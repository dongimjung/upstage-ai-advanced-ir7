import os
import json
import faiss
import numpy as np
from openai import OpenAI
import traceback
import openai
from dotenv import load_dotenv
import jsonlines
import MeCab
from rank_bm25 import BM25Okapi

# .env 파일 로드
load_dotenv('/ir/.env')

# API_KEY 값 로드
openai_api_key = os.getenv('OPENAI_API_KEY')
upstage_api_key = os.getenv('UPSTAGE_API_KEY')

os.environ["OPENAI_API_KEY"] = openai_api_key

# Upstage API 클라이언트 설정
client = OpenAI(
    api_key= upstage_api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

client_gpt = OpenAI()

mecab = MeCab.Tagger("-r /etc/mecabrc")

# JSONL 파일 경로
jsonl_file_path = '/ir/data/documents.jsonl'

# 문서와 docid 저장할 리스트
documents = []
docids = []

stoptags = {"E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"}

# JSONL 파일 읽기
with jsonlines.open(jsonl_file_path) as reader:
    for obj in reader:
        docids.append(obj['docid'])      # docid 저장
        documents.append(obj['content']) # content 저장

# Mecab을 사용하여 문서 토큰화
def tokenize_with_mecab(text):
    tokens = mecab.parse(text).splitlines()
    processed_tokens = []
    
    for token in tokens:
        if "\t" in token:  # 형태소와 품사 태그가 \t로 구분됨
            word, tag_info = token.split("\t")
            pos_tag = tag_info.split(",")[0]  # 품사 태그는 ,로 구분된 첫 번째 요소
            if pos_tag not in stoptags:  # 불필요한 품사 태그가 아닌 경우에만 추가
                processed_tokens.append(word)
    
    return processed_tokens

tokenized_corpus = [tokenize_with_mecab(doc) for doc in documents]

# BM25 인덱서 생성
bm25 = BM25Okapi(tokenized_corpus)

with open("/ir/data/eval.jsonl", "r") as f:
    eval_doc_mapping = [json.loads(line) for line in f]

doc_mapping = {}
with open("/ir/data/documents.jsonl", "r") as f:
    for line in f:
        doc = json.loads(line)
        doc_mapping[doc['docid']] = doc 

index = faiss.read_index("/ir/knn_index_cosine.faiss")

with open("/ir/chunk_mappings.json", "r") as f:
    chunk_doc_mapping = json.load(f)

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def normalize(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)

def min_max_normalize(ranked_docs):
    scores = np.array([score for _, score in ranked_docs])
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        normalized_scores = np.ones_like(scores)
    else:
        normalized_scores = (scores - min_score) / (max_score - min_score)

    normalized_ranked_docs = [(docid, score) for (docid, _), score in zip(ranked_docs, normalized_scores)]
    
    return normalized_ranked_docs

def z_score_normalize(ranked_docs):
    scores = np.array([score for _, score in ranked_docs])
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    
    if std_dev == 0:
        normalized_scores = np.zeros_like(scores)
    else:
        normalized_scores = (scores - mean_score) / std_dev

    normalized_ranked_docs = [(docid, score) for (docid, _), score in zip(ranked_docs, normalized_scores)]
    
    return normalized_ranked_docs

def merge_and_sum_scores(knn_retrieved_docs, bm25_retrieved_docs, p):
    knn_dict = {doc_id: score for doc_id, score in knn_retrieved_docs}
    bm25_dict = {doc_id: score for doc_id, score in bm25_retrieved_docs}
    
    all_ids = set(knn_dict.keys()).union(set(bm25_dict.keys()))

    merged_docs = []
    for doc_id in all_ids:
        knn_score = knn_dict.get(doc_id, 0)
        bm25_score = bm25_dict.get(doc_id, 0)
        combined_score = p * knn_score + (1 - p) * bm25_score
        merged_docs.append((doc_id, combined_score))
    
    merged_docs = sorted(merged_docs, key=lambda x: x[1], reverse=True)
    
    return merged_docs

def retrieval_with_score(query):
    """ knn """
    query_result = client.embeddings.create(
        model="solar-embedding-1-large-query",
        input=query
    ).data[0].embedding

    query_embedding = np.array(query_result).reshape(1, -1)

    normalized_query = normalize(query_embedding)

    k = 200
    distances, indices = index.search(normalized_query, k)

    retrieved_doc_ids = []

    for i, idx in enumerate(indices[0]):
        chunk_info = chunk_doc_mapping[idx]
        doc_id = chunk_info['doc_id']
        
        cosine_distance = distances[0][i]
        cosine_similarity = 1 - cosine_distance

        retrieved_doc_ids.append((doc_id, cosine_similarity))
    
    knn_retrieved_docs = z_score_normalize(retrieved_doc_ids)
    
    """ BM25 """ 
    tokenized_query = tokenize_with_mecab(query)

    doc_scores = bm25.get_scores(tokenized_query)

    ranked_docs = sorted(zip(docids, doc_scores), key=lambda x: x[1], reverse=True)[:k]
    
    bm25_retrieved_docs = z_score_normalize(ranked_docs)

    merged_docs = merge_and_sum_scores(knn_retrieved_docs, bm25_retrieved_docs, 0.7)
    
    top_docs = merged_docs[:5]

    return top_docs

def generate_response_from_documents(query, top_docs):
    """
    검색된 상위 문서를 바탕으로 Solar-1-mini-chat 모델을 사용하여 답변을 생성하는 함수
    """
    context = "\n".join([doc_mapping[doc[0]]['content'] for doc in top_docs])
    
    prompt = f"""
    ## Role: 과학 상식 전문가
    ## Instructions
    - 사용자의 질문과 주어진 문서 정보를 활용하여 간결하게 답변을 생성하세요.
    - 질문: {query}
    - 참고 문서: {context}
    """
    
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {"role": "system", "content": prompt},
            ]
        )
        generated_text = response.choices[0].message.content
        return generated_text
    except Exception as e:
        print(f"응답 생성 실패: {str(e)}")
        return "응답 생성 오류 발생"

def top_agent_without_relevance_check(queries):
    if len(queries) > 1:
        transformed_query = query_transformer(queries)
        query = [
            {"role": "user", "content": transformed_query}
        ]
        checked_query = science_query_detector(query)
        
        if checked_query == '과학 관련 질문이 아닙니다.':
            return False
    else:
        checked_query = science_query_detector(queries)
        if checked_query == '과학 관련 질문이 아닙니다.':
            return False

    retrieved_doc = retrieval_with_score(checked_query)
    
    check_list = []

    for doc in retrieved_doc:
        if doc[1] > retrieved_doc[0][1] * 0.63:
            check_list.append(doc)
    
    return check_list

# 검색된 문서를 기반으로 답변 생성
def search_and_generate_response(query):
    print(f"'{query}'에 대한 검색 및 응답 생성 시작")
    top_docs = retrieval_with_score(query)
    return generate_response_from_documents(query, top_docs)

# 실행 예시
if __name__ == "__main__":
    query = "왜 여름에는 풀잎들이 초록초록해져?"
    result = search_and_generate_response(query)
    print(result)
