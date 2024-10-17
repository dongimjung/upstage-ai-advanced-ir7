import os
import json
import faiss
import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일 로드
load_dotenv('/ir/.env')

# API_KEY 값을 가져옴
openai_api_key = os.getenv('OPENAI_API_KEY')
upstage_api_key = os.getenv('UPSTAGE_API_KEY')

# Solar API 클라이언트 설정
client = OpenAI(
    api_key= upstage_api_key,
    base_url="https://api.upstage.ai/v1/solar"
)

class SolarEmbeddings:
    def __init__(self):
        self.client = client
        print("SolarEmbeddings 인스턴스 생성 완료")

    def get_embeddings_in_batches(self, docs, batch_size):
        batch_embeddings = []
        print(f"문서 {len(docs)}개에 대해 임베딩 생성 시작, 배치 크기: {batch_size}")
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            contents = [doc["content"] for doc in batch]
            
            # Solar API로 임베딩 생성
            for content in contents:
                print(f"'{content[:30]}...'에 대한 임베딩 생성 중")
                try:
                    query_result = self.client.embeddings.create(
                        model="solar-embedding-1-large-query",
                        input=content
                    ).data[0].embedding
                    batch_embeddings.append(query_result)
                except Exception as e:
                    print(f"임베딩 생성 실패: {str(e)}")
                
        print(f"총 {len(batch_embeddings)}개의 임베딩 생성 완료")
        return batch_embeddings

# class FaissDB:
#     def __init__(self, d, index_file="/ir/knn_index_cosine.faiss"):
#         self.index_file = index_file
#         self.documents = []
#         self.index = faiss.IndexFlatL2(d)
#         print("FAISS 인덱스 초기화 완료")

#         if os.path.exists(self.index_file):
#             self.index = faiss.read_index(self.index_file)
#             print(f"FAISS 인덱스 로드 완료: {self.index_file}")
#         else:
#             print("새로운 FAISS 인덱스 생성")

#     def add_embeddings(self, embeddings, docs):
#         print(f"FAISS 인덱스에 {len(embeddings)}개의 임베딩 추가 중")
#         embeddings = np.array(embeddings).astype('float32')
#         self.index.add(embeddings)
#         self.documents.extend(docs)
#         print(f"문서 {len(docs)}개가 인덱스에 추가됨")

#     def save_index(self):
#         print(f"FAISS 인덱스 저장 중: {self.index_file}")
#         faiss.write_index(self.index, self.index_file)
#         print(f"FAISS 인덱스 저장 완료: {self.index_file}")

#     def search(self, query_embedding, k=3):
#         print(f"쿼리 임베딩에 대한 FAISS 검색 시작, k={k}")
#         query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
#         distances, indices = self.index.search(query_embedding, k)

#         if len(indices[0]) == 0 or len(distances[0]) == 0:
#             print("검색 결과가 없습니다.")
#             return []

#         results = [{"doc": self.documents[i], "distance": distances[0][idx]} for idx, i in enumerate(indices[0]) if i < len(self.documents)]
#         print(f"총 {len(results)}개의 결과 반환")
#         return results

class FaissDB:
    def __init__(self, d, index_file="/ir/knn_index_cosine.faiss", mappings_file="/ir/chunk_mappings.json"):
        self.index_file = index_file
        self.mappings_file = mappings_file  # 문서 매핑 파일 추가
        self.documents = []
        self.index = faiss.IndexFlatL2(d)
        print("FAISS 인덱스 초기화 완료")

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            print(f"FAISS 인덱스 로드 완료: {self.index_file}")
        else:
            print("새로운 FAISS 인덱스 생성")

        if os.path.exists(self.mappings_file):
            with open(self.mappings_file, "r") as f:
                self.documents = json.load(f)
            print(f"문서 매핑 로드 완료: {self.mappings_file}")
        else:
            print("문서 매핑 파일이 없습니다. 새로운 문서 매핑 생성 필요")

    def add_embeddings(self, embeddings, docs):
        print(f"FAISS 인덱스에 {len(embeddings)}개의 임베딩 추가 중")
        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(docs)
        print(f"문서 {len(docs)}개가 인덱스에 추가됨")

    def save_index(self):
        print(f"FAISS 인덱스 저장 중: {self.index_file}")
        faiss.write_index(self.index, self.index_file)
        print(f"FAISS 인덱스 저장 완료: {self.index_file}")

        with open(self.mappings_file, "w") as f:
            json.dump(self.documents, f)
        print(f"문서 매핑 저장 완료: {self.mappings_file}")

    def search(self, query_embedding, k=3):
        print(f"쿼리 임베딩에 대한 FAISS 검색 시작, k={k}")
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        if len(indices[0]) == 0 or len(distances[0]) == 0:
            print("검색 결과가 없습니다.")
            return []

        results = [{"doc": self.documents[i], "distance": distances[0][idx]} for idx, i in enumerate(indices[0]) if i < len(self.documents)]
        print(f"총 {len(results)}개의 결과 반환")
        return results

class GemmaGPT:
    def __init__(self, faiss_db=None):
        self.client = client  # Solar API 클라이언트를 사용
        self.faiss_db = faiss_db
        self.persona_qa = self.get_persona_qa_prompt()
        print("GemmaGPT 인스턴스 생성 완료")

    def get_persona_qa_prompt(self):
        return """
        ## Role: 과학 상식 전문가
        ## Instructions
        - 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
        - 한국어로 답변을 생성한다.
        """

    def generate_response(self, input_text):
        print(f"'{input_text[:30]}...'에 대한 응답 생성 중")
        # Solar API를 사용하여 텍스트 생성
        try:
            response = self.client.completions.create(
                model="solar-embedding-1-large-query",
                prompt=input_text,
                max_tokens=200
            ).choices[0].text
            print(f"응답 생성 완료: {response[:50]}...")
            return response
        except Exception as e:
            print(f"응답 생성 실패: {str(e)}")
            return "응답 생성 오류 발생"

    def answer_question(self, messages):
        response = {"answer": ""}
        try:
            msg_content = " ".join([str(message["content"]) for message in messages if "content" in message])
            print(f"'{msg_content[:30]}...'에 대한 질문 응답 시작")
            result = self.generate_response(msg_content)
        except Exception as e:
            traceback.print_exc()
            response["answer"] = "오류가 발생했습니다."
            print("응답 중 오류 발생")
            return response

        response["answer"] = result
        print("응답 완료")
        return response


def search_and_respond(query):
    print(f"'{query}'에 대한 검색 및 응답 생성 시작")
    embedding_module = SolarEmbeddings()
    faiss_db = FaissDB(768)

    if len(faiss_db.documents) == 0:
        print("문서가 존재하지 않음, 새로 문서 로드 및 임베딩 생성")
        with open("/ir/data/documents.jsonl") as f:
            docs = [json.loads(line) for line in f]
        embeddings = embedding_module.get_embeddings_in_batches(docs, 100)
        faiss_db.add_embeddings(embeddings, docs)
        faiss_db.save_index()

    print(f"'{query}'에 대한 임베딩 생성 중")
    query_embedding = client.embeddings.create(
        model="solar-embedding-1-large-query", # 쿼리 역시 faiss 인덱싱할때 사용했던거랑 같은 모델을 써서 임베딩해야함.
        input=query
    ).data[0].embedding

    search_result = faiss_db.search(query_embedding, 3)

    if search_result:
        results = [result["doc"]["content"] for result in search_result]
        return "\n".join(results)
    else:
        return "검색 결과가 없습니다."

if __name__ == "__main__":
    query = "가을 하늘은 왜 파랄까?"
    print(f"'{query}'에 대한 검색 및 응답 시작")
    result = search_and_respond(query)
    print(f"최종 결과:\n{result}")
