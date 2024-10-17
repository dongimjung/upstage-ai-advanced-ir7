from flask import Flask, request, jsonify
from kkh_baseline_solar_llm import search_and_respond
from flask_cors import CORS  # CORS 모듈 추가

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

@app.route('/rag', methods=['POST'])
def rag():
    data = request.json
    query = data.get('query')

    try:
        # RAG 시스템에서 검색한 결과를 받아옴
        result = search_and_respond(query)
        return jsonify({
            'query': query,
            'answer': result if result else "No result"
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

