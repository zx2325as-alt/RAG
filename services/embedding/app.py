from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import os

# 确保能导入 app.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.config import Config

app = Flask(__name__)

# 加载 Embedding 模型
model_name = os.getenv("EMBEDDING_MODEL_PATH", Config.EMBEDDING_MODEL_PATH)
print(f"Loading embedding model from {model_name}...")
embeddings = HuggingFaceEmbeddings(model_name=model_name)

@app.route('/embed', methods=['POST'])
def embed():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "text is required"}), 400
        
    try:
        if isinstance(text, str):
            vector = embeddings.embed_query(text)
            return jsonify({"vector": vector})
        elif isinstance(text, list):
            vectors = embeddings.embed_documents(text)
            return jsonify({"vectors": vectors})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)