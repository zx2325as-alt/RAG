from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
import os

# 确保能导入 app.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.config import Config

app = Flask(__name__)

# 加载 Rerank 模型
model_name = os.getenv("RERANK_MODEL_PATH", Config.RERANKER_MODEL_PATH)
print(f"Loading reranker model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

@app.route('/rerank', methods=['POST'])
def rerank():
    data = request.json
    query = data.get("query")
    candidates = data.get("candidates", []) # [{"doc_id": "...", "content": "..."}]
    top_k = data.get("top_k", 5)
    
    if not query or not candidates:
        return jsonify({"error": "query and candidates are required"}), 400
        
    try:
        pairs = [[query, doc["content"]] for doc in candidates]
        
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
            
        results = []
        for doc, score in zip(candidates, scores):
            doc["rerank_score"] = score.item()
            results.append(doc)
            
        # 按重排序得分降序
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return jsonify({"top_k": results[:top_k]})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)