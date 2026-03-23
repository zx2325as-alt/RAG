from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import json
import redis
from datetime import datetime, timedelta

# 这里模拟 QA API 的核心逻辑，实际调用其他微服务
app = FastAPI(title="问答API服务")

# 假设这些是其他微服务的地址
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:5001/embed")
RERANK_SERVICE_URL = os.getenv("RERANK_SERVICE_URL", "http://localhost:5002/rerank")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8000/v1/chat/completions")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis 缓存
try:
    redis_client = redis.from_url(REDIS_URL)
    redis_client.ping()
    use_redis = True
except Exception:
    use_redis = False

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = "anonymous"

class SourceInfo(BaseModel):
    content: str
    doc_id: Optional[str]
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    question = request.question
    
    # 1. 检查缓存
    cache_key = f"query:{hash(question)}"
    if use_redis:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
            
    # 2. 调用 Embedding 服务获取向量 (简化示例)
    # 实际应从 FAISS + BM25 获取 top-50
    try:
        # 这里用伪代码代表请求
        # embed_res = requests.post(EMBEDDING_SERVICE_URL, json={"text": question})
        # vector = embed_res.json()["vector"]
        # bm25_results = search_bm25(question)
        # faiss_results = search_faiss(vector)
        # top_50 = fuse_rrf(bm25_results, faiss_results)
        top_50 = [{"content": "示例上下文1", "doc_id": "doc1"}, {"content": "示例上下文2", "doc_id": "doc2"}] # Mock
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索阶段失败: {e}")
        
    # 3. 调用 Rerank 服务获取 top-5
    try:
        # rerank_res = requests.post(RERANK_SERVICE_URL, json={"query": question, "candidates": top_50})
        # top_5 = rerank_res.json()["top_k"]
        top_5 = top_50[:5] # Mock
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重排序阶段失败: {e}")
        
    # 4. 组装 Prompt 并调用 LLM 服务
    context = "\n\n".join([doc["content"] for doc in top_5])
    prompt = f"你是一个专业的运维助手。请根据以下参考资料回答用户问题。\n参考资料：\n{context}\n\n问题：{question}"
    
    try:
        # llm_res = requests.post(LLM_SERVICE_URL, json={"messages": [{"role": "user", "content": prompt}]})
        # answer = llm_res.json()["choices"][0]["message"]["content"]
        answer = "根据资料，这是模拟的生成答案。" # Mock
    except Exception as e:
        answer = "抱歉，生成答案时出现错误。请检查LLM服务状态。"
        
    response = QueryResponse(
        answer=answer,
        sources=[SourceInfo(content=doc["content"][:100], doc_id=doc.get("doc_id"), score=1.0) for doc in top_5]
    )
    
    # 5. 写入缓存
    if use_redis:
        redis_client.setex(cache_key, timedelta(hours=24), response.json())
        
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)