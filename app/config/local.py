import os
from .base import BaseConfig, BASE_DIR

class LocalConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///rag.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index')
    
    # 本地版服务绑定地址
    APP_HOST = '127.0.0.1'
    WEBUI_HOST = '127.0.0.1'
    TENSORBOARD_HOST = '127.0.0.1'
    
    # 本地版图数据库配置
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')
    
    # 本地版模型路径配置 (相对路径)
    EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', os.path.join(BASE_DIR, 'model', 'bge-large-zh'))
    RERANKER_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH', os.path.join(BASE_DIR, 'model', 'bge-reranker-large'))
    QWEN_LOCAL_MODEL_PATH = os.getenv('QWEN_LOCAL_MODEL_PATH', os.path.join(BASE_DIR, 'model', 'Qwen-7B-Chat'))
    
    # 本地版 API URLs
    QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://127.0.0.1:8000/v1/chat/completions')
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://127.0.0.1:8000/v1/chat/completions')