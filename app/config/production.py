import os
from .base import BaseConfig, BASE_DIR

class ProductionConfig(BaseConfig):
    DEBUG = False
    # 线上版数据库通常为 MySQL/PostgreSQL，这里通过环境变量覆盖，默认给出示例
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'mysql+pymysql://user:password@localhost/rag_db')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///rag.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', '/data/faiss_index')
    
    # 线上版服务绑定地址 (通常需要 0.0.0.0 对外暴露，或由 Nginx 代理)
    APP_HOST = '0.0.0.0'
    WEBUI_HOST = '0.0.0.0'
    TENSORBOARD_HOST = '0.0.0.0'
    
    # 线上版图数据库配置 (生产环境可能在其他内网 IP)
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://neo4j-server:7687')
    
    # 线上版模型路径配置 (相对路径)
    EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', os.path.join(BASE_DIR, 'model', 'bge-large-zh'))
    RERANKER_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH', os.path.join(BASE_DIR, 'model', 'bge-reranker-large'))
    QWEN_LOCAL_MODEL_PATH = os.getenv('QWEN_LOCAL_MODEL_PATH', os.path.join(BASE_DIR, 'model', 'Qwen-7B-Chat'))

    # 线上版 API URLs
    QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://127.0.0.1:8000/v1/chat/completions')
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://127.0.0.1:8000/v1/chat/completions')