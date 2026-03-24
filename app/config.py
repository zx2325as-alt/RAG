import os
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

class BaseConfig:
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    
    # --- 大模型切换配置 ---
    # 可选值: 'qwen' (本地vLLM部署), 'deepseek' (官方API), 'ollama', 或 'vllm' (本地vLLM通用引擎)
    ACTIVE_LLM = os.getenv('ACTIVE_LLM', 'deepseek') 
    
    # 允许的在线模型列表 (可在此配置新增)
    ONLINE_QUERY_MODELS = [
        {"id": "deepseek", "name": "DeepSeek (官方 API)"},
        {"id": "chatgpt", "name": "ChatGPT (OpenAI API)"},
        {"id": "qwen", "name": "Qwen (通义千问)"}
    ]

    # 1. Qwen 本地服务配置
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'EMPTY')
    QWEN_MODEL_NAME = os.getenv('QWEN_MODEL_NAME', 'Qwen/Qwen-7B-Chat')

    # 微调看板服务配置
    WEBUI_HOST = os.getenv('WEBUI_HOST', '127.0.0.1')
    WEBUI_PORT = int(os.getenv('WEBUI_PORT', 7860))
    TENSORBOARD_HOST = os.getenv('TENSORBOARD_HOST', '127.0.0.1')
    TENSORBOARD_PORT = int(os.getenv('TENSORBOARD_PORT', 6006))

    # 2. DeepSeek 服务配置
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-803936ec8add440db0dca13ec660f070')
    DEEPSEEK_MODEL_NAME = os.getenv('DEEPSEEK_MODEL_NAME', 'deepseek-chat')
    
    # 3. Ollama 服务配置
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'qwen2.5:latest')
    
    # 4. vLLM 本地通用服务配置
    VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'EMPTY')
    VLLM_MODEL_NAME = os.getenv('VLLM_MODEL_NAME', 'qwen_v1')

class LocalConfig(BaseConfig):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///rag.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index')
    
    # 本地版服务绑定地址
    WEBUI_HOST = '127.0.0.1'
    TENSORBOARD_HOST = '127.0.0.1'
    
    # 本地版模型路径配置
    EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', r"e:\python\conda\RAG\model\bge-large-zh")
    RERANKER_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH', r"e:\python\conda\RAG\model\bge-reranker-large")
    QWEN_LOCAL_MODEL_PATH = os.getenv('QWEN_LOCAL_MODEL_PATH', r"e:\python\conda\RAG\model\Qwen-7B-Chat")
    
    # 本地版 API URLs
    QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://localhost:8000/v1/chat/completions')
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1/chat/completions')

class ProductionConfig(BaseConfig):
    DEBUG = False
    # 线上版数据库通常为 MySQL/PostgreSQL，这里通过环境变量覆盖，默认给出示例
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'mysql+pymysql://user:password@localhost/rag_db')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///rag.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://redis-server:6379/0')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', '/data/faiss_index')
    
    # 线上版服务绑定地址 (通常需要 0.0.0.0 对外暴露，或由 Nginx 代理)
    WEBUI_HOST = '0.0.0.0'
    TENSORBOARD_HOST = '0.0.0.0'
    
    # 线上版模型路径配置 (线上容器或服务器中的绝对路径)
    EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH', '/root/project/RAG/model/bge-large-zh')
    RERANKER_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH', '/root/project/RAG/model/bge-reranker-large')
    QWEN_LOCAL_MODEL_PATH = os.getenv('QWEN_LOCAL_MODEL_PATH', '/root/project/RAG/model/Qwen-7B-Chat')

    # 线上版 API URLs
    QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://qwen-server:8000/v1/chat/completions')
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://ollama-server:11434')
    VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://vllm-server:8000/v1/chat/completions')

# 一键切换配置：通过环境变量 APP_ENV 控制 (local 或 production)
env = os.getenv('APP_ENV', 'local').lower()
config_map = {
    'local': LocalConfig,
    'production': ProductionConfig
}

Config = config_map.get(env, LocalConfig)

