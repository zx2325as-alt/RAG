import os

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///rag.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    
    # --- 大模型切换配置 ---
    # 可选值: 'qwen' (本地vLLM部署), 'deepseek' (官方API), 'ollama', 或 'vllm' (本地vLLM通用引擎)
    ACTIVE_LLM = os.getenv('ACTIVE_LLM', 'deepseek') 

    # 1. Qwen 本地服务配置 (vLLM 等)
    QWEN_API_URL = os.getenv('QWEN_API_URL', 'http://localhost:8000/v1/chat/completions')
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'EMPTY')
    QWEN_MODEL_NAME = os.getenv('QWEN_MODEL_NAME', 'Qwen/Qwen-7B-Chat')

    # 2. DeepSeek 服务配置 (支持官方 API 或 本地 vLLM)
    # 如果使用官方 API，URL 通常为 https://api.deepseek.com/chat/completions (注意新版DeepSeek官方去掉了 /v1 路径)
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/chat/completions')
    # 请在系统环境变量中设置 DEEPSEEK_API_KEY，或在此处填入真实的 API Key
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-803936ec8add440db0dca13ec660f070')
    # 模型名称通常为 deepseek-chat 或 deepseek-coder
    DEEPSEEK_MODEL_NAME = os.getenv('DEEPSEEK_MODEL_NAME', 'deepseek-chat')
    
    # 3. Ollama 服务配置
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    # 默认 Ollama 模型，可在前端切换
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'qwen2.5:latest')
    
    # 4. vLLM 本地通用服务配置
    VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1/chat/completions')
    VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'EMPTY')
    VLLM_MODEL_NAME = os.getenv('VLLM_MODEL_NAME', 'qwen_v1')
