import os
from dotenv import load_dotenv

# 获取项目根目录
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 加载 .env 文件（如果存在）
load_dotenv()

class BaseConfig:
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    
    # --- 服务绑定配置 ---
    APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
    APP_PORT = int(os.getenv('APP_PORT', 6008))
    
    # --- 大模型切换配置 ---
    ACTIVE_LLM = os.getenv('ACTIVE_LLM', 'deepseek') 
    
    # 允许的在线模型列表
    ONLINE_QUERY_MODELS = [
        {"id": "deepseek", "name": "DeepSeek (官方 API)"},
        {"id": "chatgpt", "name": "ChatGPT (OpenAI API)"},
        {"id": "qwen", "name": "Qwen (通义千问)"}
    ]

    # 1. Qwen 本地服务配置
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'EMPTY')
    QWEN_MODEL_NAME = os.getenv('QWEN_MODEL_NAME', 'Qwen/Qwen-7B-Chat')

    # 微调看板服务配置
    WEBUI_HOST = os.getenv('WEBUI_HOST', '0.0.0.0')
    WEBUI_PORT = int(os.getenv('WEBUI_PORT', 7860))
    TENSORBOARD_HOST = os.getenv('TENSORBOARD_HOST', '0.0.0.0')
    TENSORBOARD_PORT = int(os.getenv('TENSORBOARD_PORT', 6006))
    
    # 针对线上部署/端口映射：前端访问 TensorBoard 的实际外网地址和端口
    # 如果没配置，前端会自动使用 window.location.hostname + TENSORBOARD_PORT
    TENSORBOARD_PUBLIC_URL = os.getenv('TENSORBOARD_PUBLIC_URL', '')

    # 2. DeepSeek 服务配置
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-803936ec8add440db0dca13ec660f070')
    DEEPSEEK_MODEL_NAME = os.getenv('DEEPSEEK_MODEL_NAME', 'deepseek-chat')
    
    # 3. Ollama 服务配置
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'qwen2.5:latest')
    
    # 4. vLLM 本地通用服务配置
    VLLM_API_KEY = os.getenv('VLLM_API_KEY', 'EMPTY')
    VLLM_MODEL_NAME = os.getenv('VLLM_MODEL_NAME', 'qwen_v1')
    
    # 5. Neo4j 图数据库配置 (Graph RAG)
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', '11111111')
    
    # 6. 智能体工具调用配置 (Agent Function Calling)
    # 支持在配置中启用或禁用特定的工具
    ENABLED_TOOLS = os.getenv('ENABLED_TOOLS', 'execute_shell_command,query_api_endpoint,query_metrics').split(',')
    
    # Hugging Face 模型下载与存放目录
    HF_MODEL_DIR = os.getenv('HF_MODEL_DIR', os.path.join(BASE_DIR, 'hugface'))