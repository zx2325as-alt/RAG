import sys
import os

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 设置 HuggingFace 国内镜像源，解决 [WinError 10060] 连接超时问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 引入项目配置
from app.config import Config

from huggingface_hub import snapshot_download

local_dir = Config.EMBEDDING_MODEL_PATH
os.makedirs(local_dir, exist_ok=True)

print(f"===========================================================")
print(f"正在从国内镜像 (hf-mirror.com) 下载模型 BAAI/bge-large-zh")
print(f"目标文件夹: {local_dir}")
print(f"===========================================================")

# 仅下载运行必需的文件，避免下载过大的重复权重（如 .bin 和 .safetensors 会重复）
try:
    snapshot_download(
        repo_id="BAAI/bge-large-zh",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*"]
    )
    print("模型下载成1111功！现在本地读取不会再报错了。")
except Exception as e:
    print(f"下载失败: {e}")
