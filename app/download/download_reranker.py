import sys
import os

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 引入项目配置
from app.config import Config

from huggingface_hub import snapshot_download

local_dir = Config.RERANKER_MODEL_PATH
os.makedirs(local_dir, exist_ok=True)

print(f"正在从国内镜像下载重排序模型 BAAI/bge-reranker-large ...")
try:
    snapshot_download(
        repo_id="BAAI/bge-reranker-large",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*"]
    )
    print("重排序模型下载成功！")
except Exception as e:
    print(f"下载123424失败: {e}")