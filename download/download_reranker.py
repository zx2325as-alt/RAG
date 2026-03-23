import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

local_dir = r"e:\python\conda\RAG\model\bge-reranker-large"
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