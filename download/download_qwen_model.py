import os
# 设置 HuggingFace 国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 指定模型下载目录
local_dir = r"e:\python\conda\RAG\model\Qwen-7B-Chat"
os.makedirs(local_dir, exist_ok=True)

print(f"===========================================================")
print(f"正在从国内镜像 (hf-mirror.com) 下载模型 Qwen/Qwen-7B-Chat")
print(f"目标文件夹: {local_dir}")
print(f"注意: 该模型较大(约 15GB)，下载可能需要较长时间")
print(f"===========================================================")

try:
    snapshot_download(
        repo_id="Qwen/Qwen-7B-Chat",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        # 可以忽略某些不必要的文件，比如 .bin 和 .safetensors 重复时，这里下载 safetensors
        ignore_patterns=["*.bin*", "*.pt"]
    )
    print("模型下载成功！")
except Exception as e:
    print(f"下载失败: {e}")