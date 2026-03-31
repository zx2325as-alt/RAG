#!/bin/bash
# ==============================================================================
# 湖北移动运维知识助手 (RAG) - Linux 自动化安装与启动脚本
# 此脚本将尝试安装必要的系统依赖 (Redis, Docker用于Neo4j) 和 Python 环境，并启动所有服务。
# 请确保以有 sudo 权限的用户运行。
# ==============================================================================

set -e # 遇到错误即退出
export LC_ALL=C.UTF-8

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}        湖北移动运维知识助手 (RAG) 自动化部署脚本      ${NC}"
echo -e "${GREEN}=======================================================${NC}"

# ==========================================
# 0. 环境准备：处理 sudo
# ==========================================
# 如果当前是 root 用户，则无需使用 sudo
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null; then
        SUDO_CMD="sudo"
    else
        echo -e "${RED}当前非 root 用户且未安装 sudo，请切换到 root 用户执行此脚本。${NC}"
        exit 1
    fi
fi

# ==========================================
# 1. 检查并安装系统级依赖
# ==========================================
echo -e "\n${YELLOW}[1/4] 正在检查并安装系统级依赖...${NC}"

# 检查包管理器
if command -v apt-get >/dev/null; then
    PKG_MANAGER="apt-get"
    # 为了加快二次执行速度，可以跳过频繁的 update
    # $SUDO_CMD apt-get update -y
elif command -v yum >/dev/null; then
    PKG_MANAGER="yum"
elif command -v dnf >/dev/null; then
    PKG_MANAGER="dnf"
else
    echo -e "${RED}未找到支持的包管理器 (apt/yum/dnf)，请手动安装系统依赖。${NC}"
    exit 1
fi

# 1.1 安装 Redis
if ! command -v redis-server >/dev/null; then
    echo "正在安装 Redis..."
    if [ "$PKG_MANAGER" = "apt-get" ]; then
        $SUDO_CMD apt-get update -y
    fi
    $SUDO_CMD $PKG_MANAGER install -y redis-server redis-tools
    $SUDO_CMD systemctl enable redis-server || true
    $SUDO_CMD systemctl start redis-server || true
else
    echo "Redis 已安装，确保其正在运行..."
    # 尝试启动，如果不是 systemd 管理的系统（如某些 docker 容器内），可能会失败，这里忽略错误
    $SUDO_CMD systemctl start redis-server >/dev/null 2>&1 || true
    # 如果 redis 没跑起来，尝试直接后台启动
    if ! redis-cli ping >/dev/null 2>&1; then
        echo "尝试手动后台启动 redis-server..."
        $SUDO_CMD redis-server --daemonize yes || true
    fi
fi

# 1.2 检查 Docker (用于快速启动 Neo4j)
if ! command -v docker >/dev/null; then
    echo "正在安装 Docker (用于运行 Neo4j)..."
    if [ "$PKG_MANAGER" = "apt-get" ]; then
        $SUDO_CMD apt-get update -y
        $SUDO_CMD apt-get install -y docker.io
    else
        $SUDO_CMD $PKG_MANAGER install -y docker
    fi
    $SUDO_CMD systemctl enable docker || true
    $SUDO_CMD systemctl start docker || true
else
    echo "Docker 已安装。"
fi

# ==========================================
# 2. 安装 Python 依赖
# ==========================================
echo -e "\n${YELLOW}[2/4] 正在配置 Python 环境并安装依赖...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}未找到 requirements.txt 文件，请确保在项目根目录下执行此脚本。${NC}"
    exit 1
fi

echo "检查并安装基础 Python 依赖..."
# 使用 pip install -r 且配合 --no-cache-dir 或让 pip 自行判断缓存，pip 默认如果已安装且版本满足就会跳过
# 为了更明显的跳过提示，我们这里直接运行，pip 会自动跳过已安装的
pip install -r requirements.txt

echo "检查并安装 LLaMA-Factory..."
if ! command -v llamafactory-cli >/dev/null; then
    pip install llamafactory
else
    echo "LLaMA-Factory 已安装，跳过。"
fi

echo "检查并安装 vLLM..."
if ! python -c "import vllm" >/dev/null 2>&1; then
    pip install vllm || echo -e "${YELLOW}警告: vLLM 安装失败，可能需要特定的 CUDA 版本，请稍后手动排查。${NC}"
else
    echo "vLLM 已安装，跳过。"
fi

# ==========================================
# 3. 启动后台数据库/中间件
# ==========================================
echo -e "\n${YELLOW}[3/4] 正在启动基础服务 (Redis & Neo4j)...${NC}"

# Redis (已由 systemd 管理，这里只做检查)
echo "检查 Redis 状态..."
if redis-cli ping >/dev/null 2>&1; then
    echo "Redis 运行正常。"
else
    echo -e "${YELLOW}警告: Redis 未响应，请检查 Redis 服务状态。${NC}"
fi

# Neo4j (使用 Docker 启动)
echo "正在启动 Neo4j Docker 容器..."
# 检查是否已存在名为 rag-neo4j 的容器
if $SUDO_CMD docker ps -a --format '{{.Names}}' | grep -Eq "^rag-neo4j$"; then
    echo "Neo4j 容器已存在，检查运行状态..."
    if ! $SUDO_CMD docker ps --format '{{.Names}}' | grep -Eq "^rag-neo4j$"; then
        echo "启动已存在的 Neo4j 容器..."
        $SUDO_CMD docker start rag-neo4j
    else
        echo "Neo4j 容器已经在运行中。"
    fi
else
    echo "创建并启动新的 Neo4j 容器..."
    # 密码设置为 11111111 与 config/base.py 对应
    $SUDO_CMD docker run -d \
        --name rag-neo4j \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/11111111 \
        -v $PWD/neo4j_data:/data \
        neo4j:latest
fi

# 等待 Neo4j 初始化
echo "等待 Neo4j 服务启动 (约 10 秒)..."
sleep 10

# ==========================================
# 4. 启动 Python 应用项目
# ==========================================
echo -e "\n${YELLOW}[4/4] 正在启动所有核心应用服务...${NC}"

# 创建 logs 目录
mkdir -p logs

# 4.1 启动 LLaMA-Factory WebUI
# 修复找不到 data/dataset_info.json 的问题：需确保在 LLaMA-Factory 目录下启动，或者手动创建所需目录
echo "-> 准备 LLaMA-Factory 运行环境..."
if [ ! -d "data" ]; then
    mkdir -p data
    echo "{}" > data/dataset_info.json
fi
if [ ! -f "data/dataset_info.json" ]; then
    echo "{}" > data/dataset_info.json
fi

echo "-> 启动 LLaMA-Factory WebUI (绑定端口: 6006 以适应 AutoDL 映射)..."
# 通过环境变量强制 LLaMA-Factory 使用 6006 端口
export GRADIO_SERVER_PORT=6006
export GRADIO_SERVER_NAME=0.0.0.0
nohup llamafactory-cli webui > logs/llamafactory.log 2>&1 &
LLAMA_PID=$!
echo "LLaMA-Factory 进程 ID: $LLAMA_PID"

# 4.2 启动 vLLM API Server (可选)
# 如果配置了 vLLM 模型，可以在此启动。这里以一个占位模型为例，实际使用时需要替换模型路径
# VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
# echo "-> 启动 vLLM Server (模型: $VLLM_MODEL, 端口: 8000)..."
# nohup python -m vllm.entrypoints.openai.api_server --model $VLLM_MODEL --port 8000 > logs/vllm.log 2>&1 &
# VLLM_PID=$!
# echo "vLLM 进程 ID: $VLLM_PID"
echo "-> (提示: vLLM API Server 可在前端页面动态启动，此处跳过自动启动)"

# 4.3 启动 RAG Flask 后端
echo "-> 启动 RAG 后端服务 (端口: 6008)..."
nohup python run.py > logs/rag_backend.log 2>&1 &
RAG_PID=$!
echo "RAG 后端进程 ID: $RAG_PID"

echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}所有服务已下发启动指令！运行日志已保存至 logs/ 目录。${NC}"
echo -e "服务访问地址："
echo -e "- RAG 问答主前端:   ${YELLOW}http://<本机IP>:6008${NC} (AutoDL用户请访问 6008 映射地址)"
echo -e "- LLaMA-Factory:    ${YELLOW}http://<本机IP>:6006${NC} (AutoDL用户请访问 6006 映射地址)"
echo -e "- Neo4j Browser:    ${YELLOW}http://<本机IP>:7474${NC} (用户: neo4j, 密码: 11111111)"
echo -e "${GREEN}=======================================================${NC}"
echo "如需停止服务，请使用: kill $LLAMA_PID $RAG_PID"
