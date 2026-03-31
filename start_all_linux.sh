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

# 1.1 安装 Redis 和 Java (Neo4j 依赖 Java 17+)
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

# 1.2 检查 Java 环境 (Neo4j 5.x 需要 Java 17)
if ! command -v java >/dev/null; then
    echo "正在安装 Java 17 (Neo4j 依赖)..."
    if [ "$PKG_MANAGER" = "apt-get" ]; then
        $SUDO_CMD apt-get install -y openjdk-17-jre-headless
    else
        $SUDO_CMD $PKG_MANAGER install -y java-17-openjdk
    fi
else
    echo "Java 已安装。"
fi

# ==========================================
# 2. 安装 Python 依赖
# ==========================================
echo -e "\n${YELLOW}[2/4] 正在配置 Python 环境并安装依赖...${NC}"

# 如果您使用 requirements_online.txt 安装过了，这一步将很快跳过
echo "检查并安装基础 Python 依赖..."
if [ -f "requirements_online.txt" ]; then
    pip install -r requirements_online.txt
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${RED}未找到 requirements 文件，请确保在项目根目录下执行此脚本。${NC}"
    exit 1
fi

echo "检查并安装本地 LLaMA-Factory..."
if [ -d "LLaMA-Factory" ]; then
    echo "发现本地 LLaMA-Factory 目录，正在安装..."
    pip install -e LLaMA-Factory
else
    echo "未发现本地 LLaMA-Factory 目录，尝试通过 pip 安装..."
    if ! command -v llamafactory-cli >/dev/null; then
        pip install llamafactory
    fi
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

# Neo4j (本地目录启动)
echo "正在启动本地 Neo4j 服务..."
NEO4J_HOME="$PWD/model/neo4j-community-5.18.1"
if [ -d "$NEO4J_HOME" ]; then
    echo "发现本地 Neo4j 目录: $NEO4J_HOME"
    
    # 确保脚本有执行权限
    chmod +x "$NEO4J_HOME/bin/neo4j"
    chmod +x "$NEO4J_HOME/bin/cypher-shell" || true
    
    # 尝试启动
    echo "执行 Neo4j 启动命令..."
    "$NEO4J_HOME/bin/neo4j" start || echo -e "${YELLOW}警告: Neo4j 启动遇到问题，它可能已经在运行。${NC}"
    
    # 等待 Neo4j 初始化
    echo "等待本地 Neo4j 服务启动 (约 10 秒)..."
    sleep 10
else
    echo -e "${RED}未找到本地 Neo4j 目录 ($NEO4J_HOME)，请确认路径是否正确。${NC}"
    echo -e "${YELLOW}由于未找到 Neo4j，图谱相关功能将降级或失效。${NC}"
fi

# ==========================================
# 4. 启动 Python 应用项目
# ==========================================
echo -e "\n${YELLOW}[4/4] 正在启动所有核心应用服务...${NC}"

# 创建 logs 目录
mkdir -p logs

# 4.1 启动 LLaMA-Factory WebUI
echo "-> 准备 LLaMA-Factory 运行环境..."
LLAMA_FACTORY_DIR="$PWD/LLaMA-Factory"

if [ -d "$LLAMA_FACTORY_DIR" ]; then
    echo "进入本地 LLaMA-Factory 目录启动 WebUI..."
    cd "$LLAMA_FACTORY_DIR"
    
    # 修复找不到 data/dataset_info.json 的问题
    if [ ! -d "data" ]; then
        mkdir -p data
    fi
    if [ ! -f "data/dataset_info.json" ]; then
        echo "{}" > data/dataset_info.json
    fi

    echo "-> 启动 LLaMA-Factory WebUI (绑定端口: 6006 以适应 AutoDL 映射)..."
    export GRADIO_SERVER_PORT=6006
    export GRADIO_SERVER_NAME=0.0.0.0
    
    # 使用本地源代码启动
    nohup python src/webui.py > ../logs/llamafactory.log 2>&1 &
    LLAMA_PID=$!
    
    # 退回原目录
    cd ..
else
    echo -e "${YELLOW}未找到本地 LLaMA-Factory 目录，尝试使用系统命令启动...${NC}"
    # 修复找不到 data/dataset_info.json 的问题
    if [ ! -d "data" ]; then
        mkdir -p data
    fi
    if [ ! -f "data/dataset_info.json" ]; then
        echo "{}" > data/dataset_info.json
    fi
    
    export GRADIO_SERVER_PORT=6006
    export GRADIO_SERVER_NAME=0.0.0.0
    nohup llamafactory-cli webui > logs/llamafactory.log 2>&1 &
    LLAMA_PID=$!
fi

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
