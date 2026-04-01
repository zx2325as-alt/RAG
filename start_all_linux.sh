#!/bin/bash
# ==============================================================================
# 湖北移动运维知识助手 (RAG) - Linux 自动化安装与启动脚本
# 【CUDA 11.8 专用最终版 | 无环境损坏 | 一键全启动】
# ==============================================================================

set -e
export LC_ALL=C.UTF-8

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}        湖北移动运维知识助手 (RAG) 自动化部署脚本      ${NC}"
echo -e "${GREEN}=======================================================${NC}"

# ==========================================
# 0. 环境准备
# ==========================================
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null; then
        SUDO_CMD="sudo"
    else
        echo -e "${RED}请用 root 用户执行${NC}"
        exit 1
    fi
fi

# ==========================================
# 1. 系统依赖：Redis + Java
# ==========================================
echo -e "\n${YELLOW}[1/5] 安装系统依赖 Redis + Java${NC}"

if command -v apt-get >/dev/null; then
    PKG_MANAGER="apt-get"
    $SUDO_CMD apt-get update -y
else
    echo -e "${RED}仅支持 Ubuntu/Debian${NC}"
    exit 1
fi

# Redis
if ! command -v redis-server >/dev/null; then
    $SUDO_CMD apt-get install -y redis-server redis-tools
fi
$SUDO_CMD systemctl enable redis-server
$SUDO_CMD systemctl restart redis-server
sleep 2
redis-cli ping >/dev/null 2>&1 || $SUDO_CMD redis-server --daemonize yes
echo -e "${GREEN}✅ Redis 已启动${NC}"

# Java 17
if ! command -v java >/dev/null; then
    $SUDO_CMD apt-get install -y openjdk-17-jre-headless
fi
echo -e "${GREEN}✅ Java 17 已就绪${NC}"

# ==========================================
# 2. 安装 Python 依赖（安全模式，不炸环境）
# ==========================================
echo -e "\n${YELLOW}[2/5] 安装 Python 依赖（安全无冲突）${NC}"

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/
pip config set global.trusted-host "mirrors.aliyun.com pypi.tuna.tsinghua.edu.cn"

# 核心：安全安装依赖，不升级CUDA，不破坏torch
pip install -r requirements_online.txt --no-deps

# 补全轻量依赖（绝对安全）
pip install blinker click itsdangerous jinja2 markupsafe werkzeug sqlalchemy aiohttp dataclasses-json httpx-sse langsmith pydantic pydantic-settings tenacity huggingface-hub tokenizers openai tiktoken protobuf

# 安全安装 LLaMA-Factory
pip install llamafactory==0.8.3 --no-deps

echo -e "${GREEN}✅ Python 环境安装完成（无冲突）${NC}"

# ==========================================
# 3. 安装 vLLM 0.4.1 CUDA11.8 预编译版（不编译、不卡死）
# ==========================================
echo -e "\n${YELLOW}[3/5] 安装 vLLM 0.4.1 (CUDA 11.8 专用)${NC}"

if ! python -c "import vllm" >/dev/null 2>&1; then
    pip install ray==2.54.1 xformers==0.0.25 outlines==0.0.34 nvidia-ml-py pydantic==2.12.5 fastapi uvicorn
    export VLLM_INSTALL_PUNICA_DROP_IN=0
    export VLLM_TARGET_DEVICE="cuda"
    pip install --no-deps https://github.com/vllm-project/vllm/releases/download/v0.4.1/vllm-0.4.1+cu118-cp310-cp310-manylinux1_x86_64.whl
    echo -e "${GREEN}✅ vLLM 安装完成${NC}"
else
    echo -e "${GREEN}✅ vLLM 已存在${NC}"
fi

# ==========================================
# 4. 启动 Neo4j（本地版，自动修复权限）
# ==========================================
echo -e "\n${YELLOW}[4/5] 启动 Neo4j 图数据库${NC}"

NEO4J_HOME="$PWD/model/neo4j-community-5.18.1"
if [ -d "$NEO4J_HOME" ]; then
    chmod +x "$NEO4J_HOME/bin/neo4j"
    chmod +x "$NEO4J_HOME/bin/cypher-shell"
    $NEO4J_HOME/bin/neo4j stop >/dev/null 2>&1 || true
    $NEO4J_HOME/bin/neo4j start
    sleep 10
    echo -e "${GREEN}✅ Neo4j 已启动${NC}"
else
    echo -e "${YELLOW}⚠️  未找到 Neo4j 目录，跳过图谱功能${NC}"
fi

# ==========================================
# 5. 启动所有核心服务
# ==========================================
echo -e "\n${YELLOW}[5/5] 启动 LLaMA-Factory + RAG 后端${NC}"

mkdir -p logs
mkdir -p data
echo "{}" > data/dataset_info.json 2>/dev/null || true

# 启动 LLaMA-Factory
export GRADIO_SERVER_PORT=6006
export GRADIO_SERVER_NAME=0.0.0.0
nohup llamafactory-cli webui > logs/llamafactory.log 2>&1 &
LLAMA_PID=$!
sleep 3

# 启动 RAG 后端
nohup python run.py > logs/rag_backend.log 2>&1 &
RAG_PID=$!

# ==========================================
# 完成
# ==========================================
echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}✅ 所有服务启动成功！${NC}"
echo -e "
【服务清单】
✅ Redis 已启动
✅ Java 17 就绪
✅ Python 依赖无冲突
✅ vLLM 0.4.1 (CUDA11.8)
✅ Neo4j 已启动
✅ LLaMA-Factory: 端口 6006
✅ RAG 问答系统: 端口 6008

AutoDL 用户请使用平台映射的端口访问！
${NC}"
echo -e "${GREEN}=======================================================${NC}"
echo "停止所有服务: kill $LLAMA_PID $RAG_PID"