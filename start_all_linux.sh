#!/bin/bash
# ==============================================================================
# RAG 系统 - Linux 自动化部署与启动脚本
# 环境: Ubuntu 22.04 + Python 3.10 + PyTorch 2.1.2 + CUDA 11.8
# 说明: 支持幂等执行，已安装/已启动的服务会自动跳过
# ==============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志目录
LOGS_DIR="$PWD/logs"
mkdir -p "$LOGS_DIR"

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }

# 记录启动时间
echo "========================================" >> "$LOGS_DIR/setup.log"
echo "Setup started at: $(date)" >> "$LOGS_DIR/setup.log"
echo "========================================" >> "$LOGS_DIR/setup.log"

# ==============================================================================
# 打印欢迎信息
# ==============================================================================
echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}        RAG 知识库问答系统 - 自动化部署脚本          ${NC}"
echo -e "${GREEN}        Python 3.10 + PyTorch 2.1.2 + CUDA 11.8       ${NC}"
echo -e "${GREEN}=======================================================${NC}"

# ==============================================================================
# 检查是否为 root 用户（部分操作需要 sudo）
# ==============================================================================
SUDO_CMD=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO_CMD="sudo"
        log_info "检测到非 root 用户，将使用 sudo 执行需要权限的操作"
    else
        log_warn "非 root 用户且未安装 sudo，部分操作可能失败"
    fi
fi

# ==============================================================================
# 函数: 检查 Conda 是否已安装
# ==============================================================================
check_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# ==============================================================================
# 函数: 安装 Conda (Miniconda3)
# ==============================================================================
install_conda() {
    log_step "开始安装 Miniconda3..."
    
    # 下载 Miniconda3
    CONDA_INSTALLER="$HOME/miniconda3.sh"
    if [ ! -f "$CONDA_INSTALLER" ]; then
        log_info "下载 Miniconda3 安装包..."
        wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$CONDA_INSTALLER"
    fi
    
    # 安装
    log_info "执行 Miniconda3 安装..."
    bash "$CONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # 初始化 shell
    log_info "初始化 Conda..."
    "$HOME/miniconda3/bin/conda" init bash
    
    # 清理安装包
    rm -f "$CONDA_INSTALLER"
    
    # 重新加载配置
    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/.bashrc" 2>/dev/null || true
    
    log_info "Miniconda3 安装完成"
}

# ==============================================================================
# 函数: 检查 Conda 环境是否存在
# ==============================================================================
check_env_exists() {
    local env_name=$1
    conda env list | grep -q "^${env_name} "
}

# ==============================================================================
# 函数: 创建 Conda 环境
# ==============================================================================
create_conda_env() {
    local env_name=$1
    log_step "创建 Conda 环境: ${env_name} (Python 3.10)"
    
    # 使用阿里云镜像加速
    conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/ 2>/dev/null || true
    conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/ 2>/dev/null || true
    conda config --set show_channel_urls yes 2>/dev/null || true
    
    conda create -n "$env_name" python=3.10 -y
    log_info "Conda 环境 ${env_name} 创建完成"
}

# ==============================================================================
# 函数: 检查 PyTorch CUDA 版本
# ==============================================================================
check_pytorch_cuda() {
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; assert torch.version.cuda.startswith('11.8'), 'CUDA version mismatch'" 2>/dev/null
}

# ==============================================================================
# 函数: 安装 PyTorch (CUDA 11.8)
# ==============================================================================
install_pytorch() {
    log_step "安装 PyTorch 2.1.2 + CUDA 11.8..."
    
    # 使用阿里云镜像
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # 验证安装
    if check_pytorch_cuda; then
        log_info "PyTorch 2.1.2 + CUDA 11.8 安装成功"
        python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    else
        log_error "PyTorch CUDA 安装验证失败"
        exit 1
    fi
}

# ==============================================================================
# 函数: 安装 Python 依赖
# ==============================================================================
install_python_deps() {
    log_step "安装 Python 依赖 (requirements_online.txt)..."
    
    # 配置 pip 镜像
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true
    pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/ 2>/dev/null || true
    pip config set global.trusted-host "mirrors.aliyun.com pypi.tuna.tsinghua.edu.cn" 2>/dev/null || true
    
    # 安装依赖
    pip install -r requirements_online.txt
    
    log_info "Python 依赖安装完成"
}

# ==============================================================================
# 函数: 检查 vLLM 是否已安装
# ==============================================================================
check_vllm() {
    python -c "import vllm" 2>/dev/null
}

# 调试用: 显示 vLLM 导入错误
show_vllm_error() {
    python -c "import vllm" 2>&1 || true
}

# ==============================================================================
# 函数: 安装 vLLM (CUDA 11.8 预编译版)
# ==============================================================================
install_vllm() {
    log_step "安装 vLLM 0.4.1 (CUDA 11.8 预编译版)..."
    
    # 设置环境变量避免编译
    export VLLM_INSTALL_PUNICA_DROP_IN=0
    export VLLM_TARGET_DEVICE="cuda"
    
    # 1. 先安装 vLLM 的运行时依赖（不含 torch/xformers）
    log_step "安装 vLLM 运行时依赖..."
    pip install ray==2.9.3 outlines==0.0.34 nvidia-ml-py pydantic==2.7.4 \
        fastapi uvicorn tiktoken prometheus-client sentencepiece \
        lm-format-enforcer==0.9.8 msgpack
    
    # 2. 安装 vLLM 预编译版（--no-deps 防止拉取 torch 2.2.1 覆盖已有的 2.1.2+cu118）
    log_step "安装 vLLM 0.4.1+cu118 (--no-deps)..."
    pip install --no-deps https://github.com/vllm-project/vllm/releases/download/v0.4.1/vllm-0.4.1+cu118-cp310-cp310-manylinux1_x86_64.whl
    
    # 3. 安装与 PyTorch 2.1.2 匹配的 xformers（--no-deps）
    pip install xformers==0.0.23.post1 --no-deps
    
    if check_vllm; then
        log_info "vLLM 安装成功"
    else
        log_error "vLLM 导入失败，错误信息:"
        show_vllm_error
        exit 1
    fi
}

# ==============================================================================
# 函数: 检查 LLaMA-Factory 是否已安装
# ==============================================================================
check_llamafactory() {
    command -v llamafactory-cli >/dev/null 2>&1
}

# ==============================================================================
# 函数: 安装 LLaMA-Factory
# ==============================================================================
install_llamafactory() {
    log_step "安装 LLaMA-Factory 0.8.3..."
    
    # 使用 --no-deps 避免依赖冲突
    pip install llamafactory==0.8.3 --no-deps
    
    if check_llamafactory; then
        log_info "LLaMA-Factory 安装成功"
    else
        log_error "LLaMA-Factory 安装失败"
        exit 1
    fi
}

# ==============================================================================
# 函数: 检查 Redis 是否已安装
# ==============================================================================
check_redis() {
    command -v redis-server >/dev/null 2>&1
}

# ==============================================================================
# 函数: 安装并启动 Redis
# ==============================================================================
install_and_start_redis() {
    log_step "检查 Redis 服务..."
    
    # 检查是否已安装
    if ! check_redis; then
        log_info "Redis 未安装，开始安装..."
        $SUDO_CMD apt-get update -qq
        $SUDO_CMD apt-get install -y -qq redis-server redis-tools
    else
        log_info "Redis 已安装"
    fi
    
    # 启动 Redis
    if ! pgrep -x "redis-server" >/dev/null 2>&1; then
        log_info "启动 Redis 服务..."
        $SUDO_CMD systemctl enable redis-server 2>/dev/null || true
        $SUDO_CMD systemctl restart redis-server 2>/dev/null || $SUDO_CMD redis-server --daemonize yes
        sleep 2
    else
        log_info "Redis 服务已在运行"
    fi
    
    # 验证
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_info "Redis 服务运行正常"
    else
        log_warn "Redis 连接测试失败，尝试启动..."
        $SUDO_CMD redis-server --daemonize yes
    fi
}

# ==============================================================================
# 函数: 检查 Java 17
# ==============================================================================
check_java() {
    if command -v java >/dev/null 2>&1; then
        java -version 2>&1 | grep -q "17"
        return $?
    fi
    return 1
}

# ==============================================================================
# 函数: 安装 Java 17
# ==============================================================================
install_java() {
    log_step "检查 Java 17..."
    
    if check_java; then
        log_info "Java 17 已安装"
    else
        log_info "安装 OpenJDK 17..."
        $SUDO_CMD apt-get update -qq
        $SUDO_CMD apt-get install -y -qq openjdk-17-jre-headless
        log_info "Java 17 安装完成"
    fi
}

# ==============================================================================
# 函数: 启动 Neo4j
# ==============================================================================
start_neo4j() {
    log_step "启动 Neo4j 图数据库..."
    
    NEO4J_HOME="$PWD/model/neo4j-community-5.18.1"
    
    if [ ! -d "$NEO4J_HOME" ]; then
        log_warn "未找到 Neo4j 目录: $NEO4J_HOME"
        log_warn "图谱功能将不可用，如需使用请手动部署 Neo4j"
        return 0
    fi
    
    # 设置权限
    chmod +x "$NEO4J_HOME/bin/neo4j" 2>/dev/null || true
    chmod +x "$NEO4J_HOME/bin/cypher-shell" 2>/dev/null || true
    
    # 检查是否已在运行
    if pgrep -f "neo4j" >/dev/null 2>&1; then
        log_info "Neo4j 已在运行"
    else
        log_info "启动 Neo4j..."
        $NEO4J_HOME/bin/neo4j stop 2>/dev/null || true
        $NEO4J_HOME/bin/neo4j start
        sleep 5
    fi
    
    log_info "Neo4j 启动完成 (bolt://localhost:7687)"
}

# ==============================================================================
# 函数: 启动 RAG Flask 后端
# ==============================================================================
start_rag_backend() {
    log_step "启动 RAG Flask 后端服务..."
    
    # 检查端口是否被占用
    if lsof -Pi :6008 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "端口 6008 已被占用，尝试停止旧进程..."
        kill $(lsof -Pi :6008 -sTCP:LISTEN -t) 2>/dev/null || true
        sleep 2
    fi
    
    # 启动服务
    export APP_ENV=production
    nohup python app/main.py > "$LOGS_DIR/rag_backend.log" 2>&1 &
    RAG_PID=$!
    
    # 等待服务启动
    sleep 3
    
    # 验证启动
    if ps -p $RAG_PID > /dev/null 2>&1; then
        log_info "RAG 后端服务已启动 (PID: $RAG_PID, 端口: 6008)"
        echo $RAG_PID > "$LOGS_DIR/rag_backend.pid"
    else
        log_error "RAG 后端服务启动失败，请检查日志: $LOGS_DIR/rag_backend.log"
        return 1
    fi
}

# ==============================================================================
# 函数: 停止所有服务
# ==============================================================================
stop_all_services() {
    log_step "停止所有服务..."
    
    # 停止 RAG 后端
    if [ -f "$LOGS_DIR/rag_backend.pid" ]; then
        RAG_PID=$(cat "$LOGS_DIR/rag_backend.pid")
        if ps -p $RAG_PID > /dev/null 2>&1; then
            kill $RAG_PID 2>/dev/null || true
            log_info "已停止 RAG 后端服务 (PID: $RAG_PID)"
        fi
        rm -f "$LOGS_DIR/rag_backend.pid"
    fi
    
    # 停止 Neo4j
    NEO4J_HOME="$PWD/model/neo4j-community-5.18.1"
    if [ -d "$NEO4J_HOME" ]; then
        $NEO4J_HOME/bin/neo4j stop 2>/dev/null || true
        log_info "已停止 Neo4j"
    fi
    
    # 停止 Redis (可选，通常保持运行)
    # $SUDO_CMD systemctl stop redis-server 2>/dev/null || true
    
    log_info "所有服务已停止"
}

# ==============================================================================
# 函数: 显示服务状态
# ==============================================================================
show_service_status() {
    echo ""
    echo -e "${GREEN}=======================================================${NC}"
    echo -e "${GREEN}                  服务状态检查                        ${NC}"
    echo -e "${GREEN}=======================================================${NC}"
    
    # Redis
    if redis-cli ping 2>/dev/null | grep -q "PONG"; then
        echo -e "${GREEN}✅ Redis${NC}: 运行中"
    else
        echo -e "${RED}❌ Redis${NC}: 未运行"
    fi
    
    # Neo4j
    if pgrep -f "neo4j" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Neo4j${NC}: 运行中"
    else
        echo -e "${YELLOW}⚠️ Neo4j${NC}: 未运行"
    fi
    
    # RAG 后端
    if [ -f "$LOGS_DIR/rag_backend.pid" ]; then
        RAG_PID=$(cat "$LOGS_DIR/rag_backend.pid")
        if ps -p $RAG_PID > /dev/null 2>&1; then
            echo -e "${GREEN}✅ RAG 后端${NC}: 运行中 (PID: $RAG_PID, 端口: 6008)"
        else
            echo -e "${RED}❌ RAG 后端${NC}: 未运行"
        fi
    else
        echo -e "${RED}❌ RAG 后端${NC}: 未运行"
    fi
    
    echo -e "${GREEN}=======================================================${NC}"
}

# ==============================================================================
# 主程序
# ==============================================================================

# 处理命令行参数
case "${1:-}" in
    stop)
        stop_all_services
        exit 0
        ;;
    status)
        show_service_status
        exit 0
        ;;
    restart)
        stop_all_services
        sleep 2
        ;;
    *)
        # 继续执行安装和启动流程
        ;;
esac

# ==============================================================================
# 阶段 1: 系统依赖检查与安装
# ==============================================================================
log_step "阶段 1/3: 系统依赖检查与安装"

# 检查并安装 Conda
if check_conda; then
    log_info "Conda 已安装: $(conda --version)"
else
    log_warn "Conda 未安装，开始安装..."
    install_conda
fi

# 确保 Conda 可用
export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

# 检查并创建 Conda 环境
ENV_NAME="RAG"
if check_env_exists "$ENV_NAME"; then
    log_info "Conda 环境 '${ENV_NAME}' 已存在"
else
    create_conda_env "$ENV_NAME"
fi

# 激活环境
log_info "激活 Conda 环境: ${ENV_NAME}"
conda activate "$ENV_NAME" || source activate "$ENV_NAME"

# 检查并安装 PyTorch
if check_pytorch_cuda; then
    log_info "PyTorch + CUDA 11.8 已安装"
    python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"
else
    install_pytorch
fi

# ==============================================================================
# 阶段 2: Python 依赖安装
# ==============================================================================
log_step "阶段 2/3: Python 依赖安装"

# 安装 requirements
install_python_deps

# 检查并安装 vLLM
if check_vllm; then
    log_info "vLLM 已安装"
else
    install_vllm
fi

# 检查并安装 LLaMA-Factory
if check_llamafactory; then
    log_info "LLaMA-Factory 已安装"
else
    install_llamafactory
fi

# ==============================================================================
# 阶段 3: 服务启动
# ==============================================================================
log_step "阶段 3/3: 启动服务"

# 安装并启动 Redis
install_and_start_redis

# 安装 Java (Neo4j 需要)
install_java

# 启动 Neo4j
start_neo4j

# 启动 RAG 后端
start_rag_backend

# ==============================================================================
# 完成
# ==============================================================================
echo ""
echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}✅ 所有服务启动成功！                                  ${NC}"
echo -e "${GREEN}=======================================================${NC}"
echo ""
echo -e "【服务清单】"
show_service_status
echo ""
echo -e "【访问地址】"
echo -e "  RAG 后端:    http://localhost:6008"
echo -e "  Neo4j:       bolt://localhost:7687"
echo -e "  Redis:       localhost:6379"
echo ""
echo -e "【日志文件】"
echo -e "  部署日志:    $LOGS_DIR/setup.log"
echo -e "  后端日志:    $LOGS_DIR/rag_backend.log"
echo ""
echo -e "【管理命令】"
echo -e "  查看状态:    ./start_all_linux.sh status"
echo -e "  停止服务:    ./start_all_linux.sh stop"
echo -e "  重启服务:    ./start_all_linux.sh restart"
echo ""
echo -e "${GREEN}=======================================================${NC}"
