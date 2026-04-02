#!/bin/bash
# ==============================================================================
# RAG 系统 - 服务启动与管理脚本（适配 AutoDL root 环境）
# 环境: Ubuntu 22.04 / Python 3.12 / 依赖已预装
# 功能: 启动/停止/重启 Redis, Neo4j, RAG Flask 后端
# ==============================================================================

set -e  # 遇到错误退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志目录
LOGS_DIR="$PWD/logs"
mkdir -p "$LOGS_DIR"

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOGS_DIR/setup.log"; }

# 确保在项目根目录
cd "$(dirname "$0")"

# ==============================================================================
# 检查并激活 conda 环境
# ==============================================================================
activate_conda_env() {
    # 期望的环境名（可修改为你的实际环境名）
    TARGET_ENV="vllm-new"

    # 检查 conda 是否可用
    if ! command -v conda &> /dev/null; then
        log_error "Conda 未安装或不在 PATH 中，请先安装 Miniconda"
        exit 1
    fi

    # 初始化 conda（如果尚未初始化）
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi

    # 如果当前环境不是目标环境，则切换
    if [[ "$CONDA_DEFAULT_ENV" != "$TARGET_ENV" ]]; then
        log_info "切换到 Conda 环境: $TARGET_ENV"
        conda activate "$TARGET_ENV" || {
            log_error "无法激活环境 $TARGET_ENV，请确认环境存在"
            exit 1
        }
    else
        log_info "已处于 Conda 环境: $TARGET_ENV"
    fi

    # 验证关键包
    python -c "import vllm, langchain, faiss" 2>/dev/null || {
        log_error "关键依赖导入失败，请检查环境 $TARGET_ENV"
        exit 1
    }
}

# ==============================================================================
# 检查 Redis 并启动（root 用户无需 sudo）
# ==============================================================================
start_redis() {
    log_step "启动 Redis..."
    if pgrep -x "redis-server" >/dev/null; then
        log_info "Redis 已在运行"
        return 0
    fi

    if command -v redis-server &> /dev/null; then
        log_info "Redis 已安装，正在启动..."
        redis-server --daemonize yes
        sleep 2
        if redis-cli ping | grep -q PONG; then
            log_info "Redis 启动成功"
        else
            log_error "Redis 启动失败"
            return 1
        fi
    else
        log_warn "Redis 未安装，尝试安装..."
        apt-get update -qq
        apt-get install -y redis-server
        # 安装后启动
        redis-server --daemonize yes
        sleep 2
        if redis-cli ping | grep -q PONG; then
            log_info "Redis 安装并启动成功"
        else
            log_error "Redis 启动失败"
            return 1
        fi
    fi
}

# ==============================================================================
# 检查 Java 17
# ==============================================================================
ensure_java() {
    log_step "检查 Java 17..."
    if java -version 2>&1 | grep -q "version \"17"; then
        log_info "Java 17 已安装"
    else
        log_info "安装 OpenJDK 17..."
        apt-get update -qq
        apt-get install -y openjdk-17-jre-headless
        log_info "Java 17 安装完成"
    fi
}

# ==============================================================================
# 启动 Neo4j
# ==============================================================================
start_neo4j() {
    log_step "启动 Neo4j..."
    NEO4J_HOME="$PWD/model/neo4j-community-5.18.1"
    if [[ ! -d "$NEO4J_HOME" ]]; then
        log_warn "未找到 Neo4j 目录: $NEO4J_HOME，跳过启动"
        return 0
    fi

    chmod +x "$NEO4J_HOME/bin/neo4j" 2>/dev/null || true
    if pgrep -f "neo4j" >/dev/null; then
        log_info "Neo4j 已在运行"
    else
        "$NEO4J_HOME/bin/neo4j" start
        sleep 5
        log_info "Neo4j 启动完成 (bolt://localhost:7687)"
    fi
}

# ==============================================================================
# 启动 RAG 后端
# ==============================================================================
start_rag_backend() {
    log_step "启动 RAG 后端服务..."
    PORT=6008
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
        log_warn "端口 $PORT 已被占用，尝试停止旧进程..."
        kill -9 $(lsof -Pi :$PORT -sTCP:LISTEN -t) 2>/dev/null || true
        sleep 2
    fi

    export APP_ENV=production
    nohup python app/main.py > "$LOGS_DIR/rag_backend.log" 2>&1 &
    RAG_PID=$!
    sleep 3

    if ps -p $RAG_PID >/dev/null; then
        log_info "RAG 后端已启动 (PID: $RAG_PID, 端口: $PORT)"
        echo $RAG_PID > "$LOGS_DIR/rag_backend.pid"
    else
        log_error "RAG 后端启动失败，查看日志: $LOGS_DIR/rag_backend.log"
        return 1
    fi
}

# ==============================================================================
# 停止所有服务
# ==============================================================================
stop_services() {
    log_step "停止所有服务..."
    if [[ -f "$LOGS_DIR/rag_backend.pid" ]]; then
        kill $(cat "$LOGS_DIR/rag_backend.pid") 2>/dev/null && log_info "已停止 RAG 后端"
        rm -f "$LOGS_DIR/rag_backend.pid"
    fi

    NEO4J_HOME="$PWD/model/neo4j-community-5.18.1"
    if [[ -d "$NEO4J_HOME" ]]; then
        "$NEO4J_HOME/bin/neo4j" stop 2>/dev/null && log_info "已停止 Neo4j"
    fi

    # 可选：停止 Redis（一般不停止，避免影响其他）
    # redis-cli shutdown 2>/dev/null && log_info "已停止 Redis"
    log_info "服务已停止"
}

# ==============================================================================
# 显示状态
# ==============================================================================
show_status() {
    echo -e "\n${GREEN}========== 服务状态 ==========${NC}"
    redis-cli ping 2>/dev/null | grep -q PONG && echo -e "${GREEN}✅ Redis${NC}: 运行中" || echo -e "${RED}❌ Redis${NC}: 未运行"
    pgrep -f "neo4j" >/dev/null && echo -e "${GREEN}✅ Neo4j${NC}: 运行中" || echo -e "${YELLOW}⚠️ Neo4j${NC}: 未运行"
    if [[ -f "$LOGS_DIR/rag_backend.pid" ]] && ps -p $(cat "$LOGS_DIR/rag_backend.pid") >/dev/null; then
        echo -e "${GREEN}✅ RAG 后端${NC}: 运行中 (端口 6008)"
    else
        echo -e "${RED}❌ RAG 后端${NC}: 未运行"
    fi
}

# ==============================================================================
# 主入口
# ==============================================================================
case "${1:-}" in
    stop)
        stop_services
        exit 0
        ;;
    status)
        show_status
        exit 0
        ;;
    restart)
        stop_services
        sleep 2
        ;;
    start|"")
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

# 激活 conda 环境（必须）
activate_conda_env

# 启动服务
start_redis
ensure_java
start_neo4j
start_rag_backend

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}✅ 所有服务已启动${NC}"
show_status
echo -e "\n${GREEN}访问地址:${NC} http://localhost:6008"
echo -e "${GREEN}日志目录:${NC} $LOGS_DIR"
echo -e "${GREEN}管理命令:${NC} $0 stop | restart | status"
echo -e "${GREEN}========================================${NC}"