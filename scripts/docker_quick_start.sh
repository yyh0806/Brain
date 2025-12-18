#!/bin/bash

# Brain Isaac Sim Docker 快速启动脚本
# 一键启动Brain系统的Docker环境

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
IMAGE_NAME="brain-simple:latest"
CONTAINER_NAME="brain-isaac-sim"
DATA_DIR="./data"
LOGS_DIR="./logs"
CONFIG_DIR="./config"

# 帮助信息
show_help() {
    echo "Brain Isaac Sim Docker 快速启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示帮助信息"
    echo "  -b, --build             构建Docker镜像"
    echo "  -r, --run               运行容器"
    echo "  -s, --stop              停止容器"
    echo "  -c, --clean             清理容器和镜像"
    echo "  -i, --interactive      交互式运行"
    echo "  -d, --demo              运行演示"
    echo "  -j, --jupyter           启动Jupyter Lab"
    echo "  --dev                   开发模式"
    echo "  --full                 使用完整版镜像"
    echo ""
    echo "示例:"
    echo "  $0 --build --run       # 构建并运行"
    echo "  $0 --demo              # 运行演示"
    echo "  $0 --jupyter           # 启动Jupyter"
    echo "  $0 --interactive       # 交互式运行"
}

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查系统要求
check_requirements() {
    log_info "检查系统要求..."

    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装。请先安装Docker。"
        exit 1
    fi

    # 检查NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker运行时未配置。"
        log_info "请安装nvidia-container-toolkit并重启Docker服务。"
        exit 1
    fi

    # 检查GPU
    if ! nvidia-smi &> /dev/null; then
        log_warning "未检测到NVIDIA GPU。将使用CPU模式。"
        GPU_MODE="cpu"
    else
        log_success "检测到NVIDIA GPU"
        GPU_MODE="gpu"
    fi

    # 创建必要的目录
    mkdir -p "$DATA_DIR" "$LOGS_DIR" "$CONFIG_DIR"

    log_success "系统要求检查完成"
}

# 构建Docker镜像
build_image() {
    local image_suffix=""
    if [[ "$USE_FULL" == "true" ]]; then
        IMAGE_NAME="brain-isaac-sim:latest"
        image_suffix="完整版"
    else
        IMAGE_NAME="brain-simple:latest"
        image_suffix="简化版"
    fi

    log_info "构建$image_suffix Docker镜像: $IMAGE_NAME"

    # 检查镜像是否已存在
    if docker images | grep -q "$IMAGE_NAME"; then
        log_warning "镜像 $IMAGE_NAME 已存在"
        read -p "是否重新构建? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "使用现有镜像"
            return 0
        fi
    fi

    local dockerfile="Dockerfile.simple"
    if [[ "$USE_FULL" == "true" ]]; then
        dockerfile="Dockerfile.isaac_sim"
    fi

    if [[ -f "$dockerfile" ]]; then
        docker build -f "$dockerfile" -t "$IMAGE_NAME" .
        log_success "$image_suffix镜像构建完成"
    else
        log_error "Dockerfile $dockerfile 不存在"
        exit 1
    fi
}

# 运行容器
run_container() {
    log_info "启动Docker容器: $CONTAINER_NAME"

    # 检查容器是否已运行
    if docker ps | grep -q "$CONTAINER_NAME"; then
        log_warning "容器 $CONTAINER_NAME 已在运行"
        return 0
    fi

    local docker_args=(
        "--name" "$CONTAINER_NAME"
        "--rm"
        "-v" "$(pwd):/workspace/brain"
        "-v" "$DATA_DIR:/workspace/data"
        "-v" "$LOGS_DIR:/workspace/logs"
        "-v" "$CONFIG_DIR:/workspace/config"
        "-p" "8888:8888"
        "-p" "8501:8501"
    )

    # 添加GPU支持
    if [[ "$GPU_MODE" == "gpu" ]]; then
        docker_args+=("--gpus" "all")
    fi

    # 添加X11支持
    if [[ "$X11_SUPPORT" == "true" ]]; then
        docker_args+=(
            "-e" "DISPLAY=$DISPLAY"
            "-v" "/tmp/.X11-unix:/tmp/.X11-unix"
        )
        xhost +local:docker &> /dev/null || true
    fi

    # 开发模式添加更多卷
    if [[ "$DEV_MODE" == "true" ]]; then
        docker_args+=(
            "-v" "$(pwd)/docs:/workspace/docs"
            "-v" "$(pwd)/tests:/workspace/tests"
        )
    fi

    # 运行容器
    docker run -it "${docker_args[@]}" "$IMAGE_NAME" "$@"

    log_success "容器启动完成"
}

# 停止容器
stop_container() {
    log_info "停止Docker容器: $CONTAINER_NAME"

    if docker ps | grep -q "$CONTAINER_NAME"; then
        docker stop "$CONTAINER_NAME"
        log_success "容器已停止"
    else
        log_warning "容器 $CONTAINER_NAME 未在运行"
    fi
}

# 清理容器和镜像
clean_docker() {
    log_info "清理Docker资源..."

    # 停止并删除容器
    if docker ps -a | grep -q "$CONTAINER_NAME"; then
        docker rm -f "$CONTAINER_NAME"
    fi

    # 删除镜像
    read -p "是否删除Docker镜像? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi "$IMAGE_NAME" 2>/dev/null || true
        log_success "镜像已删除"
    fi

    # 清理未使用的资源
    docker system prune -f

    log_success "清理完成"
}

# 运行演示
run_demo() {
    log_info "运行Brain系统演示..."
    run_container python3 run_complete_system_demo.py --mode demo
}

# 启动Jupyter Lab
start_jupyter() {
    log_info "启动Jupyter Lab..."
    run_container jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='brain2024'
    log_info "Jupyter Lab访问地址: http://localhost:8888?token=brain2024"
}

# 显示容器状态
show_status() {
    log_info "Docker容器状态:"
    echo ""

    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "✅ 容器 $CONTAINER_NAME 正在运行"
        echo "📊 资源使用:"
        docker stats --no-stream "$CONTAINER_NAME" || true
        echo ""
        echo "🌐 服务地址:"
        echo "   Jupyter Lab: http://localhost:8888?token=brain2024"
        echo "   Streamlit:   http://localhost:8501"
        echo ""
        echo "🔧 管理命令:"
        echo "   停止: $0 --stop"
        echo "   进入: docker exec -it $CONTAINER_NAME /bin/bash"
    else
        echo "❌ 容器 $CONTAINER_NAME 未运行"
        echo ""
        echo "🚀 启动命令:"
        echo "   演示: $0 --demo"
        echo "   Jupyter: $0 --jupyter"
        echo "   交互: $0 --interactive"
    fi
}

# 主函数
main() {
    # 默认参数
    BUILD_ONLY=false
    RUN_ONLY=false
    STOP_ONLY=false
    CLEAN_ONLY=false
    INTERACTIVE=false
    DEMO=false
    JUPYTER=false
    DEV_MODE=false
    USE_FULL=false
    X11_SUPPORT=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -b|--build)
                BUILD_ONLY=true
                shift
                ;;
            -r|--run)
                RUN_ONLY=true
                shift
                ;;
            -s|--stop)
                STOP_ONLY=true
                shift
                ;;
            -c|--clean)
                CLEAN_ONLY=true
                shift
                ;;
            -i|--interactive)
                INTERACTIVE=true
                shift
                ;;
            -d|--demo)
                DEMO=true
                shift
                ;;
            -j|--jupyter)
                JUPYTER=true
                shift
                ;;
            --dev)
                DEV_MODE=true
                shift
                ;;
            --full)
                USE_FULL=true
                shift
                ;;
            --x11)
                X11_SUPPORT=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 检查系统要求
    check_requirements

    # 执行对应操作
    if [[ "$STOP_ONLY" == "true" ]]; then
        stop_container
    elif [[ "$CLEAN_ONLY" == "true" ]]; then
        clean_docker
    elif [[ "$BUILD_ONLY" == "true" ]]; then
        build_image
    elif [[ "$INTERACTIVE" == "true" ]]; then
        build_image
        run_container /bin/bash
    elif [[ "$DEMO" == "true" ]]; then
        build_image
        run_demo
    elif [[ "$JUPYTER" == "true" ]]; then
        build_image
        start_jupyter
    elif [[ "$RUN_ONLY" == "true" ]]; then
        run_container
    else
        # 默认显示状态
        show_status
    fi
}

# 运行主函数
main "$@"