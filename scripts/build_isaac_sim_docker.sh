#!/bin/bash

# Isaac Sim Docker构建脚本
# 用于构建和运行Isaac Sim集成的Brain系统Docker环境

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装。请先安装Docker Compose。"
        exit 1
    fi

    # 检查NVIDIA Docker运行时
    if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log_error "NVIDIA Docker运行时未配置或无法访问GPU。"
        log_info "请确保已安装NVIDIA驱动和Docker支持。"
        exit 1
    fi

    # 检查GPU
    if ! nvidia-smi &> /dev/null; then
        log_warning "未检测到NVIDIA GPU或驱动。Isaac Sim需要GPU支持。"
    fi

    log_success "系统要求检查完成"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."

    mkdir -p data
    mkdir -p logs
    mkdir -p config/docker
    mkdir -p outputs

    # 创建Docker专用的配置文件
    cat > config/docker/isaac_sim_docker_config.yaml << 'EOF'
# Isaac Sim Docker配置
isaac_sim:
  enabled: true
  docker_mode: true

  # Isaac Sim路径 (Docker内路径)
  isaac_sim_path: "/opt/nvidia/isaac_sim"

  # 仿真环境设置
  simulation:
    headless: false  # Docker中建议设为true，除非有X11转发
    physics_engine: "physx"
    gpu_device_id: 0

  # 数据存储路径 (Docker内路径)
  data_paths:
    output_dir: "/workspace/outputs"
    temp_dir: "/tmp/isaac_sim"

  # 网络设置
  networking:
    web_port: 49000
    enable_web: true

  # 资源限制
  resources:
    max_memory: "16GB"
    gpu_memory_fraction: 0.8

brain_system:
  # Brain系统Docker特定配置
  workspace_path: "/workspace/brain"
  python_path: "/workspace/brain"

  # 数据卷映射
  volumes:
    - "./data:/workspace/data"
    - "./logs:/workspace/logs"
    - "./config:/workspace/config"
    - "./outputs:/workspace/outputs"

logging:
  level: "INFO"
  file_path: "/workspace/logs/isaac_sim_docker.log"
  max_file_size: "100MB"
  backup_count: 5

development:
  # 开发模式设置
  debug_mode: true
  auto_reload: true
  jupyter_enabled: true
  jupyter_port: 8888
  jupyter_token: "brain2024"
EOF

    log_success "目录创建完成"
}

# 构建Docker镜像
build_docker_image() {
    log_info "构建Isaac Sim Docker镜像..."

    # 设置构建参数
    export DOCKER_BUILDKIT=1

    # 构建镜像
    docker build \
        -f Dockerfile.isaac_sim \
        -t brain-isaac-sim:latest \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        .

    if [ $? -eq 0 ]; then
        log_success "Docker镜像构建完成"
    else
        log_error "Docker镜像构建失败"
        exit 1
    fi
}

# 运行Docker容器
run_docker_container() {
    local service_name=${1:-"isaac-sim-brain"}

    log_info "启动Docker容器: $service_name"

    # 检查X11显示
    if [ -z "$DISPLAY" ]; then
        log_warning "未设置DISPLAY环境变量。将使用headless模式。"
        export DISPLAY=:0
    fi

    # 授权X11连接
    xhost +local:docker &>/dev/null || true

    # 启动容器
    docker-compose -f docker-compose.isaac_sim.yml up -d "$service_name"

    if [ $? -eq 0 ]; then
        log_success "Docker容器启动成功"

        case "$service_name" in
            "isaac-sim-brain")
                log_info "进入容器命令: docker exec -it brain-isaac-sim /bin/bash"
                ;;
            "jupyter-lab")
                log_info "Jupyter Lab访问: http://localhost:8889?token=brain2024"
                ;;
            "streamlit-dashboard")
                log_info "Streamlit Dashboard访问: http://localhost:8501"
                ;;
        esac
    else
        log_error "Docker容器启动失败"
        exit 1
    fi
}

# 停止Docker容器
stop_docker_container() {
    log_info "停止Docker容器..."

    docker-compose -f docker-compose.isaac_sim.yml down

    log_success "Docker容器已停止"
}

# 清理Docker资源
clean_docker_resources() {
    log_info "清理Docker资源..."

    # 停止容器
    docker-compose -f docker-compose.isaac_sim.yml down

    # 删除镜像
    docker rmi brain-isaac-sim:latest 2>/dev/null || true

    # 清理未使用的资源
    docker system prune -f

    log_success "Docker资源清理完成"
}

# 测试Docker环境
test_docker_environment() {
    log_info "测试Docker环境..."

    # 运行简单测试
    docker exec brain-isaac-sim python3 -c "
import sys
sys.path.append('/workspace/brain')
print('✓ Python环境测试通过')

try:
    import pybullet
    print('✓ PyBullet导入成功')
except ImportError:
    print('✗ PyBullet导入失败')

try:
    import brain
    print('✓ Brain模块导入成功')
except ImportError as e:
    print(f'✗ Brain模块导入失败: {e}')

print('Docker环境测试完成')
"

    if [ $? -eq 0 ]; then
        log_success "Docker环境测试通过"
    else
        log_error "Docker环境测试失败"
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "Isaac Sim Docker构建脚本"
    echo ""
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  build                   构建Docker镜像"
    echo "  run [service]           运行Docker容器"
    echo "  stop                    停止Docker容器"
    echo "  test                    测试Docker环境"
    echo "  clean                   清理Docker资源"
    echo "  all                     执行完整流程 (check + build + run)"
    echo ""
    echo "服务名称:"
    echo "  isaac-sim-brain         主Isaac Sim容器 (默认)"
    echo "  jupyter-lab             Jupyter Lab服务"
    echo "  streamlit-dashboard     Streamlit Dashboard服务"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -v, --verbose           详细输出"
    echo ""
    echo "示例:"
    echo "  $0 all                  # 执行完整流程"
    echo "  $0 build                # 仅构建镜像"
    echo "  $0 run jupyter-lab      # 运行Jupyter Lab服务"
    echo "  $0 test                 # 测试Docker环境"
}

# 主函数
main() {
    local command=""
    local service_name=""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            build|run|stop|test|clean|all)
                command="$1"
                shift
                ;;
            isaac-sim-brain|jupyter-lab|streamlit-dashboard)
                service_name="$1"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 执行命令
    case "$command" in
        "check"|"")
            check_requirements
            ;;
        "build")
            check_requirements
            create_directories
            build_docker_image
            ;;
        "run")
            run_docker_container "$service_name"
            ;;
        "stop")
            stop_docker_container
            ;;
        "test")
            test_docker_environment
            ;;
        "clean")
            clean_docker_resources
            ;;
        "all")
            check_requirements
            create_directories
            build_docker_image
            run_docker_container "$service_name"
            test_docker_environment
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"