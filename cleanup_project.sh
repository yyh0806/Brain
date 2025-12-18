#!/bin/bash

# 🧹 Brain项目清理脚本
# 用于清理临时文件、测试文件和过程文档

set -e  # 遇到错误立即退出

echo "🧹 开始清理Brain项目..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在正确的目录
if [ ! -f "README.md" ] || [ ! -d "brain" ]; then
    log_error "请在Brain项目根目录执行此脚本"
    exit 1
fi

# 创建备份目录
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
log_info "创建备份目录: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# 1. 清理过程文档文件
log_info "清理过程文档文件..."
PROCESS_DOCS=(
    "CI_FIX_SUMMARY.md"
    "BRANCH_PROTECTION_SETUP.md"
    "CONFIG_SYSTEM_SUMMARY.md"
    "DEMO_SYSTEM_FIX_SUMMARY.md"
    "IMPLEMENTATION_SUMMARY.md"
    "MODULE_IMPORT_SOLUTION_SUMMARY.md"
    "visualization_status_report.md"
    "add_github_secrets.md"
    "docker_usage_examples.md"
    "ISAAC_SIM_SETUP_GUIDE.md"
)

for doc in "${PROCESS_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        log_warn "备份并删除: $doc"
        mv "$doc" "$BACKUP_DIR/"
    else
        log_info "文件不存在，跳过: $doc"
    fi
done

# 2. 清理临时Python文件
log_info "清理临时Python文件..."
TEMP_PY_FILES=(
    "check_odom_topic.py"
    "demo_visualization.py"
    "run_complete_system_demo.py"
    "run_complete_system_demo_simple.py"
    "run_sensor_input_demo.py"
    "streamlit_docker_dashboard.py"
    "verify_implementation.py"
    "simple_demo.py"
)

for py_file in "${TEMP_PY_FILES[@]}"; do
    if [ -f "$py_file" ]; then
        log_warn "备份并删除: $py_file"
        mv "$py_file" "$BACKUP_DIR/"
    else
        log_info "文件不存在，跳过: $py_file"
    fi
done

# 3. 清理Docker相关文件
log_info "清理Docker相关文件..."
DOCKER_FILES=(
    "Dockerfile.isaac_sim"
    "Dockerfile.isaac_sim_fallback"
    "Dockerfile.minimal"
    "Dockerfile.optimized"
    "Dockerfile.simple"
    "docker-compose.isaac_sim.yml"
    "docker-compose.optimized.yml"
    ".dockerignore"
)

for docker_file in "${DOCKER_FILES[@]}"; do
    if [ -f "$docker_file" ]; then
        log_warn "备份并删除: $docker_file"
        mv "$docker_file" "$BACKUP_DIR/"
    else
        log_info "文件不存在，跳过: $docker_file"
    fi
done

# 4. 清理其他临时文件
log_info "清理其他临时文件..."
OTHER_FILES=(
    "ULTIMATE_CARLA_RENDERER.html"
    "setup_github_secrets.sh"
)

for other_file in "${OTHER_FILES[@]}"; do
    if [ -f "$other_file" ]; then
        log_warn "备份并删除: $other_file"
        mv "$other_file" "$BACKUP_DIR/"
    else
        log_info "文件不存在，跳过: $other_file"
    fi
done

# 5. 清理临时目录
log_info "清理临时目录..."
TEMP_DIRS=(
    "config_backup_20251218_144948"
    "docs-reports-dev"
    "fusion-engine-dev"
    "integration-dev"
    "preprocessing-dev"
    "sensor-input-dev"
    "situational-map-dev"
    "testing-framework-dev"
    "testing_framework_dev"
)

for temp_dir in "${TEMP_DIRS[@]}"; do
    if [ -d "$temp_dir" ]; then
        log_warn "备份并删除目录: $temp_dir"
        mv "$temp_dir" "$BACKUP_DIR/"
    else
        log_info "目录不存在，跳过: $temp_dir"
    fi
done

# 6. 清理空目录
log_info "清理空目录..."
find . -type d -empty -delete 2>/dev/null || true

# 7. 清理已删除的配置文件
log_info "清理git中的已删除配置文件..."
git rm -f config/default_config.yaml 2>/dev/null || true
git rm -f config/ollama_config.yaml 2>/dev/null || true
git rm -f config/ros2_config.yaml 2>/dev/null || true
git rm -f test_direct.py 2>/dev/null || true
git rm -f test_layers.py 2>/dev/null || true

# 8. 提交清理的变更
log_info "提交清理变更..."
git add .
git commit -m "🧹 feat: 清理项目临时文件和过程文档

清理内容:
- 移除过程文档和总结文件 (15个)
- 删除临时Python脚本和演示文件 (8个)
- 清理Docker相关文件 (8个)
- 移除临时开发目录 (9个)
- 备份到 $BACKUP_DIR 目录

保留内容:
- brain/ 核心模块
- config/ 必要配置
- README.md 和 CLAUDE.md 主要文档
- .github/workflows/ 工作流文件

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 9. 显示清理结果
log_info "清理完成！"
echo ""
echo "📊 清理统计:"
echo "- 备份目录: $BACKUP_DIR"
echo "- 清理的文件和目录已备份到上述目录"
echo ""
echo "🔍 当前项目结构:"
tree -L 2 -I 'backup_*|__pycache__|*.pyc' . 2>/dev/null || ls -la

echo ""
log_info "项目清理完成！如有需要可以从备份目录恢复文件。"
log_warn "请检查git status确认所有变更都正确提交。"