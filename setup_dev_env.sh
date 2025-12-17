#!/bin/bash

# Brain项目开发环境设置脚本
# 用于初始化Git Worktree开发环境

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查是否在正确的目录
if [ ! -d "brain" ]; then
    print_error "请在Brain项目根目录下运行此脚本"
    exit 1
fi

# 检查Git是否已初始化
if [ ! -d ".git" ]; then
    print_error "当前目录不是Git仓库，请先初始化Git"
    exit 1
fi

# 获取当前分支
CURRENT_BRANCH=$(git branch --show-current)
print_info "当前分支: $CURRENT_BRANCH"

# 创建develop分支（如果不存在）
if ! git show-ref --verify --quiet refs/heads/develop; then
    print_info "创建develop分支..."
    git checkout -b develop
    git push -u origin develop
else
    print_info "develop分支已存在"
fi

# 创建各层级的Worktree
print_info "创建各层级的Worktree..."

# 检查worktree目录是否存在
WORKTREE_BASE="../brain-worktrees"
if [ ! -d "$WORKTREE_BASE" ]; then
    mkdir -p "$WORKTREE_BASE"
fi

# 创建各层级的worktree
LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models" "core" "platforms" "recovery" "state" "utils" "visualization")

for layer in "${LAYERS[@]}"; do
    WORKTREE_PATH="$WORKTREE_BASE/brain-$layer"
    
    if [ ! -d "$WORKTREE_PATH" ]; then
        print_info "创建 $layer 层级的Worktree: $WORKTREE_PATH"
        git worktree add "$WORKTREE_PATH" "develop"
        
        # 创建层级开发分支（如果不存在）
        LAYER_BRANCH="${layer}-dev"
        if ! git show-ref --verify --quiet "refs/heads/$LAYER_BRANCH"; then
            cd "$WORKTREE_PATH"
            git checkout -b "$LAYER_BRANCH"
            git push -u "origin" "$LAYER_BRANCH"
            cd - > /dev/null
        else
            print_info "$LAYER_BRANCH 分支已存在"
        fi
    else
        print_warning "$layer 层级的Worktree已存在: $WORKTREE_PATH"
    fi
done

# 创建便捷脚本
print_info "创建便捷脚本..."

SCRIPT_DIR="./scripts"
mkdir -p "$SCRIPT_DIR"

# 创建切换到各层级的脚本
cat > "$SCRIPT_DIR/goto_layer.sh" << 'EOF'
#!/bin/bash

# 切换到指定层级的开发环境
# 用法: ./scripts/goto_layer.sh <layer_name>

set -e

LAYER_NAME=\$1

if [ -z "\$LAYER_NAME" ]; then
    echo "用法: \$0 <layer_name>"
    echo "可用层级: perception, cognitive, planning, execution, communication, models, core, platforms, recovery, state, utils, visualization"
    exit 1
fi

WORKTREE_PATH="../brain-worktrees/brain-\$LAYER_NAME"

if [ ! -d "\$WORKTREE_PATH" ]; then
    echo "错误: \$LAYER_NAME 层级的Worktree不存在"
    exit 1
fi

echo "切换到 \$LAYER_NAME 层级开发环境..."
cd "\$WORKTREE_PATH"
exec "\$SHELL"
EOF

chmod +x "$SCRIPT_DIR/goto_layer.sh"

# 创建同步所有层级的脚本
cat > "$SCRIPT_DIR/sync_all.sh" << 'EOF'
#!/bin/bash

# 同步所有层级的Worktree
set -e

echo "同步所有层级Worktree..."

LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models" "core" "platforms" "recovery" "state" "utils" "visualization")
WORKTREE_BASE="../brain-worktrees"

for layer in "\${LAYERS[@]}"; do
    WORKTREE_PATH="\$WORKTREE_BASE/brain-\$layer"
    
    if [ -d "\$WORKTREE_PATH" ]; then
        echo "同步 \$layer 层级..."
        cd "\$WORKTREE_PATH"
        git pull origin "\$layer-dev"
    fi
done

echo "所有层级同步完成"
EOF

chmod +x "$SCRIPT_DIR/sync_all.sh"

# 创建提交所有层级更改的脚本
cat > "$SCRIPT_DIR/commit_all.sh" << 'EOF'
#!/bin/bash

# 提交所有层级的更改
# 用法: ./scripts/commit_all.sh "<commit_message>"

set -e

COMMIT_MESSAGE=\$1

if [ -z "\$COMMIT_MESSAGE" ]; then
    echo "用法: \$0 \"<commit_message>\""
    exit 1
fi

LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models" "core" "platforms" "recovery" "state" "utils" "visualization")
WORKTREE_BASE="../brain-worktrees"

for layer in "\${LAYERS[@]}"; do
    WORKTREE_PATH="\$WORKTREE_BASE/brain-\$layer"
    
    if [ -d "\$WORKTREE_PATH" ]; then
        echo "检查 \$layer 层级..."
        cd "\$WORKTREE_PATH"
        
        # 检查是否有未提交的更改
        if [ -n "\$(git status --porcelain)" ]; then
            echo "提交 \$layer 层级的更改..."
            git add .
            git commit -m "\$COMMIT_MESSAGE"
            git push origin "\$layer-dev"
        else
            echo "\$layer 层级没有未提交的更改"
        fi
    fi
done

echo "所有层级提交完成"
EOF

chmod +x "$SCRIPT_DIR/commit_all.sh"

# 创建合并到develop分支的脚本
cat > "$SCRIPT_DIR/merge_to_develop.sh" << 'EOF'
#!/bin/bash

# 合并所有层级到develop分支
set -e

echo "合并所有层级到develop分支..."

# 切换到主仓库
cd ..

# 拉取最新代码
git checkout develop
git pull origin develop

# 合并各层级分支
LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models" "core" "platforms" "recovery" "state" "utils" "visualization")

for layer in "\${LAYERS[@]}"; do
    LAYER_BRANCH="\$layer-dev"
    echo "合并 \$layer 层级..."
    git merge origin/\$LAYER_BRANCH --no-ff
done

# 推送到远程
git push origin develop

echo "所有层级已合并到develop分支"
EOF

chmod +x "$SCRIPT_DIR/merge_to_develop.sh"

print_info "开发环境设置完成！"
print_info ""
print_info "可用的便捷脚本："
print_info "  ./scripts/goto_layer.sh <layer>     - 切换到指定层级开发环境"
print_info "  ./scripts/sync_all.sh              - 同步所有层级Worktree"
print_info "  ./scripts/commit_all.sh <message>    - 提交所有层级的更改"
print_info "  ./scripts/merge_to_develop.sh       - 合并所有层级到develop分支"
print_info ""
print_info "示例用法："
print_info "  ./scripts/goto_layer.sh perception  # 切换到感知层开发环境"
print_info "  ./scripts/commit_all.sh \"feat: add new feature\"  # 提交所有层级更改"
