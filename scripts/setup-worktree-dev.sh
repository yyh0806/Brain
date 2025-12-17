#!/bin/bash

# Brain Project Worktree Development Setup Script
# This script sets up the development environment for parallel development using Git worktrees

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAIN_BRANCH="main"
LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models")

echo -e "${BLUE}Brain Project Worktree Development Setup${NC}"
echo "============================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command_exists git; then
    echo -e "${RED}Error: Git is not installed${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}Dependencies check passed${NC}"

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit: Brain project setup"
    echo -e "${GREEN}Git repository initialized${NC}"
fi

# Create development branches
echo -e "${YELLOW}Creating development branches...${NC}"

for layer in "${LAYERS[@]}"; do
    branch_name="${layer}-dev"
    if ! git show-ref --verify --quiet "refs/heads/$branch_name"; then
        git checkout -b "$branch_name"
        echo -e "${GREEN}Created branch: $branch_name${NC}"
    else
        echo -e "${BLUE}Branch $branch_name already exists${NC}"
    fi
    git checkout "$MAIN_BRANCH" 2>/dev/null || git checkout master
done

# Remove existing worktrees if they exist
echo -e "${YELLOW}Cleaning up existing worktrees...${NC}"

for layer in "${LAYERS[@]}"; do
    worktree_path="../brain-$layer"
    if [ -d "$worktree_path" ]; then
        git worktree remove "$worktree_path" 2>/dev/null || rm -rf "$worktree_path"
        echo -e "${BLUE}Removed existing worktree: $worktree_path${NC}"
    fi
done

# Create worktrees
echo -e "${YELLOW}Creating Git worktrees for parallel development...${NC}"

for layer in "${LAYERS[@]}"; do
    branch_name="${layer}-dev"
    worktree_path="../brain-$layer"

    git worktree add "$worktree_path" "$branch_name"
    echo -e "${GREEN}Created worktree: $worktree_path -> $branch_name${NC}"
done

# Display worktree information
echo -e "${YELLOW}Current Git worktrees:${NC}"
git worktree list

# Set up development environment in each worktree
echo -e "${YELLOW}Setting up development environments...${NC}"

for layer in "${LAYERS[@]}"; do
    worktree_path="../brain-$layer"

    echo -e "${BLUE}Setting up $layer layer environment...${NC}"

    # Create development scripts
    cat > "$worktree_path/dev.sh" << EOF
#!/bin/bash
# Development script for $layer layer

echo "Activating $layer layer development environment"

# Set PYTHONPATH
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"

# Source ROS environment if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Activating Python virtual environment..."
    source venv/bin/activate
fi

echo "$layer layer development environment ready!"
echo ""
echo "Available commands:"
echo "  python -m pytest brain/$layer/          # Run layer tests"
echo "  python -m flake8 brain/$layer/          # Code quality check"
echo "  python -m mypy brain/$layer/            # Type checking"
echo "  git status                               # Check git status"
echo "  git add . && git commit -m 'message'    # Commit changes"
EOF

    chmod +x "$worktree_path/dev.sh"

    # Create layer-specific test script
    cat > "$worktree_path/test-layer.sh" << EOF
#!/bin/bash
# Test script for $layer layer

set -e

echo "Running tests for $layer layer..."

# Activate environment
source ./dev.sh

# Run tests with coverage
python -m pytest brain/$layer/ \\
    -v \\
    --cov=brain.$layer \\
    --cov-report=term-missing \\
    --cov-report=html:htmlcov \\
    --cov-report=xml

echo "Tests completed! Coverage report available in htmlcov/"
EOF

    chmod +x "$worktree_path/test-layer.sh"

    echo -e "${GREEN}Set up $layer layer development environment${NC}"
done

# Create main development script
cat > "dev-main.sh" << EOF
#!/bin/bash
# Main development script for Brain project

echo "Brain Project Development Environment"
echo "====================================="

# Set PYTHONPATH
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"

# Source ROS environment if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Activating Python virtual environment..."
    source venv/bin/activate
fi

echo "Main development environment ready!"
echo ""
echo "Available commands:"
echo "  python -m pytest brain/                   # Run all tests"
echo "  python -m flake8 brain/                   # Code quality check"
echo "  python -m mypy brain/                     # Type checking"
echo "  git worktree list                         # List worktrees"
echo "  ./scripts/merge-layers.sh                 # Merge layer changes"
echo ""
echo "Layer-specific worktrees:"
for layer in "${LAYERS[@]}"; do
    echo "  cd ../brain-\$layer                    # Switch to \$layer layer"
done
EOF

chmod +x "dev-main.sh"

# Create merge script
cat > "scripts/merge-layers.sh" << EOF
#!/bin/bash
# Merge script for combining layer changes

set -e

MAIN_BRANCH="main"
LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models")

echo "Brain Project Layer Merge Script"
echo "================================="

# Switch to main branch
git checkout "\$MAIN_BRANCH"

# Pull latest changes
git pull origin "\$MAIN_BRANCH"

# Merge each layer branch
for layer in "\${LAYERS[@]}"; do
    branch_name="\${layer}-dev"
    echo -e "\${YELLOW}Merging \$branch_name...\${NC}"

    if git show-ref --verify --quiet "refs/heads/\$branch_name"; then
        git merge "\$branch_name" -m "Merge \$layer layer changes"
        echo -e "\${GREEN}Merged \$branch_name successfully\${NC}"
    else
        echo -e "\${BLUE}Branch \$branch_name not found, skipping\${NC}"
    fi
done

# Push changes
git push origin "\$MAIN_BRANCH"

echo -e "\${GREEN}All layers merged and pushed successfully!\${NC}"
EOF

chmod +x "scripts/merge-layers.sh"

# Create status script
cat > "scripts/status.sh" << EOF
#!/bin/bash
# Status script for checking development progress

set -e

MAIN_BRANCH="main"
LAYERS=("perception" "cognitive" "planning" "execution" "communication" "models")

echo "Brain Project Development Status"
echo "================================="

# Check main repository status
echo -e "\${BLUE}Main Repository Status:\${NC}"
git status --porcelain

echo ""
echo -e "\${BLUE}Worktree Status:\${NC}"
git worktree list

echo ""
echo -e "\${BLUE}Branch Status:\${NC}"
for layer in "\${LAYERS[@]}"; do
    branch_name="\${layer}-dev"
    if git show-ref --verify --quiet "refs/heads/\$branch_name"; then
        echo -e "\${GREEN}✓ \$branch_name exists\${NC}"
    else
        echo -e "\${RED}✗ \$branch_name missing\${NC}"
    fi
done

echo ""
echo -e "\${BLUE}Recent Commits:\${NC}"
git log --oneline -10

echo ""
echo -e "\${BLUE}Development Worktrees:\${NC}"
for layer in "\${LAYERS[@]}"; do
    worktree_path="../brain-\$layer"
    if [ -d "\$worktree_path" ]; then
        cd "\$worktree_path"
        status=\$(git status --porcelain)
        if [ -z "\$status" ]; then
            echo -e "\${GREEN}✓ \$layer: Clean\${NC}"
        else
            echo -e "\${YELLOW}⚠ \$layer: Has changes\${NC}"
        fi
        cd - > /dev/null
    else
        echo -e "\${RED}✗ \$layer: Worktree not found\${NC}"
    fi
done
EOF

chmod +x "scripts/status.sh"

echo -e "${GREEN}Development environment setup completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Activate main development environment:"
echo "   source ./dev-main.sh"
echo ""
echo "2. Switch to a specific layer:"
echo "   cd ../brain-perception    # or other layers"
echo "   source ./dev.sh"
echo ""
echo "3. Check development status:"
echo "   ./scripts/status.sh"
echo ""
echo "4. Run tests:"
echo "   ./test-layer.sh           # In layer worktree"
echo "   python -m pytest brain/   # In main repository"
echo ""
echo "5. Merge layer changes:"
echo "   ./scripts/merge-layers.sh"
echo ""
echo -e "${GREEN}Happy parallel development! 🚀${NC}"