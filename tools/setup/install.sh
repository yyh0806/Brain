#!/bin/bash
# Brain 系统安装脚本

set -e

echo "=========================================="
echo "Brain 系统安装脚本"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 升级pip
echo "升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 安装Brain包（开发模式）
echo "安装 Brain 包..."
pip install -e .

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "1. 激活虚拟环境: source venv/bin/activate"
echo "2. 运行示例: python examples/basic_usage.py"
echo "3. 或直接运行: python3 examples/basic_usage.py"
echo ""


