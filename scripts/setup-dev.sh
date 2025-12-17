#!/bin/bash

echo "设置开发环境..."

# 创建虚拟环境
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements-dev.txt

echo "开发环境设置完成!"
echo "激活环境: source venv/bin/activate"
