#!/bin/bash

echo "运行测试套件..."

# 激活虚拟环境
source venv/bin/activate

# 运行代码格式检查
echo "代码格式检查..."
black --check brain/ tests/

# 运行代码质量检查
echo "代码质量检查..."
flake8 brain/ tests/

# 运行类型检查
echo "类型检查..."
mypy brain/

# 运行单元测试
echo "单元测试..."
python -m pytest tests/unit/ -v --cov=brain --cov-report=html

# 运行集成测试
echo "集成测试..."
python -m pytest tests/integration/ -v

echo "测试完成! 查看覆盖率报告: htmlcov/index.html"
