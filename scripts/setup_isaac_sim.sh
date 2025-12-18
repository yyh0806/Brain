#!/bin/bash

# Isaac Sim环境配置脚本
# 此脚本用于检查系统兼容性并配置Isaac Sim环境

set -e

echo "=== Isaac Sim环境配置脚本 ==="
echo "正在检查系统兼容性..."

# 检查Python版本
echo "1. 检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 检查Python版本兼容性
if [[ $python_version == 3.10.* ]]; then
    echo "✓ Python 3.10.x - 兼容Isaac Sim 4.x"
    isaac_sim_version="4.x"
elif [[ $python_version == 3.11.* ]]; then
    echo "✓ Python 3.11.x - 兼容Isaac Sim 5.x"
    isaac_sim_version="5.x"
else
    echo "❌ Python版本不兼容。需要Python 3.10.x 或 3.11.x"
    echo "请使用pyenv或conda安装正确的Python版本"
    exit 1
fi

# 检查GLIBC版本
echo "2. 检查GLIBC版本..."
glibc_version=$(ldd --version | head -n1 | awk '{print $NF}')
echo "当前GLIBC版本: $glibc_version"

# 检查GLIBC兼容性
if [[ $glibc_version == 2.35* ]] || [[ $glibc_version == 2.36* ]] || [[ $glibc_version == 2.37* ]] || [[ $glibc_version > 2.37 ]]; then
    echo "✓ GLIBC版本兼容"
else
    echo "⚠️  GLIBC版本可能不兼容。推荐2.35+"
    echo "如遇到问题，请考虑使用Docker容器"
fi

# 检查NVIDIA驱动
echo "3. 检查NVIDIA驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)
    echo "✓ NVIDIA驱动版本: $nvidia_driver"

    # 检查GPU
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
    echo "GPU: $gpu_name"

    # 检查VRAM
    vram_gb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    vram_gb=$((vram_gb / 1024))
    echo "VRAM: ${vram_gb}GB"

    if [ $vram_gb -ge 4 ]; then
        echo "✓ VRAM满足最低要求(4GB)"
    else
        echo "❌ VRAM不足。需要至少4GB VRAM"
        exit 1
    fi
else
    echo "❌ 未检测到NVIDIA驱动或GPU"
    echo "请安装NVIDIA驱动和CUDA"
    exit 1
fi

# 创建虚拟环境（如果不存在）
echo "4. 配置Python虚拟环境..."
venv_path="isaac_sim_env"

if [ ! -d "$venv_path" ]; then
    echo "创建虚拟环境: $venv_path"
    python3 -m venv $venv_path
else
    echo "虚拟环境已存在: $venv_path"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source $venv_path/bin/activate

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装基础依赖
echo "5. 安装基础依赖..."
pip install wheel setuptools

# 安装Isaac Sim（根据版本）
echo "6. 安装Isaac Sim..."
echo "注意：Isaac Sim安装可能需要较长时间..."

# 提供安装选项
echo ""
echo "请选择Isaac Sim安装方式："
echo "1) 通过pip安装（推荐用于开发）"
echo "2) 仅配置环境（手动安装）"
echo "3) 安装PyBullet作为轻量级替代"
echo ""

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo "通过pip安装Isaac Sim..."
        pip install isaacsim
        echo "✓ Isaac Sim安装完成"
        ;;
    2)
        echo "跳过Isaac Sim安装，仅配置环境..."
        echo "请手动从NVIDIA官网下载并安装Isaac Sim"
        ;;
    3)
        echo "安装PyBullet作为替代方案..."
        pip install pybullet
        echo "✓ PyBullet安装完成"
        ;;
    *)
        echo "无效选择，跳过Isaac Sim安装"
        ;;
esac

# 安装其他依赖
echo "7. 安装其他Python依赖..."
pip install numpy scipy
pip install opencv-python
pip install matplotlib

# 创建环境配置文件
echo "8. 创建环境配置..."
cat > isaac_sim_env.sh << 'EOF'
#!/bin/bash
# Isaac Sim环境激活脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 激活虚拟环境
source $SCRIPT_DIR/isaac_sim_env/bin/activate

# 设置Isaac Sim环境变量（如果通过pip安装）
export ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac_sim-*

# 添加Python路径
export PYTHONPATH=$ISAAC_SIM_PATH/kit/python:$PYTHONPATH

echo "Isaac Sim环境已激活"
echo "Python: $(which python)"
echo "Python版本: $(python --version)"
EOF

chmod +x isaac_sim_env.sh

# 创建测试脚本
echo "9. 创建测试脚本..."
cat > test_isaac_sim.py << 'EOF'
#!/usr/bin/env python3
"""
Isaac Sim安装测试脚本
"""

import sys
import os

def test_python_version():
    """测试Python版本兼容性"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor in [10, 11]:
        print("✓ Python版本兼容")
        return True
    else:
        print("❌ Python版本不兼容")
        return False

def test_isaac_sim_import():
    """测试Isaac Sim导入"""
    try:
        import isaacsim
        print("✓ Isaac Sim导入成功")
        return True
    except ImportError as e:
        print(f"❌ Isaac Sim导入失败: {e}")
        return False

def test_pybullet_import():
    """测试PyBullet导入（作为替代方案）"""
    try:
        import pybullet
        print("✓ PyBullet导入成功")
        return True
    except ImportError as e:
        print(f"❌ PyBullet导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== Isaac Sim环境测试 ===")

    # 测试Python版本
    python_ok = test_python_version()

    # 测试Isaac Sim
    if python_ok:
        isaac_ok = test_isaac_sim_import()

        # 如果Isaac Sim不可用，测试PyBullet
        if not isaac_ok:
            print("\n测试PyBullet作为替代方案...")
            pybullet_ok = test_pybullet_import()
        else:
            pybullet_ok = False
    else:
        isaac_ok = False
        pybullet_ok = False

    # 总结
    print("\n=== 测试总结 ===")
    print(f"Python兼容性: {'✓' if python_ok else '❌'}")
    print(f"Isaac Sim: {'✓' if isaac_ok else '❌'}")
    print(f"PyBullet: {'✓' if pybullet_ok else '❌'}")

    if isaac_ok:
        print("🎉 Isaac Sim环境配置成功！")
    elif pybullet_ok:
        print("⚠️  使用PyBullet作为仿真环境")
    else:
        print("❌ 仿真环境配置失败，请检查安装")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
EOF

chmod +x test_isaac_sim.py

echo ""
echo "=== Isaac Sim环境配置完成 ==="
echo ""
echo "下一步："
echo "1. 运行测试: ./test_isaac_sim.py"
echo "2. 激活环境: source isaac_sim_env.sh"
echo "3. 开始使用Isaac Sim接口"
echo ""
echo "配置文件位置："
echo "- 环境激活: isaac_sim_env.sh"
echo "- 测试脚本: test_isaac_sim.py"
echo "- 虚拟环境: isaac_sim_env/"
echo ""

# 运行测试
echo "是否现在运行测试？(y/n)"
read -p "> " run_test

if [[ $run_test == "y" ]] || [[ $run_test == "Y" ]]; then
    echo "运行Isaac Sim测试..."
    python3 test_isaac_sim.py
fi

echo "配置脚本执行完成！"