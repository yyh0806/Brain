# BRAIN Isaac Sim Docker 使用示例

## 🐳 Docker 快速使用指南

### 1. 基础使用 - 快速测试

```bash
# 🚀 快速运行World Model演示
docker run --rm \
    --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/logs:/workspace/logs \
    brain-simple:latest \
    python3 run_complete_system_demo.py --mode quick

# 📊 结果: ✅ World Model系统成功运行，所有测试通过
```

### 2. 交互式开发环境

```bash
# 🛠️ 启动交互式开发容器
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace/brain \
    -v $(pwd)/data:/workspace/data \
    -p 8888:8888 \
    --name brain-dev \
    brain-simple:latest \
    /bin/bash

# 在容器内:
root@container:/workspace/brain# python3 run_complete_system_demo.py --mode full
root@container:/workspace/brain# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### 3. Web服务部署

#### Jupyter Lab 服务
```bash
# 📓 启动Jupyter Lab (端口8889)
docker run -d \
    --gpus all \
    --name brain-jupyter \
    -v $(pwd):/workspace/brain \
    -p 8889:8888 \
    brain-simple:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='brain2024'

# 访问: http://localhost:8889?token=brain2024
```

#### Streamlit Dashboard
```bash
# 📊 启动Streamlit监控面板 (端口8502)
docker run -d \
    --gpus all \
    --name brain-dashboard \
    -v $(pwd):/workspace/brain \
    -p 8502:8501 \
    brain-simple:latest \
    streamlit run streamlit_docker_dashboard.py --server.port=8501 --server.address=0.0.0.0

# 访问: http://localhost:8502
```

### 4. 使用自动化脚本

```bash
# 🤖 一键运行演示
./scripts/docker_quick_start.sh --demo

# 🐚 一键启动Jupyter
./scripts/docker_quick_start.sh --jupyter

# 🔧 一键启动交互式环境
./scripts/docker_quick_start.sh --interactive

# 🏗️ 构建并运行
./scripts/docker_quick_start.sh --build --run
```

### 5. Docker Compose 部署

```bash
# 🎭 启动完整服务栈
docker-compose -f docker-compose.isaac_sim.yml up -d

# 📊 启动特定服务
docker-compose -f docker-compose.isaac_sim.yml up -d jupyter-lab
docker-compose -f docker-compose.isaac_sim.yml up -d streamlit-dashboard

# 🛑 停止所有服务
docker-compose -f docker-compose.isaac_sim.yml down

# 🧹 清理
docker-compose -f docker-compose.isaac_sim.yml down -v --rmi all
```

## 🔧 高级配置示例

### GPU内存限制
```bash
# 💾 限制GPU内存使用
docker run --rm \
    --gpus '"device=0,1"' \
    --env NVIDIA_VISIBLE_DEVICES=0,1 \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    brain-simple:latest \
    python3 run_complete_system_demo.py
```

### 性能优化配置
```bash
# ⚡ 高性能配置
docker run --rm \
    --gpus all \
    --shm-size=2gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace/brain \
    brain-simple:latest \
    python3 run_complete_system_demo.py --mode full
```

### 开发环境配置
```bash
# 👨‍💻 完整开发环境
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace/brain \
    -v $(pwd)/docs:/workspace/docs \
    -v $(pwd)/tests:/workspace/tests \
    -p 8888:8888 \
    -p 8501:8501 \
    -p 6006:6006 \
    --env PYTHONPATH=/workspace/brain \
    brain-simple:latest \
    /bin/bash
```

## 🎯 测试场景

### 1. 系统集成测试
```bash
# 🔬 完整系统测试
docker run --rm \
    --gpus all \
    brain-simple:latest \
    bash -c "
    python3 run_complete_system_demo.py --mode full &&
    python3 -c 'import numpy; print(f\"NumPy: {numpy.__version__}\")' &&
    python3 -c 'import pydantic; print(f\"Pydantic: {pydantic.__version__}\")'
"
```

### 2. 性能基准测试
```bash
# 📈 性能测试
docker run --rm \
    --gpus all \
    brain-simple:latest \
    python3 -c "
import time
import numpy as np
from brain.cognitive.world_model import WorldModel

# 测试World Model性能
wm = WorldModel()
start_time = time.time()

for i in range(100):
    wm.update_context({'test_data': np.random.rand(100)})

elapsed = time.time() - start_time
print(f'100次更新耗时: {elapsed:.3f}秒')
print(f'平均每次更新: {elapsed/100*1000:.2f}毫秒')
"
```

### 3. GPU测试
```bash
# 🎮 GPU功能测试
docker run --rm \
    --gpus all \
    brain-simple:latest \
    python3 -c "
try:
    import torch
    print(f'✅ PyTorch CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'🚀 GPU设备: {torch.cuda.get_device_name(0)}')
        print(f'💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
except ImportError:
    print('⚠️ PyTorch未安装，使用CPU模式')

import numpy as np
print(f'✅ NumPy版本: {np.__version__}')
print('✅ 基础科学计算环境正常')
"
```

## 📊 监控和调试

### 容器资源监控
```bash
# 📊 实时资源使用
docker stats brain-simple

# 📋 容器详细信息
docker inspect brain-simple

# 📜 容器日志
docker logs -f brain-simple

# 🔧 进入运行中的容器
docker exec -it brain-simple /bin/bash
```

### 系统健康检查
```bash
# 🏥 健康检查脚本
docker run --rm \
    brain-simple:latest \
    bash -c "
echo '=== Python环境检查 ==='
python3 --version
pip list | grep -E '(numpy|pydantic|yaml|loguru)'

echo '=== 系统组件检查 ==='
ls -la /workspace/brain/brain/
echo '=== 内存检查 ==='
free -h
echo '=== GPU检查 ==='
nvidia-smi || echo 'GPU不可用'
"
```

## 🚀 生产部署

### 环境变量配置
```bash
# 🏭 生产环境配置
docker run -d \
    --gpus all \
    --name brain-prod \
    --restart unless-stopped \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/logs:/workspace/logs \
    -v $(pwd)/config:/workspace/config \
    -p 8888:8888 \
    -p 8501:8501 \
    -e ENVIRONMENT=production \
    -e LOG_LEVEL=INFO \
    -e PYTHONPATH=/workspace/brain \
    brain-simple:latest
```

### 多实例负载均衡
```bash
# ⚖️ 启动多个实例
for i in {1..3}; do
    docker run -d \
        --gpus all \
        --name brain-worker-$i \
        -v $(pwd)/data:/workspace/data \
        -p $((8888+i)):8888 \
        brain-simple:latest
done
```

## 🛠️ 故障排除

### 常见问题解决
```bash
# 🔍 检查Docker和GPU支持
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi

# 🔍 检查镜像构建
docker history brain-simple:latest

# 🔍 检查端口占用
netstat -tulpn | grep :8888

# 🔍 清理Docker资源
docker system prune -f
docker volume prune -f
```

### 性能优化技巧
```bash
# 🚀 使用缓存卷加速构建
docker build \
    --cache-from brain-simple:latest \
    -f Dockerfile.simple \
    -t brain-simple:latest .

# 💾 使用tmpfs提升IO性能
docker run --rm \
    --tmpfs /tmp:rw,noexec,nosuid,size=1g \
    brain-simple:latest \
    python3 run_complete_system_demo.py
```

## 📈 成功案例

### 典型使用流程
```bash
# 1️⃣ 构建镜像
docker build -f Dockerfile.simple -t brain-simple:latest .

# 2️⃣ 测试基础功能
docker run --rm brain-simple:latest python3 run_complete_system_demo.py --mode quick

# 3️⃣ 启动开发环境
docker run -it --rm -v $(pwd):/workspace/brain brain-simple:latest /bin/bash

# 4️⃣ 部署Web服务
docker run -d -p 8502:8501 brain-simple:latest streamlit run streamlit_docker_dashboard.py --server.port=8501

# 5️⃣ 监控服务状态
docker ps && docker stats $(docker ps -q)
```

---

## 🎯 总结

✅ **成功验证的功能**:
- World Model系统完整运行
- Jupyter Lab交互式开发环境
- Streamlit实时监控面板
- GPU加速支持 (NVIDIA RTX 3090)
- 自动化脚本和Docker Compose编排
- 完整的开发到部署工作流

🚀 **系统特点**:
- 容器化部署，环境一致性
- GPU加速，高性能计算
- 模块化设计，灵活扩展
- 自动化运维，简化管理
- 企业级配置，生产就绪

🎉 **使用建议**:
- 开发测试: 使用 `brain-simple:latest`
- 生产部署: 使用 Docker Compose 编排
- 性能调优: 配置GPU和内存限制
- 监控运维: 使用自动化脚本和健康检查