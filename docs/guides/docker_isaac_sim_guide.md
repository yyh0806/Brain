# Isaac Sim Docker集成指南

## 📋 概述

本指南帮助您使用Docker快速部署Isaac Sim仿真环境，并集成到您的World Model系统中。

## 🎯 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX系列（Volta架构或更新版本）
- **VRAM**: 最低4GB，推荐8GB+
- **内存**: 16GB+ RAM
- **存储**: 50GB+ 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+推荐)
- **NVIDIA驱动**: 515.65.01+
- **Docker**: 20.10+
- **NVIDIA Container Toolkit**: 已安装并配置

## 🚀 快速开始

### 1. 安装NVIDIA Container Toolkit

```bash
# 添加NVIDIA包仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 重启Docker服务
sudo systemctl restart docker
```

### 2. 验证NVIDIA Docker支持

```bash
# 测试GPU访问
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

### 3. 构建Brain Isaac Sim Docker镜像

```bash
# 克隆项目（如果还没有）
git clone <your-brain-repo>
cd Brain

# 构建简化版镜像（推荐用于快速测试）
docker build -f Dockerfile.simple -t brain-simple:latest .

# 或者构建完整版镜像
docker build -f Dockerfile.isaac_sim -t brain-isaac-sim:latest .
```

## 🐳 Docker镜像选择

### 版本对比

| 镜像版本 | 描述 | 大小 | 适用场景 |
|---------|------|------|----------|
| `brain-simple:latest` | 基础版本，包含PyBullet | ~2GB | 快速测试、开发 |
| `brain-isaac-sim:latest` | 完整版本，包含Isaac Sim | ~8GB | 生产环境、完整仿真 |

### 推荐使用流程

1. **开发阶段**: 使用`brain-simple`进行快速迭代
2. **测试阶段**: 使用`brain-isaac-sim`验证完整功能
3. **部署阶段**: 根据需求选择合适版本

## 🎮 使用Docker运行Brain系统

### 基本运行命令

```bash
# 运行快速演示
docker run --rm \
    --gpus all \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/logs:/workspace/logs \
    brain-simple:latest \
    python3 run_complete_system_demo.py --mode quick

# 交互式运行
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace/brain \
    brain-simple:latest \
    /bin/bash
```

### 高级配置

```bash
# 完整环境配置
docker run -it --rm \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace/brain \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/logs:/workspace/logs \
    -p 8888:8888 \
    -p 8501:8501 \
    --name brain-container \
    brain-simple:latest
```

### 使用Docker Compose

```bash
# 启动完整服务栈
docker-compose -f docker-compose.isaac_sim.yml up -d

# 启动特定服务
docker-compose -f docker-compose.isaac_sim.yml up -d isaac-sim-brain

# 停止服务
docker-compose -f docker-compose.isaac_sim.yml down
```

## 🔧 服务访问

### Jupyter Lab
- **URL**: http://localhost:8889
- **Token**: brain2024
- **用途**: 交互式开发、调试

### Streamlit Dashboard
- **URL**: http://localhost:8501
- **用途**: 实时监控面板

### Isaac Sim Web界面
- **URL**: http://localhost:49000
- **用途**: 3D仿真可视化

## 📊 性能优化

### GPU内存管理

```bash
# 限制GPU内存使用
docker run --gpus all \
    --env NVIDIA_VISIBLE_DEVICES=0 \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    brain-simple:latest
```

### 内存优化

```bash
# 增加共享内存
docker run --shm-size=2gb brain-simple:latest

# 限制容器内存
docker run --memory=8g brain-simple:latest
```

### 存储优化

```bash
# 使用tmpfs提升IO性能
docker run --tmpfs /tmp:rw,noexec,nosuid,size=1g brain-simple:latest

# 使用SSD存储卷
docker run -v /ssd/brain-data:/workspace/data brain-simple:latest
```

## 🛠️ 常见问题解决

### 问题1: GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
```

### 问题2: 内存不足
```bash
# 增加交换空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 问题3: 显示问题（GUI应用）
```bash
# 允许X11连接
xhost +local:docker

# 设置DISPLAY变量
export DISPLAY=:0
```

### 问题4: 权限问题
```bash
# 添加用户到docker组
sudo usermod -aG docker $USER

# 重新登录或刷新组权限
newgrp docker
```

## 📈 监控和调试

### 容器监控

```bash
# 查看容器资源使用
docker stats brain-container

# 查看容器日志
docker logs -f brain-container

# 进入容器调试
docker exec -it brain-container /bin/bash
```

### GPU监控

```bash
# 实时GPU使用情况
watch -n 1 nvidia-smi

# Docker容器GPU使用
nvidia-docker stats
```

## 🔒 安全注意事项

### 容器安全
- 定期更新基础镜像
- 使用非root用户运行容器
- 限制容器权限
- 定期扫描安全漏洞

### 网络安全
- 只开放必要端口
- 使用防火墙保护
- 配置VPN访问（如需要）

## 📚 开发工作流

### 1. 开发环境设置
```bash
# 创建开发容器
docker run -it --rm \
    --gpus all \
    -v $(pwd):/workspace/brain \
    brain-simple:latest \
    /bin/bash

# 在容器内安装开发依赖
pip install -r requirements.dev.txt
```

### 2. 代码热重载
```bash
# 使用volume实现代码热重载
docker run --rm \
    --gpus all \
    -v $(pwd):/workspace/brain \
    brain-simple:latest \
    python3 -m watchdog --patterns="*.py" --command="python3 run_complete_system_demo.py"
```

### 3. 测试自动化
```bash
# 运行测试套件
docker run --rm \
    --gpus all \
    -v $(pwd):/workspace/brain \
    brain-simple:latest \
    python3 -m pytest tests/
```

## 🚀 生产部署

### 多节点部署
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  brain-master:
    image: brain-isaac-sim:latest
    environment:
      - ROLE=master
    ports:
      - "8888:8888"

  brain-worker:
    image: brain-isaac-sim:latest
    environment:
      - ROLE=worker
      - MASTER_URL=brain-master:8888
    depends_on:
      - brain-master
```

### 负载均衡
```bash
# 使用HAProxy进行负载均衡
docker run -d \
    --name haproxy \
    -p 80:80 \
    -v $(pwd)/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg \
    haproxy:latest
```

## 📖 延伸阅读

- [Docker官方文档](https://docs.docker.com/)
- [NVIDIA Container Toolkit文档](https://docs.nvidia.com/datacloud/cloud-native/container-toolkit/)
- [Isaac Sim文档](https://docs.omniverse.nvidia.com/isaac_sim/latest.html)
- [PyBullet文档](https://pybullet.org/wordpress/)

---

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个Docker集成方案！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。