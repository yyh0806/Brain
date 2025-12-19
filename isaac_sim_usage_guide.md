# 🚀 Isaac Sim 5.1.0 使用指南

## ✅ 当前状态
- **镜像**: nvcr.io/nvidia/isaac-sim:5.1.0 (15.1GB) ✅
- **容器**: isaac-sim-gui 运行中 ✅
- **GPU支持**: 已启用 ✅
- **端口映射**: 8222, 49000-49002 ✅

## 🌐 访问Isaac Sim

### 方法1: Web浏览器界面 (推荐)
```bash
# 等待容器完全启动后访问:
http://localhost:8222
```

### 方法2: Livestream Web界面
```bash
# 如果8222端口不可用，尝试:
http://localhost:49000
http://localhost:49001
```

### 方法3: 直接连接容器
```bash
# 查看容器状态
docker ps | grep isaac-sim

# 查看启动日志
docker logs isaac-sim-gui

# 进入容器
docker exec -it isaac-sim-gui /bin/bash
```

## 🛠️ 常用命令

### 检查运行状态
```bash
# 查看容器状态
docker ps | grep isaac-sim

# 查看资源使用
docker stats isaac-sim-gui

# 查看日志
docker logs -f isaac-sim-gui
```

### 重启服务
```bash
# 停止
docker stop isaac-sim-gui

# 启动
docker start isaac-sim-gui

# 重启
docker restart isaac-sim-gui
```

### 进入开发模式
```bash
# 启动Python交互模式
docker exec -it isaac-sim-gui python3

# 运行示例脚本
docker exec isaac-sim-gui python3 /isaac-sim/apps/isaacsim/standalone_examples/hello_world.py
```

## 📁 目录映射

- **本地工作区**: `./isaac-sim-workspace` ↔ `/workspace/isaac-sim`
- **缓存目录**: `~/isaac-sim-cache/kit/cache` ↔ `/root/.cache/kit`
- **数据目录**: `~/isaac-sim-cache/data` ↔ `/root/.local/share/ov/data`

## 🎯 快速开始

1. **等待启动完成** (约2-3分钟)
2. **访问Web界面**: http://localhost:8222
3. **创建新场景** 或 **打开示例**
4. **开始仿真开发**

## ⚠️ 故障排除

### 如果Web界面无法访问:
```bash
# 检查端口是否监听
netstat -tlnp | grep 8222

# 检查容器日志
docker logs isaac-sim-gui | tail -20

# 检查GPU状态
nvidia-smi
```

### 如果性能不佳:
```bash
# 增加共享内存
docker stop isaac-sim-gui
docker run -d --name isaac-sim-gui --shm-size=16gb [其他参数...]

# 限制GPU内存使用
docker run -d --name isaac-sim-gui --gpus '"device=0, memory=8GB"' [其他参数...]
```

## 📚 更多资源

- [Isaac Sim 5.1 官方文档](https://docs.omniverse.nvidia.com/isaac_sim/latest/index.html)
- [Python API 参考](https://docs.omniverse.nvidia.com/isaac_sim/latest/APIReference.html)
- [示例代码](/isaac-sim/apps/isaacsim/standalone_examples/)

---

**注意**: 首次启动可能需要2-3分钟初始化时间。