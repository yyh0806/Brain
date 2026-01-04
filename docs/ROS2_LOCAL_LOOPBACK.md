# ROS2 本地回环模式配置指南

本指南说明如何将ROS2配置为本地回环模式，避免使用UDP组播。

## 概述

本地回环模式使ROS2节点只在本地（127.0.0.1）通信，不使用UDP组播。这样可以：
- 提高本地通信性能
- 避免网络干扰
- 减少网络带宽占用
- 提高系统安全性

## 快速开始

### 1. 使用启动脚本

最简单的方法是使用提供的启动脚本：

```bash
# 启动Brain系统（本地回环模式）
./scripts/start_brain_local.sh

# 或者先加载配置，再手动启动
./scripts/start_ros2_local.sh
source venv/bin/activate
python3 examples/perception_driven_demo.py
```

### 2. 手动配置

如果需要手动配置，在启动ROS2节点之前设置以下环境变量：

```bash
# 设置ROS域ID（所有节点必须相同）
export ROS_DOMAIN_ID=42

# 使用CycloneDDS（推荐用于本地通信）
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/path/to/config/dds/cyclonedds_local.xml

# 禁用组播，只允许本地通信
export ROS_LOCALHOST_ONLY=1

# 或者使用FastDDS
# export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/config/dds/local_loopback_profile.xml
```

## 配置文件

### CycloneDDS配置 (推荐)

位置: `config/dds/cyclonedds_local.xml`

主要配置项：
- `NetworkInterfaceAddress`: 127.0.0.1（只使用本地回环）
- `AllowMulticast`: false（禁用组播）
- `Discovery`: 只监听127.0.0.1
- `SharedMemory`: 启用共享内存（提高本地性能）

### FastDDS配置

位置: `config/dds/local_loopback_profile.xml`

主要配置项：
- `transport_id`: shared_memory_transport + local_udp
- `enable_multicast`: false
- `interfaceWhiteList`: 只包含127.0.0.1

## 使用场景

### 1. 启动Brain系统

```bash
./scripts/start_brain_local.sh
```

### 2. 启动多个终端

每个终端都需要加载相同的配置：

```bash
# 终端1
./scripts/start_ros2_local.sh
python3 examples/perception_driven_demo.py

# 终端2
./scripts/start_ros2_local.sh
ros2 topic list

# 终端3
./scripts/start_ros2_local.sh
ros2 run rviz2 rviz2
```

### 3. 与Isaac Sim通信

如果需要在本地回环模式下与Isaac Sim通信：

```bash
# 1. 先启动Brain（本地模式）
./scripts/start_brain_local.sh

# 2. 在另一个终端启动RViz2（加载相同配置）
source scripts/start_ros2_local.sh
./start_rviz2.sh
```

## 验证配置

运行以下命令验证配置是否生效：

```bash
# 检查RMW实现
echo $RMW_IMPLEMENTATION

# 检查域ID
echo $ROS_DOMAIN_ID

# 列出发现的节点
ros2 daemon stop
ros2 daemon start
ros2 node list

# 检查话题
ros2 topic list

# 查看网络连接（应该只看到127.0.0.1）
netstat -an | grep 7400
```

## 性能优化

本地回环模式可以通过以下方式进一步优化：

1. **使用共享内存**: CycloneDDS和FastDDS都支持共享内存传输
2. **增加缓冲区大小**: 减少数据拷贝
3. **调整QoS策略**: 使用适合本地通信的QoS设置

示例QoS配置（在`config/modules/communication/ros2.yaml`中）：

```yaml
qos:
  sensor:
    reliability: "best_effort"
    durability: "volatile"
    depth: 10

  command:
    reliability: "reliable"
    durability: "volatile"
    depth: 10
```

## 故障排除

### 问题1: 节点无法发现

**症状**: `ros2 node list` 显示为空

**解决方案**:
1. 确保所有节点使用相同的 `ROS_DOMAIN_ID`
2. 检查环境变量是否正确设置
3. 尝试重启ROS2守护进程:
   ```bash
   ros2 daemon stop
   ros2 daemon start
   ```

### 问题2: 配置文件未生效

**症状**: 节点仍在使用UDP组播

**解决方案**:
1. 检查配置文件路径是否正确
2. 确认使用正确的RMW实现
3. 查看DDS日志输出

### 问题3: 性能不佳

**症状**: 通信延迟高或带宽不足

**解决方案**:
1. 启用共享内存传输
2. 增加DDS缓冲区大小
3. 使用更快的RMW实现（CycloneDDS通常比FastDDS更快）

## 配置对比

| 配置项 | 默认模式 | 本地回环模式 |
|--------|---------|-------------|
| 网络接口 | 所有接口 | 127.0.0.1 |
| 组播 | 启用 | 禁用 |
| 发现范围 | 子网 | 本地 |
| 传输方式 | UDP | 共享内存 + 本地UDP |
| 适用场景 | 多机通信 | 单机高性能 |

## 注意事项

1. **所有节点必须使用相同的配置**: 域ID、RMW实现等必须一致
2. **不能跨机器通信**: 本地回环模式只能用于单机通信
3. **Isaac Sim需要支持**: 确保Isaac Sim能够连接到本地回环地址
4. **端口占用**: 默认端口7400，如需更改需同步修改所有节点配置

## 相关文件

- `scripts/start_ros2_local.sh` - ROS2本地回环启动脚本
- `scripts/start_brain_local.sh` - Brain本地回环启动脚本
- `config/dds/cyclonedds_local.xml` - CycloneDDS配置文件
- `config/dds/local_loopback_profile.xml` - FastDDS配置文件
- `config/modules/communication/ros2.yaml` - ROS2通信配置

## 参考资源

- [ROS2 DDS配置指南](https://docs.ros.org/en/humble/How-To-Guides/DDS-configuration.html)
- [CycloneDDS文档](https://cyclonedds.io/)
- [FastDDS文档](https://docs.eprosima.com/)
- [ROS2域隔离](https://docs.ros.org/en/humble/Tutorials/Understanding-ROS2-Nodes/Working-with-multiple-ROS2-middleware-implementations.html)




