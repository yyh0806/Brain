# 多传感器融合文档

Brain感知模块提供多传感器融合功能，将不同传感器的数据融合为一致的环境表示。

## 融合算法概览

### 1. 位姿融合 (EKF)

使用扩展卡尔曼滤波器融合里程计和IMU数据：

```python
from brain.perception.sensors.sensor_fusion import EKFPoseFusion

# 创建EKF融合器
ekf = EKFPoseFusion()

# 更新里程计数据
odom_data = {
    "position": {"x": 1.0, "y": 0.5, "z": 0.0},
    "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
    "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
    "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
}
ekf.update_odom(odom_data)

# 更新IMU数据
imu_data = {
    "orientation": {"x": 0, "y": 0, "z": 0.12, "w": 0.993},
    "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.11},
    "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
}
ekf.update_imu(imu_data)

# 获取融合后的位姿
pose = ekf.get_pose()
print(f"位置: x={pose.x}, y={pose.y}, z={pose.z}")
print(f"姿态: roll={pose.roll}, pitch={pose.pitch}, yaw={pose.yaw}")
```

### 2. RGBD融合

融合RGB图像和深度图生成3D点云：

```python
from brain.perception.sensors.sensor_fusion import DepthRGBFusion

# 创建RGBD融合器
fusion = DepthRGBFusion(
    fx=525.0,  # 相机焦距x
    fy=525.0,  # 相机焦距y
    cx=319.5,  # 光心x
    cy=239.5   # 光心y
)

# 融合RGB和深度图
rgbd, points = fusion.fuse(rgb_image, depth_image)

# rgbd: (H, W, 4) RGBD图像
# points: (N, 6) 3D点云 (X, Y, Z, R, G, B)
```

### 3. 障碍物检测融合

融合来自多个传感器的障碍物检测：

```python
from brain.perception.sensors.sensor_fusion import ObstacleDetector

# 创建障碍物检测器
detector = ObstacleDetector({
    "depth_threshold": 3.0,    # 深度阈值 (米)
    "laser_threshold": 5.0,    # 激光阈值 (米)
    "fusion_distance": 0.5      # 融合距离 (米)
})

# 从深度图检测障碍物
depth_obstacles = detector.detect_from_depth(depth_image)

# 从激光雷达检测障碍物
laser_obstacles = detector.detect_from_laser(ranges, angles)

# 融合障碍物检测
fused_obstacles = detector.fuse_obstacles(depth_obstacles, laser_obstacles)
```

## 数据结构

### FusedPose

融合后的位姿数据：

```python
from brain.perception.sensors.sensor_fusion import FusedPose

pose = FusedPose(
    x=1.0, y=2.0, z=0.0,         # 位置
    roll=0.1, pitch=0.05, yaw=1.57, # 姿态
    covariance=np.eye(6) * 0.1      # 协方差矩阵
)

# 转换为numpy数组
pose_array = pose.to_array()  # [x, y, z, roll, pitch, yaw]

# 从数组创建
new_pose = FusedPose.from_array(pose_array)
```

## 融合配置

### EKF参数

```python
# 过程噪声协方差 (Q)
ekf.Q = np.eye(12) * 0.01
ekf.Q[6:9, 6:9] *= 0.1   # 速度噪声
ekf.Q[9:12, 9:12] *= 0.1  # 角速度噪声

# 测量噪声协方差 (R)
ekf.R_odom = np.eye(6) * 0.1  # 里程计测量噪声
ekf.R_imu = np.eye(6) * 0.05   # IMU测量噪声
```

### RGBD融合参数

```python
# 相机参数
fusion.fx = 525.0  # 焦距x
fusion.fy = 525.0  # 焦距y
fusion.cx = 320.0  # 光心x
fusion.cy = 240.0  # 光心y
```

## 融合策略

### 1. 时间对齐

确保所有传感器数据在时间上对齐：

```python
from brain.perception.sensors.sensor_manager import MultiSensorManager

# 创建多传感器管理器
manager = MultiSensorManager()

# 添加传感器
manager.add_sensor(camera, camera_config)
manager.add_sensor(lidar, lidar_config)

# 创建传感器组
manager.create_sensor_group(
    group_id="main_sensors",
    sensor_ids=["camera", "lidar"],
    sync_method=SyncMethod.TIMESTAMP_ALIGNMENT,
    sync_tolerance=0.01  # 10ms容差
)

# 启动传感器
manager.start_sensors()

# 添加同步数据回调
def on_sync_data(sync_packet):
    # 处理同步后的数据
    print(f"同步数据质量: {sync_packet.sync_quality}")
    camera_data = sync_packet.get_sensor_data("camera")
    lidar_data = sync_packet.get_sensor_data("lidar")

manager.add_sync_callback(on_sync_data)
```

### 2. 空间对齐

将不同传感器数据转换到同一坐标系：

```python
from brain.perception.utils.coordinates import transform_local_to_world

# 转换点到世界坐标系
world_x, world_y = transform_local_to_world(
    local_x=1.0,      # 本地x坐标
    local_y=2.0,      # 本地y坐标
    robot_x=5.0,      # 机器人x位置
    robot_y=3.0,      # 机器人y位置
    robot_yaw=1.57     # 机器人朝向
)
```

### 3. 质量加权

根据数据质量对传感器数据进行加权融合：

```python
from brain.perception.sensors.sensor_fusion import DepthRGBFusion

# 基于质量的融合权重
quality_weighted = True

if quality_weighted:
    # 高质量数据权重更高
    weight = data_quality_score
else:
    # 等权重
    weight = 1.0
```

## 性能优化

1. **异步处理**: 使用异步处理避免阻塞主线程
2. **数据降采样**: 对高频传感器数据进行适当降采样
3. **区域限制**: 只处理感兴趣区域的数据
4. **并行计算**: 利用多核处理器并行处理不同传感器数据

## 故障处理

1. **传感器失效**: 检测传感器故障并切换到备份方案
2. **数据异常**: 检测并过滤异常数据
3. **同步失败**: 处理多传感器同步失败情况

## 相关文档

- [传感器接口](sensor_interfaces.md)
- [ROS2集成](ros2_integration.md)
- [测试指南](testing.md)