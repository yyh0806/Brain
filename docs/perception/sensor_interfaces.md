# 传感器接口文档

Brain感知模块提供统一的传感器接口，支持多种传感器类型的数据采集和处理。

## 传感器基类

### BaseSensor

所有传感器都继承自`BaseSensor`抽象基类，提供以下功能：

- 数据采集和预处理
- 数据质量评估
- 线程安全的数据缓冲
- 传感器状态监控
- 错误处理和恢复

```python
from brain.perception.sensors.sensor_interface import BaseSensor, SensorConfig
from brain.perception.sensor_input_types import SensorType

# 创建传感器配置
config = SensorConfig(
    sensor_id="camera_1",
    sensor_type=SensorType.CAMERA,
    update_rate=30.0,  # 30Hz
    buffer_size=100
)

# 传感器将由具体子类实现
```

## 传感器类型

### 图像传感器 (ImageSensor)

支持RGB相机、深度相机和热成像相机：

```python
from brain.perception.sensors.sensor_interface import ImageSensor
from brain.perception.sensor_input_types import CameraIntrinsics, ImageData

# 相机内参
intrinsics = CameraIntrinsics(
    fx=525.0, fy=525.0,  # 焦距
    cx=320.0, cy=240.0,  # 主点
    width=640, height=480   # 图像尺寸
)

# 创建图像传感器
camera = ImageSensor(config, camera_intrinsics=intrinsics)

# 获取最新数据
data_packets = camera.get_latest_data()
for packet in data_packets:
    image_data: ImageData = packet.data
    image = image_data.image  # numpy数组 (H, W, C)
    depth = image_data.depth  # 可选深度图 (H, W)
```

### 点云传感器 (PointCloudSensor)

支持LiDAR和雷达传感器：

```python
from brain.perception.sensors.sensor_interface import PointCloudSensor
from brain.perception.sensor_input_types import PointCloudData

# 创建点云传感器
lidar = PointCloudSensor(config)

# 获取最新数据
data_packets = lidar.get_latest_data()
for packet in data_packets:
    pointcloud: PointCloudData = packet.data
    points = pointcloud.points      # (N, 3) numpy数组
    intensity = pointcloud.intensity  # (N,) numpy数组 (可选)
    rgb = pointcloud.rgb             # (N, 3) numpy数组 (可选)
```

### IMU传感器 (IMUSensor)

支持惯性测量单元：

```python
from brain.perception.sensors.sensor_interface import IMUSensor
from brain.perception.sensor_input_types import IMUData

# 创建IMU传感器
imu = IMUSensor(config)

# 获取最新数据
data_packets = imu.get_latest_data()
for packet in data_packets:
    imu_data: IMUData = packet.data
    acceleration = imu_data.linear_acceleration  # (3,) 线性加速度
    angular_velocity = imu_data.angular_velocity  # (3,) 角速度
    orientation = imu_data.orientation           # (4,) 四元数 (可选)
```

### GPS传感器 (GPSSensor)

支持全球定位系统：

```python
from brain.perception.sensors.sensor_interface import GPSSensor
from brain.perception.sensor_input_types import GPSData

# 创建GPS传感器
gps = GPSSensor(config)

# 获取最新数据
data_packets = gps.get_latest_data()
for packet in data_packets:
    gps_data: GPSData = packet.data
    latitude = gps_data.latitude      # 纬度
    longitude = gps_data.longitude    # 经度
    altitude = gps_data.altitude      # 高度
    velocity = gps_data.velocity     # (3,) 速度向量 (可选)
```

## 传感器工厂

使用传感器工厂函数创建传感器实例：

```python
from brain.perception.sensors.sensor_interface import create_sensor, SensorConfig
from brain.perception.sensor_input_types import SensorType

# 创建配置
config = SensorConfig(
    sensor_id="lidar_front",
    sensor_type=SensorType.LIDAR,
    update_rate=10.0
)

# 创建传感器
lidar = create_sensor(config)

# 启动传感器
lidar.start()
```

## 数据类型

### SensorDataPacket

所有传感器数据都被封装在`SensorDataPacket`中：

```python
from brain.perception.sensor_input_types import SensorDataPacket

# 获取数据包
packet = sensor.get_latest_data()[0]

# 访问属性
sensor_id = packet.sensor_id          # 传感器ID
sensor_type = packet.sensor_type      # 传感器类型
timestamp = packet.timestamp          # 时间戳
data = packet.data                  # 实际数据 (PointCloudData, ImageData等)
quality_score = packet.quality_score  # 数据质量评分 (0-1)
```

### 数据质量评估

每个传感器都提供数据质量评估功能：

```python
from brain.perception.sensor_input_types import validate_sensor_data_quality

# 评估数据质量
assessment = validate_sensor_data_quality(sensor_data)

# 访问评估结果
score = assessment["score"]           # 总体质量分数 (0-1)
quality = assessment["quality"]        # 质量等级
issues = assessment["issues"]         # 发现的问题列表
```

## 传感器回调

可以为传感器添加回调函数，在新数据到达时自动处理：

```python
def on_new_data(packet: SensorDataPacket):
    print(f"收到新数据: {packet.sensor_id} @ {packet.timestamp}")
    # 处理数据...

# 添加回调
sensor.add_callback(on_new_data)

# 移除回调
sensor.remove_callback(on_new_data)
```

## 传感器统计

获取传感器运行统计信息：

```python
# 获取统计信息
stats = sensor.get_statistics()

# 访问统计数据
is_running = stats["is_running"]          # 是否运行中
packets_received = stats["packets_received"]  # 接收的数据包数
packets_dropped = stats["packets_dropped"]    # 丢弃的数据包数
loss_rate = stats["loss_rate_percent"]          # 丢包率
average_update_rate = stats["average_update_rate"]  # 平均更新率
```

## 传感器配置

### SensorConfig

```python
from brain.perception.sensors.sensor_interface import SensorConfig

config = SensorConfig(
    sensor_id="sensor_1",               # 传感器唯一标识
    sensor_type=SensorType.CAMERA,       # 传感器类型
    frame_id="camera_link",              # 坐标系
    update_rate=30.0,                  # 更新频率 (Hz)
    auto_start=True,                     # 自动启动
    buffer_size=100,                    # 缓冲区大小
    enable_compression=False,             # 启用压缩
    quality_threshold=0.5,              # 质量阈值
    max_processing_time=0.1,            # 最大处理时间 (秒)
    calibration_params={},                # 校准参数
    enable_noise_filtering=True,         # 启用噪声过滤
    enable_outlier_removal=True,        # 启用异常值移除
    min_data_quality=0.3,               # 最小数据质量
    ros2_topic="/sensors/camera",        # ROS2话题 (可选)
    ros2_qos_profile="best_effort"       # ROS2 QoS配置
)
```

## 最佳实践

1. **传感器启动**: 使用`start()`方法启动传感器，检查返回值确认启动成功
2. **错误处理**: 监控传感器状态，处理异常情况
3. **资源管理**: 使用`stop()`方法释放传感器资源
4. **数据质量**: 根据应用需求设置合适的质量阈值
5. **性能优化**: 根据处理能力调整缓冲区大小和更新频率

## 相关文档

- [多传感器融合](sensor_fusion.md)
- [ROS2集成](ros2_integration.md)
- [测试指南](testing.md)









