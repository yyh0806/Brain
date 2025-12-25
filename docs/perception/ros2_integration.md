# ROS2集成文档

Brain感知模块通过ROS2与外部传感器和机器人系统进行集成，支持多种ROS2消息类型和服务。

## ROS2传感器管理器

### 基本使用

```python
from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
from brain.communication.ros2_interface import ROS2Interface

# 创建ROS2接口
ros2 = ROS2Interface()

# 创建传感器管理器
sensor_manager = ROS2SensorManager(
    ros2_interface=ros2,
    config={
        "sensors": {
            "rgb_camera": {"enabled": True},
            "depth_camera": {"enabled": True},
            "lidar": {"enabled": True},
            "imu": {"enabled": True}
        },
        "grid_resolution": 0.1,
        "map_size": 50.0
    }
)
```

### 获取融合感知数据

```python
# 异步获取融合后的感知数据
perception_data = await sensor_manager.get_fused_perception()

# 访问融合数据
pose = perception_data.pose          # 机器人位姿
rgb_image = perception_data.rgb_image   # RGB图像
depth_image = perception_data.depth_image  # 深度图像
laser_ranges = perception_data.laser_ranges  # 激光雷达距离
obstacles = perception_data.obstacles  # 障碍物列表
occupancy_grid = perception_data.occupancy_grid  # 占据栅格地图
```

## 支持的ROS2话题

### 输入话题

| 话题名称 | 消息类型 | 描述 |
|---------|---------|------|
| `/rgb/image_raw` | sensor_msgs/Image | RGB相机图像 |
| `/depth/image_raw` | sensor_msgs/Image | 深度相机图像 |
| `/camera_info` | sensor_msgs/CameraInfo | 相机内参 |
| `/scan` | sensor_msgs/LaserScan | 激光雷达扫描 |
| `/imu` | sensor_msgs/Imu | IMU数据 |
| `/odom` | nav_msgs/Odometry | 里程计数据 |
| `/pointcloud` | sensor_msgs/PointCloud2 | 点云数据 |
| `/map` | nav_msgs/OccupancyGrid | 外部占据地图 |

### 输出话题 (可选)

| 话题名称 | 消息类型 | 描述 |
|---------|---------|------|
| `/perception/objects` | custom_msgs/DetectedObjects | 检测到的物体 |
| `/perception/occupancy_grid` | nav_msgs/OccupancyGrid | 生成的占据地图 |
| `/perception/pose` | geometry_msgs/PoseStamped | 融合后的位姿 |

## 传感器状态管理

### 检查传感器健康状态

```python
# 获取所有传感器健康状态
sensor_health = sensor_manager.get_sensor_health()

# 检查特定传感器
is_camera_healthy = sensor_health.get("rgb_camera", False)
is_lidar_healthy = sensor_health.get("lidar", False)

# 等待传感器就绪
ready = await sensor_manager.wait_for_sensors(timeout=10.0)
```

### 传感器状态类型

```python
from brain.perception.sensors.ros2_sensor_manager import SensorStatus, SensorType

# 传感器状态
status = SensorStatus(
    sensor_type=SensorType.RGB_CAMERA,
    enabled=True,
    connected=False,
    last_update=None,
    update_rate=0.0,
    error_count=0
)

# 检查传感器是否健康
is_healthy = status.is_healthy(timeout=5.0)
```

## 数据获取方法

### 直接获取原始数据

```python
# 获取最新RGB图像
rgb_image = sensor_manager.get_rgb_image()

# 获取最新深度图像
depth_image = sensor_manager.get_depth_image()

# 获取最新激光雷达数据
laser_scan = sensor_manager.get_laser_scan()

# 获取当前位姿
current_pose = sensor_manager.get_current_pose()
current_pose_2d = sensor_manager.get_current_pose_2d()
```

### 障碍物和导航信息

```python
# 获取最近障碍物
nearest_obstacle = sensor_manager.get_nearest_obstacle()
if nearest_obstacle:
    print(f"最近障碍物距离: {nearest_obstacle['distance']}米")

# 获取特定方向障碍物
front_obstacles = sensor_manager.get_obstacles_in_direction("front")
left_obstacles = sensor_manager.get_obstacles_in_direction("left")

# 检查路径是否畅通
is_front_clear = perception_data.is_path_clear("front", threshold=1.0)
is_left_clear = perception_data.is_path_clear("left", threshold=1.0)
```

## 配置参数

### 传感器配置

```python
config = {
    "sensors": {
        "rgb_camera": {
            "enabled": True,
            "topic": "/rgb/image_raw",
            "queue_size": 5
        },
        "depth_camera": {
            "enabled": True,
            "topic": "/depth/image_raw",
            "queue_size": 5
        },
        "lidar": {
            "enabled": True,
            "topic": "/scan",
            "queue_size": 10
        },
        "imu": {
            "enabled": True,
            "topic": "/imu",
            "queue_size": 10
        }
    },
    "pose_filter_alpha": 0.8,      # 位姿滤波系数
    "obstacle_threshold": 0.5,       # 障碍物阈值
    "min_obstacle_size": 0.1,       # 最小障碍物尺寸
    "grid_resolution": 0.1,         # 栅格地图分辨率
    "map_size": 50.0,               # 地图大小
    "max_history": 100               # 历史数据数量
}
```

### 占据栅格参数

```python
occupancy_config = {
    "resolution": 0.1,          # 米/栅格
    "map_size": 50.0,           # 地图大小(米)
    "camera_fov": 1.57,         # 相机视场角(弧度)
    "camera_range": 10.0,        # 相机感知范围(米)
    "lidar_range": 30.0,        # 激光雷达范围(米)
    "occupied_prob": 0.7,        # 占据概率阈值
    "free_prob": 0.3,           # 自由概率阈值
    "min_depth": 0.1,           # 最小深度(米)
    "max_depth": 10.0           # 最大深度(米)
}
```

## 与Isaac Sim集成

### Isaac Sim ROS2桥接

Isaac Sim提供内置ROS2桥接功能，可将仿真环境中的传感器数据发布为ROS2话题：

1. 在Isaac Sim中启用ROS2桥接：
   ```
   Window > Robotics > ROS2 Bridge
   ```

2. 配置要发布的话题：
   ```python
   # 在Isaac Sim Python脚本中
   from omni.isaac.ros2_bridge import ROS2Bridge
   bridge = ROS2Bridge()
   bridge.add_publisher("/rgb/image_raw", "sensor_msgs/Image")
   bridge.add_publisher("/depth/image_raw", "sensor_msgs/Image")
   bridge.add_publisher("/scan", "sensor_msgs/LaserScan")
   bridge.add_publisher("/imu", "sensor_msgs/Imu")
   bridge.add_publisher("/odom", "nav_msgs/Odometry")
   ```

3. 在Brain项目中接收数据：
   ```python
   # 创建ROS2传感器管理器
   sensor_manager = ROS2SensorManager(ros2_interface)
   
   # 获取融合后的感知数据
   perception_data = await sensor_manager.get_fused_perception()
   ```

### 数据格式转换

Isaac Sim和Brain之间的数据格式需要适当转换：

```python
# ROS2消息转Brain数据类型
def convert_ros_image(ros_image):
    import cv2
    import numpy as np
    
    # 转换ROS2 Image为numpy数组
    bridge = cv_bridge.CvBridge()
    cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
    return cv_image

def convert_ros_scan(ros_scan):
    # 转换ROS2 LaserScan
    ranges = list(ros_scan.ranges)
    angles = []
    
    # 计算角度
    angle_min = ros_scan.angle_min
    angle_increment = ros_scan.angle_increment
    for i in range(len(ranges)):
        angles.append(angle_min + i * angle_increment)
    
    return ranges, angles
```

## 性能优化

### 异步处理

ROS2传感器管理器使用异步处理避免阻塞：

```python
import asyncio

async def perception_loop():
    while True:
        # 异步获取感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 处理数据...
        
        # 控制循环频率
        await asyncio.sleep(0.1)  # 10Hz

# 运行异步循环
asyncio.run(perception_loop())
```

### 数据缓存

传感器管理器提供历史数据缓存：

```python
# 获取历史数据
history = sensor_manager.get_data_history(count=10)

# 分析传感器稳定性
for data in history:
    # 分析时间戳差异
    timestamp = data.timestamp
    # ...
```

## 故障排除

### 常见问题

1. **ROS2连接问题**
   ```python
   # 检查ROS2节点状态
   ros2 node list
   ros2 topic list
   ros2 topic echo /rgb/image_raw
   ```

2. **数据不同步**
   ```python
   # 检查传感器时间戳
   perception_data = await sensor_manager.get_fused_perception()
   print(f"RGB时间戳: {rgb_timestamp}")
   print(f"深度时间戳: {depth_timestamp}")
   print(f"激光时间戳: {laser_timestamp}")
   ```

3. **性能问题**
   ```python
   # 获取传感器健康状态
   health = sensor_manager.get_sensor_health()
   for sensor, is_healthy in health.items():
       if not is_healthy:
           print(f"传感器 {sensor} 状态异常")
   ```

## 相关文档

- [传感器接口](sensor_interfaces.md)
- [多传感器融合](sensor_fusion.md)
- [Isaac Sim集成](isaac_sim_integration.md)






