# 障碍物检测说明

## 什么是障碍物（Obstacles）？

障碍物是感知层从传感器数据中检测到的**可能阻挡机器人前进的物体**。障碍物检测是机器人导航和避障的基础。

## 障碍物检测的数据来源

### 1. 激光雷达（Lidar）
- **数据格式**：`laser_ranges` 和 `laser_angles`
- **检测方法**：
  - 将激光雷达扫描点转换为笛卡尔坐标（x, y）
  - 使用聚类算法将相邻的点分组
  - 每个聚类代表一个潜在的障碍物
  - 计算障碍物的中心位置、大小和距离

### 2. 点云（Point Cloud）
- **数据格式**：3D点云数组 `(N, 3)` 或 `(N, 6)`
- **检测方法**：
  - 将3D点云投影到2D平面（去除Z轴）
  - 转换为激光雷达格式（ranges, angles）
  - 使用与激光雷达相同的聚类算法检测障碍物

### 3. 深度图（Depth Image）
- **数据格式**：深度图像数组
- **检测方法**：
  - 检测深度值小于阈值的区域
  - 将深度图分成左、中、右三个区域
  - 每个区域的最小深度值代表障碍物距离

## 障碍物数据结构

每个障碍物包含以下信息：

```python
{
    "id": "obstacle_0",           # 障碍物ID
    "source": "laser",             # 数据来源：laser/depth/pointcloud
    "local_position": {           # 局部坐标（相对于机器人）
        "x": 1.5,                  # X坐标（米）
        "y": 0.3,                  # Y坐标（米）
    },
    "world_position": {            # 世界坐标（如果有位姿信息）
        "x": 2.1,
        "y": 1.8,
    },
    "distance": 1.53,              # 距离（米）
    "size": 0.5,                   # 障碍物大小（米）
    "direction": "front",          # 方向：front/left/right/back
    "confidence": 0.9,             # 置信度（0-1）
    "region": "center"             # 区域：left/center/right
}
```

## 障碍物检测参数

### 聚类参数
- `distance_threshold`: 0.3米 - 相邻点距离阈值，超过此距离的点不属于同一聚类
- `min_points`: 3 - 最小点数，少于3个点的聚类会被忽略
- `min_obstacle_size`: 0.1米 - 最小障碍物大小，小于此大小的聚类会被忽略

### 距离过滤
- `range_min`: 0.1米 - 最小有效距离
- `range_max`: 30.0米 - 最大有效距离

## 障碍物检测流程

1. **数据获取**：从ROS2话题接收激光雷达或点云数据
2. **数据转换**：将点云转换为激光雷达格式（如果需要）
3. **聚类分析**：使用简单聚类算法将点分组
4. **障碍物生成**：为每个有效聚类创建障碍物对象
5. **坐标转换**：将局部坐标转换为世界坐标（如果有位姿信息）
6. **结果输出**：将障碍物列表添加到 `PerceptionData.obstacles`

## 可视化中的障碍物显示

在可视化界面中，障碍物显示为：
- **红色圆圈**：表示障碍物的位置和大小
- **标签**：显示障碍物ID和距离
- **机器人位置**：绿色圆点表示机器人当前位置

## 为什么没有检测到障碍物？

可能的原因：
1. **没有激光雷达数据**：`laser_ranges` 为空或 `None`
2. **点云未转换**：点云数据存在但未转换为激光雷达格式
3. **距离太远**：所有点都超过30米，被过滤掉
4. **聚类太小**：障碍物太小，不满足最小点数或最小大小要求
5. **场景空旷**：环境中确实没有障碍物

## 调试建议

1. **检查激光雷达数据**：
   ```python
   if perception_data.laser_ranges:
       print(f"Lidar points: {len(perception_data.laser_ranges)}")
       print(f"Range: [{min(perception_data.laser_ranges)}, {max(perception_data.laser_ranges)}]")
   ```

2. **检查点云数据**：
   ```python
   if perception_data.pointcloud is not None:
       print(f"Point cloud shape: {perception_data.pointcloud.shape}")
   ```

3. **检查障碍物检测逻辑**：
   - 查看 `ros2_sensor_manager.py` 中的 `_detect_obstacles_from_laser` 方法
   - 检查聚类参数是否合理
   - 检查距离过滤是否太严格



