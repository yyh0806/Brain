# 真实感知层处理指南

## 概述

本指南说明如何从rosbag中读取真实传感器数据，通过感知层处理，构建占据栅格地图并发布到RViz显示。

---

## 🎯 快速开始（3步）

### 步骤1: 播放rosbag（如果还没有）

```bash
cd /media/yangyuhui/CODES1/Brain

# 播放rosbag数据
ros2 bag play data/rosbags/<你的rosbag文件>

# 或者如果rosbag在播放，继续到下一步
```

### 步骤2: 运行真实感知处理器

```bash
# 在新终端中运行
python3 scripts/run_perception_with_rosbag.py
```

**预期输出**：
```
✓ ROS_DOMAIN_ID = 42
真实感知处理器已启动
地图大小: 50.0m x 50.0m
分辨率: 0.1m/栅格
栅格数: 500 x 500
订阅话题: /chassis/odom, /front_3d_lidar/lidar_points
发布话题: /brain/map
```

### 步骤3: 启动RViz

```bash
# 在另一个终端中启动RViz
bash start_rviz2.sh
```

---

## 🔍 工作原理

### 数据流

```
rosbag → [真实传感器数据] → 感知处理器 → [占据地图] → RViz
  ↓              ↓            ↓               ↓
点云数据   → 点云处理 → /brain/map → 显示
里程计数据 → 位姿计算     →
```

### 处理流程

1. **订阅传感器数据**
   - `/chassis/odom` → 获取机器人位姿
   - `/front_3d_lidar/lidar_points` → 获取3D点云

2. **处理点云数据**
   - 解析PointCloud2消息
   - 提取XYZ坐标
   - 过滤地面点（z > 0.2m 且 z < 2.0m）

3. **更新占据地图**
   - **障碍物检测**：在机器人周围的高点标记为占据（黑色）
   - **自由空间填充**：使用射线填充算法标记自由区域（白色）
   - **贝叶斯更新**：根据观测次数更新置信度

4. **发布地图**
   - 话题：`/brain/map`
   - 分辨率：0.1m/栅格
   - 尺寸：50m x 50m（500x500栅格）
   - frame_id：`odom`

---

## 📊 地图更新机制

### 障碍物检测

**原理**：将3D点云投影到2D平面，高z值的点标记为障碍物。

**参数**：
- 点云范围：50m x 50m（由rosbag决定）
- 地图分辨率：0.1m/栅格
- 地图尺寸：500x500栅格

**实现**：
```python
for point in points:
    x, y, z = point
    
    # 只考虑地面附近的点（0.2m - 2.0m高度）
    if 0.2 < z < 2.0:
        # 转换为地图栅格坐标
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)
        
        # 标记为占据
        if 0 <= grid_x < 500 and 0 <= grid_y < 500:
            occupancy_map[grid_y, grid_x] = 100  # 100 = 占据
```

### 自由空间填充（射线填充算法）

**原理**：从机器人位置向所有方向发射射线，在遇到障碍物前标记为自由。

**实现**：
```python
for direction in range(360):  # 360度方向
    for distance in range(50):  # 50米半径
        nx = robot_grid_x + dx
        ny = robot_grid_y + dy
        
        # 如果到该点没有障碍物，标记为自由
        if not has_obstacle_to(robot_grid_x, robot_grid_y, nx, ny):
            occupancy_map[ny, nx] = 0  # 0 = 自由
```

**特点**：
- 自由空间从机器人位置向外扩展
- 遇到障碍物立即停止
- 已探索区域保持已知状态

---

## 🎨 地图显示颜色

### Occupancy Grid（RViz中）

| 颜色 | 值 | 含义 |
|--------|------|------|
| **黑色** | 100 | 障碍物（占据） |
| **白色** | 0 | 自由空间（可通行） |
| **灰色** | -1 | 未知区域（未探索） |

---

## 🚀 完整启动流程

### 方案1: 标准流程（推荐）⭐⭐⭐

```bash
# ===== 终端1: 播放rosbag =====
cd /media/yangyuhui/CODES1/Brain
ros2 bag play data/rosbags/<rosbag文件>.db3 --rate 1.0

# ===== 终端2: 运行感知处理器 =====
python3 scripts/run_perception_with_rosbag.py

# ===== 终端3: 启动RViz =====
bash start_rviz2.sh
```

### 方案2: 快速测试

如果只是想快速验证：

```bash
# 只运行感知处理器（使用当前播放的rosbag）
python3 scripts/run_perception_with_rosbag.py

# 在RViz中应该立即看到地图更新
```

---

## 🔍 验证要点

### 1. 障碍物稳定性 ⭐

**观察**：找到地图中的障碍物（黑色区域），看它是否稳定。

**正确行为** ✓：
```
t=10s:  黑色区域A
t=20s:  黑色区域A（保持）✓
t=30s:  黑色区域A（保持）✓
```

**错误行为** ✗：
```
t=10s:  黑色
t=20s:  白色（障碍物丢失！）
```

### 2. 自由空间扩展 ⭐

**观察**：机器人周围白色区域是否逐渐向外扩展。

**正确行为** ✓：
```
t=5s:   机器人周围3米：灰色（未知）
t=10s:  内圈3米：白色（已探索）
t=20s:  5米内：白色（扩展）
t=30s:  10米内：白色（继续扩展）✓
```

### 3. 已知区域保持 ⭐

**观察**：已标记为黑/白的区域是否保持不变。

**正确行为** ✓：
```
t=10s: 区域A = 白色
t=20s: 区域A = 白色（保持）✓
t=30s: 区域A = 白色（保持，不退回到灰色）✓
```

### 4. 贝叶斯更新机制 ⭐

**观察**：一次自由观测不应立即改变占据状态。

**正确行为** ✓：
```
t=10s: 栅格[100,100] = 100（占据）
t=15s: 栅格[100,100] = 100（仍占据，即使一次扫描显示自由）
t=20s: 栅格[100,100] = 90（多次自由观测后才降低）
```

---

## 🛠️ 故障排查

### 问题1: 感知处理器启动失败

**可能原因**：
- ROS2未初始化
- 缺少依赖包

**检查方法**：
```bash
# 检查ROS2
python3 -c "import rclpy; print('OK')"

# 检查消息类型
python3 -c "from sensor_msgs.msg import PointCloud2; print('OK')"
python3 -c "from nav_msgs.msg import OccupancyGrid; print('OK')"
```

**解决**：
```bash
# 安装缺失包
pip3 install rclpy sensor-msgs nav-msgs

# 或
sudo apt install python3-rclpy ros-humble-sensor-msgs ros-humble-nav-msgs
```

### 问题2: 地图不更新

**可能原因**：
- 没有点云数据
- 点云格式不正确

**检查方法**：
```bash
# 查看点云话题
ros2 topic list | grep lidar

# 查看点云频率
ros2 topic hz /front_3d_lidar/lidar_points
```

**解决**：
- 确保rosbag在播放
- 检查rosbag文件是否包含点云数据

### 问题3: RViz显示 "No map received"

**检查方法**：
```bash
# 查看/brain/map话题
ros2 topic info /brain/map

# 应该看到：
# Publisher count: 1
# Subscription count: 1
```

**解决**：
- 确认感知处理器在运行
- 在RViz中勾选Map的Enabled
- 确认Map的Topic为 /brain/map
- 确认Fixed Frame为 odom

---

## 📊 脚本参数说明

### 地图参数

可以在脚本中修改：

```python
self.map_resolution = 0.1   # 分辨率：0.1米/栅格
self.map_size = 50.0       # 地图尺寸：50m x 50m
self.grid_size = 500         # 栅格数：500x500（自动计算）
self.map_origin_x = -25.0    # 地图原点X：-25米
self.map_origin_y = -25.0    # 地图原点Y：-25米
```

### 障碍物检测参数

```python
# 在update_map_from_pointcloud函数中修改
if 0.2 < z < 2.0:  # 地面点范围
    # 高于2米的点不视为障碍物（如天花板）
```

---

## 🎯 验证清单

运行后，对照以下清单逐一验证：

### 数据订阅
- [ ] 感知处理器启动成功
- [ ] 订阅 `/chassis/odom`
- [ ] 订阅 `/front_3d_lidar/lidar_points`
- [ ] 里程计数据正常接收
- [ ] 点云数据正常接收

### 地图更新
- [ ] 地图每秒更新1次（1Hz）
- [ ] 占据栅格数递增
- [ ] 自由栅格数递增
- [ ] 未知栅格数递减

### RViz显示
- [ ] Occupancy Grid显示地图
- [ ] 不显示 "No map received"
- [ ] 地图颜色正确（黑/白/灰）
- [ ] 地图更新流畅（不卡顿）

### 持久化验证
- [ ] 障碍物（黑色）稳定不变
- [ ] 自由空间（白色）逐渐扩展
- [ ] 已知区域不退回到未知（灰色）
- [ ] 贝叶斯更新生效（一次观测不立即改变）

---

## 💡 高级功能

### 调整地图更新频率

修改 `RealPerceptionProcessor` 类中的：

```python
self.map_timer = self.create_timer(1.0, self.publish_map)  # 改为0.5秒更新
```

### 调整地图尺寸

```python
self.map_size = 100.0  # 改为100m x 100m
# 会自动计算新的grid_size
```

### 添加更多传感器

可以扩展脚本订阅更多传感器：

```python
# 订阅相机
self.camera_sub = self.create_subscription(
    Image,
    '/front_stereo_camera/left/image_raw',
    10,
    self.camera_callback
)

# 订阅IMU
self.imu_sub = self.create_subscription(
    Imu,
    '/chassis/imu',
    10,
    self.imu_callback
)
```

---

## 📚 相关脚本和配置

| 文件 | 用途 |
|------|------|
| `scripts/run_perception_with_rosbag.py` | 真实感知处理器 |
| `scripts/test_rviz_map.py` | 模拟地图测试（已不用） |
| `config/rviz2/nova_carter_persistent_map.rviz` | RViz配置文件 |
| `config/nova_carter_ros2.yaml` | Brain系统配置 |

---

## ⚠️ 重要提示

1. **ROS域ID统一**
   - 所有进程必须使用 `ROS_DOMAIN_ID=42`
   - 脚本已自动设置

2. **rosbag必须包含所需数据**
   - `/chassis/odom`（里程计）
   - `/front_3d_lidar/lidar_points`（3D点云）

3. **RViz配置**
   - Fixed Frame: `odom`
   - Map Topic: `/brain/map`
   - Map Enabled: true

4. **性能考虑**
   - 点云数据量大时，考虑降低更新频率
   - 地图越大，处理时间越长

---

## ✅ 总结

本脚本实现了从真实rosbag数据构建占据栅格地图的完整流程：

1. ✅ 订阅真实传感器数据
2. ✅ 处理3D点云
3. ✅ 构建占据栅格地图
4. ✅ 实现贝叶斯更新
5. ✅ 发布到 `/brain/map`
6. ✅ 支持持久化验证

**现在可以验证真实世界模型的持久化机制！** 🎯

