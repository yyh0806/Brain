# RViz "No map received" 问题解决方案

## ⚠️ 重要提示：ROS域ID配置

**所有脚本都使用 `ROS_DOMAIN_ID=42`**

确保所有进程在同一个ROS域中通信！

- ✅ `start_rviz2.sh` - 自动设置 `ROS_DOMAIN_ID=42`
- ✅ `test_rviz_map.py` - 自动设置 `ROS_DOMAIN_ID=42`
- ✅ `verify_world_model_rviz.py` - 自动设置 `ROS_DOMAIN_ID=42`
- ✅ `ros2 bag play` - 会继承环境变量的 `ROS_DOMAIN_ID=42`

**验证方法**：
```bash
# 检查当前域ID
echo $ROS_DOMAIN_ID

# 应该输出: 42
```

---

## 问题描述

在RViz中启动后，Occupancy Grid面板显示 **"No map received"**，说明没有接收到占据栅格地图数据。

## 原因分析

**原因**：`/brain/map` 话题未发布

- RViz配置文件订阅 `/brain/map` 话题
- 但Brain系统没有运行，所以没有发布地图数据
- 或者Brain系统发布了地图，但话题名称不匹配
- 或者ROS域ID不匹配（例如一个用42，一个用0）

## 解决方案

### 方案1: 测试RViz配置（快速验证）⭐

如果你只是想快速验证RViz配置是否正确，可以运行测试脚本：

```bash
# 终端1: 启动RViz
bash start_rviz2.sh

# 终端2: 运行测试脚本，发布模拟地图
python3 scripts/test_rviz_map.py
```

**预期结果**：
- RViz中会显示一个模拟的占据地图
- 包含各种障碍物形状：
  - 中心一个矩形障碍物（黑色）
  - 左上角一个L形障碍物（黑色）
  - 右下角一个圆形障碍物（黑色）
  - 中心向外的自由空间射线（白色）

**停止测试**：
- 在运行测试脚本的终端按 `Ctrl+C`

---

### 方案2: 启动Brain系统（获取真实数据）⭐⭐⭐

如果你想看到真实的占据地图和验证持久化机制，需要启动Brain系统：

#### 步骤1: 停止当前的rosbag播放

```bash
# 按Ctrl+C停止rosbag播放
```

#### 步骤2: 启动Brain系统

Brain系统会：
- 订阅rosbag中的传感器数据
- 处理数据并更新占据地图
- 发布地图到 `/brain/map` 话题

```bash
python3 -c "
from brain.core.brain import Brain
import asyncio

async def main():
    print('初始化Brain系统...')
    brain = Brain(config_path='config/nova_carter_ros2.yaml')
    
    print('启动感知模块...')
    await brain.initialize()
    await brain.start_perception()
    
    print('Brain已启动，正在发布地图到 /brain/map')
    print('按Ctrl+C停止')
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print('\\n停止Brain系统...')
        await brain.shutdown()

asyncio.run(main())
"
```

#### 步骤3: 启动RViz

```bash
# 新终端: 启动RViz
bash start_rviz2.sh
```

#### 步骤4: 重新播放rosbag

```bash
# 新终端: 播放rosbag
ros2 bag play data/rosbags/<你的rosbag文件>
```

**预期结果**：
- RViz中显示真实的占据地图
- 地图随着rosbag播放逐渐更新
- 可以验证持久化机制：
  - 障碍物（黑色）保持稳定
  - 自由空间（白色）逐渐扩展
  - 已探索区域不变回灰色（未知）

---

### 方案3: 检查话题名称

如果Brain系统正在运行，但RViz仍显示"No map received"，检查话题名称：

#### 检查当前发布的话题

```bash
# 列出所有话题
ros2 topic list

# 查找地图相关话题
ros2 topic list | grep -i map
```

#### 预期结果

应该看到：
```
/brain/map          ← 占据栅格地图
/brain/robot_path   ← 机器人轨迹
/brain/visualization_markers  ← 可视化标记
```

#### 如果话题名称不同

如果在RViz配置中看到 `/brain/map`，但实际发布的是其他名称（如 `/map`），需要：

**方法1: 修改RViz配置文件**

编辑 `config/rviz2/nova_carter_persistent_map.rviz`：

找到 `Occupancy Grid` 部分，修改 `Topic`：

```yaml
Topic:
  Value: /map  # 改成实际的话题名称
```

**方法2: 在RViz中手动修改**

1. 在左侧Displays面板，展开 `Occupancy Grid`
2. 找到 `Topic` 参数
3. 输入实际的话题名称（如 `/map`）
4. 按回车

---

## 验证地图发布

### 方法1: 使用命令行查看

```bash
# 查看话题信息
ros2 topic info /brain/map

# 查看话题数据（会打印地图消息）
ros2 topic echo /brain/map --once
```

### 方法2: 使用测试脚本

```bash
# 运行测试脚本
python3 scripts/test_rviz_map.py

# 在另一个终端查看话题
ros2 topic echo /brain/map --once
```

应该看到类似这样的输出：

```yaml
header:
  stamp:
    sec: 1703567890
    nanosec: 123456789
  frame_id: "map"
info:
  map_load_time:
    sec: 0
    nanosec: 0
  resolution: 0.1
  width: 500
  height: 500
  origin:
    position:
      x: -25.0
      y: -25.0
      z: 0.0
    orientation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
data: [-1, -1, -1, ..., 100, 100, 100, ..., 0, 0, 0, ...]
```

---

## 常见问题排查

### Q1: 测试脚本运行后，RViz仍显示"No map received"

**检查步骤**：

1. **确认RViz订阅的话题**
   - 在RViz左侧面板，展开 `Occupancy Grid`
   - 查看 `Topic` 参数是否为 `/brain/map`
   - 如果不是，手动修改为 `/brain/map`

2. **确认话题正在发布**
   ```bash
   ros2 topic list | grep brain/map
   ```
   应该看到 `/brain/map`

3. **确认RViz的Fixed Frame**
   - 在RViz顶部 `Fixed Frame` 下拉框
   - 选择 `map` 或 `odom`
   - 如果选择其他（如 `base_link`），可能看不到地图

### Q2: Brain系统启动后，仍看不到地图

**检查步骤**：

1. **查看Brain系统日志**
   - Brain启动时是否有错误
   - 是否显示"发布地图到 /brain/map"

2. **检查ROS2域ID**
   ```bash
   echo $ROS_DOMAIN_ID
   ```
   Brain系统和RViz应该使用相同的域ID（默认42）

3. **检查OccupancyMapper是否启用**
   - 查看Brain配置文件
   - 确认 `perception.mapping.enabled: true`

### Q3: 地图显示全灰（无数据）

**可能原因**：
- OccupancyMapper未初始化
- 传感器数据未正确接收
- 地图更新未触发

**解决方法**：
1. 检查Point Cloud是否显示
2. 检查Odometry是否更新
3. 查看Brain系统日志

### Q4: 地图显示但不更新

**检查更新频率**：

1. **查看地图发布频率**
   ```bash
   ros2 topic hz /brain/map
   ```
   应该看到 `average rate: 2.0` 左右

2. **检查RViz的Frame Rate**
   - RViz顶部菜单 → `Panels` → `Views` → `Frame Rate`
   - 设置为 30 或更高

---

## 快速参考

### 启动顺序

#### 测试模式（模拟数据）
```bash
# 终端1
bash start_rviz2.sh

# 终端2
python3 scripts/test_rviz_map.py
```

#### 真实模式（Brain + rosbag）
```bash
# 终端1: 启动Brain
python3 -c "..."  # 见方案2

# 终端2: 启动RViz
bash start_rviz2.sh

# 终端3: 播放rosbag
ros2 bag play data/rosbags/*.db3
```

### 话题名称对照

| 话题名称 | 用途 | 发布者 |
|---------|------|--------|
| `/brain/map` | 占据栅格地图 | RViz2Visualizer |
| `/brain/robot_path` | 机器人轨迹 | RViz2Visualizer |
| `/brain/visualization_markers` | 可视化标记 | RViz2Visualizer |
| `/brain/robot_pose` | 机器人位姿 | RViz2Visualizer |

### RViz配置文件

- `nova_carter_persistent_map.rviz` - 持久化地图专用（推荐）
- `perception_world_model.rviz` - 完整感知可视化
- `nova_carter_perfect.rviz` - 通用配置

所有配置文件的Occupancy Grid都订阅 `/brain/map`。

---

## 总结

**核心问题**：`/brain/map` 话题未发布

**解决方法**：
1. **快速测试**：运行 `test_rviz_map.py`
2. **真实数据**：启动Brain系统
3. **检查话题**：确认话题名称和发布状态

**验证步骤**：
1. 运行 `ros2 topic list | grep brain/map` 查看话题是否存在
2. 运行 `ros2 topic echo /brain/map --once` 查看话题数据
3. 在RViz中检查Occupancy Grid的Topic参数

如果问题仍未解决，请提供：
- `ros2 topic list` 的输出
- RViz截图
- Brain系统日志

