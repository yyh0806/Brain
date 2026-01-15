# WebViz感知层可视化 - 快速开始

## 概述

本系统使用WebViz（开源Web可视化工具）来全面展示Brain感知层输出，包括rosbag回放和实时ROS2连接两种模式。

## 快速开始（3步即可）

### 方式A: 使用现有rosbag文件

```bash
# 步骤1: 运行快速开始脚本
cd /media/yangyuhui/CODES1/Brain
bash scripts/quickstart_webviz.sh

# 步骤2: 按照提示在浏览器中打开WebViz
#    https://webviz.io/app/

# 步骤3: 按照脚本提示配置5个面板
#    - Image Panel: 查看RGB相机
#    - 3D Panel: 查看点云
#    - Map Panel: 查看占据地图
#    - Plot Panel: 查看IMU和里程计
#    - Raw Messages Panel: 查看世界模型状态
```

### 方式B: 录制新的rosbag然后回放

```bash
# 终端1: 启动Brain系统
cd /media/yangyuhui/CODES1/Brain
python3 brain/brain.py

# 终端2: 录制感知数据
cd /media/yangyuhui/CODES1/Brain
bash scripts/record_perception_bag.sh

# 让系统运行一段时间（比如30秒到2分钟），收集数据

# 终端3: 回放并可视化
cd /media/yangyuhui/CODES1/Brain
bash scripts/quickstart_webviz.sh
```

## 文件结构

```
Brain/
├── scripts/                           # 新增脚本
│   ├── record_perception_bag.sh      # rosbag录制脚本
│   ├── play_and_visualize_webviz.sh  # rosbag回放脚本
│   ├── start_webviz_viz.sh          # 综合启动脚本
│   ├── verify_world_model.py        # 世界模型验证工具
│   └── quickstart_webviz.sh          # 快速开始脚本（推荐）
│
├── config/webviz/                    # WebViz配置文件
│   ├── nova_carter_topics.yaml     # 话题配置
│   └── perception_layout.json         # 面板布局配置
│
├── docs/
│   ├── WEBVIZ_USAGE_GUIDE.md        # 详细使用指南
│   └── WEBVIZ_QUICKSTART.md          # 本文件
│
└── data/rosbags/                     # rosbag存储目录
```

## WebViz面板配置详解

### 1. Image Panel（RGB相机）

**添加方法：**
1. 点击右上角 "Add Panel"
2. 选择 "Image"
3. 配置：
   - Image topic: `/front_stereo_camera/left/image_raw`

**观察内容：**
- RGB相机实时画面
- 帧率和延迟信息

### 2. 3D Panel（点云可视化）

**添加方法：**
1. 点击 "Add Panel" → 选择 "3D"
2. 配置：
   - Topic: `/front_3d_lidar/lidar_points`
   - Color: #ff0000（红色）
   - Size: 0.05
   - Color by: intensity

**观察内容：**
- 3D点云可视化
- 机器人位置和轨迹
- 障碍物分布

### 3. Map Panel（占据地图）

**添加方法：**
1. 点击 "Add Panel" → 选择 "Map"
2. 配置：
   - Map topic: `/map`
   - Color scheme: map（标准占据地图）
   - Robot base: `base_link`

**颜色含义：**
- **黑色** = 占据（障碍物）
- **白色** = 自由（可通行）
- **灰色** = 未知（未探索）

**观察内容（验证持久化）：**
- 障碍物位置是否稳定（不应频繁闪烁）
- 自由空间是否逐渐扩展（从机器人到障碍物的射线变白）
- 已探索区域是否保持已知（不会变回灰色）

### 4. Plot Panel（IMU和里程计）

**添加方法：**
1. 点击 "Add Panel" → 选择 "Plot"
2. 添加以下路径：
   - `/chassis/odom.pose.pose.position.x` (X Position, 红色)
   - `/chassis/odom.pose.pose.position.y` (Y Position, 绿色)
   - `/chassis/imu.linear_acceleration.x` (Accel X, 蓝色)

**观察内容：**
- 机器人位置轨迹
- IMU加速度数据
- 数据的连续性和合理性

### 5. Raw Messages Panel（世界模型状态）

**添加方法：**
1. 点击 "Add Panel" → 选择 "Raw Messages"
2. 配置：
   - Topic: `/world_model`
   - Subscribe: 勾选
   - Show full message: 勾选

**观察内容（验证持久化）：**
- `metadata.confidence`: 整体置信度（应递增）
- `update_count`: 更新次数（应递增）
- `semantic_objects`: 语义物体列表

## 世界模型持久化验证

### 验证清单

使用此清单验证世界模型是否正确工作：

#### 占据地图持久化
- [ ] 障碍物在多次扫描后保持稳定
- [ ] 自由空间逐渐扩展
- [ ] 已探索区域不退回到未知状态
- [ ] 贝叶斯更新生效（一次观测不立即改变已占据区域）

#### 语义物体持久化
- [ ] 同一物体的`update_count`随时间递增
- [ ] 物体`confidence`随观测次数提升
- [ ] 长时间未观测的物体`confidence`下降
- [ ] 低置信度物体被自动移除（< 0.1）

#### 时间衰减机制
- [ ] 短期观测维持高置信度
- [ ] 长期未观测置信度指数下降
- [ ] 衰减率可配置（`semantic_decay`参数）

### 在WebViz中的验证方法

#### 方法1: 查看Raw Messages Panel

```json
// 典型的持久化世界模型状态
{
  "metadata": {
    "confidence": 0.85,        // 随时间递增
    "update_count": 150,          // 持续递增
    "map_age_seconds": 300
  },
  "semantic_objects": {
    "door": {
      "confidence": 0.92,      // 多次观测后提升
      "update_count": 5         // 持久化证据
    },
    "shelf": {
      "confidence": 0.65,      // 持续观测但置信度较低
      "update_count": 3
    }
  },
  "persistence": {
    "decay_rate": 0.05,
    "is_persistent": true         // 数据已持久化
  }
}
```

#### 方法2: 观察Map Panel变化

记录不同时刻的地图状态：
- t=0s: 地图中心区域为灰色（未知）
- t=10s: 机器人周围变成白色（自由）
- t=20s: 发现障碍物，显示黑色
- t=30s: 障碍物仍为黑色（持久化）
- t=40s: 机器人移动，更多区域变成白色
- t=60s: 早期障碍物仍为黑色（持久化验证）

#### 方法3: 使用验证工具

```bash
# 运行验证工具（需要Brain系统在运行）
cd /media/yangyuhui/CODES1/Brain

# 实时验证模式
python3 scripts/verify_world_model.py --live
```

验证工具会输出详细报告，包括：
- 占据地图持久化分数
- 语义物体持久化统计
- 时间衰减机制验证
- 贝叶斯更新机制验证

## 故障排查

### 问题1: WebViz无法连接到rosbag

**症状：** 面板显示"Waiting for data..."

**可能原因：**
1. rosbag文件未开始播放
2. 话题名称不匹配
3. ROS2域ID未设置

**解决方案：**
```bash
# 检查rosbag是否在播放
ros2 bag info data/rosbags/your_bag.db3

# 重新启动rosbag（确保设置ROS2域ID）
export ROS_DOMAIN_ID=42
ros2 bag play data/rosbags/your_bag.db3 --rate 1.0
```

### 问题2: 某些话题没有数据

**症状：** 特定面板一直显示"Waiting for data..."

**可能原因：**
1. Brain系统未发布该话题
2. 话题配置名称不匹配

**解决方案：**
```bash
# 查看当前发布的话题
ros2 topic list

# 查看话题详细信息
ros2 topic info /front_stereo_camera/left/image_raw
ros2 topic info /front_3d_lidar/lidar_points
ros2 topic info /map
ros2 topic info /world_model  # 如果存在
```

### 问题3: 地图显示全灰（未知）

**症状：** Map面板中所有区域都是灰色

**可能原因：**
1. `/map`话题未发布
2. OccupancyMapper未启用
3. 传感器数据未到达

**解决方案：**
1. 确认Brain系统正在运行
2. 检查Brain日志中的OccupancyMapper相关消息
3. 确认激光雷达/点云数据正常到达

### 问题4: 点云显示异常

**症状：** 3D面板中点云闪烁、缺失或颜色错误

**可能原因：**
1. 点云数据量过大
2. WebViz渲染性能问题
3. 网络带宽不足

**解决方案：**
1. 调整WebViz中的Point size参数（减小到0.03）
2. 减少Decay time（从默认改为0.5）
3. 检查网络连接
4. 考虑使用本地WebViz（需额外安装）

## 高级功能

### 自定义布局保存

1. 配置好所有面板
2. 点击右上角"Layout"图标
3. 选择"Save layout..."
4. 保存为JSON文件
5. 下次加载：点击"Load layout..."选择保存的文件

### 录制演示

1. 使用屏幕录制软件（如OBS Studio）
2. 在WebViz中演示世界模型持久化
3. 说明关键观察点

### 批量处理

如果想批量处理多个rosbag文件：

```bash
cd /media/yangyuhui/CODES1/Brain

# 处理所有rosbag文件
for bag in data/rosbags/*.db3; do
    echo "处理: $bag"
    bash scripts/play_and_visualize_webviz.sh "$bag"
    
    # 提示用户手动确认下一步
    read -p "按Enter继续下一个rosbag..."
done
```

## 性能优化建议

### WebViz端优化

1. **减少渲染复杂度**
   - 降低点云点大小
   - 减少点云保留时间
   - 关闭不必要的面板

2. **调整刷新频率**
   - Image Panel可以降低刷新率
   - Map Panel可以降低刷新率

3. **网络优化**
   - 使用本地WebViz而非在线版
   - 确保网络稳定

### Brain系统端优化

1. **降低数据发布频率**
   - 适当降低地图更新频率（如从10Hz降到5Hz）
   - 降低点云发布频率

2. **数据压缩**
   - 启用图像压缩
   - 降采样点云数据

3. **选择性发布**
   - 只在需要时发布世界模型状态
   - 降低语义物体发布频率

## 下一步

完成WebViz可视化后，你可以：

1. **调优参数**
   - 调整地图分辨率、衰减率等
   - 优化发布频率

2. **扩展功能**
   - 添加更多可视化面板（如2D Pose、IMU图表等）
   - 实现自定义数据类型
   - 添加实时图表分析

3. **自动化测试**
   - 集成到CI/CD流程
   - 自动化验证报告生成
   - 创建性能基准测试

4. **用户界面改进**
   - 创建自定义Web界面
   - 添加交互式控制
   - 实现历史数据回放

## 参考资料

- WebViz官方文档: https://github.com/cruise-automation/webviz
- WebViz在线应用: https://webviz.io/app/
- ROS2 bag文档: https://docs.ros.org/en/humble/Tutorials/Ros2bag/Tutorials.html
- 详细使用指南: `/media/yangyuhui/CODES1/Brain/docs/WEBVIZ_USAGE_GUIDE.md`

## 获取帮助

如果遇到问题：

1. 查看详细使用指南: `cat docs/WEBVIZ_USAGE_GUIDE.md`
2. 运行验证工具获取诊断信息: `python3 scripts/verify_world_model.py --live`
3. 检查Brain系统日志
4. 确认所有传感器正常工作

祝你可视化愉快！




