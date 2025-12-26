# WebViz感知层可视化使用指南

## 概述

本指南帮助你使用WebViz可视化Brain系统的感知层输出，验证世界模型的持久化机制。

## 前置条件

1. **ROS2环境**: 确保ROS2已正确安装和配置
2. **Python环境**: Python 3.8+
3. **浏览器**: Chrome/Firefox（支持WebGL）
4. **网络连接**（如使用在线WebViz）

## 快速开始

### 方式1: 使用现有rosbag文件（推荐）

```bash
# 1. 进入Brain项目目录
cd /media/yangyuhui/CODES1/Brain

# 2. 如果有rosbag文件，直接使用
bash scripts/play_and_visualize_webviz.sh data/rosbags/your_bag.db3

# 3. 按提示选择方法1（在线WebViz）

# 4. 打开浏览器访问
#    https://webviz.io/app/
```

### 方式2: 记录新的rosbag然后回放

```bash
# 1. 启动Brain系统（在终端1）
cd /media/yangyuhui/CODES1/Brain
python3 brain/brain.py

# 2. 在另一个终端开始记录（终端2）
cd /media/yangyuhui/CODES1/Brain
bash scripts/record_perception_bag.sh

# 3. 让系统运行一段时间，收集足够的数据

# 4. 按Ctrl+C停止录制

# 5. 回放并可视化（终端3）
bash scripts/play_and_visualize_webviz.sh data/rosbags/perception_20251226_xxxx.db3
```

### 方式3: 实时连接到正在运行的系统

```bash
# 1. 启动Brain系统（终端1）
cd /media/yangyuhui/CODES1/Brain
python3 brain/brain.py

# 2. 启动WebViz连接（终端2）
cd /media/yangyuhui/CODES1/Brain
bash scripts/start_webviz_viz.sh live

# 3. 打开浏览器访问
#    https://webviz.io/app/
```

### 方式4: 验证世界模型持久化

```bash
# 使用验证工具检查世界模型行为
cd /media/yangyuhui/CODES1/Brain

# 从rosbag验证
python3 scripts/verify_world_model.py --bag data/rosbags/perception.db3

# 或实时验证（需要Brain系统正在运行）
python3 scripts/verify_world_model.py --live
```

## WebViz面板配置

### 1. 3D Visualization面板（查看点云）

**配置步骤：**
1. 点击"Add Panel" → 选择"3D"
2. 在面板设置中：
   - 添加话题: `/front_3d_lidar/lidar_points`
   - Color: #ff0000（红色）
   - Size: 0.05
   - Color by: intensity

**观察重点：**
- 点云是否完整显示
- 点云密度是否合理
- 机器人位置轨迹是否平滑

### 2. Image面板（查看RGB图像）

**配置步骤：**
1. 点击"Add Panel" → 选择"Image"
2. 配置:
   - Image topic: `/front_stereo_camera/left/image_raw`
   - Scale: 1.0
   - Show timestamp: 勾选

**观察重点：**
- 图像是否清晰
- 帧率是否流畅
- 是否有延迟

### 3. Map面板（查看占据地图）

**配置步骤：**
1. 点击"Add Panel" → 选择"Map"
2. 配置:
   - Map topic: `/map`
   - Color scheme: map（标准占据地图颜色）
   - Show path: 勾选
   - Robot base: `base_link`

**观察重点：**
- **黑色区域** = 占据（障碍物）
- **白色区域** = 自由（可通行）
- **灰色区域** = 未知（未探索）
- 地图是否随时间更新（持久化）

### 4. Plot面板（查看IMU和里程计）

**配置步骤：**
1. 点击"Add Panel" → 选择"Plot"
2. 添加以下路径：
   - `/chassis/odom.pose.pose.position.x` (X Position, 红色)
   - `/chassis/odom.pose.pose.position.y` (Y Position, 绿色)
   - `/chassis/imu.linear_acceleration.x` (Accel X, 蓝色)
   - `/chassis/imu.linear_acceleration.y` (Accel Y, 紫色)

**观察重点：**
- 位置轨迹是否连续
- 加速度是否合理
- 是否有异常跳变

### 5. Raw Messages面板（查看世界模型状态）

**配置步骤：**
1. 点击"Add Panel" → 选择"Raw Messages"
2. 配置:
   - Topic: `/world_model`
   - Subscribe: 勾选
   - Show full message: 勾选

**观察重点：**
- `confidence`: 整体置信度
- `update_count`: 更新次数（应递增）
- `semantic_objects`: 语义物体列表

## 世界模型持久化验证指标

### 1. 占据地图持久化

**期望行为：**
- 多次扫描后，障碍物位置保持稳定
- 自由空间逐渐扩展（从机器人到障碍物的路径变白）
- 已探索区域保持已知（不会变回未知）

**在WebViz Map面板中观察：**
- ✓ 黑色区域（障碍物）在多次更新后保持不变
- ✓ 白色区域（自由空间）逐渐扩大
- ✓ 已探索区域不会变回灰色

**失败表现：**
- ✗ 障碍物位置频繁变化（不稳定）
- ✗ 自由空间不扩展或收缩
- ✗ 已探索区域变回未知

### 2. 语义物体持久化

**期望行为：**
- 同一物体的`update_count`随时间递增
- 物体`confidence`随观测次数提升
- 长时间未观测的物体，`confidence`逐渐下降

**在WebViz Raw Messages面板中观察：**
- ✓ 查看特定物体（如"door"）的状态
- ✓ `update_count`应随时间增加：1 → 2 → 3...
- ✓ `confidence`应递增：0.7 → 0.75 → 0.82...

**失败表现：**
- ✗ 物体每次都被视为新物体（update_count始终为1）
- ✗ 置信度不提升或随机波动
- ✗ 物体立即消失

### 3. 时间衰减机制

**期望行为：**
- 物体在持续观测时，confidence保持高位
- 物体停止观测后，confidence按指数衰减
- 低于阈值（0.1）的物体被自动移除

**验证方法：**
1. 在Raw Messages面板中记录某物体的confidence
2. 观察几分钟（rosbag回放速度）
3. 如果物体持续出现，confidence应保持或上升
4. 如果物体消失，confidence应下降

**在WebViz中观察时间线：**
- 在Plot面板中添加路径：`/world_model/semantic_objects.door.confidence`
- 观察confidence随时间的变化曲线

### 4. 贝叶斯更新机制

**核心原理：**
- 不是简单覆盖：`grid[10,10] = OCCUPIED`
- 而是概率更新：
  - 已占据的栅格，需要多次观测为自由才会变
  - 已自由的栅格，一次观测占据就会变
  - 未知栅格，直接设置观测值

**验证方法：**
1. 记录某区域的栅格状态（黑色/白色/灰色）
2. 观察后续更新
3. 已占据的黑色区域应该稳定存在
4. 从起点到障碍物的射线应该显示为自由

**在WebViz Map面板中观察：**
- ✓ 障碍物（黑色）即使一次观测为自由，也保持黑色
- ✓ 自由路径（白色）射线穿过未知区域时会将其填充
- ✓ 已知区域（黑或白）不会变回未知（灰色）

## 故障排查

### 问题1: WebViz无法连接到rosbag

**症状：** 3D/Map面板显示"Waiting for data..."

**解决方案：**
1. 检查rosbag文件是否存在
2. 确认rosbag是否在播放：`ros2 bag info your_bag.db3`
3. 检查话题名称是否正确
4. 查看rosbag中的话题列表：`ros2 bag info your_bag.db3 | grep topics`

### 问题2: 某些话题没有数据

**症状：** 特定面板一直显示"Waiting for data..."

**解决方案：**
1. 检查Brain系统是否发布该话题
2. 查看配置文件中的话题映射：`config/nova_carter_ros2.yaml`
3. 确认话题名称是否匹配
4. 使用`ros2 topic list`查看发布的话题

### 问题3: 地图显示全灰（未知）

**症状：** Map面板中所有区域都是灰色

**解决方案：**
1. 检查`/map`话题是否发布
2. 检查话题类型：`ros2 topic info /map`
3. 确认Brain系统中启用了OccupancyMapper
4. 查看是否有`/world_model`话题的更新

### 问题4: 点云显示异常

**症状：** 3D面板中点云闪烁、缺失或颜色错误

**解决方案：**
1. 检查点云数据源：`ros2 topic hz /front_3d_lidar/lidar_points`
2. 检查数据类型：`ros2 topic info /front_3d_lidar/lidar_points`
3. 调整WebViz中的点大小和颜色配置
4. 检查QoS设置（可能存在丢包）

## 高级用法

### 自定义WebViz布局

1. 保存当前布局：WebViz → 布局图标 → 保存布局
2. 加载自定义布局：使用`config/webviz/perception_layout.json`
3. 导出布局：复制当前配置到新JSON文件

### 调整显示参数

**3D面板：**
- Point size: 0.03-0.08（调整点大小）
- Decay time: 0.0-0.5（调整点保留时间）
- Color by: height/intensity（调整着色方式）

**Map面板：**
- Alpha: 0.3-0.7（调整地图透明度）
- Threshold: 0.1-0.5（调整显示阈值）
- Resolution: 0.05-0.2（调整地图分辨率）

### 录制WebViz视图

1. 使用WebViz的"Record"功能（如果支持）
2. 或使用屏幕录制软件
3. 记录关键场景用于分析和演示

## 验证清单

使用此清单确保世界模型持久化正确工作：

### 基础功能
- [ ] rosbag成功录制
- [ ] rosbag包含所有感知话题
- [ ] WebViz能够连接到rosbag
- [ ] 所有5个面板正常显示数据

### 占据地图持久化
- [ ] 障碍物位置稳定（多次更新后保持）
- [ ] 自由空间逐渐扩展
- [ ] 已探索区域保持已知
- [ ] 射线填充正确（自由路径可见）

### 语义物体持久化
- [ ] 同一物体update_count递增
- [ ] 物体confidence随观测提升
- [ ] 长时间未观测confidence下降
- [ ] 低confidence物体被自动移除

### 时间衰减机制
- [ ] 短期观测维持高confidence
- [ ] 长期未观测confidence下降
- [ ] 衰减曲线符合指数规律
- [ ] 阈值（0.1）正确工作

### 整体性能
- [ ] 地图覆盖率逐渐提升
- [ ] 整体confidence合理（>0.5）
- [ ] 更新速率稳定
- [ ] 无明显内存泄漏

## 常见问题

### Q: 为什么看不到世界模型状态？

A: 可能原因：
1. `/world_model`话题未发布
2. Brain系统中的WorldModel未初始化
3. Raw Messages面板未正确订阅

解决方案：检查Brain日志，确认世界模型初始化和话题发布

### Q: 占据地图一直显示全灰？

A: 可能原因：
1. OccupancyMapper未启用
2. 激光雷达/点云数据未到达
3. 坐标变换不正确

解决方案：检查Brain日志，查看"OccupancyMapper update"相关消息

### Q: 点云显示不完整？

A: 可能原因：
1. 点云数据量过大，浏览器无法渲染
2. WebViz配置限制显示的点数
3. 网络带宽不足

解决方案：调整WebViz中的渲染参数，或使用本地WebViz

## 下一步

验证完成后，你可以：
1. 调整WorldModel参数（分辨率、衰减率等）
2. 优化感知层性能
3. 集成到自动化测试流程
4. 扩展更多可视化面板

## 参考资料

- WebViz官方文档: https://github.com/cruise-automation/webviz
- WebViz在线应用: https://webviz.io/app/
- ROS2 bag文档: https://docs.ros.org/en/humble/Tutorials/Ros2bag/Tutorials.html
- Brain系统文档: `/media/yangyuhui/CODES1/Brain/docs/`

