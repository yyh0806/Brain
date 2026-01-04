# WebViz感知层可视化系统 - 完整指南

## 概述

本系统使用WebViz（开源Web可视化工具）来全面展示Brain感知层输出，并提供世界模型持久化机制的验证能力。

## 快速开始（3步即可）

### 方法1: 使用现有rosbag（最简单，推荐）

```bash
# 步骤1: 运行快速开始脚本
cd /media/yangyuhui/CODES1/Brain
bash scripts/quickstart_webviz.sh

# 步骤2: 打开浏览器访问WebViz
#    https://webviz.io/app/

# 步骤3: 按照脚本提示配置5个面板
#    - Image Panel: RGB相机
#    - 3D Panel: 点云
#    - Map Panel: 占据地图
#    - Plot Panel: IMU和里程计
#    - Raw Messages Panel: 世界模型状态
```

### 方法2: 录制新的rosbag然后回放

```bash
# 步骤1: 启动Brain系统
cd /media/yangyuhui/CODES1/Brain
python3 brain/brain.py

# 步骤2: 在另一个终端开始录制
cd /media/yangyuhui/CODES1/Brain
bash scripts/record_perception_bag.sh

# 步骤3: 让系统运行一段时间（30秒-2分钟）收集数据

# 步骤4: 停止录制（Ctrl+C）

# 步骤5: 回放并可视化
cd /media/yangyuhui/CODES1/Brain
bash scripts/quickstart_webviz.sh
```

### 方法3: 连接到实时运行的Brain系统

```bash
# 步骤1: 启动Brain系统
cd /media/yangyuhui/CODES1/Brain
python3 brain/brain.py

# 步骤2: 在另一个终端启动WebViz连接
cd /media/yangyuhui/CODES1/Brain
bash scripts/start_webviz_viz.sh live

# 步骤3: 打开浏览器访问
#    https://webviz.io/app/

# 步骤4: 在WebViz中配置面板（同方法1）
```

## 文件结构

```
Brain/
├── scripts/                           # 新增的脚本文件
│   ├── record_perception_bag.sh      # rosbag录制脚本
│   ├── play_and_visualize_webviz.sh  # rosbag回放脚本
│   ├── start_webviz_viz.sh          # 综合启动脚本
│   ├── verify_world_model.py        # 世界模型验证工具
│   └── quickstart_webviz.sh          # ⭐ 快速开始脚本（推荐）
│
├── config/webviz/                    # WebViz配置文件
│   ├── nova_carter_topics.yaml     # 话题配置清单
│   └── perception_layout.json         # 面板布局配置
│
├── docs/                              # 文档文件
│   ├── WEBVIZ_README.md              # 本文件
│   ├── WEBVIZ_QUICKSTART.md          # ⭐ 快速开始指南（推荐阅读）
│   ├── WEBVIZ_USAGE_GUIDE.md        # 详细使用指南
│   └── WEBVIZ_IMPLEMENTATION_SUMMARY.md  # 实现总结文档
│
├── brain/communication/
│   └── publish_world_model.py        # ROS2世界模型话题发布器
│
└── data/rosbags/                      # rosbag存储目录
```

## WebViz面板配置详解

### 1. Image Panel（RGB相机）

**添加方法：**
1. 点击"Add Panel" → 选择"Image"
2. 配置：
   - Image topic: `/front_stereo_camera/left/image_raw`
   - Show timestamp: 勾选
   - Synchronize: 勾选

**观察内容：**
- RGB相机实时画面
- 图像质量和帧率信息

**用于验证：**
- [ ] 图像是否清晰
- [ ] 帧率是否流畅（~30fps）
- [ ] 是否有延迟

### 2. 3D Panel（点云可视化）

**添加方法：**
1. 点击"Add Panel" → 选择"3D"
2. 配置：
   - Topic: `/front_3d_lidar/lidar_points`
   - Color: #ff0000（红色）
   - Size: 0.05
   - Color by: intensity
   - Decay time: 0.0（不衰减）

**观察内容：**
- 3D点云可视化
- 机器人位置和轨迹
- 障碍物分布

**用于验证：**
- [ ] 点云是否完整显示
- [ ] 点云密度是否合理
- [ ] 机器人位置轨迹是否平滑

### 3. Map Panel（占据地图）- ⭐最重要

**添加方法：**
1. 点击"Add Panel" → 选择"Map"
2. 配置：
   - Map topic: `/map`
   - Color scheme: map（标准占据地图）
   - Show path: 勾选
   - Robot base: `base_link`

**颜色含义：**
- **黑色** = 占据（障碍物）
- **白色** = 自由（可通行）
- **灰色** = 未知（未探索）

**用于验证：**
- [ ] 障碍物位置是否稳定（持久化）
- [ ] 自由空间是否逐渐扩展
- [ ] 已探索区域是否保持已知（不变回灰色）

**持久化验证指标：**
1. 障碍物稳定性：多次扫描后，同一位置的障碍物应保持黑色
2. 自由空间扩展：从机器人到障碍物的射线路径应逐渐变为白色
3. 已知区域保持：已探索的区域不应变回灰色

### 4. Plot Panel（IMU和里程计）

**添加方法：**
1. 点击"Add Panel" → 选择"Plot"
2. 添加以下路径：
   - `/chassis/odom.pose.pose.position.x` (X Position, 红色)
   - `/chassis/odom.pose.pose.position.y` (Y Position, 绿色)
   - `/chassis/imu.linear_acceleration.x` (Accel X, 蓝色)
   - `/chassis/imu.linear_acceleration.y` (Accel Y, 紫色)

**配置：**
- X Axis: time
- X Axis Val: timestamp
- Show Legend: 勾选

**观察内容：**
- 机器人位置轨迹
- IMU加速度数据
- 数据的连续性和合理性

**用于验证：**
- [ ] 位置轨迹是否连续
- [ ] 加速度是否合理（无异常跳变）
- [ ] 数据更新频率是否稳定

### 5. Raw Messages Panel（世界模型状态）

**添加方法：**
1. 点击"Add Panel" → 选择"Raw Messages"
2. 配置：
   - Topic: `/world_model`
   - Subscribe: 勾选
   - Show full message: 勾选

**观察内容：**
- JSON格式的世界模型完整状态
- 包括元数据、占据地图、语义物体等

**用于验证：**
- [ ] `confidence`: 整体置信度（应递增）
- [ ] `update_count`: 更新次数（应递增）
- [ ] `semantic_objects`: 语义物体列表
- [ ] 持久化指标是否正确

## 世界模型持久化验证

### 核心验证点

#### 1. 占据地图持久化（Map Panel观察）

**期望行为：**
- ✅ 多次扫描后，障碍物位置保持稳定
- ✅ 自由空间逐渐扩展（从机器人到障碍物路径变白）
- ✅ 已探索区域保持已知（不变回未知）

**验证方法：**
1. 在Map面板中记录某区域的状态（黑/白/灰）
2. 观察10秒、20秒、30秒后的变化
3. 确认障碍物（黑色）在后续扫描中保持不变
4. 确认自由路径（白色）逐渐扩大

**在WebViz中观察：**
- [ ] 黑色区域（障碍物）在多次更新后保持不变
- [ ] 白色区域（自由空间）逐渐扩大
- [ ] 已探索区域（黑或白）不会变回灰色（未知）

**失败表现：**
- ✗ 障碍物位置频繁变化（不稳定）
- ✗ 自由空间不扩展或收缩
- ✗ 已探索区域变回未知

#### 2. 语义物体持久化（Raw Messages Panel观察）

**期望行为：**
- ✅ 同一物体的`update_count`随时间递增
- ✅ 物体`confidence`随观测次数提升
- ✅ 长时间未观测的物体，`confidence`逐渐下降
- ✅ 低于阈值的物体被自动移除

**验证方法：**
1. 在Raw Messages面板中找到`semantic_objects`字段
2. 查看特定物体（如"door"）的状态
3. 记录`update_count`和`confidence`的值
4. 等待30秒，再次观察

**在WebViz中观察：**
- [ ] 查看特定物体（如"door"）的状态
- [ ] `update_count`应随时间增加：1 → 2 → 3...
- [ ] `confidence`应递增：0.7 → 0.75 → 0.8...
- [ ] 物体消失后，`confidence`应下降

**失败表现：**
- ✗ 物体每次都被视为新物体（update_count始终为1）
- ✗ 置信度不提升或随机波动
- ✗ 物体立即消失

#### 3. 时间衰减机制（验证）

**期望行为：**
- ✅ 物体在持续观测时，confidence保持高位
- ✅ 物体停止观测后，confidence按指数衰减
- ✅ 衰减曲线符合：conf(t) = conf(0) * exp(-k * t)

**验证方法：**
1. 在Plot面板中添加路径查看某个物体confidence
   - Topic: `/world_model/semantic_objects.door.confidence`
2. 观察confidence随时间的变化曲线
3. 让物体消失（机器人离开），观察曲线下降

**在WebViz中观察：**
- [ ] 观察confidence随时间的变化曲线
- [ ] 持续观测时，confidence保持或上升
- [ ] 物体消失后，confidence按指数下降
- [ ] 低于阈值（0.1）的物体从列表中消失

**失败表现：**
- ✗ confidence随机波动
- ✗ 不衰减或衰减过快
- ✗ 衰减不符合指数规律

#### 4. 贝叶斯更新机制（Map Panel观察）

**核心原理：**
- 不是简单覆盖：`grid[10,10] = OCCUPIED`
- 而是概率更新：
  - 已占据的栅格，需要多次观测为自由才会变
  - 已自由的栅格，一次观测占据就会变
  - 未知栅格，直接设置观测值

**验证方法：**
1. 记录某区域的栅格状态（黑/白/灰）
2. 观察后续更新
3. 已占据的黑色区域应该稳定存在
4. 从起点到障碍物的射线应该显示为自由

**在WebViz中观察：**
- [ ] 已占据的黑色区域即使一次观测为自由，也保持黑色
- [ ] 自由路径（白色）射线穿过未知区域时会将其填充
- [ ] 已知区域（黑或白）不会变回未知（灰色）

**失败表现：**
- ✗ 一次观测就立即改变状态
- ✗ 自由路径不显示
- ✗ 已知区域变回未知

### 验证工具使用

如果想要自动化的验证报告：

```bash
cd /media/yangyuhui/CODES1/Brain

# 从rosbag验证（离线）
python3 scripts/verify_world_model.py --bag data/rosbags/perception_xxx.db3

# 实时验证（需要Brain系统运行）
python3 scripts/verify_world_model.py --live
```

验证工具会输出详细的报告，包括：
- 占据地图持久化分数
- 语义物体持久化统计
- 时间衰减机制验证
- 贝叶斯更新机制验证
- 整体性能评估

## 常见问题排查

### 问题1: WebViz无法连接到rosbag

**症状：**
- 3D/Map面板显示"Waiting for data..."

**解决方案：**
1. 检查rosbag文件是否存在
2. 确认rosbag是否在播放
3. 检查话题名称是否正确
4. 查看rosbag中的话题列表：`ros2 bag info your_bag.db3 | grep topics`

### 问题2: 某些话题没有数据

**症状：**
- 特定面板一直显示"Waiting for data..."

**解决方案：**
1. 检查Brain系统是否发布该话题
2. 查看配置文件中的话题映射：`config/nova_carter_ros2.yaml`
3. 确认话题名称是否匹配
4. 使用`ros2 topic list`查看发布的话题

### 问题3: 地图显示全灰（未知）

**症状：**
- Map面板中所有区域都是灰色

**解决方案：**
1. 检查`/map`话题是否发布
2. 检查话题类型：`ros2 topic info /map`
3. 确认Brain系统中启用了OccupancyMapper
4. 查看是否有`/world_model`话题的更新

### 问题4: 点云显示异常

**症状：**
- 3D面板中点云闪烁、缺失或颜色错误

**解决方案：**
1. 检查点云数据源：`ros2 topic hz /front_3d_lidar/lidar_points`
2. 检查数据类型：`ros2 topic info /front_3d_lidar/lidar_points`
3. 调整WebViz中的点大小和颜色配置
4. 检查QoS设置（可能存在丢包）

## 性能优化建议

### WebViz端优化

1. **减少渲染复杂度**
   - 降低3D面板点大小
   - 减少点云保留时间
   - 关闭不必要的面板

2. **调整刷新频率**
   - Image面板可以降低刷新率
   - Map面板可以降低刷新率

3. **网络优化**
   - 使用本地WebViz而非在线版
   - 确保网络带宽充足

### Brain系统端优化

1. **降低数据发布频率**
   - 适当降低地图更新频率
   - 降低点云发布频率

2. **数据压缩**
   - 启用图像压缩
   - 降采样点云数据

3. **选择性发布**
   - 只在需要时发布世界模型状态
   - 降低语义物体发布频率

## 下一步建议

### 短期目标

1. **测试验证**
   - 测试所有脚本功能
   - 验证世界模型持久化机制
   - 收集性能数据

2. **参数调优**
   - 调整地图分辨率、衰减率等参数
   - 优化发布频率
   - 优化QoS设置

3. **文档完善**
   - 补充更多使用示例
   - 添加更多故障排查指南
   - 编写测试报告

### 长期目标

1. **功能扩展**
   - 添加更多可视化面板
   - 实现自定义数据类型
   - 添加历史数据回放功能

2. **自动化测试**
   - 集成到CI/CD流程
   - 自动化验证报告生成
   - 创建性能基准测试

3. **用户界面改进**
   - 创建自定义Web界面
   - 添加交互式控制
   - 实现性能监控和分析

## 参考资料

- WebViz官方文档: https://github.com/cruise-automation/webviz
- WebViz在线应用: https://webviz.io/app/
- ROS2 bag文档: https://docs.ros.org/en/humble/Tutorials/Ros2bag/Tutorials.html
- Brain项目文档: `/media/yangyuhui/CODES1/Brain/docs/`

## 获取帮助

如果遇到问题：

1. **查看详细使用指南**
   ```bash
   cat docs/WEBVIZ_USAGE_GUIDE.md
   ```

2. **查看快速开始指南**
   ```bash
   cat docs/WEBVIZ_QUICKSTART.md
   ```

3. **查看实现总结**
   ```bash
   cat docs/WEBVIZ_IMPLEMENTATION_SUMMARY.md
   ```

4. **运行验证工具**
   ```bash
   python3 scripts/verify_world_model.py --live
   ```

5. **检查Brain系统日志**
   - 查看是否有OccupancyMapper相关的错误
   - 确认所有传感器正常工作

祝你使用愉快！



