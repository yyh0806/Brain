# RViz 持久化地图 - 快速参考卡

## ⚡ 快速开始（3步）

### 步骤1: 启动RViz
```bash
bash start_rviz2.sh
# ✅ 自动设置 ROS_DOMAIN_ID=42
```

### 步骤2: 启动地图发布器（二选一）

**选项A: 测试模式（模拟数据）**
```bash
python3 scripts/test_rviz_map.py
# ✅ 自动设置 ROS_DOMAIN_ID=42
# 发布模拟地图到 /brain/map
```

**选项B: 真实模式（Brain系统）**
```bash
# 启动Brain系统（让它从rosbag读取并发布地图）
python3 -c "from brain.core.brain import Brain; import asyncio; brain = Brain(); asyncio.run(brain.initialize())"
# ✅ 需要Brain系统支持
```

### 步骤3: 验证地图显示

在RViz中应该看到：
- ✅ Occupancy Grid面板显示地图（不再是"No map received"）
- ✅ 黑色区域 = 障碍物
- ✅ 白色区域 = 自由空间
- ✅ 灰色区域 = 未知

---

## ⚠️ 关键配置：ROS域ID

**所有脚本都使用 `ROS_DOMAIN_ID=42`**

| 脚本 | 域ID设置 |
|------|----------|
| `start_rviz2.sh` | ✅ 自动设置为 42 |
| `test_rviz_map.py` | ✅ 自动设置为 42 |
| `verify_world_model_rviz.py` | ✅ 自动设置为 42 |
| `ros2 bag play` | ✅ 继承环境变量 |

**验证域ID**：
```bash
echo $ROS_DOMAIN_ID
# 应该输出: 42
```

---

## 🎯 RViz配置要点

### Occupancy Grid面板配置

**必查项**：
- ✅ `Topic` = `/brain/map`（不是 `/map`！）
- ✅ `Color Scheme` = `map`
- ✅ `Alpha` = `0.7-0.9`（越大越清晰）

**如果看不到地图**：
1. 展开 `Occupancy Grid`
2. 检查 `Topic` 是否为 `/brain/map`
3. 检查 `Enabled` 是否勾选
4. 检查 `Fixed Frame` 是否为 `map` 或 `odom`

---

## 🔍 持久化验证要点

### 1. 障碍物稳定性 ⭐

**观察**：找一个黑色障碍物，看它是否保持黑色

**正确行为** ✓：
```
t=5s:   黑色
t=10s:  黑色（保持）
t=15s:  黑色（保持）
```

**错误行为** ✗：
```
t=5s:   黑色
t=10s:  白色（丢失！）
```

### 2. 自由空间扩展 ⭐

**观察**：机器人周围白色区域是否逐渐向外扩展

**正确行为** ✓：
```
t=0s:   灰色（未知）
t=10s:  3米内变白
t=20s:  5米内变白
t=30s:  10米内变白（扩展）
```

### 3. 已知区域保持 ⭐

**观察**：已标记区域（黑或白）是否保持颜色

**正确行为** ✓：
```
t=10s:  白色
t=20s:  白色（保持）
t=30s:  白色（保持，不变回灰色）
```

---

## 📊 话题订阅对照

| RViz面板 | 话题名称 | 数据来源 |
|----------|----------|----------|
| Occupancy Grid | `/brain/map` | RViz2Visualizer |
| Point Cloud | `/front_3d_lidar/lidar_points` | rosbag |
| Odometry | `/chassis/odom` | rosbag |
| RGB Camera | `/front_stereo_camera/left/image_raw` | rosbag |
| Robot Path | `/brain/robot_path` | RViz2Visualizer |

**注意**：Occupancy Grid 使用 `/brain/map`（不是 `/map`）

---

## 🛠️ 故障排查

### 问题1: RViz显示 "No map received"

**检查1**: `/brain/map` 话题是否存在？
```bash
ros2 topic list | grep brain/map
# 应该看到: /brain/map
```

**检查2**: 域ID是否一致？
```bash
echo $ROS_DOMAIN_ID
# 应该是: 42
```

**解决**:
- 如果话题不存在 → 运行 `test_rviz_map.py` 或启动Brain系统
- 如果域ID不一致 → 确保 `export ROS_DOMAIN_ID=42`

### 问题2: 地图全灰（无数据）

**检查**: 话题是否有数据？
```bash
ros2 topic echo /brain/map --once
# 应该看到地图消息
```

**解决**:
- 等待地图发布（可能需要几秒）
- 检查发布者是否正常运行

### 问题3: 地图不清晰

**解决**:
- 调整Occupancy Grid的Alpha到0.9
- 调整视图到俯视角度
- 暂时关闭Point Cloud减少干扰

---

## 📚 相关文档

| 文档 | 用途 |
|------|------|
| `RVIZ_PERSISTENT_MAP_GUIDE.md` | 详细使用指南 |
| `RVIZ_FIX_NO_MAP_RECEIVED.md` | 问题排查 |
| `RVIZ_QUICK_REFERENCE.md` | 本文档（快速参考）|

---

## ✅ 验证清单

- [ ] RViz成功启动
- [ ] ROS_DOMAIN_ID = 42（所有进程一致）
- [ ] `/brain/map` 话题存在
- [ ] Occupancy Grid显示地图（不再"No map received"）
- [ ] 障碍物（黑色）稳定不变
- [ ] 自由空间（白色）逐渐扩展
- [ ] 已知区域保持（不变回灰色）
- [ ] 贝叶斯更新生效（一次观测不立即改变）

---

## 💡 快速命令参考

```bash
# 启动RViz
bash start_rviz2.sh

# 测试地图发布
python3 scripts/test_rviz_map.py

# 启动监控助手
python3 scripts/verify_world_model_rviz.py

# 查看话题列表
ros2 topic list

# 查看地图话题
ros2 topic info /brain/map

# 查看地图数据
ros2 topic echo /brain/map --once

# 查看话题频率
ros2 topic hz /brain/map

# 播放rosbag
ros2 bag play data/rosbags/*.db3

# 检查域ID
echo $ROS_DOMAIN_ID

# 设置域ID
export ROS_DOMAIN_ID=42
```

---

**最后更新**: 2025-12-26
**ROS域ID**: 42
**地图话题**: /brain/map

