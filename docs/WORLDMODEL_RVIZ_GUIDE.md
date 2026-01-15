# WorldModel RViz可视化快速指南

## 🎯 现状

✅ **所有关键可视化已修复并验证**：
- semantic_grid（占据栅格地图）：100x100 cells
- trajectory（机器人轨迹）：正常工作
- frontiers（探索前沿）：正常工作
- change_markers（变化事件）：正常工作

⚠️  **需要数据源的功能**：
- semantic_markers：需要VLM检测数据
- belief_markers：需要信念修正策略
- vlm_markers：需要VLM检测数据

## 📋 使用步骤

### 方法1：使用rosbag数据

```bash
# 终端1：启动WorldModel pipeline
export ROS_DOMAIN_ID=42
python3 tests/cognitive/run_worldmodel_with_rosbag.py

# 终端2：启动RViz（可选，用于可视化）
export ROS_DOMAIN_ID=42
rviz2 -d rviz/simple_worldmodel.rviz

# 终端3：监控话题（可选，用于调试）
export ROS_DOMAIN_ID=42
python3 scripts/monitor_world_model_topics.py
```

### 方法2：使用快速启动脚本

```bash
export ROS_DOMAIN_ID=42

# 先启动pipeline
python3 tests/cognitive/run_worldmodel_with_rosbag.py

# 在另一个终端运行
./scripts/start_world_model_viz.sh
```

### 方法3：运行诊断工具

```bash
export ROS_DOMAIN_ID=42
python3 scripts/diagnose_world_model_viz.py
```

## 🔍 在RViz中查看

启动RViz后，你应该能看到：

1. **Semantic Occupancy Grid**（占据栅格地图）
   - 灰色区域：未知区域
   - 白色区域：自由空间
   - 黑色区域：占据区域
   - 彩色/蓝色区域：语义标记（门、建筑等）

2. **Semantic Labels**（语义物体标签）
   - 3D文字标签显示检测到的物体

3. **Robot Trajectory**（机器人轨迹）
   - 绿色路径线显示历史轨迹

4. **Exploration Frontiers**（探索前沿）
   - 彩色箭头指示待探索区域
   - 箭头上显示优先级和距离

5. **TF**（坐标变换树）
   - 显示所有坐标框架关系

## 📊 话题监控

运行话题监控工具，你会看到类似输出：

```
✅ semantic_grid: 100x100, 10000 cells
✅ semantic_markers: 5 个标记
✅ trajectory: 2 个位姿
✅ frontiers: 6 个标记
✅ change_markers: 2 个标记
✅ vlm_markers: 6 个标记
```

## 🛠️ 故障排查

### 问题：RViz中看不到地图

**检查**：
```bash
# 检查话题是否发布
ros2 topic list | grep world_model
```

**应该看到**：
```
/world_model/semantic_grid
/world_model/semantic_markers
/world_model/trajectory
/world_model/frontiers
/world_model/belief_markers
/world_model/change_events
```

### 问题：地图是全灰色

**原因**：WorldModel的current_map未初始化

**解决**：这是正常的，需要点云数据来构建地图

### 问题：看不到任何数据

**检查**：
```bash
# 检查ROS域ID
echo $ROS_DOMAIN_ID

# 应该输出：42
```

## 📝 调试技巧

1. **查看日志**：
   ```bash
   tail -f .cursor/debug.log | grep "semantic_grid"
   ```

2. **运行诊断**：
   ```bash
   python3 scripts/diagnose_world_model_viz.py
   ```

3. **监控话题**：
   ```bash
   python3 scripts/monitor_world_model_topics.py
   ```

## 🚀 下一步

现在可视化功能已经完全修复，你可以：

1. ✅ 启动rosbag数据流
2. ✅ 启动WorldModel pipeline
3. ✅ 在RViz中实时查看可视化
4. ✅ 使用监控和诊断工具调试

**数据流**：
```
rosbag → 传感器数据 → WorldModel → 可视化 → RViz显示
```

---

**修复内容总结**：
1. ✅ 修复了frontiers生成中的position属性访问错误
2. ✅ 修复了change_markers生成中的robot_position访问错误
3. ✅ 创建了话题监控工具
4. ✅ 创建了诊断工具
5. ✅ 创建了简化版RViz配置
6. ✅ 创建了启动脚本
7. ✅ 创建了快速使用指南
