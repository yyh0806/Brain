# 认知层世界模型RViz可视化指南

## 概述

本指南介绍如何使用RViz2可视化认知层世界模型，包括信念状态、语义物体、探索边界和环境变化事件的可视化。

## 前置条件

### 1. 启动Ollama LLaVA模型

```bash
ollama run llava:7b
```

### 2. 启动Brain系统

```bash
cd /media/yangyuhui/CODES1/Brain
python3 brain/brain.py
```

### 3. 设置ROS Domain ID

```bash
export ROS_DOMAIN_ID=42
```

## 快速开始

### 方法1: 使用启动脚本（推荐）

```bash
cd /media/yangyuhui/CODES1/Brain
bash scripts/start_cognitive_rviz_viz.sh
```

### 方法2: 使用测试脚本（包含Brain启动）

```bash
cd /media/yangyuhui/CODES1/Brain
bash scripts/test_cognitive_viz_with_real_data.sh
```

### 方法3: 手动启动

```bash
# 设置Domain ID
export ROS_DOMAIN_ID=42

# 启动RViz2
ros2 run rviz2 rviz2 -d /media/yangyuhui/CODES1/Brain/config/rviz2/cognitive_world_model.rviz
```

## RViz2面板说明

### 1. 语义占据地图

- **话题：** /world_model/semantic_grid
- **颜色含义：**
  - 灰色（-1）：未知
  - 白色（0）：自由
  - 黑色（100）：占据
  - 蓝色（101）：门
  - 红色（102）：人
  - 绿色（103）：建筑
  - 橙色（104）：障碍物
  - 紫色（105）：目标
  - 黄色（106）：兴趣点

- **Alpha:** 0.7
- **Color Scheme:** map

### 2. 语义物体标注

- **话题：** /world_model/semantic_markers
- **Namespace:** semantic_labels
- **显示内容：** 物体标签（悬浮文字）
- **颜色编码：** 门（蓝色）、人（红色）、建筑（绿色）、障碍物（橙色）

### 3. 信念状态标记

- **话题：** /world_model/belief_markers
- **Namespace:** belief_markers
- **Marker类型:** SPHERE（球体）
- **颜色含义：**
  - 绿色：高置信度（>0.8）
  - 黄色：中等置信度（0.5-0.8）
  - 红色：低置信度（<0.5）
  - 灰色：已证伪
- **大小编码：** 置信度越高，球体越大（0.1-0.3米）

### 4. 机器人轨迹

- **话题：** /world_model/trajectory
- **颜色：** 绿色
- **线宽：** 0.1米
- **显示：** 最近100个位姿点的路径

### 5. 探索边界

- **话题：** /world_model/frontiers
- **Marker类型:** ARROW（箭头）+ TEXT_VIEW_FACING（文字）
- **颜色编码：**
  - 亮绿色：高优先级（>0.8）大箭头
  - 黄色：中等优先级（0.5-0.8）中箭头
  - 灰色：低优先级（<0.5）小箭头
- **文字显示：** 优先级+距离（如：P:0.85 D:5m）

### 6. 变化事件

- **话题：** /world_model/change_events
- **显示内容：** 临时Marker，5秒后自动消失
- **事件类型：** 新障碍物（橙色圆柱）、目标移动（紫色圆柱）、路径阻塞（红色X）、障碍物移动（黄色圆柱）、目标出现（绿色圆柱）、障碍物移除（蓝色虚线框）

### 7. VLM检测

- **话题：** /vlm/detections
- **显示内容：** 边界框、标签、置信度
- **来源标识：** 只显示source='vlm'的物体

### 8. RGB相机图像

- **话题：** /front_stereo_camera/left/image_raw
- **用途：** 查看实际相机图像

### 9. 3D点云

- **话题：** /front_3d_lidar/lidar_points
- **Size (m):** 0.05
- **Color Transformer:** AxisColor

## 常见问题排查

### 问题1: RViz2无法连接到话题

**解决方案：**
```bash
# 检查Domain ID
echo $ROS_DOMAIN_ID

# 检查话题列表
ros2 topic list | grep world_model
```

### 问题2: 信念标记不显示

**可能原因：**
- 信念修正策略未启用
- 没有注册任何信念

**解决方案：**
- 检查WorldModel是否有信念修正策略
- 查看Brain系统日志

### 问题3: VLM检测不显示

**可能原因：**
- VLM服务未运行
- 没有VLM检测到物体

**解决方案：**
```bash
# 检查VLM是否运行
pgrep -f "ollama run llava"

# 检查话题
ros2 topic list | grep vlm
```

## 总结

本指南介绍了认知层世界模型的完整RViz可视化系统，包括信念状态可视化、VLM检测结果可视化、探索边界可视化和环境变化事件可视化。所有可视化功能由单一的WorldModelVisualizer节点管理，具有统一管理、性能优化、时间同步、配置集中、易于维护等优势。
