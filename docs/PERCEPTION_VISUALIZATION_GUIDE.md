# 感知数据可视化指南

## 1. RGB图像（双目相机）

### 当前配置
- **左眼**: `/front_stereo_camera/left/image_raw`
- **右眼**: 未配置（需要添加）

### 建议配置
在`config/environments/isaac_sim/nova_carter.yaml`中添加：
```yaml
ros2_interface:
  topics:
    rgb_image: "/front_stereo_camera/left/image_raw"
    rgb_image_right: "/front_stereo_camera/right/image_raw"  # 添加右眼
```

### 可视化改进
- 可以同时显示左右两个图像
- 用于立体视觉和深度估计

---

## 2. 激光雷达数据（Lidar Data）

### 数据来源
- **话题**: `/front_3d_lidar/lidar_points` (PointCloud2)
- **处理**: 转换为极坐标格式（ranges, angles）

### 显示内容
- 极坐标转笛卡尔坐标的点云
- 机器人位置（绿色圆点）
- 扫描范围可视化

---

## 3. 障碍物（Obstacles）

### 数据含义
障碍物是从**激光雷达数据**中检测出来的物体，包含以下信息：

```python
{
    "id": "obs_0",                    # 障碍物ID
    "type": "unknown",                 # 类型（未知/静态/动态）
    "local_position": {                # 局部坐标（相对于机器人）
        "x": 2.5,                      # 前方距离
        "y": 0.3                       # 左右偏移
    },
    "world_position": {                # 世界坐标
        "x": 10.2,
        "y": 5.1
    },
    "size": 0.8,                       # 障碍物大小（米）
    "distance": 2.52,                  # 距离机器人距离（米）
    "angle": 0.12,                     # 角度（弧度）
    "direction": "front",               # 方向（front/left/right/back）
    "point_count": 15                  # 激光点数量
}
```

### 检测方法
1. **聚类算法**: 将相邻的激光点聚合成障碍物
2. **距离阈值**: 相邻点距离 < 0.3米
3. **最小点数**: 至少3个点才认为是障碍物
4. **大小过滤**: 过滤掉太小的障碍物（< 最小尺寸）

### 可视化
- 红色圆圈表示障碍物
- 圆圈大小 = 障碍物尺寸
- 标签显示ID和距离

---

## 4. 语义物体（Semantic Objects）

### 数据来源
- **VLM（视觉语言模型）**: 使用Ollama `llava:7b`进行场景理解
- **检测频率**: 根据配置的`analysis_interval`（默认2秒）

### 数据结构
```python
DetectedObject(
    id="vlm_0",
    label="door",                      # 物体标签
    confidence=0.7,                    # 置信度
    bbox=BoundingBox(...),             # 边界框（图像坐标）
    description="A red door",          # 描述
    position_description="center",      # 位置描述
    estimated_distance=5.0,            # 估计距离（米）
    estimated_direction="front"         # 方向
)
```

### 为什么可能没有显示？
1. **VLM未运行**: 检查VLM是否启用
2. **分析频率**: 可能还没到分析时间
3. **图像为空**: RGB图像为空时不会运行VLM
4. **Ollama未启动**: 确保Ollama服务运行中

### 检查方法
```bash
# 检查Ollama是否运行
curl http://localhost:11434/api/tags

# 检查VLM配置
grep -A 5 "vlm:" config/environments/isaac_sim/nova_carter.yaml
```

---

## 5. 占据栅格地图（Occupancy Grid Map）

### 颜色含义

| 颜色 | 值 | 含义 | 说明 |
|------|-----|------|------|
| **黑色** | 100 | **占据（Occupied）** | 有障碍物的区域，不可通行 |
| **白色** | 0 | **自由（Free）** | 可通行区域 |
| **灰色** | -1 | **未知（Unknown）** | 未探索区域 |

### 地图生成
1. **从深度图**: 使用深度信息更新地图
2. **从激光雷达**: 使用激光扫描数据更新
3. **从点云**: 使用3D点云数据更新

### 地图参数
- **分辨率**: 0.1米/栅格（默认）
- **地图大小**: 50米 x 50米（默认）
- **原点**: 地图中心（-25, -25）到（25, 25）

### 可视化说明
- **红色圆点**: 机器人位置
- **红色箭头**: 机器人朝向
- **黑色区域**: 障碍物/不可通行
- **白色区域**: 可通行
- **灰色区域**: 未探索

---

## 改进建议

### 1. 添加右眼图像显示
修改可视化脚本，支持显示左右两个RGB图像。

### 2. 改进语义物体显示
- 在RGB图像上绘制边界框
- 显示物体标签和置信度
- 在地图上标记语义物体的位置

### 3. 障碍物详细信息
- 显示障碍物类型
- 显示障碍物方向
- 显示最近障碍物的详细信息

### 4. 占据地图改进
- 显示地图更新频率
- 显示已探索区域百分比
- 显示障碍物密度




