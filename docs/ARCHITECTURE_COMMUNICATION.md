# 架构通信和数据流分析

## 1. L2测试结果

### 测试状态
- **测试执行**: 已启动，正在收集传感器数据
- **VLM初始化**: ✅ 成功（已集成到感知层）
- **传感器数据**: ✅ 正在接收（点云、里程计）
- **ROS2接口**: ✅ 正常工作

### 测试输出摘要
```
VLM已初始化并传入感知层
ROS2SensorManager 初始化完成 (VLM已启用)
传感器数据正在接收：
  - 点云数据: 2000-4000个点/帧
  - 里程计数据: 正常更新
```

---

## 2. 认知层的信念（Belief）是什么？

### 信念的定义

**信念（Belief）**是认知层维护的关于世界的**假设**，例如：
- "杯子在厨房"
- "从A到B的路径畅通"
- "目标物体存在"
- "天气晴朗"

### 信念的数据结构

```python
@dataclass
class Belief:
    id: str                    # 信念ID
    content: str              # 信念内容，如 "杯子在厨房"
    confidence: float         # 置信度 0-1
    evidence_count: int       # 支持证据数量
    falsified: bool           # 是否已被证伪
    created_at: datetime      # 创建时间
    last_updated: datetime    # 最后更新时间
```

### 信念的类型

根据`BeliefType`枚举，信念分为：

1. **OBJECT_LOCATION** - 物体位置信念
   - 例如："cup在kitchen"
   - 用途：追踪物体位置

2. **PATH_ACCESSIBLE** - 路径可达性信念
   - 例如："从A到B的路径畅通"
   - 用途：路径规划

3. **OBJECT_EXISTS** - 物体存在性信念
   - 例如："目标物体存在"
   - 用途：目标搜索

4. **ENVIRONMENT_STATE** - 环境状态信念
   - 例如："天气晴朗"
   - 用途：环境理解

### 信念的生命周期

```
创建信念 → 收集证据 → 更新置信度 → 可能被证伪 → 移除
```

**信念更新机制：**
- **成功操作** → 置信度增加（+0.1）
- **失败操作** → 置信度降低（-0.2）
- **多次失败** → 置信度降至阈值以下 → 标记为已证伪
- **长时间未观测** → 置信度衰减

### 信念修正策略

认知层通过`BeliefUpdatePolicy`管理信念：

```python
# 根据观测结果更新信念
belief_update = await cognitive_layer.update_belief(observation_result)

# 返回：
# - 更新的信念列表
# - 已证伪的信念ID列表
# - 需要重新评估的假设
```

**核心原则：**
- 搜索失败 → 降低概率
- 多次失败 → 移除假设
- 执行成功 → 提高置信度
- 长时间未观测 → 置信度衰减

---

## 3. 感知层输出和通信机制

### 感知层最终输出：PerceptionData

感知层通过`ROS2SensorManager.get_fused_perception()`返回`PerceptionData`对象，包含：

#### 层次1：原始传感器数据
```python
- pose: Pose3D              # 机器人位姿 (x, y, z, roll, pitch, yaw)
- velocity: Velocity        # 速度 (linear, angular)
- rgb_image: np.ndarray     # RGB图像 (H, W, 3)
- depth_image: np.ndarray   # 深度图像 (H, W)
- pointcloud: np.ndarray    # 点云 (N, 3)
- laser_ranges: List[float] # 激光雷达距离
- laser_angles: List[float] # 激光雷达角度
```

#### 层次2：几何处理结果
```python
- obstacles: List[Dict]     # 障碍物列表（从激光雷达聚类）
  - id, type, position, size, distance, angle, direction
- occupancy_grid: np.ndarray # 占据栅格地图 (H, W)
- grid_resolution: float    # 栅格分辨率（米/栅格）
- grid_origin: Tuple[float, float] # 地图原点
```

#### 层次3：语义理解结果（新增）
```python
- semantic_objects: List[DetectedObject]  # 语义物体列表
  - id, label, confidence, bbox, description, position_description
- scene_description: Optional[SceneDescription]  # 场景描述
  - summary, objects, spatial_relations, navigation_hints
- spatial_relations: List[Dict]  # 空间关系
- navigation_hints: List[str]   # 导航提示
```

#### 传感器状态
```python
- sensor_status: Dict[str, bool]  # 传感器健康状态
```

---

### L1 → L2 通信（传感器 → 感知层）

**数据流：**
```
ROS2话题 → ROS2Interface → ROS2SensorManager → PerceptionData
```

**通信接口：**

1. **ROS2Interface** (L1层)
   - 订阅ROS2话题：
     - `/chassis/odom` → 里程计
     - `/front_3d_lidar/lidar_points` → 点云
     - `/front_stereo_camera/left/image_raw` → RGB图像
     - `/chassis/imu` → IMU数据
   - 将ROS2消息转换为内部格式（`SensorData`）

2. **ROS2SensorManager** (L2层)
   - 调用：`ros2_interface.get_sensor_data()` 获取原始数据
   - 进行传感器融合：
     - 位姿融合（里程计 + IMU）
     - 障碍物检测（激光雷达聚类）
     - 占据栅格生成（点云/深度图/激光）
     - **VLM场景理解**（RGB图像 → 语义信息）
   - 返回：`PerceptionData`（融合后的感知数据）

**代码示例：**
```python
# L1: ROS2Interface接收数据
sensor_data = ros2_interface.get_sensor_data()

# L2: ROS2SensorManager融合数据
perception_data = await sensor_manager.get_fused_perception()
```

---

### L2 → L3 通信（感知层 → 认知层）

**数据流：**
```
ROS2SensorManager → PerceptionData → CognitiveLayer.process_perception() → CognitiveOutput
```

**通信接口：**

1. **Brain._update_perception_loop()** (主循环)
   ```python
   async def _update_perception_loop(self):
       while self._running:
           # 从感知层获取数据
           perception_data = await self.sensor_manager.get_fused_perception()
           
           # 传递给认知层
           cognitive_output = await self.cognitive_layer.process_perception(perception_data)
   ```

2. **CognitiveLayer.process_perception()** (L3层)
   ```python
   async def process_perception(
       self, 
       perception_data: 'PerceptionData'
   ) -> CognitiveOutput:
       # 更新世界模型
       changes = self.world_model.update_from_perception(perception_data)
       
       # 从感知数据中获取语义对象（如果存在）
       if perception_data.semantic_objects:
           for obj in perception_data.semantic_objects:
               # 更新追踪对象到WorldModel
               self.world_model.update_tracked_object(...)
       
       # 获取规划上下文
       planning_context = self.world_model.get_context_for_planning()
       
       return CognitiveOutput(
           planning_context=planning_context,
           environment_changes=changes
       )
   ```

**数据传递：**
- **输入**: `PerceptionData`（来自感知层）
- **处理**: 
  - 更新`WorldModel`（机器人位置、地图、语义物体）
  - 检测环境变化（新障碍物、目标出现等）
  - 生成规划上下文
- **输出**: `CognitiveOutput`
  - `planning_context`: 规划上下文（位置、障碍物、目标等）
  - `environment_changes`: 环境变化列表

---

## 完整数据流图

```
┌─────────────────────────────────────────────────────────────┐
│ L1: 传感器层 (ROS2Interface)                                 │
│  - 订阅ROS2话题                                              │
│  - 转换消息格式                                              │
│  - 输出: SensorData                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ get_sensor_data()
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ L2: 感知层 (ROS2SensorManager)                              │
│  - 传感器融合                                                │
│  - 障碍物检测                                                │
│  - 占据栅格生成                                              │
│  - VLM场景理解 ← 新增                                        │
│  - 输出: PerceptionData                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ get_fused_perception()
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Brain主循环 (_update_perception_loop)                       │
│  - 持续获取感知数据                                          │
│  - 传递给认知层                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ process_perception(PerceptionData)
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ L3: 认知层 (CognitiveLayer)                                 │
│  - 更新WorldModel                                           │
│  - 检测环境变化                                              │
│  - 生成规划上下文                                            │
│  - 输出: CognitiveOutput                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ WorldModel (世界模型)                                        │
│  - 机器人状态                                                │
│  - 占据地图                                                  │
│  - 语义物体追踪                                              │
│  - 信念管理 (Belief)                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键改进（VLM迁移后）

### 改造前
```
感知层 → PerceptionData（无语义） → 认知层 → VLM分析 → WorldModel
```

### 改造后
```
感知层（包含VLM） → PerceptionData（含语义） → 认知层 → WorldModel
```

**优势：**
- ✅ 感知层输出完整（包含语义信息）
- ✅ 认知层专注于信念更新（不做感知处理）
- ✅ 数据流清晰，职责明确

---

## 总结

1. **L2测试**: 正在运行，VLM已成功集成到感知层
2. **信念**: 认知层维护的关于世界的假设，包含置信度、证据、可被证伪
3. **感知层输出**: `PerceptionData`包含原始数据、几何处理、语义理解三个层次
4. **L1→L2通信**: ROS2Interface → ROS2SensorManager（通过`get_sensor_data()`）
5. **L2→L3通信**: ROS2SensorManager → CognitiveLayer（通过`process_perception()`）





