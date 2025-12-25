# 目标检测文档

Brain感知模块提供目标检测与跟踪功能，支持多种检测模型和跟踪算法。

## 目标检测器使用

### 基本使用

```python
from brain.perception.object_detector import ObjectDetector

# 创建目标检测器
detector = ObjectDetector({
    "mode": "fast",               # 检测模式: fast/accurate/tracking
    "confidence_threshold": 0.5,  # 置信度阈值
})

# 检测图像中的目标
image = get_image()  # 获取图像 (H, W, 3) numpy数组
detections = await detector.detect(image)

# 处理检测结果
for detection in detections:
    print(f"检测到 {detection.object_type}, 置信度: {detection.confidence}")
    bbox = detection.bounding_box_2d  # (x, y, w, h)
    position = detection.position_3d   # 3D位置 (如果有深度信息)
```

### 目标跟踪

```python
# 检测并跟踪目标
tracked_objects = await detector.detect_and_track(image, depth_map)

# 处理跟踪结果
for obj in tracked_objects:
    print(f"跟踪ID: {obj.track_id}, 类型: {obj.object_type}")
    print(f"当前位置: {obj.position.x}, {obj.position.y}, {obj.position.z}")
    print(f"速度: vx={obj.velocity['vx']}, vy={obj.velocity['vy']}")
    
    # 预测未来位置
    future_pos = obj.predict_position(dt=1.0)
    print(f"1秒后预测位置: {future_pos.x}, {future_pos.y}, {future_pos.z}")
```

### 检测特定类型目标

```python
from brain.perception.environment import ObjectType

# 只检测特定类型的目标
target_types = [ObjectType.PERSON, ObjectType.VEHICLE]
person_vehicle_detections = await detector.detect_specific(image, target_types)

# 在特定区域检测
roi = (100, 100, 200, 200)  # (x, y, w, h)
roi_detections = await detector.detect_in_area(image, roi)
```

## 检测模式

### 快速模式 (Fast)

适用于实时应用，速度快但精度可能较低：

```python
detector = ObjectDetector({"mode": "fast"})
detections = await detector.detect(image)
```

### 精确模式 (Accurate)

适用于精度要求高的场景，速度较慢：

```python
detector = ObjectDetector({"mode": "accurate"})
detections = await detector.detect(image)
```

### 跟踪模式 (Tracking)

在连续帧之间跟踪目标：

```python
detector = ObjectDetector({"mode": "tracking"})
tracked_objects = await detector.detect_and_track(image)
```

## 数据结构

### Detection

```python
from brain.perception.object_detector import Detection

detection = Detection(
    object_type=ObjectType.PERSON,     # 目标类型
    confidence=0.85,                  # 置信度 (0-1)
    bounding_box_2d=(100, 100, 50, 120),  # (x, y, w, h)
    position_3d=Position3D(x=2.0, y=1.0, z=0.0),  # 3D位置(可选)
    attributes={"color": "red"}        # 其他属性
)
```

### TrackedObject

```python
from brain.perception.object_detector import TrackedObject

tracked_obj = TrackedObject(
    track_id="person_1",              # 跟踪ID
    object_type=ObjectType.PERSON,      # 目标类型
    position=Position3D(x=2.0, y=1.0, z=0.0),  # 当前位置
    velocity={"vx": 0.5, "vy": 0.2, "vz": 0.0},   # 速度
    history=[Position3D(...)]          # 位置历史
)

# 获取跟踪历史
history = tracked_obj.history  # 位置历史列表

# 跟踪状态
lost_frames = tracked_obj.lost_frames  # 丢失帧数
age = tracked_obj.age                 # 跟踪年龄
```

## 目标类型

```python
from brain.perception.environment import ObjectType

# 支持的目标类型
print(ObjectType.PERSON)      # 人
print(ObjectType.VEHICLE)     # 车辆
print(ObjectType.BUILDING)     # 建筑
print(ObjectType.TREE)        # 树木
print(ObjectType.OBSTACLE)    # 障碍物
print(ObjectType.UNKNOWN)      # 未知
```

## 跟踪管理

### 获取所有跟踪

```python
# 获取所有活跃跟踪
all_tracks = detector.get_all_tracks()

# 获取特定跟踪
track = detector.get_track("person_1")
if track:
    print(f"跟踪对象位置: {track.position}")
```

### 清除跟踪

```python
# 清除所有跟踪
detector.clear_tracks()

# 手动移除特定跟踪
if "person_1" in detector.tracks:
    del detector.tracks["person_1"]
```

## 转换为环境对象

```python
from brain.perception.environment import DetectedObject

# 转换跟踪为环境对象
tracks = detector.get_all_tracks()
env_objects = detector.to_detected_objects(tracks)

# 处理环境对象
for obj in env_objects:
    print(f"环境对象ID: {obj.id}, 类型: {obj.object_type}")
    print(f"位置: {obj.position.x}, {obj.position.y}")
    print(f"速度: {obj.velocity}")
```

## 检测配置

```python
config = {
    "mode": "tracking",              # 检测模式
    "confidence_threshold": 0.5,      # 置信度阈值
    "max_tracking_distance": 3.0,     # 最大跟踪距离(米)
    "max_lost_frames": 10,           # 最大丢失帧数
    "min_track_length": 3,           # 最小跟踪长度
    "tracking_memory": 50,            # 跟踪历史大小
    "detection_model": "yolov8n",     # 检测模型
    "tracking_algorithm": "kalman"     # 跟踪算法
}

detector = ObjectDetector(config)
```

## 性能优化

### 区域检测

只检测感兴趣区域，提高处理速度：

```python
# 定义多个感兴趣区域
regions = [
    (0, 0, 320, 240),      # 左半边
    (320, 0, 320, 240)      # 右半边
]

# 并行检测区域
detections = []
for roi in regions:
    roi_detections = await detector.detect_in_area(image, roi)
    detections.extend(roi_detections)
```

### 帧跳跃

对于高帧率视频，可以跳过部分帧：

```python
import asyncio

frame_count = 0
skip_frames = 2  # 每处理一帧，跳过2帧

async def process_frames():
    global frame_count
    while True:
        image = get_next_frame()
        frame_count += 1
        
        if frame_count % (skip_frames + 1) == 0:
            # 处理当前帧
            detections = await detector.detect(image)
            # 处理检测结果...
        
        await asyncio.sleep(0.033)  # 约30fps
```

## 故障处理

### 检测失败处理

```python
try:
    detections = await detector.detect(image)
except Exception as e:
    print(f"检测失败: {e}")
    # 使用默认值或上一帧结果
    detections = []
```

### 跟踪丢失处理

```python
# 检查跟踪状态
for track_id, track in detector.tracks.items():
    if track.lost_frames > 5:
        print(f"跟踪 {track_id} 可能已丢失")
    
    if track.lost_frames > 10:
        print(f"移除丢失的跟踪 {track_id}")
        del detector.tracks[track_id]
```

## 相关文档

- [传感器接口](sensor_interfaces.md)
- [多传感器融合](sensor_fusion.md)
- [视觉语言模型感知](vlm_perception.md)






