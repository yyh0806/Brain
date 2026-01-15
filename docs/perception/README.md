# Brain 感知模块文档

本文档提供Brain感知模块的详细说明，包括架构、接口、使用示例和测试方法。

## 目录

1. [模块概述](overview.md)
2. [传感器接口](sensor_interfaces.md)
3. [多传感器融合](sensor_fusion.md)
4. [ROS2集成](ros2_integration.md)
5. [目标检测](object_detection.md)
6. [视觉语言模型感知](vlm_perception.md)
7. [占据栅格地图](occupancy_mapping.md)
8. [测试指南](testing.md)
9. [Isaac Sim集成](isaac_sim_integration.md)

## 快速开始

要使用感知模块，请参考以下示例：

```python
from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
from brain.perception.object_detector import ObjectDetector
from brain.perception.vlm.vlm_perception import VLMPerception

# 初始化ROS2传感器管理器
sensor_manager = ROS2SensorManager(ros2_interface)

# 获取融合后的感知数据
perception_data = await sensor_manager.get_fused_perception()

# 使用目标检测器
detector = ObjectDetector()
detections = await detector.detect(perception_data.rgb_image)

# 使用VLM进行场景理解
vlm = VLMPerception()
scene_description = await vlm.describe_scene(perception_data.rgb_image)
```

更多详细示例请参考各模块的文档页面。









