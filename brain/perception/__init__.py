"""感知模块"""
from brain.perception.sensors.sensor_manager import MultiSensorManager
# EnvironmentModel 已删除，功能合并到 WorldModel
from brain.perception.object_detector import ObjectDetector

# 向后兼容：SensorManager 作为 MultiSensorManager 的别名
SensorManager = MultiSensorManager

__all__ = ["SensorManager", "MultiSensorManager", "ObjectDetector"]

