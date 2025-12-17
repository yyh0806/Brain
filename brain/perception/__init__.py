"""感知模块"""
from brain.perception.sensors.sensor_manager import SensorManager
# EnvironmentModel 已删除，功能合并到 WorldModel
from brain.perception.object_detector import ObjectDetector

__all__ = ["SensorManager", "ObjectDetector"]

