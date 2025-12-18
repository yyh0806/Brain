# -*- coding: utf-8 -*-
"""
Brain Cognitive World Model Module

This module provides the cognitive world model implementation for the Brain system,
including sensor input processing, multi-sensor fusion, and situational awareness.
"""

from brain.cognitive.world_model.world_model import WorldModel, EnvironmentChange, ChangeType, ChangePriority

# Sensor input types and interfaces
from brain.cognitive.world_model.sensor_input_types import (
    SensorDataPacket,
    PointCloudData,
    ImageData,
    IMUData,
    GPSData,
    WeatherData,
    CameraIntrinsics,
    SensorType,
    SensorQuality,
)

from brain.cognitive.world_model.sensor_interface import (
    BaseSensor,
    PointCloudSensor,
    ImageSensor,
    IMUSensor,
    GPSSensor,
)

from brain.cognitive.world_model.sensor_manager import (
    MultiSensorManager,
    SensorSyncStatus,
    DataQualityAssessment,
    SynchronizedDataPacket,
)

from brain.cognitive.world_model.data_converter import (
    DataConverter,
    ROS2Converter,
    StandardFormatConverter,
)

__all__ = [
    # Existing world model
    "WorldModel",
    "EnvironmentChange",
    "ChangeType",
    "ChangePriority",

    # Data types
    "SensorDataPacket",
    "PointCloudData",
    "ImageData",
    "IMUData",
    "GPSData",
    "WeatherData",
    "CameraIntrinsics",
    "SensorType",
    "SensorQuality",

    # Sensor interfaces
    "BaseSensor",
    "PointCloudSensor",
    "ImageSensor",
    "IMUSensor",
    "GPSSensor",

    # Management
    "MultiSensorManager",
    "SensorSyncStatus",
    "DataQualityAssessment",

    # Converters
    "DataConverter",
    "ROS2Converter",
    "StandardFormatConverter",
]
