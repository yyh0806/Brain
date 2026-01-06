# -*- coding: utf-8 -*-
"""
统一的数据类型定义
"""

from enum import Enum, IntEnum


class SensorType(Enum):
    """传感器类型"""
    POINT_CLOUD = "point_cloud"
    IMAGE = "image"
    IMU = "imu"
    GPS = "gps"
    WEATHER = "weather"
    LIDAR = "lidar"
    RADAR = "radar"
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    THERMAL_CAMERA = "thermal_camera"


class SensorQuality(IntEnum):
    """传感器质量评估"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1
    INVALID = 0


class ObjectType(Enum):
    """物体类型"""
    UNKNOWN = "unknown"
    PERSON = "person"
    VEHICLE = "vehicle"
    BUILDING = "building"
    TREE = "tree"
    OBSTACLE = "obstacle"
    LANDING_ZONE = "landing_zone"
    TARGET = "target"
    WATER = "water"
    ROAD = "road"


class TerrainType(Enum):
    """地形类型"""
    UNKNOWN = "unknown"
    FLAT = "flat"
    SLOPE = "slope"
    ROUGH = "rough"
    WATER = "water"
    URBAN = "urban"
    FOREST = "forest"
