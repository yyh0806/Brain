"""
感知层枚举类型定义
"""

from enum import Enum, IntEnum


class CellState(IntEnum):
    """栅格状态"""
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100


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


class DetectionSource(Enum):
    """检测来源"""
    VLM = "vlm"
    YOLO = "yolo"
    FUSED = "fused"


class DetectionMode(Enum):
    """检测模式"""
    FAST = "fast"
    ACCURATE = "accurate"
    TRACKING = "tracking"


class PerceptionEventType(Enum):
    """感知事件类型"""
    SENSOR_DATA = "sensor_data"
    FUSION_COMPLETE = "fusion_complete"
    OBJECT_DETECTED = "object_detected"
    OBSTACLE_DETECTED = "obstacle_detected"
    MAP_UPDATED = "map_updated"
    VLM_ANALYSIS = "vlm_analysis"
    SENSOR_ERROR = "sensor_error"
    FUSION_ERROR = "fusion_error"


class SensorQuality(IntEnum):
    """传感器质量评估等级"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1
    INVALID = 0
