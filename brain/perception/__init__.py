"""
感知模块 - Perception Module

统一的传感器管理、数据融合、目标检测、地图构建等功能。

Architecture:
    data/         - 数据模型和类型
    sensors/       - 传感器管理和ROS2集成
    detection/     - 目标检测和跟踪
    understanding/ - VLM场景理解
    mapping/       - 地图构建（占据栅格、语义地图）
    utils/         - 工具函数（坐标转换、验证等）

ROS_DOMAIN_ID Support:
    - 自动从环境变量读取 ROS_DOMAIN_ID
    - 支持通过配置文件设置
    - 默认值为 0

Author: Brain Development Team
Date: 2025-01-06
"""

# ==============================================================================
# 数据模型
# ==============================================================================
from .data.types import SensorType, SensorQuality, ObjectType, TerrainType
from .data.models import Pose2D, Pose3D, Position3D, Velocity, BoundingBox, Detection

# ==============================================================================
# 传感器管理
# ==============================================================================
from .sensors.base import BaseSensor
from .sensors.manager import MultiSensorManager
from .sensors.ros2_sensor_manager import ROS2SensorManager, PerceptionData

# 传感器融合
from .sensors.fusion import FusedPose, EKFPoseFusion, DepthRGBFusion, ObstacleDetector

# 传感器模型
from .sensors.models.camera_model import CameraModel
from .sensors.models.lidar_model import LidarModel
from .sensors.models.imu_model import IMUModel
from .sensors.models.gps_model import GPSModel

# ==============================================================================
# 目标检测
# ==============================================================================
try:
    from .detection.detector import ObjectDetector, DetectionMode
    from .detection.tracker import ObjectTracker, TrackedObject
except ImportError:
    ObjectDetector = None
    DetectionMode = None
    ObjectTracker = None
    TrackedObject = None

# ==============================================================================
# 场景理解 (VLM)
# ==============================================================================
try:
    from .understanding.vlm_perception import VLMPerception
    from .understanding.vlm_service import VLMService
except ImportError:
    VLMPerception = None
    VLMService = None

# ==============================================================================
# 地图构建
# ==============================================================================
try:
    from .mapping.occupancy_mapper import OccupancyMapper, OccupancyGrid
except ImportError:
    OccupancyMapper = None
    OccupancyGrid = None

# ==============================================================================
# 工具
# ==============================================================================
from .utils.coordinates import quaternion_to_euler, transform_local_to_world
from .utils.math_utils import compute_laser_angles, angle_to_direction
try:
    from .utils.validation import validate_sensor_data, validate_pose
except ImportError:
    validate_sensor_data = None
    validate_pose = None

# ==============================================================================
# 统一导出接口
# ==============================================================================

__version__ = "2.0.0"
__author__ = "Brain Development Team"

__all__ = [
    # 数据模型
    "Pose2D",
    "Pose3D",
    "Position3D",
    "Velocity",
    "BoundingBox",
    "Detection",
    "SensorType",
    "SensorQuality",
    "ObjectType",
    "TerrainType",

    # 传感器
    "BaseSensor",
    "MultiSensorManager",
    "ROS2SensorManager",
    "PerceptionData",
    "create_sensor",

    # 融合
    "FusedPose",
    "EKFPoseFusion",
    "DepthRGBFusion",
    "ObstacleDetector",

    # 传感器模型
    "CameraModel",
    "LidarModel",
    "IMUModel",
    "GPSModel",

    # 目标检测
    "ObjectDetector",
    "DetectionMode",
    "ObjectTracker",
    "TrackedObject",

    # 场景理解
    "VLMPerception",
    "VLMService",

    # 地图
    "OccupancyMapper",
    "OccupancyGrid",

    # 工具
    "quaternion_to_euler",
    "transform_local_to_world",
    "compute_laser_angles",
    "angle_to_direction",
    "validate_sensor_data",
    "validate_pose",
]
