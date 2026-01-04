"""
感知模块 - Perception Module

统一的传感器管理、数据融合、目标检测、地图构建等功能。

Architecture:
    data/         - 数据模型和类型
    sensors/       - 传感器管理和ROS2集成
    detection/     - 目标检测和跟踪
    understanding/  - VLM场景理解
    fusion/        - 多传感器数据融合
    processing/    - 数据预处理和处理
    infrastructure/- 基础设施（事件总线、异步处理等）
    mapping/       - 地图构建（占据栅格、语义地图）
    utils/         - 工具函数（坐标转换、验证等）

ROS_DOMAIN_ID Support:
    - 自动从环境变量读取 ROS_DOMAIN_ID
    - 支持通过配置文件设置
    - 默认值为 0

Author: Brain Development Team
Date: 2025-01-04
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
from .sensors.manager import MultiSensorManager, SensorManager
from .sensors.ros2_manager import ROS2SensorManager

# 传感器特定导入
try:
    from .sensors.camera import CameraSensor
    from .sensors.lidar import LidarSensor
    from .sensors.imu import IMUSensor
    from .sensors.gps import GPSSensor
except ImportError as e:
    CameraSensor = None
    LidarSensor = None
    IMUSensor = None
    GPSSensor = None

# 传感器模型
from .sensors.models.camera_model import CameraModel
from .sensors.models.lidar_model import LidarModel
from .sensors.models.imu_model import IMUModel
from .sensors.models.gps_model import GPSModel

from .sensors.fusion import SensorFusion

# 向后兼容的别名
create_sensor = BaseSensor.create

# ==============================================================================
# 目标检测
# ==============================================================================
try:
    from .detection.detector import ObjectDetector, DetectionMode
    from .detection.tracker import ObjectTracker, TrackedObject
except ImportError as e:
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
except ImportError as e:
    VLMPerception = None
    VLMService = None

# ==============================================================================
# 地图构建
# ==============================================================================
from .mapping.occupancy_mapper import OccupancyMapper, OccupancyGrid

# ==============================================================================
# 数据融合
# ==============================================================================
from .sensors.sensor_fusion import EKFPoseFusion
from .sensors.sensor_fusion import DepthRGBFusion
from .sensors.sensor_fusion import ObstacleDetector

# 别名以避免命名冲突
PoseFusion = EKFPoseFusion
DepthRGBFusion = DepthRGBFusion

# ==============================================================================
# 数据处理
# ==============================================================================
try:
    from .processing.pointcloud_processor import LidarProcessor
except ImportError as e:
    LidarProcessor = None

# ==============================================================================
# 基础设施
# ==============================================================================
from .infrastructure.event_bus import PerceptionEventBus
from .infrastructure.async_processor import AsyncProcessor
from .infrastructure.circuit_breaker import CircuitBreaker
from .infrastructure.performance_monitor import PerformanceMonitor
from .infrastructure.converter import DataConverter
from .infrastructure.exceptions import (
    PerceptionError,
    SensorError,
    FusionError,
    DetectionError,
)

# ==============================================================================
# 工具
# ==============================================================================
from .utils.coordinates import quaternion_to_euler, transform_local_to_world
from .utils.math import compute_laser_angles, angle_to_direction
from .utils.validation import validate_sensor_data, validate_pose

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
    "CameraSensor",
    "LidarSensor",
    "IMUSensor",
    "GPSSensor",
    "MultiSensorManager",
    "SensorManager",
    "ROS2SensorManager",
    "create_sensor",
    "SensorFusion",
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
    
    # 融合
    "EKFPoseFusion",
    "PoseFusion",
    "DepthRGBFusion",
    "ObstacleDetector",
    
    # 处理
    "LidarProcessor",
    
    # 基础设施
    "PerceptionEventBus",
    "AsyncProcessor",
    "CircuitBreaker",
    "PerformanceMonitor",
    "DataConverter",
    "PerceptionError",
    "SensorError",
    "FusionError",
    "DetectionError",
    
    # 工具
    "quaternion_to_euler",
    "transform_local_to_world",
    "compute_laser_angles",
    "angle_to_direction",
    "validate_sensor_data",
    "validate_pose",
]
