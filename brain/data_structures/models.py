"""
Core Data Models for Brain System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


@dataclass
class SensorData:
    """Generic sensor data container"""
    timestamp: float
    sensor_type: str
    sensor_id: str
    data: Dict[str, Any]
    quality: float = 1.0

    # For specific sensor types
    points: Optional[List] = None  # For point cloud
    image: Optional[np.ndarray] = None  # For camera
    orientation: Optional[Dict[str, float]] = None  # For IMU
    position: Optional[Dict[str, float]] = None  # For GPS


@dataclass
class OccupancyGrid:
    """Occupancy grid representation"""
    data: np.ndarray
    resolution: float
    origin: tuple  # (x, y, yaw)
    width: int
    height: int


@dataclass
class WorldObject:
    """Object in the world model"""
    id: str
    type: str
    position: tuple  # (x, y, z)
    orientation: tuple  # (roll, pitch, yaw)
    size: tuple  # (width, height, depth)
    velocity: tuple = (0.0, 0.0, 0.0)
    confidence: float = 1.0
    timestamp: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicObject:
    """Dynamic object with trajectory prediction"""
    id: str
    type: str
    position: tuple  # (x, y, z)
    velocity: tuple  # (vx, vy, vz)
    acceleration: tuple = (0.0, 0.0, 0.0)
    trajectory: List[tuple] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: float = 0.0
    prediction_horizon: float = 5.0  # seconds
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldModel:
    """World model representation"""
    timestamp: float
    objects: List[WorldObject] = field(default_factory=list)
    grid: Optional[OccupancyGrid] = None
    robot_position: tuple = (0.0, 0.0, 0.0)
    robot_orientation: tuple = (0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of sensor data fusion"""
    timestamp: float
    fused_objects: List[WorldObject]
    confidence_scores: List[float]
    source_sensors: List[str]
    fusion_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ROS2 message stubs for when ROS2 is not available
@dataclass
class ROS2Header:
    """ROS2 Header message stub"""
    stamp: int
    frame_id: str


@dataclass
class ROS2PointField:
    """ROS2 PointField message stub"""
    name: str
    offset: int
    datatype: int
    count: int


@dataclass
class ROS2PointCloud2:
    """ROS2 PointCloud2 message stub"""
    header: ROS2Header
    height: int
    width: int
    fields: List[ROS2PointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool


@dataclass
class ROS2Image:
    """ROS2 Image message stub"""
    header: ROS2Header
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: bytes


@dataclass
class ROS2LaserScan:
    """ROS2 LaserScan message stub"""
    header: ROS2Header
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float
    ranges: List[float]
    intensities: List[float]


@dataclass
class ROS2Twist:
    """ROS2 Twist message stub"""
    linear: Dict[str, float]
    angular: Dict[str, float]


@dataclass
class ROS2PoseStamped:
    """ROS2 PoseStamped message stub"""
    header: ROS2Header
    pose: Dict[str, Any]


@dataclass
class ROS2Odometry:
    """ROS2 Odometry message stub"""
    header: ROS2Header
    child_frame_id: str
    pose: Dict[str, Any]
    twist: ROS2Twist


@dataclass
class ROS2OccupancyGrid:
    """ROS2 OccupancyGrid message stub"""
    header: ROS2Header
    info: Dict[str, Any]
    data: List[int]


@dataclass
class ROS2NavSatFix:
    """ROS2 NavSatFix message stub"""
    header: ROS2Header
    status: Dict[str, int]
    latitude: float
    longitude: float
    altitude: float
    position_covariance: List[float]
    position_covariance_type: int


@dataclass
class ROS2Imu:
    """ROS2 IMU message stub"""
    header: ROS2Header
    orientation: Dict[str, float]
    orientation_covariance: List[float]
    angular_velocity: Dict[str, float]
    angular_velocity_covariance: List[float]
    linear_acceleration: Dict[str, float]
    linear_acceleration_covariance: List[float]