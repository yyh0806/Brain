# -*- coding: utf-8 -*-
"""
Sensor Input Data Types

This module defines the core data structures for sensor input processing
in the Brain cognitive world model system.

Author: Brain Development Team
Date: 2025-12-17
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Union, Dict, Any, List, Tuple
import numpy as np
import time
import threading
from collections import deque


class SensorType(Enum):
    """Enumeration of supported sensor types."""
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
    """Sensor quality assessment levels."""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    VERY_POOR = 1
    INVALID = 0


@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters for image processing and projection.

    This class encapsulates all necessary intrinsic parameters for
    camera projection and distortion correction.
    """
    # Camera matrix K
    fx: float  # Focal length in x direction (pixels)
    fy: float  # Focal length in y direction (pixels)
    cx: float  # Principal point x coordinate (pixels)
    cy: float  # Principal point y coordinate (pixels)

    # Distortion coefficients
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    k3: float = 0.0  # Radial distortion coefficient 3
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2

    # Image dimensions
    width: int = 0
    height: int = 0

    # Distortion model
    distortion_model: str = "plumb_bob"

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")
        if self.width < 0 or self.height < 0:
            raise ValueError("Image dimensions must be non-negative")

    @property
    def camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

    @property
    def distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients array."""
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


@dataclass
class PointCloudData:
    """
    Point cloud data structure for 3D spatial information.

    This class represents point cloud data from LiDAR, radar, or other
    3D sensors with optional intensity and color information.
    """
    points: np.ndarray  # Nx3 array [x, y, z]
    intensity: Optional[np.ndarray] = None  # Nx1 intensity values
    rgb: Optional[np.ndarray] = None  # Nx3 RGB values
    timestamp: float = field(default_factory=time.time)
    frame_id: str = "base_link"
    sensor_pose: Optional[np.ndarray] = None  # 4x4 transformation matrix

    # Point cloud metadata
    point_count: int = field(init=False)
    has_intensity: bool = field(init=False)
    has_color: bool = field(init=False)

    def __post_init__(self):
        """Initialize derived fields and validate data."""
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")

        self.point_count = self.points.shape[0]
        self.has_intensity = self.intensity is not None
        self.has_color = self.rgb is not None

        # Validate optional arrays
        if self.intensity is not None:
            if len(self.intensity) != self.point_count:
                raise ValueError("Intensity array length must match point count")

        if self.rgb is not None:
            if self.rgb.ndim != 2 or self.rgb.shape[1] != 3:
                raise ValueError("RGB must be Nx3 array")
            if self.rgb.shape[0] != self.point_count:
                raise ValueError("RGB array length must match point count")

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D bounding box of point cloud."""
        return np.min(self.points, axis=0), np.max(self.points, axis=0)

    def transform(self, transformation: np.ndarray) -> 'PointCloudData':
        """Apply 4x4 transformation matrix to point cloud."""
        if transformation.shape != (4, 4):
            raise ValueError("Transformation must be 4x4 matrix")

        # Convert points to homogeneous coordinates
        homogeneous_points = np.hstack([
            self.points,
            np.ones((self.point_count, 1))
        ])

        # Apply transformation
        transformed = (transformation @ homogeneous_points.T).T
        transformed_points = transformed[:, :3]

        return PointCloudData(
            points=transformed_points,
            intensity=self.intensity,
            rgb=self.rgb,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            sensor_pose=transformation @ self.sensor_pose if self.sensor_pose is not None else None
        )


@dataclass
class ImageData:
    """
    Image data structure for visual sensor information.

    This class represents image data from cameras with optional depth
    information and camera parameters.
    """
    image: np.ndarray  # HxWxC image array
    depth: Optional[np.ndarray] = None  # HxW depth map
    camera_params: Optional[CameraIntrinsics] = None
    timestamp: float = field(default_factory=time.time)
    frame_id: str = "camera_link"

    # Image metadata
    height: int = field(init=False)
    width: int = field(init=False)
    channels: int = field(init=False)
    has_depth: bool = field(init=False)
    encoding: str = "bgr8"  # ROS2 image encoding

    def __post_init__(self):
        """Initialize derived fields and validate data."""
        if self.image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")

        self.height, self.width = self.image.shape[:2]
        self.channels = 1 if self.image.ndim == 2 else self.image.shape[2]
        self.has_depth = self.depth is not None

        # Validate depth map
        if self.depth is not None:
            if self.depth.shape != (self.height, self.width):
                raise ValueError("Depth map dimensions must match image")

    def get_region_of_interest(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Extract region of interest from image."""
        return self.image[y:y+h, x:x+w]

    def resize(self, new_width: int, new_height: int) -> 'ImageData':
        """Resize image to new dimensions."""
        import cv2
        resized_image = cv2.resize(self.image, (new_width, new_height))

        resized_depth = None
        if self.depth is not None:
            resized_depth = cv2.resize(self.depth, (new_width, new_height))

        return ImageData(
            image=resized_image,
            depth=resized_depth,
            camera_params=self.camera_params,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            encoding=self.encoding
        )


@dataclass
class IMUData:
    """
    IMU (Inertial Measurement Unit) data structure.

    This class represents IMU sensor data including linear acceleration,
    angular velocity, and orientation information.
    """
    linear_acceleration: np.ndarray  # [ax, ay, az] in m/s²
    angular_velocity: np.ndarray     # [wx, wy, wz] in rad/s
    orientation: Optional[np.ndarray] = None  # Quaternion [x, y, z, w]

    # Covariance matrices
    linear_acceleration_covariance: Optional[np.ndarray] = None  # 3x3
    angular_velocity_covariance: Optional[np.ndarray] = None     # 3x3
    orientation_covariance: Optional[np.ndarray] = None          # 3x3

    timestamp: float = field(default_factory=time.time)
    frame_id: str = "imu_link"

    def __post_init__(self):
        """Validate IMU data arrays."""
        if self.linear_acceleration.shape != (3,):
            raise ValueError("Linear acceleration must be 3-element vector")
        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be 3-element vector")
        if self.orientation is not None and self.orientation.shape != (4,):
            raise ValueError("Orientation must be 4-element quaternion")

    def get_magnitude(self) -> Tuple[float, float]:
        """Get magnitude of acceleration and angular velocity."""
        accel_mag = np.linalg.norm(self.linear_acceleration)
        gyro_mag = np.linalg.norm(self.angular_velocity)
        return accel_mag, gyro_mag


@dataclass
class GPSData:
    """
    GPS (Global Positioning System) data structure.

    This class represents GPS sensor data including position, velocity,
    and accuracy information.
    """
    latitude: float   # Latitude in degrees
    longitude: float  # Longitude in degrees
    altitude: float   # Altitude in meters

    # Position accuracy
    position_covariance: Optional[np.ndarray] = None  # 3x3 covariance matrix

    # Velocity information (ENU frame)
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz] in m/s
    velocity_covariance: Optional[np.ndarray] = None  # 3x3 covariance matrix

    # GPS status
    fix_type: int = 0  # 0=No fix, 1=2D fix, 2=3D fix, 3=RTK fix
    satellites_used: int = 0
    hdop: float = 0.0  # Horizontal dilution of precision
    vdop: float = 0.0  # Vertical dilution of precision

    timestamp: float = field(default_factory=time.time)
    frame_id: str = "gps_link"

    def __post_init__(self):
        """Validate GPS data ranges."""
        if not -90 <= self.latitude <= 90:
            raise ValueError("Latitude must be in range [-90, 90]")
        if not -180 <= self.longitude <= 180:
            raise ValueError("Longitude must be in range [-180, 180]")

        if self.velocity is not None and self.velocity.shape != (3,):
            raise ValueError("Velocity must be 3-element vector")

    def to_utm(self) -> Tuple[float, float]:
        """Convert latitude/longitude to UTM coordinates."""
        # Simplified UTM conversion - in production use proper library
        import math

        zone = int((self.longitude + 180) / 6) + 1
        k0 = 0.9996

        # Simplified calculation - use proper UTM library in production
        x = self.longitude * 111320 * math.cos(math.radians(self.latitude))
        y = self.latitude * 110574

        return x, y


@dataclass
class WeatherData:
    """
    Weather data structure for environmental sensor information.

    This class represents weather and environmental conditions from
    meteorological sensors.
    """
    temperature: float          # Temperature in Celsius
    humidity: float            # Relative humidity in percentage (0-100)
    pressure: float            # Atmospheric pressure in hPa
    wind_speed: float          # Wind speed in m/s
    wind_direction: float      # Wind direction in degrees (0-360)

    # Optional weather parameters
    visibility: Optional[float] = None      # Visibility in meters
    precipitation: Optional[float] = None   # Precipitation rate in mm/h
    uv_index: Optional[float] = None        # UV index
    air_quality_index: Optional[int] = None # AQI value

    timestamp: float = field(default_factory=time.time)
    frame_id: str = "weather_station"

    def __post_init__(self):
        """Validate weather data ranges."""
        if not -50 <= self.temperature <= 60:
            raise ValueError("Temperature out of reasonable range")
        if not 0 <= self.humidity <= 100:
            raise ValueError("Humidity must be in range [0, 100]")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
        if self.wind_speed < 0:
            raise ValueError("Wind speed cannot be negative")
        if not 0 <= self.wind_direction < 360:
            raise ValueError("Wind direction must be in range [0, 360)")

    def get_heat_index(self) -> float:
        """Calculate heat index from temperature and humidity."""
        # Simplified heat index calculation
        if self.temperature < 27:
            return self.temperature

        # More accurate calculation would use the full NWS formula
        hi = self.temperature + 0.5 * (self.humidity / 100.0)
        return hi


@dataclass
class SensorDataPacket:
    """
    Universal sensor data packet for all sensor types.

    This class provides a standardized interface for all sensor data
    types in the Brain system, enabling unified processing and fusion.
    """
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]
    quality_score: float = 1.0
    calibration_params: Dict[str, Any] = field(default_factory=dict)

    # Packet metadata
    sequence_number: int = 0
    frame_id: str = "base_link"
    processing_time: float = 0.0

    # Thread-safe access
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        """Validate packet data."""
        if not 0 <= self.quality_score <= 1:
            raise ValueError("Quality score must be in range [0, 1]")

        # Extract frame_id from embedded data if not provided
        if hasattr(self.data, 'frame_id') and self.frame_id == "base_link":
            self.frame_id = self.data.frame_id

    def get_timestamp_delta(self, reference_time: Optional[float] = None) -> float:
        """Get time delta from reference time or current time."""
        if reference_time is None:
            reference_time = time.time()
        return reference_time - self.timestamp

    def is_stale(self, max_age_seconds: float = 1.0) -> bool:
        """Check if data packet is stale based on age."""
        return self.get_timestamp_delta() > max_age_seconds

    def validate_data_integrity(self) -> bool:
        """Validate the integrity of the sensor data."""
        try:
            # Basic validation based on sensor type
            if isinstance(self.data, PointCloudData):
                return self.data.points.size > 0
            elif isinstance(self.data, ImageData):
                return self.data.image.size > 0
            elif isinstance(self.data, IMUData):
                return not np.any(np.isnan(self.data.linear_acceleration))
            elif isinstance(self.data, GPSData):
                return -90 <= self.data.latitude <= 90
            elif isinstance(self.data, WeatherData):
                return 0 <= self.data.humidity <= 100

            return True
        except Exception:
            return False

    def with_lock(self):
        """Context manager for thread-safe access."""
        return self._lock


# Type aliases for better readability
SensorDataUnion = Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]
QualityAssessment = Dict[str, Union[float, SensorQuality, str]]


@dataclass
class SensorMetadata:
    """Metadata for sensor configuration and capabilities."""
    sensor_id: str
    sensor_type: SensorType
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""

    # Physical specifications
    fov_h: Optional[float] = None  # Horizontal field of view in degrees
    fov_v: Optional[float] = None  # Vertical field of view in degrees
    range_min: Optional[float] = None  # Minimum sensing range
    range_max: Optional[float] = None  # Maximum sensing range

    # Performance specifications
    update_rate: float = 0.0  # Update rate in Hz
    accuracy: float = 0.0    # Accuracy specification
    resolution: str = ""     # Resolution string

    # Calibration information
    last_calibration: Optional[float] = None  # Unix timestamp
    calibration_validity: float = 86400.0     # Calibration validity in seconds

    def needs_calibration(self) -> bool:
        """Check if sensor needs recalibration."""
        if self.last_calibration is None:
            return True

        current_time = time.time()
        return (current_time - self.last_calibration) > self.calibration_validity


# Utility functions for sensor data processing
def create_timestamp() -> float:
    """Create a high-precision timestamp."""
    return time.time()


def validate_sensor_data_quality(data: SensorDataUnion) -> QualityAssessment:
    """Assess quality of sensor data and return assessment."""
    assessment = {
        "score": 1.0,
        "quality": SensorQuality.EXCELLENT,
        "issues": []
    }

    try:
        if isinstance(data, PointCloudData):
            if data.point_count == 0:
                assessment["score"] = 0.0
                assessment["quality"] = SensorQuality.INVALID
                assessment["issues"].append("Empty point cloud")
            elif data.point_count < 100:
                assessment["score"] = 0.5
                assessment["quality"] = SensorQuality.POOR
                assessment["issues"].append("Low point density")

        elif isinstance(data, ImageData):
            if data.image.size == 0:
                assessment["score"] = 0.0
                assessment["quality"] = SensorQuality.INVALID
                assessment["issues"].append("Empty image")
            else:
                # Check for over/under exposure
                mean_intensity = np.mean(data.image)
                if mean_intensity < 10 or mean_intensity > 245:
                    assessment["score"] = 0.7
                    assessment["quality"] = SensorQuality.FAIR
                    assessment["issues"].append("Poor exposure")

        elif isinstance(data, IMUData):
            # Check for unreasonable values
            accel_magnitude = np.linalg.norm(data.linear_acceleration)
            if accel_magnitude > 50:  # 5g threshold
                assessment["score"] = 0.5
                assessment["quality"] = SensorQuality.POOR
                assessment["issues"].append("Unreasonable acceleration")

    except Exception as e:
        assessment["score"] = 0.0
        assessment["quality"] = SensorQuality.INVALID
        assessment["issues"].append(f"Validation error: {str(e)}")

    return assessment


# Sensor data buffering for temporal processing
class SensorDataBuffer:
    """Thread-safe buffer for sensor data with temporal ordering."""

    def __init__(self, max_size: int = 100, max_age_seconds: float = 5.0):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, packet: SensorDataPacket) -> None:
        """Add a data packet to the buffer."""
        with self._lock:
            self._buffer.append(packet)

    def get_latest(self, count: int = 1) -> List[SensorDataPacket]:
        """Get the latest N packets from the buffer."""
        with self._lock:
            if count == 1:
                return [self._buffer[-1]] if self._buffer else []
            return list(self._buffer)[-count:]

    def get_by_timerange(self, start_time: float, end_time: float) -> List[SensorDataPacket]:
        """Get packets within a specific time range."""
        with self._lock:
            return [
                packet for packet in self._buffer
                if start_time <= packet.timestamp <= end_time
            ]

    def cleanup_old_data(self) -> None:
        """Remove old data from the buffer."""
        current_time = time.time()
        cutoff_time = current_time - self.max_age_seconds

        with self._lock:
            while self._buffer and self._buffer[0].timestamp < cutoff_time:
                self._buffer.popleft()

    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)