# -*- coding: utf-8 -*-
"""
Data Format Converter Module

This module provides comprehensive data format conversion capabilities for
different sensor data formats, including ROS2 message conversion and
standard format output for the Brain cognitive world model system.

Author: Brain Development Team
Date: 2025-12-17
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import json
import base64
import struct
import threading
import time
import logging

from .sensor_input_types import (
    SensorDataPacket,
    SensorType,
    PointCloudData,
    ImageData,
    IMUData,
    GPSData,
    WeatherData,
    CameraIntrinsics,
)

from .sensor_manager import SynchronizedDataPacket

# Configure logging
logger = logging.getLogger(__name__)


class DataFormat(Enum):
    """Supported data formats."""
    BRAIN_NATIVE = "brain_native"
    ROS2 = "ros2"
    JSON = "json"
    BINARY = "binary"
    PROTOBUF = "protobuf"
    CSV = "csv"
    PCD = "pcd"  # Point Cloud Data format
    PNM = "pnm"  # Portable image formats


@dataclass
class ConversionResult:
    """Result of data format conversion."""
    success: bool
    data: Optional[Any] = None
    format: DataFormat = DataFormat.BRAIN_NATIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    conversion_time: float = 0.0

    def is_valid(self) -> bool:
        """Check if conversion was successful."""
        return self.success and self.data is not None


@dataclass
class ConversionOptions:
    """Options for data format conversion."""
    target_format: DataFormat
    compression: bool = False
    include_metadata: bool = True
    validate_output: bool = True
    precision: str = "single"  # single, double, or integer
    coordinate_frame: str = "base_link"
    time_format: str = "unix"  # unix, ros, or iso

    # Image-specific options
    image_encoding: str = "bgr8"
    image_quality: int = 90  # For JPEG compression

    # Point cloud-specific options
    point_cloud_format: str = "xyz"  # xyz, xyzrgb, xyzi
    downsample: bool = False
    voxel_size: float = 0.05

    # Validation options
    strict_validation: bool = True
    max_size_mb: int = 100


class DataConverter(ABC):
    """
    Abstract base class for data format converters.

    This class defines the interface for converting between different
    data formats in the Brain system.
    """

    def __init__(self):
        """Initialize data converter."""
        self._conversion_stats = {
            "total_conversions": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "average_conversion_time": 0.0,
        }
        self._stats_lock = threading.Lock()

    @abstractmethod
    def convert(self, data: Any, options: ConversionOptions) -> ConversionResult:
        """
        Convert data to target format.

        Args:
            data: Input data to convert
            options: Conversion options

        Returns:
            Conversion result with converted data
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[DataFormat]:
        """Get list of supported input/output formats."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        with self._stats_lock:
            return self._conversion_stats.copy()

    def _update_stats(self, success: bool, conversion_time: float) -> None:
        """Update conversion statistics."""
        with self._stats_lock:
            self._conversion_stats["total_conversions"] += 1
            if success:
                self._conversion_stats["successful_conversions"] += 1
            else:
                self._conversion_stats["failed_conversions"] += 1

            # Update average conversion time
            total = self._conversion_stats["total_conversions"]
            current_avg = self._conversion_stats["average_conversion_time"]
            self._conversion_stats["average_conversion_time"] = (
                (current_avg * (total - 1) + conversion_time) / total
            )


class ROS2Converter(DataConverter):
    """
    ROS2 message format converter.

    This class converts between Brain native formats and ROS2 message types
    for integration with ROS2-based systems.
    """

    def __init__(self):
        """Initialize ROS2 converter."""
        super().__init__()
        self.ros2_available = self._check_ros2_availability()

    def _check_ros2_availability(self) -> bool:
        """Check if ROS2 is available."""
        try:
            import rclpy
            from sensor_msgs.msg import PointCloud2, Image, Imu, NavSatFix
            from geometry_msgs.msg import Quaternion, Vector3
            from std_msgs.msg import Header
            return True
        except ImportError:
            logger.warning("ROS2 not available, ROS2 conversion will be disabled")
            return False

    def convert(self, data: Any, options: ConversionOptions) -> ConversionResult:
        """
        Convert to/from ROS2 format.

        Args:
            data: Input data (Brain native or ROS2 message)
            options: Conversion options

        Returns:
            Conversion result
        """
        start_time = time.time()
        success = False
        result_data = None
        error_message = None

        try:
            if not self.ros2_available:
                raise ImportError("ROS2 not available")

            if isinstance(data, SensorDataPacket):
                # Convert Brain to ROS2
                result_data = self._brain_to_ros2(data, options)
            else:
                # Convert ROS2 to Brain
                result_data = self._ros2_to_brain(data, options)

            success = True

        except Exception as e:
            error_message = str(e)
            logger.error(f"ROS2 conversion failed: {e}")

        conversion_time = time.time() - start_time
        self._update_stats(success, conversion_time)

        return ConversionResult(
            success=success,
            data=result_data,
            format=DataFormat.ROS2,
            error_message=error_message,
            conversion_time=conversion_time
        )

    def _brain_to_ros2(self, packet: SensorDataPacket, options: ConversionOptions) -> Any:
        """Convert Brain native packet to ROS2 message."""
        from std_msgs.msg import Header

        # Create header
        header = Header()
        header.stamp = self._time_to_ros_time(packet.timestamp)
        header.frame_id = options.coordinate_frame

        # Convert based on sensor type
        if packet.sensor_type == SensorType.POINT_CLOUD:
            return self._pointcloud_to_ros2(packet.data, header)
        elif packet.sensor_type == SensorType.IMAGE:
            return self._image_to_ros2(packet.data, header, options)
        elif packet.sensor_type == SensorType.IMU:
            return self._imu_to_ros2(packet.data, header)
        elif packet.sensor_type == SensorType.GPS:
            return self._gps_to_ros2(packet.data, header)
        else:
            raise ValueError(f"Unsupported sensor type for ROS2 conversion: {packet.sensor_type}")

    def _ros2_to_brain(self, ros2_msg: Any, options: ConversionOptions) -> SensorDataPacket:
        """Convert ROS2 message to Brain native packet."""
        from std_msgs.msg import Header

        # Extract header information
        header = getattr(ros2_msg, 'header', None)
        timestamp = self._ros_time_to_time(header.stamp) if header else time.time()
        frame_id = header.frame_id if header else options.coordinate_frame

        # Convert based on message type
        if hasattr(ros2_msg, 'fields'):  # Likely PointCloud2
            data = self._ros2_to_pointcloud(ros2_msg)
            sensor_type = SensorType.POINT_CLOUD
        elif hasattr(ros2_msg, 'height') and hasattr(ros2_msg, 'width'):  # Likely Image
            data = self._ros2_to_image(ros2_msg)
            sensor_type = SensorType.IMAGE
        elif hasattr(ros2_msg, 'linear_acceleration'):  # Likely IMU
            data = self._ros2_to_imu(ros2_msg)
            sensor_type = SensorType.IMU
        elif hasattr(ros2_msg, 'latitude'):  # Likely GPS
            data = self._ros2_to_gps(ros2_msg)
            sensor_type = SensorType.GPS
        else:
            raise ValueError(f"Unsupported ROS2 message type: {type(ros2_msg)}")

        return SensorDataPacket(
            sensor_id="ros2_sensor",
            sensor_type=sensor_type,
            timestamp=timestamp,
            data=data,
            frame_id=frame_id
        )

    def _pointcloud_to_ros2(self, pc_data: PointCloudData, header) -> Any:
        """Convert point cloud to ROS2 PointCloud2 message."""
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2

        # Create point fields
        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=7, count=1),  # FLOAT32
            point_cloud2.PointField(name='y', offset=4, datatype=7, count=1),  # FLOAT32
            point_cloud2.PointField(name='z', offset=8, datatype=7, count=1),  # FLOAT32
        ]

        # Add intensity field if available
        if pc_data.intensity is not None:
            fields.append(point_cloud2.PointField(name='intensity', offset=12, datatype=7, count=1))

        # Add RGB field if available
        if pc_data.rgb is not None:
            fields.append(point_cloud2.PointField(name='rgb', offset=16, datatype=7, count=1))

        # Create point cloud message
        pc2_msg = point_cloud2.create_cloud(header, fields, pc_data.points)

        # Add additional data
        if pc_data.intensity is not None:
            pc2_msg = point_cloud2.create_cloud(header, fields,
                                               np.column_stack([pc_data.points, pc_data.intensity]))

        return pc2_msg

    def _image_to_ros2(self, img_data: ImageData, header, options: ConversionOptions) -> Any:
        """Convert image to ROS2 Image message."""
        from sensor_msgs.msg import Image

        image_msg = Image()
        image_msg.header = header
        image_msg.height = img_data.height
        image_msg.width = img_data.width
        image_msg.encoding = options.image_encoding
        image_msg.is_bigendian = False
        image_msg.step = img_data.width * img_data.channels

        # Convert image data to bytes
        if img_data.image.dtype == np.uint8:
            image_msg.data = img_data.image.tobytes()
        else:
            # Convert to uint8 if needed
            image_msg.data = (img_data.image * 255).astype(np.uint8).tobytes()

        return image_msg

    def _imu_to_ros2(self, imu_data: IMUData, header) -> Any:
        """Convert IMU data to ROS2 Imu message."""
        from sensor_msgs.msg import Imu
        from geometry_msgs.msg import Quaternion, Vector3

        imu_msg = Imu()
        imu_msg.header = header

        # Linear acceleration
        imu_msg.linear_acceleration = Vector3(
            x=imu_data.linear_acceleration[0],
            y=imu_data.linear_acceleration[1],
            z=imu_data.linear_acceleration[2]
        )

        # Angular velocity
        imu_msg.angular_velocity = Vector3(
            x=imu_data.angular_velocity[0],
            y=imu_data.angular_velocity[1],
            z=imu_data.angular_velocity[2]
        )

        # Orientation
        if imu_data.orientation is not None:
            imu_msg.orientation = Quaternion(
                x=imu_data.orientation[0],
                y=imu_data.orientation[1],
                z=imu_data.orientation[2],
                w=imu_data.orientation[3]
            )

        return imu_msg

    def _gps_to_ros2(self, gps_data: GPSData, header) -> Any:
        """Convert GPS data to ROS2 NavSatFix message."""
        from sensor_msgs.msg import NavSatFix
        from geographic_msgs.msg import GeoPoint

        fix_msg = NavSatFix()
        fix_msg.header = header
        fix_msg.latitude = gps_data.latitude
        fix_msg.longitude = gps_data.longitude
        fix_msg.altitude = gps_data.altitude

        # Set status
        fix_msg.status.status = gps_data.fix_type - 1  # Convert to ROS2 convention
        fix_msg.status.service = NavSatFix.SERVICE_GPS

        # Set covariance
        if gps_data.position_covariance is not None:
            fix_msg.position_covariance = gps_data.position_covariance.flatten().tolist()
            fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN

        return fix_msg

    def _ros2_to_pointcloud(self, pc2_msg) -> PointCloudData:
        """Convert ROS2 PointCloud2 to PointCloudData."""
        from sensor_msgs_py import point_cloud2

        # Convert to numpy array
        pc_array = point_cloud2.read_points(pc2_msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_array))

        # Extract intensity if available
        intensity = None
        if 'intensity' in pc2_msg.fields:
            intensity_array = point_cloud2.read_points(pc2_msg, field_names=("intensity",), skip_nans=True)
            intensity = np.array([p[0] for p in intensity_array])

        return PointCloudData(
            points=points,
            intensity=intensity,
            timestamp=time.time(),
            frame_id=pc2_msg.header.frame_id
        )

    def _ros2_to_image(self, image_msg) -> ImageData:
        """Convert ROS2 Image to ImageData."""
        # Convert from bytes to numpy array
        if image_msg.encoding in ['rgb8', 'bgr8', 'mono8']:
            dtype = np.uint8
        elif image_msg.encoding in ['16UC1', 'mono16']:
            dtype = np.uint16
        elif image_msg.encoding in ['32FC1']:
            dtype = np.float32
        else:
            dtype = np.uint8

        # Reshape image data
        image = np.frombuffer(image_msg.data, dtype=dtype)
        if image_msg.encoding in ['rgb8', 'bgr8']:
            image = image.reshape((image_msg.height, image_msg.width, 3))
        else:
            image = image.reshape((image_msg.height, image_msg.width))

        return ImageData(
            image=image,
            timestamp=time.time(),
            frame_id=image_msg.header.frame_id,
            encoding=image_msg.encoding
        )

    def _ros2_to_imu(self, imu_msg) -> IMUData:
        """Convert ROS2 Imu to IMUData."""
        linear_acceleration = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        angular_velocity = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        orientation = None
        if imu_msg.orientation_covariance[0] != -1:  # Check if orientation is valid
            orientation = np.array([
                imu_msg.orientation.x,
                imu_msg.orientation.y,
                imu_msg.orientation.z,
                imu_msg.orientation.w
            ])

        return IMUData(
            linear_acceleration=linear_acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
            timestamp=time.time(),
            frame_id=imu_msg.header.frame_id
        )

    def _ros2_to_gps(self, fix_msg) -> GPSData:
        """Convert ROS2 NavSatFix to GPSData."""
        # Extract position covariance
        position_covariance = None
        if fix_msg.position_covariance_type != NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            position_covariance = np.array(fix_msg.position_covariance).reshape(3, 3)

        return GPSData(
            latitude=fix_msg.latitude,
            longitude=fix_msg.longitude,
            altitude=fix_msg.altitude,
            position_covariance=position_covariance,
            fix_type=fix_msg.status.status + 1,  # Convert from ROS2 convention
            timestamp=time.time(),
            frame_id=fix_msg.header.frame_id
        )

    def _time_to_ros_time(self, timestamp: float) -> Any:
        """Convert Unix timestamp to ROS2 time."""
        if self.ros2_available:
            from builtin_interfaces.msg import Time
            seconds = int(timestamp)
            nanoseconds = int((timestamp - seconds) * 1e9)
            return Time(sec=seconds, nanosec=nanoseconds)
        return timestamp

    def _ros_time_to_time(self, ros_time) -> float:
        """Convert ROS2 time to Unix timestamp."""
        return ros_time.sec + ros_time.nanosec * 1e-9

    def get_supported_formats(self) -> List[DataFormat]:
        """Get supported ROS2 formats."""
        return [DataFormat.BRAIN_NATIVE, DataFormat.ROS2]


class StandardFormatConverter(DataConverter):
    """
    Standard format converter for JSON, binary, and other common formats.

    This class provides conversion to standard formats for data storage,
    transmission, and interoperability with external systems.
    """

    def __init__(self):
        """Initialize standard format converter."""
        super().__init__()

    def convert(self, data: Any, options: ConversionOptions) -> ConversionResult:
        """
        Convert data to standard format.

        Args:
            data: Input data
            options: Conversion options

        Returns:
            Conversion result
        """
        start_time = time.time()
        success = False
        result_data = None
        error_message = None

        try:
            if options.target_format == DataFormat.JSON:
                result_data = self._to_json(data, options)
            elif options.target_format == DataFormat.BINARY:
                result_data = self._to_binary(data, options)
            elif options.target_format == DataFormat.CSV:
                result_data = self._to_csv(data, options)
            elif options.target_format == DataFormat.PCD:
                result_data = self._to_pcd(data, options)
            else:
                raise ValueError(f"Unsupported target format: {options.target_format}")

            success = True

        except Exception as e:
            error_message = str(e)
            logger.error(f"Standard format conversion failed: {e}")

        conversion_time = time.time() - start_time
        self._update_stats(success, conversion_time)

        return ConversionResult(
            success=success,
            data=result_data,
            format=options.target_format,
            error_message=error_message,
            conversion_time=conversion_time
        )

    def _to_json(self, data: Any, options: ConversionOptions) -> str:
        """Convert data to JSON format."""
        if isinstance(data, SensorDataPacket):
            return self._sensor_packet_to_json(data, options)
        elif isinstance(data, (PointCloudData, ImageData, IMUData, GPSData, WeatherData)):
            return self._sensor_data_to_json(data, options)
        else:
            return json.dumps(data, default=str, indent=2)

    def _sensor_packet_to_json(self, packet: SensorDataPacket, options: ConversionOptions) -> str:
        """Convert sensor data packet to JSON."""
        json_data = {
            "sensor_id": packet.sensor_id,
            "sensor_type": packet.sensor_type.value,
            "timestamp": packet.timestamp,
            "quality_score": packet.quality_score,
            "frame_id": packet.frame_id,
            "sequence_number": packet.sequence_number,
            "processing_time": packet.processing_time,
        }

        # Add sensor-specific data
        if isinstance(packet.data, PointCloudData):
            json_data["data"] = self._pointcloud_to_dict(packet.data, options)
        elif isinstance(packet.data, ImageData):
            json_data["data"] = self._image_to_dict(packet.data, options)
        elif isinstance(packet.data, IMUData):
            json_data["data"] = self._imu_to_dict(packet.data, options)
        elif isinstance(packet.data, GPSData):
            json_data["data"] = self._gps_to_dict(packet.data, options)
        elif isinstance(packet.data, WeatherData):
            json_data["data"] = self._weather_to_dict(packet.data, options)

        # Add calibration parameters if requested
        if options.include_metadata:
            json_data["calibration_params"] = packet.calibration_params

        return json.dumps(json_data, indent=2)

    def _pointcloud_to_dict(self, pc_data: PointCloudData, options: ConversionOptions) -> Dict[str, Any]:
        """Convert point cloud data to dictionary."""
        result = {
            "point_count": pc_data.point_count,
            "frame_id": pc_data.frame_id,
            "timestamp": pc_data.timestamp,
            "has_intensity": pc_data.has_intensity,
            "has_color": pc_data.has_color,
        }

        # Include points (may be large)
        if options.max_size_mb and pc_data.points.nbytes > options.max_size_mb * 1024 * 1024:
            result["points_encoded"] = "base64"
            result["points"] = base64.b64encode(pc_data.points.tobytes()).decode('ascii')
        else:
            result["points"] = pc_data.points.tolist()

        # Include intensity if available
        if pc_data.intensity is not None:
            result["intensity"] = pc_data.intensity.tolist()

        # Include RGB if available
        if pc_data.rgb is not None:
            result["rgb"] = pc_data.rgb.tolist()

        return result

    def _image_to_dict(self, img_data: ImageData, options: ConversionOptions) -> Dict[str, Any]:
        """Convert image data to dictionary."""
        result = {
            "height": img_data.height,
            "width": img_data.width,
            "channels": img_data.channels,
            "frame_id": img_data.frame_id,
            "timestamp": img_data.timestamp,
            "has_depth": img_data.has_depth,
            "encoding": img_data.encoding,
        }

        # Include image data (may be large)
        if options.max_size_mb and img_data.image.nbytes > options.max_size_mb * 1024 * 1024:
            result["image_encoded"] = "base64"
            result["image"] = base64.b64encode(img_data.image.tobytes()).decode('ascii')
        else:
            result["image"] = img_data.image.tolist()

        # Include depth if available
        if img_data.depth is not None:
            result["depth"] = img_data.depth.tolist()

        # Include camera parameters
        if img_data.camera_params is not None:
            result["camera_params"] = {
                "fx": img_data.camera_params.fx,
                "fy": img_data.camera_params.fy,
                "cx": img_data.camera_params.cx,
                "cy": img_data.camera_params.cy,
                "width": img_data.camera_params.width,
                "height": img_data.camera_params.height,
            }

        return result

    def _imu_to_dict(self, imu_data: IMUData, options: ConversionOptions) -> Dict[str, Any]:
        """Convert IMU data to dictionary."""
        result = {
            "frame_id": imu_data.frame_id,
            "timestamp": imu_data.timestamp,
            "linear_acceleration": imu_data.linear_acceleration.tolist(),
            "angular_velocity": imu_data.angular_velocity.tolist(),
        }

        # Include orientation if available
        if imu_data.orientation is not None:
            result["orientation"] = imu_data.orientation.tolist()

        return result

    def _gps_to_dict(self, gps_data: GPSData, options: ConversionOptions) -> Dict[str, Any]:
        """Convert GPS data to dictionary."""
        result = {
            "latitude": gps_data.latitude,
            "longitude": gps_data.longitude,
            "altitude": gps_data.altitude,
            "frame_id": gps_data.frame_id,
            "timestamp": gps_data.timestamp,
            "fix_type": gps_data.fix_type,
            "satellites_used": gps_data.satellites_used,
            "hdop": gps_data.hdop,
            "vdop": gps_data.vdop,
        }

        # Include velocity if available
        if gps_data.velocity is not None:
            result["velocity"] = gps_data.velocity.tolist()

        return result

    def _weather_to_dict(self, weather_data: WeatherData, options: ConversionOptions) -> Dict[str, Any]:
        """Convert weather data to dictionary."""
        result = {
            "temperature": weather_data.temperature,
            "humidity": weather_data.humidity,
            "pressure": weather_data.pressure,
            "wind_speed": weather_data.wind_speed,
            "wind_direction": weather_data.wind_direction,
            "frame_id": weather_data.frame_id,
            "timestamp": weather_data.timestamp,
        }

        # Include optional parameters
        if weather_data.visibility is not None:
            result["visibility"] = weather_data.visibility
        if weather_data.precipitation is not None:
            result["precipitation"] = weather_data.precipitation
        if weather_data.uv_index is not None:
            result["uv_index"] = weather_data.uv_index
        if weather_data.air_quality_index is not None:
            result["air_quality_index"] = weather_data.air_quality_index

        return result

    def _to_binary(self, data: Any, options: ConversionOptions) -> bytes:
        """Convert data to binary format."""
        if isinstance(data, PointCloudData):
            return self._pointcloud_to_binary(data, options)
        elif isinstance(data, ImageData):
            return self._image_to_binary(data, options)
        else:
            # Generic binary serialization
            return self._generic_to_binary(data, options)

    def _to_csv(self, data: Any, options: ConversionOptions) -> str:
        """Convert data to CSV format."""
        if isinstance(data, PointCloudData):
            return self._pointcloud_to_csv(data, options)
        elif isinstance(data, (IMUData, GPSData, WeatherData)):
            return self._tabular_to_csv(data, options)
        else:
            raise ValueError(f"CSV conversion not supported for data type: {type(data)}")

    def _to_pcd(self, data: Any, options: ConversionOptions) -> str:
        """Convert point cloud data to PCD format."""
        if not isinstance(data, PointCloudData):
            raise ValueError("PCD format only supports point cloud data")

        # PCD header
        header = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            f"FIELDS x y z",
            f"SIZE 4 4 4",
            f"TYPE F F F",
            f"COUNT 1 1 1",
            f"WIDTH {data.point_count}",
            "HEIGHT 1",
            f"VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {data.point_count}",
            "DATA ascii"
        ]

        # Add intensity field if available
        if data.intensity is not None:
            header[2] = "FIELDS x y z intensity"
            header[3] = "SIZE 4 4 4 4"
            header[4] = "TYPE F F F F"
            header[5] = "COUNT 1 1 1 1"

        # Convert points to strings
        if data.intensity is not None:
            point_strings = [
                f"{x:.6f} {y:.6f} {z:.6f} {i:.6f}"
                for (x, y, z), i in zip(data.points, data.intensity)
            ]
        else:
            point_strings = [
                f"{x:.6f} {y:.6f} {z:.6f}"
                for x, y, z in data.points
            ]

        return "\n".join(header + point_strings)

    def _pointcloud_to_binary(self, pc_data: PointCloudData, options: ConversionOptions) -> bytes:
        """Convert point cloud to binary format."""
        # Simple binary format: [point_count][points][intensity][rgb]
        binary_data = bytearray()

        # Header
        binary_data.extend(struct.pack('<I', pc_data.point_count))
        binary_data.extend(struct.pack('<B', 1 if pc_data.has_intensity else 0))
        binary_data.extend(struct.pack('<B', 1 if pc_data.has_color else 0))

        # Points
        binary_data.extend(pc_data.points.astype(np.float32).tobytes())

        # Intensity
        if pc_data.intensity is not None:
            binary_data.extend(pc_data.intensity.astype(np.float32).tobytes())

        # RGB
        if pc_data.rgb is not None:
            binary_data.extend(pc_data.rgb.astype(np.uint8).tobytes())

        return bytes(binary_data)

    def _image_to_binary(self, img_data: ImageData, options: ConversionOptions) -> bytes:
        """Convert image to binary format."""
        # Simple format: [height][width][channels][image_data]
        binary_data = bytearray()

        binary_data.extend(struct.pack('<I', img_data.height))
        binary_data.extend(struct.pack('<I', img_data.width))
        binary_data.extend(struct.pack('<B', img_data.channels))
        binary_data.extend(img_data.image.tobytes())

        return bytes(binary_data)

    def _generic_to_binary(self, data: Any, options: ConversionOptions) -> bytes:
        """Generic binary serialization using pickle."""
        import pickle
        return pickle.dumps(data)

    def _pointcloud_to_csv(self, pc_data: PointCloudData, options: ConversionOptions) -> str:
        """Convert point cloud to CSV format."""
        header = "x,y,z"
        if pc_data.intensity is not None:
            header += ",intensity"
        if pc_data.rgb is not None:
            header += ",r,g,b"

        lines = [header]

        for i, (x, y, z) in enumerate(pc_data.points):
            row = [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]

            if pc_data.intensity is not None:
                row.append(f"{pc_data.intensity[i]:.6f}")

            if pc_data.rgb is not None:
                r, g, b = pc_data.rgb[i]
                row.extend([str(r), str(g), str(b)])

            lines.append(",".join(row))

        return "\n".join(lines)

    def _tabular_to_csv(self, data: Any, options: ConversionOptions) -> str:
        """Convert tabular sensor data to CSV."""
        if isinstance(data, IMUData):
            header = "timestamp,ax,ay,az,wx,wy,wz,qx,qy,qz,qw"
            values = [
                data.timestamp,
                data.linear_acceleration[0], data.linear_acceleration[1], data.linear_acceleration[2],
                data.angular_velocity[0], data.angular_velocity[1], data.angular_velocity[2],
            ]
            if data.orientation is not None:
                values.extend([data.orientation[0], data.orientation[1], data.orientation[2], data.orientation[3]])
            else:
                values.extend(["", "", "", ""])

        elif isinstance(data, GPSData):
            header = "timestamp,latitude,longitude,altitude,fix_type,satellites_used,hdop,vdop"
            values = [
                data.timestamp, data.latitude, data.longitude, data.altitude,
                data.fix_type, data.satellites_used, data.hdop, data.vdop
            ]

        elif isinstance(data, WeatherData):
            header = "timestamp,temperature,humidity,pressure,wind_speed,wind_direction"
            values = [
                data.timestamp, data.temperature, data.humidity, data.pressure,
                data.wind_speed, data.wind_direction
            ]
        else:
            raise ValueError(f"CSV conversion not supported for data type: {type(data)}")

        return f"{header}\n{','.join(map(str, values))}"

    def get_supported_formats(self) -> List[DataFormat]:
        """Get supported standard formats."""
        return [
            DataFormat.BRAIN_NATIVE,
            DataFormat.JSON,
            DataFormat.BINARY,
            DataFormat.CSV,
            DataFormat.PCD,
        ]


# Factory function for creating converters
def create_converter(format_type: DataFormat) -> DataConverter:
    """
    Factory function to create appropriate converter.

    Args:
        format_type: Target format type

    Returns:
        Configured converter instance

    Raises:
        ValueError: If format type is not supported
    """
    if format_type == DataFormat.ROS2:
        return ROS2Converter()
    elif format_type in [DataFormat.JSON, DataFormat.BINARY, DataFormat.CSV, DataFormat.PCD]:
        return StandardFormatConverter()
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


# Utility functions for batch conversion
def convert_batch(data_list: List[Any], options: ConversionOptions) -> List[ConversionResult]:
    """
    Convert a batch of data items.

    Args:
        data_list: List of data items to convert
        options: Conversion options

    Returns:
        List of conversion results
    """
    converter = create_converter(options.target_format)
    results = []

    for data in data_list:
        result = converter.convert(data, options)
        results.append(result)

    return results


def validate_conversion_result(result: ConversionResult, expected_format: DataFormat) -> bool:
    """
    Validate conversion result.

    Args:
        result: Conversion result to validate
        expected_format: Expected output format

    Returns:
        True if result is valid
    """
    if not result.success:
        return False

    if result.format != expected_format:
        return False

    if result.data is None:
        return False

    return True