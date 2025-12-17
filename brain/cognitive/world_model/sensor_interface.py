# -*- coding: utf-8 -*-
"""
Sensor Interface Module

This module defines the base sensor interface and concrete sensor implementations
for different sensor types in the Brain cognitive world model system.

Author: Brain Development Team
Date: 2025-12-17
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Union
import threading
import time
import queue
import numpy as np
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
    SensorMetadata,
    SensorDataBuffer,
    validate_sensor_data_quality,
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SensorConfig:
    """Configuration parameters for sensor initialization."""
    sensor_id: str
    sensor_type: SensorType
    frame_id: str = "base_link"
    update_rate: float = 10.0  # Hz
    auto_start: bool = True
    buffer_size: int = 100
    enable_compression: bool = False
    quality_threshold: float = 0.5
    max_processing_time: float = 0.1  # seconds

    # Calibration parameters
    calibration_params: Dict[str, Any] = field(default_factory=dict)

    # Data filtering parameters
    enable_noise_filtering: bool = True
    enable_outlier_removal: bool = True
    min_data_quality: float = 0.3

    # ROS2 specific (if applicable)
    ros2_topic: Optional[str] = None
    ros2_qos_profile: str = "best_effort"


class BaseSensor(ABC):
    """
    Abstract base class for all sensor implementations.

    This class provides the common interface and functionality for all sensors
    in the Brain system, including data acquisition, buffering, and quality
    assessment.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize the sensor with configuration.

        Args:
            config: Sensor configuration parameters
        """
        self.config = config
        self.metadata: Optional[SensorMetadata] = None
        self.is_running = False
        self.last_update_time = 0.0
        self.total_packets_received = 0
        self.total_packets_dropped = 0

        # Thread safety
        self._lock = threading.Lock()
        self._data_lock = threading.Lock()

        # Data buffering
        self._buffer = SensorDataBuffer(
            max_size=config.buffer_size,
            max_age_seconds=5.0
        )

        # Callback system for data processing
        self._callbacks: List[Callable[[SensorDataPacket], None]] = []

        # Threading
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "start_time": 0.0,
            "last_data_time": 0.0,
            "average_update_rate": 0.0,
            "quality_history": [],
            "processing_times": [],
        }

        logger.info(f"Initialized sensor {config.sensor_id} of type {config.sensor_type}")

    @abstractmethod
    def _initialize_sensor(self) -> bool:
        """
        Initialize the physical sensor hardware or data source.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def _acquire_data(self) -> Optional[Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]]:
        """
        Acquire raw data from the sensor.

        Returns:
            Raw sensor data or None if acquisition failed
        """
        pass

    @abstractmethod
    def _cleanup_sensor(self) -> None:
        """Cleanup sensor resources and connections."""
        pass

    def start(self) -> bool:
        """
        Start the sensor data acquisition thread.

        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self.is_running:
                logger.warning(f"Sensor {self.config.sensor_id} is already running")
                return False

            if not self._initialize_sensor():
                logger.error(f"Failed to initialize sensor {self.config.sensor_id}")
                return False

            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._acquisition_loop,
                name=f"Sensor-{self.config.sensor_id}",
                daemon=True
            )

            self.is_running = True
            self._stats["start_time"] = time.time()
            self._worker_thread.start()

            logger.info(f"Started sensor {self.config.sensor_id}")
            return True

    def stop(self) -> None:
        """Stop the sensor data acquisition thread."""
        with self._lock:
            if not self.is_running:
                return

            self._stop_event.set()
            self.is_running = False

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)

            self._cleanup_sensor()
            logger.info(f"Stopped sensor {self.config.sensor_id}")

    def get_latest_data(self, count: int = 1) -> List[SensorDataPacket]:
        """
        Get the latest sensor data packets.

        Args:
            count: Number of packets to retrieve

        Returns:
            List of latest data packets
        """
        return self._buffer.get_latest(count)

    def get_data_by_timerange(self, start_time: float, end_time: float) -> List[SensorDataPacket]:
        """
        Get sensor data packets within a time range.

        Args:
            start_time: Start time (Unix timestamp)
            end_time: End time (Unix timestamp)

        Returns:
            List of data packets within time range
        """
        return self._buffer.get_by_timerange(start_time, end_time)

    def add_callback(self, callback: Callable[[SensorDataPacket], None]) -> None:
        """
        Add a callback function to be called on new data.

        Args:
            callback: Function to call with new data packets
        """
        with self._data_lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SensorDataPacket], None]) -> None:
        """
        Remove a callback function.

        Args:
            callback: Function to remove
        """
        with self._data_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get sensor statistics and performance metrics.

        Returns:
            Dictionary containing sensor statistics
        """
        with self._lock:
            current_time = time.time()
            runtime = current_time - self._stats["start_time"]

            # Calculate average update rate
            if runtime > 0:
                self._stats["average_update_rate"] = self.total_packets_received / runtime

            # Calculate packet loss rate
            total_packets = self.total_packets_received + self.total_packets_dropped
            loss_rate = (self.total_packets_dropped / total_packets * 100) if total_packets > 0 else 0

            return {
                "sensor_id": self.config.sensor_id,
                "sensor_type": self.config.sensor_type.value,
                "is_running": self.is_running,
                "runtime_seconds": runtime,
                "packets_received": self.total_packets_received,
                "packets_dropped": self.total_packets_dropped,
                "loss_rate_percent": loss_rate,
                "average_update_rate": self._stats["average_update_rate"],
                "last_update_time": self.last_update_time,
                "buffer_size": self._buffer.size(),
                "config": {
                    "update_rate": self.config.update_rate,
                    "frame_id": self.config.frame_id,
                    "quality_threshold": self.config.quality_threshold,
                }
            }

    def _acquisition_loop(self) -> None:
        """Main data acquisition loop running in worker thread."""
        logger.info(f"Starting acquisition loop for sensor {self.config.sensor_id}")

        while not self._stop_event.is_set():
            loop_start_time = time.time()

            try:
                # Acquire raw data
                raw_data = self._acquire_data()
                if raw_data is None:
                    continue

                # Process raw data into packet
                packet = self._process_raw_data(raw_data)
                if packet is None:
                    continue

                # Add to buffer
                self._buffer.add(packet)

                # Update statistics
                self.total_packets_received += 1
                self.last_update_time = time.time()

                # Call callbacks
                self._notify_callbacks(packet)

                # Clean old data
                self._buffer.cleanup_old_data()

            except Exception as e:
                logger.error(f"Error in acquisition loop for sensor {self.config.sensor_id}: {e}")
                self.total_packets_dropped += 1

            # Control update rate
            loop_time = time.time() - loop_start_time
            target_period = 1.0 / self.config.update_rate
            sleep_time = max(0, target_period - loop_time)

            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

        logger.info(f"Acquisition loop ended for sensor {self.config.sensor_id}")

    def _process_raw_data(self, raw_data: Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]) -> Optional[SensorDataPacket]:
        """
        Process raw sensor data into a standardized packet.

        Args:
            raw_data: Raw sensor data

        Returns:
            Processed sensor data packet or None if processing failed
        """
        processing_start_time = time.time()

        try:
            # Validate data quality
            quality_assessment = validate_sensor_data_quality(raw_data)
            quality_score = quality_assessment["score"]

            # Check quality threshold
            if quality_score < self.config.min_data_quality:
                logger.warning(f"Data quality below threshold for sensor {self.config.sensor_id}: {quality_score}")
                self.total_packets_dropped += 1
                return None

            # Apply filtering if enabled
            if self.config.enable_noise_filtering:
                raw_data = self._apply_noise_filtering(raw_data)

            if self.config.enable_outlier_removal:
                raw_data = self._remove_outliers(raw_data)

            # Create data packet
            processing_time = time.time() - processing_start_time

            packet = SensorDataPacket(
                sensor_id=self.config.sensor_id,
                sensor_type=self.config.sensor_type,
                timestamp=time.time(),
                data=raw_data,
                quality_score=quality_score,
                calibration_params=self.config.calibration_params,
                frame_id=self.config.frame_id,
                processing_time=processing_time,
                sequence_number=self.total_packets_received
            )

            # Check processing time
            if processing_time > self.config.max_processing_time:
                logger.warning(f"Processing time exceeded threshold for sensor {self.config.sensor_id}: {processing_time:.3f}s")

            self._stats["processing_times"].append(processing_time)
            if len(self._stats["processing_times"]) > 100:
                self._stats["processing_times"].pop(0)

            return packet

        except Exception as e:
            logger.error(f"Error processing data for sensor {self.config.sensor_id}: {e}")
            self.total_packets_dropped += 1
            return None

    def _apply_noise_filtering(self, raw_data: Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]) -> Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]:
        """Apply noise filtering to raw sensor data."""
        # Base implementation - subclasses should override with specific filtering
        return raw_data

    def _remove_outliers(self, raw_data: Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]) -> Union[PointCloudData, ImageData, IMUData, GPSData, WeatherData]:
        """Remove outliers from raw sensor data."""
        # Base implementation - subclasses should override with specific filtering
        return raw_data

    def _notify_callbacks(self, packet: SensorDataPacket) -> None:
        """Notify all registered callbacks with new data packet."""
        with self._data_lock:
            for callback in self._callbacks:
                try:
                    callback(packet)
                except Exception as e:
                    logger.error(f"Error in callback for sensor {self.config.sensor_id}: {e}")

    def __del__(self):
        """Destructor - ensure sensor is properly stopped."""
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()


class PointCloudSensor(BaseSensor):
    """
    Point cloud sensor implementation for LiDAR, radar, and 3D scanners.

    This class handles point cloud data acquisition and processing with
    specialized filtering and preprocessing capabilities.
    """

    def __init__(self, config: SensorConfig, metadata: Optional[SensorMetadata] = None):
        """
        Initialize point cloud sensor.

        Args:
            config: Sensor configuration
            metadata: Sensor metadata and capabilities
        """
        super().__init__(config)
        self.metadata = metadata

        # Point cloud specific parameters
        self.voxel_size: float = 0.05  # For downsampling
        self.max_range: float = 100.0   # Maximum range in meters
        self.min_range: float = 0.5     # Minimum range in meters
        self.remove_ground_plane: bool = True
        self.ground_threshold: float = 0.1  # meters

    def _initialize_sensor(self) -> bool:
        """Initialize point cloud sensor hardware or data source."""
        # Implementation would connect to actual hardware or data source
        # For now, we simulate initialization
        logger.info(f"Initializing point cloud sensor {self.config.sensor_id}")
        return True

    def _acquire_data(self) -> Optional[PointCloudData]:
        """
        Acquire point cloud data from sensor.

        Returns:
            Point cloud data or None if acquisition failed
        """
        # Simulated data acquisition - replace with actual sensor interface
        try:
            # Generate synthetic point cloud for demonstration
            num_points = 1000
            points = self._generate_synthetic_point_cloud(num_points)
            intensity = np.random.uniform(0, 1, num_points)

            return PointCloudData(
                points=points,
                intensity=intensity,
                timestamp=time.time(),
                frame_id=self.config.frame_id
            )

        except Exception as e:
            logger.error(f"Failed to acquire point cloud data: {e}")
            return None

    def _cleanup_sensor(self) -> None:
        """Cleanup point cloud sensor resources."""
        logger.info(f"Cleaning up point cloud sensor {self.config.sensor_id}")

    def _generate_synthetic_point_cloud(self, num_points: int) -> np.ndarray:
        """Generate synthetic point cloud for testing."""
        # Create a simple cone pattern
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        r = np.random.uniform(0, 10, num_points)
        z = np.random.uniform(-2, 2, num_points)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.column_stack([x, y, z])

    def _apply_noise_filtering(self, raw_data: PointCloudData) -> PointCloudData:
        """Apply noise filtering specific to point clouds."""
        # Remove points outside valid range
        distances = np.linalg.norm(raw_data.points, axis=1)
        valid_mask = (distances >= self.min_range) & (distances <= self.max_range)

        filtered_points = raw_data.points[valid_mask]
        filtered_intensity = raw_data.intensity[valid_mask] if raw_data.intensity is not None else None
        filtered_rgb = raw_data.rgb[valid_mask] if raw_data.rgb is not None else None

        return PointCloudData(
            points=filtered_points,
            intensity=filtered_intensity,
            rgb=filtered_rgb,
            timestamp=raw_data.timestamp,
            frame_id=raw_data.frame_id,
            sensor_pose=raw_data.sensor_pose
        )

    def _remove_outliers(self, raw_data: PointCloudData) -> PointCloudData:
        """Remove statistical outliers from point cloud."""
        # Simple statistical outlier removal
        if len(raw_data.points) < 10:
            return raw_data

        # Calculate distances to neighbors (simplified)
        distances = np.linalg.norm(raw_data.points - np.mean(raw_data.points, axis=0), axis=1)
        threshold = np.mean(distances) + 2 * np.std(distances)

        inlier_mask = distances < threshold

        filtered_points = raw_data.points[inlier_mask]
        filtered_intensity = raw_data.intensity[inlier_mask] if raw_data.intensity is not None else None
        filtered_rgb = raw_data.rgb[inlier_mask] if raw_data.rgb is not None else None

        return PointCloudData(
            points=filtered_points,
            intensity=filtered_intensity,
            rgb=filtered_rgb,
            timestamp=raw_data.timestamp,
            frame_id=raw_data.frame_id,
            sensor_pose=raw_data.sensor_pose
        )


class ImageSensor(BaseSensor):
    """
    Image sensor implementation for cameras and vision systems.

    This class handles image data acquisition with support for various
    camera types and image processing capabilities.
    """

    def __init__(self, config: SensorConfig, camera_intrinsics: Optional[CameraIntrinsics] = None):
        """
        Initialize image sensor.

        Args:
            config: Sensor configuration
            camera_intrinsics: Camera intrinsic parameters
        """
        super().__init__(config)
        self.camera_intrinsics = camera_intrinsics

        # Image specific parameters
        self.auto_exposure: bool = True
        self.auto_white_balance: bool = True
        self.image_width: int = 1920
        self.image_height: int = 1080
        self.enable_rectification: bool = True

    def _initialize_sensor(self) -> bool:
        """Initialize image sensor hardware or data source."""
        logger.info(f"Initializing image sensor {self.config.sensor_id}")
        # Implementation would connect to actual camera
        return True

    def _acquire_data(self) -> Optional[ImageData]:
        """
        Acquire image data from sensor.

        Returns:
            Image data or None if acquisition failed
        """
        try:
            # Simulated image acquisition
            image = self._generate_synthetic_image()

            return ImageData(
                image=image,
                camera_params=self.camera_intrinsics,
                timestamp=time.time(),
                frame_id=self.config.frame_id
            )

        except Exception as e:
            logger.error(f"Failed to acquire image data: {e}")
            return None

    def _cleanup_sensor(self) -> None:
        """Cleanup image sensor resources."""
        logger.info(f"Cleaning up image sensor {self.config.sensor_id}")

    def _generate_synthetic_image(self) -> np.ndarray:
        """Generate synthetic image for testing."""
        # Create a gradient image with noise
        image = np.random.randint(0, 256, (self.image_height, self.image_width, 3), dtype=np.uint8)

        # Add some pattern
        y, x = np.ogrid[:self.image_height, :self.image_width]
        center_x, center_y = self.image_width // 2, self.image_height // 2

        # Create circular gradient
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        gradient = np.clip(255 - distance * 2, 0, 255).astype(np.uint8)

        image[:, :, 0] = gradient  # Red channel
        image[:, :, 1] = gradient // 2  # Green channel
        image[:, :, 2] = gradient // 3  # Blue channel

        return image

    def _apply_noise_filtering(self, raw_data: ImageData) -> ImageData:
        """Apply noise filtering specific to images."""
        # Simple Gaussian blur for noise reduction
        import cv2

        filtered_image = cv2.GaussianBlur(raw_data.image, (3, 3), 0)

        return ImageData(
            image=filtered_image,
            depth=raw_data.depth,
            camera_params=raw_data.camera_params,
            timestamp=raw_data.timestamp,
            frame_id=raw_data.frame_id,
            encoding=raw_data.encoding
        )


class IMUSensor(BaseSensor):
    """
    IMU sensor implementation for inertial measurement units.

    This class handles IMU data acquisition including acceleration,
    angular velocity, and orientation measurements.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize IMU sensor.

        Args:
            config: Sensor configuration
        """
        super().__init__(config)

        # IMU specific parameters
        self.gravity_compensation: bool = True
        self.bias_estimation: bool = True
        self.integration_window: float = 0.1  # seconds

    def _initialize_sensor(self) -> bool:
        """Initialize IMU sensor hardware or data source."""
        logger.info(f"Initializing IMU sensor {self.config.sensor_id}")
        return True

    def _acquire_data(self) -> Optional[IMUData]:
        """
        Acquire IMU data from sensor.

        Returns:
            IMU data or None if acquisition failed
        """
        try:
            # Simulated IMU data
            acceleration = np.random.randn(3) * 0.1 + np.array([0, 0, 9.81])  # Include gravity
            angular_velocity = np.random.randn(3) * 0.01
            orientation = self._generate_random_quaternion()

            return IMUData(
                linear_acceleration=acceleration,
                angular_velocity=angular_velocity,
                orientation=orientation,
                timestamp=time.time(),
                frame_id=self.config.frame_id
            )

        except Exception as e:
            logger.error(f"Failed to acquire IMU data: {e}")
            return None

    def _cleanup_sensor(self) -> None:
        """Cleanup IMU sensor resources."""
        logger.info(f"Cleaning up IMU sensor {self.config.sensor_id}")

    def _generate_random_quaternion(self) -> np.ndarray:
        """Generate random unit quaternion."""
        q = np.random.randn(4)
        q = q / np.linalg.norm(q)  # Normalize
        return q

    def _apply_noise_filtering(self, raw_data: IMUData) -> IMUData:
        """Apply noise filtering specific to IMU data."""
        # Simple low-pass filter
        alpha = 0.1  # Filter coefficient

        # For demonstration, we'll just slightly smooth the values
        filtered_acceleration = raw_data.linear_acceleration * (1 - alpha) + np.array([0, 0, 9.81]) * alpha
        filtered_angular_velocity = raw_data.angular_velocity * (1 - alpha)

        return IMUData(
            linear_acceleration=filtered_acceleration,
            angular_velocity=filtered_angular_velocity,
            orientation=raw_data.orientation,
            timestamp=raw_data.timestamp,
            frame_id=raw_data.frame_id
        )


class GPSSensor(BaseSensor):
    """
    GPS sensor implementation for global positioning systems.

    This class handles GPS data acquisition including position,
    velocity, and satellite information.
    """

    def __init__(self, config: SensorConfig):
        """
        Initialize GPS sensor.

        Args:
            config: Sensor configuration
        """
        super().__init__(config)

        # GPS specific parameters
        self.coordinate_system: str = "WGS84"
        self.enable_dgps: bool = True
        self.min_satellites: int = 4

    def _initialize_sensor(self) -> bool:
        """Initialize GPS sensor hardware or data source."""
        logger.info(f"Initializing GPS sensor {self.config.sensor_id}")
        return True

    def _acquire_data(self) -> Optional[GPSData]:
        """
        Acquire GPS data from sensor.

        Returns:
            GPS data or None if acquisition failed
        """
        try:
            # Simulated GPS data (around Beijing area)
            latitude = 39.9042 + np.random.randn() * 0.0001
            longitude = 116.4074 + np.random.randn() * 0.0001
            altitude = 50.0 + np.random.randn() * 0.5

            velocity = np.random.randn(3) * 0.5

            return GPSData(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                velocity=velocity,
                fix_type=np.random.choice([1, 2, 3]),
                satellites_used=np.random.randint(4, 12),
                timestamp=time.time(),
                frame_id=self.config.frame_id
            )

        except Exception as e:
            logger.error(f"Failed to acquire GPS data: {e}")
            return None

    def _cleanup_sensor(self) -> None:
        """Cleanup GPS sensor resources."""
        logger.info(f"Cleaning up GPS sensor {self.config.sensor_id}")

    def _apply_noise_filtering(self, raw_data: GPSData) -> GPSData:
        """Apply noise filtering specific to GPS data."""
        # Kalman filter would be ideal here, but we'll use simple smoothing
        return raw_data  # Placeholder for proper filtering


# Factory function for creating sensors
def create_sensor(config: SensorConfig, **kwargs) -> BaseSensor:
    """
    Factory function to create appropriate sensor instance.

    Args:
        config: Sensor configuration
        **kwargs: Additional sensor-specific parameters

    Returns:
        Configured sensor instance

    Raises:
        ValueError: If sensor type is not supported
    """
    sensor_classes = {
        SensorType.POINT_CLOUD: PointCloudSensor,
        SensorType.LIDAR: PointCloudSensor,
        SensorType.RADAR: PointCloudSensor,
        SensorType.IMAGE: ImageSensor,
        SensorType.CAMERA: ImageSensor,
        SensorType.DEPTH_CAMERA: ImageSensor,
        SensorType.THERMAL_CAMERA: ImageSensor,
        SensorType.IMU: IMUSensor,
        SensorType.GPS: GPSSensor,
    }

    if config.sensor_type not in sensor_classes:
        raise ValueError(f"Unsupported sensor type: {config.sensor_type}")

    sensor_class = sensor_classes[config.sensor_type]
    return sensor_class(config, **kwargs)