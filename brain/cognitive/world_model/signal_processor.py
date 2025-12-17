"""
Signal Processing Module for World Model

This module provides comprehensive signal processing capabilities including
IMU filtering, GPS correction, and multi-sensor data fusion preprocessing.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import time
import logging
from scipy import signal as scipy_signal
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal processing parameters."""
    # IMU filtering
    imu_complementary_alpha: float = 0.98  # Complementary filter parameter
    imu_lowpass_cutoff: float = 5.0  # Hz
    imu_highpass_cutoff: float = 0.1  # Hz
    imu_noise_std: float = 0.01

    # GPS correction
    gps_outlier_threshold: float = 3.0  # Standard deviations
    gps_min_satellites: int = 4
    gps_dop_threshold: float = 2.0  # Dilution of precision
    gps_kalman_process_noise: float = 0.1
    gps_kalman_measurement_noise: float = 1.0

    # Sensor fusion
    fusion_window_size: int = 100
    fusion_interpolation_method: str = 'linear'
    fusion_time_alignment_tolerance: float = 0.01  # seconds

    # Calibration
    accel_bias: np.ndarray = None
    gyro_bias: np.ndarray = None
    magnetometer_bias: np.ndarray = None
    accel_scale: np.ndarray = None
    gyro_scale: np.ndarray = None

    # Performance
    buffer_size: int = 1000
    processing_frequency: float = 100.0  # Hz


class IMUFilter:
    """IMU data filtering and processing."""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.orientation = Rotation.identity()
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)

        # Initialize bias and scale factors
        self.accel_bias = config.accel_bias if config.accel_bias is not None else np.zeros(3)
        self.gyro_bias = config.gyro_bias if config.gyro_bias is not None else np.zeros(3)
        self.accel_scale = config.accel_scale if config.accel_scale is not None else np.ones(3)
        self.gyro_scale = config.gyro_scale if config.gyro_scale is not None else np.ones(3)

        # Filter states
        self.lowpass_filter = None
        self.highpass_filter = None
        self._initialize_filters()

        # Complementary filter state
        self.complementary_state = {'angle': np.zeros(3), 'gyro_integral': np.zeros(3)}

    def _initialize_filters(self):
        """Initialize digital filters."""
        nyquist = self.config.processing_frequency / 2

        # Low-pass Butterworth filter
        lowpass_normalized_cutoff = self.config.imu_lowpass_cutoff / nyquist
        self.lowpass_filter = scipy_signal.butter(
            4, lowpass_normalized_cutoff, btype='low'
        )

        # High-pass Butterworth filter
        highpass_normalized_cutoff = self.config.imu_highpass_cutoff / nyquist
        self.highpass_filter = scipy_signal.butter(
            2, highpass_normalized_cutoff, btype='high'
        )

    def calibrate(self, accel_data: np.ndarray, gyro_data: np.ndarray,
                  duration: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Calibrate IMU by estimating biases and scale factors.

        Args:
            accel_data: Nx3 array of accelerometer measurements
            gyro_data: Nx3 array of gyroscope measurements
            duration: Duration of calibration data in seconds

        Returns:
            Calibration parameters
        """
        try:
            # Assume the sensor is stationary for calibration
            # Accelerometer bias: average acceleration should be [0, 0, g] when stationary
            accel_mean = np.mean(accel_data, axis=0)
            expected_gravity = np.array([0, 0, 9.81])
            self.accel_bias = accel_mean - expected_gravity

            # Gyroscope bias: average angular velocity should be [0, 0, 0] when stationary
            self.gyro_bias = np.mean(gyro_data, axis=0)

            # Scale factors (assuming nominal scale for now)
            self.accel_scale = np.ones(3)
            self.gyro_scale = np.ones(3)

            calibration_params = {
                'accel_bias': self.accel_bias,
                'gyro_bias': self.gyro_bias,
                'accel_scale': self.accel_scale,
                'gyro_scale': self.gyro_scale
            }

            logger.info("IMU calibration completed")
            return calibration_params

        except Exception as e:
            logger.error(f"Error during IMU calibration: {str(e)}")
            raise

    def filter_accelerometer(self, accel: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Filter accelerometer data.

        Args:
            accel: 3D acceleration measurement [x, y, z] (m/s^2)
            timestamp: Measurement timestamp

        Returns:
            Filtered acceleration
        """
        try:
            # Apply bias correction and scaling
            accel_corrected = (accel - self.accel_bias) * self.accel_scale

            # Apply low-pass filter to remove high-frequency noise
            if self.lowpass_filter is not None:
                accel_filtered = scipy_signal.filtfilt(
                    self.lowpass_filter[0], self.lowpass_filter[1],
                    accel_corrected.reshape(-1, 1), axis=0
                ).flatten()
            else:
                accel_filtered = accel_corrected

            # Remove gravity component (simplified)
            # This should be done with current orientation estimate
            gravity = self.orientation.apply(np.array([0, 0, 9.81]))
            accel_linear = accel_filtered - gravity

            return accel_linear

        except Exception as e:
            logger.error(f"Error filtering accelerometer data: {str(e)}")
            return accel

    def filter_gyroscope(self, gyro: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Filter gyroscope data.

        Args:
            gyro: 3D angular velocity measurement [x, y, z] (rad/s)
            timestamp: Measurement timestamp

        Returns:
            Filtered angular velocity
        """
        try:
            # Apply bias correction and scaling
            gyro_corrected = (gyro - self.gyro_bias) * self.gyro_scale

            # Apply high-pass filter to remove drift
            if self.highpass_filter is not None:
                gyro_filtered = scipy_signal.filtfilt(
                    self.highpass_filter[0], self.highpass_filter[1],
                    gyro_corrected.reshape(-1, 1), axis=0
                ).flatten()
            else:
                gyro_filtered = gyro_corrected

            return gyro_filtered

        except Exception as e:
            logger.error(f"Error filtering gyroscope data: {str(e)}")
            return gyro

    def update_orientation(self, accel: np.ndarray, gyro: np.ndarray,
                          magnetometer: Optional[np.ndarray] = None,
                          dt: float = 0.01) -> Rotation:
        """
        Update orientation using complementary filter.

        Args:
            accel: Filtered accelerometer data
            gyro: Filtered gyroscope data
            magnetometer: Optional magnetometer data
            dt: Time step

        Returns:
            Updated rotation object
        """
        try:
            # Complementary filter for orientation estimation
            alpha = self.config.imu_complementary_alpha

            # Gyroscope integration
            gyro_angle = gyro * dt
            self.complementary_state['gyro_integral'] += gyro_angle

            # Accelerometer-based tilt correction
            if np.linalg.norm(accel) > 0.1:  # Avoid division by zero
                # Roll and pitch from accelerometer
                accel_norm = accel / np.linalg.norm(accel)
                accel_roll = np.arctan2(accel_norm[1], accel_norm[2])
                accel_pitch = np.arctan2(-accel_norm[0], np.sqrt(accel_norm[1]**2 + accel_norm[2]**2))

                # Correct gyroscope integration with accelerometer
                self.complementary_state['angle'][0] = alpha * self.complementary_state['gyro_integral'][0] + (1 - alpha) * accel_roll
                self.complementary_state['angle'][1] = alpha * self.complementary_state['gyro_integral'][1] + (1 - alpha) * accel_pitch
                self.complementary_state['angle'][2] = self.complementary_state['gyro_integral'][2]  # Yaw from gyroscope only

            # Update rotation object
            self.orientation = Rotation.from_euler('xyz', self.complementary_state['angle'])

            return self.orientation

        except Exception as e:
            logger.error(f"Error updating orientation: {str(e)}")
            return self.orientation

    def integrate_motion(self, accel: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate acceleration to get velocity and position.

        Args:
            accel: Linear acceleration in world frame
            dt: Time step

        Returns:
            Tuple of (velocity, position)
        """
        try:
            # Integrate acceleration to get velocity
            self.velocity += accel * dt

            # Apply velocity damping to prevent drift
            self.velocity *= 0.999

            # Integrate velocity to get position
            self.position += self.velocity * dt

            return self.velocity.copy(), self.position.copy()

        except Exception as e:
            logger.error(f"Error integrating motion: {str(e)}")
            return self.velocity, self.position


class GPSCorrector:
    """GPS data correction and processing."""

    def __init__(self, config: SignalConfig):
        self.config = config

        # Kalman filter for GPS smoothing
        self.kalman_state = np.zeros(6)  # [x, y, z, vx, vy, vz]
        self.kalman_covariance = np.eye(6) * 100

        # History for outlier detection
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)

    def correct_position(self, latitude: float, longitude: float, altitude: float,
                        accuracy: float, num_satellites: int, hdop: float,
                        timestamp: float) -> Tuple[np.ndarray, bool]:
        """
        Correct and validate GPS position.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters
            accuracy: Position accuracy in meters
            num_satellites: Number of satellites
            hdop: Horizontal dilution of precision
            timestamp: Measurement timestamp

        Returns:
            Tuple of (corrected_position_xyz, is_valid)
        """
        try:
            # Validate GPS data quality
            if not self._validate_gps_quality(num_satellites, hdop, accuracy):
                return np.array([0, 0, 0]), False

            # Convert to local ENU coordinates (simplified)
            # In practice, this should use proper coordinate transformation
            position_enu = self._lat_lon_alt_to_enu(latitude, longitude, altitude)

            # Outlier detection
            is_outlier = self._detect_outlier(position_enu)
            if is_outlier:
                logger.warning(f"GPS outlier detected at position {position_enu}")
                return self.kalman_state[:3], False

            # Kalman filter update
            corrected_position = self._kalman_update(position_enu, timestamp)

            # Update history
            self.position_history.append(corrected_position)

            return corrected_position, True

        except Exception as e:
            logger.error(f"Error correcting GPS position: {str(e)}")
            return np.array([0, 0, 0]), False

    def _validate_gps_quality(self, num_satellites: int, hdop: float, accuracy: float) -> bool:
        """Validate GPS data quality."""
        return (num_satellites >= self.config.gps_min_satellites and
                hdop <= self.config.gps_dop_threshold and
                accuracy <= 100.0)  # 100m maximum acceptable accuracy

    def _detect_outlier(self, position: np.ndarray) -> bool:
        """Detect position outliers using statistical methods."""
        if len(self.position_history) < 3:
            return False

        positions = np.array(list(self.position_history))
        mean_position = np.mean(positions, axis=0)
        std_position = np.std(positions, axis=0)

        # Check if position is too far from recent mean
        distance = np.linalg.norm(position - mean_position)
        threshold = self.config.gps_outlier_threshold * np.mean(std_position)

        return distance > threshold

    def _lat_lon_alt_to_enu(self, latitude: float, longitude: float, altitude: float) -> np.ndarray:
        """Convert latitude, longitude, altitude to local ENU coordinates."""
        # Simplified conversion - in practice, use proper geodetic transformations
        # This is a placeholder implementation
        earth_radius = 6371000.0  # meters

        # Assume origin at 0, 0 for simplicity
        x = longitude * np.pi / 180.0 * earth_radius * np.cos(latitude * np.pi / 180.0)
        y = latitude * np.pi / 180.0 * earth_radius
        z = altitude

        return np.array([x, y, z])

    def _kalman_update(self, position: np.ndarray, timestamp: float) -> np.ndarray:
        """Update Kalman filter with new measurement."""
        try:
            # State transition matrix (constant velocity model)
            dt = 0.1  # Assumed time step
            F = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

            # Process noise
            Q = np.eye(6) * self.config.gps_kalman_process_noise

            # Measurement matrix (only position measurements)
            H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])

            # Measurement noise
            R = np.eye(3) * self.config.gps_kalman_measurement_noise

            # Prediction step
            self.kalman_state = F @ self.kalman_state
            self.kalman_covariance = F @ self.kalman_covariance @ F.T + Q

            # Update step
            y = position - H @ self.kalman_state
            S = H @ self.kalman_covariance @ H.T + R
            K = self.kalman_covariance @ H.T @ np.linalg.inv(S)

            self.kalman_state += K @ y
            self.kalman_covariance = (np.eye(6) - K @ H) @ self.kalman_covariance

            return self.kalman_state[:3]

        except Exception as e:
            logger.error(f"Error in Kalman filter update: {str(e)}")
            return position


class MultiSensorFusion:
    """Multi-sensor data fusion and synchronization."""

    def __init__(self, config: SignalConfig):
        self.config = config
        self.sensor_buffers = {}
        self.synchronized_data = {}
        self.reference_time = 0.0

    def add_sensor_data(self, sensor_id: str, data: Dict[str, Any], timestamp: float):
        """
        Add sensor data to buffer for fusion.

        Args:
            sensor_id: Unique sensor identifier
            data: Sensor data dictionary
            timestamp: Data timestamp
        """
        try:
            if sensor_id not in self.sensor_buffers:
                self.sensor_buffers[sensor_id] = deque(maxlen=self.config.buffer_size)

            self.sensor_buffers[sensor_id].append({
                'data': data,
                'timestamp': timestamp
            })

        except Exception as e:
            logger.error(f"Error adding sensor data: {str(e)}")

    def synchronize_sensors(self, reference_timestamp: float) -> Dict[str, Any]:
        """
        Synchronize sensor data to reference timestamp.

        Args:
            reference_timestamp: Reference time for synchronization

        Returns:
            Dictionary of synchronized sensor data
        """
        try:
            synchronized = {}

            for sensor_id, buffer in self.sensor_buffers.items():
                if not buffer:
                    continue

                # Find closest data point to reference timestamp
                closest_data = None
                min_time_diff = float('inf')

                for item in buffer:
                    time_diff = abs(item['timestamp'] - reference_timestamp)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_data = item

                # Check if within tolerance
                if min_time_diff <= self.config.fusion_time_alignment_tolerance:
                    synchronized[sensor_id] = closest_data['data']
                else:
                    # Interpolate if within reasonable range
                    if min_time_diff <= 0.1:  # 100ms max for interpolation
                        interpolated = self._interpolate_sensor_data(
                            sensor_id, buffer, reference_timestamp
                        )
                        if interpolated is not None:
                            synchronized[sensor_id] = interpolated

            return synchronized

        except Exception as e:
            logger.error(f"Error synchronizing sensors: {str(e)}")
            return {}

    def _interpolate_sensor_data(self, sensor_id: str, buffer: deque,
                                target_timestamp: float) -> Optional[Dict[str, Any]]:
        """Interpolate sensor data between two measurements."""
        try:
            # Sort buffer by timestamp
            sorted_data = sorted(buffer, key=lambda x: x['timestamp'])

            # Find surrounding data points
            before = None
            after = None

            for item in sorted_data:
                if item['timestamp'] < target_timestamp:
                    before = item
                elif item['timestamp'] > target_timestamp and after is None:
                    after = item
                    break

            if before is None or after is None:
                return None

            # Interpolation factor
            total_time = after['timestamp'] - before['timestamp']
            if total_time == 0:
                return before['data']

            alpha = (target_timestamp - before['timestamp']) / total_time

            # Linear interpolation for numeric data
            interpolated = self._interpolate_data_dict(
                before['data'], after['data'], alpha
            )

            return interpolated

        except Exception as e:
            logger.error(f"Error interpolating sensor data: {str(e)}")
            return None

    def _interpolate_data_dict(self, data1: Dict[str, Any], data2: Dict[str, Any],
                              alpha: float) -> Dict[str, Any]:
        """Interpolate between two data dictionaries."""
        interpolated = {}

        for key in data1.keys():
            if key not in data2:
                interpolated[key] = data1[key]
                continue

            val1 = data1[key]
            val2 = data2[key]

            # Handle different data types
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                interpolated[key] = val1 * (1 - alpha) + val2 * alpha
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                interpolated[key] = val1 * (1 - alpha) + val2 * alpha
            else:
                # For non-numeric data, choose closer one
                interpolated[key] = data1 if alpha < 0.5 else data2

        return interpolated

    def fuse_imu_gps(self, imu_data: Dict[str, Any], gps_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse IMU and GPS data for improved positioning.

        Args:
            imu_data: Processed IMU data
            gps_data: Processed GPS data

        Returns:
            Fused positioning data
        """
        try:
            # Simple fusion: use GPS for absolute position, IMU for orientation and motion
            fused_data = {
                'position': gps_data.get('position', np.zeros(3)),
                'position_valid': gps_data.get('valid', False),
                'velocity': imu_data.get('velocity', np.zeros(3)),
                'orientation': imu_data.get('orientation', Rotation.identity()),
                'acceleration': imu_data.get('acceleration', np.zeros(3)),
                'angular_velocity': imu_data.get('angular_velocity', np.zeros(3)),
                'confidence': self._compute_fusion_confidence(imu_data, gps_data)
            }

            return fused_data

        except Exception as e:
            logger.error(f"Error fusing IMU and GPS data: {str(e)}")
            return {}

    def _compute_fusion_confidence(self, imu_data: Dict[str, Any],
                                  gps_data: Dict[str, Any]) -> float:
        """Compute confidence score for fused data."""
        try:
            imu_confidence = imu_data.get('quality_score', 0.5)
            gps_confidence = 1.0 if gps_data.get('valid', False) else 0.0

            # Weight GPS higher for position, IMU higher for orientation
            position_confidence = 0.7 * gps_confidence + 0.3 * imu_confidence
            orientation_confidence = imu_confidence

            return (position_confidence + orientation_confidence) / 2.0

        except Exception:
            return 0.5


class SignalProcessor:
    """Main signal processing class."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

        # Initialize processing components
        self.imu_filter = IMUFilter(self.config)
        self.gps_corrector = GPSCorrector(self.config)
        self.multi_sensor_fusion = MultiSensorFusion(self.config)

        # Processing state
        self.is_processing = False
        self.processing_thread = None

    def process_imu_data(self, accel: np.ndarray, gyro: np.ndarray,
                         magnetometer: Optional[np.ndarray] = None,
                         timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Process IMU data with full pipeline.

        Args:
            accel: 3D acceleration [x, y, z]
            gyro: 3D angular velocity [x, y, z]
            magnetometer: Optional 3D magnetic field [x, y, z]
            timestamp: Measurement timestamp

        Returns:
            Dictionary containing processed IMU data
        """
        try:
            # Filter accelerometer data
            accel_filtered = self.imu_filter.filter_accelerometer(accel, timestamp)

            # Filter gyroscope data
            gyro_filtered = self.imu_filter.filter_gyroscope(gyro, timestamp)

            # Update orientation
            orientation = self.imu_filter.update_orientation(
                accel_filtered, gyro_filtered, magnetometer
            )

            # Integrate motion
            velocity, position = self.imu_filter.integrate_motion(
                accel_filtered, 1.0 / self.config.processing_frequency
            )

            # Quality assessment
            quality_score = self._assess_imu_quality(accel, gyro, accel_filtered, gyro_filtered)

            return {
                'acceleration': accel_filtered,
                'angular_velocity': gyro_filtered,
                'orientation': orientation,
                'velocity': velocity,
                'position': position,
                'quality_score': quality_score,
                'timestamp': timestamp
            }

        except Exception as e:
            logger.error(f"Error processing IMU data: {str(e)}")
            raise

    def process_gps_data(self, latitude: float, longitude: float, altitude: float,
                        accuracy: float, num_satellites: int, hdop: float,
                        timestamp: float) -> Dict[str, Any]:
        """
        Process GPS data with correction.

        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in meters
            accuracy: Position accuracy
            num_satellites: Number of satellites
            hdop: Horizontal dilution of precision
            timestamp: Measurement timestamp

        Returns:
            Dictionary containing processed GPS data
        """
        try:
            # Correct and validate position
            position, is_valid = self.gps_corrector.correct_position(
                latitude, longitude, altitude, accuracy, num_satellites, hdop, timestamp
            )

            # Quality assessment
            quality_score = self._assess_gps_quality(
                accuracy, num_satellites, hdop, is_valid
            )

            return {
                'position': position,
                'valid': is_valid,
                'accuracy': accuracy,
                'num_satellites': num_satellites,
                'hdop': hdop,
                'quality_score': quality_score,
                'timestamp': timestamp
            }

        except Exception as e:
            logger.error(f"Error processing GPS data: {str(e)}")
            raise

    def process_multi_sensor_data(self, sensor_data_list: List[Tuple[str, Dict[str, Any], float]]) -> Dict[str, Any]:
        """
        Process and fuse multiple sensor data streams.

        Args:
            sensor_data_list: List of (sensor_id, data, timestamp) tuples

        Returns:
            Dictionary containing fused results
        """
        try:
            # Add all sensor data to fusion buffers
            for sensor_id, data, timestamp in sensor_data_list:
                self.multi_sensor_fusion.add_sensor_data(sensor_id, data, timestamp)

            # Use the most recent timestamp as reference
            if sensor_data_list:
                reference_timestamp = max(timestamp for _, _, timestamp in sensor_data_list)

                # Synchronize sensor data
                synchronized = self.multi_sensor_fusion.synchronize_sensors(reference_timestamp)

                # Perform fusion if we have IMU and GPS data
                if 'imu' in synchronized and 'gps' in synchronized:
                    fused = self.multi_sensor_fusion.fuse_imu_gps(
                        synchronized['imu'], synchronized['gps']
                    )
                    synchronized['fused'] = fused

                return synchronized

            return {}

        except Exception as e:
            logger.error(f"Error processing multi-sensor data: {str(e)}")
            raise

    def calibrate_imu(self, accel_data: np.ndarray, gyro_data: np.ndarray,
                      duration: float = 10.0) -> Dict[str, np.ndarray]:
        """
        Calibrate IMU sensors.

        Args:
            accel_data: Nx3 accelerometer measurements
            gyro_data: Nx3 gyroscope measurements
            duration: Duration of calibration data

        Returns:
            Calibration parameters
        """
        return self.imu_filter.calibrate(accel_data, gyro_data, duration)

    def _assess_imu_quality(self, raw_accel: np.ndarray, raw_gyro: np.ndarray,
                           filtered_accel: np.ndarray, filtered_gyro: np.ndarray) -> float:
        """Assess quality of IMU processing."""
        try:
            # Noise reduction factor
            accel_noise_reduction = np.std(raw_accel) / (np.std(filtered_accel) + 1e-7)
            gyro_noise_reduction = np.std(raw_gyro) / (np.std(filtered_gyro) + 1e-7)

            # Reasonable range check
            accel_magnitude = np.linalg.norm(filtered_accel)
            gyro_magnitude = np.linalg.norm(filtered_gyro)

            accel_reasonable = 1.0 if accel_magnitude < 50.0 else 0.5  # 50g max
            gyro_reasonable = 1.0 if gyro_magnitude < 35.0 else 0.5  # 2000 deg/s max

            # Combine metrics
            quality = (0.3 * min(accel_noise_reduction, 10.0) / 10.0 +
                      0.3 * min(gyro_noise_reduction, 10.0) / 10.0 +
                      0.2 * accel_reasonable +
                      0.2 * gyro_reasonable)

            return float(np.clip(quality, 0.0, 1.0))

        except Exception:
            return 0.5

    def _assess_gps_quality(self, accuracy: float, num_satellites: int,
                           hdop: float, is_valid: bool) -> float:
        """Assess quality of GPS data."""
        try:
            if not is_valid:
                return 0.0

            # Accuracy score (inverse relationship)
            accuracy_score = 1.0 / (1.0 + accuracy / 10.0)

            # Satellite count score
            satellite_score = min(num_satellites / 12.0, 1.0)

            # DOP score (inverse relationship)
            dop_score = 1.0 / (1.0 + hdop)

            # Weighted combination
            quality = 0.4 * accuracy_score + 0.3 * satellite_score + 0.3 * dop_score

            return float(np.clip(quality, 0.0, 1.0))

        except Exception:
            return 0.0

    def set_config(self, config: SignalConfig):
        """Update processing configuration."""
        self.config = config

        # Reinitialize components with new config
        self.imu_filter = IMUFilter(config)
        self.gps_corrector = GPSCorrector(config)
        self.multi_sensor_fusion = MultiSensorFusion(config)