"""
Data Quality Assessment Module for World Model

This module provides comprehensive data quality assessment including
integrity checks, quality scoring, anomaly detection, and validation.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import time
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality assessment parameters."""
    # Data integrity thresholds
    point_cloud_min_points: int = 100
    point_cloud_max_points: int = 1000000
    image_min_resolution: Tuple[int, int] = (32, 32)
    image_max_resolution: Tuple[int, int] = (8192, 8192)
    signal_min_samples: int = 10
    signal_max_samples: int = 10000

    # Quality scoring weights
    completeness_weight: float = 0.3
    consistency_weight: float = 0.25
    accuracy_weight: float = 0.25
    timeliness_weight: float = 0.2

    # Anomaly detection
    anomaly_contamination: float = 0.1
    anomaly_n_estimators: int = 100
    statistical_outlier_threshold: float = 3.0

    # Temporal consistency
    max_timestamp_gap: float = 1.0  # seconds
    max_position_change: float = 10.0  # meters
    max_orientation_change: float = np.pi  # radians

    # Sensor-specific thresholds
    lidar_max_range: float = 100.0  # meters
    lidar_min_intensity: float = 0.0
    lidar_max_intensity: float = 255.0
    gps_max_accuracy: float = 100.0  # meters
    imu_max_acceleration: float = 50.0  # m/s^2
    imu_max_angular_velocity: float = 35.0  # rad/s


class DataIntegrityChecker:
    """Check data integrity and completeness."""

    def __init__(self, config: QualityConfig):
        self.config = config

    def check_point_cloud_integrity(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        """
        Check point cloud data integrity.

        Args:
            point_cloud: Nx3 array of 3D points

        Returns:
            Dictionary containing integrity check results
        """
        integrity_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            if point_cloud is None:
                integrity_results['is_valid'] = False
                integrity_results['errors'].append("Point cloud is None")
                return integrity_results

            # Check array shape
            if len(point_cloud.shape) != 2 or point_cloud.shape[1] != 3:
                integrity_results['is_valid'] = False
                integrity_results['errors'].append(
                    f"Invalid point cloud shape: {point_cloud.shape}, expected (N, 3)"
                )
                return integrity_results

            # Check point count
            num_points = point_cloud.shape[0]
            integrity_results['statistics']['num_points'] = num_points

            if num_points < self.config.point_cloud_min_points:
                integrity_results['warnings'].append(
                    f"Low point count: {num_points} < {self.config.point_cloud_min_points}"
                )

            if num_points > self.config.point_cloud_max_points:
                integrity_results['warnings'].append(
                    f"High point count: {num_points} > {self.config.point_cloud_max_points}"
                )

            # Check for NaN or infinite values
            nan_mask = np.isnan(point_cloud)
            inf_mask = np.isinf(point_cloud)
            invalid_mask = nan_mask | inf_mask

            if np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)
                integrity_results['warnings'].append(
                    f"Found {num_invalid} invalid (NaN/Inf) points"
                )
                integrity_results['statistics']['invalid_points'] = num_invalid

            # Check coordinate ranges
            valid_points = point_cloud[~invalid_mask] if np.any(invalid_mask) else point_cloud

            if len(valid_points) > 0:
                ranges = np.ptp(valid_points, axis=0)
                integrity_results['statistics']['coordinate_ranges'] = ranges

                # Check for reasonable ranges
                if np.any(ranges > 10000):  # 10km max range
                    integrity_results['warnings'].append(
                        f"Unusually large coordinate range: {ranges}"
                    )

            # Check point density (if we have enough points)
            if num_points > 100:
                bounds = np.min(valid_points, axis=0), np.max(valid_points, axis=0)
                volume = np.prod(bounds[1] - bounds[0])
                density = num_points / max(volume, 1e-10)
                integrity_results['statistics']['density'] = density

                if density < 1e-6:  # Very low density
                    integrity_results['warnings'].append(
                        f"Very low point density: {density:.2e} points/m³"
                    )

        except Exception as e:
            integrity_results['is_valid'] = False
            integrity_results['errors'].append(f"Error checking point cloud integrity: {str(e)}")

        return integrity_results

    def check_image_integrity(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Check image data integrity.

        Args:
            image: HxWxC image array

        Returns:
            Dictionary containing integrity check results
        """
        integrity_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            if image is None:
                integrity_results['is_valid'] = False
                integrity_results['errors'].append("Image is None")
                return integrity_results

            # Check array dimensions
            if len(image.shape) not in [2, 3]:
                integrity_results['is_valid'] = False
                integrity_results['errors'].append(
                    f"Invalid image dimensions: {len(image.shape)}, expected 2 or 3"
                )
                return integrity_results

            # Check image resolution
            height, width = image.shape[:2]
            min_h, min_w = self.config.image_min_resolution
            max_h, max_w = self.config.image_max_resolution

            integrity_results['statistics']['resolution'] = (height, width)
            integrity_results['statistics']['channels'] = image.shape[2] if len(image.shape) == 3 else 1

            if height < min_h or width < min_w:
                integrity_results['warnings'].append(
                    f"Low resolution: {height}x{width} < {min_h}x{min_w}"
                )

            if height > max_h or width > max_w:
                integrity_results['warnings'].append(
                    f"High resolution: {height}x{width} > {max_h}x{max_w}"
                )

            # Check for NaN or infinite values
            nan_mask = np.isnan(image)
            inf_mask = np.isinf(image)
            invalid_mask = nan_mask | inf_mask

            if np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)
                integrity_results['warnings'].append(
                    f"Found {num_invalid} invalid (NaN/Inf) pixels"
                )
                integrity_results['statistics']['invalid_pixels'] = num_invalid

            # Check data type and range
            if image.dtype == np.uint8:
                min_val, max_val = 0, 255
            elif image.dtype == np.uint16:
                min_val, max_val = 0, 65535
            elif image.dtype in [np.float32, np.float64]:
                min_val, max_val = 0.0, 1.0
            else:
                integrity_results['warnings'].append(
                    f"Unusual image data type: {image.dtype}"
                )
                min_val, max_val = np.min(image), np.max(image)

            actual_min, actual_max = np.min(image), np.max(image)
            integrity_results['statistics']['value_range'] = (actual_min, actual_max)

            # Check if values are in expected range
            if actual_min < min_val or actual_max > max_val:
                integrity_results['warnings'].append(
                    f"Image values outside expected range [{min_val}, {max_val}]: "
                    f"actual range [{actual_min}, {actual_max}]"
                )

        except Exception as e:
            integrity_results['is_valid'] = False
            integrity_results['errors'].append(f"Error checking image integrity: {str(e)}")

        return integrity_results

    def check_signal_integrity(self, signal_data: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """
        Check signal data integrity.

        Args:
            signal_data: Signal data array
            sample_rate: Sampling rate in Hz

        Returns:
            Dictionary containing integrity check results
        """
        integrity_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }

        try:
            if signal_data is None:
                integrity_results['is_valid'] = False
                integrity_results['errors'].append("Signal data is None")
                return integrity_results

            # Check array dimensions
            if len(signal_data.shape) != 1 and len(signal_data.shape) != 2:
                integrity_results['is_valid'] = False
                integrity_results['errors'].append(
                    f"Invalid signal dimensions: {signal_data.shape}"
                )
                return integrity_results

            # Check number of samples
            if len(signal_data.shape) == 1:
                num_samples = signal_data.shape[0]
                num_channels = 1
            else:
                num_samples, num_channels = signal_data.shape

            integrity_results['statistics']['num_samples'] = num_samples
            integrity_results['statistics']['num_channels'] = num_channels

            if num_samples < self.config.signal_min_samples:
                integrity_results['warnings'].append(
                    f"Low sample count: {num_samples} < {self.config.signal_min_samples}"
                )

            if num_samples > self.config.signal_max_samples:
                integrity_results['warnings'].append(
                    f"High sample count: {num_samples} > {self.config.signal_max_samples}"
                )

            # Check for NaN or infinite values
            nan_mask = np.isnan(signal_data)
            inf_mask = np.isinf(signal_data)
            invalid_mask = nan_mask | inf_mask

            if np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)
                integrity_results['warnings'].append(
                    f"Found {num_invalid} invalid (NaN/Inf) samples"
                )
                integrity_results['statistics']['invalid_samples'] = num_invalid

            # Check signal characteristics
            valid_data = signal_data[~np.any(invalid_mask, axis=-1)] if len(signal_data.shape) > 1 else signal_data[~invalid_mask]

            if len(valid_data) > 0:
                signal_mean = np.mean(valid_data, axis=0)
                signal_std = np.std(valid_data, axis=0)
                integrity_results['statistics']['mean'] = signal_mean
                integrity_results['statistics']['std'] = signal_std

                # Check for reasonable signal ranges based on sensor type
                if num_channels == 3:  # Likely IMU data
                    if np.any(np.abs(signal_mean) > 10):  # Large DC offset
                        integrity_results['warnings'].append(
                            f"Large DC offset detected: {signal_mean}"
                        )

                # Duration check
                duration = num_samples / sample_rate
                integrity_results['statistics']['duration'] = duration

                if duration > 3600:  # More than 1 hour
                    integrity_results['warnings'].append(
                        f"Long signal duration: {duration:.1f} seconds"
                    )

        except Exception as e:
            integrity_results['is_valid'] = False
            integrity_results['errors'].append(f"Error checking signal integrity: {str(e)}")

        return integrity_results


class QualityScorer:
    """Compute quality scores for sensor data."""

    def __init__(self, config: QualityConfig):
        self.config = config

    def compute_overall_score(self, integrity_results: Dict[str, Any],
                            consistency_score: float,
                            accuracy_score: float,
                            timeliness_score: float) -> float:
        """
        Compute overall quality score.

        Args:
            integrity_results: Results from integrity checking
            consistency_score: Temporal consistency score (0-1)
            accuracy_score: Accuracy score (0-1)
            timeliness_score: Timeliness score (0-1)

        Returns:
            Overall quality score (0-1)
        """
        try:
            # Completeness score based on integrity
            completeness_score = 1.0
            if integrity_results['errors']:
                completeness_score = 0.0
            elif integrity_results['warnings']:
                # Reduce score based on number of warnings
                completeness_score = max(0.5, 1.0 - len(integrity_results['warnings']) * 0.1)

            # Weighted combination
            overall_score = (
                self.config.completeness_weight * completeness_score +
                self.config.consistency_weight * consistency_score +
                self.config.accuracy_weight * accuracy_score +
                self.config.timeliness_weight * timeliness_score
            )

            return float(np.clip(overall_score, 0.0, 1.0))

        except Exception as e:
            logger.error(f"Error computing overall score: {str(e)}")
            return 0.5

    def assess_point_cloud_quality(self, point_cloud: np.ndarray,
                                 integrity_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess point cloud quality.

        Args:
            point_cloud: Point cloud data
            integrity_results: Integrity check results

        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}

        try:
            if point_cloud is None or len(point_cloud) == 0:
                return {'overall': 0.0, 'completeness': 0.0, 'density': 0.0, 'coverage': 0.0}

            # Point density quality
            num_points = len(point_cloud)
            density_score = min(num_points / 10000.0, 1.0)  # 10k points for perfect score
            quality_metrics['density'] = density_score

            # Spatial coverage quality
            if 'coordinate_ranges' in integrity_results['statistics']:
                ranges = integrity_results['statistics']['coordinate_ranges']
                coverage_score = 1.0 - np.exp(-np.mean(ranges) / 100.0)  # 100m for good coverage
                quality_metrics['coverage'] = coverage_score

            # Completeness based on integrity
            completeness_score = 1.0 if integrity_results['is_valid'] else 0.5
            if integrity_results['warnings']:
                completeness_score *= (1.0 - len(integrity_results['warnings']) * 0.1)
            quality_metrics['completeness'] = completeness_score

            # Overall quality
            quality_metrics['overall'] = (
                0.4 * completeness_score +
                0.3 * density_score +
                0.3 * coverage_score
            )

        except Exception as e:
            logger.error(f"Error assessing point cloud quality: {str(e)}")
            quality_metrics = {'overall': 0.0}

        return quality_metrics

    def assess_image_quality(self, image: np.ndarray,
                           integrity_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess image quality.

        Args:
            image: Image data
            integrity_results: Integrity check results

        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}

        try:
            if image is None or image.size == 0:
                return {'overall': 0.0, 'completeness': 0.0, 'resolution': 0.0, 'contrast': 0.0}

            # Resolution quality
            height, width = image.shape[:2]
            resolution_score = min((height * width) / (1920 * 1080), 1.0)  # Full HD as reference
            quality_metrics['resolution'] = resolution_score

            # Contrast quality (using standard deviation)
            if len(image.shape) == 3:
                # Convert to grayscale for contrast calculation
                gray = np.mean(image, axis=2)
            else:
                gray = image

            contrast = np.std(gray)
            contrast_score = min(contrast / 64.0, 1.0)  # 64 std for good contrast
            quality_metrics['contrast'] = contrast_score

            # Completeness based on integrity
            completeness_score = 1.0 if integrity_results['is_valid'] else 0.5
            if integrity_results['warnings']:
                completeness_score *= (1.0 - len(integrity_results['warnings']) * 0.1)
            quality_metrics['completeness'] = completeness_score

            # Overall quality
            quality_metrics['overall'] = (
                0.4 * completeness_score +
                0.3 * resolution_score +
                0.3 * contrast_score
            )

        except Exception as e:
            logger.error(f"Error assessing image quality: {str(e)}")
            quality_metrics = {'overall': 0.0}

        return quality_metrics

    def assess_signal_quality(self, signal_data: np.ndarray,
                            integrity_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess signal quality.

        Args:
            signal_data: Signal data
            integrity_results: Integrity check results

        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}

        try:
            if signal_data is None or len(signal_data) == 0:
                return {'overall': 0.0, 'completeness': 0.0, 'snr': 0.0, 'stability': 0.0}

            # Signal-to-noise ratio (simplified)
            valid_data = signal_data[~np.isnan(signal_data) & ~np.isinf(signal_data)]
            if len(valid_data) > 0:
                signal_power = np.mean(valid_data ** 2)
                noise_estimate = np.var(np.diff(valid_data))  # High-frequency content as noise
                snr = signal_power / (noise_estimate + 1e-10)
                snr_score = min(snr / 100.0, 1.0)  # 20dB SNR for good score
            else:
                snr_score = 0.0
            quality_metrics['snr'] = snr_score

            # Signal stability (inverse of variance)
            if len(valid_data) > 0:
                stability_score = 1.0 / (1.0 + np.var(valid_data))
            else:
                stability_score = 0.0
            quality_metrics['stability'] = stability_score

            # Completeness based on integrity
            completeness_score = 1.0 if integrity_results['is_valid'] else 0.5
            if integrity_results['warnings']:
                completeness_score *= (1.0 - len(integrity_results['warnings']) * 0.1)
            quality_metrics['completeness'] = completeness_score

            # Overall quality
            quality_metrics['overall'] = (
                0.4 * completeness_score +
                0.3 * snr_score +
                0.3 * stability_score
            )

        except Exception as e:
            logger.error(f"Error assessing signal quality: {str(e)}")
            quality_metrics = {'overall': 0.0}

        return quality_metrics


class AnomalyDetector:
    """Detect anomalies in sensor data."""

    def __init__(self, config: QualityConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.is_trained = False

    def train_point_cloud_anomaly_model(self, point_clouds: List[np.ndarray]):
        """
        Train anomaly detection model for point clouds.

        Args:
            point_clouds: List of point cloud arrays
        """
        try:
            # Extract features from point clouds
            features = []
            for pc in point_clouds:
                if pc is not None and len(pc) > 0:
                    feature_vector = self._extract_point_cloud_features(pc)
                    features.append(feature_vector)

            if len(features) < 2:
                logger.warning("Insufficient point cloud data for training anomaly detector")
                return

            features = np.array(features)

            # Normalize features
            self.scalers['pointcloud'] = StandardScaler()
            features_normalized = self.scalers['pointcloud'].fit_transform(features)

            # Train Isolation Forest
            self.models['pointcloud'] = IsolationForest(
                contamination=self.config.anomaly_contamination,
                n_estimators=self.config.anomaly_n_estimators,
                random_state=42
            )
            self.models['pointcloud'].fit(features_normalized)

            logger.info("Trained point cloud anomaly detection model")
            self.is_trained = True

        except Exception as e:
            logger.error(f"Error training point cloud anomaly model: {str(e)}")

    def train_image_anomaly_model(self, images: List[np.ndarray]):
        """
        Train anomaly detection model for images.

        Args:
            images: List of image arrays
        """
        try:
            # Extract features from images
            features = []
            for img in images:
                if img is not None and img.size > 0:
                    feature_vector = self._extract_image_features(img)
                    features.append(feature_vector)

            if len(features) < 2:
                logger.warning("Insufficient image data for training anomaly detector")
                return

            features = np.array(features)

            # Normalize features
            self.scalers['image'] = StandardScaler()
            features_normalized = self.scalers['image'].fit_transform(features)

            # Train Isolation Forest
            self.models['image'] = IsolationForest(
                contamination=self.config.anomaly_contamination,
                n_estimators=self.config.anomaly_n_estimators,
                random_state=42
            )
            self.models['image'].fit(features_normalized)

            logger.info("Trained image anomaly detection model")
            self.is_trained = True

        except Exception as e:
            logger.error(f"Error training image anomaly model: {str(e)}")

    def detect_point_cloud_anomaly(self, point_cloud: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in point cloud data.

        Args:
            point_cloud: Point cloud array

        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            if 'pointcloud' not in self.models:
                return {'is_anomaly': False, 'anomaly_score': 0.0, 'method': 'none'}

            # Extract features
            features = self._extract_point_cloud_features(point_cloud)
            features_normalized = self.scalers['pointcloud'].transform([features])

            # Predict anomaly
            anomaly_prediction = self.models['pointcloud'].predict(features_normalized)[0]
            anomaly_score = self.models['pointcloud'].decision_function(features_normalized)[0]

            return {
                'is_anomaly': anomaly_prediction == -1,
                'anomaly_score': float(-anomaly_score),  # Negative for anomalies
                'method': 'isolation_forest'
            }

        except Exception as e:
            logger.error(f"Error detecting point cloud anomaly: {str(e)}")
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'method': 'error'}

    def detect_image_anomaly(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in image data.

        Args:
            image: Image array

        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            if 'image' not in self.models:
                return {'is_anomaly': False, 'anomaly_score': 0.0, 'method': 'none'}

            # Extract features
            features = self._extract_image_features(image)
            features_normalized = self.scalers['image'].transform([features])

            # Predict anomaly
            anomaly_prediction = self.models['image'].predict(features_normalized)[0]
            anomaly_score = self.models['image'].decision_function(features_normalized)[0]

            return {
                'is_anomaly': anomaly_prediction == -1,
                'anomaly_score': float(-anomaly_score),  # Negative for anomalies
                'method': 'isolation_forest'
            }

        except Exception as e:
            logger.error(f"Error detecting image anomaly: {str(e)}")
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'method': 'error'}

    def detect_statistical_anomaly(self, data: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect statistical anomalies using z-score method.

        Args:
            data: Data array
            threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            if threshold is None:
                threshold = self.config.statistical_outlier_threshold

            # Calculate z-scores
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            anomaly_mask = z_scores > threshold

            return {
                'is_anomaly': np.any(anomaly_mask),
                'anomaly_count': np.sum(anomaly_mask),
                'anomaly_indices': np.where(anomaly_mask)[0].tolist(),
                'z_scores': z_scores.tolist(),
                'method': 'statistical_zscore'
            }

        except Exception as e:
            logger.error(f"Error detecting statistical anomaly: {str(e)}")
            return {'is_anomaly': False, 'method': 'error'}

    def _extract_point_cloud_features(self, point_cloud: np.ndarray) -> np.ndarray:
        """Extract features for point cloud anomaly detection."""
        try:
            if point_cloud is None or len(point_cloud) == 0:
                return np.zeros(10)

            # Basic statistics
            num_points = len(point_cloud)
            mean_coords = np.mean(point_cloud, axis=0)
            std_coords = np.std(point_cloud, axis=0)
            min_coords = np.min(point_cloud, axis=0)
            max_coords = np.max(point_cloud, axis=0)

            # Features: num_points, mean_xyz, std_xyz, range_xyz
            features = np.array([
                num_points,
                mean_coords[0], mean_coords[1], mean_coords[2],
                std_coords[0], std_coords[1], std_coords[2],
                max_coords[0] - min_coords[0],
                max_coords[1] - min_coords[1],
                max_coords[2] - min_coords[2]
            ])

            return features

        except Exception:
            return np.zeros(10)

    def _extract_image_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for image anomaly detection."""
        try:
            if image is None or image.size == 0:
                return np.zeros(10)

            # Basic statistics
            mean_intensity = np.mean(image)
            std_intensity = np.std(image)
            min_intensity = np.min(image)
            max_intensity = np.max(image)

            # Additional features
            height, width = image.shape[:2]
            num_channels = image.shape[2] if len(image.shape) == 3 else 1

            # Contrast and brightness
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image

            contrast = np.std(gray)
            brightness = np.mean(gray)

            # Features: height, width, channels, mean, std, min, max, contrast, brightness
            features = np.array([
                height, width, num_channels,
                mean_intensity, std_intensity,
                min_intensity, max_intensity,
                contrast, brightness, image.size
            ])

            return features

        except Exception:
            return np.zeros(10)


class DataQualityAssessor:
    """Main data quality assessment class."""

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()

        # Initialize components
        self.integrity_checker = DataIntegrityChecker(self.config)
        self.quality_scorer = QualityScorer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)

        # Historical data for temporal consistency
        self.data_history = {}

    def assess_point_cloud_quality(self, point_cloud: np.ndarray,
                                  timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Comprehensive quality assessment for point cloud data.

        Args:
            point_cloud: Point cloud data
            timestamp: Data timestamp

        Returns:
            Dictionary containing comprehensive quality assessment
        """
        try:
            # Integrity check
            integrity_results = self.integrity_checker.check_point_cloud_integrity(point_cloud)

            # Quality scoring
            quality_metrics = self.quality_scorer.assess_point_cloud_quality(
                point_cloud, integrity_results
            )

            # Anomaly detection
            anomaly_results = self.anomaly_detector.detect_point_cloud_anomaly(point_cloud)

            # Temporal consistency
            consistency_score = self._check_temporal_consistency(
                'pointcloud', point_cloud, timestamp
            )

            # Combine results
            assessment = {
                'integrity': integrity_results,
                'quality_metrics': quality_metrics,
                'anomaly': anomaly_results,
                'temporal_consistency': consistency_score,
                'overall_score': quality_metrics['overall'],
                'timestamp': timestamp,
                'recommendations': self._generate_recommendations(
                    integrity_results, quality_metrics, anomaly_results
                )
            }

            # Update history
            self._update_history('pointcloud', point_cloud, timestamp)

            return assessment

        except Exception as e:
            logger.error(f"Error assessing point cloud quality: {str(e)}")
            return {'error': str(e), 'overall_score': 0.0}

    def assess_image_quality(self, image: np.ndarray, timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Comprehensive quality assessment for image data.

        Args:
            image: Image data
            timestamp: Data timestamp

        Returns:
            Dictionary containing comprehensive quality assessment
        """
        try:
            # Integrity check
            integrity_results = self.integrity_checker.check_image_integrity(image)

            # Quality scoring
            quality_metrics = self.quality_scorer.assess_image_quality(
                image, integrity_results
            )

            # Anomaly detection
            anomaly_results = self.anomaly_detector.detect_image_anomaly(image)

            # Temporal consistency
            consistency_score = self._check_temporal_consistency(
                'image', image, timestamp
            )

            # Combine results
            assessment = {
                'integrity': integrity_results,
                'quality_metrics': quality_metrics,
                'anomaly': anomaly_results,
                'temporal_consistency': consistency_score,
                'overall_score': quality_metrics['overall'],
                'timestamp': timestamp,
                'recommendations': self._generate_recommendations(
                    integrity_results, quality_metrics, anomaly_results
                )
            }

            # Update history
            self._update_history('image', image, timestamp)

            return assessment

        except Exception as e:
            logger.error(f"Error assessing image quality: {str(e)}")
            return {'error': str(e), 'overall_score': 0.0}

    def assess_signal_quality(self, signal_data: np.ndarray, sample_rate: float,
                            timestamp: float = 0.0) -> Dict[str, Any]:
        """
        Comprehensive quality assessment for signal data.

        Args:
            signal_data: Signal data
            sample_rate: Sampling rate in Hz
            timestamp: Data timestamp

        Returns:
            Dictionary containing comprehensive quality assessment
        """
        try:
            # Integrity check
            integrity_results = self.integrity_checker.check_signal_integrity(
                signal_data, sample_rate
            )

            # Quality scoring
            quality_metrics = self.quality_scorer.assess_signal_quality(
                signal_data, integrity_results
            )

            # Anomaly detection
            anomaly_results = self.anomaly_detector.detect_statistical_anomaly(signal_data)

            # Temporal consistency
            consistency_score = self._check_temporal_consistency(
                'signal', signal_data, timestamp
            )

            # Combine results
            assessment = {
                'integrity': integrity_results,
                'quality_metrics': quality_metrics,
                'anomaly': anomaly_results,
                'temporal_consistency': consistency_score,
                'overall_score': quality_metrics['overall'],
                'timestamp': timestamp,
                'recommendations': self._generate_recommendations(
                    integrity_results, quality_metrics, anomaly_results
                )
            }

            # Update history
            self._update_history('signal', signal_data, timestamp)

            return assessment

        except Exception as e:
            logger.error(f"Error assessing signal quality: {str(e)}")
            return {'error': str(e), 'overall_score': 0.0}

    def train_anomaly_models(self, training_data: Dict[str, List[np.ndarray]]):
        """
        Train anomaly detection models.

        Args:
            training_data: Dictionary with sensor types as keys and data lists as values
        """
        try:
            for sensor_type, data_list in training_data.items():
                if sensor_type == 'pointcloud':
                    self.anomaly_detector.train_point_cloud_anomaly_model(data_list)
                elif sensor_type == 'image':
                    self.anomaly_detector.train_image_anomaly_model(data_list)

            logger.info("Completed anomaly model training")

        except Exception as e:
            logger.error(f"Error training anomaly models: {str(e)}")

    def _check_temporal_consistency(self, sensor_type: str, data: Any,
                                   timestamp: float) -> float:
        """Check temporal consistency with historical data."""
        try:
            if sensor_type not in self.data_history:
                return 1.0  # No history, assume consistent

            history = self.data_history[sensor_type]
            if len(history) == 0:
                return 1.0

            # Get most recent data
            last_entry = history[-1]
            last_timestamp = last_entry['timestamp']
            last_data = last_entry['data']

            # Check timestamp gap
            time_gap = timestamp - last_timestamp
            if time_gap > self.config.max_timestamp_gap:
                return 0.5  # Large gap, reduced consistency

            # Data-specific consistency checks
            if sensor_type == 'pointcloud' and isinstance(data, np.ndarray) and isinstance(last_data, np.ndarray):
                # Check for reasonable changes in point cloud
                if len(data) > 0 and len(last_data) > 0:
                    point_count_change = abs(len(data) - len(last_data)) / max(len(last_data), 1)
                    consistency_score = max(0.0, 1.0 - point_count_change)
                else:
                    consistency_score = 0.5
            elif sensor_type == 'image' and isinstance(data, np.ndarray) and isinstance(last_data, np.ndarray):
                # Check for reasonable changes in image statistics
                mean_diff = abs(np.mean(data) - np.mean(last_data))
                consistency_score = max(0.0, 1.0 - mean_diff / 255.0)
            else:
                consistency_score = 0.8  # Default for unknown types

            return consistency_score

        except Exception:
            return 0.5

    def _update_history(self, sensor_type: str, data: Any, timestamp: float):
        """Update historical data for temporal consistency checking."""
        try:
            if sensor_type not in self.data_history:
                self.data_history[sensor_type] = []

            # Add new entry
            self.data_history[sensor_type].append({
                'data': data,
                'timestamp': timestamp
            })

            # Keep only recent history (last 10 entries)
            if len(self.data_history[sensor_type]) > 10:
                self.data_history[sensor_type] = self.data_history[sensor_type][-10:]

        except Exception as e:
            logger.error(f"Error updating history: {str(e)}")

    def _generate_recommendations(self, integrity_results: Dict[str, Any],
                                quality_metrics: Dict[str, Any],
                                anomaly_results: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []

        try:
            # Based on integrity
            if not integrity_results['is_valid']:
                recommendations.append("Data validation failed - check sensor configuration")
            elif integrity_results['warnings']:
                recommendations.append("Address data warnings for improved quality")

            # Based on quality metrics
            if quality_metrics.get('overall', 0) < 0.5:
                recommendations.append("Low overall quality - consider sensor calibration")
            if quality_metrics.get('completeness', 0) < 0.5:
                recommendations.append("Incomplete data - check sensor connectivity")
            if quality_metrics.get('density', 0) < 0.3:
                recommendations.append("Low data density - adjust sensor settings")

            # Based on anomaly detection
            if anomaly_results.get('is_anomaly', False):
                recommendations.append("Anomalous data detected - investigate sensor environment")

        except Exception:
            pass

        return recommendations

    def set_config(self, config: QualityConfig):
        """Update assessment configuration."""
        self.config = config

        # Reinitialize components with new config
        self.integrity_checker = DataIntegrityChecker(config)
        self.quality_scorer = QualityScorer(config)
        self.anomaly_detector = AnomalyDetector(config)