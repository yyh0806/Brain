"""
Comprehensive Unit Tests for Data Preprocessing Modules

This module tests all components of the preprocessing pipeline including:
- Point cloud processing
- Image processing
- Signal processing
- Quality assessment
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brain.cognitive.world_model.pointcloud_processor import (
    PointCloudProcessor, PointCloudConfig, VoxelGridSampler,
    StatisticalOutlierRemover, ICPRegistration, FeatureExtractor
)
from brain.cognitive.world_model.image_processor import (
    ImageProcessor, ImageConfig, ImageEnhancer, YOLODetector,
    SemanticSegmentator, FeatureExtractor as ImageFeatureExtractor
)
from brain.cognitive.world_model.signal_processor import (
    SignalProcessor, SignalConfig, IMUFilter, GPSCorrector, MultiSensorFusion
)
from brain.cognitive.world_model.quality_assessor import (
    DataQualityAssessor, QualityConfig, DataIntegrityChecker,
    QualityScorer, AnomalyDetector
)


class TestPointCloudProcessor(unittest.TestCase):
    """Test point cloud processing components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PointCloudConfig(
            voxel_size=0.1,
            outlier_nb_neighbors=10,
            outlier_std_ratio=1.5,
            icp_max_distance=1.0,
            use_gpu=False
        )
        self.processor = PointCloudProcessor(self.config)

        # Generate test point cloud
        np.random.seed(42)
        self.test_points = np.random.randn(1000, 3)

    def test_voxel_grid_sampler(self):
        """Test voxel grid downsampling."""
        sampler = VoxelGridSampler(voxel_size=0.1)

        # Test with valid input
        downsampled = sampler.sample(self.test_points)
        self.assertIsInstance(downsampled, np.ndarray)
        self.assertEqual(downsampled.shape[1], 3)
        self.assertLessEqual(len(downsampled), len(self.test_points))

        # Test with empty input
        empty_result = sampler.sample(np.array([]))
        self.assertEqual(len(empty_result), 0)

        # Test batch processing
        points_list = [self.test_points, self.test_points[:500]]
        batch_results = sampler.sample_batch(points_list)
        self.assertEqual(len(batch_results), 2)

    def test_statistical_outlier_remover(self):
        """Test statistical outlier removal."""
        remover = StatisticalOutlierRemover(nb_neighbors=10, std_ratio=1.5)

        # Add some outliers
        outlier_points = np.vstack([self.test_points, np.array([[100, 100, 100]])])

        # Test filtering
        filtered = remover.filter(outlier_points)
        self.assertIsInstance(filtered, np.ndarray)
        self.assertEqual(filtered.shape[1], 3)
        self.assertLessEqual(len(filtered), len(outlier_points))

        # Test with small point cloud
        small_cloud = np.random.randn(5, 3)
        small_result = remover.filter(small_cloud)
        self.assertEqual(len(small_result), len(small_cloud))

    def test_icp_registration(self):
        """Test ICP registration."""
        registrator = ICPRegistration(max_distance=1.0, max_iterations=10)

        # Create source and target point clouds
        source = np.random.randn(100, 3)
        target = source + np.random.randn(100, 3) * 0.01  # Small noise

        # Test registration
        registered = registrator.register(source, target)
        self.assertIsInstance(registered, np.ndarray)
        self.assertEqual(registered.shape, source.shape)

        # Test without target
        registered_no_target = registrator.register(source)
        self.assertEqual(registered_no_target.shape, source.shape)

        # Test transformation matrix retrieval
        transform = registrator.get_transformation_matrix()
        self.assertEqual(transform.shape, (4, 4))

    def test_feature_extractor(self):
        """Test point cloud feature extraction."""
        extractor = FeatureExtractor(feature_radius=0.1, normal_radius=0.05)

        # Test feature extraction
        features = extractor.extract_features(self.test_points)
        self.assertIsInstance(features, dict)

        # Check for expected feature types
        expected_features = ['normals', 'curvature', 'density', 'pca']
        for feature in expected_features:
            self.assertIn(feature, features)

        # Test with empty point cloud
        empty_features = extractor.extract_features(np.array([]))
        self.assertEqual(empty_features, {})

    def test_point_cloud_processor_pipeline(self):
        """Test complete point cloud processing pipeline."""
        result = self.processor.process(self.test_points)

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('points', result)
        self.assertIn('features', result)
        self.assertIn('quality_score', result)
        self.assertIn('processing_time', result)
        self.assertIn('original_count', result)
        self.assertIn('final_count', result)

        # Check processed points
        self.assertIsInstance(result['points'], np.ndarray)
        self.assertEqual(result['points'].shape[1], 3)
        self.assertEqual(result['original_count'], len(self.test_points))

        # Check quality score
        self.assertIsInstance(result['quality_score'], float)
        self.assertGreaterEqual(result['quality_score'], 0.0)
        self.assertLessEqual(result['quality_score'], 1.0)

    def test_point_cloud_processor_batch(self):
        """Test batch processing of point clouds."""
        points_list = [self.test_points, self.test_points[:500]]
        results = self.processor.process_batch(points_list)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('points', result)

    def test_point_cloud_quality_assessment(self):
        """Test point cloud quality assessment."""
        quality = self.processor._assess_quality(self.test_points, self.test_points)
        self.assertIsInstance(quality, float)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)


class TestImageProcessor(unittest.TestCase):
    """Test image processing components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ImageConfig(
            brightness_factor=1.1,
            contrast_factor=1.1,
            detection_confidence_threshold=0.5,
            use_gpu=False
        )
        self.processor = ImageProcessor(self.config)

        # Generate test image
        np.random.seed(42)
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_image_enhancer(self):
        """Test image enhancement."""
        enhancer = ImageEnhancer(self.config)

        # Test enhancement
        enhanced = enhancer.enhance(self.test_image)
        self.assertIsInstance(enhanced, np.ndarray)
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

        # Test histogram equalization
        hist_eq = enhancer.histogram_equalization(self.test_image)
        self.assertEqual(hist_eq.shape, self.test_image.shape)

        # Test CLAHE
        clahe = enhancer.adaptive_histogram_equalization(self.test_image)
        self.assertEqual(clahe.shape, self.test_image.shape)

    def test_yolo_detector(self):
        """Test YOLO object detection."""
        detector = YOLODetector(self.config)

        # Mock YOLO model to avoid dependency
        with patch.object(detector, 'load_model', return_value=True), \
             patch.object(detector, 'model') as mock_model:

            # Mock model output
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = [Mock()]
            mock_result.boxes.xyxy[0].cpu.return_value.numpy.return_value = np.array([10, 20, 100, 200])
            mock_result.boxes.conf = [Mock()]
            mock_result.boxes.conf[0].cpu.return_value.numpy.return_value = np.array([0.8])
            mock_result.boxes.cls = [Mock()]
            mock_result.boxes.cls[0].cpu.return_value.numpy.return_value = np.array([0])
            mock_model.return_value = [mock_result]

            # Test detection
            result = detector.detect(self.test_image)
            self.assertIsInstance(result, dict)
            self.assertIn('detections', result)
            self.assertIn('num_detections', result)

    def test_semantic_segmentator(self):
        """Test semantic segmentation."""
        segmentator = SemanticSegmentator(self.config)

        # Mock model to avoid dependency
        with patch.object(segmentator, 'load_model', return_value=True), \
             patch('torch.no_grad'), \
             patch.object(segmentator, 'model') as mock_model:

            # Mock model output
            mock_output = {'out': [Mock()]}
            mock_output['out'][0] = Mock()
            mock_output['out'][0].argmax.return_value.cpu.return_value.numpy.return_value = np.zeros((480, 640))
            mock_output['out'][0].softmax.return_value.max.return_value[0].cpu.return_value.numpy.return_value = np.ones((480, 640))
            mock_model.return_value = mock_output

            # Test segmentation
            result = segmentator.segment(self.test_image)
            self.assertIsInstance(result, dict)
            self.assertIn('segmentation_map', result)
            self.assertIn('confidence_map', result)

    def test_image_feature_extractor(self):
        """Test image feature extraction."""
        extractor = ImageFeatureExtractor(self.config)

        # Test feature extraction
        features = extractor.extract_features(self.test_image)
        self.assertIsInstance(features, dict)

        # Check for expected feature types
        expected_features = ['sift', 'color_histogram', 'texture', 'edges', 'statistics']
        for feature in expected_features:
            self.assertIn(feature, features)

        # Test SIFT features
        self.assertIn('keypoints', features['sift'])
        self.assertIn('num_keypoints', features['sift'])

    def test_image_processor_pipeline(self):
        """Test complete image processing pipeline."""
        # Mock YOLO and segmentation to avoid dependencies
        with patch.object(self.processor.detector, 'detect') as mock_detect, \
             patch.object(self.processor.segmentator, 'segment') as mock_segment:

            mock_detect.return_value = {'detections': [], 'num_detections': 0}
            mock_segment.return_value = {'segmentation_map': None, 'confidence_map': None}

            result = self.processor.process(self.test_image)

            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn('enhanced_image', result)
            self.assertIn('detections', result)
            self.assertIn('segmentation', result)
            self.assertIn('features', result)
            self.assertIn('quality_score', result)
            self.assertIn('processing_time', result)

            # Check enhanced image
            self.assertIsInstance(result['enhanced_image'], np.ndarray)
            self.assertEqual(result['enhanced_image'].shape, self.test_image.shape)

    def test_image_quality_assessment(self):
        """Test image quality assessment."""
        # Mock skimage to avoid dependency
        with patch('brain.cognitive.world_model.image_processor.structural_similarity') as mock_ssim:
            mock_ssim.return_value = 0.8

            quality = self.processor._assess_quality(self.test_image, self.test_image)
            self.assertIsInstance(quality, float)
            self.assertGreaterEqual(quality, 0.0)
            self.assertLessEqual(quality, 1.0)


class TestSignalProcessor(unittest.TestCase):
    """Test signal processing components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SignalConfig(
            imu_complementary_alpha=0.95,
            gps_outlier_threshold=2.0,
            processing_frequency=100.0
        )
        self.processor = SignalProcessor(self.config)

        # Generate test sensor data
        np.random.seed(42)
        self.test_accel = np.random.randn(100, 3) * 0.1  # 0.1 m/s^2 noise
        self.test_gyro = np.random.randn(100, 3) * 0.01  # 0.01 rad/s noise

    def test_imu_filter(self):
        """Test IMU filtering."""
        imu_filter = IMUFilter(self.config)

        # Test calibration
        accel_bias = imu_filter.calibrate(self.test_accel, self.test_gyro)
        self.assertIsInstance(accel_bias, dict)
        self.assertIn('accel_bias', accel_bias)
        self.assertIn('gyro_bias', accel_bias)

        # Test accelerometer filtering
        accel_filtered = imu_filter.filter_accelerometer(self.test_accel[0], 0.0)
        self.assertIsInstance(accel_filtered, np.ndarray)
        self.assertEqual(accel_filtered.shape, (3,))

        # Test gyroscope filtering
        gyro_filtered = imu_filter.filter_gyroscope(self.test_gyro[0], 0.0)
        self.assertIsInstance(gyro_filtered, np.ndarray)
        self.assertEqual(gyro_filtered.shape, (3,))

        # Test orientation update
        orientation = imu_filter.update_orientation(
            self.test_accel[0], self.test_gyro[0]
        )
        self.assertIsNotNone(orientation)

        # Test motion integration
        velocity, position = imu_filter.integrate_motion(self.test_accel[0], 0.01)
        self.assertIsInstance(velocity, np.ndarray)
        self.assertIsInstance(position, np.ndarray)

    def test_gps_corrector(self):
        """Test GPS correction."""
        gps_corrector = GPSCorrector(self.config)

        # Test position correction with valid data
        position, is_valid = gps_corrector.correct_position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            accuracy=5.0,
            num_satellites=8,
            hdop=1.0,
            timestamp=time.time()
        )
        self.assertIsInstance(position, np.ndarray)
        self.assertEqual(position.shape, (3,))
        self.assertIsInstance(is_valid, bool)

        # Test with poor quality GPS
        position_bad, is_valid_bad = gps_corrector.correct_position(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            accuracy=500.0,  # Poor accuracy
            num_satellites=2,  # Few satellites
            hdop=5.0,  # Poor DOP
            timestamp=time.time()
        )
        self.assertFalse(is_valid_bad)

    def test_multi_sensor_fusion(self):
        """Test multi-sensor data fusion."""
        fusion = MultiSensorFusion(self.config)

        # Test adding sensor data
        fusion.add_sensor_data('imu', {'accel': [1, 0, 9.8]}, time.time())
        fusion.add_sensor_data('gps', {'position': [0, 0, 0]}, time.time())

        # Test synchronization
        synchronized = fusion.synchronize_sensors(time.time())
        self.assertIsInstance(synchronized, dict)

        # Test IMU-GPS fusion
        imu_data = {'acceleration': np.array([0, 0, 9.8]), 'orientation': None}
        gps_data = {'position': np.array([1, 2, 3]), 'valid': True}
        fused = fusion.fuse_imu_gps(imu_data, gps_data)
        self.assertIsInstance(fused, dict)
        self.assertIn('position', fused)
        self.assertIn('orientation', fused)

    def test_signal_processor_imu_pipeline(self):
        """Test complete IMU processing pipeline."""
        result = self.processor.process_imu_data(
            accel=self.test_accel[0],
            gyro=self.test_gyro[0],
            timestamp=time.time()
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('acceleration', result)
        self.assertIn('angular_velocity', result)
        self.assertIn('orientation', result)
        self.assertIn('velocity', result)
        self.assertIn('position', result)
        self.assertIn('quality_score', result)

    def test_signal_processor_gps_pipeline(self):
        """Test complete GPS processing pipeline."""
        result = self.processor.process_gps_data(
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            accuracy=5.0,
            num_satellites=8,
            hdop=1.0,
            timestamp=time.time()
        )

        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('position', result)
        self.assertIn('valid', result)
        self.assertIn('quality_score', result)

    def test_multi_sensor_processing(self):
        """Test multi-sensor data processing."""
        sensor_data = [
            ('imu', {'accel': [1, 0, 9.8], 'gyro': [0, 0, 0]}, time.time()),
            ('gps', {'position': [1, 2, 3], 'valid': True}, time.time())
        ]

        result = self.processor.process_multi_sensor_data(sensor_data)
        self.assertIsInstance(result, dict)


class TestDataQualityAssessor(unittest.TestCase):
    """Test data quality assessment components."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = QualityConfig(
            point_cloud_min_points=100,
            image_min_resolution=(32, 32),
            signal_min_samples=10
        )
        self.assessor = DataQualityAssessor(self.config)

        # Generate test data
        np.random.seed(42)
        self.test_pointcloud = np.random.randn(1000, 3)
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_signal = np.random.randn(1000, 3)

    def test_data_integrity_checker(self):
        """Test data integrity checking."""
        checker = DataIntegrityChecker(self.config)

        # Test point cloud integrity
        pc_integrity = checker.check_point_cloud_integrity(self.test_pointcloud)
        self.assertIsInstance(pc_integrity, dict)
        self.assertIn('is_valid', pc_integrity)
        self.assertIn('errors', pc_integrity)
        self.assertIn('warnings', pc_integrity)
        self.assertIn('statistics', pc_integrity)

        # Test with None data
        none_integrity = checker.check_point_cloud_integrity(None)
        self.assertFalse(none_integrity['is_valid'])

        # Test image integrity
        img_integrity = checker.check_image_integrity(self.test_image)
        self.assertIsInstance(img_integrity, dict)
        self.assertIn('is_valid', img_integrity)

        # Test signal integrity
        signal_integrity = checker.check_signal_integrity(self.test_signal, 100.0)
        self.assertIsInstance(signal_integrity, dict)
        self.assertIn('is_valid', signal_integrity)

    def test_quality_scorer(self):
        """Test quality scoring."""
        scorer = QualityScorer(self.config)

        # Test overall score computation
        integrity = {'is_valid': True, 'errors': [], 'warnings': []}
        overall_score = scorer.compute_overall_score(
            integrity, 0.8, 0.9, 0.7
        )
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)

        # Test point cloud quality assessment
        pc_quality = scorer.assess_point_cloud_quality(
            self.test_pointcloud, integrity
        )
        self.assertIsInstance(pc_quality, dict)
        self.assertIn('overall', pc_quality)

        # Test image quality assessment
        img_quality = scorer.assess_image_quality(
            self.test_image, integrity
        )
        self.assertIsInstance(img_quality, dict)
        self.assertIn('overall', img_quality)

        # Test signal quality assessment
        signal_quality = scorer.assess_signal_quality(
            self.test_signal, integrity
        )
        self.assertIsInstance(signal_quality, dict)
        self.assertIn('overall', signal_quality)

    def test_anomaly_detector(self):
        """Test anomaly detection."""
        detector = AnomalyDetector(self.config)

        # Test feature extraction
        pc_features = detector._extract_point_cloud_features(self.test_pointcloud)
        self.assertIsInstance(pc_features, np.ndarray)
        self.assertEqual(len(pc_features), 10)

        img_features = detector._extract_image_features(self.test_image)
        self.assertIsInstance(img_features, np.ndarray)
        self.assertEqual(len(img_features), 10)

        # Test statistical anomaly detection
        anomaly_result = detector.detect_statistical_anomaly(
            np.random.randn(1000)
        )
        self.assertIsInstance(anomaly_result, dict)
        self.assertIn('is_anomaly', anomaly_result)
        self.assertIn('method', anomaly_result)

        # Test training models
        detector.train_point_cloud_anomaly_model([self.test_pointcloud])
        detector.train_image_anomaly_model([self.test_image])

    def test_comprehensive_quality_assessment(self):
        """Test comprehensive quality assessment."""
        # Test point cloud assessment
        pc_assessment = self.assessor.assess_point_cloud_quality(
            self.test_pointcloud, time.time()
        )
        self.assertIsInstance(pc_assessment, dict)
        self.assertIn('integrity', pc_assessment)
        self.assertIn('quality_metrics', pc_assessment)
        self.assertIn('anomaly', pc_assessment)
        self.assertIn('overall_score', pc_assessment)

        # Test image assessment
        img_assessment = self.assessor.assess_image_quality(
            self.test_image, time.time()
        )
        self.assertIsInstance(img_assessment, dict)
        self.assertIn('integrity', img_assessment)
        self.assertIn('quality_metrics', img_assessment)

        # Test signal assessment
        signal_assessment = self.assessor.assess_signal_quality(
            self.test_signal, 100.0, time.time()
        )
        self.assertIsInstance(signal_assessment, dict)
        self.assertIn('integrity', signal_assessment)
        self.assertIn('quality_metrics', signal_assessment)

    def test_training_anomaly_models(self):
        """Test training of anomaly detection models."""
        training_data = {
            'pointcloud': [self.test_pointcloud, self.test_pointcloud[:500]],
            'image': [self.test_image, self.test_image[:240, :320]]
        }

        # Should not raise any exceptions
        self.assessor.train_anomaly_models(training_data)


class TestIntegration(unittest.TestCase):
    """Integration tests for the preprocessing pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.pointcloud_processor = PointCloudProcessor(PointCloudConfig(use_gpu=False))
        self.image_processor = ImageProcessor(ImageConfig(use_gpu=False))
        self.signal_processor = SignalProcessor(SignalConfig())
        self.quality_assessor = DataQualityAssessor(QualityConfig())

        # Generate realistic test data
        np.random.seed(42)
        self.test_pointcloud = np.random.randn(5000, 3)
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_accel = np.random.randn(100, 3) * 0.1
        self.test_gyro = np.random.randn(100, 3) * 0.01

    def test_end_to_end_point_cloud_processing(self):
        """Test end-to-end point cloud processing."""
        # Process point cloud
        processed = self.pointcloud_processor.process(self.test_pointcloud)

        # Assess quality
        quality = self.quality_assessor.assess_point_cloud_quality(
            processed['points'], time.time()
        )

        # Check results
        self.assertIsInstance(processed, dict)
        self.assertIsInstance(quality, dict)
        self.assertGreater(processed['quality_score'], 0.0)
        self.assertGreater(quality['overall_score'], 0.0)

    def test_end_to_end_image_processing(self):
        """Test end-to-end image processing."""
        # Mock YOLO and segmentation for integration test
        with patch.object(self.image_processor.detector, 'detect') as mock_detect, \
             patch.object(self.image_processor.segmentator, 'segment') as mock_segment:

            mock_detect.return_value = {'detections': [], 'num_detections': 0}
            mock_segment.return_value = {'segmentation_map': None, 'confidence_map': None}

            # Process image
            processed = self.image_processor.process(self.test_image)

            # Assess quality
            quality = self.quality_assessor.assess_image_quality(
                processed['enhanced_image'], time.time()
            )

            # Check results
            self.assertIsInstance(processed, dict)
            self.assertIsInstance(quality, dict)
            self.assertGreater(processed['quality_score'], 0.0)
            self.assertGreater(quality['overall_score'], 0.0)

    def test_multi_sensor_data_flow(self):
        """Test multi-sensor data flow."""
        # Process IMU data
        imu_result = self.signal_processor.process_imu_data(
            self.test_accel[0], self.test_gyro[0], time.time()
        )

        # Process GPS data
        gps_result = self.signal_processor.process_gps_data(
            37.7749, -122.4194, 100.0, 5.0, 8, 1.0, time.time()
        )

        # Fuse sensors
        sensor_data = [
            ('imu', imu_result, time.time()),
            ('gps', gps_result, time.time())
        ]
        fused = self.signal_processor.process_multi_sensor_data(sensor_data)

        # Check results
        self.assertIsInstance(imu_result, dict)
        self.assertIsInstance(gps_result, dict)
        self.assertIsInstance(fused, dict)

    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        # Point cloud processing performance
        start_time = time.time()
        self.pointcloud_processor.process(self.test_pointcloud)
        pc_time = time.time() - start_time

        # Image processing performance
        with patch.object(self.image_processor.detector, 'detect'), \
             patch.object(self.image_processor.segmentator, 'segment'):
            start_time = time.time()
            self.image_processor.process(self.test_image)
            img_time = time.time() - start_time

        # Signal processing performance
        start_time = time.time()
        self.signal_processor.process_imu_data(
            self.test_accel[0], self.test_gyro[0], time.time()
        )
        signal_time = time.time() - start_time

        # Performance should be reasonable (less than 1 second each)
        self.assertLess(pc_time, 1.0)
        self.assertLess(img_time, 1.0)
        self.assertLess(signal_time, 0.1)

    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process large datasets
        large_pointcloud = np.random.randn(50000, 3)
        large_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        self.pointcloud_processor.process(large_pointcloud)

        with patch.object(self.image_processor.detector, 'detect'), \
             patch.object(self.image_processor.segmentator, 'segment'):
            self.image_processor.process(large_image)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB)
        self.assertLess(memory_increase, 500)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2)