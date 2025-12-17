#!/usr/bin/env python3
"""
Data Preprocessing Demo for World Model

This script demonstrates the usage of all preprocessing modules:
- Point cloud processing
- Image processing
- Signal processing
- Quality assessment

Usage:
    python preprocessing_demo.py [--demo-type all|pointcloud|image|signal|quality]
"""

import sys
import os
import time
import argparse
import logging
from typing import Dict, Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import preprocessing modules
from brain.cognitive.world_model.pointcloud_processor import (
    PointCloudProcessor, PointCloudConfig
)
from brain.cognitive.world_model.image_processor import (
    ImageProcessor, ImageConfig
)
from brain.cognitive.world_model.signal_processor import (
    SignalProcessor, SignalConfig
)
from brain.cognitive.world_model.quality_assessor import (
    DataQualityAssessor, QualityConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data():
    """Generate synthetic test data for demonstration."""
    logger.info("Generating synthetic test data...")

    # Generate point cloud
    np.random.seed(42)
    point_cloud = np.random.randn(5000, 3)

    # Add some structure to the point cloud
    theta = np.random.uniform(0, 2*np.pi, 1000)
    r = np.random.uniform(0, 5, 1000)
    point_cloud[:1000, 0] = r * np.cos(theta)
    point_cloud[:1000, 1] = r * np.sin(theta)
    point_cloud[:1000, 2] = np.random.randn(1000) * 0.1

    # Generate image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add some structure to the image
    # Create a gradient
    for i in range(480):
        for j in range(640):
            image[i, j] = [
                int(255 * (i / 480)),
                int(255 * (j / 640)),
                int(255 * ((i + j) / (480 + 640)))
            ]

    # Generate IMU data
    duration = 10.0  # seconds
    sample_rate = 100.0  # Hz
    num_samples = int(duration * sample_rate)

    t = np.linspace(0, duration, num_samples)

    # Simulate motion
    accel = np.zeros((num_samples, 3))
    accel[:, 0] = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Sinusoidal acceleration
    accel[:, 1] = 0.05 * np.cos(2 * np.pi * 0.3 * t)
    accel[:, 2] = 9.81 + 0.02 * np.sin(2 * np.pi * 1.0 * t)  # Gravity + vibration

    gyro = np.zeros((num_samples, 3))
    gyro[:, 0] = 0.01 * np.sin(2 * np.pi * 0.2 * t)  # Angular velocity
    gyro[:, 1] = 0.005 * np.cos(2 * np.pi * 0.1 * t)
    gyro[:, 2] = 0.001 * np.sin(2 * np.pi * 0.05 * t)

    # Add noise
    accel += np.random.randn(*accel.shape) * 0.01
    gyro += np.random.randn(*gyro.shape) * 0.001

    return {
        'point_cloud': point_cloud,
        'image': image,
        'accel': accel,
        'gyro': gyro,
        'sample_rate': sample_rate
    }


def demo_point_cloud_processing(test_data: Dict[str, Any]):
    """Demonstrate point cloud processing."""
    logger.info("\n=== Point Cloud Processing Demo ===")

    # Configure point cloud processor
    config = PointCloudConfig(
        voxel_size=0.05,
        outlier_nb_neighbors=20,
        outlier_std_ratio=2.0,
        use_gpu=False  # Set to True if GPU is available
    )
    processor = PointCloudProcessor(config)

    # Process point cloud
    logger.info("Processing point cloud...")
    start_time = time.time()
    result = processor.process(test_data['point_cloud'])
    processing_time = time.time() - start_time

    logger.info(f"Point cloud processing completed in {processing_time:.3f} seconds")
    logger.info(f"Original points: {result['original_count']}")
    logger.info(f"Processed points: {result['final_count']}")
    logger.info(f"Quality score: {result['quality_score']:.3f}")
    logger.info(f"Features extracted: {list(result['features'].keys())}")

    # Display some statistics
    if 'density' in result['features']:
        density = result['features']['density']
        logger.info(f"Point density: {np.mean(density):.3e} points/m³")

    return result


def demo_image_processing(test_data: Dict[str, Any]):
    """Demonstrate image processing."""
    logger.info("\n=== Image Processing Demo ===")

    # Configure image processor
    config = ImageConfig(
        brightness_factor=1.2,
        contrast_factor=1.1,
        detection_confidence_threshold=0.5,
        use_gpu=False  # Set to True if GPU is available
    )
    processor = ImageProcessor(config)

    # Process image
    logger.info("Processing image...")
    start_time = time.time()
    result = processor.process(test_data['image'])
    processing_time = time.time() - start_time

    logger.info(f"Image processing completed in {processing_time:.3f} seconds")
    logger.info(f"Image shape: {result['image_shape']}")
    logger.info(f"Quality score: {result['quality_score']:.3f}")

    # Display detection results
    detections = result['detections']
    logger.info(f"Objects detected: {detections['num_detections']}")

    # Display feature statistics
    features = result['features']
    if 'sift' in features:
        logger.info(f"SIFT keypoints extracted: {features['sift']['num_keypoints']}")

    return result


def demo_signal_processing(test_data: Dict[str, Any]):
    """Demonstrate signal processing."""
    logger.info("\n=== Signal Processing Demo ===")

    # Configure signal processor
    config = SignalConfig(
        imu_complementary_alpha=0.98,
        gps_outlier_threshold=3.0,
        processing_frequency=100.0
    )
    processor = SignalProcessor(config)

    # Process IMU data
    logger.info("Processing IMU data...")
    start_time = time.time()

    # Process a single sample
    imu_result = processor.process_imu_data(
        accel=test_data['accel'][0],
        gyro=test_data['gyro'][0],
        timestamp=time.time()
    )

    # Process GPS data (simulated)
    gps_result = processor.process_gps_data(
        latitude=37.7749,
        longitude=-122.4194,
        altitude=100.0,
        accuracy=5.0,
        num_satellites=8,
        hdop=1.0,
        timestamp=time.time()
    )

    processing_time = time.time() - start_time

    logger.info(f"Signal processing completed in {processing_time:.3f} seconds")
    logger.info(f"IMU quality score: {imu_result['quality_score']:.3f}")
    logger.info(f"GPS valid: {gps_result['valid']}")
    logger.info(f"GPS quality score: {gps_result['quality_score']:.3f}")

    # Multi-sensor fusion
    sensor_data = [
        ('imu', imu_result, time.time()),
        ('gps', gps_result, time.time())
    ]
    fused_result = processor.process_multi_sensor_data(sensor_data)

    logger.info(f"Fused data available: {list(fused_result.keys())}")

    return {
        'imu': imu_result,
        'gps': gps_result,
        'fused': fused_result
    }


def demo_quality_assessment(test_data: Dict[str, Any], processing_results: Dict[str, Any]):
    """Demonstrate quality assessment."""
    logger.info("\n=== Quality Assessment Demo ===")

    # Configure quality assessor
    config = QualityConfig(
        completeness_weight=0.3,
        consistency_weight=0.25,
        accuracy_weight=0.25,
        timeliness_weight=0.2
    )
    assessor = DataQualityAssessor(config)

    # Assess point cloud quality
    if 'pointcloud' in processing_results:
        logger.info("Assessing point cloud quality...")
        pc_quality = assessor.assess_point_cloud_quality(
            processing_results['pointcloud']['points'],
            time.time()
        )
        logger.info(f"Point cloud quality: {pc_quality['overall_score']:.3f}")
        logger.info(f"Integrity valid: {pc_quality['integrity']['is_valid']}")
        logger.info(f"Anomaly detected: {pc_quality['anomaly']['is_anomaly']}")

    # Assess image quality
    if 'image' in processing_results:
        logger.info("Assessing image quality...")
        img_quality = assessor.assess_image_quality(
            processing_results['image']['enhanced_image'],
            time.time()
        )
        logger.info(f"Image quality: {img_quality['overall_score']:.3f}")
        logger.info(f"Resolution score: {img_quality['quality_metrics']['resolution']:.3f}")
        logger.info(f"Contrast score: {img_quality['quality_metrics']['contrast']:.3f}")

    # Assess signal quality
    if 'signal' in processing_results:
        logger.info("Assessing signal quality...")
        signal_quality = assessor.assess_signal_quality(
            test_data['accel'],
            test_data['sample_rate'],
            time.time()
        )
        logger.info(f"Signal quality: {signal_quality['overall_score']:.3f}")
        logger.info(f"SNR score: {signal_quality['quality_metrics']['snr']:.3f}")
        logger.info(f"Stability score: {signal_quality['quality_metrics']['stability']:.3f}")

    # Train anomaly models
    logger.info("Training anomaly detection models...")
    training_data = {
        'pointcloud': [test_data['point_cloud'], test_data['point_cloud'][:2500]],
        'image': [test_data['image'], test_data['image'][:240, :320]]
    }
    assessor.train_anomaly_models(training_data)
    logger.info("Anomaly models trained successfully")


def demo_performance_analysis(test_data: Dict[str, Any]):
    """Demonstrate performance analysis."""
    logger.info("\n=== Performance Analysis Demo ===")

    # Point cloud performance
    config = PointCloudConfig(use_gpu=False)
    processor = PointCloudProcessor(config)

    logger.info("Testing point cloud processing performance...")
    sizes = [1000, 5000, 10000, 50000]

    for size in sizes:
        pc_data = test_data['point_cloud'][:size]
        start_time = time.time()
        result = processor.process(pc_data)
        processing_time = time.time() - start_time

        points_per_second = size / processing_time
        logger.info(f"Size: {size:5d} points, Time: {processing_time:6.3f}s, "
                   f"Rate: {points_per_second:8.0f} points/s")

    # Image performance
    config = ImageConfig(use_gpu=False)
    processor = ImageProcessor(config)

    logger.info("\nTesting image processing performance...")
    with patch.object(processor.detector, 'detect'), \
         patch.object(processor.segmentator, 'segment'):

        resolutions = [(320, 240), (640, 480), (1280, 720), (1920, 1080)]

        for h, w in resolutions:
            img_data = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            start_time = time.time()
            result = processor.process(img_data)
            processing_time = time.time() - start_time

            pixels = h * w
            pixels_per_second = pixels / processing_time
            logger.info(f"Size: {w:4d}x{h:4d}, Time: {processing_time:6.3f}s, "
                       f"Rate: {pixels_per_second:8.0f} pixels/s")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Data Preprocessing Demo")
    parser.add_argument(
        '--demo-type',
        choices=['all', 'pointcloud', 'image', 'signal', 'quality', 'performance'],
        default='all',
        help='Type of demonstration to run'
    )
    args = parser.parse_args()

    logger.info("Starting Data Preprocessing Demo")
    logger.info("=" * 50)

    # Generate test data
    test_data = generate_test_data()
    logger.info(f"Generated test data: {len(test_data['point_cloud'])} points, "
               f"{test_data['image'].shape} image, "
               f"{len(test_data['accel'])} signal samples")

    # Store processing results
    processing_results = {}

    # Run demonstrations based on selection
    if args.demo_type in ['all', 'pointcloud']:
        processing_results['pointcloud'] = demo_point_cloud_processing(test_data)

    if args.demo_type in ['all', 'image']:
        processing_results['image'] = demo_image_processing(test_data)

    if args.demo_type in ['all', 'signal']:
        processing_results['signal'] = demo_signal_processing(test_data)

    if args.demo_type in ['all', 'quality']:
        demo_quality_assessment(test_data, processing_results)

    if args.demo_type in ['all', 'performance']:
        demo_performance_analysis(test_data)

    logger.info("\n" + "=" * 50)
    logger.info("Demo completed successfully!")


if __name__ == '__main__':
    # Import for patching in performance demo
    try:
        from unittest.mock import patch
    except ImportError:
        patch = None

    main()