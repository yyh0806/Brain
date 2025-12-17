# Data Preprocessing Modules for World Model

This directory contains comprehensive data preprocessing modules for the World Model fusion architecture. The modules provide robust processing capabilities for point cloud, image, and signal data from multiple sensors.

## Branch Information
- Work Tree: preprocessing-dev
- Branch: preprocessing-dev
- Base Branch: master

## Module Responsibilities
Data preprocessing module development:
- Point cloud preprocessing algorithms
- Image preprocessing algorithms
- Signal filtering processing
- Data quality assessment

## Overview

The preprocessing pipeline consists of four main modules:

1. **Point Cloud Processing** - 3D LiDAR/radar data processing
2. **Image Processing** - Visual data enhancement and analysis
3. **Signal Processing** - IMU, GPS, and multi-sensor fusion
4. **Quality Assessment** - Data integrity and quality validation

## Architecture

```
brain/cognitive/world_model/
├── pointcloud_processor.py    # 3D point cloud processing
├── image_processor.py         # 2D image processing
├── signal_processor.py        # 1D signal processing
└── quality_assessor.py        # Data quality assessment
```

## Features

### Point Cloud Processing (`pointcloud_processor.py`)

- **Voxel Grid Downsampling** - Efficient point cloud reduction
- **Statistical Outlier Removal** - Noise filtering
- **ICP Registration** - Point cloud alignment
- **Feature Extraction** - Geometric and local descriptors
- **GPU Acceleration** - CUDA support when available
- **Batch Processing** - Handle multiple point clouds efficiently

```python
from brain.cognitive.world_model.pointcloud_processor import PointCloudProcessor

processor = PointCloudProcessor()
result = processor.process(point_cloud_data)
print(f"Processed {result['original_count']} -> {result['final_count']} points")
print(f"Quality score: {result['quality_score']:.3f}")
```

### Image Processing (`image_processor.py`)

- **Image Enhancement** - Brightness, contrast, saturation, sharpness
- **YOLO Object Detection** - State-of-the-art object detection
- **Semantic Segmentation** - DeepLabV3 pixel-level segmentation
- **Feature Extraction** - SIFT, HOG, color histograms, texture analysis
- **GPU Acceleration** - PyTorch CUDA support
- **Batch Processing** - Process multiple images in parallel

```python
from brain.cognitive.world_model.image_processor import ImageProcessor

processor = ImageProcessor()
result = processor.process(image_data)
print(f"Detected {result['detections']['num_detections']} objects")
print(f"Image quality: {result['quality_score']:.3f}")
```

### Signal Processing (`signal_processor.py`)

- **IMU Filtering** - Complementary filter, Kalman filtering
- **GPS Correction** - Outlier detection, Kalman smoothing
- **Sensor Fusion** - Multi-sensor data synchronization and fusion
- **Motion Integration** - Position and velocity estimation
- **Calibration** - Automated bias and scale factor estimation

```python
from brain.cognitive.world_model.signal_processor import SignalProcessor

processor = SignalProcessor()
imu_result = processor.process_imu_data(accel, gyro, timestamp)
gps_result = processor.process_gps_data(lat, lon, alt, accuracy, sats, hdop, timestamp)
fused = processor.process_multi_sensor_data([('imu', imu_result, timestamp), ('gps', gps_result, timestamp)])
```

### Quality Assessment (`quality_assessor.py`)

- **Data Integrity Checks** - Validate data format and completeness
- **Quality Scoring** - Multi-dimensional quality metrics
- **Anomaly Detection** - Statistical and ML-based anomaly detection
- **Temporal Consistency** - Time-series consistency validation
- **Recommendations** - Automated quality improvement suggestions

```python
from brain.cognitive.world_model.quality_assessor import DataQualityAssessor

assessor = DataQualityAssessor()
quality = assessor.assess_point_cloud_quality(point_cloud, timestamp)
print(f"Overall quality: {quality['overall_score']:.3f}")
print(f"Anomalies detected: {quality['anomaly']['is_anomaly']}")
```

## Installation

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For GPU acceleration, install CUDA-enabled versions:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Development Guidelines
1. Develop based on the latest code from the main branch
2. Follow the project's coding standards
3. Commit code promptly and write tests
4. Regularly sync updates from the main branch

## Testing Commands

### Unit Tests
```bash
# Run unit tests
python -m pytest tests/unit/test_preprocessing.py -v

# Run with coverage
python -m pytest tests/unit/test_preprocessing.py --cov=brain/cognitive/world_model --cov-report=html
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/

# Generate test coverage report
python -m pytest --cov=brain tests/
```

## Demonstration

Run the demo script to see all modules in action:

```bash
python examples/preprocessing_demo.py --demo-type all
```

Available demo types:
- `all` - Run all demonstrations
- `pointcloud` - Point cloud processing only
- `image` - Image processing only
- `signal` - Signal processing only
- `quality` - Quality assessment only
- `performance` - Performance benchmarks

## Performance Considerations

### GPU Acceleration

- Enable GPU processing for large datasets
- Automatically detects CUDA availability
- Falls back to CPU if GPU unavailable

```python
config = PointCloudConfig(use_gpu=True)  # Enable GPU acceleration
```

### Memory Optimization

- Process data in batches for large datasets
- Use appropriate voxel sizes for point clouds
- Configure image resolution based on requirements

## Commit Standards

Commit message format: `module: brief description`

Examples:
- `pointcloud: Implement voxel grid downsampling with GPU support`
- `image: Add YOLO object detection with configurable confidence threshold`
- `signal: Implement IMU complementary filter with automatic calibration`
- `quality: Add statistical anomaly detection using Isolation Forest`

---
Created: 2025-12-17
Updated: 2025-12-17
