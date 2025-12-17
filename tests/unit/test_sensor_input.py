# -*- coding: utf-8 -*-
"""
Unit Tests for Sensor Input Module

This module contains comprehensive unit tests for the sensor input
components of the Brain cognitive world model system.

Author: Brain Development Team
Date: 2025-12-17
"""

import unittest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from brain.cognitive.world_model.sensor_input_types import (
    SensorDataPacket,
    SensorType,
    PointCloudData,
    ImageData,
    IMUData,
    GPSData,
    WeatherData,
    CameraIntrinsics,
    SensorQuality,
    SensorMetadata,
    SensorDataBuffer,
    validate_sensor_data_quality,
)

from brain.cognitive.world_model.sensor_interface import (
    BaseSensor,
    SensorConfig,
    PointCloudSensor,
    ImageSensor,
    IMUSensor,
    GPSSensor,
    create_sensor,
)

from brain.cognitive.world_model.sensor_manager import (
    MultiSensorManager,
    SensorGroup,
    SyncMethod,
    SensorSyncStatus,
    SynchronizedDataPacket,
    DataQualityAssessment,
)

from brain.cognitive.world_model.data_converter import (
    DataConverter,
    DataFormat,
    ConversionResult,
    ConversionOptions,
    ROS2Converter,
    StandardFormatConverter,
    create_converter,
)


class TestSensorDataTypes(unittest.TestCase):
    """Test cases for sensor data types."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_points = np.random.rand(100, 3)
        self.test_intensity = np.random.rand(100)
        self.test_rgb = np.random.randint(0, 256, (100, 3))
        self.test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.test_depth = np.random.rand(480, 640) * 10.0

    def test_point_cloud_data_creation(self):
        """Test PointCloudData creation and validation."""
        pc_data = PointCloudData(
            points=self.test_points,
            intensity=self.test_intensity,
            rgb=self.test_rgb,
            timestamp=time.time(),
            frame_id="test_frame"
        )

        self.assertEqual(pc_data.point_count, 100)
        self.assertTrue(pc_data.has_intensity)
        self.assertTrue(pc_data.has_color)
        self.assertEqual(pc_data.points.shape, (100, 3))
        self.assertEqual(pc_data.intensity.shape, (100,))
        self.assertEqual(pc_data.rgb.shape, (100, 3))

    def test_point_cloud_data_transformation(self):
        """Test PointCloudData transformation."""
        pc_data = PointCloudData(points=self.test_points)

        # Create translation transformation
        transformation = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        ])

        transformed = pc_data.transform(transformation)

        # Check translation
        np.testing.assert_array_almost_equal(
            transformed.points[:, 0],
            pc_data.points[:, 0] + 1
        )
        np.testing.assert_array_almost_equal(
            transformed.points[:, 1],
            pc_data.points[:, 1] + 2
        )
        np.testing.assert_array_almost_equal(
            transformed.points[:, 2],
            pc_data.points[:, 2] + 3
        )

    def test_point_cloud_data_bounds(self):
        """Test PointCloudData bounds calculation."""
        pc_data = PointCloudData(points=self.test_points)
        min_bounds, max_bounds = pc_data.get_bounds()

        self.assertEqual(min_bounds.shape, (3,))
        self.assertEqual(max_bounds.shape, (3,))

        # Check that all points are within bounds
        for i in range(3):
            self.assertGreaterEqual(np.min(pc_data.points[:, i]), min_bounds[i])
            self.assertLessEqual(np.max(pc_data.points[:, i]), max_bounds[i])

    def test_image_data_creation(self):
        """Test ImageData creation and validation."""
        camera_params = CameraIntrinsics(
            fx=500.0, fy=500.0,
            cx=320.0, cy=240.0,
            width=640, height=480
        )

        img_data = ImageData(
            image=self.test_image,
            depth=self.test_depth,
            camera_params=camera_params,
            timestamp=time.time(),
            frame_id="camera_frame"
        )

        self.assertEqual(img_data.width, 640)
        self.assertEqual(img_data.height, 480)
        self.assertEqual(img_data.channels, 3)
        self.assertTrue(img_data.has_depth)
        self.assertIsNotNone(img_data.camera_params)

    def test_camera_intrinsics(self):
        """Test CameraIntrinsics validation and properties."""
        camera_params = CameraIntrinsics(
            fx=500.0, fy=500.0,
            cx=320.0, cy=240.0
        )

        # Test camera matrix
        camera_matrix = camera_params.camera_matrix
        self.assertEqual(camera_matrix.shape, (3, 3))
        self.assertEqual(camera_matrix[0, 0], 500.0)
        self.assertEqual(camera_matrix[1, 1], 500.0)
        self.assertEqual(camera_matrix[0, 2], 320.0)
        self.assertEqual(camera_matrix[1, 2], 240.0)

        # Test distortion coefficients
        dist_coeffs = camera_params.distortion_coeffs
        self.assertEqual(len(dist_coeffs), 5)

    def test_imu_data_creation(self):
        """Test IMUData creation and validation."""
        acceleration = np.array([1.0, 2.0, 3.0])
        angular_velocity = np.array([0.1, 0.2, 0.3])
        orientation = np.array([0.0, 0.0, 0.0, 1.0])

        imu_data = IMUData(
            linear_acceleration=acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
            timestamp=time.time(),
            frame_id="imu_frame"
        )

        np.testing.assert_array_equal(imu_data.linear_acceleration, acceleration)
        np.testing.assert_array_equal(imu_data.angular_velocity, angular_velocity)
        np.testing.assert_array_equal(imu_data.orientation, orientation)

        # Test magnitude calculation
        accel_mag, gyro_mag = imu_data.get_magnitude()
        expected_accel = np.linalg.norm(acceleration)
        expected_gyro = np.linalg.norm(angular_velocity)
        self.assertAlmostEqual(accel_mag, expected_accel, places=5)
        self.assertAlmostEqual(gyro_mag, expected_gyro, places=5)

    def test_gps_data_creation(self):
        """Test GPSData creation and validation."""
        gps_data = GPSData(
            latitude=39.9042,
            longitude=116.4074,
            altitude=50.0,
            velocity=np.array([1.0, 2.0, 0.0]),
            fix_type=3,
            satellites_used=8,
            timestamp=time.time(),
            frame_id="gps_frame"
        )

        self.assertEqual(gps_data.latitude, 39.9042)
        self.assertEqual(gps_data.longitude, 116.4074)
        self.assertEqual(gps_data.altitude, 50.0)
        self.assertEqual(gps_data.fix_type, 3)
        self.assertEqual(gps_data.satellites_used, 8)

        # Test UTM conversion
        x, y = gps_data.to_utm()
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)

    def test_weather_data_creation(self):
        """Test WeatherData creation and validation."""
        weather_data = WeatherData(
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25,
            wind_speed=5.0,
            wind_direction=180.0,
            visibility=10000.0,
            timestamp=time.time(),
            frame_id="weather_frame"
        )

        self.assertEqual(weather_data.temperature, 25.0)
        self.assertEqual(weather_data.humidity, 60.0)
        self.assertEqual(weather_data.pressure, 1013.25)
        self.assertEqual(weather_data.wind_speed, 5.0)
        self.assertEqual(weather_data.wind_direction, 180.0)

        # Test heat index calculation
        heat_index = weather_data.get_heat_index()
        self.assertIsInstance(heat_index, float)

    def test_sensor_data_packet_creation(self):
        """Test SensorDataPacket creation and validation."""
        pc_data = PointCloudData(points=self.test_points)
        packet = SensorDataPacket(
            sensor_id="test_sensor",
            sensor_type=SensorType.POINT_CLOUD,
            timestamp=time.time(),
            data=pc_data,
            quality_score=0.95,
            calibration_params={"test_param": "test_value"}
        )

        self.assertEqual(packet.sensor_id, "test_sensor")
        self.assertEqual(packet.sensor_type, SensorType.POINT_CLOUD)
        self.assertEqual(packet.quality_score, 0.95)
        self.assertIn("test_param", packet.calibration_params)

        # Test data integrity validation
        self.assertTrue(packet.validate_data_integrity())

        # Test staleness check
        self.assertFalse(packet.is_stale(max_age_seconds=1.0))

    def test_sensor_data_buffer(self):
        """Test SensorDataBuffer functionality."""
        buffer = SensorDataBuffer(max_size=5, max_age_seconds=1.0)

        # Add test packets
        packets = []
        for i in range(3):
            pc_data = PointCloudData(points=np.random.rand(10, 3))
            packet = SensorDataPacket(
                sensor_id="test_sensor",
                sensor_type=SensorType.POINT_CLOUD,
                timestamp=time.time() + i * 0.1,
                data=pc_data
            )
            packets.append(packet)
            buffer.add(packet)

        # Test retrieval
        latest = buffer.get_latest(1)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0].sensor_id, "test_sensor")

        # Test time range retrieval
        start_time = packets[0].timestamp
        end_time = packets[-1].timestamp
        range_packets = buffer.get_by_timerange(start_time, end_time)
        self.assertEqual(len(range_packets), 3)

        # Test buffer size
        self.assertEqual(buffer.size(), 3)

    def test_data_quality_validation(self):
        """Test sensor data quality validation."""
        # Test valid point cloud
        pc_data = PointCloudData(points=np.random.rand(100, 3))
        assessment = validate_sensor_data_quality(pc_data)
        self.assertEqual(assessment["quality"], SensorQuality.EXCELLENT)

        # Test empty point cloud
        empty_pc = PointCloudData(points=np.array([]).reshape(0, 3))
        assessment = validate_sensor_data_quality(empty_pc)
        self.assertEqual(assessment["quality"], SensorQuality.INVALID)

        # Test image data
        img_data = ImageData(image=self.test_image)
        assessment = validate_sensor_data_quality(img_data)
        self.assertEqual(assessment["quality"], SensorQuality.EXCELLENT)

        # Test IMU data
        imu_data = IMUData(
            linear_acceleration=np.array([1.0, 2.0, 3.0]),
            angular_velocity=np.array([0.1, 0.2, 0.3])
        )
        assessment = validate_sensor_data_quality(imu_data)
        self.assertEqual(assessment["quality"], SensorQuality.EXCELLENT)


class TestSensorInterfaces(unittest.TestCase):
    """Test cases for sensor interfaces."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SensorConfig(
            sensor_id="test_sensor",
            sensor_type=SensorType.POINT_CLOUD,
            update_rate=10.0,
            buffer_size=50
        )

    def test_sensor_config_creation(self):
        """Test SensorConfig creation."""
        config = SensorConfig(
            sensor_id="test",
            sensor_type=SensorType.IMAGE,
            update_rate=30.0
        )

        self.assertEqual(config.sensor_id, "test")
        self.assertEqual(config.sensor_type, SensorType.IMAGE)
        self.assertEqual(config.update_rate, 30.0)

    def test_point_cloud_sensor_creation(self):
        """Test PointCloudSensor creation."""
        sensor = PointCloudSensor(self.config)

        self.assertEqual(sensor.config.sensor_id, "test_sensor")
        self.assertEqual(sensor.config.sensor_type, SensorType.POINT_CLOUD)
        self.assertFalse(sensor.is_running)

    def test_point_cloud_sensor_lifecycle(self):
        """Test PointCloudSensor start/stop lifecycle."""
        sensor = PointCloudSensor(self.config)

        # Test start
        self.assertTrue(sensor.start())
        self.assertTrue(sensor.is_running)

        # Test stop
        sensor.stop()
        self.assertFalse(sensor.is_running)

    def test_point_cloud_sensor_data_acquisition(self):
        """Test PointCloudSensor data acquisition."""
        sensor = PointCloudSensor(self.config)

        # Mock the data acquisition method
        with patch.object(sensor, '_acquire_data') as mock_acquire:
            test_points = np.random.rand(100, 3)
            test_data = PointCloudData(points=test_points)
            mock_acquire.return_value = test_data

            # Start sensor and wait for data
            sensor.start()
            time.sleep(0.2)  # Wait for at least one acquisition cycle

            # Check that data was acquired
            latest_data = sensor.get_latest_data(1)
            self.assertGreater(len(latest_data), 0)

            sensor.stop()

    def test_image_sensor_creation(self):
        """Test ImageSensor creation."""
        camera_params = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        sensor = ImageSensor(self.config, camera_params)

        self.assertEqual(sensor.config.sensor_id, "test_sensor")
        self.assertEqual(sensor.camera_intrinsics, camera_params)

    def test_imu_sensor_creation(self):
        """Test IMUSensor creation."""
        sensor = IMUSensor(self.config)

        self.assertEqual(sensor.config.sensor_id, "test_sensor")
        self.assertTrue(sensor.gravity_compensation)
        self.assertTrue(sensor.bias_estimation)

    def test_gps_sensor_creation(self):
        """Test GPSSensor creation."""
        sensor = GPSSensor(self.config)

        self.assertEqual(sensor.config.sensor_id, "test_sensor")
        self.assertEqual(sensor.coordinate_system, "WGS84")
        self.assertTrue(sensor.enable_dgps)

    def test_sensor_factory(self):
        """Test sensor factory function."""
        # Test point cloud sensor creation
        pc_sensor = create_sensor(self.config)
        self.assertIsInstance(pc_sensor, PointCloudSensor)

        # Test image sensor creation
        img_config = SensorConfig(
            sensor_id="img_sensor",
            sensor_type=SensorType.IMAGE
        )
        img_sensor = create_sensor(img_config)
        self.assertIsInstance(img_sensor, ImageSensor)

        # Test IMU sensor creation
        imu_config = SensorConfig(
            sensor_id="imu_sensor",
            sensor_type=SensorType.IMU
        )
        imu_sensor = create_sensor(imu_config)
        self.assertIsInstance(imu_sensor, IMUSensor)

        # Test GPS sensor creation
        gps_config = SensorConfig(
            sensor_id="gps_sensor",
            sensor_type=SensorType.GPS
        )
        gps_sensor = create_sensor(gps_config)
        self.assertIsInstance(gps_sensor, GPSSensor)

        # Test unsupported sensor type
        invalid_config = SensorConfig(
            sensor_id="invalid_sensor",
            sensor_type=SensorType.WEATHER
        )
        with self.assertRaises(ValueError):
            create_sensor(invalid_config)

    def test_sensor_callbacks(self):
        """Test sensor callback functionality."""
        sensor = PointCloudSensor(self.config)
        callback_called = threading.Event()
        received_packet = None

        def test_callback(packet):
            nonlocal received_packet
            received_packet = packet
            callback_called.set()

        sensor.add_callback(test_callback)

        # Mock data acquisition
        with patch.object(sensor, '_acquire_data') as mock_acquire:
            test_points = np.random.rand(10, 3)
            test_data = PointCloudData(points=test_points)
            mock_acquire.return_value = test_data

            sensor.start()

            # Wait for callback
            self.assertTrue(callback_called.wait(timeout=1.0))
            self.assertIsNotNone(received_packet)
            self.assertEqual(received_packet.sensor_id, "test_sensor")

            sensor.stop()
            sensor.remove_callback(test_callback)

    def test_sensor_statistics(self):
        """Test sensor statistics reporting."""
        sensor = PointCloudSensor(self.config)

        # Mock data acquisition
        with patch.object(sensor, '_acquire_data') as mock_acquire:
            test_points = np.random.rand(10, 3)
            test_data = PointCloudData(points=test_points)
            mock_acquire.return_value = test_data

            sensor.start()
            time.sleep(0.2)  # Allow some data to be processed
            sensor.stop()

            stats = sensor.get_statistics()
            self.assertIn("sensor_id", stats)
            self.assertIn("packets_received", stats)
            self.assertIn("average_update_rate", stats)
            self.assertEqual(stats["sensor_id"], "test_sensor")


class TestSensorManager(unittest.TestCase):
    """Test cases for multi-sensor manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MultiSensorManager(max_sync_window=0.1)

        # Create test sensor configs
        self.pc_config = SensorConfig(
            sensor_id="pc_sensor",
            sensor_type=SensorType.POINT_CLOUD,
            update_rate=10.0
        )

        self.img_config = SensorConfig(
            sensor_id="img_sensor",
            sensor_type=SensorType.IMAGE,
            update_rate=10.0
        )

    def test_manager_initialization(self):
        """Test MultiSensorManager initialization."""
        self.assertEqual(len(self.manager.sensors), 0)
        self.assertEqual(len(self.manager.active_sensors), 0)
        self.assertEqual(len(self.manager.sensor_groups), 0)

    def test_sensor_addition(self):
        """Test adding sensors to manager."""
        sensor = PointCloudSensor(self.pc_config)

        self.assertTrue(self.manager.add_sensor(sensor, self.pc_config))
        self.assertEqual(len(self.manager.sensors), 1)
        self.assertIn("pc_sensor", self.manager.sensors)

        # Test duplicate addition
        self.assertFalse(self.manager.add_sensor(sensor, self.pc_config))

    def test_sensor_removal(self):
        """Test removing sensors from manager."""
        sensor = PointCloudSensor(self.pc_config)
        self.manager.add_sensor(sensor, self.pc_config)

        self.assertTrue(self.manager.remove_sensor("pc_sensor"))
        self.assertEqual(len(self.manager.sensors), 0)
        self.assertNotIn("pc_sensor", self.manager.sensors)

        # Test removal of non-existent sensor
        self.assertFalse(self.manager.remove_sensor("non_existent"))

    def test_sensor_group_creation(self):
        """Test creating sensor groups."""
        # Add sensors first
        pc_sensor = PointCloudSensor(self.pc_config)
        img_sensor = ImageSensor(self.img_config)
        self.manager.add_sensor(pc_sensor, self.pc_config)
        self.manager.add_sensor(img_sensor, self.img_config)

        # Create group
        self.assertTrue(self.manager.create_sensor_group(
            group_id="test_group",
            sensor_ids=["pc_sensor", "img_sensor"],
            sync_method=SyncMethod.TIMESTAMP_ALIGNMENT
        ))

        self.assertIn("test_group", self.manager.sensor_groups)
        group = self.manager.sensor_groups["test_group"]
        self.assertEqual(len(group.sensor_ids), 2)
        self.assertIn("pc_sensor", group.sensor_ids)
        self.assertIn("img_sensor", group.sensor_ids)

    def test_sensor_start_stop(self):
        """Test starting and stopping sensors."""
        sensor = PointCloudSensor(self.pc_config)
        self.manager.add_sensor(sensor, self.pc_config)

        # Test start
        self.assertTrue(self.manager.start_sensors(["pc_sensor"]))
        self.assertIn("pc_sensor", self.manager.active_sensors)

        # Test stop
        self.manager.stop_sensors(["pc_sensor"])
        self.assertNotIn("pc_sensor", self.manager.active_sensors)

    def test_sensor_quality_assessment(self):
        """Test sensor data quality assessment."""
        sensor = PointCloudSensor(self.pc_config)
        self.manager.add_sensor(sensor, self.pc_config)

        # Mock data generation
        with patch.object(sensor, '_acquire_data') as mock_acquire:
            test_points = np.random.rand(100, 3)
            test_data = PointCloudData(points=test_points)
            mock_acquire.return_value = test_data

            self.manager.start_sensors(["pc_sensor"])
            time.sleep(0.2)

            # Assess quality
            assessment = self.manager.assess_sensor_quality("pc_sensor")
            self.assertIsInstance(assessment, DataQualityAssessment)
            self.assertEqual(assessment.sensor_id, "pc_sensor")
            self.assertGreaterEqual(assessment.overall_quality, 0.0)
            self.assertLessEqual(assessment.overall_quality, 1.0)

            self.manager.stop_sensors(["pc_sensor"])

    def test_synchronization(self):
        """Test multi-sensor synchronization."""
        # Add sensors
        pc_sensor = PointCloudSensor(self.pc_config)
        img_sensor = ImageSensor(self.img_config)
        self.manager.add_sensor(pc_sensor, self.pc_config)
        self.manager.add_sensor(img_sensor, self.img_config)

        # Create group
        self.manager.create_sensor_group(
            group_id="sync_group",
            sensor_ids=["pc_sensor", "img_sensor"],
            sync_method=SyncMethod.TIMESTAMP_ALIGNMENT,
            sync_tolerance=0.05
        )

        # Mock data generation
        with patch.object(pc_sensor, '_acquire_data') as mock_pc, \
             patch.object(img_sensor, '_acquire_data') as mock_img:

            mock_pc.return_value = PointCloudData(points=np.random.rand(50, 3))

            test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            mock_img.return_value = ImageData(image=test_image)

            # Start sensors and synchronization
            self.manager.start_sensors()
            self.manager.start_synchronization()

            time.sleep(0.3)  # Allow synchronization to occur

            # Check for synchronized data
            sync_data = self.manager.get_synchronized_data("sync_group", 1)

            # Clean up
            self.manager.stop_sensors()
            self.manager.stop_synchronization()

    def test_manager_statistics(self):
        """Test manager statistics reporting."""
        stats = self.manager.get_manager_statistics()

        self.assertIn("runtime_seconds", stats)
        self.assertIn("total_sensors", stats)
        self.assertIn("active_sensors", stats)
        self.assertIn("sync_operations", stats)
        self.assertIn("sensor_status", stats)
        self.assertIn("group_status", stats)


class TestDataConverter(unittest.TestCase):
    """Test cases for data format converters."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_points = np.random.rand(10, 3)
        self.test_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        self.pc_data = PointCloudData(points=self.test_points)
        self.img_data = ImageData(image=self.test_image)

        self.pc_packet = SensorDataPacket(
            sensor_id="pc_sensor",
            sensor_type=SensorType.POINT_CLOUD,
            timestamp=time.time(),
            data=self.pc_data
        )

    def test_standard_format_converter_creation(self):
        """Test StandardFormatConverter creation."""
        converter = StandardFormatConverter()
        self.assertIsInstance(converter, DataConverter)

        supported_formats = converter.get_supported_formats()
        self.assertIn(DataFormat.JSON, supported_formats)
        self.assertIn(DataFormat.CSV, supported_formats)
        self.assertIn(DataFormat.BINARY, supported_formats)

    def test_json_conversion(self):
        """Test JSON format conversion."""
        converter = StandardFormatConverter()
        options = ConversionOptions(target_format=DataFormat.JSON)

        # Test sensor packet conversion
        result = converter.convert(self.pc_packet, options)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.format, DataFormat.JSON)

        # Verify JSON can be parsed
        import json
        parsed = json.loads(result.data)
        self.assertEqual(parsed["sensor_id"], "pc_sensor")
        self.assertEqual(parsed["sensor_type"], "point_cloud")

    def test_csv_conversion(self):
        """Test CSV format conversion."""
        converter = StandardFormatConverter()
        options = ConversionOptions(target_format=DataFormat.CSV)

        # Test point cloud conversion
        result = converter.convert(self.pc_data, options)
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, str)

        # Verify CSV format
        lines = result.data.strip().split('\n')
        self.assertGreater(len(lines), 1)  # Header + data
        self.assertIn('x,y,z', lines[0])

    def test_binary_conversion(self):
        """Test binary format conversion."""
        converter = StandardFormatConverter()
        options = ConversionOptions(target_format=DataFormat.BINARY)

        # Test point cloud conversion
        result = converter.convert(self.pc_data, options)
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, bytes)
        self.assertGreater(len(result.data), 0)

    def test_pcd_conversion(self):
        """Test PCD format conversion."""
        converter = StandardFormatConverter()
        options = ConversionOptions(target_format=DataFormat.PCD)

        # Test point cloud conversion
        result = converter.convert(self.pc_data, options)
        self.assertTrue(result.success)
        self.assertIsInstance(result.data, str)

        # Verify PCD format
        lines = result.data.strip().split('\n')
        self.assertIn("VERSION 0.7", lines[0])
        self.assertIn("FIELDS x y z", lines[2])

    def test_ros2_converter_creation(self):
        """Test ROS2Converter creation."""
        converter = ROS2Converter()
        self.assertIsInstance(converter, DataConverter)

        supported_formats = converter.get_supported_formats()
        self.assertIn(DataFormat.BRAIN_NATIVE, supported_formats)
        self.assertIn(DataFormat.ROS2, supported_formats)

    @patch('brain.cognitive.world_model.data_converter.ROS2Converter._check_ros2_availability')
    def test_ros2_conversion_without_ros2(self, mock_check):
        """Test ROS2 conversion when ROS2 is not available."""
        mock_check.return_value = False

        converter = ROS2Converter()
        options = ConversionOptions(target_format=DataFormat.ROS2)

        result = converter.convert(self.pc_packet, options)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
        self.assertIn("ROS2 not available", result.error_message)

    def test_converter_factory(self):
        """Test converter factory function."""
        # Test JSON converter creation
        json_converter = create_converter(DataFormat.JSON)
        self.assertIsInstance(json_converter, StandardFormatConverter)

        # Test ROS2 converter creation
        ros2_converter = create_converter(DataFormat.ROS2)
        self.assertIsInstance(ros2_converter, ROS2Converter)

        # Test unsupported format
        with self.assertRaises(ValueError):
            create_converter(DataFormat.PROTOBUF)

    def test_conversion_options(self):
        """Test ConversionOptions creation and validation."""
        options = ConversionOptions(
            target_format=DataFormat.JSON,
            compression=True,
            include_metadata=True,
            coordinate_frame="test_frame"
        )

        self.assertEqual(options.target_format, DataFormat.JSON)
        self.assertTrue(options.compression)
        self.assertTrue(options.include_metadata)
        self.assertEqual(options.coordinate_frame, "test_frame")

    def test_conversion_result_validation(self):
        """Test ConversionResult validation."""
        # Test successful result
        success_result = ConversionResult(
            success=True,
            data="test_data",
            format=DataFormat.JSON
        )
        self.assertTrue(success_result.is_valid())

        # Test failed result
        failed_result = ConversionResult(
            success=False,
            error_message="Test error"
        )
        self.assertFalse(failed_result.is_valid())

        # Test result with no data
        no_data_result = ConversionResult(
            success=True,
            data=None,
            format=DataFormat.JSON
        )
        self.assertFalse(no_data_result.is_valid())

    def test_converter_statistics(self):
        """Test converter statistics tracking."""
        converter = StandardFormatConverter()
        options = ConversionOptions(target_format=DataFormat.JSON)

        # Perform conversion
        converter.convert(self.pc_packet, options)

        # Check statistics
        stats = converter.get_statistics()
        self.assertEqual(stats["total_conversions"], 1)
        self.assertGreaterEqual(stats["successful_conversions"], 0)
        self.assertGreater(stats["average_conversion_time"], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete sensor input system."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = MultiSensorManager()

        # Create multiple sensors
        self.sensors = []
        self.configs = []

        sensor_types = [
            (SensorType.POINT_CLOUD, "pc_sensor_1"),
            (SensorType.IMAGE, "img_sensor_1"),
            (SensorType.IMU, "imu_sensor_1"),
            (SensorType.GPS, "gps_sensor_1")
        ]

        for sensor_type, sensor_id in sensor_types:
            config = SensorConfig(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                update_rate=5.0
            )
            sensor = create_sensor(config)

            self.configs.append(config)
            self.sensors.append(sensor)
            self.manager.add_sensor(sensor, config)

    def test_complete_sensor_pipeline(self):
        """Test complete sensor data pipeline."""
        # Create sensor group for synchronization
        self.manager.create_sensor_group(
            group_id="main_group",
            sensor_ids=["pc_sensor_1", "img_sensor_1", "imu_sensor_1"],
            sync_method=SyncMethod.TIMESTAMP_ALIGNMENT
        )

        # Start all sensors
        self.assertTrue(self.manager.start_sensors())
        self.manager.start_synchronization()

        # Wait for data processing
        time.sleep(0.5)

        # Check that sensors are active
        self.assertGreater(len(self.manager.active_sensors), 0)

        # Check for synchronized data
        sync_data = self.manager.get_synchronized_data("main_group", 1)

        # Assess data quality
        for sensor_id in self.manager.active_sensors:
            assessment = self.manager.assess_sensor_quality(sensor_id)
            self.assertIsInstance(assessment, DataQualityAssessment)

        # Check manager statistics
        stats = self.manager.get_manager_statistics()
        self.assertGreater(stats["runtime_seconds"], 0)

        # Stop all sensors
        self.manager.stop_sensors()
        self.manager.stop_synchronization()

    def test_data_format_conversion_pipeline(self):
        """Test data format conversion pipeline."""
        # Generate test data packet
        pc_data = PointCloudData(points=np.random.rand(50, 3))
        packet = SensorDataPacket(
            sensor_id="test_pc",
            sensor_type=SensorType.POINT_CLOUD,
            timestamp=time.time(),
            data=pc_data
        )

        # Test multiple format conversions
        formats = [DataFormat.JSON, DataFormat.CSV, DataFormat.BINARY, DataFormat.PCD]
        results = []

        for format_type in formats:
            converter = create_converter(format_type)
            options = ConversionOptions(target_format=format_type)
            result = converter.convert(packet, options)
            results.append(result)

            self.assertTrue(result.success, f"Conversion to {format_type} failed")
            self.assertIsNotNone(result.data)

        # Verify all conversions were successful
        successful_conversions = [r for r in results if r.success]
        self.assertEqual(len(successful_conversions), len(formats))

    def test_error_handling(self):
        """Test error handling in the sensor input system."""
        # Test invalid sensor addition
        invalid_sensor = None
        with self.assertRaises(AttributeError):
            self.manager.add_sensor(invalid_sensor, None)

        # Test invalid sensor group creation
        self.assertFalse(self.manager.create_sensor_group(
            group_id="invalid_group",
            sensor_ids=["non_existent_sensor"]
        ))

        # Test data quality assessment for non-existent sensor
        assessment = self.manager.assess_sensor_quality("non_existent")
        self.assertTrue(assessment.is_invalid)

        # Test conversion with invalid data
        converter = StandardFormatConverter()
        options = ConversionOptions(target_format=DataFormat.JSON)
        result = converter.convert(None, options)
        self.assertFalse(result.success)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestSensorDataTypes,
        TestSensorInterfaces,
        TestSensorManager,
        TestDataConverter,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")

    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)