"""
传感器接口单元测试

测试各种传感器类的初始化、数据采集和处理功能。
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch

from brain.perception.sensors.sensor_interface import (
    BaseSensor, ImageSensor, PointCloudSensor, IMUSensor, GPSSensor, 
    SensorConfig, create_sensor
)
from brain.perception.sensor_input_types import (
    SensorType, ImageData, PointCloudData, IMUData, GPSData,
    SensorDataPacket, CameraIntrinsics, SensorMetadata
)


class TestSensorConfig:
    """测试传感器配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SensorConfig(
            sensor_id="test_sensor",
            sensor_type=SensorType.CAMERA
        )
        
        assert config.sensor_id == "test_sensor"
        assert config.sensor_type == SensorType.CAMERA
        assert config.frame_id == "base_link"
        assert config.update_rate == 10.0
        assert config.auto_start is True
        assert config.buffer_size == 100
        assert config.enable_compression is False
        assert config.quality_threshold == 0.5
        assert config.max_processing_time == 0.1
        assert config.enable_noise_filtering is True
        assert config.enable_outlier_removal is True
        assert config.min_data_quality == 0.3
        assert config.ros2_topic is None
        assert config.ros2_qos_profile == "best_effort"
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SensorConfig(
            sensor_id="custom_sensor",
            sensor_type=SensorType.LIDAR,
            frame_id="custom_frame",
            update_rate=30.0,
            auto_start=False,
            buffer_size=200,
            enable_compression=True,
            quality_threshold=0.8,
            max_processing_time=0.05,
            calibration_params={"key": "value"},
            enable_noise_filtering=False,
            enable_outlier_removal=False,
            min_data_quality=0.6,
            ros2_topic="/custom/topic",
            ros2_qos_profile="reliable"
        )
        
        assert config.sensor_id == "custom_sensor"
        assert config.sensor_type == SensorType.LIDAR
        assert config.frame_id == "custom_frame"
        assert config.update_rate == 30.0
        assert config.auto_start is False
        assert config.buffer_size == 200
        assert config.enable_compression is True
        assert config.quality_threshold == 0.8
        assert config.max_processing_time == 0.05
        assert config.calibration_params == {"key": "value"}
        assert config.enable_noise_filtering is False
        assert config.enable_outlier_removal is False
        assert config.min_data_quality == 0.6
        assert config.ros2_topic == "/custom/topic"
        assert config.ros2_qos_profile == "reliable"


class TestImageSensor:
    """测试图像传感器"""
    
    @pytest.fixture
    def config(self):
        """图像传感器配置"""
        return SensorConfig(
            sensor_id="test_camera",
            sensor_type=SensorType.CAMERA,
            update_rate=30.0,
            buffer_size=10
        )
    
    @pytest.fixture
    def camera_intrinsics(self):
        """相机内参"""
        return CameraIntrinsics(
            fx=525.0, fy=525.0,
            cx=320.0, cy=240.0,
            width=640, height=480
        )
    
    def test_initialization(self, config, camera_intrinsics):
        """测试图像传感器初始化"""
        sensor = ImageSensor(config, camera_intrinsics=camera_intrinsics)
        
        assert sensor.config.sensor_id == "test_camera"
        assert sensor.config.sensor_type == SensorType.CAMERA
        assert sensor.camera_intrinsics == camera_intrinsics
        assert not sensor.is_running
        assert sensor.auto_exposure is True
        assert sensor.auto_white_balance is True
        assert sensor.image_width == 1920
        assert sensor.image_height == 1080
        assert sensor.enable_rectification is True
    
    def test_start_stop(self, config, camera_intrinsics):
        """测试传感器启动和停止"""
        sensor = ImageSensor(config, camera_intrinsics=camera_intrinsics)
        
        # 测试启动
        assert sensor.start()
        assert sensor.is_running
        
        # 测试重复启动
        assert not sensor.start()
        assert sensor.is_running
        
        # 测试停止
        sensor.stop()
        assert not sensor.is_running
        
        # 测试重复停止
        sensor.stop()
        assert not sensor.is_running
    
    def test_data_acquisition(self, config, camera_intrinsics):
        """测试数据采集"""
        sensor = ImageSensor(config, camera_intrinsics=camera_intrinsics)
        sensor.start()
        
        # 等待数据生成
        time.sleep(0.5)
        
        data_packets = sensor.get_latest_data(5)
        assert len(data_packets) > 0
        
        for packet in data_packets:
            assert isinstance(packet.data, ImageData)
            assert packet.sensor_id == "test_camera"
            assert 0 <= packet.quality_score <= 1.0
        
        sensor.stop()
    
    def test_image_generation(self, config, camera_intrinsics):
        """测试图像生成"""
        sensor = ImageSensor(config, camera_intrinsics=camera_intrinsics)
        
        # 测试合成图像生成
        image = sensor._generate_synthetic_image()
        assert image.shape == (sensor.image_height, sensor.image_width, 3)
        assert image.dtype == np.uint8
        
        # 检查图像范围
        assert 0 <= np.min(image) <= 255
        assert 0 <= np.max(image) <= 255
    
    def test_statistics(self, config, camera_intrinsics):
        """测试统计信息"""
        sensor = ImageSensor(config, camera_intrinsics=camera_intrinsics)
        sensor.start()
        
        # 等待数据生成
        time.sleep(0.5)
        
        stats = sensor.get_statistics()
        assert "sensor_id" in stats
        assert "sensor_type" in stats
        assert "is_running" in stats
        assert "runtime_seconds" in stats
        assert "packets_received" in stats
        assert "packets_dropped" in stats
        assert "loss_rate_percent" in stats
        assert "average_update_rate" in stats
        assert "last_update_time" in stats
        assert "buffer_size" in stats
        assert "config" in stats
        
        assert stats["sensor_id"] == "test_camera"
        assert stats["sensor_type"] == "camera"
        assert stats["is_running"] is True
        
        sensor.stop()
    
    def test_callbacks(self, config, camera_intrinsics):
        """测试回调机制"""
        sensor = ImageSensor(config, camera_intrinsics=camera_intrinsics)
        
        # 创建回调函数
        callback_data = []
        def test_callback(packet):
            callback_data.append(packet)
        
        # 添加回调
        sensor.add_callback(test_callback)
        
        # 启动传感器
        sensor.start()
        
        # 等待数据生成
        time.sleep(0.5)
        
        # 验证回调被调用
        assert len(callback_data) > 0
        for packet in callback_data:
            assert isinstance(packet, SensorDataPacket)
            assert packet.sensor_id == "test_camera"
        
        # 移除回调
        sensor.remove_callback(test_callback)
        callback_data.clear()
        
        # 等待更多数据
        time.sleep(0.5)
        
        # 验证回调未被调用
        assert len(callback_data) == 0
        
        sensor.stop()


class TestPointCloudSensor:
    """测试点云传感器"""
    
    @pytest.fixture
    def config(self):
        """点云传感器配置"""
        return SensorConfig(
            sensor_id="test_lidar",
            sensor_type=SensorType.LIDAR,
            update_rate=10.0,
            buffer_size=10
        )
    
    def test_initialization(self, config):
        """测试点云传感器初始化"""
        sensor = PointCloudSensor(config)
        
        assert sensor.config.sensor_id == "test_lidar"
        assert sensor.config.sensor_type == SensorType.LIDAR
        assert not sensor.is_running
        assert sensor.voxel_size == 0.05
        assert sensor.max_range == 100.0
        assert sensor.min_range == 0.5
        assert sensor.remove_ground_plane is True
        assert sensor.ground_threshold == 0.1
    
    def test_point_cloud_generation(self, config):
        """测试点云生成"""
        sensor = PointCloudSensor(config)
        
        # 测试合成点云生成
        points = sensor._generate_synthetic_point_cloud(1000)
        assert points.shape == (1000, 3)
        assert not np.isnan(points).any()
        
        # 检查点云范围
        distances = np.linalg.norm(points, axis=1)
        assert np.all(distances <= 10.0)  # 应该在10米范围内
    
    def test_noise_filtering(self, config):
        """测试噪声过滤"""
        sensor = PointCloudSensor(config)
        
        # 创建测试点云
        points = np.random.randn(100, 3)
        intensity = np.random.rand(100)
        
        # 测试超出范围的点
        points[0] = [200, 200, 200]  # 超出max_range
        points[1] = [0.01, 0.01, 0.01]  # 低于min_range
        
        test_data = PointCloudData(points=points, intensity=intensity)
        filtered_data = sensor._apply_noise_filtering(test_data)
        
        # 验证超出范围的点被过滤
        assert filtered_data.points.shape[0] < 100
        assert np.all(np.linalg.norm(filtered_data.points, axis=1) >= sensor.min_range)
        assert np.all(np.linalg.norm(filtered_data.points, axis=1) <= sensor.max_range)
    
    def test_outlier_removal(self, config):
        """测试异常值移除"""
        sensor = PointCloudSensor(config)
        
        # 创建包含异常值的点云
        normal_points = np.random.randn(50, 3)
        outlier_points = np.array([[10, 10, 10], [-10, -10, -10]])
        all_points = np.vstack([normal_points, outlier_points])
        
        test_data = PointCloudData(points=all_points)
        filtered_data = sensor._remove_outliers(test_data)
        
        # 验证异常值被过滤
        assert filtered_data.points.shape[0] < all_points.shape[0]
        
        # 验证大部分正常点保留
        # 注意：由于实现方式不同，这个断言可能需要调整
        assert filtered_data.points.shape[0] >= 45  # 至少保留大部分正常点


class TestIMUSensor:
    """测试IMU传感器"""
    
    @pytest.fixture
    def config(self):
        """IMU传感器配置"""
        return SensorConfig(
            sensor_id="test_imu",
            sensor_type=SensorType.IMU,
            update_rate=100.0,
            buffer_size=10
        )
    
    def test_initialization(self, config):
        """测试IMU传感器初始化"""
        sensor = IMUSensor(config)
        
        assert sensor.config.sensor_id == "test_imu"
        assert sensor.config.sensor_type == SensorType.IMU
        assert not sensor.is_running
        assert sensor.gravity_compensation is True
        assert sensor.bias_estimation is True
        assert sensor.integration_window == 0.1
    
    def test_imu_data_generation(self, config):
        """测试IMU数据生成"""
        sensor = IMUSensor(config)
        
        # 生成合成IMU数据
        imu_data = sensor._acquire_data()
        
        assert isinstance(imu_data, IMUData)
        assert imu_data.linear_acceleration.shape == (3,)
        assert imu_data.angular_velocity.shape == (3,)
        assert imu_data.orientation is not None
        assert imu_data.orientation.shape == (4,)
        
        # 验证四元数归一化
        quaternion_norm = np.linalg.norm(imu_data.orientation)
        assert abs(quaternion_norm - 1.0) < 1e-6
    
    def test_noise_filtering(self, config):
        """测试IMU噪声过滤"""
        sensor = IMUSensor(config)
        
        # 创建测试IMU数据
        acceleration = np.array([1.0, 2.0, 15.0])  # 包含重力
        angular_velocity = np.array([0.1, 0.2, 0.3])
        orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        test_data = IMUData(
            linear_acceleration=acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation
        )
        
        filtered_data = sensor._apply_noise_filtering(test_data)
        
        # 验证加速度滤波
        assert filtered_data.linear_acceleration is not None
        assert filtered_data.angular_velocity is not None
        assert filtered_data.orientation is not None


class TestGPSSensor:
    """测试GPS传感器"""
    
    @pytest.fixture
    def config(self):
        """GPS传感器配置"""
        return SensorConfig(
            sensor_id="test_gps",
            sensor_type=SensorType.GPS,
            update_rate=1.0,
            buffer_size=10
        )
    
    def test_initialization(self, config):
        """测试GPS传感器初始化"""
        sensor = GPSSensor(config)
        
        assert sensor.config.sensor_id == "test_gps"
        assert sensor.config.sensor_type == SensorType.GPS
        assert not sensor.is_running
        assert sensor.coordinate_system == "WGS84"
        assert sensor.enable_dgps is True
        assert sensor.min_satellites == 4
    
    def test_gps_data_generation(self, config):
        """测试GPS数据生成"""
        sensor = GPSSensor(config)
        
        # 生成合成GPS数据
        gps_data = sensor._acquire_data()
        
        assert isinstance(gps_data, GPSData)
        assert -90 <= gps_data.latitude <= 90
        assert -180 <= gps_data.longitude <= 180
        assert gps_data.altitude > 0
        assert gps_data.fix_type in [0, 1, 2, 3]
        assert gps_data.satellites_used >= 4
        assert gps_data.hdop >= 0
        assert gps_data.vdop >= 0
    
    def test_utm_conversion(self, config):
        """测试UTM坐标转换"""
        sensor = GPSSensor(config)
        
        # 创建测试GPS数据
        gps_data = GPSData(
            latitude=39.9042,  # 北京天安门
            longitude=116.4074,
            altitude=50.0
        )
        
        # 转换为UTM坐标
        utm_x, utm_y = gps_data.to_utm()
        
        # 验证UTM坐标
        assert isinstance(utm_x, (float, np.float32, np.float64))
        assert isinstance(utm_y, (float, np.float32, np.float64))
        assert utm_x > 0
        assert utm_y > 0


class TestSensorFactory:
    """测试传感器工厂"""
    
    def test_create_image_sensor(self):
        """测试创建图像传感器"""
        config = SensorConfig(
            sensor_id="test_camera",
            sensor_type=SensorType.CAMERA
        )
        
        sensor = create_sensor(config)
        assert isinstance(sensor, ImageSensor)
        assert sensor.config.sensor_id == "test_camera"
    
    def test_create_point_cloud_sensor(self):
        """测试创建点云传感器"""
        config = SensorConfig(
            sensor_id="test_lidar",
            sensor_type=SensorType.LIDAR
        )
        
        sensor = create_sensor(config)
        assert isinstance(sensor, PointCloudSensor)
        assert sensor.config.sensor_id == "test_lidar"
    
    def test_create_imu_sensor(self):
        """测试创建IMU传感器"""
        config = SensorConfig(
            sensor_id="test_imu",
            sensor_type=SensorType.IMU
        )
        
        sensor = create_sensor(config)
        assert isinstance(sensor, IMUSensor)
        assert sensor.config.sensor_id == "test_imu"
    
    def test_create_gps_sensor(self):
        """测试创建GPS传感器"""
        config = SensorConfig(
            sensor_id="test_gps",
            sensor_type=SensorType.GPS
        )
        
        sensor = create_sensor(config)
        assert isinstance(sensor, GPSSensor)
        assert sensor.config.sensor_id == "test_gps"
    
    def test_create_sensor_with_alias(self):
        """测试使用别名创建传感器"""
        config = SensorConfig(
            sensor_id="test_depth_camera",
            sensor_type=SensorType.DEPTH_CAMERA
        )
        
        sensor = create_sensor(config)
        assert isinstance(sensor, ImageSensor)
    
    def test_create_invalid_sensor(self):
        """测试创建不支持的传感器类型"""
        # 模拟不支持的传感器类型
        config = SensorConfig(
            sensor_id="test_invalid",
            sensor_type="invalid_type"
        )
        
        with pytest.raises(ValueError):
            create_sensor(config)
    
    def test_create_sensor_with_metadata(self):
        """测试创建带元数据的传感器"""
        config = SensorConfig(
            sensor_id="test_lidar_with_meta",
            sensor_type=SensorType.LIDAR
        )
        
        metadata = SensorMetadata(
            sensor_id="test_lidar_with_meta",
            sensor_type=SensorType.LIDAR,
            manufacturer="TestCorp",
            model="TestLidar",
            serial_number="12345",
            fov_h=360.0,
            fov_v=30.0,
            range_min=0.1,
            range_max=100.0,
            update_rate=10.0,
            accuracy=0.05,
            resolution="0.1m"
        )
        
        sensor = create_sensor(config, metadata=metadata)
        assert isinstance(sensor, PointCloudSensor)
        assert sensor.metadata == metadata


class TestSensorLifecycle:
    """测试传感器生命周期"""
    
    @pytest.fixture
    def config(self):
        """传感器配置"""
        return SensorConfig(
            sensor_id="test_lifecycle",
            sensor_type=SensorType.CAMERA,
            update_rate=10.0,
            buffer_size=10
        )
    
    def test_resource_cleanup(self, config):
        """测试资源清理"""
        sensor = ImageSensor(config)
        
        # 启动传感器
        sensor.start()
        assert sensor.is_running
        
        # 删除传感器，应该自动调用清理
        del sensor
        # 由于Python的垃圾回收机制，这里很难直接测试
        # 但可以确保没有资源泄漏
    
    def test_thread_safety(self, config):
        """测试线程安全性"""
        sensor = ImageSensor(config)
        
        # 启动传感器
        sensor.start()
        
        # 创建多个线程同时访问数据
        results = []
        def access_data():
            for _ in range(10):
                data = sensor.get_latest_data(1)
                results.append(len(data))
                time.sleep(0.01)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=access_data)
            threads.append(t)
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
        
        # 验证所有访问都成功
        assert len(results) == 50  # 5个线程 × 10次访问
        assert all(0 <= r <= 1 for r in results)  # 每次访问返回0或1个数据包
        
        sensor.stop()
    
    def test_buffer_overflow(self, config):
        """测试缓冲区溢出"""
        # 使用小缓冲区
        small_buffer_config = SensorConfig(
            sensor_id="test_overflow",
            sensor_type=SensorType.CAMERA,
            buffer_size=3
        )
        
        sensor = ImageSensor(small_buffer_config)
        sensor.start()
        
        # 等待缓冲区填满
        time.sleep(1.0)
        
        # 获取所有数据
        all_data = sensor.get_latest_data(100)  # 请求超过缓冲区大小的数据
        
        # 验证数据不超过缓冲区大小
        assert len(all_data) <= 3
        
        sensor.stop()


if __name__ == "__main__":
    pytest.main([__file__])




