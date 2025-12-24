"""
ROS2集成测试

测试Brain感知模块与ROS2的集成功能，包括传感器数据接收、多传感器同步和感知数据融合。
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, Optional

# 模拟ROS2依赖
class MockROS2Node:
    """模拟ROS2节点"""
    def __init__(self, name):
        self.name = name
        self.subscriptions = {}
        self.publishers = {}
    
    def create_subscription(self, msg_type, topic, callback, qos_profile=None):
        """创建订阅"""
        self.subscriptions[topic] = (msg_type, callback)
        return Mock()
    
    def create_publisher(self, msg_type, topic, qos_profile=None):
        """创建发布者"""
        self.publishers[topic] = (msg_type, None)
        return Mock()
    
    def get_clock(self):
        """获取时钟"""
        return Mock()
    
    def destroy_node(self):
        """销毁节点"""
        pass

class MockROS2Interface:
    """模拟ROS2接口"""
    def __init__(self):
        self.node = MockROS2Node("test_brain_perception")
        self.latest_sensor_data = None
        self.rgb_image = np.random.randint(0, 256, (480, 640, 3))
        self.depth_image = np.random.rand(480, 640) * 10.0
        self.laser_scan = {
            "ranges": [5.0] * 360,
            "angles": [i * 0.017 for i in range(360)]
        }
        self.imu_data = {
            "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
            "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.1},
            "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
        }
        self.odometry_data = {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
        }
    
    def get_sensor_data(self):
        """获取传感器数据"""
        if self.latest_sensor_data is None:
            self.latest_sensor_data = Mock(
                timestamp=time.time(),
                rgb_image=self.rgb_image,
                depth_image=self.depth_image,
                laser_scan=self.laser_scan,
                imu=self.imu_data,
                odometry=self.odometry_data,
                pointcloud=None
            )
        return self.latest_sensor_data
    
    def get_rgb_image(self):
        """获取RGB图像"""
        return self.rgb_image
    
    def get_depth_image(self):
        """获取深度图像"""
        return self.depth_image
    
    def get_laser_scan(self):
        """获取激光雷达数据"""
        return self.laser_scan
    
    def destroy(self):
        """销毁接口"""
        pass


@pytest.fixture
def ros2_interface():
    """ROS2接口模拟对象"""
    return MockROS2Interface()


@pytest.fixture
async def sensor_manager(ros2_interface):
    """传感器管理器"""
    # 模拟导入
    with patch('brain.communication.ros2_interface.ROS2Interface', return_value=ros2_interface):
        with patch('brain.perception.sensors.ros2_sensor_manager.OccupancyMapper') as MockOccupancyMapper:
            # 模拟占据栅格映射器
            mock_mapper = Mock()
            mock_mapper.get_grid.return_value = Mock(
                data=np.random.randint(-1, 2, (500, 500)),
                resolution=0.1,
                origin_x=-25.0,
                origin_y=-25.0
            )
            MockOccupancyMapper.return_value = mock_mapper
            
            # 导入并创建传感器管理器
            from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
            
            config = {
                "sensors": {
                    "rgb_camera": {"enabled": True},
                    "depth_camera": {"enabled": True},
                    "lidar": {"enabled": True},
                    "imu": {"enabled": True}
                },
                "grid_resolution": 0.1,
                "map_size": 50.0,
                "pose_filter_alpha": 0.8,
                "obstacle_threshold": 0.5,
                "min_obstacle_size": 0.1,
                "max_history": 100
            }
            
            manager = ROS2SensorManager(ros2_interface, config)
            return manager


class TestROS2SensorManager:
    """测试ROS2传感器管理器"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, sensor_manager):
        """测试传感器管理器初始化"""
        # 验证配置
        assert sensor_manager.config["sensors"]["rgb_camera"]["enabled"] is True
        assert sensor_manager.config["sensors"]["depth_camera"]["enabled"] is True
        assert sensor_manager.config["sensors"]["lidar"]["enabled"] is True
        assert sensor_manager.config["sensors"]["imu"]["enabled"] is True
        
        assert sensor_manager.config["grid_resolution"] == 0.1
        assert sensor_manager.config["map_size"] == 50.0
        assert sensor_manager.config["pose_filter_alpha"] == 0.8
        assert sensor_manager.config["obstacle_threshold"] == 0.5
        assert sensor_manager.config["min_obstacle_size"] == 0.1
        assert sensor_manager.config["max_history"] == 100
        
        # 验证传感器状态初始化
        from brain.perception.sensors.ros2_sensor_manager import SensorType, SensorStatus
        assert SensorType.RGB_CAMERA in sensor_manager.sensor_status
        assert SensorType.DEPTH_CAMERA in sensor_manager.sensor_status
        assert SensorType.LIDAR in sensor_manager.sensor_status
        assert SensorType.IMU in sensor_manager.sensor_status
        
        assert all(status.enabled for status in sensor_manager.sensor_status.values())
    
    @pytest.mark.asyncio
    async def test_get_fused_perception(self, sensor_manager, ros2_interface):
        """测试获取融合感知数据"""
        # 获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 验证数据结构
        assert perception_data is not None
        assert hasattr(perception_data, "timestamp")
        assert hasattr(perception_data, "pose")
        assert hasattr(perception_data, "rgb_image")
        assert hasattr(perception_data, "depth_image")
        assert hasattr(perception_data, "laser_ranges")
        assert hasattr(perception_data, "laser_angles")
        assert hasattr(perception_data, "obstacles")
        assert hasattr(perception_data, "occupancy_grid")
        assert hasattr(perception_data, "grid_resolution")
        assert hasattr(perception_data, "grid_origin")
        assert hasattr(perception_data, "sensor_status")
        
        # 验证位姿数据
        if perception_data.pose:
            assert hasattr(perception_data.pose, "x")
            assert hasattr(perception_data.pose, "y")
            assert hasattr(perception_data.pose, "z")
            assert hasattr(perception_data.pose, "roll")
            assert hasattr(perception_data.pose, "pitch")
            assert hasattr(perception_data.pose, "yaw")
        
        # 验证图像数据
        if perception_data.rgb_image is not None:
            assert isinstance(perception_data.rgb_image, np.ndarray)
            assert perception_data.rgb_image.shape == (480, 640, 3)
        
        if perception_data.depth_image is not None:
            assert isinstance(perception_data.depth_image, np.ndarray)
            assert perception_data.depth_image.shape == (480, 640)
        
        # 验证激光雷达数据
        if perception_data.laser_ranges is not None:
            assert isinstance(perception_data.laser_ranges, list)
            assert len(perception_data.laser_ranges) == 360
        
        if perception_data.laser_angles is not None:
            assert isinstance(perception_data.laser_angles, list)
            assert len(perception_data.laser_angles) == 360
        
        # 验证占据栅格地图
        if perception_data.occupancy_grid is not None:
            assert isinstance(perception_data.occupancy_grid, np.ndarray)
            assert perception_data.occupancy_grid.shape == (500, 500)
        
        # 验证传感器状态
        assert isinstance(perception_data.sensor_status, dict)
        assert "rgb_camera" in perception_data.sensor_status
        assert "depth_camera" in perception_data.sensor_status
        assert "lidar" in perception_data.sensor_status
        assert "imu" in perception_data.sensor_status
    
    @pytest.mark.asyncio
    async def test_pose_fusion(self, sensor_manager, ros2_interface):
        """测试位姿融合"""
        # 设置IMU数据有差异
        ros2_interface.imu_data = {
            "orientation": {"x": 0.05, "y": 0.1, "z": 0.0, "w": 0.995},
            "angular_velocity": {"x": 0.02, "y": 0.01, "z": 0.15},
            "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
        }
        
        # 获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 验证位姿数据
        if perception_data.pose:
            # 由于IMU数据有差异，pose应该与里程计不同
            assert abs(perception_data.pose.roll - 0.1) < 0.1  # 互补滤波后的值
            assert abs(perception_data.pose.pitch - 0.2) < 0.1  # 互补滤波后的值
    
    @pytest.mark.asyncio
    async def test_obstacle_detection(self, sensor_manager, ros2_interface):
        """测试障碍物检测"""
        # 设置有障碍物的激光数据
        ros2_interface.laser_scan = {
            "ranges": [5.0] * 180 + [2.0] * 20 + [5.0] * 160,
            "angles": [i * 0.017 for i in range(360)]
        }
        
        # 获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 验证障碍物检测
        assert perception_data.obstacles is not None
        assert len(perception_data.obstacles) > 0
        
        # 验证障碍物信息
        obstacle = perception_data.obstacles[0]
        assert "id" in obstacle
        assert "type" in obstacle
        assert "local_position" in obstacle
        assert "world_position" in obstacle
        assert "size" in obstacle
        assert "distance" in obstacle
        assert "angle" in obstacle
        assert "direction" in obstacle
        assert "point_count" in obstacle
    
    @pytest.mark.asyncio
    async def test_occupancy_grid_generation(self, sensor_manager, ros2_interface):
        """测试占据栅格地图生成"""
        # 设置有障碍物的深度数据
        ros2_interface.depth_image = np.ones((240, 320)) * 5.0
        ros2_interface.depth_image[:, 150:170] = 2.0  # 中央区域障碍物
        
        # 设置机器人位姿
        ros2_interface.odometry_data = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0.0}
        }
        
        # 获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 验证占据栅格地图
        if perception_data.occupancy_grid is not None:
            assert isinstance(perception_data.occupancy_grid, np.ndarray)
            assert perception_data.occupancy_grid.shape == (500, 500)
            assert perception_data.grid_resolution == 0.1
            assert perception_data.grid_origin == (-25.0, -25.0)
    
    @pytest.mark.asyncio
    async def test_sensor_health(self, sensor_manager):
        """测试传感器健康检查"""
        # 获取传感器健康状态
        health = sensor_manager.get_sensor_health()
        
        # 验证健康状态
        assert isinstance(health, dict)
        assert "rgb_camera" in health
        assert "depth_camera" in health
        assert "lidar" in health
        assert "imu" in health
        
        # 在模拟环境中，所有传感器应该是健康的
        assert all(status is True for status in health.values())
    
    @pytest.mark.asyncio
    async def test_wait_for_sensors(self, sensor_manager):
        """测试等待传感器就绪"""
        # 在模拟环境中，传感器应该立即可用
        ready = await sensor_manager.wait_for_sensors(timeout=1.0)
        assert ready is True
    
    @pytest.mark.asyncio
    async def test_data_methods(self, sensor_manager, ros2_interface):
        """测试数据获取方法"""
        # 获取RGB图像
        rgb_image = sensor_manager.get_rgb_image()
        assert isinstance(rgb_image, np.ndarray)
        assert rgb_image.shape == (480, 640, 3)
        
        # 获取深度图像
        depth_image = sensor_manager.get_depth_image()
        assert isinstance(depth_image, np.ndarray)
        assert depth_image.shape == (480, 640)
        
        # 获取激光雷达数据
        laser_scan = sensor_manager.get_laser_scan()
        assert isinstance(laser_scan, dict)
        assert "ranges" in laser_scan
        assert "angles" in laser_scan
        assert len(laser_scan["ranges"]) == 360
        
        # 获取当前位姿
        current_pose = sensor_manager.get_current_pose()
        if current_pose:
            assert hasattr(current_pose, "x")
            assert hasattr(current_pose, "y")
            assert hasattr(current_pose, "z")
            assert hasattr(current_pose, "roll")
            assert hasattr(current_pose, "pitch")
            assert hasattr(current_pose, "yaw")
        
        # 获取当前2D位姿
        current_pose_2d = sensor_manager.get_current_pose_2d()
        if current_pose_2d:
            assert hasattr(current_pose_2d, "x")
            assert hasattr(current_pose_2d, "y")
            assert hasattr(current_pose_2d, "theta")
    
    @pytest.mark.asyncio
    async def test_obstacle_query_methods(self, sensor_manager, ros2_interface):
        """测试障碍物查询方法"""
        # 设置有障碍物的激光数据
        ros2_interface.laser_scan = {
            "ranges": [2.0] + [5.0] * 178 + [2.0] + [5.0] * 179,
            "angles": [i * 0.017 for i in range(360)]
        }
        
        # 获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 获取最近障碍物
        nearest = sensor_manager.get_nearest_obstacle()
        assert nearest is not None
        assert "distance" in nearest
        assert nearest["distance"] <= 2.0  # 最近障碍物应该是我们设置的2.0
        
        # 获取前方障碍物
        front_obstacles = sensor_manager.get_obstacles_in_direction("front")
        assert len(front_obstacles) > 0
        
        # 获取左侧障碍物
        left_obstacles = sensor_manager.get_obstacles_in_direction("left")
        assert len(left_obstacles) >= 0  # 可能为空
        
        # 获取右侧障碍物
        right_obstacles = sensor_manager.get_obstacles_in_direction("right")
        assert len(right_obstacles) > 0
    
    @pytest.mark.asyncio
    async def test_path_clear_check(self, sensor_manager, ros2_interface):
        """测试路径畅通检查"""
        # 设置无障碍物的激光数据
        ros2_interface.laser_scan = {
            "ranges": [10.0] * 360,  # 全部10米远
            "angles": [i * 0.017 for i in range(360)]
        }
        
        # 获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 检查前方路径是否畅通
        assert perception_data.is_path_clear("front", threshold=1.0) is True
        assert perception_data.is_path_clear("left", threshold=1.0) is True
        assert perception_data.is_path_clear("right", threshold=1.0) is True
        
        # 设置有障碍物的激光数据
        ros2_interface.laser_scan = {
            "ranges": [0.5] + [10.0] * 179 + [10.0] * 180,
            "angles": [i * 0.017 for i in range(360)]
        }
        
        # 重新获取融合感知数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 检查前方路径是否畅通
        assert perception_data.is_path_clear("front", threshold=1.0) is False
        assert perception_data.is_path_clear("left", threshold=1.0) is True
        assert perception_data.is_path_clear("right", threshold=1.0) is True
    
    @pytest.mark.asyncio
    async def test_data_history(self, sensor_manager, ros2_interface):
        """测试数据历史记录"""
        # 获取融合感知数据多次
        for _ in range(5):
            await sensor_manager.get_fused_perception()
            await asyncio.sleep(0.01)
        
        # 获取数据历史
        history = sensor_manager.get_data_history(count=3)
        assert len(history) == 3
        
        # 验证数据结构
        for data in history:
            assert hasattr(data, "timestamp")
            assert hasattr(data, "pose")
            assert hasattr(data, "rgb_image")
        
        # 验证时间顺序
        for i in range(1, len(history)):
            assert history[i].timestamp >= history[i-1].timestamp


class TestSensorDataProcessing:
    """测试传感器数据处理"""
    
    @pytest.mark.asyncio
    async def test_pose_extraction(self, ros2_interface):
        """测试位姿提取"""
        # 模拟导入
        with patch('brain.communication.ros2_interface.ROS2Interface', return_value=ros2_interface):
            from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
            
            # 创建传感器管理器
            manager = ROS2SensorManager(ros2_interface, {})
            
            # 测试位姿提取
            pose = manager._extract_pose(ros2_interface.odometry_data)
            
            assert pose.x == 1.0
            assert pose.y == 2.0
            assert pose.z == 0.0
            # 四元数(0, 0, 0, 1)对应欧拉角(0, 0, 0)
            assert abs(pose.roll) < 1e-6
            assert abs(pose.pitch) < 1e-6
            assert abs(pose.yaw) < 1e-6
    
    @pytest.mark.asyncio
    async def test_velocity_extraction(self, ros2_interface):
        """测试速度提取"""
        # 模拟导入
        with patch('brain.communication.ros2_interface.ROS2Interface', return_value=ros2_interface):
            from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
            
            # 创建传感器管理器
            manager = ROS2SensorManager(ros2_interface, {})
            
            # 测试速度提取
            velocity = manager._extract_velocity(ros2_interface.odometry_data)
            
            assert velocity.linear_x == 0.5
            assert velocity.linear_y == 0.0
            assert velocity.linear_z == 0.0
            assert velocity.angular_x == 0.0
            assert velocity.angular_y == 0.0
            assert velocity.angular_z == 0.1
    
    @pytest.mark.asyncio
    async def test_imu_pose_fusion(self, ros2_interface):
        """测试IMU位姿融合"""
        # 模拟导入
        with patch('brain.communication.ros2_interface.ROS2Interface', return_value=ros2_interface):
            from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
            
            # 创建传感器管理器
            manager = ROS2SensorManager(ros2_interface, {
                "pose_filter_alpha": 0.8  # 80%来自原始位姿，20%来自IMU
            })
            
            # 创建原始位姿
            from brain.perception.data_models import Pose3D
            original_pose = Pose3D(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
            
            # 测试IMU融合
            fused_pose = manager._fuse_imu_pose(original_pose, ros2_interface.imu_data)
            
            # 验证融合结果
            # 四元数(0.05, 0.1, 0, 0.995)对应欧拉角(约0.2, 0.1, 0)
            assert abs(fused_pose.roll - 0.04) < 0.01  # 0.0 * 0.8 + 0.2 * 0.2
            assert abs(fused_pose.pitch - 0.02) < 0.01  # 0.0 * 0.8 + 0.1 * 0.1
            # yaw主要来自里程计，IMU漂移大
            assert abs(fused_pose.yaw) < 1e-6
    
    @pytest.mark.asyncio
    async def test_laser_obstacle_detection(self, ros2_interface):
        """测试激光雷达障碍物检测"""
        # 模拟导入
        with patch('brain.communication.ros2_interface.ROS2Interface', return_value=ros2_interface):
            from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
            
            # 创建传感器管理器
            manager = ROS2SensorManager(ros2_interface, {
                "min_obstacle_size": 0.1
            })
            
            # 创建机器人位姿
            from brain.perception.data_models import Pose3D
            pose = Pose3D(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
            
            # 测试激光雷达障碍物检测
            obstacles = manager._detect_obstacles_from_laser(
                ros2_interface.laser_scan["ranges"],
                ros2_interface.laser_scan["angles"],
                pose
            )
            
            # 默认情况下，应该没有障碍物
            assert len(obstacles) == 0
            
            # 设置有障碍物的激光数据
            ros2_interface.laser_scan["ranges"][90] = 2.0  # 正前方2米
            
            obstacles = manager._detect_obstacles_from_laser(
                ros2_interface.laser_scan["ranges"],
                ros2_interface.laser_scan["angles"],
                pose
            )
            
            # 应该检测到障碍物
            assert len(obstacles) > 0
            
            # 验证障碍物信息
            obstacle = obstacles[0]
            assert "id" in obstacle
            assert "type" in obstacle
            assert "local_position" in obstacle
            assert "world_position" in obstacle
            assert "size" in obstacle
            assert "distance" in obstacle
            assert "angle" in obstacle
            assert "direction" in obstacle
            assert "point_count" in obstacle
            
            # 验证位置
            assert abs(obstacle["local_position"]["x"]) < 0.1  # 应该接近0
            assert abs(obstacle["local_position"]["y"] - 2.0) < 0.1  # 应该接近2.0


if __name__ == "__main__":
    pytest.main([__file__])




