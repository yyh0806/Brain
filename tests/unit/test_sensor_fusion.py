"""
传感器融合算法单元测试

测试多传感器融合算法，包括位姿融合、RGBD融合和障碍物检测融合。
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch

from brain.perception.sensors.sensor_fusion import (
    EKFPoseFusion, DepthRGBFusion, ObstacleDetector,
    FusedPose, DepthRGBFusion, ObstacleDetector
)
from brain.perception.sensor_input_types import (
    PointCloudData, ImageData
)


class TestEKFPoseFusion:
    """测试扩展卡尔曼滤波位姿融合"""
    
    @pytest.fixture
    def ekf(self):
        """创建EKF融合器实例"""
        return EKFPoseFusion()
    
    def test_initialization(self, ekf):
        """测试EKF初始化"""
        assert not ekf.initialized
        assert ekf.state.shape == (12,)
        assert ekf.P.shape == (12, 12)
        assert ekf.Q.shape == (12, 12)
        assert ekf.R_odom.shape == (6, 6)
        assert ekf.R_imu.shape == (6, 6)
        assert ekf.last_update is None
        
        # 验证过程噪声协方差设置
        assert ekf.Q[6:9, 6:9].sum() > ekf.Q[:6, :6].sum()  # 速度噪声更大
        assert ekf.Q[9:12, 9:12].sum() > ekf.Q[:6, :6].sum()  # 角速度噪声更大
    
    def test_odometry_update_initializes(self, ekf):
        """测试里程计更新初始化EKF"""
        odom_data = {
            "position": {"x": 1.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
            "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
        }
        
        ekf.update_odom(odom_data)
        
        assert ekf.initialized
        assert ekf.state[0] == 1.0  # x
        assert ekf.state[1] == 2.0  # y
        assert ekf.state[2] == 0.0  # z
        assert ekf.state[6] == 0.5  # vx
        assert ekf.state[7] == 0.0  # vy
        assert ekf.state[8] == 0.0  # vz
        assert ekf.state[9] == 0.0  # wx
        assert ekf.state[10] == 0.0  # wy
        assert ekf.state[11] == 0.1  # wz
        
        assert ekf.last_update is not None
    
    def test_imu_update_without_initialization(self, ekf):
        """测试未初始化时IMU更新"""
        imu_data = {
            "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
            "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.1},
            "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
        }
        
        # 不应该更新状态
        initial_state = ekf.state.copy()
        ekf.update_imu(imu_data)
        
        assert np.array_equal(ekf.state, initial_state)
    
    def test_imu_updates_roll_pitch(self, ekf):
        """测试IMU更新roll和pitch"""
        # 首先初始化
        odom_data = {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0, "y": 0, "z": 0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0}
        }
        ekf.update_odom(odom_data)
        
        initial_roll = ekf.state[3]
        initial_pitch = ekf.state[4]
        initial_yaw = ekf.state[5]
        
        # 更新IMU
        imu_data = {
            "orientation": {"x": 0.1, "y": 0.05, "z": 0.0, "w": 0.9987},  # roll≈0.2, pitch≈0.1
            "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.1},
            "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
        }
        
        ekf.update_imu(imu_data)
        
        # roll和pitch应该更新，yaw变化较小
        assert ekf.state[3] != initial_roll  # roll更新
        assert ekf.state[4] != initial_pitch  # pitch更新
        # yaw可能有一些变化，但应该较小
        
        # 角速度应该更新
        assert ekf.state[9] == 0.01  # wx
        assert ekf.state[10] == 0.02  # wy
        assert ekf.state[11] == 0.1   # wz
    
    def test_prediction(self, ekf):
        """测试EKF预测"""
        # 初始化
        odom_data = {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 1.0, "y": 0.5, "z": 0.0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
        }
        ekf.update_odom(odom_data)
        
        initial_x = ekf.state[0]
        initial_y = ekf.state[1]
        initial_z = ekf.state[2]
        initial_roll = ekf.state[3]
        initial_pitch = ekf.state[4]
        initial_yaw = ekf.state[5]
        
        # 预测1秒
        ekf.predict(dt=1.0)
        
        # 位置应该根据速度更新
        assert ekf.state[0] > initial_x  # x增加
        assert ekf.state[1] > initial_y  # y增加
        assert ekf.state[2] == initial_z  # z不变(速度为0)
        
        # 姿态应该根据角速度更新
        assert ekf.state[3] > initial_roll   # roll增加
        assert ekf.state[4] > initial_pitch  # pitch增加
        assert ekf.state[5] > initial_yaw    # yaw增加
        
        # 速度应该保持不变
        assert ekf.state[6] == 1.0  # vx
        assert ekf.state[7] == 0.5  # vy
        assert ekf.state[8] == 0.0  # vz
        assert ekf.state[9] == 0.0   # wx
        assert ekf.state[10] == 0.0  # wy
        assert ekf.state[11] == 0.1  # wz
    
    def test_get_pose(self, ekf):
        """测试获取位姿"""
        # 初始化
        odom_data = {
            "position": {"x": 1.0, "y": 2.0, "z": 0.5},
            "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
            "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
        }
        ekf.update_odom(odom_data)
        
        pose = ekf.get_pose()
        
        assert isinstance(pose, FusedPose)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 0.5
        assert pose.covariance.shape == (6, 6)
        assert pose.timestamp is not None
    
    def test_get_velocity(self, ekf):
        """测试获取速度"""
        # 初始化
        odom_data = {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            "linear_velocity": {"x": 0.5, "y": 0.3, "z": 0.1},
            "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.1}
        }
        ekf.update_odom(odom_data)
        
        linear_vel, angular_vel = ekf.get_velocity()
        
        assert linear_vel.shape == (3,)
        assert angular_vel.shape == (3,)
        assert linear_vel[0] == 0.5  # vx
        assert linear_vel[1] == 0.3  # vy
        assert linear_vel[2] == 0.1  # vz
        assert angular_vel[0] == 0.01  # wx
        assert angular_vel[1] == 0.02  # wy
        assert angular_vel[2] == 0.1   # wz


class TestDepthRGBFusion:
    """测试RGBD融合"""
    
    @pytest.fixture
    def fusion(self):
        """创建RGBD融合器实例"""
        return DepthRGBFusion()
    
    @pytest.fixture
    def rgb_image(self):
        """创建测试RGB图像"""
        h, w = 240, 320
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    
    @pytest.fixture
    def depth_image(self):
        """创建测试深度图"""
        h, w = 240, 320
        return np.random.rand(h, w) * 10.0
    
    def test_initialization(self, fusion):
        """测试RGBD融合器初始化"""
        assert fusion.fx == 525.0
        assert fusion.fy == 525.0
        assert fusion.cx == 319.5
        assert fusion.cy == 239.5
    
    def test_custom_initialization(self):
        """测试自定义参数初始化"""
        custom_fusion = DepthRGBFusion(
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0
        )
        
        assert custom_fusion.fx == 600.0
        assert custom_fusion.fy == 600.0
        assert custom_fusion.cx == 320.0
        assert custom_fusion.cy == 240.0
    
    def test_fuse(self, fusion, rgb_image, depth_image):
        """测试RGBD融合"""
        rgbd, points = fusion.fuse(rgb_image, depth_image)
        
        # 验证RGBD图像
        assert rgbd.shape == (rgb_image.shape[0], rgb_image.shape[1], 4)
        assert rgbd[:, :, :3].shape == rgb_image.shape
        assert np.array_equal(rgbd[:, :, :3], rgb_image.astype(np.float32) / 255.0)
        assert np.array_equal(rgbd[:, :, 3], depth_image)
        
        # 验证点云
        assert points.shape[1] == 6  # X, Y, Z, R, G, B
        assert points.shape[0] > 0  # 应该有点
        
        # 验证点云XYZ与深度图一致
        valid_depth_mask = depth_image > 0
        expected_points = valid_depth_mask.sum()
        assert points.shape[0] <= expected_points  # 可能有些点被过滤
    
    def test_depth_to_pointcloud(self, fusion):
        """测试深度图转点云"""
        h, w = 100, 100
        depth = np.ones((h, w))
        
        # 创建简单的深度图：中心深度为5米，边缘深度为1米
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - w/2)**2 + (y - h/2)**2)
                depth[y, x] = 1.0 + 4.0 * (1.0 - dist / (w/2))
        
        points = fusion._depth_to_pointcloud(np.ones((h, w, 3)), depth)
        
        # 验证点云数量
        assert points.shape[0] > 0
        assert points.shape[1] == 6
        
        # 验证点云范围
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        
        # 深度范围应该与输入一致
        assert np.all(z_coords >= 1.0)
        assert np.all(z_coords <= 5.0)
        
        # 中心点应该更远
        center_mask = (np.abs(x_coords) < 1.0) & (np.abs(y_coords) < 1.0)
        if np.any(center_mask):
            center_z = z_coords[center_mask]
            edge_z = z_coords[~center_mask]
            assert np.mean(center_z) > np.mean(edge_z)
    
    def test_project_point_to_image(self, fusion):
        """测试3D点投影到图像"""
        # 测试不同距离的点
        test_points = [
            (1.0, 0.0, 1.0),    # 右侧1米
            (0.0, 1.0, 1.0),    # 上方1米
            (-1.0, 0.0, 1.0),   # 左侧1米
            (0.0, 0.0, 1.0),    # 正前方1米
            (0.0, 0.0, -1.0),   # 相机后面（无效）
            (0.0, 0.0, 0.0),   # 相机位置（无效）
        ]
        
        for x, y, z in test_points:
            pixel = fusion.project_point_to_image((x, y, z))
            
            if z <= 0:
                # 负深度应该返回None
                assert pixel is None
            else:
                # 正深度应该返回有效像素
                assert pixel is not None
                px, py = pixel
                
                # 验证投影正确性
                # u = fx * x / z + cx
                # v = fy * y / z + cy
                expected_u = fusion.fx * x / z + fusion.cx
                expected_v = fusion.fy * y / z + fusion.cy
                
                assert abs(px - expected_u) < 1e-6
                assert abs(py - expected_v) < 1e-6


class TestObstacleDetector:
    """测试障碍物检测与融合"""
    
    @pytest.fixture
    def detector(self):
        """创建障碍物检测器实例"""
        return ObstacleDetector({
            "depth_threshold": 3.0,    # 深度阈值 (米)
            "laser_threshold": 5.0,    # 激光阈值 (米)
            "fusion_distance": 0.5      # 融合距离 (米)
        })
    
    def test_initialization(self, detector):
        """测试障碍物检测器初始化"""
        assert detector.depth_threshold == 3.0
        assert detector.laser_threshold == 5.0
        assert detector.fusion_distance == 0.5
    
    def test_detect_from_depth(self, detector):
        """测试从深度图检测障碍物"""
        # 创建测试深度图
        h, w = 240, 320
        
        # 创建有障碍物的深度图
        depth = np.ones((h, w)) * 10.0  # 默认远距离
        
        # 在不同区域添加障碍物
        # 左侧障碍物
        depth[:, :w//3] = 2.0
        # 中央障碍物
        depth[:, w//3:2*w//3] = 1.0
        # 右侧障碍物
        depth[:, 2*w//3:] = 2.5
        
        obstacles = detector.detect_from_depth(depth)
        
        # 验证检测结果
        assert len(obstacles) == 3  # 左、中、右各一个
        
        # 验证区域信息
        regions = [obs["region"] for obs in obstacles]
        assert "left" in regions
        assert "center" in regions
        assert "right" in regions
        
        # 验证距离信息
        distances = [obs["distance"] for obs in obstacles]
        left_dist = distances[regions.index("left")]
        center_dist = distances[regions.index("center")]
        right_dist = distances[regions.index("right")]
        
        assert center_dist < left_dist   # 中央障碍物更近
        assert center_dist < right_dist  # 中央障碍物更近
        
        # 验证置信度
        confidences = [obs["confidence"] for obs in obstacles]
        assert all(c > 0 for c in confidences)
    
    def test_detect_from_laser(self, detector):
        """测试从激光雷达检测障碍物"""
        # 创建测试激光数据
        angles = np.linspace(-np.pi, np.pi, 360)
        
        # 创建有障碍物的距离数据
        ranges = np.ones(360) * 10.0  # 默认远距离
        
        # 在不同区域添加障碍物
        # 前方障碍物
        front_indices = (angles > -np.pi/6) & (angles < np.pi/6)
        ranges[front_indices] = 2.0
        
        # 左侧障碍物
        left_indices = (angles > np.pi/2 - np.pi/6) & (angles < np.pi/2 + np.pi/6)
        ranges[left_indices] = 3.0
        
        # 右侧障碍物
        right_indices = (angles > -np.pi/2 - np.pi/6) & (angles < -np.pi/2 + np.pi/6)
        ranges[right_indices] = 2.5
        
        obstacles = detector.detect_from_laser(ranges.tolist(), angles.tolist())
        
        # 验证检测结果
        assert len(obstacles) == 3  # 前、左、右各一个
        
        # 验证区域信息
        regions = [obs["region"] for obs in obstacles]
        assert "front" in regions
        assert "left" in regions
        assert "right" in regions
        
        # 验证距离信息
        distances = [obs["distance"] for obs in obstacles]
        front_dist = distances[regions.index("front")]
        left_dist = distances[regions.index("left")]
        right_dist = distances[regions.index("right")]
        
        assert front_dist < left_dist   # 前方障碍物更近
        assert front_dist < right_dist  # 前方障碍物更近
    
    def test_fuse_obstacles(self, detector):
        """测试障碍物融合"""
        # 创建深度和激光障碍物
        depth_obstacles = [
            {"source": "depth", "region": "left", "distance": 2.0, "confidence": 0.8},
            {"source": "depth", "region": "center", "distance": 3.0, "confidence": 0.8},
            {"source": "depth", "region": "right", "distance": 4.0, "confidence": 0.8}
        ]
        
        laser_obstacles = [
            {"source": "laser", "region": "front", "distance": 2.5, "confidence": 0.9},
            {"source": "laser", "region": "left", "distance": 2.2, "confidence": 0.9},
            {"source": "laser", "region": "right", "distance": 3.8, "confidence": 0.9}
        ]
        
        fused_obstacles = detector.fuse_obstacles(depth_obstacles, laser_obstacles)
        
        # 验证融合结果
        assert len(fused_obstacles) == 4  # front, left, right, center
        
        # 验证融合障碍物
        fused_obstacle = None
        for obs in fused_obstacles:
            if obs["source"] == "fused" and obs["region"] == "left":
                fused_obstacle = obs
                break
        
        assert fused_obstacle is not None
        assert fused_obstacle["region"] == "left"
        assert abs(fused_obstacle["distance"] - 2.1) < 0.1  # (2.0 + 2.2) / 2
        assert fused_obstacle["confidence"] > 0.8  # 应该高于原始置信度
        
        # 验证未融合障碍物
        front_obstacle = None
        center_obstacle = None
        right_obstacle = None
        
        for obs in fused_obstacles:
            if obs["region"] == "front" and obs["source"] == "laser":
                front_obstacle = obs
            elif obs["region"] == "center" and obs["source"] == "depth":
                center_obstacle = obs
            elif obs["region"] == "right" and obs["source"] == "fused":
                right_obstacle = obs
        
        assert front_obstacle is not None
        assert center_obstacle is not None
        assert right_obstacle is not None
        
        assert front_obstacle["distance"] == 2.5
        assert center_obstacle["distance"] == 3.0
        assert right_obstacle["source"] == "fused"


class TestFusedPose:
    """测试融合位姿数据结构"""
    
    def test_creation(self):
        """测试创建融合位姿"""
        pose = FusedPose(
            x=1.0, y=2.0, z=0.5,
            roll=0.1, pitch=0.05, yaw=1.57
        )
        
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 0.5
        assert pose.roll == 0.1
        assert pose.pitch == 0.05
        assert pose.yaw == 1.57
        assert pose.covariance.shape == (6, 6)
        assert pose.timestamp is not None
    
    def test_to_array(self):
        """测试转换为数组"""
        pose = FusedPose(
            x=1.0, y=2.0, z=0.5,
            roll=0.1, pitch=0.05, yaw=1.57
        )
        
        pose_array = pose.to_array()
        
        assert pose_array.shape == (6,)
        assert pose_array[0] == 1.0   # x
        assert pose_array[1] == 2.0   # y
        assert pose_array[2] == 0.5   # z
        assert pose_array[3] == 0.1   # roll
        assert pose_array[4] == 0.05  # pitch
        assert pose_array[5] == 1.57  # yaw
    
    def test_from_array(self):
        """测试从数组创建"""
        pose_array = np.array([1.0, 2.0, 0.5, 0.1, 0.05, 1.57])
        pose = FusedPose.from_array(pose_array)
        
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 0.5
        assert pose.roll == 0.1
        assert pose.pitch == 0.05
        assert pose.yaw == 1.57
    
    def test_roundtrip(self):
        """测试往返转换"""
        original_pose = FusedPose(
            x=1.0, y=2.0, z=0.5,
            roll=0.1, pitch=0.05, yaw=1.57
        )
        
        pose_array = original_pose.to_array()
        new_pose = FusedPose.from_array(pose_array)
        
        assert original_pose.x == new_pose.x
        assert original_pose.y == new_pose.y
        assert original_pose.z == new_pose.z
        assert original_pose.roll == new_pose.roll
        assert original_pose.pitch == new_pose.pitch
        assert original_pose.yaw == new_pose.yaw


if __name__ == "__main__":
    pytest.main([__file__])








