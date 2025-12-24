"""
占据栅格地图单元测试

测试占据栅格地图生成器，包括地图创建、更新、查询和坐标转换功能。
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch

from brain.perception.mapping.occupancy_mapper import (
    OccupancyGrid, OccupancyMapper, CellState
)


class TestOccupancyGrid:
    """测试占据栅格地图数据结构"""
    
    @pytest.fixture
    def grid(self):
        """创建测试栅格地图"""
        return OccupancyGrid(width=100, height=100, resolution=0.1)
    
    def test_initialization(self, grid):
        """测试栅格地图初始化"""
        assert grid.width == 100
        assert grid.height == 100
        assert grid.resolution == 0.1
        assert grid.origin_x == 0.0
        assert grid.origin_y == 0.0
        assert grid.data.shape == (100, 100)
        
        # 初始状态应该全部未知
        assert np.all(grid.data == CellState.UNKNOWN)
    
    def test_custom_initialization(self):
        """测试自定义初始化参数"""
        custom_grid = OccupancyGrid(
            width=200, 
            height=150, 
            resolution=0.05,
            origin_x=-10.0,
            origin_y=-5.0
        )
        
        assert custom_grid.width == 200
        assert custom_grid.height == 150
        assert custom_grid.resolution == 0.05
        assert custom_grid.origin_x == -10.0
        assert custom_grid.origin_y == -5.0
    
    def test_world_to_grid_conversion(self, grid):
        """测试世界坐标到栅格坐标转换"""
        # 测试基本转换
        grid_x, grid_y = grid.world_to_grid(5.0, 3.0)
        expected_x = int((5.0 - grid.origin_x) / grid.resolution)
        expected_y = int((3.0 - grid.origin_y) / grid.resolution)
        assert grid_x == expected_x
        assert grid_y == expected_y
        
        # 测试负坐标
        grid_x, grid_y = grid.world_to_grid(-2.0, -1.5)
        expected_x = int((-2.0 - grid.origin_x) / grid.resolution)
        expected_y = int((-1.5 - grid.origin_y) / grid.resolution)
        assert grid_x == expected_x
        assert grid_y == expected_y
        
        # 测试分数坐标（应该向下取整）
        grid_x, grid_y = grid.world_to_grid(5.05, 3.07)
        expected_x = int((5.05 - grid.origin_x) / grid.resolution)
        expected_y = int((3.07 - grid.origin_y) / grid.resolution)
        assert grid_x == 50
        assert grid_y == 30
    
    def test_grid_to_world_conversion(self, grid):
        """测试栅格坐标到世界坐标转换"""
        # 测试基本转换
        world_x, world_y = grid.grid_to_world(50, 30)
        expected_x = 50 * grid.resolution + grid.origin_x
        expected_y = 30 * grid.resolution + grid.origin_y
        assert abs(world_x - expected_x) < 1e-10
        assert abs(world_y - expected_y) < 1e-10
        
        # 测试负栅格坐标
        world_x, world_y = grid.grid_to_world(-20, -15)
        expected_x = -20 * grid.resolution + grid.origin_x
        expected_y = -15 * grid.resolution + grid.origin_y
        assert abs(world_x - expected_x) < 1e-10
        assert abs(world_y - expected_y) < 1e-10
    
    def test_coordinate_conversion_roundtrip(self, grid):
        """测试坐标转换往返"""
        # 世界坐标 -> 栅格坐标 -> 世界坐标
        original_world_x, original_world_y = 5.05, 3.07
        grid_x, grid_y = grid.world_to_grid(original_world_x, original_world_y)
        converted_world_x, converted_world_y = grid.grid_to_world(grid_x, grid_y)
        
        # 由于栅格化，可能存在精度损失
        assert abs(converted_world_x - original_world_x) < grid.resolution
        assert abs(converted_world_y - original_world_y) < grid.resolution
        
        # 栅格坐标 -> 世界坐标 -> 栅格坐标
        original_grid_x, original_grid_y = 50, 30
        world_x, world_y = grid.grid_to_world(original_grid_x, original_grid_y)
        converted_grid_x, converted_grid_y = grid.world_to_grid(world_x, world_y)
        
        assert converted_grid_x == original_grid_x
        assert converted_grid_y == original_grid_y
    
    def test_is_valid(self, grid):
        """测试栅格坐标有效性检查"""
        # 有效坐标
        assert grid.is_valid(0, 0)
        assert grid.is_valid(50, 50)
        assert grid.is_valid(99, 99)
        
        # 无效坐标
        assert not grid.is_valid(-1, 0)
        assert not grid.is_valid(0, -1)
        assert not grid.is_valid(100, 0)
        assert not grid.is_valid(0, 100)
        assert not grid.is_valid(100, 100)
    
    def test_cell_operations(self, grid):
        """测试栅格单元操作"""
        # 测试设置和获取
        grid.set_cell(50, 50, CellState.OCCUPIED)
        assert grid.get_cell(50, 50) == CellState.OCCUPIED
        
        grid.set_cell(25, 25, CellState.FREE)
        assert grid.get_cell(25, 25) == CellState.FREE
        
        # 测试无效坐标
        assert grid.get_cell(-1, 0) == CellState.UNKNOWN
        assert grid.get_cell(100, 0) == CellState.UNKNOWN
        
        # 无效坐标的set_cell不应该抛出异常
        grid.set_cell(-1, 0, CellState.FREE)
        grid.set_cell(100, 0, CellState.OCCUPIED)
    
    def test_state_checkers(self, grid):
        """测试状态检查方法"""
        # 设置不同状态的栅格
        grid.set_cell(10, 10, CellState.FREE)
        grid.set_cell(20, 20, CellState.OCCUPIED)
        grid.set_cell(30, 30, CellState.UNKNOWN)
        
        # 检查状态
        assert grid.is_free(10, 10)
        assert not grid.is_free(20, 20)
        assert not grid.is_free(30, 30)
        
        assert grid.is_occupied(20, 20)
        assert not grid.is_occupied(10, 10)
        assert not grid.is_occupied(30, 30)
        
        assert grid.is_unknown(30, 30)
        assert not grid.is_unknown(10, 10)
        assert not grid.is_unknown(20, 20)
        
        # 检查无效坐标
        assert not grid.is_occupied(-1, 0)
        assert not grid.is_free(100, 0)
        assert not grid.is_unknown(0, -1)


class TestOccupancyMapper:
    """测试占据栅格地图生成器"""
    
    @pytest.fixture
    def mapper(self):
        """创建占据栅格地图生成器"""
        return OccupancyMapper(
            resolution=0.1,
            map_size=10.0,
            camera_fov=1.57,  # 90度
            camera_range=10.0,
            lidar_range=30.0
        )
    
    def test_initialization(self, mapper):
        """测试地图生成器初始化"""
        assert mapper.resolution == 0.1
        assert mapper.map_size == 10.0
        assert mapper.camera_fov == 1.57
        assert mapper.camera_range == 10.0
        assert mapper.lidar_range == 30.0
        
        # 检查栅格地图尺寸
        grid_size = int(mapper.map_size / mapper.resolution)
        assert mapper.grid.width == grid_size
        assert mapper.grid.height == grid_size
        
        # 检查相机参数
        assert mapper.camera_fx == 525.0
        assert mapper.camera_fy == 525.0
        assert mapper.camera_cx == 320.0
        assert mapper.camera_cy == 240.0
        
        # 检查更新参数
        assert mapper.occupied_prob == 0.7
        assert mapper.free_prob == 0.3
        assert mapper.min_depth == 0.1
        assert mapper.max_depth == 10.0
    
    def test_custom_initialization(self):
        """测试自定义初始化参数"""
        custom_mapper = OccupancyMapper(
            resolution=0.05,
            map_size=20.0,
            camera_fov=2.09,  # 120度
            camera_range=15.0,
            lidar_range=50.0,
            config={
                "camera_fx": 600.0,
                "camera_fy": 600.0,
                "camera_cx": 320.0,
                "camera_cy": 240.0,
                "occupied_prob": 0.8,
                "free_prob": 0.2,
                "min_depth": 0.05,
                "max_depth": 15.0
            }
        )
        
        assert custom_mapper.resolution == 0.05
        assert custom_mapper.map_size == 20.0
        assert custom_mapper.camera_fov == 2.09
        assert custom_mapper.camera_range == 15.0
        assert custom_mapper.lidar_range == 50.0
        
        assert custom_mapper.camera_fx == 600.0
        assert custom_mapper.camera_fy == 600.0
        assert custom_mapper.camera_cx == 320.0
        assert custom_mapper.camera_cy == 240.0
        
        assert custom_mapper.occupied_prob == 0.8
        assert custom_mapper.free_prob == 0.2
        assert custom_mapper.min_depth == 0.05
        assert custom_mapper.max_depth == 15.0
    
    def test_depth_update(self, mapper):
        """测试从深度图更新地图"""
        # 创建测试深度图
        h, w = 240, 320
        depth = np.ones((h, w)) * 5.0  # 5米远
        
        # 在不同区域添加障碍物（更近的距离）
        # 左侧障碍物
        depth[:, :w//3] = 2.0
        # 中央障碍物
        depth[:, w//3:2*w//3] = 1.0
        # 右侧障碍物
        depth[:, 2*w//3:] = 2.5
        
        # 更新地图
        mapper.update_from_depth(depth, pose=(0, 0, 0))
        
        # 检查地图更新
        grid = mapper.get_grid()
        
        # 应该有占据和自由空间
        occupied_count = np.sum(grid.data == CellState.OCCUPIED)
        free_count = np.sum(grid.data == CellState.FREE)
        
        assert occupied_count > 0
        assert free_count > 0
        
        # 检查障碍物位置
        # 中央障碍物应该在机器人正前方
        center_x = w // 2
        center_y = h // 2
        grid_x, grid_y = mapper.grid.world_to_grid(0, 1.0)  # 前方1米
        assert grid.is_occupied(grid_x, grid_y)
    
    def test_depth_update_with_pose(self, mapper):
        """测试带位姿的深度图更新"""
        # 创建测试深度图
        h, w = 240, 320
        depth = np.ones((h, w)) * 5.0
        
        # 在中央添加障碍物
        depth[:, w//3:2*w//3] = 1.0
        
        # 机器人位于(2, 1)，朝向PI/2（向左）
        robot_x, robot_y, robot_yaw = 2.0, 1.0, math.pi / 2
        
        # 更新地图
        mapper.update_from_depth(depth, pose=(robot_x, robot_y, robot_yaw))
        
        # 障碍物应该在机器人上方（在机器人坐标系中前方）
        grid = mapper.get_grid()
        
        # 机器人位置
        robot_grid_x, robot_grid_y = mapper.grid.world_to_grid(robot_x, robot_y)
        assert grid.is_free(robot_grid_x, robot_grid_y)
        
        # 障碍物应该在机器人的上方
        obstacle_grid_x, obstacle_grid_y = mapper.grid.world_to_grid(robot_x, robot_y + 1.0)
        assert grid.is_occupied(obstacle_grid_x, obstacle_grid_y)
    
    def test_depth_update_with_camera_pose(self, mapper):
        """测试带相机位姿的深度图更新"""
        # 创建测试深度图
        h, w = 240, 320
        depth = np.ones((h, w)) * 5.0
        
        # 在中央添加障碍物
        depth[:, w//3:2*w//3] = 1.0
        
        # 机器人位姿
        robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        
        # 相机相对于机器人
        camera_x, camera_y, camera_yaw = 1.0, 0.0, 0.0
        
        # 更新地图
        mapper.update_from_depth(depth, pose=(robot_x, robot_y, robot_yaw), camera_pose=(camera_x, camera_y, camera_yaw))
        
        # 障碍物应该在机器人右方1米处
        grid = mapper.get_grid()
        
        # 机器人位置
        robot_grid_x, robot_grid_y = mapper.grid.world_to_grid(robot_x, robot_y)
        assert grid.is_free(robot_grid_x, robot_grid_y)
        
        # 障碍物位置
        obstacle_grid_x, obstacle_grid_y = mapper.grid.world_to_grid(camera_x, camera_y)
        assert grid.is_occupied(obstacle_grid_x, obstacle_grid_y)
    
    def test_laser_update(self, mapper):
        """测试从激光雷达更新地图"""
        # 创建测试激光数据
        angles = np.linspace(-np.pi, np.pi, 360)
        ranges = np.ones(360) * 5.0  # 默认5米远
        
        # 在不同区域添加障碍物（更近的距离）
        # 前方障碍物
        front_indices = (angles > -np.pi/6) & (angles < np.pi/6)
        ranges[front_indices] = 2.0
        
        # 左侧障碍物
        left_indices = (angles > np.pi/2 - np.pi/6) & (angles < np.pi/2 + np.pi/6)
        ranges[left_indices] = 1.5
        
        # 右侧障碍物
        right_indices = (angles > -np.pi/2 - np.pi/6) & (angles < -np.pi/2 + np.pi/6)
        ranges[right_indices] = 3.0
        
        # 更新地图
        mapper.update_from_laser(ranges.tolist(), angles.tolist(), pose=(0, 0, 0))
        
        # 检查地图更新
        grid = mapper.get_grid()
        
        # 应该有占据和自由空间
        occupied_count = np.sum(grid.data == CellState.OCCUPIED)
        free_count = np.sum(grid.data == CellState.FREE)
        
        assert occupied_count > 0
        assert free_count > 0
        
        # 检查障碍物位置
        # 前方障碍物
        front_grid_x, front_grid_y = mapper.grid.world_to_grid(0, 2.0)
        assert grid.is_occupied(front_grid_x, front_grid_y)
        
        # 左侧障碍物
        left_grid_x, left_grid_y = mapper.grid.world_to_grid(-1.5, 0)
        assert grid.is_occupied(left_grid_x, left_grid_y)
        
        # 右侧障碍物
        right_grid_x, right_grid_y = mapper.grid.world_to_grid(3.0, 0)
        assert grid.is_occupied(right_grid_x, right_grid_y)
    
    def test_laser_update_with_pose(self, mapper):
        """测试带位姿的激光雷达更新"""
        # 创建测试激光数据
        angles = np.linspace(-np.pi, np.pi, 360)
        ranges = np.ones(360) * 5.0
        
        # 在前方添加障碍物
        front_indices = (angles > -np.pi/6) & (angles < np.pi/6)
        ranges[front_indices] = 2.0
        
        # 机器人位于(2, 1)，朝向PI/2（向左）
        robot_x, robot_y, robot_yaw = 2.0, 1.0, math.pi / 2
        
        # 更新地图
        mapper.update_from_laser(ranges.tolist(), angles.tolist(), pose=(robot_x, robot_y, robot_yaw))
        
        # 障碍物应该在机器人上方（在机器人坐标系中前方）
        grid = mapper.get_grid()
        
        # 机器人位置
        robot_grid_x, robot_grid_y = mapper.grid.world_to_grid(robot_x, robot_y)
        assert grid.is_free(robot_grid_x, robot_grid_y)
        
        # 障碍物应该在机器人的上方
        obstacle_grid_x, obstacle_grid_y = mapper.grid.world_to_grid(robot_x, robot_y + 2.0)
        assert grid.is_occupied(obstacle_grid_x, obstacle_grid_y)
    
    def test_pointcloud_update(self, mapper):
        """测试从点云更新地图"""
        # 创建测试点云
        # 机器人周围的点（自由空间）
        free_points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ])
        
        # 障碍物点
        obstacle_points = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0]
        ])
        
        # 组合点云
        pointcloud = np.vstack([free_points, obstacle_points])
        
        # 更新地图
        mapper.update_from_pointcloud(pointcloud, pose=(0, 0, 0))
        
        # 检查地图更新
        grid = mapper.get_grid()
        
        # 障碍物点应该被标记为占据
        for point in obstacle_points:
            grid_x, grid_y = mapper.grid.world_to_grid(point[0], point[1])
            assert grid.is_occupied(grid_x, grid_y)
    
    def test_pointcloud_update_with_pose(self, mapper):
        """测试带位姿的点云更新"""
        # 创建测试点云（机器人坐标系）
        free_points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        obstacle_points = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0]
        ])
        
        # 组合点云
        pointcloud = np.vstack([free_points, obstacle_points])
        
        # 机器人位于(2, 1)，朝向PI/2（向左）
        robot_x, robot_y, robot_yaw = 2.0, 1.0, math.pi / 2
        
        # 更新地图
        mapper.update_from_pointcloud(pointcloud, pose=(robot_x, robot_y, robot_yaw))
        
        # 检查地图更新
        grid = mapper.get_grid()
        
        # 障碍物点应该在转换后的位置被标记为占据
        # (2.0, 0.0) 在机器人坐标系中应该转换为(2.0, 2.0) 在世界坐标系中
        obstacle_world_x = robot_x + 2.0  # 4.0
        obstacle_world_y = robot_y + 0.0  # 1.0
        
        obstacle_grid_x, obstacle_grid_y = mapper.grid.world_to_grid(obstacle_world_x, obstacle_world_y)
        assert grid.is_occupied(obstacle_grid_x, obstacle_grid_y)
    
    def test_invalid_data_handling(self, mapper):
        """测试无效数据处理"""
        # 空深度图
        mapper.update_from_depth(None, pose=(0, 0, 0))
        mapper.update_from_depth(np.array([]), pose=(0, 0, 0))
        
        # 空激光数据
        mapper.update_from_laser([], [], pose=(0, 0, 0))
        mapper.update_from_laser(None, None, pose=(0, 0, 0))
        mapper.update_from_laser([1, 2, 3], [1, 2], pose=(0, 0, 0))
        
        # 空点云
        mapper.update_from_pointcloud(None, pose=(0, 0, 0))
        mapper.update_from_pointcloud(np.array([]), pose=(0, 0, 0))
        
        # 这些操作应该不会抛出异常
        grid = mapper.get_grid()
        assert grid is not None
    
    def test_nearest_obstacle(self, mapper):
        """测试最近障碍物查找"""
        # 手动设置一些障碍物
        mapper.grid.set_cell(50, 50, CellState.OCCUPIED)
        mapper.grid.set_cell(60, 60, CellState.OCCUPIED)
        mapper.grid.set_cell(70, 70, CellState.OCCUPIED)
        
        # 查询最近障碍物
        nearest = mapper.get_nearest_obstacle(0, 0, max_range=10.0)
        
        assert nearest is not None
        obs_x, obs_y, distance = nearest
        
        # 验证返回的是最近障碍物
        # 将栅格坐标转换为世界坐标
        world_x = obs_x * mapper.resolution + mapper.grid.origin_x
        world_y = obs_y * mapper.resolution + mapper.grid.origin_y
        
        # 检查返回的坐标是已设置的障碍物之一
        is_obstacle = (
            (obs_x == 50 and obs_y == 50) or
            (obs_x == 60 and obs_y == 60) or
            (obs_x == 70 and obs_y == 70)
        )
        assert is_obstacle
        
        # 检查距离计算
        expected_distance = math.sqrt(world_x**2 + world_y**2)
        assert abs(distance - expected_distance) < 1e-6
    
    def test_nearest_obstacle_not_found(self, mapper):
        """测试未找到障碍物的情况"""
        # 清空地图，确保没有障碍物
        mapper.reset()
        
        # 查询最近障碍物
        nearest = mapper.get_nearest_obstacle(0, 0, max_range=10.0)
        
        # 应该返回None
        assert nearest is None
    
    def test_nearest_obstacle_out_of_range(self, mapper):
        """测试超出范围的障碍物"""
        # 手动设置一些障碍物
        mapper.grid.set_cell(50, 50, CellState.OCCUPIED)
        
        # 查询最近障碍物，但设置最大范围小于障碍物距离
        world_x = 50 * mapper.resolution + mapper.grid.origin_x  # 5.0
        world_y = 50 * mapper.resolution + mapper.grid.origin_y  # 5.0
        obstacle_distance = math.sqrt(world_x**2 + world_y**2)
        
        # 设置比障碍物距离小的最大范围
        max_range = obstacle_distance * 0.9
        
        nearest = mapper.get_nearest_obstacle(0, 0, max_range=max_range)
        
        # 应该返回None
        assert nearest is None
    
    def test_is_occupied_at(self, mapper):
        """测试世界坐标位置占据检查"""
        # 手动设置一个障碍物
        grid_x, grid_y = 50, 50
        mapper.grid.set_cell(grid_x, grid_y, CellState.OCCUPIED)
        
        # 转换为世界坐标
        world_x, world_y = mapper.grid.grid_to_world(grid_x, grid_y)
        
        # 检查占据状态
        assert mapper.is_occupied_at(world_x, world_y)
        
        # 检查非占据位置
        assert not mapper.is_occupied_at(0, 0)
    
    def test_is_free_at(self, mapper):
        """测试世界坐标位置自由检查"""
        # 手动设置一个自由位置
        grid_x, grid_y = 25, 25
        mapper.grid.set_cell(grid_x, grid_y, CellState.FREE)
        
        # 转换为世界坐标
        world_x, world_y = mapper.grid.grid_to_world(grid_x, grid_y)
        
        # 检查自由状态
        assert mapper.is_free_at(world_x, world_y)
        
        # 检查占据位置
        mapper.grid.set_cell(50, 50, CellState.OCCUPIED)
        occupied_world_x, occupied_world_y = mapper.grid.grid_to_world(50, 50)
        assert not mapper.is_free_at(occupied_world_x, occupied_world_y)
    
    def test_statistics(self, mapper):
        """测试地图统计信息"""
        # 手动设置一些栅格状态
        # 占据栅格
        for i in range(10):
            for j in range(10):
                mapper.grid.set_cell(i, j, CellState.OCCUPIED)
        
        # 自由栅格
        for i in range(10, 20):
            for j in range(10, 20):
                mapper.grid.set_cell(i, j, CellState.FREE)
        
        # 获取统计信息
        stats = mapper.get_statistics()
        
        # 验证统计信息
        assert stats["total_cells"] == mapper.grid.width * mapper.grid.height
        assert stats["occupied"] == 100
        assert stats["free"] == 100
        assert stats["unknown"] == stats["total_cells"] - 200
        assert stats["coverage"] == (stats["occupied"] + stats["free"]) / stats["total_cells"]
        assert stats["resolution"] == mapper.resolution
        assert stats["map_size"] == mapper.map_size
    
    def test_reset(self, mapper):
        """测试地图重置"""
        # 设置一些栅格状态
        mapper.grid.set_cell(50, 50, CellState.OCCUPIED)
        mapper.grid.set_cell(25, 25, CellState.FREE)
        
        # 确认设置成功
        assert mapper.grid.is_occupied(50, 50)
        assert mapper.grid.is_free(25, 25)
        
        # 重置地图
        mapper.reset()
        
        # 确认所有栅格回到未知状态
        assert np.all(mapper.grid.data == CellState.UNKNOWN)


if __name__ == "__main__":
    pytest.main([__file__])


