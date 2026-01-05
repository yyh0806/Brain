"""
单元测试 - 核心数据类型

测试感知层的核心数据类型，包括：
- Position2D, Position3D
- Pose2D, Pose3D
- Velocity
- BoundingBox
- DetectedObject
- OccupancyGrid
"""

import pytest
import numpy as np

from brain.perception.core.types import (
    Position2D,
    Position3D,
    Pose2D,
    Pose3D,
    Velocity,
    BoundingBox,
    DetectedObject,
    OccupancyGrid
)
from brain.perception.core.enums import CellState


class TestPosition2D:
    """测试Position2D"""

    def test_creation(self):
        """测试创建"""
        pos = Position2D(x=1.0, y=2.0)
        assert pos.x == 1.0
        assert pos.y == 2.0

    def test_default_values(self):
        """测试默认值"""
        pos = Position2D()
        assert pos.x == 0.0
        assert pos.y == 0.0

    def test_distance_to(self):
        """测试距离计算"""
        pos1 = Position2D(x=0.0, y=0.0)
        pos2 = Position2D(x=3.0, y=4.0)
        distance = pos1.distance_to(pos2)
        assert abs(distance - 5.0) < 1e-6

    def test_to_dict(self):
        """测试转换为字典"""
        pos = Position2D(x=1.0, y=2.0)
        d = pos.to_dict()
        assert d == {"x": 1.0, "y": 2.0}

    def test_from_dict(self):
        """测试从字典创建"""
        d = {"x": 1.0, "y": 2.0}
        pos = Position2D.from_dict(d)
        assert pos.x == 1.0
        assert pos.y == 2.0


class TestPosition3D:
    """测试Position3D"""

    def test_creation(self):
        """测试创建"""
        pos = Position3D(x=1.0, y=2.0, z=3.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0

    def test_default_values(self):
        """测试默认值"""
        pos = Position3D()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.z == 0.0

    def test_distance_to(self):
        """测试距离计算"""
        pos1 = Position3D(x=0.0, y=0.0, z=0.0)
        pos2 = Position3D(x=1.0, y=2.0, z=2.0)
        distance = pos1.distance_to(pos2)
        assert abs(distance - 3.0) < 1e-6

    def test_to_dict(self):
        """测试转换为字典"""
        pos = Position3D(x=1.0, y=2.0, z=3.0)
        d = pos.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_to_tuple(self):
        """测试转换为元组"""
        pos = Position3D(x=1.0, y=2.0, z=3.0)
        t = pos.to_tuple()
        assert t == (1.0, 2.0, 3.0)

    def test_from_dict(self):
        """测试从字典创建"""
        d = {"x": 1.0, "y": 2.0, "z": 3.0}
        pos = Position3D.from_dict(d)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0


class TestPose2D:
    """测试Pose2D"""

    def test_creation(self):
        """测试创建"""
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.theta == 0.5

    def test_to_dict(self):
        """测试转换为字典"""
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)
        d = pose.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "theta": 0.5}

    def test_to_tuple(self):
        """测试转换为元组"""
        pose = Pose2D(x=1.0, y=2.0, theta=0.5)
        t = pose.to_tuple()
        assert t == (1.0, 2.0, 0.5)


class TestPose3D:
    """测试Pose3D"""

    def test_creation(self):
        """测试创建"""
        pose = Pose3D(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.5)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.roll == 0.1
        assert pose.pitch == 0.2
        assert pose.yaw == 0.5

    def test_to_dict(self):
        """测试转换为字典"""
        pose = Pose3D(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.5)
        d = pose.to_dict()
        assert d == {
            "x": 1.0, "y": 2.0, "z": 3.0,
            "roll": 0.1, "pitch": 0.2, "yaw": 0.5
        }

    def test_to_2d(self):
        """测试转换为2D位姿"""
        pose3d = Pose3D(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.5)
        pose2d = pose3d.to_2d()
        assert pose2d.x == 1.0
        assert pose2d.y == 2.0
        assert pose2d.theta == 0.5

    def test_to_tuple_2d(self):
        """测试转换为2D元组"""
        pose = Pose3D(x=1.0, y=2.0, z=3.0, yaw=0.5)
        t = pose.to_tuple_2d()
        assert t == (1.0, 2.0, 0.5)


class TestVelocity:
    """测试Velocity"""

    def test_creation(self):
        """测试创建"""
        vel = Velocity(
            linear_x=1.0,
            linear_y=0.5,
            linear_z=0.0,
            angular_z=0.1
        )
        assert vel.linear_x == 1.0
        assert vel.linear_y == 0.5
        assert vel.angular_z == 0.1

    def test_to_dict(self):
        """测试转换为字典"""
        vel = Velocity(linear_x=1.0, angular_z=0.1)
        d = vel.to_dict()
        assert "linear_x" in d
        assert d["linear_x"] == 1.0

    def test_from_dict(self):
        """测试从字典创建"""
        d = {
            "linear_x": 1.0,
            "linear_y": 0.5,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.1
        }
        vel = Velocity.from_dict(d)
        assert vel.linear_x == 1.0
        assert vel.angular_z == 0.1


class TestBoundingBox:
    """测试BoundingBox"""

    def test_creation(self, sample_position_3d):
        """测试创建"""
        bbox = BoundingBox(
            min_point=Position3D(x=0.0, y=0.0, z=0.0),
            max_point=Position3D(x=2.0, y=2.0, z=2.0)
        )
        assert bbox.min_point.x == 0.0
        assert bbox.max_point.x == 2.0

    def test_center(self, sample_position_3d):
        """测试中心点计算"""
        bbox = BoundingBox(
            min_point=Position3D(x=0.0, y=0.0, z=0.0),
            max_point=Position3D(x=2.0, y=2.0, z=2.0)
        )
        center = bbox.center
        assert center.x == 1.0
        assert center.y == 1.0
        assert center.z == 1.0

    def test_size(self):
        """测试尺寸计算"""
        bbox = BoundingBox(
            min_point=Position3D(x=0.0, y=0.0, z=0.0),
            max_point=Position3D(x=2.0, y=3.0, z=4.0)
        )
        size = bbox.size
        assert size == (2.0, 3.0, 4.0)

    def test_contains(self):
        """测试包含检查"""
        bbox = BoundingBox(
            min_point=Position3D(x=0.0, y=0.0, z=0.0),
            max_point=Position3D(x=2.0, y=2.0, z=2.0)
        )
        # 内部点
        assert bbox.contains(Position3D(x=1.0, y=1.0, z=1.0))
        # 外部点
        assert not bbox.contains(Position3D(x=3.0, y=3.0, z=3.0))

    def test_intersects(self):
        """测试相交检查"""
        bbox1 = BoundingBox(
            min_point=Position3D(x=0.0, y=0.0, z=0.0),
            max_point=Position3D(x=2.0, y=2.0, z=2.0)
        )
        # 相交的边界框
        bbox2 = BoundingBox(
            min_point=Position3D(x=1.0, y=1.0, z=1.0),
            max_point=Position3D(x=3.0, y=3.0, z=3.0)
        )
        assert bbox1.intersects(bbox2)

        # 不相交的边界框
        bbox3 = BoundingBox(
            min_point=Position3D(x=3.0, y=3.0, z=3.0),
            max_point=Position3D(x=5.0, y=5.0, z=5.0)
        )
        assert not bbox1.intersects(bbox3)


class TestDetectedObject:
    """测试DetectedObject"""

    def test_creation(self, sample_position_3d):
        """测试创建"""
        obj = DetectedObject(
            id="obj_001",
            label="person",
            confidence=0.95,
            position=sample_position_3d
        )
        assert obj.id == "obj_001"
        assert obj.label == "person"
        assert obj.confidence == 0.95
        assert obj.position == sample_position_3d

    def test_to_dict(self, sample_position_3d):
        """测试转换为字典"""
        obj = DetectedObject(
            id="obj_001",
            label="person",
            confidence=0.95,
            position=sample_position_3d,
            description="A person"
        )
        d = obj.to_dict()
        assert d["id"] == "obj_001"
        assert d["label"] == "person"
        assert d["confidence"] == 0.95
        assert "position" in d
        assert d["description"] == "A person"


class TestOccupancyGrid:
    """测试OccupancyGrid"""

    def test_creation(self):
        """测试创建"""
        grid = OccupancyGrid(
            width=100,
            height=100,
            resolution=0.1,
            origin_x=-5.0,
            origin_y=-5.0
        )
        assert grid.width == 100
        assert grid.height == 100
        assert grid.resolution == 0.1
        assert grid.origin_x == -5.0
        assert grid.origin_y == -5.0
        # 检查数据初始化
        assert grid.data.shape == (100, 100)
        assert np.all(grid.data == CellState.UNKNOWN)

    def test_world_to_grid(self):
        """测试世界坐标转栅格坐标"""
        grid = OccupancyGrid(
            width=100,
            height=100,
            resolution=0.1,
            origin_x=-5.0,
            origin_y=-5.0
        )
        # 原点应该对应栅格(50, 50)
        gx, gy = grid.world_to_grid(0.0, 0.0)
        assert gx == 50
        assert gy == 50

    def test_grid_to_world(self):
        """测试栅格坐标转世界坐标"""
        grid = OccupancyGrid(
            width=100,
            height=100,
            resolution=0.1,
            origin_x=-5.0,
            origin_y=-5.0
        )
        # 栅格(50, 50)应该对应原点
        x, y = grid.grid_to_world(50, 50)
        assert abs(x - 0.0) < 1e-6
        assert abs(y - 0.0) < 1e-6

    def test_is_valid(self):
        """测试有效性检查"""
        grid = OccupancyGrid(width=100, height=100, resolution=0.1)
        # 有效坐标
        assert grid.is_valid(0, 0)
        assert grid.is_valid(99, 99)
        # 无效坐标
        assert not grid.is_valid(-1, 0)
        assert not grid.is_valid(100, 0)
        assert not grid.is_valid(0, 100)

    def test_set_and_get_cell(self):
        """测试设置和获取栅格"""
        grid = OccupancyGrid(width=100, height=100, resolution=0.1)
        # 设置栅格
        grid.set_cell(50, 50, CellState.OCCUPIED)
        # 获取栅格
        state = grid.get_cell(50, 50)
        assert state == CellState.OCCUPIED

    def test_is_occupied(self):
        """测试占据检查"""
        grid = OccupancyGrid(width=100, height=100, resolution=0.1)
        grid.set_cell(50, 50, CellState.OCCUPIED)
        assert grid.is_occupied(50, 50)
        assert not grid.is_occupied(51, 51)

    def test_is_free(self):
        """测试自由检查"""
        grid = OccupancyGrid(width=100, height=100, resolution=0.1)
        grid.set_cell(50, 50, CellState.FREE)
        assert grid.is_free(50, 50)
        assert not grid.is_free(51, 51)

    def test_is_unknown(self):
        """测试未知检查"""
        grid = OccupancyGrid(width=100, height=100, resolution=0.1)
        # 默认状态是未知
        assert grid.is_unknown(50, 50)
        # 设置为占据后不再未知
        grid.set_cell(50, 50, CellState.OCCUPIED)
        assert not grid.is_unknown(50, 50)
