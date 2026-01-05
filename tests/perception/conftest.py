"""
Perception模块测试配置文件

提供测试fixtures和通用工具
"""

import pytest
import numpy as np
from typing import Dict, Any
from datetime import datetime

from brain.perception.core.types import (
    Position2D,
    Position3D,
    Pose2D,
    Pose3D,
    Velocity,
    BoundingBox,
    DetectedObject,
    SceneDescription,
    OccupancyGrid
)
from brain.perception.core.enums import (
    CellState,
    ObjectType,
    DetectionMode
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_position_2d():
    """示例2D位置"""
    return Position2D(x=1.0, y=2.0)


@pytest.fixture
def sample_position_3d():
    """示例3D位置"""
    return Position3D(x=1.0, y=2.0, z=3.0)


@pytest.fixture
def sample_pose_2d():
    """示例2D位姿"""
    return Pose2D(x=1.0, y=2.0, theta=0.5)


@pytest.fixture
def sample_pose_3d():
    """示例3D位姿"""
    return Pose3D(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.5)


@pytest.fixture
def sample_velocity():
    """示例速度"""
    return Velocity(
        linear_x=1.0,
        linear_y=0.5,
        linear_z=0.0,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.1
    )


@pytest.fixture
def sample_bounding_box(sample_position_3d):
    """示例边界框"""
    return BoundingBox(
        min_point=Position3D(x=0.0, y=0.0, z=0.0),
        max_point=Position3D(x=2.0, y=2.0, z=2.0)
    )


@pytest.fixture
def sample_detected_object(sample_position_3d):
    """示例检测物体"""
    return DetectedObject(
        id="obj_001",
        label="person",
        confidence=0.95,
        position=sample_position_3d,
        description="A person standing"
    )


@pytest.fixture
def sample_occupancy_grid():
    """示例占据栅格"""
    return OccupancyGrid(
        width=100,
        height=100,
        resolution=0.1,
        origin_x=-5.0,
        origin_y=-5.0
    )


@pytest.fixture
def sample_scene_description(sample_detected_object):
    """示例场景描述"""
    return SceneDescription(
        summary="A scene with a person",
        objects=[sample_detected_object],
        spatial_relations=["person is in the center"],
        navigation_hints=["Path is clear"],
        potential_targets=["person"]
    )


@pytest.fixture
def mock_image():
    """模拟图像数据"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_depth_map():
    """模拟深度图"""
    return np.random.uniform(0.5, 10.0, (480, 640)).astype(np.float32)


@pytest.fixture
def mock_laser_scan():
    """模拟激光扫描数据"""
    import math
    num_points = 360
    ranges = []
    angles = []
    for i in range(num_points):
        angle = math.radians(i)
        ranges.append(5.0 + np.random.randn() * 0.1)
        angles.append(angle)
    return ranges, angles


@pytest.fixture
def mock_pointcloud():
    """模拟点云数据"""
    num_points = 1000
    points = np.random.randn(num_points, 3) * 2.0
    return points


@pytest.fixture
def perception_config() -> Dict[str, Any]:
    """感知层配置"""
    return {
        "mode": "fast",
        "confidence_threshold": 0.5,
        "resolution": 0.1,
        "map_size": 50.0
    }


# ==================== 辅助函数 ====================

def assert_position_equal(pos1: Position3D, pos2: Position3D, tol: float = 1e-6):
    """断言两个位置相等"""
    assert abs(pos1.x - pos2.x) < tol
    assert abs(pos1.y - pos2.y) < tol
    assert abs(pos1.z - pos2.z) < tol


def assert_bbox_equal(bbox1: BoundingBox, bbox2: BoundingBox, tol: float = 1e-6):
    """断言两个边界框相等"""
    assert_position_equal(bbox1.min_point, bbox2.min_point, tol)
    assert_position_equal(bbox1.max_point, bbox2.max_point, tol)


def create_mock_perception_data(
    has_laser: bool = True,
    has_pointcloud: bool = False,
    has_vlm: bool = False
):
    """创建模拟感知数据"""
    from dataclasses import dataclass

    @dataclass
    class MockPerceptionData:
        laser_ranges: list = None
        laser_angles: list = None
        pointcloud: np.ndarray = None
        pose: Pose2D = None
        scene_description: SceneDescription = None
        semantic_objects: list = None

    data = MockPerceptionData()

    if has_laser:
        import math
        num_points = 360
        data.laser_ranges = [5.0 + np.random.randn() * 0.1 for _ in range(num_points)]
        data.laser_angles = [math.radians(i) for i in range(num_points)]

    if has_pointcloud:
        num_points = 1000
        data.pointcloud = np.random.randn(num_points, 3) * 2.0

    data.pose = Pose2D(x=0.0, y=0.0, theta=0.0)

    if has_vlm:
        data.scene_description = sample_scene_description()
        data.semantic_objects = [sample_detected_object(Position3D())]

    return data
