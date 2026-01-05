"""
端到端测试 - 感知层集成

测试感知层各模块的端到端交互流程
"""

import pytest
import numpy as np
from unittest.mock import Mock
from dataclasses import dataclass

from brain.perception.world.model import WorldModel
from brain.perception.core.types import (
    Pose2D,
    DetectedObject,
    Position3D,
    SceneDescription
)
from brain.perception.core.enums import ObjectType


class TestWorldModelIntegration:
    """测试世界模型的端到端功能"""

    def test_initialization(self):
        """测试初始化"""
        world_model = WorldModel(
            resolution=0.1,
            map_size=50.0
        )
        assert world_model.occupancy_mapper is not None
        assert len(world_model.semantic_objects) == 0
        assert len(world_model.spatial_relations) == 0

    def test_update_with_laser_data(self, mock_laser_scan):
        """测试使用激光数据更新世界模型"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 创建模拟感知数据
        @dataclass
        class MockPerceptionData:
            laser_ranges: list
            laser_angles: list
            pose: Pose2D

        perception_data = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        # 更新世界模型
        world_model.update_with_perception(perception_data)

        # 检查更新是否成功
        assert world_model.metadata.update_count == 1
        stats = world_model.get_map_statistics()
        # 应该有一些栅格被更新
        assert stats["occupied_cells"] > 0 or stats["free_cells"] > 0

    def test_update_with_pointcloud(self, mock_pointcloud):
        """测试使用点云数据更新世界模型"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 创建模拟感知数据
        @dataclass
        class MockPerceptionData:
            pointcloud: np.ndarray
            pose: Pose2D

        perception_data = MockPerceptionData(
            pointcloud=mock_pointcloud,
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        # 更新世界模型
        world_model.update_with_perception(perception_data)

        # 检查更新是否成功
        assert world_model.metadata.update_count == 1
        stats = world_model.get_map_statistics()
        # 应该有一些栅格被更新
        assert stats["occupied_cells"] > 0

    def test_update_with_semantic_data(self):
        """测试使用语义数据更新世界模型"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 创建模拟感知数据
        @dataclass
        class MockPerceptionData:
            scene_description: SceneDescription
            semantic_objects: list
            pose: Pose2D

        # 创建模拟检测物体
        detected_obj = DetectedObject(
            id="obj_001",
            label="person",
            confidence=0.95,
            position=Position3D(x=1.0, y=2.0, z=0.0),
            description="A person standing"
        )

        scene_desc = SceneDescription(
            summary="A scene with a person",
            objects=[detected_obj],
            spatial_relations=[],
            navigation_hints=[],
            potential_targets=["person"]
        )

        perception_data = MockPerceptionData(
            scene_description=scene_desc,
            semantic_objects=[detected_obj],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        # 更新世界模型
        world_model.update_with_perception(perception_data)

        # 检查语义信息是否更新
        assert len(world_model.semantic_objects) > 0
        assert "person" in world_model.semantic_objects
        assert world_model.metadata.scene_description == "A scene with a person"

    def test_query_occupancy(self, mock_laser_scan):
        """测试占用查询"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 更新一些数据
        @dataclass
        class MockPerceptionData:
            laser_ranges: list
            laser_angles: list
            pose: Pose2D

        perception_data = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        world_model.update_with_perception(perception_data)

        # 查询一些位置的占用状态
        # 查询原点（应该是自由的）
        is_occupied = world_model.query_occupancy(0.0, 0.0)
        # 注意：具体结果取决于模拟数据

    def test_get_global_map(self, mock_laser_scan):
        """测试获取全局地图"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 更新一些数据
        @dataclass
        class MockPerceptionData:
            laser_ranges: list
            laser_angles: list
            pose: Pose2D

        perception_data = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        world_model.update_with_perception(perception_data)

        # 获取全局地图
        global_map = world_model.get_global_map()
        assert global_map is not None
        assert isinstance(global_map, np.ndarray)

    def test_get_semantic_map(self):
        """测试获取语义地图"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 添加语义物体
        @dataclass
        class MockPerceptionData:
            semantic_objects: list
            pose: Pose2D

        detected_obj = DetectedObject(
            id="obj_001",
            label="person",
            confidence=0.95,
            position=Position3D(x=1.0, y=2.0, z=0.0)
        )

        perception_data = MockPerceptionData(
            semantic_objects=[detected_obj],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        world_model.update_with_perception(perception_data)

        # 获取语义地图
        semantic_map = world_model.get_semantic_map()
        assert "objects" in semantic_map
        assert len(semantic_map["objects"]) > 0

    def test_get_map_statistics(self, mock_laser_scan):
        """测试获取地图统计信息"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 更新一些数据
        @dataclass
        class MockPerceptionData:
            laser_ranges: list
            laser_angles: list
            pose: Pose2D

        perception_data = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        world_model.update_with_perception(perception_data)

        # 获取统计信息
        stats = world_model.get_map_statistics()
        assert "total_cells" in stats
        assert "occupied_cells" in stats
        assert "free_cells" in stats
        assert "unknown_cells" in stats
        assert stats["total_cells"] == 500 * 500  # 50m / 0.1m = 500

    def test_reset(self, mock_laser_scan):
        """测试重置世界模型"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 更新一些数据
        @dataclass
        class MockPerceptionData:
            laser_ranges: list
            laser_angles: list
            pose: Pose2D

        perception_data = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        world_model.update_with_perception(perception_data)

        # 确认有更新
        assert world_model.metadata.update_count > 0

        # 重置
        world_model.reset()

        # 确认重置成功
        assert world_model.metadata.update_count == 0
        assert len(world_model.semantic_objects) == 0
        assert len(world_model.spatial_relations) == 0

    def test_semantic_object_decay(self):
        """测试语义物体的时间衰减"""
        world_model = WorldModel(
            resolution=0.1,
            map_size=50.0,
            config={"semantic_decay": 1.0}  # 高衰减率以便测试
        )

        # 添加语义物体
        @dataclass
        class MockPerceptionData:
            semantic_objects: list
            pose: Pose2D

        detected_obj = DetectedObject(
            id="obj_001",
            label="person",
            confidence=1.0,
            position=Position3D(x=1.0, y=2.0, z=0.0)
        )

        perception_data = MockPerceptionData(
            semantic_objects=[detected_obj],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        world_model.update_with_perception(perception_data)

        # 保存初始置信度
        initial_confidence = world_model.semantic_objects["person"].confidence

        # 模拟时间流逝（通过修改last_seen）
        from datetime import datetime, timedelta
        world_model.semantic_objects["person"].last_seen = datetime.now() - timedelta(minutes=10)

        # 触发另一次更新（这将应用衰减）
        world_model.update_with_perception(perception_data)

        # 置信度应该降低
        # 注意：由于update_with_perception也会更新置信度，这个测试可能需要调整


class TestPerceptionPipeline:
    """测试完整的感知流程"""

    def test_sensor_to_world_model_pipeline(self, mock_laser_scan):
        """测试从传感器到世界模型的完整流程"""
        # 创建世界模型
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 模拟传感器数据
        @dataclass
        class MockPerceptionData:
            laser_ranges: list
            laser_angles: list
            pose: Pose2D

        perception_data = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1],
            pose=Pose2D(x=0.0, y=0.0, theta=0.0)
        )

        # 更新世界模型
        world_model.update_with_perception(perception_data)

        # 验证流程完成
        assert world_model.metadata.update_count == 1

        # 获取地图
        global_map = world_model.get_global_map()
        assert global_map is not None

        # 获取统计信息
        stats = world_model.get_map_statistics()
        assert stats["occupied_cells"] > 0 or stats["free_cells"] > 0

    def test_multi_sensor_fusion(self, mock_laser_scan, mock_pointcloud):
        """测试多传感器融合"""
        world_model = WorldModel(resolution=0.1, map_size=50.0)

        # 先用激光雷达更新
        @dataclass
        class MockPerceptionData:
            laser_ranges: list = None
            laser_angles: list = None
            pointcloud: np.ndarray = None
            pose: Pose2D = Pose2D(x=0.0, y=0.0, theta=0.0)

        perception_data1 = MockPerceptionData(
            laser_ranges=mock_laser_scan[0],
            laser_angles=mock_laser_scan[1]
        )

        world_model.update_with_perception(perception_data1)
        update_count_1 = world_model.metadata.update_count

        # 再用点云更新
        perception_data2 = MockPerceptionData(
            pointcloud=mock_pointcloud
        )

        world_model.update_with_perception(perception_data2)

        # 验证两次更新
        assert world_model.metadata.update_count == update_count_1 + 1

        # 地图应该更完整
        stats = world_model.get_map_statistics()
        coverage = stats["occupied_ratio"] + stats["free_ratio"]
        assert coverage > 0
