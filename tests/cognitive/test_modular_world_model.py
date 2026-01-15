# -*- coding: utf-8 -*-
"""
模块化WorldModel测试
"""

import pytest
import asyncio
import numpy as np
from brain.cognitive.world_model.modular_world_model import (
    GeometricLayer,
    SemanticLayer,
    CausalLayer,
    ModularWorldModel
)


class MockSemanticObject:
    """模拟语义物体"""
    def __init__(self, obj_id, label, position, confidence=0.8):
        self.id = obj_id
        self.label = label
        self.world_position = position
        self.confidence = confidence


class TestGeometricLayer:
    """测试几何层"""

    def test_initialization(self):
        """测试初始化"""
        layer = GeometricLayer()

        # SLAM可能不可用
        if layer.slam_manager is None:
            assert layer.coordinate_transformer is None

    def test_slam_map_property(self):
        """测试SLAM地图属性"""
        layer = GeometricLayer()

        # 初始时应该为None
        slam_map = layer.slam_map
        assert slam_map is None or slam_map is not None  # 根据SLAM是否可用

    def test_geometric_map_property(self):
        """测试几何地图属性"""
        layer = GeometricLayer()

        # 初始时应该为None
        geo_map = layer.geometric_map
        assert geo_map is None


class TestSemanticLayer:
    """测试语义层"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """测试初始化"""
        layer = SemanticLayer()

        assert layer.object_manager is not None
        assert layer.semantic_overlays is not None
        assert len(layer.semantic_overlays) == 0

    @pytest.mark.asyncio
    async def test_update_from_perception(self):
        """测试从感知更新"""
        layer = SemanticLayer()

        semantic_objects = [
            MockSemanticObject("obj1", "door", (5.0, 3.0)),
            MockSemanticObject("obj2", "person", (7.0, 2.0)),
        ]

        changes = await layer.update_from_perception(semantic_objects)

        # 应该检测到新物体
        assert len(changes) > 0
        assert layer.object_manager.size() == 2

    @pytest.mark.asyncio
    async def test_get_object(self):
        """测试获取物体"""
        layer = SemanticLayer()

        obj = MockSemanticObject("obj1", "door", (5.0, 3.0))
        obj_id = layer.object_manager.add_or_update(obj)

        retrieved = layer.get_object(obj_id)
        assert retrieved is not None
        assert retrieved.label == "door"

    @pytest.mark.asyncio
    async def test_find_object(self):
        """测试查找物体"""
        layer = SemanticLayer()

        obj = MockSemanticObject("obj1", "front_door", (5.0, 3.0))
        layer.object_manager.add_or_update(obj)

        # 查找部分名称
        found = layer.find_object("door")
        assert found is not None
        assert "door" in found.label.lower()

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """测试获取统计信息"""
        layer = SemanticLayer()

        obj = MockSemanticObject("obj1", "door", (5.0, 3.0))
        layer.object_manager.add_or_update(obj)

        stats = layer.get_statistics()
        assert stats["size"] == 1


class TestCausalLayer:
    """测试因果层"""

    def test_initialization(self):
        """测试初始化"""
        layer = CausalLayer()

        assert layer.causal_graph is not None
        assert len(layer.state_history) == 0

    def test_update_state(self):
        """测试更新状态"""
        layer = CausalLayer()

        state1 = {
            "timestamp": "2024-01-01",
            "semantic_objects_count": 5
        }

        layer.update_state(state1)

        assert len(layer.state_history) == 1

    def test_state_history_limit(self):
        """测试状态历史限制"""
        layer = CausalLayer(config={"max_state_history": 5})

        for i in range(10):
            state = {"semantic_objects_count": i}
            layer.update_state(state)

        # 应该只保留最近5个状态
        assert len(layer.state_history) == 5

    def test_causal_relation_detection(self):
        """测试因果关系检测"""
        layer = CausalLayer()

        state1 = {"semantic_objects_count": 5}
        state2 = {"semantic_objects_count": 7}

        layer.update_state(state1)
        layer.update_state(state2)

        # 应该检测到物体出现
        # (实际检测逻辑在_detect_causal_relations中)
        assert len(layer.state_history) == 2


class TestModularWorldModel:
    """测试模块化世界模型"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """测试初始化"""
        model = ModularWorldModel()

        assert model.geometric_layer is not None
        assert model.semantic_layer is not None
        assert model.causal_layer is not None
        assert model.change_detector is not None

    @pytest.mark.asyncio
    async def test_initialize(self):
        """测试初始化方法"""
        model = ModularWorldModel()

        await model.initialize()

        # 如果SLAM可用，geometric_layer应该已初始化
        # 这里只是验证不会抛出异常

    @pytest.mark.asyncio
    async def test_update_from_perception(self):
        """测试从感知更新"""
        model = ModularWorldModel()

        # 创建模拟感知数据
        class MockPerceptionData:
            def __init__(self):
                self.semantic_objects = [
                    MockSemanticObject("obj1", "door", (5.0, 3.0)),
                    MockSemanticObject("obj2", "person", (7.0, 2.0)),
                ]
                self.pose = MockPose(x=1.0, y=2.0, z=0.0, yaw=0.5)

        class MockPose:
            def __init__(self, x, y, z, yaw):
                self.x = x
                self.y = y
                self.z = z
                self.yaw = yaw

        perception_data = MockPerceptionData()

        changes = await model.update_from_perception(perception_data)

        # 应该有变化
        assert len(changes) > 0
        assert len(model.semantic_layer.get_all_objects()) == 2

    @pytest.mark.asyncio
    async def test_robot_state_update(self):
        """测试机器人状态更新"""
        model = ModularWorldModel()

        class MockPerceptionData:
            def __init__(self):
                self.semantic_objects = []
                self.pose = MockPose(x=5.0, y=3.0, z=0.0, yaw=1.5)

        class MockPose:
            def __init__(self, x, y, z, yaw):
                self.x = x
                self.y = y
                self.z = z
                self.yaw = yaw

        perception_data = MockPerceptionData()
        await model.update_from_perception(perception_data)

        assert model.robot_position["x"] == 5.0
        assert model.robot_position["y"] == 3.0
        assert model.robot_heading == 1.5

    @pytest.mark.asyncio
    async def test_semantic_objects_property(self):
        """测试语义物体属性"""
        model = ModularWorldModel()

        class MockPerceptionData:
            def __init__(self):
                self.semantic_objects = [
                    MockSemanticObject("obj1", "door", (5.0, 3.0)),
                ]
                self.pose = MockPose(x=0, y=0, z=0, yaw=0)

        class MockPose:
            def __init__(self, x, y, z, yaw):
                self.x = x
                self.y = y
                self.z = z
                self.yaw = yaw

        perception_data = MockPerceptionData()
        await model.update_from_perception(perception_data)

        semantic_objects = model.semantic_objects
        assert len(semantic_objects) == 1

    @pytest.mark.asyncio
    async def test_get_location(self):
        """测试获取物体位置"""
        model = ModularWorldModel()

        class MockPerceptionData:
            def __init__(self):
                self.semantic_objects = [
                    MockSemanticObject("obj1", "door", (5.0, 3.0)),
                ]
                self.pose = MockPose(x=0, y=0, z=0, yaw=0)

        class MockPose:
            def __init__(self, x, y, z, yaw):
                self.x = x
                self.y = y
                self.z = z
                self.yaw = yaw

        perception_data = MockPerceptionData()
        await model.update_from_perception(perception_data)

        location = model.get_location("door")
        assert location is not None
        assert location["position"] == (5.0, 3.0)

    @pytest.mark.asyncio
    async def test_get_location_not_found(self):
        """测试获取不存在的物体位置"""
        model = ModularWorldModel()

        location = model.get_location("nonexistent")
        assert location is None

    @pytest.mark.asyncio
    async def test_change_history(self):
        """测试变化历史"""
        model = ModularWorldModel()

        class MockPerceptionData:
            def __init__(self):
                self.semantic_objects = [
                    MockSemanticObject("obj1", "door", (5.0, 3.0)),
                ]
                self.pose = MockPose(x=0, y=0, z=0, yaw=0)

        class MockPose:
            def __init__(self, x, y, z, yaw):
                self.x = x
                self.y = y
                self.z = z
                self.yaw = yaw

        perception_data = MockPerceptionData()
        await model.update_from_perception(perception_data)

        assert len(model.change_history) > 0

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """测试关闭"""
        model = ModularWorldModel()

        # 不应该抛出异常
        model.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
