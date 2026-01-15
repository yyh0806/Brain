# -*- coding: utf-8 -*-
"""
认知层完整流程集成测试
"""

import pytest
import asyncio
import time
import numpy as np
from brain.cognitive.world_model.modular_world_model import ModularWorldModel
from brain.cognitive.reasoning.async_cot_engine import AsyncCoTEngine
from brain.cognitive.world_model.risk_calculator import RiskAreaCalculator


class MockSemanticObject:
    """模拟语义物体"""
    def __init__(self, obj_id, label, position, confidence=0.8):
        self.id = obj_id
        self.label = label
        self.world_position = position
        self.confidence = confidence


class MockPose:
    """模拟位姿"""
    def __init__(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw


class MockPerceptionData:
    """模拟感知数据"""
    def __init__(self, semantic_objects=None, pose=None):
        self.semantic_objects = semantic_objects or []
        self.pose = pose or MockPose(0, 0, 0, 0)


@pytest.mark.asyncio
class TestCognitiveFullPipeline:
    """测试认知层完整流程"""

    async def test_world_model_update_cycle(self):
        """测试世界模型更新周期"""
        model = ModularWorldModel()
        await model.initialize()

        # 初始状态
        assert len(model.semantic_layer.get_all_objects()) == 0

        # 第一次更新
        perception_data = MockPerceptionData(
            semantic_objects=[
                MockSemanticObject("obj1", "door", (5.0, 3.0)),
                MockSemanticObject("obj2", "person", (7.0, 2.0)),
            ],
            pose=MockPose(1.0, 2.0, 0.0, 0.5)
        )

        changes = await model.update_from_perception(perception_data)

        assert len(changes) == 2  # 两个新物体
        assert len(model.semantic_layer.get_all_objects()) == 2
        assert model.robot_position["x"] == 1.0

        # 第二次更新（部分物体变化）
        perception_data2 = MockPerceptionData(
            semantic_objects=[
                MockSemanticObject("obj1", "door", (5.1, 3.1)),  # 位置微调
                MockSemanticObject("obj3", "chair", (8.0, 4.0)),  # 新物体
            ],
            pose=MockPose(1.5, 2.5, 0.0, 0.6)
        )

        changes2 = await model.update_from_perception(perception_data2)

        # 应该有新物体和变化
        assert len(changes2) >= 1
        assert len(model.semantic_layer.get_all_objects()) >= 2

    async def test_risk_assessment_pipeline(self):
        """测试风险评估流程"""
        model = ModularWorldModel()
        risk_calculator = RiskAreaCalculator()

        # 更新感知数据
        perception_data = MockPerceptionData(
            semantic_objects=[
                MockSemanticObject("obj1", "person", (10.0, 10.0)),
                MockSemanticObject("obj2", "car", (15.0, 15.0)),
            ],
            pose=MockPose(5.0, 5.0, 0.0, 0.0)
        )

        await model.update_from_perception(perception_data)

        # 计算风险地图
        geometric_map = model.geometric_map
        if geometric_map is None:
            geometric_map = np.zeros((50, 50), dtype=np.int8)

        risk_map = risk_calculator.compute_risk_map(
            geometric_map,
            model.semantic_objects,
            (model.robot_position["x"], model.robot_position["y"])
        )

        assert risk_map is not None
        assert risk_map.shape == geometric_map.shape

    async def test_reasoning_pipeline(self):
        """测试推理流程"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            model = ModularWorldModel()
            await model.initialize()

            # 更新感知数据
            perception_data = MockPerceptionData(
                semantic_objects=[
                    MockSemanticObject("obj1", "door", (5.0, 3.0)),
                ],
                pose=MockPose(0, 0, 0, 0)
            )

            await model.update_from_perception(perception_data)

            # 使用推理引擎查询
            result = await engine.reason(
                query="门在哪里？",
                context=model.semantic_objects,
                mode="location"
            )

            assert result is not None
            assert len(result.chain) > 0
            assert result.confidence > 0

        finally:
            engine.stop()

    async def test_change_detection_integration(self):
        """测试变化检测集成"""
        model = ModularWorldModel()

        # 第一次更新
        perception_data1 = MockPerceptionData(
            semantic_objects=[
                MockSemanticObject("obj1", "door", (5.0, 3.0)),
            ],
            pose=MockPose(0, 0, 0, 0)
        )

        changes1 = await model.update_from_perception(perception_data1)
        assert len(changes1) > 0

        # 第二次更新（相同数据）
        changes2 = await model.update_from_perception(perception_data1)

        # 变化检测器应该能检测到是否有变化
        assert isinstance(changes2, list)

    async def test_exploration_frontier_integration(self):
        """测试探索边界集成"""
        from brain.cognitive.world_model.risk_calculator import ExplorationFrontierDetector

        model = ModularWorldModel()
        frontier_detector = ExplorationFrontierDetector()

        # 创建包含未知区域的地图
        geometric_map = np.zeros((50, 50), dtype=np.int8)
        geometric_map[20:30, 20:30] = -1  # 未知区域

        # 检测探索边界
        frontiers = frontier_detector.detect_frontiers(
            geometric_map,
            set(),
            (25.0, 25.0)
        )

        assert len(frontiers) > 0

        # 更新模型的探索边界
        model.exploration_frontiers = frontiers

        assert len(model.exploration_frontiers) == len(frontiers)


@pytest.mark.asyncio
class TestPerformanceIntegration:
    """测试性能集成"""

    async def test_world_model_update_performance(self):
        """测试世界模型更新性能"""
        model = ModularWorldModel()
        await model.initialize()

        # 创建大量感知数据
        semantic_objects = [
            MockSemanticObject(f"obj{i}", f"object{i}", (float(i), float(i)))
            for i in range(100)
        ]

        perception_data = MockPerceptionData(
            semantic_objects=semantic_objects,
            pose=MockPose(0, 0, 0, 0)
        )

        # 测量更新时间
        start = time.time()
        await model.update_from_perception(perception_data)
        duration = (time.time() - start) * 1000  # 毫秒

        # 更新时间应该合理（<1秒）
        assert duration < 1000

    async def test_reasoning_cache_performance(self):
        """测试推理缓存性能"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            # 第一次推理
            start = time.time()
            result1 = await engine.reason("测试查询", {}, "default")
            duration1 = (time.time() - start) * 1000

            # 第二次推理（缓存命中）
            start = time.time()
            result2 = await engine.reason("测试查询", {}, "default")
            duration2 = (time.time() - start) * 1000

            # 缓存命中应该更快
            assert result2.from_cache is True
            assert duration2 < duration1

        finally:
            engine.stop()

    async def test_memory_stability(self):
        """测试内存稳定性"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        model = ModularWorldModel()

        # 执行多次更新
        for i in range(50):
            perception_data = MockPerceptionData(
                semantic_objects=[
                    MockSemanticObject(f"obj{i}", "object", (float(i), float(i)))
                ],
                pose=MockPose(float(i), float(i), 0, 0)
            )
            await model.update_from_perception(perception_data)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # 内存增长应该合理（<100MB）
        assert memory_growth < 100


@pytest.mark.asyncio
class TestErrorHandling:
    """测试错误处理"""

    async def test_none_perception_data(self):
        """测试None感知数据"""
        model = ModularWorldModel()

        # 应该不抛出异常
        changes = await model.update_from_perception(None)
        assert isinstance(changes, list)

    async def test_empty_perception_data(self):
        """测试空感知数据"""
        model = ModularWorldModel()

        perception_data = MockPerceptionData(
            semantic_objects=[],
            pose=MockPose(0, 0, 0, 0)
        )

        changes = await model.update_from_perception(perception_data)
        assert isinstance(changes, list)

    async def test_invalid_object_data(self):
        """测试无效物体数据"""
        model = ModularWorldModel()

        # 缺少必要属性的物体
        class InvalidObject:
            pass

        perception_data = MockPerceptionData(
            semantic_objects=[InvalidObject()],
            pose=MockPose(0, 0, 0, 0)
        )

        # 应该不抛出异常
        changes = await model.update_from_perception(perception_data)
        assert isinstance(changes, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
