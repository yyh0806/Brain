# -*- coding: utf-8 -*-
"""
风险计算器测试
"""

import pytest
import numpy as np
from brain.cognitive.world_model.risk_calculator import (
    RiskArea,
    RiskAreaCalculator,
    ExplorationFrontierDetector
)


class MockSemanticObject:
    """模拟语义物体"""
    def __init__(self, label, position):
        self.label = label
        self.world_position = position


class TestRiskArea:
    """测试风险区域"""

    def test_creation(self):
        """测试创建风险区域"""
        area = RiskArea(
            grid_position=(5, 10),
            risk_level=0.8,
            risk_type="dynamic_obstacle",
            description="高风险区域"
        )

        assert area.grid_position == (5, 10)
        assert area.risk_level == 0.8
        assert area.risk_type == "dynamic_obstacle"
        assert area.description == "高风险区域"


class TestRiskAreaCalculator:
    """测试风险区域计算器"""

    def test_initialization(self):
        """测试初始化"""
        calculator = RiskAreaCalculator()

        assert calculator.dynamic_obstacle_radius == 2.0
        assert calculator.narrow_passage_threshold == 1.5
        assert calculator.unknown_area_risk == 0.5

    def test_initialization_with_config(self):
        """测试使用配置初始化"""
        config = {
            "dynamic_obstacle_radius": 3.0,
            "narrow_passage_threshold": 2.0,
            "unknown_area_risk": 0.7
        }

        calculator = RiskAreaCalculator(config)

        assert calculator.dynamic_obstacle_radius == 3.0
        assert calculator.narrow_passage_threshold == 2.0
        assert calculator.unknown_area_risk == 0.7

    def test_compute_risk_map_basic(self):
        """测试基本风险地图计算"""
        calculator = RiskAreaCalculator()

        # 创建测试地图
        test_map = np.zeros((50, 50), dtype=np.int8)

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 风险地图应该与输入地图大小相同
        assert risk_map.shape == test_map.shape
        assert risk_map.dtype == np.float32

    def test_compute_risk_map_with_occupied(self):
        """测试包含占据区域的风险地图"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[20:30, 20:30] = 100  # 占据区域

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 应该有风险区域
        assert np.max(risk_map) > 0

    def test_compute_risk_map_with_unknown(self):
        """测试包含未知区域的风险地图"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[5:15, 5:15] = -1  # 未知区域

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 未知区域应该有风险
        assert np.max(risk_map) >= calculator.unknown_area_risk

    def test_compute_risk_map_with_dynamic_obstacles(self):
        """测试包含动态障碍物的风险地图"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)

        semantic_objects = {
            "obj1": MockSemanticObject("人", (10.0, 10.0)),
            "obj2": MockSemanticObject("车", (15.0, 15.0)),
        }

        risk_map = calculator.compute_risk_map(
            test_map,
            semantic_objects,
            (25.0, 25.0)
        )

        # 应该检测到动态障碍物风险
        assert np.max(risk_map) > 0

    def test_compute_risk_map_none_input(self):
        """测试None输入"""
        calculator = RiskAreaCalculator()

        risk_map = calculator.compute_risk_map(None, {}, (0, 0))

        assert risk_map is None

    def test_dynamic_obstacle_risk(self):
        """测试动态障碍物风险计算"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)

        # 中文人
        semantic_objects = {
            "obj1": MockSemanticObject("人", (10.0, 10.0)),
        }

        risk_map = calculator.compute_risk_map(
            test_map,
            semantic_objects,
            (25.0, 25.0)
        )

        # 应该有风险
        assert np.max(risk_map) >= 0.3

    def test_dynamic_obstacle_risk_english(self):
        """测试英文动态障碍物"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)

        # 英文person和car
        semantic_objects = {
            "obj1": MockSemanticObject("person", (10.0, 10.0)),
            "obj2": MockSemanticObject("car", (15.0, 15.0)),
        }

        risk_map = calculator.compute_risk_map(
            test_map,
            semantic_objects,
            (25.0, 25.0)
        )

        # 应该检测到风险
        assert np.max(risk_map) >= 0.3

    def test_narrow_passage_risk(self):
        """测试狭窄通道风险"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)
        # 创建狭窄通道
        test_map[20:25, 10:20] = 100  # 左墙
        test_map[20:25, 30:40] = 100  # 右墙

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 应该检测到狭窄通道风险
        assert np.max(risk_map) >= 0.4

    def test_unknown_area_risk(self):
        """测试未知区域风险"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[5:15, 5:15] = -1  # 未知区域

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 未知区域应该有风险
        unknown_mask = (test_map == -1)
        assert np.all(risk_map[unknown_mask] == calculator.unknown_area_risk)

    def test_get_high_risk_areas(self):
        """测试获取高风险区域"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[5:15, 5:15] = -1  # 未知区域

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 使用低阈值获取高风险区域
        high_risk_areas = calculator.get_high_risk_areas(risk_map, threshold=0.1)

        assert len(high_risk_areas) > 0

        # 检查第一个高风险区域
        area = high_risk_areas[0]
        assert isinstance(area, RiskArea)
        assert area.risk_level >= 0.1
        assert area.risk_type == "high_risk"

    def test_get_high_risk_areas_empty(self):
        """测试获取高风险区域（无风险）"""
        calculator = RiskAreaCalculator()

        test_map = np.zeros((50, 50), dtype=np.int8)

        risk_map = calculator.compute_risk_map(
            test_map,
            {},
            (25.0, 25.0)
        )

        # 使用高阈值，应该没有高风险区域
        high_risk_areas = calculator.get_high_risk_areas(risk_map, threshold=1.0)

        assert len(high_risk_areas) == 0


class TestExplorationFrontierDetector:
    """测试探索边界检测器"""

    def test_initialization(self):
        """测试初始化"""
        detector = ExplorationFrontierDetector()

        assert detector.config is not None

    def test_detect_frontiers_basic(self):
        """测试基本探索边界检测"""
        detector = ExplorationFrontierDetector()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[5:15, 5:15] = -1  # 未知区域

        frontiers = detector.detect_frontiers(
            test_map,
            set(),
            (25.0, 25.0)
        )

        # 应该检测到探索边界
        assert len(frontiers) > 0

    def test_detect_frontiers_none_input(self):
        """测试None输入"""
        detector = ExplorationFrontierDetector()

        frontiers = detector.detect_frontiers(
            None,
            set(),
            (25.0, 25.0)
        )

        assert len(frontiers) == 0

    def test_detect_frontiers_max_limit(self):
        """测试探索边界数量限制"""
        config = {"max_frontiers": 10}
        detector = ExplorationFrontierDetector(config)

        test_map = np.zeros((100, 100), dtype=np.int8)
        test_map[10:90, 10:90] = -1  # 大片未知区域

        frontiers = detector.detect_frontiers(
            test_map,
            set(),
            (50.0, 50.0)
        )

        # 应该限制在最大数量
        assert len(frontiers) <= 10

    def test_frontier_priority(self):
        """测试探索边界优先级"""
        detector = ExplorationFrontierDetector()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[20:30, 20:30] = -1  # 未知区域（靠近中心）

        frontiers = detector.detect_frontiers(
            test_map,
            set(),
            (25.0, 25.0)  # 机器人位置靠近未知区域
        )

        if frontiers:
            # 检查优先级在0-1之间
            for frontier in frontiers:
                priority = frontier["priority"]
                assert 0 <= priority <= 1

    def test_frontier_structure(self):
        """测试探索边界结构"""
        detector = ExplorationFrontierDetector()

        test_map = np.zeros((50, 50), dtype=np.int8)
        test_map[20:30, 20:30] = -1

        frontiers = detector.detect_frontiers(
            test_map,
            set(),
            (25.0, 25.0)
        )

        if frontiers:
            # 检查边界结构
            frontier = frontiers[0]
            assert "position" in frontier
            assert "priority" in frontier
            assert isinstance(frontier["position"], tuple)
            assert len(frontier["position"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
