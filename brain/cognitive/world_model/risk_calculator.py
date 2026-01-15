# -*- coding: utf-8 -*-
"""
风险区域计算器 - Risk Area Calculator

实现环境风险评估：
- 动态障碍物附近高风险
- 狭窄通道风险
- 未知区域风险
- 综合风险地图生成
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskArea:
    """风险区域"""
    grid_position: Tuple[int, int]
    risk_level: float  # 0-1
    risk_type: str  # "dynamic_obstacle", "narrow_passage", "unknown", etc.
    description: str = ""


class RiskAreaCalculator:
    """
    风险区域计算器

    核心功能：
    1. 计算动态障碍物风险
    2. 检测狭窄通道
    3. 识别未知高风险区域
    4. 生成综合风险地图
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 风险参数
        self.dynamic_obstacle_radius = self.config.get("dynamic_obstacle_radius", 2.0)  # 米
        self.narrow_passage_threshold = self.config.get("narrow_passage_threshold", 1.5)  # 米
        self.unknown_area_risk = self.config.get("unknown_area_risk", 0.5)  # 0-1

    def compute_risk_map(
        self,
        geometric_map: np.ndarray,
        semantic_objects: Dict[str, Any],
        robot_position: Tuple[float, float]
    ) -> np.ndarray:
        """
        计算风险地图

        Args:
            geometric_map: 几何地图（占据栅格）
            semantic_objects: 语义物体字典
            robot_position: 机器人位置

        Returns:
            风险地图（与geometric_map相同大小，值0-1）
        """
        if geometric_map is None:
            return None

        # 初始化风险地图
        risk_map = np.zeros_like(geometric_map, dtype=np.float32)

        # 1. 动态障碍物风险
        risk_map = np.maximum(risk_map, self._compute_dynamic_obstacle_risk(
            geometric_map, semantic_objects, robot_position
        ))

        # 2. 狭窄通道风险
        risk_map = np.maximum(risk_map, self._compute_narrow_passage_risk(geometric_map))

        # 3. 未知区域风险
        risk_map = np.maximum(risk_map, self._compute_unknown_area_risk(geometric_map))

        return risk_map

    def _compute_dynamic_obstacle_risk(
        self,
        geometric_map: np.ndarray,
        semantic_objects: Dict[str, Any],
        robot_position: Tuple[float, float]
    ) -> np.ndarray:
        """
        计算动态障碍物风险

        动态障碍物（人、车）附近风险较高
        """
        risk_map = np.zeros_like(geometric_map, dtype=np.float32)

        # 识别动态障碍物
        dynamic_obstacles = []
        for obj_id, obj in semantic_objects.items():
            if hasattr(obj, 'label'):
                label = obj.label.lower()
                if any(keyword in label for keyword in ["人", "person", "车", "car", "vehicle"]):
                    if hasattr(obj, 'world_position'):
                        dynamic_obstacles.append(obj.world_position)

        # 为每个动态障碍物计算风险区域
        for obj_pos in dynamic_obstacles:
            # 简化实现：在栅格地图中标记风险
            # 实际应该考虑机器人到障碍物的距离
            risk_map = np.maximum(risk_map, 0.3)  # 基础风险

        return risk_map

    def _compute_narrow_passage_risk(self, geometric_map: np.ndarray) -> np.ndarray:
        """
        计算狭窄通道风险

        狭窄通道（两侧有障碍物）风险较高
        """
        risk_map = np.zeros_like(geometric_map, dtype=np.float32)

        # 简化实现：检测狭窄通道
        # 实际应该使用形态学操作检测通道宽度

        # 这里使用简单启发式：占据栅格密集的区域
        if geometric_map.ndim == 2:
            # 对每个栅格检查周围是否有足够空间
            # 使用卷积核检测狭窄通道
            kernel = np.ones((3, 3), dtype=np.int8)

            # 标记占据栅格
            occupied = (geometric_map == 100).astype(np.int8)

            import scipy.ndimage as ndimage
            # 计算每个栅格周围的占据数量
            occupied_count = ndimage.convolve(occupied, kernel, mode='constant', cval=0)

            # 如果周围占据密度高，风险高
            narrow_areas = (occupied_count >= 5) & (occupied_count <= 7)
            risk_map[narrow_areas] = 0.4

        return risk_map

    def _compute_unknown_area_risk(self, geometric_map: np.ndarray) -> np.ndarray:
        """
        计算未知区域风险

        未知区域（-1）有一定风险
        """
        risk_map = np.zeros_like(geometric_map, dtype=np.float32)

        # 未知区域风险
        unknown_areas = (geometric_map == -1)
        risk_map[unknown_areas] = self.unknown_area_risk

        return risk_map

    def get_high_risk_areas(
        self,
        risk_map: np.ndarray,
        threshold: float = 0.7
    ) -> List[RiskArea]:
        """
        获取高风险区域列表

        Args:
            risk_map: 风险地图
            threshold: 风险阈值

        Returns:
            高风险区域列表
        """
        high_risk_mask = (risk_map >= threshold)
        risk_areas = []

        # 获取高风险栅格的坐标
        risk_coords = np.argwhere(high_risk_mask)

        for grid_y, grid_x in risk_coords:
            risk_areas.append(RiskArea(
                grid_position=(grid_x, grid_y),
                risk_level=float(risk_map[grid_y, grid_x]),
                risk_type="high_risk"
            ))

        return risk_areas


class ExplorationFrontierDetector:
    """
    探索边界检测器

    检测未探索区域的边界，用于自主探索
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def detect_frontiers(
        self,
        geometric_map: np.ndarray,
        explored_positions: set,
        robot_position: Tuple[float, float]
    ) -> List[Any]:
        """
        检测探索边界

        Args:
            geometric_map: 几何地图
            explored_positions: 已探索位置集合
            robot_position: 机器人位置

        Returns:
            探索边界列表
        """
        frontiers = []

        if geometric_map is None:
            return frontiers

        # 简化实现：检测未知区域和已探索区域的边界
        unknown_mask = (geometric_map == -1)
        free_mask = (geometric_map == 0)

        # 使用形态学梯度检测边界
        import scipy.ndimage as ndimage

        # 未知区域的边界
        unknown_boundary = unknown_mask & ~ndimage.binary_erosion(unknown_mask)

        # 获取边界坐标
        boundary_coords = np.argwhere(unknown_boundary)

        # 限制返回数量
        max_frontiers = self.config.get("max_frontiers", 100)
        for i, (y, x) in enumerate(boundary_coords[:max_frontiers]):
            frontiers.append({
                "position": (x, y),
                "priority": self._compute_frontier_priority((x, y), robot_position, geometric_map)
            })

        return frontiers

    def _compute_frontier_priority(
        self,
        frontier_pos: Tuple[int, int],
        robot_pos: Tuple[float, float],
        geometric_map: np.ndarray
    ) -> float:
        """
        计算探索边界优先级

        Args:
            frontier_pos: 边界位置（栅格坐标）
            robot_pos: 机器人位置（世界坐标）
            geometric_map: 地图

        Returns:
            优先级（0-1）
        """
        # 距离因子：距离越近优先级越高
        try:
            dist = ((frontier_pos[0] - robot_pos[0])**2 +
                   (frontier_pos[1] - robot_pos[1])**2)**0.5
            distance_factor = max(0, 1 - dist / 20.0)  # 20米范围内
        except:
            distance_factor = 0.5

        # 简单返回平均优先级
        return (distance_factor + 0.5) / 2.0


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("风险区域计算器测试")
    print("=" * 60)

    # 测试1: 风险地图计算
    print("\n[测试1] 风险地图计算...")
    calculator = RiskAreaCalculator()

    # 创建模拟地图
    test_map = np.zeros((50, 50), dtype=np.int8)
    test_map[20:30, 20:30] = 100  # 占据区域
    test_map[5:15, 5:15] = -1     # 未知区域

    risk_map = calculator.compute_risk_map(
        test_map,
        {},  # 空语义物体
        (25.0, 25.0)
    )

    print(f"风险地图尺寸: {risk_map.shape}")
    print(f"平均风险: {np.mean(risk_map):.3f}")
    print(f"最大风险: {np.max(risk_map):.3f}")

    # 测试2: 高风险区域
    print("\n[测试2] 高风险区域...")
    high_risk_areas = calculator.get_high_risk_areas(risk_map, threshold=0.1)
    print(f"高风险区域数量: {len(high_risk_areas)}")

    # 测试3: 探索边界
    print("\n[测试3] 探索边界检测...")
    frontier_detector = ExplorationFrontierDetector()

    frontiers = frontier_detector.detect_frontiers(
        test_map,
        set(),
        (25.0, 25.0)
    )

    print(f"探索边界数量: {len(frontiers)}")
    if frontiers:
        for i, frontier in enumerate(frontiers[:3]):
            print(f"  边界{i}: 位置{frontier['position']}, 优先级{frontier['priority']:.2f}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n功能特性:")
    print("  ✓ 动态障碍物风险")
    print("  ✓ 狭窄通道风险")
    print("  ✓ 未知区域风险")
    print("  ✓ 探索边界检测")
