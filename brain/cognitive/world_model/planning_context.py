# -*- coding: utf-8 -*-
"""
规划上下文类型定义 - Planning Context Types

从 world_model.py 拆分出来的规划上下文类型定义。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import math


@dataclass
class PlanningContext:
    """规划上下文 - 提供给规划器的环境信息

    这是认知层向规划层提供认知状态的核心接口。
    规划层消费此上下文，自主决定行动。

    支持三模态融合：几何地图、语义地图、因果地图
    """
    current_position: Dict[str, float]
    current_heading: float
    obstacles: List[Dict[str, Any]]
    targets: List[Dict[str, Any]]
    points_of_interest: List[Dict[str, Any]]
    weather: Dict[str, Any]
    battery_level: float
    signal_strength: float
    available_paths: List[Dict[str, Any]]
    constraints: List[str]
    recent_changes: List[Dict[str, Any]]
    risk_areas: List[Dict[str, Any]]

    # NEW: 语义物体（语义地图模态）
    semantic_objects: List[Dict[str, Any]] = field(default_factory=list)

    # NEW: 因果图数据（因果地图模态）
    causal_graph: Dict[str, Any] = field(default_factory=dict)

    # NEW: 状态预测
    state_predictions: List[Dict[str, Any]] = field(default_factory=list)

    def get_objects_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别获取物体

        Args:
            category: 类别名称（如"door", "chair"）

        Returns:
            匹配的语义物体列表
        """
        return [obj for obj in self.semantic_objects
                if category.lower() in obj.get("label", "").lower()]

    def get_objects_near_position(
        self,
        x: float,
        y: float,
        radius: float = 5.0
    ) -> List[Dict[str, Any]]:
        """获取位置附近的物体

        Args:
            x: X坐标
            y: Y坐标
            radius: 搜索半径（米）

        Returns:
            附近的语义物体列表
        """
        nearby_objects = []
        for obj in self.semantic_objects:
            pos = obj.get("position", ())
            if len(pos) >= 2:
                distance = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
                if distance <= radius:
                    nearby_objects.append(obj)
        return nearby_objects

    def to_prompt_context(self) -> str:
        """转换为LLM可理解的上下文描述"""
        lines = []
        
        lines.append("## 当前状态")
        lines.append(f"- 位置: ({self.current_position.get('x', 0):.1f}, {self.current_position.get('y', 0):.1f}, {self.current_position.get('z', 0):.1f})")
        lines.append(f"- 航向: {self.current_heading:.1f}°")
        lines.append(f"- 电池: {self.battery_level:.1f}%")
        lines.append(f"- 信号: {self.signal_strength:.1f}%")
        
        if self.obstacles:
            lines.append(f"\n## 障碍物 ({len(self.obstacles)}个)")
            for obs in self.obstacles[:5]:  # 最多显示5个
                lines.append(f"- {obs.get('type', '未知')}: 距离{obs.get('distance', 0):.1f}m, 方向{obs.get('direction', '未知')}")
        
        if self.targets:
            lines.append(f"\n## 目标 ({len(self.targets)}个)")
            for target in self.targets[:5]:
                lines.append(f"- {target.get('type', '未知')}: 距离{target.get('distance', 0):.1f}m")
        
        if self.recent_changes:
            lines.append(f"\n## 最近变化")
            for change in self.recent_changes[:3]:
                lines.append(f"- [{change.get('priority', 'info')}] {change.get('description', '未知变化')}")
        
        if self.constraints:
            lines.append(f"\n## 约束条件")
            for constraint in self.constraints:
                lines.append(f"- {constraint}")
        
        if self.weather.get("condition") != "clear":
            lines.append(f"\n## 天气")
            lines.append(f"- 状况: {self.weather.get('condition', '未知')}")
            lines.append(f"- 风速: {self.weather.get('wind_speed', 0):.1f}m/s")
            lines.append(f"- 能见度: {self.weather.get('visibility', '良好')}")

        # NEW: 语义物体信息
        if self.semantic_objects:
            lines.append(f"\n## 语义物体 ({len(self.semantic_objects)}个)")
            for obj in self.semantic_objects[:5]:
                label = obj.get('label', '未知')
                pos = obj.get('position', ())
                if len(pos) >= 2:
                    lines.append(f"- {label}: 位置({pos[0]:.1f}, {pos[1]:.1f}), 置信度{obj.get('confidence', 0):.2f}")

        # NEW: 因果关系
        if self.causal_graph:
            relations = self.causal_graph.get("relations", [])
            if relations:
                lines.append(f"\n## 因果关系")
                for cause, effect in relations[:3]:
                    lines.append(f"- {cause} → {effect}")

        # NEW: 状态预测
        if self.state_predictions:
            lines.append(f"\n## 预测")
            for pred in self.state_predictions[:3]:
                lines.append(f"- {pred.get('description', '未知预测')}")

        return "\n".join(lines)











