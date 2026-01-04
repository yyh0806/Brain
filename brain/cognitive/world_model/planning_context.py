# -*- coding: utf-8 -*-
"""
规划上下文类型定义 - Planning Context Types

从 world_model.py 拆分出来的规划上下文类型定义。
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class PlanningContext:
    """规划上下文 - 提供给规划器的环境信息
    
    这是认知层向规划层提供认知状态的核心接口。
    规划层消费此上下文，自主决定行动。
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
        
        return "\n".join(lines)










