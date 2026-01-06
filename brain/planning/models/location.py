"""
Location Data Models

位置相关数据模型
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Location:
    """
    位置信息

    表示环境中的一个位置（房间、地点等）

    Attributes:
        name: 位置名称
        position: 位置坐标 {x, y, z}
        type: 位置类型 (room, door, object)
    """
    name: str
    position: Dict[str, float]  # {x: float, y: float, z: float}
    type: str = "room"  # room, door, object

    def distance_to(self, other: 'Location') -> float:
        """
        计算到另一个位置的直线距离

        Args:
            other: 另一个位置

        Returns:
            距离（米）
        """
        x1, y1, z1 = self.position.get('x', 0), self.position.get('y', 0), self.position.get('z', 0)
        x2, y2, z2 = other.position.get('x', 0), other.position.get('y', 0), other.position.get('z', 0)
        return ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5


@dataclass
class Door:
    """
    门信息

    表示一个门及其状态

    Attributes:
        name: 门名称
        position: 门位置坐标 {x, y, z}
        state: 门状态 ('open' 或 'closed')
    """
    name: str
    position: Dict[str, float]
    state: str = "closed"  # open, closed

    def is_open(self) -> bool:
        """门是否打开"""
        return self.state == "open"

    def is_closed(self) -> bool:
        """门是否关闭"""
        return self.state == "closed"


@dataclass
class ObjectInfo:
    """
    物体信息

    表示环境中的一个物体

    Attributes:
        name: 物体名称
        location: 物体所在位置名称
        position: 物体位置坐标 {x, y, z}
        visible: 是否可见
    """
    name: str
    location: str
    position: Dict[str, float]
    visible: bool = True
