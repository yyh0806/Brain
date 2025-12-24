"""
统一的数据模型

标准化感知层使用的数据结构，包括：
- Pose2D, Pose3D: 位姿数据
- Position3D: 3D位置
- Velocity: 速度
- BoundingBox: 边界框
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Pose2D:
    """2D位姿"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # yaw角，弧度
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {"x": self.x, "y": self.y, "theta": self.theta}
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为元组 (x, y, theta)"""
        return (self.x, self.y, self.theta)


@dataclass
class Pose3D:
    """3D位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw
        }
    
    def to_2d(self) -> 'Pose2D':
        """转换为2D位姿"""
        return Pose2D(x=self.x, y=self.y, theta=self.yaw)
    
    def to_tuple_2d(self) -> Tuple[float, float, float]:
        """转换为2D位姿元组 (x, y, yaw)"""
        return (self.x, self.y, self.yaw)
    
    def to_tuple_3d(self) -> Tuple[float, float, float, float, float, float]:
        """转换为3D位姿元组 (x, y, z, roll, pitch, yaw)"""
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


@dataclass
class Position3D:
    """3D位置"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """转换为元组"""
        return (self.x, self.y, self.z)
    
    def distance_to(self, other: 'Position3D') -> float:
        """计算到另一点的距离"""
        import math
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position3D':
        """从字典创建"""
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0)
        )


@dataclass
class Velocity:
    """速度"""
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "linear_x": self.linear_x,
            "linear_y": self.linear_y,
            "linear_z": self.linear_z,
            "angular_x": self.angular_x,
            "angular_y": self.angular_y,
            "angular_z": self.angular_z
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Velocity':
        """从字典创建"""
        return cls(
            linear_x=data.get("linear_x", 0.0),
            linear_y=data.get("linear_y", 0.0),
            linear_z=data.get("linear_z", 0.0),
            angular_x=data.get("angular_x", 0.0),
            angular_y=data.get("angular_y", 0.0),
            angular_z=data.get("angular_z", 0.0)
        )


@dataclass
class BoundingBox:
    """边界框（3D）"""
    min_point: Position3D
    max_point: Position3D
    
    @property
    def center(self) -> Position3D:
        """计算中心点"""
        return Position3D(
            x=(self.min_point.x + self.max_point.x) / 2,
            y=(self.min_point.y + self.max_point.y) / 2,
            z=(self.min_point.z + self.max_point.z) / 2
        )
    
    @property
    def size(self) -> Tuple[float, float, float]:
        """计算尺寸 (width, height, depth)"""
        return (
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )
    
    def contains(self, point: Position3D) -> bool:
        """检查点是否在边界框内"""
        return (
            self.min_point.x <= point.x <= self.max_point.x and
            self.min_point.y <= point.y <= self.max_point.y and
            self.min_point.z <= point.z <= self.max_point.z
        )
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """检查是否与另一个边界框相交"""
        return (
            self.min_point.x <= other.max_point.x and
            self.max_point.x >= other.min_point.x and
            self.min_point.y <= other.max_point.y and
            self.max_point.y >= other.min_point.y and
            self.min_point.z <= other.max_point.z and
            self.max_point.z >= other.min_point.z
        )





