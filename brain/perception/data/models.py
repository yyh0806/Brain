# -*- coding: utf-8 -*-
"""
统一的数据模型 - Unified Data Models
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class Pose2D:
    """2D位姿"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}


@dataclass
class Pose3D:
    """3D位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def to_2d(self):
        return Pose2D(x=self.x, y=self.y, theta=self.yaw)


@dataclass
class Position3D:
    """3D位置"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def distance_to(self, other):
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
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


@dataclass
class BoundingBox:
    """3D边界框"""
    min_point: Position3D
    max_point: Position3D


@dataclass
class Detection:
    """检测结果"""
    object_type: Optional[str] = None
    confidence: float = 0.0
    bounding_box_2d: Optional[Tuple] = None
    position_3d: Optional[Position3D] = None
