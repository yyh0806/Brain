"""
感知层统一数据模型

这是感知层的核心数据类型定义，所有模块都应使用这里定义的类型。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math


# ==================== 基础数据类 ====================

@dataclass
class Position2D:
    """二维位置"""
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: 'Position2D') -> float:
        """计算到另一点的距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position2D':
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0))


@dataclass
class Position3D:
    """三维位置"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: 'Position3D') -> float:
        """计算到另一点的距离"""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Position3D':
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0)
        )


@dataclass
class Pose2D:
    """二维位姿"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # yaw角，弧度

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.theta)


@dataclass
class Pose3D:
    """三维位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x, "y": self.y, "z": self.z,
            "roll": self.roll, "pitch": self.pitch, "yaw": self.yaw
        }

    def to_2d(self) -> Pose2D:
        """转换为2D位姿"""
        return Pose2D(x=self.x, y=self.y, theta=self.yaw)

    def to_tuple_2d(self) -> Tuple[float, float, float]:
        """转换为2D位姿元组 (x, y, yaw)"""
        return (self.x, self.y, self.yaw)


@dataclass
class Velocity:
    """速度（线速度和角速度）"""
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0

    def to_dict(self) -> Dict[str, float]:
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
    """3D边界框"""
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


# ==================== 感知结果数据类 ====================

@dataclass
class DetectedObject:
    """检测到的物体（统一版本）"""
    id: str
    label: str
    confidence: float
    position: Position3D
    bounding_box: Optional[BoundingBox] = None
    velocity: Optional[Velocity] = None
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict() if self.velocity else None,
            "description": self.description,
            "attributes": self.attributes,
            "timestamp": self.timestamp.isoformat(),
            "track_id": self.track_id
        }


@dataclass
class SceneDescription:
    """场景描述"""
    summary: str
    objects: List[DetectedObject]
    spatial_relations: List[str]
    navigation_hints: List[str]
    potential_targets: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "objects": [obj.to_dict() for obj in self.objects],
            "spatial_relations": self.spatial_relations,
            "navigation_hints": self.navigation_hints,
            "potential_targets": self.potential_targets,
            "timestamp": self.timestamp.isoformat()
        }


# ==================== 地图数据类 ====================

from brain.perception.core.enums import CellState


@dataclass
class OccupancyGrid:
    """占据栅格地图（统一版本）"""
    width: int
    height: int
    resolution: float  # 米/栅格
    origin_x: float = 0.0
    origin_y: float = 0.0

    # 栅格数据: -1=未知, 0=自由, 100=占据
    data: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = np.full((self.height, self.width), CellState.UNKNOWN, dtype=np.int8)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        x = gx * self.resolution + self.origin_x
        y = gy * self.resolution + self.origin_y
        return x, y

    def is_valid(self, gx: int, gy: int) -> bool:
        """检查栅格坐标是否有效"""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def set_cell(self, gx: int, gy: int, state: int):
        """设置栅格状态"""
        if self.is_valid(gx, gy):
            self.data[gy, gx] = state

    def get_cell(self, gx: int, gy: int) -> int:
        """获取栅格状态"""
        if self.is_valid(gx, gy):
            return int(self.data[gy, gx])
        return CellState.UNKNOWN

    def is_occupied(self, gx: int, gy: int) -> bool:
        """检查栅格是否被占据"""
        return self.get_cell(gx, gy) == CellState.OCCUPIED

    def is_free(self, gx: int, gy: int) -> bool:
        """检查栅格是否自由"""
        return self.get_cell(gx, gy) == CellState.FREE

    def is_unknown(self, gx: int, gy: int) -> bool:
        """检查栅格是否未知"""
        return self.get_cell(gx, gy) == CellState.UNKNOWN
