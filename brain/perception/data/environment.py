"""
环境工具类 - Environment Utilities

注意：EnvironmentModel 类已删除，功能已合并到 brain.cognitive.world_model.WorldModel

本文件保留工具类供其他模块使用：
- ObjectType, TerrainType: 枚举类型
- Position3D, BoundingBox, DetectedObject: 数据类
- MapCell, OccupancyGrid: 地图相关工具类
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from loguru import logger


class ObjectType(Enum):
    """物体类型"""
    UNKNOWN = "unknown"
    PERSON = "person"
    VEHICLE = "vehicle"
    BUILDING = "building"
    TREE = "tree"
    OBSTACLE = "obstacle"
    LANDING_ZONE = "landing_zone"
    TARGET = "target"
    WATER = "water"
    ROAD = "road"


class TerrainType(Enum):
    """地形类型"""
    UNKNOWN = "unknown"
    FLAT = "flat"
    SLOPE = "slope"
    ROUGH = "rough"
    WATER = "water"
    URBAN = "urban"
    FOREST = "forest"


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


@dataclass
class BoundingBox:
    """边界框"""
    min_point: Position3D
    max_point: Position3D
    
    @property
    def center(self) -> Position3D:
        return Position3D(
            x=(self.min_point.x + self.max_point.x) / 2,
            y=(self.min_point.y + self.max_point.y) / 2,
            z=(self.min_point.z + self.max_point.z) / 2
        )
    
    @property
    def size(self) -> Tuple[float, float, float]:
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


@dataclass
class DetectedObject:
    """检测到的物体"""
    id: str
    object_type: ObjectType
    position: Position3D
    bounding_box: Optional[BoundingBox] = None
    velocity: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    track_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.object_type.value,
            "position": self.position.to_dict(),
            "confidence": self.confidence,
            "velocity": self.velocity,
            "attributes": self.attributes
        }


@dataclass
class MapCell:
    """地图单元格"""
    x: int
    y: int
    terrain: TerrainType = TerrainType.UNKNOWN
    elevation: float = 0.0
    traversable: bool = True
    cost: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)


class OccupancyGrid:
    """占用栅格地图"""
    
    def __init__(
        self, 
        resolution: float = 1.0,  # 米/格
        width: int = 100,
        height: int = 100
    ):
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # 占用概率网格 (0=空闲, 1=占用, 0.5=未知)
        self.grid: List[List[float]] = [
            [0.5 for _ in range(width)]
            for _ in range(height)
        ]
        
        # 原点偏移
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return (
            max(0, min(gx, self.width - 1)),
            max(0, min(gy, self.height - 1))
        )
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        x = gx * self.resolution + self.origin_x + self.resolution / 2
        y = gy * self.resolution + self.origin_y + self.resolution / 2
        return (x, y)
    
    def update_cell(self, x: float, y: float, occupied: bool):
        """更新单元格"""
        gx, gy = self.world_to_grid(x, y)
        
        # 贝叶斯更新
        prior = self.grid[gy][gx]
        if occupied:
            # 传感器检测到障碍物
            self.grid[gy][gx] = min(0.95, prior * 0.9 + 0.1)
        else:
            # 传感器检测到空闲
            self.grid[gy][gx] = max(0.05, prior * 0.9)
    
    def is_occupied(self, x: float, y: float, threshold: float = 0.6) -> bool:
        """检查位置是否被占用"""
        gx, gy = self.world_to_grid(x, y)
        return self.grid[gy][gx] > threshold
    
    def get_occupancy(self, x: float, y: float) -> float:
        """获取位置的占用概率"""
        gx, gy = self.world_to_grid(x, y)
        return self.grid[gy][gx]


# EnvironmentModel 类已删除
# 功能已合并到 brain.cognitive.world_model.WorldModel
# 保留工具类（Position3D, BoundingBox, OccupancyGrid等）供其他模块使用

