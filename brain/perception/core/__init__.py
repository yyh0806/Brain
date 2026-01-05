"""
感知层核心模块

提供统一的数据类型、枚举和异常定义。
"""

from brain.perception.core.types import (
    Position2D,
    Position3D,
    Pose2D,
    Pose3D,
    Velocity,
    BoundingBox,
    DetectedObject,
    SceneDescription,
    OccupancyGrid
)

from brain.perception.core.enums import (
    CellState,
    ObjectType,
    TerrainType,
    SensorType,
    DetectionSource,
    DetectionMode,
    PerceptionEventType,
    SensorQuality
)

from brain.perception.core.exceptions import (
    PerceptionError,
    SensorError,
    SensorNotFoundError,
    SensorDisconnectedError,
    SensorDataError,
    DetectionError,
    MapError,
    FusionError,
    VLMError
)


__all__ = [
    # Types
    "Position2D",
    "Position3D",
    "Pose2D",
    "Pose3D",
    "Velocity",
    "BoundingBox",
    "DetectedObject",
    "SceneDescription",
    "OccupancyGrid",
    # Enums
    "CellState",
    "ObjectType",
    "TerrainType",
    "SensorType",
    "DetectionSource",
    "DetectionMode",
    "PerceptionEventType",
    "SensorQuality",
    # Exceptions
    "PerceptionError",
    "SensorError",
    "SensorNotFoundError",
    "SensorDisconnectedError",
    "SensorDataError",
    "DetectionError",
    "MapError",
    "FusionError",
    "VLMError",
]
