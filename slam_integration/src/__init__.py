"""
SLAM集成模块 - Brain项目

提供统一的SLAM接口，集成FAST-LIVO、LIO-SAM等SLAM系统
"""

from .slam_manager import (
    SLAMManager,
    SLAMConfig,
    MapMetadata,
    CoordinateTransformer,
    MockSLAMManager
)

__all__ = [
    'SLAMManager',
    'SLAMConfig',
    'MapMetadata',
    'CoordinateTransformer',
    'MockSLAMManager'
]

__version__ = '1.0.0'
