"""
机器人平台模块

定义不同平台类型的能力和约束
"""

from brain.platforms.robot_capabilities import (
    RobotCapabilities,
    PlatformType,
    SensorCapability,
    MotionCapability
)

__all__ = [
    "RobotCapabilities",
    "PlatformType",
    "SensorCapability",
    "MotionCapability"
]

