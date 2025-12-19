"""
感知层工具模块

提供统一的数学工具和坐标变换函数
"""

from brain.perception.utils.coordinates import (
    quaternion_to_euler,
    euler_to_quaternion,
    transform_local_to_world,
    transform_world_to_local,
    normalize_angle,
    normalize_angles
)

from brain.perception.utils.math_utils import (
    angle_to_direction,
    compute_laser_angles,
    distance_2d,
    distance_3d
)

__all__ = [
    # 坐标变换
    "quaternion_to_euler",
    "euler_to_quaternion",
    "transform_local_to_world",
    "transform_world_to_local",
    "normalize_angle",
    "normalize_angles",
    # 数学工具
    "angle_to_direction",
    "compute_laser_angles",
    "distance_2d",
    "distance_3d",
]
