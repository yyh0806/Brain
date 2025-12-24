"""
坐标变换工具

提供统一的坐标变换函数，包括：
- 四元数与欧拉角转换
- 局部坐标与世界坐标转换
- 角度归一化
"""

import math
from typing import Tuple, Union
import numpy as np


def quaternion_to_euler(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """
    四元数转欧拉角（ZYX顺序，即yaw-pitch-roll）
    
    Args:
        q: 四元数 (x, y, z, w)
        
    Returns:
        Tuple[float, float, float]: 欧拉角 (roll, pitch, yaw) 单位：弧度
    """
    x, y, z, w = q
    
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # 使用90度，如果超出范围
    else:
        pitch = math.asin(sinp)
    
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    欧拉角转四元数（ZYX顺序）
    
    Args:
        roll: 绕x轴旋转（弧度）
        pitch: 绕y轴旋转（弧度）
        yaw: 绕z轴旋转（弧度）
        
    Returns:
        Tuple[float, float, float, float]: 四元数 (x, y, z, w)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return (x, y, z, w)


def transform_local_to_world(
    local_x: float,
    local_y: float,
    world_x: float,
    world_y: float,
    yaw: float
) -> Tuple[float, float]:
    """
    将局部坐标转换为世界坐标
    
    Args:
        local_x: 局部x坐标
        local_y: 局部y坐标
        world_x: 世界坐标系原点x
        world_y: 世界坐标系原点y
        yaw: 世界坐标系朝向（弧度）
        
    Returns:
        Tuple[float, float]: 世界坐标 (x, y)
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    
    world_x_out = world_x + local_x * cos_yaw - local_y * sin_yaw
    world_y_out = world_y + local_x * sin_yaw + local_y * cos_yaw
    
    return (world_x_out, world_y_out)


def transform_world_to_local(
    world_x: float,
    world_y: float,
    origin_x: float,
    origin_y: float,
    yaw: float
) -> Tuple[float, float]:
    """
    将世界坐标转换为局部坐标
    
    Args:
        world_x: 世界x坐标
        world_y: 世界y坐标
        origin_x: 局部坐标系原点x（世界坐标）
        origin_y: 局部坐标系原点y（世界坐标）
        yaw: 局部坐标系朝向（弧度）
        
    Returns:
        Tuple[float, float]: 局部坐标 (x, y)
    """
    dx = world_x - origin_x
    dy = world_y - origin_y
    
    cos_yaw = math.cos(-yaw)  # 反向旋转
    sin_yaw = math.sin(-yaw)
    
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    
    return (local_x, local_y)


def normalize_angle(angle: float) -> float:
    """
    归一化角度到 [-pi, pi] 范围
    
    Args:
        angle: 输入角度（弧度）
        
    Returns:
        float: 归一化后的角度（弧度）
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_angles(angles: Union[np.ndarray, list]) -> np.ndarray:
    """
    归一化角度数组到 [-pi, pi] 范围
    
    Args:
        angles: 角度数组（弧度）
        
    Returns:
        np.ndarray: 归一化后的角度数组
    """
    if isinstance(angles, list):
        angles = np.array(angles)
    return np.arctan2(np.sin(angles), np.cos(angles))





