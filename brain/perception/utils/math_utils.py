"""
数学工具函数

提供感知层常用的数学计算函数
"""

import math
from typing import List, Dict, Any, Tuple


def angle_to_direction(angle: float) -> str:
    """
    将角度转换为方向描述（中文）
    
    Args:
        angle: 角度（弧度），范围 [-pi, pi]
        
    Returns:
        str: 方向描述（"前方"、"左前方"等）
    """
    angle_deg = math.degrees(angle)
    
    if -22.5 <= angle_deg < 22.5:
        return "前方"
    elif 22.5 <= angle_deg < 67.5:
        return "左前方"
    elif 67.5 <= angle_deg < 112.5:
        return "左侧"
    elif 112.5 <= angle_deg < 157.5:
        return "左后方"
    elif angle_deg >= 157.5 or angle_deg < -157.5:
        return "后方"
    elif -157.5 <= angle_deg < -112.5:
        return "右后方"
    elif -112.5 <= angle_deg < -67.5:
        return "右侧"
    else:
        return "右前方"


def compute_laser_angles(laser_scan: Dict[str, Any]) -> List[float]:
    """
    计算激光雷达角度数组
    
    Args:
        laser_scan: 激光雷达扫描数据字典，包含：
            - angle_min: 最小角度（弧度）
            - angle_max: 最大角度（弧度）
            - angle_increment: 角度增量（弧度）
            - ranges: 距离数组（可选，用于推断角度数量）
            
    Returns:
        List[float]: 角度数组（弧度）
    """
    angle_min = laser_scan.get("angle_min", -math.pi)
    angle_max = laser_scan.get("angle_max", math.pi)
    angle_increment = laser_scan.get("angle_increment", 0)
    
    if angle_increment > 0:
        # 使用角度增量计算
        angles = []
        angle = angle_min
        while angle <= angle_max:
            angles.append(angle)
            angle += angle_increment
        return angles
    
    # 如果没有角度增量，使用ranges数组推断
    ranges = laser_scan.get("ranges", [])
    n = len(ranges)
    if n > 0:
        return [angle_min + i * (angle_max - angle_min) / n for i in range(n)]
    
    return []


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    计算2D距离
    
    Args:
        x1, y1: 第一个点的坐标
        x2, y2: 第二个点的坐标
        
    Returns:
        float: 距离
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """
    计算3D距离
    
    Args:
        x1, y1, z1: 第一个点的坐标
        x2, y2, z2: 第二个点的坐标
        
    Returns:
        float: 距离
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)







