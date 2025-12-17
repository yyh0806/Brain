"""
局部路径规划器 - Local Planner

负责:
- 短距离路径规划
- 避障
- 速度平滑
- 轨迹跟踪
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from loguru import logger


class PlannerState(Enum):
    """规划器状态"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    OBSTACLE_DETECTED = "obstacle_detected"
    GOAL_REACHED = "goal_reached"


@dataclass
class Waypoint:
    """路径点"""
    x: float
    y: float
    yaw: Optional[float] = None
    velocity: float = 0.5
    
    def distance_to(self, other: 'Waypoint') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class VelocityCommand:
    """速度命令"""
    linear: float
    angular: float
    
    def scale(self, factor: float) -> 'VelocityCommand':
        return VelocityCommand(
            linear=self.linear * factor,
            angular=self.angular * factor
        )


class LocalPlanner:
    """
    局部路径规划器
    
    使用简化的DWA（Dynamic Window Approach）方法
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 机器人参数
        self.max_linear_vel = self.config.get("max_linear_vel", 1.0)
        self.max_angular_vel = self.config.get("max_angular_vel", 1.0)
        self.max_linear_acc = self.config.get("max_linear_acc", 0.5)
        self.max_angular_acc = self.config.get("max_angular_acc", 1.0)
        
        # 规划参数
        self.goal_tolerance = self.config.get("goal_tolerance", 0.3)
        self.lookahead_distance = self.config.get("lookahead_distance", 1.0)
        self.obstacle_margin = self.config.get("obstacle_margin", 0.5)
        
        # 控制参数
        self.control_frequency = self.config.get("control_frequency", 10.0)
        
        # 状态
        self.state = PlannerState.IDLE
        self.current_path: List[Waypoint] = []
        self.current_waypoint_idx = 0
        
        logger.info("LocalPlanner 初始化完成")
    
    def set_path(self, waypoints: List[Tuple[float, float]]):
        """设置路径"""
        self.current_path = [Waypoint(x=p[0], y=p[1]) for p in waypoints]
        self.current_waypoint_idx = 0
        self.state = PlannerState.PLANNING
        logger.info(f"设置路径: {len(self.current_path)} 个路径点")
    
    def compute_velocity(
        self,
        current_pose: Tuple[float, float, float],
        laser_ranges: Optional[List[float]] = None
    ) -> VelocityCommand:
        """
        计算速度命令
        
        Args:
            current_pose: 当前位姿 (x, y, yaw)
            laser_ranges: 激光雷达距离数据
            
        Returns:
            VelocityCommand
        """
        if not self.current_path or self.current_waypoint_idx >= len(self.current_path):
            self.state = PlannerState.GOAL_REACHED
            return VelocityCommand(linear=0, angular=0)
        
        x, y, yaw = current_pose
        target = self.current_path[self.current_waypoint_idx]
        
        # 计算到目标的距离和方向
        dx = target.x - x
        dy = target.y - y
        distance = math.sqrt(dx**2 + dy**2)
        
        # 检查是否到达当前路径点
        if distance < self.goal_tolerance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.current_path):
                self.state = PlannerState.GOAL_REACHED
                return VelocityCommand(linear=0, angular=0)
            target = self.current_path[self.current_waypoint_idx]
            dx = target.x - x
            dy = target.y - y
            distance = math.sqrt(dx**2 + dy**2)
        
        # 计算目标方向
        target_yaw = math.atan2(dy, dx)
        yaw_error = self._normalize_angle(target_yaw - yaw)
        
        # 检查障碍物
        obstacle_factor = 1.0
        if laser_ranges:
            min_range = self._get_front_range(laser_ranges)
            if min_range < self.obstacle_margin:
                self.state = PlannerState.OBSTACLE_DETECTED
                obstacle_factor = 0.0
            elif min_range < self.obstacle_margin * 2:
                obstacle_factor = (min_range - self.obstacle_margin) / self.obstacle_margin
        
        # 计算线速度
        linear = self.max_linear_vel * obstacle_factor
        if abs(yaw_error) > 0.5:
            linear *= 0.3  # 转弯时减速
        linear = min(linear, distance)  # 接近目标时减速
        
        # 计算角速度（P控制）
        angular = 2.0 * yaw_error
        angular = max(-self.max_angular_vel, min(self.max_angular_vel, angular))
        
        self.state = PlannerState.EXECUTING
        
        return VelocityCommand(linear=linear, angular=angular)
    
    def _get_front_range(self, ranges: List[float]) -> float:
        """获取前方最小距离"""
        n = len(ranges)
        center = n // 2
        window = n // 6  # 前方60度
        front_ranges = ranges[center - window:center + window]
        valid = [r for r in front_ranges if 0.1 < r < 30.0]
        return min(valid) if valid else float('inf')
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def is_goal_reached(self) -> bool:
        """检查是否到达目标"""
        return self.state == PlannerState.GOAL_REACHED
    
    def is_obstacle_detected(self) -> bool:
        """检查是否检测到障碍物"""
        return self.state == PlannerState.OBSTACLE_DETECTED
    
    def get_remaining_distance(
        self,
        current_pose: Tuple[float, float, float]
    ) -> float:
        """获取剩余距离"""
        if not self.current_path:
            return 0.0
        
        x, y, _ = current_pose
        total = 0.0
        
        # 到当前路径点的距离
        if self.current_waypoint_idx < len(self.current_path):
            target = self.current_path[self.current_waypoint_idx]
            total += math.sqrt((target.x - x)**2 + (target.y - y)**2)
        
        # 剩余路径点之间的距离
        for i in range(self.current_waypoint_idx, len(self.current_path) - 1):
            total += self.current_path[i].distance_to(self.current_path[i + 1])
        
        return total
    
    def reset(self):
        """重置规划器"""
        self.current_path = []
        self.current_waypoint_idx = 0
        self.state = PlannerState.IDLE


class PurePursuitController:
    """
    Pure Pursuit路径跟踪控制器
    """
    
    def __init__(
        self,
        lookahead_distance: float = 1.0,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 1.0
    ):
        self.lookahead_distance = lookahead_distance
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
    
    def compute_control(
        self,
        current_pose: Tuple[float, float, float],
        path: List[Tuple[float, float]]
    ) -> VelocityCommand:
        """
        计算Pure Pursuit控制命令
        
        Args:
            current_pose: (x, y, yaw)
            path: 路径点列表
            
        Returns:
            VelocityCommand
        """
        if not path:
            return VelocityCommand(linear=0, angular=0)
        
        x, y, yaw = current_pose
        
        # 找到lookahead点
        lookahead_point = self._find_lookahead_point(x, y, path)
        
        if lookahead_point is None:
            return VelocityCommand(linear=0, angular=0)
        
        # 计算曲率
        dx = lookahead_point[0] - x
        dy = lookahead_point[1] - y
        
        # 转换到机器人坐标系
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # 计算曲率
        L = math.sqrt(dx**2 + dy**2)
        if L < 0.01:
            return VelocityCommand(linear=0, angular=0)
        
        curvature = 2 * local_y / (L * L)
        
        # 计算速度
        linear = self.max_linear_vel
        angular = curvature * linear
        angular = max(-self.max_angular_vel, min(self.max_angular_vel, angular))
        
        return VelocityCommand(linear=linear, angular=angular)
    
    def _find_lookahead_point(
        self,
        x: float,
        y: float,
        path: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """找到lookahead点"""
        # 简化实现：找到距离最接近lookahead_distance的路径点
        for point in path:
            dist = math.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist >= self.lookahead_distance:
                return point
        
        # 如果没有足够远的点，返回最后一个
        return path[-1] if path else None

