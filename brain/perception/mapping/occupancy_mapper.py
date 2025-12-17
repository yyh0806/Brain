"""
占据栅格地图生成器 - Occupancy Grid Mapper

负责:
- 从深度图生成占据栅格
- 可选融合激光雷达/点云数据
- 维护局部/全局地图
- 支持地图更新和查询
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from loguru import logger


class CellState(IntEnum):
    """栅格状态"""
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100


@dataclass
class OccupancyGrid:
    """占据栅格地图"""
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


class OccupancyMapper:
    """
    占据栅格地图生成器
    
    支持从深度图、激光雷达、点云生成占据栅格
    """
    
    def __init__(
        self,
        resolution: float = 0.1,  # 米/栅格
        map_size: float = 50.0,   # 地图大小（米）
        camera_fov: float = 1.57,  # 相机视场角（弧度，默认90度）
        camera_range: float = 10.0,  # 相机感知范围（米）
        lidar_range: float = 30.0,   # 激光雷达范围（米）
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.resolution = resolution
        self.map_size = map_size
        self.camera_fov = camera_fov
        self.camera_range = camera_range
        self.lidar_range = lidar_range
        
        # 地图尺寸（栅格数）
        grid_size = int(map_size / resolution)
        self.grid = OccupancyGrid(
            width=grid_size,
            height=grid_size,
            resolution=resolution,
            origin_x=-map_size / 2,
            origin_y=-map_size / 2
        )
        
        # 相机内参（默认值，可从配置读取）
        self.camera_fx = self.config.get("camera_fx", 525.0)
        self.camera_fy = self.config.get("camera_fy", 525.0)
        self.camera_cx = self.config.get("camera_cx", 320.0)
        self.camera_cy = self.config.get("camera_cy", 240.0)
        
        # 更新参数
        self.occupied_prob = self.config.get("occupied_prob", 0.7)
        self.free_prob = self.config.get("free_prob", 0.3)
        self.min_depth = self.config.get("min_depth", 0.1)
        self.max_depth = self.config.get("max_depth", 10.0)
        
        logger.info(f"OccupancyMapper 初始化: 分辨率={resolution}m, 地图大小={map_size}m")
    
    def update_from_depth(
        self,
        depth_image: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None,
        camera_pose: Optional[Tuple[float, float, float]] = None
    ):
        """
        从深度图更新占据栅格
        
        Args:
            depth_image: 深度图 (H, W)
            pose: 机器人位姿 (x, y, yaw)，如果为None则使用地图原点
            camera_pose: 相机相对机器人的位姿 (x, y, yaw)，默认(0, 0, 0)
        """
        if depth_image is None or depth_image.size == 0:
            return
        
        if pose is None:
            pose = (0.0, 0.0, 0.0)
        
        if camera_pose is None:
            camera_pose = (0.0, 0.0, 0.0)
        
        h, w = depth_image.shape
        
        # 机器人位姿
        robot_x, robot_y, robot_yaw = pose
        cam_x, cam_y, cam_yaw = camera_pose
        
        # 相机在世界坐标系中的位姿
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        world_cam_x = robot_x + cam_x * cos_yaw - cam_y * sin_yaw
        world_cam_y = robot_y + cam_x * sin_yaw + cam_y * cos_yaw
        world_cam_yaw = robot_yaw + cam_yaw
        
        # 遍历深度图像素
        for v in range(0, h, 2):  # 降采样以提高速度
            for u in range(0, w, 2):
                depth = depth_image[v, u]
                
                # 过滤无效深度
                if depth < self.min_depth or depth > self.max_depth:
                    continue
                
                # 计算3D点（相机坐标系）
                z = depth
                x = (u - self.camera_cx) * z / self.camera_fx
                y = (v - self.camera_cy) * z / self.camera_fy
                
                # 转换到世界坐标系
                cos_cam = math.cos(world_cam_yaw)
                sin_cam = math.sin(world_cam_yaw)
                world_x = world_cam_x + x * cos_cam - y * sin_cam
                world_y = world_cam_y + x * sin_cam + y * cos_cam
                
                # 更新占据栅格
                gx, gy = self.grid.world_to_grid(world_x, world_y)
                if self.grid.is_valid(gx, gy):
                    self.grid.set_cell(gx, gy, CellState.OCCUPIED)
                
                # 更新从相机到障碍物之间的自由空间
                self._update_free_space(
                    (world_cam_x, world_cam_y),
                    (world_x, world_y)
                )
    
    def update_from_laser(
        self,
        ranges: List[float],
        angles: List[float],
        pose: Optional[Tuple[float, float, float]] = None
    ):
        """
        从激光雷达数据更新占据栅格
        
        Args:
            ranges: 距离数组
            angles: 角度数组（相对于机器人朝向）
            pose: 机器人位姿 (x, y, yaw)
        """
        if not ranges or not angles or len(ranges) != len(angles):
            return
        
        if pose is None:
            pose = (0.0, 0.0, 0.0)
        
        robot_x, robot_y, robot_yaw = pose
        
        for r, a in zip(ranges, angles):
            # 过滤无效距离
            if r < 0.1 or r > self.lidar_range:
                continue
            
            # 计算障碍物位置（机器人坐标系）
            local_x = r * math.cos(a)
            local_y = r * math.sin(a)
            
            # 转换到世界坐标系
            cos_yaw = math.cos(robot_yaw)
            sin_yaw = math.sin(robot_yaw)
            world_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
            world_y = robot_y + local_x * sin_yaw + local_y * cos_yaw
            
            # 更新占据栅格
            gx, gy = self.grid.world_to_grid(world_x, world_y)
            if self.grid.is_valid(gx, gy):
                self.grid.set_cell(gx, gy, CellState.OCCUPIED)
            
            # 更新自由空间
            self._update_free_space(
                (robot_x, robot_y),
                (world_x, world_y)
            )
    
    def update_from_pointcloud(
        self,
        pointcloud: np.ndarray,
        pose: Optional[Tuple[float, float, float]] = None
    ):
        """
        从点云更新占据栅格
        
        Args:
            pointcloud: 点云数组 (N, 3) 或 (N, 6)，包含XYZ坐标
            pose: 机器人位姿 (x, y, yaw)，用于转换点云到世界坐标
        """
        if pointcloud is None or pointcloud.size == 0:
            return
        
        if pose is None:
            pose = (0.0, 0.0, 0.0)
        
        robot_x, robot_y, robot_yaw = pose
        
        # 提取XYZ坐标（前3列）
        if pointcloud.shape[1] >= 3:
            points = pointcloud[:, :3]
        else:
            return
        
        # 转换到世界坐标系
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        
        for point in points:
            x, y, z = point[0], point[1], point[2]
            
            # 过滤太远或太近的点
            dist = math.sqrt(x**2 + y**2)
            if dist < self.min_depth or dist > self.max_depth:
                continue
            
            # 转换到世界坐标（假设点云在机器人坐标系）
            world_x = robot_x + x * cos_yaw - y * sin_yaw
            world_y = robot_y + x * sin_yaw + y * cos_yaw
            
            # 更新占据栅格
            gx, gy = self.grid.world_to_grid(world_x, world_y)
            if self.grid.is_valid(gx, gy):
                self.grid.set_cell(gx, gy, CellState.OCCUPIED)
    
    def _update_free_space(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ):
        """更新从起点到终点之间的自由空间"""
        sx, sy = start
        ex, ey = end
        
        # 使用Bresenham算法画线
        gx1, gy1 = self.grid.world_to_grid(sx, sy)
        gx2, gy2 = self.grid.world_to_grid(ex, ey)
        
        # 简化的直线遍历
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        sx_step = 1 if gx1 < gx2 else -1
        sy_step = 1 if gy1 < gy2 else -1
        err = dx - dy
        
        x, y = gx1, gy1
        while True:
            # 不更新终点（那是障碍物）
            if x == gx2 and y == gy2:
                break
            
            if self.grid.is_valid(x, y):
                # 只标记未知或占据的为自由（不覆盖已标记的自由）
                if self.grid.get_cell(x, y) == CellState.UNKNOWN:
                    self.grid.set_cell(x, y, CellState.FREE)
            
            if x == gx2 and y == gy2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx_step
            if e2 < dx:
                err += dx
                y += sy_step
    
    def get_grid(self) -> OccupancyGrid:
        """获取占据栅格地图"""
        return self.grid
    
    def is_occupied_at(
        self,
        x: float,
        y: float
    ) -> bool:
        """检查世界坐标位置是否被占据"""
        gx, gy = self.grid.world_to_grid(x, y)
        return self.grid.is_occupied(gx, gy)
    
    def is_free_at(
        self,
        x: float,
        y: float
    ) -> bool:
        """检查世界坐标位置是否自由"""
        gx, gy = self.grid.world_to_grid(x, y)
        return self.grid.is_free(gx, gy)
    
    def get_nearest_obstacle(
        self,
        x: float,
        y: float,
        max_range: float = 5.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        获取最近的障碍物
        
        Returns:
            (obstacle_x, obstacle_y, distance) 或 None
        """
        gx, gy = self.grid.world_to_grid(x, y)
        max_grid_range = int(max_range / self.resolution)
        
        min_dist = float('inf')
        nearest = None
        
        # 搜索周围栅格
        for dy in range(-max_grid_range, max_grid_range + 1):
            for dx in range(-max_grid_range, max_grid_range + 1):
                check_gx = gx + dx
                check_gy = gy + dy
                
                if self.grid.is_occupied(check_gx, check_gy):
                    obs_x, obs_y = self.grid.grid_to_world(check_gx, check_gy)
                    dist = math.sqrt((obs_x - x)**2 + (obs_y - y)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest = (obs_x, obs_y, dist)
        
        return nearest
    
    def reset(self):
        """重置地图"""
        self.grid.data.fill(CellState.UNKNOWN)
        logger.info("占据栅格地图已重置")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取地图统计信息"""
        total = self.grid.width * self.grid.height
        occupied = np.sum(self.grid.data == CellState.OCCUPIED)
        free = np.sum(self.grid.data == CellState.FREE)
        unknown = np.sum(self.grid.data == CellState.UNKNOWN)
        
        return {
            "total_cells": total,
            "occupied": occupied,
            "free": free,
            "unknown": unknown,
            "coverage": (occupied + free) / total if total > 0 else 0.0,
            "resolution": self.resolution,
            "map_size": self.map_size
        }

