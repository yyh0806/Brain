"""
多传感器融合模块 - Sensor Fusion

负责:
- 里程计与IMU融合（扩展卡尔曼滤波）
- RGB与深度图融合
- 激光雷达与点云融合
- 多源位姿估计
"""

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from brain.perception.utils.coordinates import quaternion_to_euler, normalize_angles


@dataclass
class FusedPose:
    """融合后的位姿"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # 协方差
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.1)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FusedPose':
        return cls(
            x=arr[0], y=arr[1], z=arr[2],
            roll=arr[3], pitch=arr[4], yaw=arr[5]
        )


class EKFPoseFusion:
    """
    扩展卡尔曼滤波器用于位姿融合
    
    状态向量: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    """
    
    def __init__(self):
        # 状态向量 (12维)
        self.state = np.zeros(12)
        
        # 状态协方差
        self.P = np.eye(12) * 0.1
        
        # 过程噪声协方差
        self.Q = np.eye(12) * 0.01
        self.Q[6:9, 6:9] *= 0.1   # 速度噪声
        self.Q[9:12, 9:12] *= 0.1  # 角速度噪声
        
        # 测量噪声协方差
        self.R_odom = np.eye(6) * 0.1  # 里程计测量噪声
        self.R_imu = np.eye(6) * 0.05  # IMU测量噪声
        
        self.last_update = None
        self.initialized = False
    
    def predict(self, dt: float):
        """预测步骤"""
        if not self.initialized:
            return
        
        # 状态转移矩阵
        F = np.eye(12)
        F[0, 6] = dt  # x += vx * dt
        F[1, 7] = dt  # y += vy * dt
        F[2, 8] = dt  # z += vz * dt
        F[3, 9] = dt  # roll += wx * dt
        F[4, 10] = dt  # pitch += wy * dt
        F[5, 11] = dt  # yaw += wz * dt
        
        # 状态预测
        self.state = F @ self.state
        
        # 协方差预测
        self.P = F @ self.P @ F.T + self.Q * dt
    
    def update_odom(self, odom: Dict[str, Any]):
        """里程计更新"""
        pos = odom.get("position", {})
        orient = odom.get("orientation", {})
        vel = odom.get("linear_velocity", {})
        ang_vel = odom.get("angular_velocity", {})
        
        # 提取测量值
        z_pos = np.array([
            pos.get("x", 0),
            pos.get("y", 0),
            pos.get("z", 0)
        ])
        
        # 四元数转欧拉角
        q = (orient.get("x", 0), orient.get("y", 0),
             orient.get("z", 0), orient.get("w", 1))
        roll, pitch, yaw = quaternion_to_euler(q)
        z_orient = np.array([roll, pitch, yaw])
        
        z = np.concatenate([z_pos, z_orient])
        
        if not self.initialized:
            self.state[:6] = z
            self.state[6:9] = [vel.get("x", 0), vel.get("y", 0), vel.get("z", 0)]
            self.state[9:12] = [ang_vel.get("x", 0), ang_vel.get("y", 0), ang_vel.get("z", 0)]
            self.initialized = True
            self.last_update = datetime.now()
            return
        
        # 更新速度
        self.state[6:9] = [vel.get("x", 0), vel.get("y", 0), vel.get("z", 0)]
        self.state[9:12] = [ang_vel.get("x", 0), ang_vel.get("y", 0), ang_vel.get("z", 0)]
        
        # 测量矩阵
        H = np.zeros((6, 12))
        H[:6, :6] = np.eye(6)
        
        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R_odom
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 状态更新
        y = z - H @ self.state
        # 角度归一化
        y[3:6] = normalize_angles(y[3:6])
        
        self.state = self.state + K @ y
        
        # 协方差更新
        I = np.eye(12)
        self.P = (I - K @ H) @ self.P
        
        self.last_update = datetime.now()
    
    def update_imu(self, imu: Dict[str, Any]):
        """IMU更新"""
        if not self.initialized:
            return
        
        orient = imu.get("orientation", {})
        ang_vel = imu.get("angular_velocity", {})
        lin_acc = imu.get("linear_acceleration", {})
        
        # 四元数转欧拉角
        q = (orient.get("x", 0), orient.get("y", 0),
             orient.get("z", 0), orient.get("w", 1))
        roll, pitch, yaw = quaternion_to_euler(q)
        
        # IMU只更新姿态（roll, pitch）和角速度
        # yaw受磁场影响可能不准确
        z = np.array([roll, pitch])
        
        # 测量矩阵（只更新roll和pitch）
        H = np.zeros((2, 12))
        H[0, 3] = 1  # roll
        H[1, 4] = 1  # pitch
        
        # 卡尔曼增益
        R = self.R_imu[:2, :2]
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 状态更新
        y = z - H @ self.state
        y = normalize_angles(y)
        
        self.state = self.state + K @ y
        
        # 协方差更新
        I = np.eye(12)
        self.P = (I - K @ H) @ self.P
        
        # 更新角速度
        self.state[9:12] = [
            ang_vel.get("x", 0),
            ang_vel.get("y", 0),
            ang_vel.get("z", 0)
        ]
    
    def get_pose(self) -> FusedPose:
        """获取融合后的位姿"""
        return FusedPose(
            x=self.state[0],
            y=self.state[1],
            z=self.state[2],
            roll=self.state[3],
            pitch=self.state[4],
            yaw=self.state[5],
            covariance=self.P[:6, :6].copy(),
            timestamp=self.last_update or datetime.now()
        )
    
    def get_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取速度 (线速度, 角速度)"""
        return self.state[6:9].copy(), self.state[9:12].copy()
    


class DepthRGBFusion:
    """
    深度图与RGB图像融合
    
    生成RGBD数据和3D点
    """
    
    def __init__(
        self,
        fx: float = 525.0,  # 相机焦距x
        fy: float = 525.0,  # 相机焦距y
        cx: float = 319.5,  # 光心x
        cy: float = 239.5   # 光心y
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def fuse(
        self,
        rgb: np.ndarray,
        depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        融合RGB和深度图
        
        Returns:
            rgbd: RGBD图像 (H, W, 4) - RGB + Depth
            points: 3D点云 (N, 6) - XYZ + RGB
        """
        if rgb.shape[:2] != depth.shape[:2]:
            logger.warning("RGB和深度图尺寸不匹配")
            return None, None
        
        # 创建RGBD图像
        h, w = depth.shape
        rgbd = np.zeros((h, w, 4), dtype=np.float32)
        rgbd[:, :, :3] = rgb.astype(np.float32) / 255.0
        rgbd[:, :, 3] = depth
        
        # 生成点云
        points = self._depth_to_pointcloud(rgb, depth)
        
        return rgbd, points
    
    def _depth_to_pointcloud(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        max_depth: float = 10.0
    ) -> np.ndarray:
        """将深度图转换为点云"""
        h, w = depth.shape
        
        # 创建像素坐标网格
        u = np.arange(w)
        v = np.arange(h)
        u, v = np.meshgrid(u, v)
        
        # 过滤有效深度
        valid = (depth > 0) & (depth < max_depth)
        
        # 计算3D坐标
        z = depth[valid]
        x = (u[valid] - self.cx) * z / self.fx
        y = (v[valid] - self.cy) * z / self.fy
        
        # 获取颜色
        colors = rgb[valid]
        
        # 组合点云 (N, 6): X, Y, Z, R, G, B
        points = np.column_stack([
            x, y, z,
            colors[:, 0], colors[:, 1], colors[:, 2]
        ])
        
        return points.astype(np.float32)
    
    def project_point_to_image(
        self,
        point_3d: Tuple[float, float, float]
    ) -> Optional[Tuple[int, int]]:
        """将3D点投影到图像平面"""
        x, y, z = point_3d
        
        if z <= 0:
            return None
        
        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)
        
        return (u, v)


class ObstacleDetector:
    """
    多传感器障碍物检测与融合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 检测参数
        self.depth_threshold = self.config.get("depth_threshold", 3.0)  # 米
        self.laser_threshold = self.config.get("laser_threshold", 5.0)  # 米
        self.fusion_distance = self.config.get("fusion_distance", 0.5)  # 米
    
    def detect_from_depth(
        self,
        depth: np.ndarray,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """从深度图检测障碍物"""
        threshold = threshold or self.depth_threshold
        
        # 简化实现：检测近距离区域
        h, w = depth.shape
        obstacles = []
        
        # 将深度图分成区域
        regions = [
            ("left", depth[:, :w//3]),
            ("center", depth[:, w//3:2*w//3]),
            ("right", depth[:, 2*w//3:])
        ]
        
        for name, region in regions:
            valid = (region > 0) & (region < threshold)
            if np.any(valid):
                min_depth = np.min(region[valid]) if np.any(valid) else float('inf')
                if min_depth < threshold:
                    obstacles.append({
                        "source": "depth",
                        "region": name,
                        "distance": float(min_depth),
                        "confidence": 0.8
                    })
        
        return obstacles
    
    def detect_from_laser(
        self,
        ranges: List[float],
        angles: List[float],
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """从激光雷达检测障碍物"""
        threshold = threshold or self.laser_threshold
        
        obstacles = []
        
        if not ranges:
            return obstacles
        
        n = len(ranges)
        
        # 分区域检测
        regions = [
            ("front", n//4, 3*n//4),
            ("left", 3*n//4, n),
            ("right", 0, n//4)
        ]
        
        for name, start, end in regions:
            region_ranges = ranges[start:end]
            valid = [r for r in region_ranges if 0.1 < r < threshold]
            
            if valid:
                min_dist = min(valid)
                obstacles.append({
                    "source": "laser",
                    "region": name,
                    "distance": min_dist,
                    "confidence": 0.9
                })
        
        return obstacles
    
    def fuse_obstacles(
        self,
        depth_obstacles: List[Dict[str, Any]],
        laser_obstacles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """融合多源障碍物检测结果"""
        fused = []
        
        # 按区域匹配
        for region in ["front", "center", "left", "right"]:
            depth_obs = [o for o in depth_obstacles if o.get("region") == region or 
                        (region == "front" and o.get("region") == "center")]
            laser_obs = [o for o in laser_obstacles if o.get("region") == region]
            
            if depth_obs and laser_obs:
                # 融合
                avg_dist = (depth_obs[0]["distance"] + laser_obs[0]["distance"]) / 2
                fused.append({
                    "source": "fused",
                    "region": region,
                    "distance": avg_dist,
                    "confidence": 0.95,
                    "depth_dist": depth_obs[0]["distance"],
                    "laser_dist": laser_obs[0]["distance"]
                })
            elif laser_obs:
                fused.append(laser_obs[0])
            elif depth_obs:
                fused.append(depth_obs[0])
        
        return fused

