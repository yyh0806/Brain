"""
ROS2多传感器管理器 - ROS2 Sensor Manager

负责:
- 管理多种传感器数据源（RGB/深度/激光雷达/点云/里程计/IMU）
- 传感器数据融合
- 提供统一的感知数据接口
- 支持传感器数据缓存和历史记录
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from functools import partial
import numpy as np
import math
from loguru import logger

from brain.communication.ros2_interface import ROS2Interface, SensorData, ROS2Config
from brain.perception.mapping.occupancy_mapper import OccupancyMapper
from brain.perception.utils.coordinates import quaternion_to_euler, transform_local_to_world
from brain.perception.utils.math_utils import angle_to_direction, compute_laser_angles
from brain.perception.data_models import Pose2D, Pose3D, Velocity

if TYPE_CHECKING:
    from brain.perception.vlm.vlm_perception import DetectedObject, SceneDescription, VLMPerception


class SensorType(Enum):
    """传感器类型"""
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    POINTCLOUD = "pointcloud"
    ODOMETRY = "odometry"
    IMU = "imu"
    OCCUPANCY_GRID = "occupancy_grid"


@dataclass
class SensorStatus:
    """传感器状态"""
    sensor_type: SensorType
    enabled: bool = True
    connected: bool = False
    last_update: Optional[datetime] = None
    update_rate: float = 0.0  # Hz
    error_count: int = 0
    
    def is_healthy(self, timeout: float = 5.0) -> bool:
        """检查传感器是否健康"""
        if not self.enabled or not self.connected:
            return False
        if self.last_update is None:
            return False
        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed < timeout


@dataclass
class PerceptionData:
    """融合后的感知数据"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 位姿信息
    pose: Optional[Pose3D] = None
    velocity: Optional[Velocity] = None
    
    # 原始图像
    rgb_image: Optional[np.ndarray] = None
    rgb_image_right: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    
    # 激光雷达
    laser_ranges: Optional[List[float]] = None
    laser_angles: Optional[List[float]] = None
    
    # 点云
    pointcloud: Optional[np.ndarray] = None
    
    # 障碍物信息（从激光/深度推导）
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    
    # 占据栅格
    occupancy_grid: Optional[np.ndarray] = None
    grid_resolution: float = 0.05  # 米/栅格
    grid_origin: Tuple[float, float] = (0.0, 0.0)
    
    # 传感器状态
    sensor_status: Dict[str, bool] = field(default_factory=dict)
    
    # === 全局世界模型快照 ===
    # 全局占据地图（持久环境表示）
    global_map: Optional['OccupancyGrid'] = None
    # 语义物体（来自世界模型）
    semantic_objects: List['DetectedObject'] = field(default_factory=list)
    # 场景描述（来自世界模型）
    scene_description: Optional['SceneDescription'] = None
    # 空间关系（来自世界模型）
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)
    # 导航提示（来自世界模型）
    navigation_hints: List[str] = field(default_factory=list)
    # 世界模型元数据
    world_metadata: Optional[Dict[str, Any]] = None
    
    def get_front_distance(self) -> float:
        """获取前方障碍物距离"""
        if self.laser_ranges:
            # 取前方30度范围内的最小距离
            n = len(self.laser_ranges)
            center = n // 2
            window = n // 12  # 约30度
            front_ranges = self.laser_ranges[center - window:center + window]
            valid_ranges = [r for r in front_ranges if r > 0.1 and r < 30.0]
            if valid_ranges:
                return min(valid_ranges)
        return float('inf')
    
    def get_left_distance(self) -> float:
        """获取左侧障碍物距离"""
        if self.laser_ranges:
            n = len(self.laser_ranges)
            left_idx = n * 3 // 4  # 90度位置
            window = n // 12
            left_ranges = self.laser_ranges[left_idx - window:left_idx + window]
            valid_ranges = [r for r in left_ranges if r > 0.1 and r < 30.0]
            if valid_ranges:
                return min(valid_ranges)
        return float('inf')
    
    def get_right_distance(self) -> float:
        """获取右侧障碍物距离"""
        if self.laser_ranges:
            n = len(self.laser_ranges)
            right_idx = n // 4  # -90度位置
            window = n // 12
            right_ranges = self.laser_ranges[right_idx - window:right_idx + window]
            valid_ranges = [r for r in right_ranges if r > 0.1 and r < 30.0]
            if valid_ranges:
                return min(valid_ranges)
        return float('inf')
    
    def is_path_clear(self, direction: str = "front", threshold: float = 1.0) -> bool:
        """检查指定方向路径是否畅通"""
        if direction == "front":
            return self.get_front_distance() > threshold
        elif direction == "left":
            return self.get_left_distance() > threshold
        elif direction == "right":
            return self.get_right_distance() > threshold
        return True


class ROS2SensorManager:
    """
    ROS2传感器管理器
    
    统一管理多种传感器，提供融合后的感知数据，
    并集成全局世界模型进行持久环境表示。
    """
    
    def __init__(
        self,
        ros2_interface: ROS2Interface,
        config: Optional[Dict[str, Any]] = None,
        vlm: Optional['VLMPerception'] = None,
        world_model: Optional['WorldModel'] = None
    ):
        self.ros2 = ros2_interface
        # 公共属性用于测试框架
        self.ros2_interface_for_test = ros2_interface
        self.config = config or {}
        
        # 传感器状态
        self.sensor_status: Dict[SensorType, SensorStatus] = {}
        self._init_sensor_status()
        
        # 数据缓存（使用循环缓冲区防止内存泄漏）
        self._latest_data: Optional[PerceptionData] = None
        max_history = self.config.get("max_history", 100)
        self._data_history: deque = deque(maxlen=max_history)  # 固定大小的循环缓冲区
        
        # 传感器融合参数
        self._pose_filter_alpha = self.config.get("pose_filter_alpha", 0.8)
        
        # 障碍物检测参数
        self._obstacle_threshold = self.config.get("obstacle_threshold", 0.5)
        self._min_obstacle_size = self.config.get("min_obstacle_size", 0.1)
        
        # 占据栅格生成器（保留以兼容性，但主要使用WorldModel）
        # 注意：WorldModel内部已经包含OccupancyMapper
        grid_resolution = self.config.get("grid_resolution", 0.1)
        map_size = self.config.get("map_size", 50.0)
        self.occupancy_mapper = OccupancyMapper(
            resolution=grid_resolution,
            map_size=map_size,
            config=self.config.get("occupancy", {})
        )
        
        # 初始化VLM（如果提供）- 已弃用，改用异步VLMService
        self.vlm_sync = vlm  # 保留向后兼容
        self._vlm_service = None  # 新的异步VLM服务
        
        # 初始化WorldModel（全局世界模型）
        if world_model is None:
            from brain.perception.world_model import WorldModel
            world_model = WorldModel(
                resolution=grid_resolution,
                map_size=map_size,
                config=self.config.get("world_model", {})
            )
        self.world_model = world_model
        
        # 初始化VLMService（异步VLM分析服务）
        if vlm is not None and self._vlm_service is None:
            from brain.perception.vlm_service import VLMService
            self._vlm_service = VLMService(
                vlm=vlm,
                max_workers=self.config.get("vlm", {}).get("max_workers", 1),
                cache_size=self.config.get("vlm", {}).get("cache_size", 10),
                timeout=self.config.get("vlm", {}).get("timeout", 30.0),
                enable_cache=self.config.get("vlm", {}).get("enable_cache", True)
            )
            logger.info("VLM异步服务已初始化")
        elif vlm is None and self._vlm_service is not None:
            # VLM被禁用但服务存在，停止服务
            logger.info("VLM被禁用，停止VLM异步服务")
            self._vlm_service = None
        
        # 集成WorldModel（全局世界模型）
        if world_model is None:
            from brain.perception.world_model import WorldModel
            world_model = WorldModel(
                resolution=grid_resolution,
                map_size=map_size,
                config=self.config.get("world_model", {})
            )
        self.world_model = world_model
        
        # 初始化VLM（如果提供）- 已弃用，改用异步VLMService
        self.vlm_sync = vlm  # 保留向后兼容
        self._vlm_service = None  # 新的异步VLM服务
        
        logger.info("ROS2SensorManager 初始化完成" + 
                   (" (VLM已启用)" if vlm is not None else "") +
                   " (WorldModel已集成)")
    
    def _init_sensor_status(self):
        """初始化传感器状态"""
        for sensor_type in SensorType:
            enabled = self.config.get(f"sensors.{sensor_type.value}.enabled", True)
            self.sensor_status[sensor_type] = SensorStatus(
                sensor_type=sensor_type,
                enabled=enabled
            )
    
    async def get_fused_perception(self) -> PerceptionData:
        """
        获取融合后的感知数据
        
        Returns:
            PerceptionData: 融合后的感知数据
        """
        # 从ROS2接口获取原始数据
        raw_data = self.ros2.get_sensor_data()
        
        # 创建感知数据
        perception = PerceptionData(timestamp=raw_data.timestamp)
        
        # 处理位姿
        if raw_data.odometry:
            perception.pose = self._extract_pose(raw_data.odometry)
            perception.velocity = self._extract_velocity(raw_data.odometry)
            self._update_sensor_status(SensorType.ODOMETRY, True)
        
        # 处理IMU（用于姿态融合）
        if raw_data.imu:
            if perception.pose:
                perception.pose = self._fuse_imu_pose(perception.pose, raw_data.imu)
            self._update_sensor_status(SensorType.IMU, True)
        
        # 处理RGB图像（左眼）
        if raw_data.rgb_image is not None:
            perception.rgb_image = raw_data.rgb_image
            self._update_sensor_status(SensorType.RGB_CAMERA, True)
        
        # 处理RGB图像（右眼）
        if hasattr(raw_data, 'rgb_image_right') and raw_data.rgb_image_right is not None:
            perception.rgb_image_right = raw_data.rgb_image_right
        
        # 处理深度图像
        if raw_data.depth_image is not None:
            perception.depth_image = raw_data.depth_image
            self._update_sensor_status(SensorType.DEPTH_CAMERA, True)
        
        # 处理激光雷达
        if raw_data.laser_scan:
            perception.laser_ranges = raw_data.laser_scan.get("ranges", [])
            perception.laser_angles = compute_laser_angles(raw_data.laser_scan)
            perception.obstacles = self._detect_obstacles_from_laser(
                perception.laser_ranges,
                perception.laser_angles,
                perception.pose
            )
            self._update_sensor_status(SensorType.LIDAR, True)
        
        # 处理点云
        if raw_data.pointcloud is not None:
            perception.pointcloud = raw_data.pointcloud
            self._update_sensor_status(SensorType.POINTCLOUD, True)
            
            # 如果激光雷达数据不存在，从点云转换
            if not raw_data.laser_scan and perception.pose:
                laser_data = self._convert_pointcloud_to_laser(
                    raw_data.pointcloud,
                    perception.pose
                )
                if laser_data:
                    perception.laser_ranges = laser_data["ranges"]
                    perception.laser_angles = laser_data["angles"]
                    # 使用转换后的激光雷达数据检测障碍物
                    perception.obstacles = self._detect_obstacles_from_laser(
                        perception.laser_ranges,
                        perception.laser_angles,
                        perception.pose
                    )
                    self._update_sensor_status(SensorType.LIDAR, True)
                    logger.debug(f"从点云转换激光雷达数据: {len(perception.laser_ranges)} 个点")
        
        # 处理占据地图（从外部地图）
        if raw_data.occupancy_grid is not None:
            perception.occupancy_grid = raw_data.occupancy_grid
            if raw_data.map_info:
                perception.grid_resolution = raw_data.map_info.get("resolution", 0.05)
                origin = raw_data.map_info.get("origin", {})
                perception.grid_origin = (origin.get("x", 0), origin.get("y", 0))
            self._update_sensor_status(SensorType.OCCUPANCY_GRID, True)
        
        # 从深度图/激光/点云生成占据栅格（异步处理CPU密集型操作）
        if perception.pose:
            pose_2d = (perception.pose.x, perception.pose.y, perception.pose.yaw)
            
            # 获取事件循环（Python 3.8兼容）
            loop = asyncio.get_event_loop()
            
            # 从深度图更新（异步）
            if perception.depth_image is not None:
                # 使用异步处理避免阻塞（Python 3.8兼容）
                # 使用partial包装函数以支持关键字参数
                func = partial(self.occupancy_mapper.update_from_depth, pose=pose_2d)
                await loop.run_in_executor(None, func, perception.depth_image)
            
            # 从激光雷达更新（异步）
            if perception.laser_ranges and perception.laser_angles:
                func = partial(self.occupancy_mapper.update_from_laser, pose=pose_2d)
                await loop.run_in_executor(
                    None, func, perception.laser_ranges, perception.laser_angles
                )
            
            # 从点云更新（异步）
            if perception.pointcloud is not None:
                func = partial(self.occupancy_mapper.update_from_pointcloud, pose=pose_2d)
                await loop.run_in_executor(None, func, perception.pointcloud)
            
            # 将生成的栅格添加到感知数据
            perception.occupancy_grid = self.occupancy_mapper.get_grid().data
            perception.grid_resolution = self.occupancy_mapper.resolution
            perception.grid_origin = (
                self.occupancy_mapper.grid.origin_x,
                self.occupancy_mapper.grid.origin_y
            )
        
        # 更新传感器状态到感知数据
        perception.sensor_status = {
            sensor_type.value: status.is_healthy()
            for sensor_type, status in self.sensor_status.items()
        }
        
        # 更新全局世界模型（持久地图）
        if self.world_model:
            self.world_model.update_with_perception(perception)
        
        # 从世界模型获取地图快照（持久、融合的环境表示）
        if self.world_model:
            perception.global_map = self.world_model.get_global_map()
            perception.semantic_objects = list(self.world_model.semantic_objects.values())
            perception.scene_description = self.world_model.metadata.scene_description if hasattr(self.world_model.metadata, 'scene_description') else None
            perception.spatial_relations = self.world_model.spatial_relations.copy()
            perception.navigation_hints = self.world_model.metadata.navigation_hints if hasattr(self.world_model.metadata, 'navigation_hints') else []
            perception.world_metadata = {
                "created_at": self.world_model.metadata.created_at.isoformat(),
                "last_updated": self.world_model.metadata.last_updated.isoformat(),
                "update_count": self.world_model.metadata.update_count,
                "confidence": self.world_model.metadata.confidence,
                "map_stats": self.world_model.get_map_statistics()
            }
        
        # 触发异步VLM分析（不阻塞主循环）
        if self._vlm_service and perception.rgb_image is not None:
            try:
                # 提交VLM分析请求到异步服务
                request_id = await self._vlm_service.analyze_image(perception.rgb_image)
                logger.debug(f"VLM分析请求已提交: {request_id}")
            except Exception as e:
                logger.error(f"提交VLM分析请求失败: {e}")
        
        # 缓存数据（使用循环缓冲区，自动限制大小）
        self._latest_data = perception
        self._data_history.append(perception)  # deque会自动移除最旧的元素
        
        return perception
    
    async def start_vlm_service(self) -> None:
        """启动VLM异步服务"""
        if self.vlm_sync and not self._vlm_service:
            from brain.perception.vlm_service import VLMService
            self._vlm_service = VLMService(
                vlm=self.vlm_sync,
                max_workers=self.config.get("vlm", {}).get("max_workers", 1),
                cache_size=self.config.get("vlm", {}).get("cache_size", 10),
                timeout=self.config.get("vlm", {}).get("timeout", 30.0),
                enable_cache=self.config.get("vlm", {}).get("enable_cache", True)
            )
            await self._vlm_service.start()
            logger.info("VLM异步服务已启动")
    
    async def stop_vlm_service(self) -> None:
        """停止VLM异步服务"""
        if self._vlm_service:
            await self._vlm_service.stop()
            logger.info("VLM异步服务已停止")
    
    def get_vlm_service_statistics(self) -> Dict[str, Any]:
        """获取VLM服务统计"""
        if self._vlm_service:
            return self._vlm_service.get_statistics()
        return {"running": False, "total_requests": 0}
    
    def get_world_model_statistics(self) -> Dict[str, Any]:
        """获取世界模型统计"""
        if self.world_model:
            return self.world_model.get_map_statistics()
        return None
    
    def _extract_pose(self, odom: Dict[str, Any]) -> Pose3D:
        """从里程计提取位姿"""
        pos = odom.get("position", {})
        orient = odom.get("orientation", {})
        
        # 从四元数计算欧拉角
        q = (orient.get("x", 0), orient.get("y", 0), 
             orient.get("z", 0), orient.get("w", 1))
        roll, pitch, yaw = quaternion_to_euler(q)
        
        return Pose3D(
            x=pos.get("x", 0),
            y=pos.get("y", 0),
            z=pos.get("z", 0),
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )
    
    def _extract_velocity(self, odom: Dict[str, Any]) -> Velocity:
        """从里程计提取速度"""
        linear = odom.get("linear_velocity", {})
        angular = odom.get("angular_velocity", {})
        
        return Velocity(
            linear_x=linear.get("x", 0),
            linear_y=linear.get("y", 0),
            linear_z=linear.get("z", 0),
            angular_x=angular.get("x", 0),
            angular_y=angular.get("y", 0),
            angular_z=angular.get("z", 0)
        )
    
    def _fuse_imu_pose(self, pose: Pose3D, imu: Dict[str, Any]) -> Pose3D:
        """融合IMU数据到位姿"""
        imu_orient = imu.get("orientation", {})
        q = (imu_orient.get("x", 0), imu_orient.get("y", 0),
             imu_orient.get("z", 0), imu_orient.get("w", 1))
        imu_roll, imu_pitch, imu_yaw = quaternion_to_euler(q)
        
        # 互补滤波
        alpha = self._pose_filter_alpha
        pose.roll = alpha * pose.roll + (1 - alpha) * imu_roll
        pose.pitch = alpha * pose.pitch + (1 - alpha) * imu_pitch
        # yaw主要来自里程计，IMU漂移大
        
        return pose
    
    
    def _convert_pointcloud_to_laser(
        self,
        pointcloud: np.ndarray,
        pose: Optional[Pose3D] = None
    ) -> Optional[Dict[str, List[float]]]:
        """
        将点云转换为激光雷达格式（ranges, angles）
        
        Args:
            pointcloud: 点云数组 (N, 3) 或 (N, 6)，包含XYZ坐标
            pose: 机器人位姿，用于转换到机器人坐标系
            
        Returns:
            Dict包含 "ranges" 和 "angles" 列表，如果转换失败返回None
        """
        if pointcloud is None or pointcloud.size == 0:
            return None
        
        # 提取XYZ坐标（前3列）
        if pointcloud.shape[1] >= 3:
            points = pointcloud[:, :3]
        else:
            return None
        
        # 如果有点云但为空，返回None
        if len(points) == 0:
            return None
        
        # 转换到机器人坐标系（如果提供了位姿）
        if pose:
            # 将世界坐标转换为机器人局部坐标
            cos_yaw = math.cos(-pose.yaw)  # 反向旋转
            sin_yaw = math.sin(-pose.yaw)
            
            # 相对于机器人的位置
            rel_points = points.copy()
            rel_points[:, 0] -= pose.x
            rel_points[:, 1] -= pose.y
            
            # 旋转到机器人坐标系
            x_local = rel_points[:, 0] * cos_yaw - rel_points[:, 1] * sin_yaw
            y_local = rel_points[:, 0] * sin_yaw + rel_points[:, 1] * cos_yaw
        else:
            # 假设点云已经在机器人坐标系
            x_local = points[:, 0]
            y_local = points[:, 1]
        
        # 转换为极坐标
        ranges = np.sqrt(x_local**2 + y_local**2)
        angles = np.arctan2(y_local, x_local)
        
        # 过滤无效点
        valid_mask = (ranges > 0.1) & (ranges < 30.0) & np.isfinite(ranges) & np.isfinite(angles)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return None
        
        # 按角度排序
        sort_indices = np.argsort(valid_angles)
        sorted_ranges = valid_ranges[sort_indices].tolist()
        sorted_angles = valid_angles[sort_indices].tolist()
        
        return {
            "ranges": sorted_ranges,
            "angles": sorted_angles
        }
    
    def _detect_obstacles_from_laser(
        self,
        ranges: List[float],
        angles: List[float],
        pose: Optional[Pose3D]
    ) -> List[Dict[str, Any]]:
        """从激光雷达检测障碍物"""
        obstacles = []
        
        if not ranges or not angles:
            return obstacles
        
        # 聚类参数
        distance_threshold = 0.3  # 相邻点距离阈值
        min_points = 3  # 最小点数
        
        # 简单聚类算法
        current_cluster = []
        clusters = []
        
        for i, (r, a) in enumerate(zip(ranges, angles)):
            if r < 0.1 or r > 30.0:  # 过滤无效点
                if current_cluster:
                    if len(current_cluster) >= min_points:
                        clusters.append(current_cluster)
                    current_cluster = []
                continue
            
            # 转换为笛卡尔坐标
            x = r * math.cos(a)
            y = r * math.sin(a)
            
            if current_cluster:
                # 检查与上一个点的距离
                last_x, last_y, _, _ = current_cluster[-1]
                dist = math.sqrt((x - last_x)**2 + (y - last_y)**2)
                
                if dist > distance_threshold:
                    if len(current_cluster) >= min_points:
                        clusters.append(current_cluster)
                    current_cluster = []
            
            current_cluster.append((x, y, r, a))
        
        if len(current_cluster) >= min_points:
            clusters.append(current_cluster)
        
        # 为每个聚类创建障碍物
        for cluster in clusters:
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            
            # 计算中心
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            # 计算大小
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            size = max(width, height)
            
            if size < self._min_obstacle_size:
                continue
            
            # 计算距离
            distance = math.sqrt(center_x**2 + center_y**2)
            
            # 计算方向
            angle = math.atan2(center_y, center_x)
            direction = angle_to_direction(angle)
            
            # 转换到世界坐标（如果有位姿）
            if pose:
                world_x, world_y = transform_local_to_world(
                    center_x, center_y, pose.x, pose.y, pose.yaw
                )
            else:
                world_x, world_y = center_x, center_y
            
            obstacles.append({
                "id": f"obs_{len(obstacles)}",
                "type": "unknown",
                "local_position": {"x": center_x, "y": center_y},
                "world_position": {"x": world_x, "y": world_y},
                "size": size,
                "distance": distance,
                "angle": angle,
                "direction": direction,
                "point_count": len(cluster)
            })
        
        return obstacles
    
    def _should_run_vlm_analysis(self) -> bool:
        """判断是否应该运行VLM分析（频率控制）"""
        if not self._vlm_enabled:
            return False
        if self._last_vlm_analysis is None:
            return True
        elapsed = (datetime.now() - self._last_vlm_analysis).total_seconds()
        return elapsed >= self._vlm_analysis_interval
    
    def _update_sensor_status(self, sensor_type: SensorType, connected: bool):
        """更新传感器状态"""
        status = self.sensor_status.get(sensor_type)
        if status:
            status.connected = connected
            now = datetime.now()
            if status.last_update:
                dt = (now - status.last_update).total_seconds()
                if dt > 0:
                    status.update_rate = 1.0 / dt
            status.last_update = now
    
    def get_current_pose(self) -> Optional[Pose3D]:
        """获取当前位姿"""
        if self._latest_data:
            return self._latest_data.pose
        return None
    
    def get_current_pose_2d(self) -> Optional[Pose2D]:
        """获取当前2D位姿"""
        pose3d = self.get_current_pose()
        if pose3d:
            return pose3d.to_2d()
        return None
    
    def get_rgb_image(self) -> Optional[np.ndarray]:
        """获取最新RGB图像"""
        return self.ros2.get_rgb_image()
    
    def get_depth_image(self) -> Optional[np.ndarray]:
        """获取最新深度图像"""
        return self.ros2.get_depth_image()
    
    def get_laser_scan(self) -> Optional[Dict[str, Any]]:
        """获取最新激光雷达数据"""
        return self.ros2.get_laser_scan()
    
    def get_nearest_obstacle(self) -> Optional[Dict[str, Any]]:
        """获取最近的障碍物"""
        if self._latest_data and self._latest_data.obstacles:
            return min(self._latest_data.obstacles, key=lambda o: o["distance"])
        return None
    
    def get_obstacles_in_direction(self, direction: str) -> List[Dict[str, Any]]:
        """获取指定方向的障碍物"""
        if not self._latest_data:
            return []
        
        return [
            obs for obs in self._latest_data.obstacles
            if obs.get("direction", "").startswith(direction)
        ]
    
    def get_sensor_health(self) -> Dict[str, bool]:
        """获取所有传感器健康状态"""
        return {
            sensor_type.value: status.is_healthy()
            for sensor_type, status in self.sensor_status.items()
        }
    
    def get_data_history(self, count: int = 10) -> List[PerceptionData]:
        """获取历史数据"""
        # 返回最近的count条数据
        return list(self._data_history)[-count:] if len(self._data_history) > 0 else []
    
    async def wait_for_sensors(self, timeout: float = 10.0) -> bool:
        """等待传感器就绪"""
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            # 尝试获取数据
            await self.get_fused_perception()
            
            # 检查关键传感器
            essential_sensors = [SensorType.ODOMETRY]
            all_ready = all(
                self.sensor_status[s].is_healthy()
                for s in essential_sensors
            )
            
            if all_ready:
                logger.info("传感器已就绪")
                return True
            
            await asyncio.sleep(0.5)
        
        logger.warning("等待传感器超时")
        return False
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            Dictionary containing manager statistics
        """
        return {
            "total_sensors": len(self.sensor_status),
            "active_sensors": sum(1 for status in self.sensor_status.values() if status.connected),
            "enabled_sensors": sum(1 for status in self.sensor_status.values() if status.enabled),
            "data_history_size": len(self._data_history),
            "last_update": str(self._latest_data.timestamp) if self._latest_data else "N/A",
            "sensor_types": {s.name: status.connected for s, status in self.sensor_status.items()},
            "ros2_interface_initialized": self.ros2._initialized if hasattr(self.ros2, '_initialized') else False
        }

