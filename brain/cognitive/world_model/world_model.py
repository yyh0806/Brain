"""
感知驱动的世界模型 - World Model

负责：
- 融合多传感器数据构建环境表示
- 追踪环境变化（新障碍物、目标移动、天气变化等）
- 评估变化的显著性，决定是否触发重规划
- 为规划提供丰富的环境上下文
- 语义物体追踪和探索管理（合并自 SemanticWorldModel）
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from copy import deepcopy
import math
import numpy as np
from loguru import logger

# 导入类型定义
from brain.cognitive.world_model.environment_change import (
    ChangeType,
    ChangePriority,
    EnvironmentChange
)
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.object_tracking.tracked_object import TrackedObject
from brain.cognitive.world_model.semantic.semantic_object import (
    ObjectState,
    SemanticObject,
    ExplorationFrontier
)
from brain.cognitive.world_model.causal_graph import (
    CausalGraph,
    CausalNode,
    CausalRelationType
)

# 导入资源管理器
try:
    from brain.utils.resource_manager import get_resource_manager, ResourceType, track_resource
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("资源管理器不可用，将运行在无资源管理模式")
    RESOURCE_MANAGER_AVAILABLE = False

# 导入 VLM 相关类型
try:
    from brain.perception.understanding.vlm_perception import (
        DetectedObject, SceneDescription, BoundingBox
    )
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logger.warning("VLM perception not available")

# 导入 PerceptionData
try:
    from brain.perception.ros2_sensor_manager import PerceptionData
    PERCEPTION_DATA_AVAILABLE = True
except ImportError:
    PERCEPTION_DATA_AVAILABLE = False
    PerceptionData = None

# 导入信念修正策略
try:
    from brain.cognitive.world_model.belief_revision import (
        BeliefRevisionPolicy, BeliefType, OperationType, BeliefEntry
    )
    BELIEF_REVISION_AVAILABLE = True
except ImportError:
    BELIEF_REVISION_AVAILABLE = False
    logger.warning("信念修正模块不可用")
    BeliefRevisionPolicy = None
    BeliefType = None
    OperationType = None
    BeliefEntry = None


class WorldModel:
    """
    感知驱动的世界模型（统一版本）
    
    实时维护环境认知，融合感知数据，检测显著变化
    整合了语义物体追踪和探索管理功能
    """
    
    # 感兴趣的物体类型（用于导航，从 SemanticWorldModel 迁移）
    NAVIGATION_OBJECTS = {
        "门", "door", "entrance", "入口", "出口", "exit",
        "建筑", "building", "房子", "house",
        "路", "road", "path", "道路", "走廊", "corridor",
        "楼梯", "stairs", "电梯", "elevator"
    }
    
    # 障碍物类型（从 SemanticWorldModel 迁移）
    OBSTACLE_TYPES = {
        "墙", "wall", "障碍", "obstacle", "栏杆", "fence",
        "车", "car", "vehicle", "人", "person", "pedestrian"
    }
    
    # 变化类型的优先级和阈值配置
    CHANGE_CONFIG = {
        ChangeType.NEW_OBSTACLE: {
            "priority": ChangePriority.HIGH,
            "threshold": 0.7,
            "requires_replan": True,
            "requires_confirmation": False
        },
        ChangeType.PATH_BLOCKED: {
            "priority": ChangePriority.CRITICAL,
            "threshold": 0.9,
            "requires_replan": True,
            "requires_confirmation": True
        },
        ChangeType.TARGET_APPEARED: {
            "priority": ChangePriority.HIGH,
            "threshold": 0.6,
            "requires_replan": False,
            "requires_confirmation": True
        },
        ChangeType.TARGET_MOVED: {
            "priority": ChangePriority.MEDIUM,
            "threshold": 0.5,
            "requires_replan": True,
            "requires_confirmation": False
        },
        ChangeType.WEATHER_CHANGED: {
            "priority": ChangePriority.MEDIUM,
            "threshold": 0.7,
            "requires_replan": True,
            "requires_confirmation": True
        },
        ChangeType.BATTERY_LOW: {
            "priority": ChangePriority.CRITICAL,
            "threshold": 0.95,
            "requires_replan": True,
            "requires_confirmation": False
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 资源管理
        self.resource_manager = None
        self.tracker = None
        if RESOURCE_MANAGER_AVAILABLE:
            self.resource_manager = get_resource_manager()
            self.tracker = self.resource_manager.create_tracker("world_model")

        # 机器人状态
        self.robot_position: Dict[str, float] = {"x": 0, "y": 0, "z": 0, "lat": 0, "lon": 0, "alt": 0}
        self.robot_velocity: Dict[str, float] = {"vx": 0, "vy": 0, "vz": 0}
        self.robot_heading: float = 0.0
        self.battery_level: float = 100.0
        self.signal_strength: float = 100.0

        # 环境状态
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.weather: Dict[str, Any] = {
            "condition": "clear",
            "wind_speed": 0.0,
            "wind_direction": 0.0,
            "visibility": "good",
            "temperature": 25.0
        }

        # 空间信息（优化：限制地图内存使用）
        # 注意：不再维护 occupied_cells，直接使用 OccupancyMapper 生成的地图
        self.current_map: Optional[np.ndarray] = None  # 占据栅格地图（来自 OccupancyMapper）
        self.map_resolution: float = 0.1  # 地图分辨率
        self.map_origin: Tuple[float, float] = (0.0, 0.0)  # 地图原点
        self.max_map_size = self.config.get("max_map_size", 1000*1000)  # 最大地图尺寸

        self.known_paths: List[Dict[str, Any]] = []
        self.points_of_interest: Dict[str, Dict[str, Any]] = {}
        self.geofence: Optional[Dict[str, Any]] = None

        # 语义物体追踪（从 SemanticWorldModel 迁移）- 优化内存使用
        self.semantic_objects: Dict[str, SemanticObject] = {}
        self._object_counter = 0
        self.max_semantic_objects = self.config.get("max_semantic_objects", 500)

        # NEW: 因果图（三模态融合 - 因果地图模态）
        self.causal_graph = CausalGraph()
        self.state_history: List[Dict[str, Any]] = []
        self.last_state: Dict[str, Any] = {}

        # 探索边界管理（从 SemanticWorldModel 迁移）- 优化内存使用
        self.exploration_frontiers: List[ExplorationFrontier] = []
        self._frontier_counter = 0
        self.explored_positions: Set[Tuple[int, int]] = set()  # 栅格化坐标
        self.grid_resolution = self.config.get("grid_resolution", 0.5)  # 米
        self.max_frontiers = self.config.get("max_frontiers", 100)

        # 目标管理（从 SemanticWorldModel 迁移）
        self.current_target: Optional[SemanticObject] = None
        self.target_description: str = ""

        # 物体匹配参数（从 SemanticWorldModel 迁移）
        self._position_threshold = self.config.get("position_threshold", 2.0)  # 米
        self._label_similarity_threshold = 0.5

        # 场景历史（从 SemanticWorldModel 迁移）- 优化内存使用
        self._scene_history: List[Any] = []  # SceneDescription
        self._max_history = self.config.get("max_history", 50)

        # 位姿轨迹记录 - 优化内存使用
        self.pose_history: List[Dict[str, Any]] = []
        self.max_pose_history = self.config.get("max_pose_history", 1000)
        self.trajectory_start_time: Optional[datetime] = None

        # 变化检测 - 优化内存使用
        self.previous_state: Optional[Dict[str, Any]] = None
        self.pending_changes: List[EnvironmentChange] = []
        self.change_history: List[EnvironmentChange] = []
        self.max_history = self.config.get("max_change_history", 100)

        # 上一次更新时间
        self.last_update: datetime = datetime.now()

        # 内存管理配置
        self.enable_memory_management = self.config.get("enable_memory_management", True)
        self.cleanup_threshold = self.config.get("cleanup_threshold", 0.8)  # 80%内存使用率触发清理

        # 信念修正策略（Belief Revision Policy）
        self.belief_revision_policy = None
        if BELIEF_REVISION_AVAILABLE:
            belief_config = self.config.get("belief_revision", {})
            # 尝试从配置文件加载
            try:
                from brain.utils.config import load_config
                cognitive_config = load_config("modules/cognitive/belief_revision.yaml")
                if cognitive_config and "belief_revision" in cognitive_config:
                    belief_config.update(cognitive_config["belief_revision"])
            except Exception as e:
                logger.debug(f"无法加载信念修正配置文件: {e}")
            
            self.belief_revision_policy = BeliefRevisionPolicy(config=belief_config)
            logger.info("信念修正策略已启用")
        else:
            logger.warning("信念修正策略不可用")

        logger.info("WorldModel 初始化完成 (支持资源管理)" if RESOURCE_MANAGER_AVAILABLE else "WorldModel 初始化完成")
    
    def update_from_perception(
        self,
        perception_data: Union['PerceptionData', Dict[str, Any]]
    ) -> List[EnvironmentChange]:
        """
        从感知数据更新世界模型（优化：增量更新）

        Args:
            perception_data: PerceptionData 对象或传感器数据字典（向后兼容）

        Returns:
            List[EnvironmentChange]: 检测到的显著变化列表
        """
        # 快速检查是否有必要进行更新
        self._save_previous_state()

        # 优化：如果没有显著变化且距离上次更新时间很短，跳过处理
        if not self.has_significant_change() and (datetime.now() - self.last_update).total_seconds() < 0.1:
            logger.debug("WorldModel跳过更新：无显著变化")
            return []

        changes = []

        # 处理 PerceptionData 或 Dict 格式（向后兼容）
        if isinstance(perception_data, dict):
            # 旧格式：Dict[str, Any]
            return self._update_from_dict(perception_data)

        # 新格式：PerceptionData
        if not PERCEPTION_DATA_AVAILABLE:
            logger.error("PerceptionData not available, falling back to dict format")
            return []

        # 增量更新：只更新变化的部分
        if perception_data.pose:
            # 检查位姿是否有显著变化
            new_x, new_y, new_z = perception_data.pose.x, perception_data.pose.y, perception_data.pose.z
            old_x, old_y, old_z = self.robot_position.get("x", 0), self.robot_position.get("y", 0), self.robot_position.get("z", 0)
            pose_change = math.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)

            if pose_change > 0.5:  # 位姿变化超过0.5米才更新
                self.robot_position.update({"x": new_x, "y": new_y, "z": new_z})
                self.robot_heading = perception_data.pose.yaw

                # 记录位姿轨迹（限制频率）
                if len(self.pose_history) == 0 or pose_change > 2.0:
                    self._record_pose({
                        "x": new_x, "y": new_y, "z": new_z,
                        "yaw": perception_data.pose.yaw,
                        "velocity": perception_data.velocity.to_dict() if perception_data.velocity else {}
                    })

        # NEW: 保存当前状态用于因果检测
        current_state = {
            "timestamp": datetime.now(),
            "robot_position": self.robot_position.copy(),
            "objects": {
                obj_id: {
                    "position": obj.position.copy() if hasattr(obj, 'position') else {},
                    "state": obj.state if hasattr(obj, 'state') else None
                }
                for obj_id, obj in self.tracked_objects.items()
            }
        }

        # 地图更新（检查是否有变化）
        if perception_data.occupancy_grid is not None:
            if self.current_map is None or not np.array_equal(self.current_map, perception_data.occupancy_grid):
                self.current_map = perception_data.occupancy_grid
                self.map_resolution = perception_data.grid_resolution
                self.map_origin = perception_data.grid_origin
                logger.debug(f"地图更新: 分辨率={self.map_resolution}m, 大小={perception_data.occupancy_grid.shape}")

        # 从障碍物列表更新追踪物体（增量更新）
        if perception_data.obstacles:
            object_changes = self._update_tracked_objects_from_obstacles(perception_data.obstacles)
            changes.extend(object_changes)

        # 检测路径阻塞（仅在障碍物变化后）
        if object_changes:
            path_changes = self._check_path_blocking()
            changes.extend(path_changes)

        # 记录变化（限制变化数量）
        significant_changes = changes[:10]  # 限制每帧最多处理10个变化
        for change in significant_changes:
            self.pending_changes.append(change)
            self.change_history.append(change)

        # 限制历史长度
        if len(self.change_history) > self.max_history:
            self.change_history = self.change_history[-self.max_history:]

        self.last_update = datetime.now()

        # NEW: 检测因果关系（三模态融合 - 因果地图模态）
        if hasattr(self, 'last_state') and self.last_state:
            self._detect_and_update_causal_relations(
                self.last_state,
                current_state
            )

        # 保存当前状态作为下次的上一次状态
        self.last_state = current_state
        self.state_history.append(current_state)

        # 限制状态历史长度
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

        if changes:
            logger.info(f"WorldModel增量更新: 检测到 {len(changes)} 个变化")

        return significant_changes
    
    def _update_from_dict(self, sensor_data: Dict[str, Any]) -> List[EnvironmentChange]:
        """向后兼容：从字典格式更新（旧接口）"""
        changes = []
        
        # 更新机器人状态
        if "gps" in sensor_data:
            gps = sensor_data["gps"].get("data", {})
            self.robot_position.update({
                "lat": gps.get("latitude", self.robot_position["lat"]),
                "lon": gps.get("longitude", self.robot_position["lon"]),
                "alt": gps.get("altitude", self.robot_position["alt"])
            })
        
        if "imu" in sensor_data:
            imu = sensor_data["imu"].get("data", {})
            if "orientation" in imu:
                self.robot_heading = imu["orientation"].get("yaw", self.robot_heading)
        
        # 记录位姿轨迹
        if "pose" in sensor_data:
            pose_data = sensor_data["pose"]
            self._record_pose(pose_data)
        
        # 更新电池状态
        if "battery" in sensor_data:
            old_battery = self.battery_level
            self.battery_level = sensor_data["battery"]
            
            # 检测电池低电量
            if self.battery_level < 20 and old_battery >= 20:
                changes.append(self._create_change(
                    ChangeType.BATTERY_LOW,
                    f"电池电量低: {self.battery_level:.1f}%",
                    {"level": self.battery_level}
                ))
        
        # 更新检测到的物体
        if "detections" in sensor_data:
            object_changes = self._update_tracked_objects(sensor_data["detections"])
            changes.extend(object_changes)
        
        # 更新天气
        if "weather" in sensor_data:
            weather_changes = self._update_weather(sensor_data["weather"])
            changes.extend(weather_changes)
        
        return changes
    
    def _update_tracked_objects_from_obstacles(
        self, 
        obstacles: List[Dict[str, Any]]
    ) -> List[EnvironmentChange]:
        """从障碍物列表更新追踪物体"""
        changes = []
        current_ids = set()
        
        for obs in obstacles:
            obj_id = obs.get("id", f"obs_{len(self.tracked_objects)}")
            current_ids.add(obj_id)
            
            position = obs.get("world_position", obs.get("local_position", {}))
            if not position:
                continue
            
            if obj_id in self.tracked_objects:
                # 更新现有物体
                obj = self.tracked_objects[obj_id]
                old_position = obj.position.copy()
                
                # 计算移动距离
                distance_moved = math.sqrt(
                    (position.get("x", 0) - old_position.get("x", 0)) ** 2 +
                    (position.get("y", 0) - old_position.get("y", 0)) ** 2
                )
                
                # 更新位置
                obj.position = position
                obj.position_history.append(position)
                obj.last_seen = datetime.now()
                
                # 如果移动显著
                if distance_moved > 2.0:
                    changes.append(self._create_change(
                        ChangeType.OBSTACLE_MOVED,
                        f"障碍物 {obj_id} 移动了 {distance_moved:.1f}m",
                        {"object_id": obj_id, "old_position": old_position, "new_position": position}
                    ))
            else:
                # 新障碍物
                self.tracked_objects[obj_id] = TrackedObject(
                    id=obj_id,
                    object_type=obs.get("type", "unknown"),
                    position=position,
                    confidence=obs.get("confidence", 0.5),
                    is_obstacle=True
                )
                
                changes.append(self._create_change(
                    ChangeType.NEW_OBSTACLE,
                    f"发现新障碍物: {obs.get('type', 'unknown')}",
                    {"object_id": obj_id, "type": obs.get("type"), "position": position}
                ))
        
        # 检测消失的物体
        disappeared = set(self.tracked_objects.keys()) - current_ids
        for obj_id in disappeared:
            obj = self.tracked_objects[obj_id]
            if (datetime.now() - obj.last_seen).total_seconds() > 10:
                changes.append(self._create_change(
                    ChangeType.OBSTACLE_REMOVED,
                    f"障碍物消失: {obj_id}",
                    {"object_id": obj_id, "last_position": obj.position}
                ))
                del self.tracked_objects[obj_id]
        
        return changes
    
    def _save_previous_state(self):
        """保存当前状态用于变化检测（优化：仅保存关键状态）"""
        # 优化：只保存关键状态哈希值，减少内存开销
        self.previous_state = {
            "robot_position_hash": self._hash_position(self.robot_position),
            "tracked_objects_hash": self._hash_tracked_objects(),
            "weather_hash": self._hash_weather(),
            "battery_level": self.battery_level,
            "last_check_time": datetime.now()
        }

    def _hash_position(self, position: Dict[str, float]) -> int:
        """计算位置哈希值"""
        # 使用网格化位置减少微小变化的检测
        grid_size = 1.0  # 1米网格
        grid_x = int(position.get("x", 0) / grid_size)
        grid_y = int(position.get("y", 0) / grid_size)
        grid_z = int(position.get("z", 0) / grid_size)
        return hash((grid_x, grid_y, grid_z))

    def _hash_tracked_objects(self) -> int:
        """计算追踪对象的哈希值"""
        # 只计算关键对象的状态哈希
        key_objects = []
        for obj_id, obj in list(self.tracked_objects.items())[:50]:  # 限制对象数量
            key_objects.append((
                obj_id,
                self._hash_position(obj.position),
                obj.is_obstacle,
                obj.is_target
            ))
        return hash(tuple(key_objects))

    def _hash_weather(self) -> int:
        """计算天气状态哈希值"""
        key_weather = (
            self.weather.get("condition", "clear"),
            int(self.weather.get("wind_speed", 0)),
            int(self.weather.get("visibility", "good") != "good")
        )
        return hash(key_weather)

    def has_significant_change(self) -> bool:
        """快速检查是否有显著变化"""
        if not self.previous_state:
            return True

        current_pos_hash = self._hash_position(self.robot_position)
        current_objects_hash = self._hash_tracked_objects()
        current_weather_hash = self._hash_weather()

        # 检查关键状态变化
        position_changed = current_pos_hash != self.previous_state.get("robot_position_hash")
        objects_changed = current_objects_hash != self.previous_state.get("tracked_objects_hash")
        weather_changed = current_weather_hash != self.previous_state.get("weather_hash")
        battery_critical = self.battery_level < 20 and self.previous_state.get("battery_level", 100) >= 20

        return position_changed or objects_changed or weather_changed or battery_critical
    
    def _update_tracked_objects(self, detections: List[Dict[str, Any]]) -> List[EnvironmentChange]:
        """更新跟踪的物体"""
        changes = []
        current_ids = set()
        
        for detection in detections:
            obj_id = detection.get("id", f"obj_{len(self.tracked_objects)}")
            current_ids.add(obj_id)
            
            position = {
                "x": detection.get("x", 0),
                "y": detection.get("y", 0),
                "z": detection.get("z", 0)
            }
            
            if obj_id in self.tracked_objects:
                # 更新现有物体
                obj = self.tracked_objects[obj_id]
                old_position = obj.position.copy()
                
                # 计算移动距离
                distance_moved = math.sqrt(
                    (position["x"] - old_position["x"]) ** 2 +
                    (position["y"] - old_position["y"]) ** 2 +
                    (position["z"] - old_position["z"]) ** 2
                )
                
                # 更新位置
                obj.position = position
                obj.position_history.append(position)
                obj.last_seen = datetime.now()
                
                # 如果移动显著
                if distance_moved > 2.0:  # 移动超过2米
                    if obj.is_target:
                        changes.append(self._create_change(
                            ChangeType.TARGET_MOVED,
                            f"目标 {obj_id} 移动了 {distance_moved:.1f}m",
                            {"object_id": obj_id, "old_position": old_position, "new_position": position}
                        ))
                    elif obj.is_obstacle:
                        changes.append(self._create_change(
                            ChangeType.OBSTACLE_MOVED,
                            f"障碍物 {obj_id} 移动了 {distance_moved:.1f}m",
                            {"object_id": obj_id, "old_position": old_position, "new_position": position}
                        ))
            else:
                # 新物体
                obj_type = detection.get("type", "unknown")
                is_obstacle = obj_type in ["obstacle", "person", "vehicle", "building", "tree"]
                is_target = detection.get("is_target", False)
                
                self.tracked_objects[obj_id] = TrackedObject(
                    id=obj_id,
                    object_type=obj_type,
                    position=position,
                    confidence=detection.get("confidence", 0.5),
                    is_obstacle=is_obstacle,
                    is_target=is_target
                )
                
                if is_obstacle:
                    changes.append(self._create_change(
                        ChangeType.NEW_OBSTACLE,
                        f"发现新障碍物: {obj_type}",
                        {"object_id": obj_id, "type": obj_type, "position": position}
                    ))
                elif is_target:
                    changes.append(self._create_change(
                        ChangeType.TARGET_APPEARED,
                        f"发现目标: {obj_type}",
                        {"object_id": obj_id, "type": obj_type, "position": position}
                    ))
        
        # 检测消失的物体
        disappeared = set(self.tracked_objects.keys()) - current_ids
        for obj_id in disappeared:
            obj = self.tracked_objects[obj_id]
            # 如果超过10秒没有看到
            if (datetime.now() - obj.last_seen).total_seconds() > 10:
                if obj.is_target:
                    changes.append(self._create_change(
                        ChangeType.TARGET_LOST,
                        f"目标丢失: {obj_id}",
                        {"object_id": obj_id, "last_position": obj.position}
                    ))
                elif obj.is_obstacle:
                    changes.append(self._create_change(
                        ChangeType.OBSTACLE_REMOVED,
                        f"障碍物消失: {obj_id}",
                        {"object_id": obj_id, "last_position": obj.position}
                    ))
                del self.tracked_objects[obj_id]
        
        return changes
    
    # 删除 _update_occupancy_grid 方法，不再自己维护 occupied_cells
    # 地图由 OccupancyMapper 生成，通过 PerceptionData 传递
    
    def _update_weather(self, weather_data: Dict[str, Any]) -> List[EnvironmentChange]:
        """更新天气信息"""
        changes = []
        
        old_condition = self.weather.get("condition", "clear")
        new_condition = weather_data.get("condition", "clear")
        
        self.weather.update(weather_data)
        
        # 天气显著变化
        severe_conditions = ["storm", "heavy_rain", "fog", "strong_wind"]
        if new_condition in severe_conditions and old_condition not in severe_conditions:
            changes.append(self._create_change(
                ChangeType.WEATHER_CHANGED,
                f"天气恶化: {new_condition}",
                {"old": old_condition, "new": new_condition}
            ))
        
        return changes
    
    def _check_path_blocking(self) -> List[EnvironmentChange]:
        """检测当前路径是否被阻塞"""
        changes = []
        
        # 方法1：检查追踪的物体
        for obj_id, obj in self.tracked_objects.items():
            if obj.is_obstacle:
                # 计算到障碍物的距离
                dx = obj.position.get("x", 0) - self.robot_position.get("x", 0)
                dy = obj.position.get("y", 0) - self.robot_position.get("y", 0)
                distance = math.sqrt(dx * dx + dy * dy)
                
                # 计算是否在前进方向
                angle_to_obj = math.atan2(dy, dx) * 180 / math.pi
                angle_diff = abs(angle_to_obj - self.robot_heading)
                
                # 如果在前进方向30度内且距离小于10米
                if angle_diff < 30 and distance < 10:
                    changes.append(self._create_change(
                        ChangeType.PATH_BLOCKED,
                        f"前方路径被阻塞，距离 {distance:.1f}m",
                        {"obstacle_id": obj_id, "distance": distance}
                    ))
        
        # 方法2：使用占据地图检查（如果可用）
        if self.current_map is not None:
            # 检查前方路径上的占据情况
            robot_x = self.robot_position.get("x", 0)
            robot_y = self.robot_position.get("y", 0)
            
            # 检查前方5米内的路径
            for dist in [1.0, 2.0, 3.0, 4.0, 5.0]:
                check_x = robot_x + dist * math.cos(math.radians(self.robot_heading))
                check_y = robot_y + dist * math.sin(math.radians(self.robot_heading))
                
                if self.is_occupied_at(check_x, check_y):
                    changes.append(self._create_change(
                        ChangeType.PATH_BLOCKED,
                        f"前方路径被阻塞（地图检测），距离 {dist:.1f}m",
                        {"distance": dist, "position": {"x": check_x, "y": check_y}}
                    ))
                    break
        
        return changes
    
    def _create_change(
        self,
        change_type: ChangeType,
        description: str,
        data: Dict[str, Any]
    ) -> EnvironmentChange:
        """创建环境变化记录"""
        config = self.CHANGE_CONFIG.get(change_type, {
            "priority": ChangePriority.INFO,
            "threshold": 0.5,
            "requires_replan": False,
            "requires_confirmation": False
        })
        
        return EnvironmentChange(
            change_type=change_type,
            priority=config["priority"],
            description=description,
            data=data,
            requires_replan=config["requires_replan"],
            requires_confirmation=config["requires_confirmation"]
        )
    
    def detect_significant_changes(self) -> List[EnvironmentChange]:
        """
        检测需要触发重规划的显著变化
        
        Returns:
            需要处理的变化列表
        """
        significant = [c for c in self.pending_changes if c.requires_replan]
        self.pending_changes = [c for c in self.pending_changes if not c.requires_replan]
        return significant
    
    def get_pending_confirmations(self) -> List[EnvironmentChange]:
        """获取需要用户确认的变化"""
        confirmations = [c for c in self.pending_changes if c.requires_confirmation]
        return confirmations
    
    def acknowledge_change(self, change: EnvironmentChange):
        """确认变化已处理"""
        if change in self.pending_changes:
            self.pending_changes.remove(change)
    
    def get_context_for_planning(self) -> PlanningContext:
        """
        获取规划上下文
        
        Returns:
            PlanningContext: 包含完整环境信息的上下文
        """
        # 收集障碍物信息（从追踪物体和语义物体）
        obstacles = []
        
        # 从追踪物体收集
        for obj_id, obj in self.tracked_objects.items():
            if obj.is_obstacle:
                distance = self._calculate_distance(obj.position)
                direction = self._calculate_direction(obj.position)
                obstacles.append({
                    "id": obj_id,
                    "type": obj.object_type,
                    "position": obj.position,
                    "distance": distance,
                    "direction": direction,
                    "confidence": obj.confidence
                })
        
        # 从语义物体收集障碍物
        for obj_id, obj in self.semantic_objects.items():
            if obj.is_valid() and obj.target_type == "obstacle":
                # 计算相对位置
                obj_pos_dict = {"x": obj.world_position[0], "y": obj.world_position[1], "z": 0}
                distance = self._calculate_distance(obj_pos_dict)
                direction = self._calculate_direction(obj_pos_dict)
                obstacles.append({
                    "id": obj_id,
                    "type": obj.label,
                    "position": obj_pos_dict,
                    "distance": distance,
                    "direction": direction,
                    "confidence": obj.confidence,
                    "semantic": True
                })
        
        # 收集目标信息
        targets = []
        for obj_id, obj in self.tracked_objects.items():
            if obj.is_target:
                distance = self._calculate_distance(obj.position)
                targets.append({
                    "id": obj_id,
                    "type": obj.object_type,
                    "position": obj.position,
                    "distance": distance,
                    "confidence": obj.confidence
                })
        
        # 从语义物体收集目标
        if self.current_target:
            target_pos_dict = {"x": self.current_target.world_position[0], 
                              "y": self.current_target.world_position[1], "z": 0}
            distance = self._calculate_distance(target_pos_dict)
            targets.append({
                "id": self.current_target.id,
                "type": self.current_target.label,
                "position": target_pos_dict,
                "distance": distance,
                "confidence": self.current_target.confidence,
                "semantic": True,
                "description": self.target_description
            })
        
        # 收集兴趣点
        pois = list(self.points_of_interest.values())
        
        # 构建约束列表
        constraints = []
        if self.battery_level < 30:
            constraints.append(f"电池电量低({self.battery_level:.0f}%)，优先考虑返航")
        if self.weather.get("wind_speed", 0) > 10:
            constraints.append(f"风速较大({self.weather['wind_speed']:.0f}m/s)，注意飞行稳定性")
        if self.geofence:
            constraints.append("注意地理围栏边界")
        
        # 最近变化
        recent_changes = [
            c.to_dict() for c in self.change_history[-5:]
        ]

        # NEW: 收集语义物体（三模态融合 - 语义地图）
        semantic_objects_list = []
        for obj_id, obj in self.semantic_objects.items():
            if obj.is_valid() and obj.state != ObjectState.LOST:
                semantic_objects_list.append({
                    "id": obj_id,
                    "label": obj.label,
                    "position": obj.world_position,
                    "confidence": obj.confidence,
                    "state": obj.state.value,
                    "attributes": obj.attributes
                })

        # NEW: 收集因果图数据（三模态融合 - 因果地图）
        causal_graph_data = {}
        if hasattr(self, 'causal_graph'):
            # 收集因果关系摘要
            relations = []
            for (cause, effect), edge in self.causal_graph.edges.items():
                if edge.confidence > 0.6:
                    cause_node = self.causal_graph.nodes.get(cause)
                    effect_node = self.causal_graph.nodes.get(effect)
                    cause_label = cause_node.label if cause_node else cause
                    effect_label = effect_node.label if effect_node else effect
                    relations.append(f"{cause_label} → {effect_label}")

            causal_graph_data = {
                "relations": relations,
                "num_edges": len(self.causal_graph.edges),
                "num_nodes": len(self.causal_graph.nodes)
            }

        # NEW: 收集状态预测
        state_predictions = []
        if hasattr(self, 'causal_graph') and hasattr(self, 'predict_future_state'):
            prediction_result = self.predict_future_state(horizon=5.0)
            state_predictions = prediction_result.get("predictions", [])

        return PlanningContext(
            current_position=self.robot_position,
            current_heading=self.robot_heading,
            obstacles=obstacles,
            targets=targets,
            points_of_interest=pois,
            weather=self.weather,
            battery_level=self.battery_level,
            signal_strength=self.signal_strength,
            available_paths=self.known_paths,
            constraints=constraints,
            recent_changes=recent_changes,
            risk_areas=[],  # TODO: 实现风险区域计算
            # NEW: 三模态融合字段
            semantic_objects=semantic_objects_list,
            causal_graph=causal_graph_data,
            state_predictions=state_predictions
        )
    
    def _calculate_distance(self, position: Dict[str, float]) -> float:
        """计算到指定位置的距离"""
        dx = position.get("x", 0) - self.robot_position.get("x", 0)
        dy = position.get("y", 0) - self.robot_position.get("y", 0)
        dz = position.get("z", 0) - self.robot_position.get("z", 0)
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    
    def _calculate_direction(self, position: Dict[str, float]) -> str:
        """计算方向描述"""
        dx = position.get("x", 0) - self.robot_position.get("x", 0)
        dy = position.get("y", 0) - self.robot_position.get("y", 0)
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        
        if -22.5 <= angle < 22.5:
            return "东"
        elif 22.5 <= angle < 67.5:
            return "东北"
        elif 67.5 <= angle < 112.5:
            return "北"
        elif 112.5 <= angle < 157.5:
            return "西北"
        elif angle >= 157.5 or angle < -157.5:
            return "西"
        elif -157.5 <= angle < -112.5:
            return "西南"
        elif -112.5 <= angle < -67.5:
            return "南"
        else:
            return "东南"
    
    def add_point_of_interest(
        self,
        poi_id: str,
        position: Dict[str, float],
        poi_type: str,
        description: str = ""
    ):
        """添加兴趣点"""
        self.points_of_interest[poi_id] = {
            "id": poi_id,
            "position": position,
            "type": poi_type,
            "description": description,
            "added_at": datetime.now().isoformat()
        }
    
    def set_target(self, obj_id: str, is_target: bool = True):
        """标记物体为目标"""
        if obj_id in self.tracked_objects:
            self.tracked_objects[obj_id].is_target = is_target
    
    def _record_pose(self, pose_data: Dict[str, Any]):
        """记录位姿到轨迹历史"""
        if self.trajectory_start_time is None:
            self.trajectory_start_time = datetime.now()
        
        pose_record = {
            "timestamp": datetime.now(),
            "x": pose_data.get("x", self.robot_position.get("x", 0)),
            "y": pose_data.get("y", self.robot_position.get("y", 0)),
            "z": pose_data.get("z", self.robot_position.get("z", 0)),
            "yaw": pose_data.get("yaw", self.robot_heading),
            "velocity": pose_data.get("velocity", {}),
            "elapsed_time": (datetime.now() - self.trajectory_start_time).total_seconds()
        }
        
        self.pose_history.append(pose_record)
        
        # 限制历史长度
        if len(self.pose_history) > self.max_pose_history:
            self.pose_history.pop(0)
    
    def get_pose_history(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取位姿轨迹历史"""
        if count is None:
            return self.pose_history.copy()
        return self.pose_history[-count:] if count > 0 else []
    
    def get_trajectory_distance(self) -> float:
        """计算轨迹总距离"""
        if len(self.pose_history) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.pose_history)):
            p1 = self.pose_history[i-1]
            p2 = self.pose_history[i]
            dx = p2["x"] - p1["x"]
            dy = p2["y"] - p1["y"]
            total += math.sqrt(dx**2 + dy**2)
        
        return total

    # ============ 三模态融合 - 因果地图方法 ============

    def _detect_and_update_causal_relations(
        self,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any]
    ):
        """检测和更新因果关系（三模态融合 - 因果地图模态）

        分析状态变化，检测可能的因果关系并更新因果图。

        Args:
            old_state: 上一次的状态
            new_state: 当前状态
        """
        old_objects = old_state.get("objects", {})
        new_objects = new_state.get("objects", {})

        # 检测新物体出现
        for obj_id in new_objects:
            if obj_id not in old_objects:
                # 新物体出现
                self.causal_graph.add_or_update_node(
                    node_id=obj_id,
                    node_type="object",
                    label=f"object_{obj_id}",
                    state=new_objects[obj_id]
                )
                # 推测：机器人探索导致了发现
                self.causal_graph.add_causal_relation(
                    cause="robot",
                    effect=obj_id,
                    relation_type=CausalRelationType.FUNCTIONAL,
                    confidence=0.6,
                    metadata={"reason": "object_discovery"}
                )

        # 检测物体移动
        for obj_id in new_objects:
            if obj_id not in old_objects:
                continue

            old_pos = old_objects[obj_id].get("position", {})
            new_pos = new_objects[obj_id].get("position", {})

            # 计算移动距离
            dx = new_pos.get("x", 0) - old_pos.get("x", 0)
            dy = new_pos.get("y", 0) - old_pos.get("y", 0)
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > 2.0:  # 显著移动
                # 推测：机器人交互导致移动
                self.causal_graph.add_causal_relation(
                    cause="robot",
                    effect=obj_id,
                    relation_type=CausalRelationType.FUNCTIONAL,
                    confidence=0.7,
                    metadata={"reason": "object_movement", "distance": distance}
                )

        # 检测状态变化
        for obj_id in new_objects:
            if obj_id not in old_objects:
                continue

            old_obj_state = old_objects[obj_id].get("state")
            new_obj_state = new_objects[obj_id].get("state")

            if old_obj_state and new_obj_state and old_obj_state != new_obj_state:
                # 状态发生变化
                self.causal_graph.add_causal_relation(
                    cause="robot",
                    effect=obj_id,
                    relation_type=CausalRelationType.STATE_CHANGE,
                    confidence=0.6,
                    metadata={
                        "reason": "state_change",
                        "old_state": str(old_obj_state),
                        "new_state": str(new_obj_state)
                    }
                )

    def predict_future_state(
        self,
        horizon: float = 5.0
    ) -> Dict[str, Any]:
        """预测未来状态（三模态融合 - 因果推理）

        基于因果图预测可能的未来状态变化。

        Args:
            horizon: 预测时间范围（秒）

        Returns:
            预测结果 {"predictions": List[Dict]}
        """
        predictions = []

        # 使用因果图预测
        for node_id, node in self.causal_graph.nodes.items():
            if not node.active:
                continue

            effects = self.causal_graph.predict_effects(node_id)

            for effect_id, confidence in effects:
                if confidence > 0.7:
                    effect_node = self.causal_graph.nodes.get(effect_id)
                    effect_label = effect_node.label if effect_node else effect_id

                    predictions.append({
                        "cause": node_id,
                        "cause_label": node.label,
                        "effect": effect_id,
                        "effect_label": effect_label,
                        "confidence": confidence,
                        "expected_time": horizon,
                        "description": f"{node.label} 可能导致 {effect_label} "
                                      f"(置信度{confidence:.0%})"
                    })

        return {"predictions": predictions}

    def get_causal_graph_statistics(self) -> Dict[str, Any]:
        """获取因果图统计信息"""
        return self.causal_graph.get_statistics()

    def explain_object_state(self, obj_id: str) -> Optional[str]:
        """解释物体当前状态的原因

        Args:
            obj_id: 物体ID

        Returns:
            原因描述字符串
        """
        return self.causal_graph.explain_state_change(obj_id)

    # ============ 原有方法 ============

    def get_summary(self) -> Dict[str, Any]:
        """获取世界模型摘要"""
        return {
            "robot_position": self.robot_position,
            "robot_heading": self.robot_heading,
            "battery_level": self.battery_level,
            "obstacles_count": sum(1 for o in self.tracked_objects.values() if o.is_obstacle),
            "targets_count": sum(1 for o in self.tracked_objects.values() if o.is_target),
            "semantic_objects_count": len(self.semantic_objects),
            "pending_changes": len(self.pending_changes),
            "weather": self.weather.get("condition", "unknown"),
            "last_update": self.last_update.isoformat(),
            "trajectory_length": len(self.pose_history),
            "trajectory_distance": self.get_trajectory_distance(),
            "map_available": self.current_map is not None
        }
    
    # ========== 地图查询方法（新增） ==========
    
    def get_occupancy_at(self, x: float, y: float) -> int:
        """
        查询位置的占用状态
        
        Args:
            x, y: 世界坐标
            
        Returns:
            -1: 未知, 0: 自由, 100: 占据
        """
        if self.current_map is None:
            return -1
        
        # 转换到栅格坐标
        gx = int((x - self.map_origin[0]) / self.map_resolution)
        gy = int((y - self.map_origin[1]) / self.map_resolution)
        
        # 检查边界
        if (0 <= gx < self.current_map.shape[1] and 
            0 <= gy < self.current_map.shape[0]):
            return int(self.current_map[gy, gx])
        
        return -1
    
    def is_occupied_at(self, x: float, y: float) -> bool:
        """检查位置是否被占据"""
        return self.get_occupancy_at(x, y) == 100
    
    def is_free_at(self, x: float, y: float) -> bool:
        """检查位置是否自由"""
        return self.get_occupancy_at(x, y) == 0
    
    def get_nearest_obstacle(
        self,
        x: float,
        y: float,
        max_range: float = 5.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        获取最近的障碍物（从地图提取）
        
        Returns:
            (obstacle_x, obstacle_y, distance) 或 None
        """
        if self.current_map is None:
            return None
        
        # 转换到栅格坐标
        gx = int((x - self.map_origin[0]) / self.map_resolution)
        gy = int((y - self.map_origin[1]) / self.map_resolution)
        max_grid_range = int(max_range / self.map_resolution)
        
        min_dist = float('inf')
        nearest = None
        
        # 搜索周围栅格
        for dy in range(-max_grid_range, max_grid_range + 1):
            for dx in range(-max_grid_range, max_grid_range + 1):
                check_gx = gx + dx
                check_gy = gy + dy
                
                if (0 <= check_gx < self.current_map.shape[1] and 
                    0 <= check_gy < self.current_map.shape[0]):
                    
                    if self.current_map[check_gy, check_gx] == 100:  # 占据
                        # 转换回世界坐标
                        obs_x = check_gx * self.map_resolution + self.map_origin[0]
                        obs_y = check_gy * self.map_resolution + self.map_origin[1]
                        dist = math.sqrt((obs_x - x)**2 + (obs_y - y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            nearest = (obs_x, obs_y, dist)
        
        return nearest
    
    # ========== 语义物体追踪方法（从 SemanticWorldModel 迁移） ==========
    
    def update_from_vlm(
        self,
        scene: 'SceneDescription',
        robot_pose: Tuple[float, float, float] = None
    ):
        """
        从VLM结果更新世界模型
        
        Args:
            scene: VLM场景描述
            robot_pose: 机器人位姿 (x, y, yaw)
        """
        if not VLM_AVAILABLE:
            logger.warning("VLM not available, skipping update_from_vlm")
            return
        
        if robot_pose is None:
            robot_pose = (
                self.robot_position.get("x", 0),
                self.robot_position.get("y", 0),
                self.robot_heading
            )
        
        robot_x, robot_y, robot_yaw = robot_pose
        
        # 更新场景历史
        self._scene_history.append(scene)
        if len(self._scene_history) > self._max_history:
            self._scene_history.pop(0)
        
        # 处理检测到的物体
        for detected in scene.objects:
            # 计算世界坐标
            world_pos = self._compute_world_position(
                detected, robot_x, robot_y, robot_yaw
            )
            
            # 尝试匹配已有物体
            matched_obj = self._find_matching_object(detected, world_pos)
            
            if matched_obj:
                # 更新已有物体
                matched_obj.update_observation(detected.confidence, world_pos)
                matched_obj.description = detected.description
                matched_obj.bbox = detected.bbox
            else:
                # 创建新物体
                new_obj = self._create_semantic_object(detected, world_pos)
                self.semantic_objects[new_obj.id] = new_obj
        
        # 更新探索状态
        self._update_exploration(robot_x, robot_y)
        
        # 清理过期物体
        self._cleanup_stale_objects()
        
        logger.debug(f"世界模型更新: {len(self.semantic_objects)} 个语义物体")
    
    def _compute_world_position(
        self,
        detected: 'DetectedObject',
        robot_x: float,
        robot_y: float,
        robot_yaw: float
    ) -> Tuple[float, float]:
        """计算物体的世界坐标"""
        # 从位置描述估计相对位置
        local_x, local_y = self._estimate_local_position(detected)
        
        # 如果有估计距离，使用更准确的计算
        if detected.estimated_distance:
            dist = detected.estimated_distance
            # 使用边界框中心估计方向
            if detected.bbox:
                # 图像x轴映射到方向
                angle_offset = (detected.bbox.x - 0.5) * math.pi / 2  # ±45度
                local_x = dist * math.cos(angle_offset)
                local_y = dist * math.sin(angle_offset)
        
        # 转换到世界坐标
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        
        world_x = robot_x + local_x * cos_yaw - local_y * sin_yaw
        world_y = robot_y + local_x * sin_yaw + local_y * cos_yaw
        
        return (world_x, world_y)
    
    def _estimate_local_position(
        self,
        detected: 'DetectedObject'
    ) -> Tuple[float, float]:
        """从检测结果估计相对位置"""
        # 默认距离
        default_distance = 5.0
        
        position = detected.position_description.lower()
        
        # 估计距离
        distance = default_distance
        if "近" in position or "close" in position:
            distance = 2.0
        elif "远" in position or "far" in position:
            distance = 10.0
        
        # 估计方向
        angle = 0.0  # 正前方
        if "左" in position or "left" in position:
            angle = math.pi / 4  # 45度
            if "大" in position:
                angle = math.pi / 2  # 90度
        elif "右" in position or "right" in position:
            angle = -math.pi / 4
            if "大" in position:
                angle = -math.pi / 2
        
        local_x = distance * math.cos(angle)
        local_y = distance * math.sin(angle)
        
        return (local_x, local_y)
    
    def _find_matching_object(
        self,
        detected: 'DetectedObject',
        world_pos: Tuple[float, float]
    ) -> Optional[SemanticObject]:
        """查找匹配的已有物体"""
        best_match = None
        best_score = 0.0
        
        for obj in self.semantic_objects.values():
            if not obj.is_valid():
                continue
            
            # 计算匹配分数
            score = self._compute_match_score(obj, detected, world_pos)
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = obj
        
        return best_match
    
    def _compute_match_score(
        self,
        obj: SemanticObject,
        detected: 'DetectedObject',
        world_pos: Tuple[float, float]
    ) -> float:
        """计算物体匹配分数"""
        # 标签相似度
        label_score = 0.0
        if obj.label.lower() == detected.label.lower():
            label_score = 1.0
        elif obj.label.lower() in detected.label.lower() or \
             detected.label.lower() in obj.label.lower():
            label_score = 0.7
        
        # 位置距离
        dist = math.sqrt(
            (obj.world_position[0] - world_pos[0]) ** 2 +
            (obj.world_position[1] - world_pos[1]) ** 2
        )
        position_score = max(0, 1 - dist / self._position_threshold)
        
        # 综合分数
        return label_score * 0.6 + position_score * 0.4
    
    def _create_semantic_object(
        self,
        detected: 'DetectedObject',
        world_pos: Tuple[float, float]
    ) -> SemanticObject:
        """创建语义物体"""
        self._object_counter += 1
        
        obj = SemanticObject(
            id=f"sem_{self._object_counter}",
            label=detected.label,
            world_position=world_pos,
            bbox=detected.bbox,
            confidence=detected.confidence,
            description=detected.description
        )
        
        # 判断是否为导航相关物体
        label_lower = detected.label.lower()
        if any(nav_type in label_lower for nav_type in self.NAVIGATION_OBJECTS):
            obj.target_type = "navigation"
        elif any(obs_type in label_lower for obs_type in self.OBSTACLE_TYPES):
            obj.target_type = "obstacle"
        
        return obj
    
    def _update_exploration(self, robot_x: float, robot_y: float):
        """更新探索状态"""
        # 标记当前位置为已探索
        grid_x = int(robot_x / self.grid_resolution)
        grid_y = int(robot_y / self.grid_resolution)
        
        # 探索周围区域
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.explored_positions.add((grid_x + dx, grid_y + dy))
        
        # 更新探索边界
        self._update_frontiers(robot_x, robot_y)
    
    def _update_frontiers(self, robot_x: float, robot_y: float):
        """更新探索边界"""
        # 移除已探索的边界
        self.exploration_frontiers = [
            f for f in self.exploration_frontiers
            if not self._is_explored(f.position)
        ]
        
        # 生成新边界（基于当前位置周围未探索区域）
        for angle in [0, math.pi/4, math.pi/2, 3*math.pi/4, 
                      math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4]:
            dist = 5.0  # 5米外
            fx = robot_x + dist * math.cos(angle)
            fy = robot_y + dist * math.sin(angle)
            
            if not self._is_explored((fx, fy)):
                # 检查是否已有类似边界
                has_similar = any(
                    math.sqrt((f.position[0]-fx)**2 + (f.position[1]-fy)**2) < 2.0
                    for f in self.exploration_frontiers
                )
                
                if not has_similar:
                    self._frontier_counter += 1
                    frontier = ExplorationFrontier(
                        id=f"frontier_{self._frontier_counter}",
                        position=(fx, fy),
                        direction=angle,
                        priority=self._compute_frontier_priority(fx, fy)
                    )
                    self.exploration_frontiers.append(frontier)
    
    def _is_explored(self, position: Tuple[float, float]) -> bool:
        """检查位置是否已探索"""
        grid_x = int(position[0] / self.grid_resolution)
        grid_y = int(position[1] / self.grid_resolution)
        return (grid_x, grid_y) in self.explored_positions
    
    def _compute_frontier_priority(self, fx: float, fy: float) -> float:
        """计算探索边界优先级"""
        priority = 0.5
        
        # 如果有目标，优先朝目标方向探索
        if self.current_target:
            target_pos = self.current_target.world_position
            dist_to_target = math.sqrt(
                (fx - target_pos[0])**2 + (fy - target_pos[1])**2
            )
            # 越接近目标方向，优先级越高
            priority = max(0, 1 - dist_to_target / 20.0)
        
        # 附近有导航物体则提高优先级
        for obj in self.semantic_objects.values():
            if obj.target_type == "navigation":
                dist = math.sqrt(
                    (fx - obj.world_position[0])**2 + 
                    (fy - obj.world_position[1])**2
                )
                if dist < 5.0:
                    priority = min(1.0, priority + 0.2)
        
        return priority
    
    def _cleanup_stale_objects(self, max_age: float = 120.0):
        """清理过期物体"""
        now = datetime.now()
        stale_ids = []
        
        for obj_id, obj in self.semantic_objects.items():
            age = (now - obj.last_seen).total_seconds()
            if age > max_age and obj.state != ObjectState.CONFIRMED:
                stale_ids.append(obj_id)
            elif age > 30.0 and obj.state == ObjectState.DETECTED:
                obj.mark_lost()
        
        for obj_id in stale_ids:
            del self.semantic_objects[obj_id]
    
    def find_semantic_target(
        self,
        description: str
    ) -> Optional[SemanticObject]:
        """
        根据描述查找语义目标
        
        Args:
            description: 目标描述（如"建筑的门"）
        """
        description_lower = description.lower()
        best_match = None
        best_score = 0.0
        
        for obj in self.semantic_objects.values():
            if not obj.is_valid():
                continue
            
            # 计算匹配分数
            score = self._compute_description_match(obj, description_lower)
            
            if score > best_score:
                best_score = score
                best_match = obj
        
        if best_match and best_score > 0.3:
            best_match.is_target = True
            self.current_target = best_match
            self.target_description = description
            return best_match
        
        return None
    
    def _compute_description_match(
        self,
        obj: SemanticObject,
        description: str
    ) -> float:
        """计算物体与描述的匹配度"""
        score = 0.0
        
        # 标签匹配
        if obj.label.lower() in description:
            score += 0.5
        
        # 描述关键词匹配
        keywords = ["门", "door", "建筑", "building", "入口", "entrance"]
        for kw in keywords:
            if kw in description and kw in obj.label.lower():
                score += 0.3
            if kw in description and kw in obj.description.lower():
                score += 0.2
        
        # 确认状态加分
        if obj.state == ObjectState.CONFIRMED:
            score *= 1.2
        
        # 观测次数加分
        score *= min(1.5, 1 + obj.observation_count * 0.1)
        
        return min(1.0, score)
    
    def get_exploration_target(self) -> Optional[Tuple[float, float]]:
        """获取下一个探索目标"""
        if not self.exploration_frontiers:
            return None
        
        # 按优先级排序
        sorted_frontiers = sorted(
            self.exploration_frontiers,
            key=lambda f: f.priority,
            reverse=True
        )
        
        for frontier in sorted_frontiers:
            if frontier.reachable and not frontier.explored:
                return frontier.position
        
        return None
    
    def get_objects_by_type(self, obj_type: str) -> List[SemanticObject]:
        """根据类型获取物体"""
        return [
            obj for obj in self.semantic_objects.values()
            if obj.is_valid() and obj.target_type == obj_type
        ]
    
    def get_nearest_object(
        self,
        position: Tuple[float, float],
        label_filter: str = None
    ) -> Optional[SemanticObject]:
        """获取最近的物体"""
        nearest = None
        min_dist = float('inf')
        
        for obj in self.semantic_objects.values():
            if not obj.is_valid():
                continue
            
            if label_filter and label_filter.lower() not in obj.label.lower():
                continue
            
            dist = math.sqrt(
                (obj.world_position[0] - position[0])**2 +
                (obj.world_position[1] - position[1])**2
            )
            
            if dist < min_dist:
                min_dist = dist
                nearest = obj
        
        return nearest
    
    def reset_target(self):
        """重置目标"""
        if self.current_target:
            self.current_target.is_target = False
        self.current_target = None
        self.target_description = ""

    def _manage_memory_usage(self):
        """管理内存使用"""
        if not self.enable_memory_management:
            return

        # 检查是否需要清理
        memory_usage = self._estimate_memory_usage()
        if memory_usage > self.cleanup_threshold:
            logger.warning(f"内存使用率 {memory_usage*100:.1f}% 超过阈值，执行清理")
            self._perform_memory_cleanup()

    def _estimate_memory_usage(self) -> float:
        """估算内存使用率"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_percent() / 100.0
        except ImportError:
            # 如果没有psutil，使用简单的对象计数估算
            object_count = (
                len(self.tracked_objects) +
                len(self.semantic_objects) +
                len(self.exploration_frontiers) +
                len(self.pose_history) +
                len(self.change_history)
            )
            return min(object_count / 10000.0, 1.0)  # 简化估算

    def _perform_memory_cleanup(self):
        """执行内存清理"""
        cleaned_count = 0

        # 清理过期或过多的语义物体
        if len(self.semantic_objects) > self.max_semantic_objects:
            # 按最后访问时间排序，移除最旧的
            sorted_objects = sorted(
                self.semantic_objects.items(),
                key=lambda x: x[1].last_seen
            )
            to_remove = sorted_objects[:len(self.semantic_objects) - self.max_semantic_objects]
            for obj_id, obj in to_remove:
                if obj.state != ObjectState.CONFIRMED:
                    del self.semantic_objects[obj_id]
                    cleaned_count += 1

        # 清理过多的探索边界
        if len(self.exploration_frontiers) > self.max_frontiers:
            self.exploration_frontiers = self.exploration_frontiers[-self.max_frontiers:]
            cleaned_count += len(self.exploration_frontiers) - self.max_frontiers

        # 清理位姿历史
        if len(self.pose_history) > self.max_pose_history:
            old_count = len(self.pose_history)
            self.pose_history = self.pose_history[-self.max_pose_history:]
            cleaned_count += old_count - len(self.pose_history)

        # 清理变化历史
        if len(self.change_history) > self.max_history:
            old_count = len(self.change_history)
            self.change_history = self.change_history[-self.max_history:]
            cleaned_count += old_count - len(self.change_history)

        # 清理场景历史
        if len(self._scene_history) > self._max_history:
            old_count = len(self._scene_history)
            self._scene_history = self._scene_history[-self._max_history:]
            cleaned_count += old_count - len(self._scene_history)

        # 清理已探索的位置（保留最近的部分）
        if len(self.explored_positions) > 10000:
            self.explored_positions = set(list(self.explored_positions)[-5000:])
            cleaned_count += 1

        # 清理地图（如果过大）
        if (self.current_map is not None and
            self.current_map.size > self.max_map_size):
            # 降采样地图
            self.current_map = self._downsample_map(self.current_map)
            cleaned_count += 1

        logger.info(f"内存清理完成: 清理了 {cleaned_count} 个项目")
        return cleaned_count

    def _downsample_map(self, map_array: np.ndarray, factor: int = 2) -> np.ndarray:
        """降采样地图以减少内存使用"""
        try:
            # 使用最大值池化降采样
            h, w = map_array.shape
            new_h, new_w = h // factor, w // factor
            downsampled = np.zeros((new_h, new_w), dtype=map_array.dtype)

            for i in range(new_h):
                for j in range(new_w):
                    patch = map_array[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
                    downsampled[i, j] = np.max(patch)

            return downsampled
        except Exception as e:
            logger.error(f"地图降采样失败: {e}")
            return map_array

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存使用统计"""
        stats = {
            "tracked_objects": len(self.tracked_objects),
            "semantic_objects": len(self.semantic_objects),
            "exploration_frontiers": len(self.exploration_frontiers),
            "pose_history": len(self.pose_history),
            "change_history": len(self.change_history),
            "scene_history": len(self._scene_history),
            "explored_positions": len(self.explored_positions),
        }

        # 估算各部分的内存使用
        if self.current_map is not None:
            stats["map_memory_mb"] = self.current_map.nbytes / (1024 * 1024)
            stats["map_shape"] = self.current_map.shape

        # 获取资源管理器统计
        if self.tracker:
            tracker_stats = self.tracker.get_stats()
            stats["resource_tracker"] = tracker_stats

        return stats

    def force_cleanup(self):
        """强制执行内存清理"""
        logger.info("强制执行世界模型内存清理")
        return self._perform_memory_cleanup()
    
    # ========== 信念修正接口（Belief Revision） ==========
    
    def register_belief(
        self,
        belief_id: str,
        belief_type: Union[BeliefType, str],
        description: str,
        initial_confidence: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[BeliefEntry]:
        """
        注册新信念
        
        Args:
            belief_id: 信念唯一标识
            belief_type: 信念类型（BeliefType枚举或字符串）
            description: 信念描述（如"cup在kitchen"）
            initial_confidence: 初始置信度
            metadata: 元数据
            
        Returns:
            BeliefEntry: 创建的信念条目，如果信念修正不可用则返回None
        """
        if not self.belief_revision_policy:
            logger.warning("信念修正策略不可用，无法注册信念")
            return None
        
        # 转换字符串为枚举
        if isinstance(belief_type, str):
            try:
                belief_type = BeliefType(belief_type)
            except ValueError:
                logger.error(f"无效的信念类型: {belief_type}")
                return None
        
        return self.belief_revision_policy.register_belief(
            belief_id=belief_id,
            belief_type=belief_type,
            description=description,
            initial_confidence=initial_confidence,
            metadata=metadata
        )
    
    def report_operation_failure(
        self,
        belief_id: str,
        operation_type: Union[OperationType, str],
        error: Optional[str] = None
    ) -> Optional[BeliefEntry]:
        """
        报告操作失败
        
        当基于某个信念的操作失败时，降低该信念的置信度
        
        Args:
            belief_id: 相关的信念ID
            operation_type: 失败的操作类型（OperationType枚举或字符串）
            error: 错误信息（可选）
            
        Returns:
            BeliefEntry: 更新后的信念条目，如果信念不存在或不可用则返回None
            
        Example:
            >>> world_model.report_operation_failure(
            ...     belief_id="cup_in_kitchen",
            ...     operation_type="search",
            ...     error="在厨房未找到cup"
            ... )
        """
        if not self.belief_revision_policy:
            logger.warning("信念修正策略不可用，无法报告失败")
            return None
        
        # 转换字符串为枚举
        if isinstance(operation_type, str):
            try:
                operation_type = OperationType(operation_type)
            except ValueError:
                logger.error(f"无效的操作类型: {operation_type}")
                return None
        
        return self.belief_revision_policy.report_operation_failure(
            belief_id=belief_id,
            operation_type=operation_type,
            error=error
        )
    
    def report_operation_success(
        self,
        belief_id: str,
        operation_type: Union[OperationType, str]
    ) -> Optional[BeliefEntry]:
        """
        报告操作成功
        
        当基于某个信念的操作成功时，提高该信念的置信度
        
        Args:
            belief_id: 相关的信念ID
            operation_type: 成功的操作类型（OperationType枚举或字符串）
            
        Returns:
            BeliefEntry: 更新后的信念条目，如果信念不存在或不可用则返回None
            
        Example:
            >>> world_model.report_operation_success(
            ...     belief_id="cup_in_kitchen",
            ...     operation_type="search"
            ... )
        """
        if not self.belief_revision_policy:
            logger.warning("信念修正策略不可用，无法报告成功")
            return None
        
        # 转换字符串为枚举
        if isinstance(operation_type, str):
            try:
                operation_type = OperationType(operation_type)
            except ValueError:
                logger.error(f"无效的操作类型: {operation_type}")
                return None
        
        return self.belief_revision_policy.report_operation_success(
            belief_id=belief_id,
            operation_type=operation_type
        )
    
    def update_belief_decay(self) -> int:
        """
        更新信念衰减
        
        对长时间未观测的信念进行置信度衰减
        应该在定期任务中调用（如每秒或每几秒）
        
        Returns:
            int: 更新的信念数量
        """
        if not self.belief_revision_policy:
            return 0
        
        return self.belief_revision_policy.update_belief_decay()
    
    def get_belief_confidence(self, belief_id: str) -> Optional[float]:
        """
        获取信念的当前置信度
        
        Args:
            belief_id: 信念ID
            
        Returns:
            float: 置信度，如果信念不存在或不可用则返回None
        """
        if not self.belief_revision_policy:
            return None
        
        return self.belief_revision_policy.get_belief_confidence(belief_id)
    
    def get_belief(self, belief_id: str) -> Optional[BeliefEntry]:
        """
        获取信念条目
        
        Args:
            belief_id: 信念ID
            
        Returns:
            BeliefEntry: 信念条目，如果不存在或不可用则返回None
        """
        if not self.belief_revision_policy:
            return None
        
        return self.belief_revision_policy.get_belief(belief_id)
    
    def update_belief_observation(
        self,
        belief_id: str,
        confidence: Optional[float] = None
    ) -> Optional[BeliefEntry]:
        """
        更新信念观测
        
        当有新的观测证据时，更新信念的置信度
        
        Args:
            belief_id: 信念ID
            confidence: 观测置信度（如果为None，则使用默认的observation_boost）
            
        Returns:
            BeliefEntry: 更新后的信念条目
        """
        if not self.belief_revision_policy:
            return None
        
        return self.belief_revision_policy.update_observation(
            belief_id=belief_id,
            confidence=confidence
        )
    
    def get_belief_statistics(self) -> Dict[str, Any]:
        """
        获取信念统计信息
        
        Returns:
            Dict[str, Any]: 统计信息，如果信念修正不可用则返回空字典
        """
        if not self.belief_revision_policy:
            return {}
        
        return self.belief_revision_policy.get_statistics()

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if self.tracker:
                self.tracker.cleanup_all()
        except:
            pass

