"""
路口导航器 - Intersection Navigator

负责:
- 基于地图/位姿的路口右转策略（非固定三步）
- VLM路口检测和方向识别
- 动态路径规划
"""

import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from brain.communication.control_adapter import ControlAdapter, PlatformType
from brain.perception.vlm_perception import VLMPerception, SceneDescription
from brain.perception.ros2_sensor_manager import ROS2SensorManager, PerceptionData, Pose2D
from brain.cognitive.world_model import WorldModel
from brain.navigation.smooth_executor import SmoothExecutor


@dataclass
class IntersectionInfo:
    """路口信息"""
    detected: bool = False
    direction: str = "unknown"  # "left", "right", "straight", "unknown"
    distance: float = 0.0  # 到路口的距离（米）
    confidence: float = 0.0
    available_directions: List[str] = field(default_factory=list)
    turn_radius: float = 2.0  # 建议转弯半径（米）


@dataclass
class TurnPlan:
    """转弯计划"""
    approach_phase: List[Tuple[float, float]]  # [(speed, angular), ...]
    turn_phase: List[Tuple[float, float]]
    exit_phase: List[Tuple[float, float]]
    estimated_duration: float = 0.0
    total_distance: float = 0.0


class IntersectionNavigator:
    """
    路口导航器
    
    基于VLM感知和地图信息进行路口导航
    """
    
    def __init__(
        self,
        control_adapter: ControlAdapter,
        smooth_executor: SmoothExecutor,
        sensor_manager: ROS2SensorManager,
        world_model: WorldModel,
        vlm: VLMPerception,
        config: Optional[Dict[str, Any]] = None
    ):
        self.control = control_adapter
        self.executor = smooth_executor
        self.sensors = sensor_manager
        self.world_model = world_model
        self.vlm = vlm
        self.config = config or {}
        
        # 路口检测参数
        self.intersection_detection_distance = self.config.get("intersection_detection_distance", 5.0)
        self.turn_radius = self.config.get("turn_radius", 2.0)
        self.approach_speed = self.config.get("approach_speed", 0.3)
        self.turn_speed = self.config.get("turn_speed", 0.3)
        self.exit_speed = self.config.get("exit_speed", 0.5)
        
        # 状态
        self.current_intersection: Optional[IntersectionInfo] = None
        self.turn_plan: Optional[TurnPlan] = None
        
        logger.info("IntersectionNavigator 初始化完成")
    
    async def detect_intersection(
        self,
        perception: PerceptionData
    ) -> IntersectionInfo:
        """
        检测路口
        
        Args:
            perception: 当前感知数据
            
        Returns:
            IntersectionInfo: 路口信息
        """
        info = IntersectionInfo()
        
        # 使用VLM分析场景
        if perception.rgb_image is not None:
            try:
                scene = await self.vlm.describe_scene(perception.rgb_image)
                
                # 检查场景描述中是否包含路口关键词
                summary_lower = scene.summary.lower()
                has_intersection = any(keyword in summary_lower for keyword in [
                    "intersection", "路口", "交叉", "junction", "crossroad"
                ])
                
                if has_intersection:
                    info.detected = True
                    info.confidence = 0.7
                    
                    # 尝试识别方向
                    if "右" in summary_lower or "right" in summary_lower:
                        info.direction = "right"
                        info.available_directions.append("right")
                    if "左" in summary_lower or "left" in summary_lower:
                        info.available_directions.append("left")
                    if "直" in summary_lower or "straight" in summary_lower:
                        info.available_directions.append("straight")
                    
                    # 估算距离（简化：基于深度图）
                    if perception.depth_image is not None:
                        h, w = perception.depth_image.shape
                        center_depth = perception.depth_image[h//2, w//2]
                        if 0.1 < center_depth < 20.0:
                            info.distance = center_depth
                    
            except Exception as e:
                logger.warning(f"VLM路口检测异常: {e}")
        
        # 使用地图信息辅助检测
        if perception.pose:
            # 检查地图中是否有路口特征（简化实现）
            # TODO: 基于占据栅格检测路口形状
            pass
        
        self.current_intersection = info
        return info
    
    def plan_turn(
        self,
        turn_direction: str,
        intersection_info: IntersectionInfo,
        current_pose: Pose2D
    ) -> TurnPlan:
        """
        规划转弯路径
        
        Args:
            turn_direction: 转弯方向 ("left", "right", "straight")
            intersection_info: 路口信息
            current_pose: 当前位置
            
        Returns:
            TurnPlan: 转弯计划
        """
        # 根据平台能力计算转弯参数
        capabilities = self.control.get_capabilities()
        turn_radius = intersection_info.turn_radius
        
        # 如果平台有最小转弯半径限制
        if capabilities.min_turn_radius > 0:
            turn_radius = max(turn_radius, capabilities.min_turn_radius)
        
        plan = TurnPlan(
            approach_phase=[],
            turn_phase=[],
            exit_phase=[],
            estimated_duration=0.0,
            total_distance=0.0
        )
        
        if turn_direction == "straight":
            # 直行：简单前进
            plan.approach_phase = [(self.approach_speed, 0.0)]
            plan.exit_phase = [(self.exit_speed, 0.0)]
            plan.estimated_duration = 5.0
            plan.total_distance = 5.0
        
        elif turn_direction == "right":
            # 右转
            # 1. 接近路口（减速）
            approach_dist = max(0.5, intersection_info.distance - 1.0)
            approach_time = approach_dist / self.approach_speed
            plan.approach_phase = [(self.approach_speed, 0.0)]
            
            # 2. 转弯（根据转弯半径计算角速度）
            # 对于Ackermann: angular = linear / radius
            # 对于Differential: angular = linear / radius
            turn_angular = -self.turn_speed / turn_radius  # 负值表示右转
            
            # 转弯角度（90度）
            turn_angle = math.pi / 2
            turn_arc_length = turn_radius * turn_angle
            turn_time = turn_arc_length / self.turn_speed
            
            plan.turn_phase = [(self.turn_speed, turn_angular)]
            
            # 3. 转弯后直行
            exit_dist = 3.0
            exit_time = exit_dist / self.exit_speed
            plan.exit_phase = [(self.exit_speed, 0.0)]
            
            plan.estimated_duration = approach_time + turn_time + exit_time
            plan.total_distance = approach_dist + turn_arc_length + exit_dist
        
        elif turn_direction == "left":
            # 左转（类似右转，但角速度为正）
            approach_dist = max(0.5, intersection_info.distance - 1.0)
            approach_time = approach_dist / self.approach_speed
            plan.approach_phase = [(self.approach_speed, 0.0)]
            
            turn_angular = self.turn_speed / turn_radius  # 正值表示左转
            turn_angle = math.pi / 2
            turn_arc_length = turn_radius * turn_angle
            turn_time = turn_arc_length / self.turn_speed
            
            plan.turn_phase = [(self.turn_speed, turn_angular)]
            
            exit_dist = 3.0
            exit_time = exit_dist / self.exit_speed
            plan.exit_phase = [(self.exit_speed, 0.0)]
            
            plan.estimated_duration = approach_time + turn_time + exit_time
            plan.total_distance = approach_dist + turn_arc_length + exit_dist
        
        self.turn_plan = plan
        logger.info(f"规划转弯: {turn_direction}, 预计时间={plan.estimated_duration:.1f}秒")
        
        return plan
    
    async def execute_turn(
        self,
        turn_direction: str,
        replan_callback: Optional[Callable] = None
    ) -> bool:
        """
        执行转弯
        
        Args:
            turn_direction: 转弯方向
            replan_callback: 重规划回调（当检测到障碍或偏差时调用）
            
        Returns:
            bool: 是否成功
        """
        # 等待获取位姿数据（最多等待5秒）
        import asyncio
        perception = None
        
        # 检查ROS2接口的里程计数据
        ros2_data = self.sensors.ros2.get_sensor_data()
        odom_callback_count = self.sensors.ros2._callback_counts.get("odom", 0)
        logger.info(f"里程计状态: 回调次数={odom_callback_count}, 有数据={ros2_data.odometry is not None}")
        
        for i in range(10):
            perception = await self.sensors.get_fused_perception()
            if perception and perception.pose:
                logger.info(f"成功获取位姿: ({perception.pose.x:.2f}, {perception.pose.y:.2f}, {perception.pose.yaw:.2f})")
                break
            if i == 0:
                logger.warning("等待位姿数据...")
                # 检查话题状态
                if odom_callback_count == 0:
                    odom_topic = self.sensors.ros2.config.topics.get("odom", "/odom")
                    logger.warning(f"里程计话题 {odom_topic} 没有收到数据，可能原因:")
                    logger.warning(f"  1. 话题没有发布者（Publisher count = 0）")
                    logger.warning(f"  2. 话题名称不正确")
                    logger.warning(f"  3. 仿真环境未启动或car3未激活")
            await asyncio.sleep(0.5)
        
        if not perception or not perception.pose:
            logger.error("无法获取位姿，无法执行转弯")
            logger.warning("提示: 请确保里程计话题已发布数据")
            # 即使没有位姿，也尝试执行（使用默认值）
            logger.info("尝试使用默认位姿继续执行...")
            # 创建一个默认位姿
            from brain.perception.ros2_sensor_manager import Pose3D
            if perception:
                perception.pose = Pose3D(x=0.0, y=0.0, z=0.0, yaw=0.0)
            else:
                perception = await self.sensors.get_fused_perception()
                if not perception:
                    return False
                perception.pose = Pose3D(x=0.0, y=0.0, z=0.0, yaw=0.0)
        
        current_pose = perception.pose.to_2d()
        
        # 检测路口
        intersection_info = await self.detect_intersection(perception)
        
        if not intersection_info.detected:
            logger.warning("未检测到路口，尝试基于指令执行转弯")
            # 使用默认参数
            intersection_info.distance = 3.0
            intersection_info.turn_radius = self.turn_radius
        
        # 规划转弯
        plan = self.plan_turn(turn_direction, intersection_info, current_pose)
        
        # 执行转弯计划
        try:
            # 阶段1: 接近路口
            for speed, angular in plan.approach_phase:
                await self.executor.execute_continuous(
                    target_speed=speed,
                    target_angular=angular,
                    duration=1.0,  # 每阶段执行1秒，实际由感知调整
                    obstacle_check=lambda: self._check_obstacle(perception),
                    progress_callback=replan_callback
                )
                
                # 检查是否需要重规划
                if replan_callback:
                    new_perception = await self.sensors.get_fused_perception()
                    if self._needs_replan(new_perception, perception):
                        logger.info("检测到需要重规划，调用回调")
                        replan_callback()
                        return False
            
            # 阶段2: 转弯
            for speed, angular in plan.turn_phase:
                await self.executor.execute_continuous(
                    target_speed=speed,
                    target_angular=angular,
                    duration=2.0,  # 转弯阶段
                    obstacle_check=lambda: self._check_obstacle(perception),
                    progress_callback=replan_callback
                )
                
                if replan_callback:
                    new_perception = await self.sensors.get_fused_perception()
                    if self._needs_replan(new_perception, perception):
                        replan_callback()
                        return False
            
            # 阶段3: 转弯后直行
            for speed, angular in plan.exit_phase:
                await self.executor.execute_continuous(
                    target_speed=speed,
                    target_angular=angular,
                    duration=3.0,
                    obstacle_check=lambda: self._check_obstacle(perception),
                    progress_callback=replan_callback
                )
            
            logger.info("转弯执行完成")
            return True
            
        except Exception as e:
            logger.error(f"转弯执行异常: {e}")
            await self.control.stop()
            return False
    
    def _check_obstacle(self, perception: PerceptionData) -> bool:
        """检查是否有障碍物"""
        if not perception:
            return False
        
        front_dist = perception.get_front_distance()
        return front_dist < 0.5  # 0.5米内有障碍物
    
    def _needs_replan(
        self,
        new_perception: PerceptionData,
        old_perception: PerceptionData
    ) -> bool:
        """检查是否需要重规划"""
        if not new_perception or not old_perception:
            return False
        
        # 检查障碍物变化
        new_obstacles = len(new_perception.obstacles)
        old_obstacles = len(old_perception.obstacles)
        
        if new_obstacles > old_obstacles + 1:
            return True
        
        # 检查前方距离
        new_front = new_perception.get_front_distance()
        old_front = old_perception.get_front_distance()
        
        if new_front < 0.5 and old_front > 1.0:
            return True
        
        return False
    
    def get_current_intersection(self) -> Optional[IntersectionInfo]:
        """获取当前检测到的路口信息"""
        return self.current_intersection
    
    def get_turn_plan(self) -> Optional[TurnPlan]:
        """获取当前转弯计划"""
        return self.turn_plan

