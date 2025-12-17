"""
探索式导航规划器 - Exploration Planner

负责:
- 根据目标描述规划探索路径
- 持续更新感知直到找到目标
- 生成导航操作序列
- 处理目标搜索和接近策略
"""

import asyncio
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

from brain.cognitive.world_model import WorldModel, SemanticObject
from brain.perception.vlm_perception import VLMPerception, TargetSearchResult
from brain.perception.ros2_sensor_manager import ROS2SensorManager, PerceptionData
from brain.execution.operations.ros2_ugv import (
    ROS2UGVOperations, UGVOperation, UGVOperationType
)
from brain.platforms.robot_capabilities import RobotCapabilities


class NavigationState(Enum):
    """导航状态"""
    IDLE = "idle"
    SEARCHING = "searching"       # 搜索目标中
    TARGET_FOUND = "target_found" # 已找到目标
    NAVIGATING = "navigating"     # 导航中
    APPROACHING = "approaching"   # 接近目标中
    ARRIVED = "arrived"           # 已到达
    FAILED = "failed"             # 失败
    ABORTED = "aborted"           # 中止


@dataclass
class NavigationResult:
    """导航结果"""
    success: bool
    state: NavigationState
    target_found: bool = False
    target_position: Optional[Tuple[float, float]] = None
    final_distance: float = 0.0
    elapsed_time: float = 0.0
    operations_executed: int = 0
    message: str = ""


@dataclass
class ExplorationConfig:
    """探索配置"""
    # 时间限制
    max_exploration_time: float = 300.0  # 秒
    max_single_search_time: float = 30.0  # 单次搜索时间
    
    # 速度
    exploration_speed: float = 0.3
    approach_speed: float = 0.2
    
    # 距离
    target_arrival_distance: float = 1.0   # 到达判定距离
    obstacle_distance: float = 0.5         # 障碍物距离
    
    # VLM
    vlm_interval: float = 2.0              # VLM调用间隔
    
    # 探索策略
    exploration_pattern: str = "spiral"    # spiral, frontier, random
    max_exploration_steps: int = 50


class ExplorationPlanner:
    """
    探索式导航规划器
    
    通过探索环境来寻找并到达目标
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        vlm: VLMPerception,
        sensor_manager: ROS2SensorManager,
        ugv_ops: ROS2UGVOperations,
        robot_capabilities: RobotCapabilities,
        config: Optional[ExplorationConfig] = None
    ):
        self.world_model = world_model
        self.vlm = vlm
        self.sensor_manager = sensor_manager
        self.ugv_ops = ugv_ops
        self.capabilities = robot_capabilities
        self.config = config or ExplorationConfig()
        
        # 状态
        self.state = NavigationState.IDLE
        self.current_target_description = ""
        self.found_target: Optional[SemanticObject] = None
        
        # 控制
        self._abort_requested = False
        self._pause_requested = False
        
        # 统计
        self._start_time: Optional[datetime] = None
        self._operations_count = 0
        self._vlm_calls = 0
        
        # 回调
        self._progress_callback: Optional[Callable] = None
        self._target_found_callback: Optional[Callable] = None
        
        logger.info("ExplorationPlanner 初始化完成")
    
    async def plan_exploration(
        self,
        target_description: str,
        initial_pose: Tuple[float, float, float] = None
    ) -> List[UGVOperation]:
        """
        规划探索路径以找到目标
        
        Args:
            target_description: 目标描述（如"建筑的门"）
            initial_pose: 初始位姿 (x, y, yaw)
            
        Returns:
            操作序列（初始规划）
        """
        self.current_target_description = target_description
        self.state = NavigationState.SEARCHING
        
        logger.info(f"开始规划探索: 目标='{target_description}'")
        
        # 首先检查目标是否已知
        known_target = self.world_model.find_semantic_target(target_description)
        
        if known_target:
            logger.info(f"目标已在世界模型中: {known_target.label}")
            return await self.plan_approach(known_target)
        
        # 目标未知，需要探索
        operations = self._generate_exploration_sequence()
        
        logger.info(f"生成探索序列: {len(operations)} 个操作")
        
        return operations
    
    async def plan_approach(
        self,
        target: SemanticObject
    ) -> List[UGVOperation]:
        """
        规划接近目标的路径
        
        Args:
            target: 目标物体
            
        Returns:
            操作序列
        """
        self.found_target = target
        self.state = NavigationState.TARGET_FOUND
        
        operations = []
        
        # 获取当前位置
        current_pose = self.sensor_manager.get_current_pose_2d()
        if not current_pose:
            logger.warning("无法获取当前位姿")
            return [self.ugv_ops.stop()]
        
        # 计算到目标的方向和距离
        dx = target.world_position[0] - current_pose.x
        dy = target.world_position[1] - current_pose.y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - current_pose.theta)
        
        logger.info(f"规划接近: 距离={distance:.1f}m, 角度差={math.degrees(angle_diff):.0f}度")
        
        # 1. 先转向目标
        if abs(angle_diff) > 0.1:  # 大于约6度才需要转
            if angle_diff > 0:
                operations.append(self.ugv_ops.rotate_left(abs(angle_diff)))
            else:
                operations.append(self.ugv_ops.rotate_right(abs(angle_diff)))
        
        # 2. 前进到目标附近
        approach_distance = distance - self.config.target_arrival_distance
        if approach_distance > 0:
            operations.append(
                self.ugv_ops.approach_target(
                    target.world_position[0],
                    target.world_position[1],
                    stop_distance=self.config.target_arrival_distance,
                    speed=self.config.approach_speed
                )
            )
        
        # 3. 最后调整
        operations.append(self.ugv_ops.stop())
        
        return operations
    
    async def execute_exploration(
        self,
        target_description: str,
        progress_callback: Callable = None,
        target_found_callback: Callable = None
    ) -> NavigationResult:
        """
        执行完整的探索导航
        
        这是主要的入口函数，会持续探索直到找到目标或超时
        
        Args:
            target_description: 目标描述
            progress_callback: 进度回调
            target_found_callback: 找到目标时的回调
            
        Returns:
            NavigationResult
        """
        self._progress_callback = progress_callback
        self._target_found_callback = target_found_callback
        self._abort_requested = False
        self._start_time = datetime.now()
        self._operations_count = 0
        self._vlm_calls = 0
        
        self.current_target_description = target_description
        self.state = NavigationState.SEARCHING
        
        logger.info(f"开始探索导航: '{target_description}'")
        self._report_progress(f"开始搜索: {target_description}")
        
        try:
            while not self._abort_requested:
                # 检查超时
                elapsed = (datetime.now() - self._start_time).total_seconds()
                if elapsed > self.config.max_exploration_time:
                    logger.warning("探索超时")
                    self.state = NavigationState.FAILED
                    return NavigationResult(
                        success=False,
                        state=NavigationState.FAILED,
                        elapsed_time=elapsed,
                        operations_executed=self._operations_count,
                        message="探索超时，未找到目标"
                    )
                
                # 暂停检查
                while self._pause_requested:
                    await asyncio.sleep(0.1)
                
                # 步骤1: 获取感知数据
                perception = await self.sensor_manager.get_fused_perception()
                
                # 步骤2: VLM分析场景
                if perception.rgb_image is not None:
                    scene = await self.vlm.describe_scene(perception.rgb_image)
                    self._vlm_calls += 1
                    
                    # 更新世界模型
                    robot_pose = perception.pose
                    if robot_pose:
                        self.world_model.update_from_vlm(
                            scene,
                            (robot_pose.x, robot_pose.y, robot_pose.yaw)
                        )
                    
                    # 搜索目标
                    search_result = await self.vlm.find_target(
                        perception.rgb_image,
                        target_description
                    )
                    
                    if search_result.found and search_result.confidence > 0.6:
                        logger.info(f"VLM找到目标! 置信度: {search_result.confidence}")
                        self._report_progress(f"找到目标: {search_result.explanation}")
                        
                        # 更新世界模型中的目标
                        target = self.world_model.find_semantic_target(target_description)
                        
                        if target:
                            self.found_target = target
                            self.state = NavigationState.TARGET_FOUND
                            
                            if self._target_found_callback:
                                await self._target_found_callback(target)
                            
                            # 执行接近
                            return await self._execute_approach_phase(target)
                        else:
                            # 根据VLM建议的动作移动
                            await self._execute_vlm_suggestion(search_result)
                
                # 步骤3: 检查世界模型中是否有目标
                target = self.world_model.find_semantic_target(target_description)
                if target and target.confidence > 0.7:
                    logger.info(f"世界模型中找到目标: {target.label}")
                    return await self._execute_approach_phase(target)
                
                # 步骤4: 继续探索
                await self._execute_exploration_step(perception)
                
                # 等待VLM间隔
                await asyncio.sleep(self.config.vlm_interval)
            
            # 被中止
            self.state = NavigationState.ABORTED
            return NavigationResult(
                success=False,
                state=NavigationState.ABORTED,
                elapsed_time=(datetime.now() - self._start_time).total_seconds(),
                operations_executed=self._operations_count,
                message="导航被中止"
            )
            
        except Exception as e:
            logger.error(f"探索导航异常: {e}")
            self.state = NavigationState.FAILED
            return NavigationResult(
                success=False,
                state=NavigationState.FAILED,
                message=f"导航异常: {str(e)}"
            )
    
    async def _execute_approach_phase(
        self,
        target: SemanticObject
    ) -> NavigationResult:
        """执行接近阶段"""
        self.state = NavigationState.APPROACHING
        self._report_progress(f"开始接近目标: {target.label}")
        
        # 规划接近路径
        operations = await self.plan_approach(target)
        
        # 执行操作
        for op in operations:
            if self._abort_requested:
                break
            
            # 检查障碍物
            def obstacle_check():
                perception = self.sensor_manager._latest_data
                if perception:
                    front_dist = perception.get_front_distance()
                    return front_dist < self.config.obstacle_distance
                return False
            
            success = await self.ugv_ops.execute(op, obstacle_check)
            self._operations_count += 1
            
            if not success:
                logger.warning(f"操作失败: {op.name}")
                # 尝试绕行
                await self._handle_obstacle()
        
        # 检查是否到达
        current_pose = self.sensor_manager.get_current_pose_2d()
        if current_pose:
            dx = target.world_position[0] - current_pose.x
            dy = target.world_position[1] - current_pose.y
            final_distance = math.sqrt(dx**2 + dy**2)
            
            if final_distance < self.config.target_arrival_distance * 1.5:
                self.state = NavigationState.ARRIVED
                self._report_progress(f"已到达目标! 距离: {final_distance:.1f}m")
                
                return NavigationResult(
                    success=True,
                    state=NavigationState.ARRIVED,
                    target_found=True,
                    target_position=target.world_position,
                    final_distance=final_distance,
                    elapsed_time=(datetime.now() - self._start_time).total_seconds(),
                    operations_executed=self._operations_count,
                    message="成功到达目标"
                )
        
        # 未完全到达，但找到了目标
        self.state = NavigationState.TARGET_FOUND
        return NavigationResult(
            success=True,
            state=NavigationState.TARGET_FOUND,
            target_found=True,
            target_position=target.world_position,
            elapsed_time=(datetime.now() - self._start_time).total_seconds(),
            operations_executed=self._operations_count,
            message="找到目标但接近过程中断"
        )
    
    async def _execute_exploration_step(self, perception: PerceptionData):
        """执行一步探索"""
        self.state = NavigationState.SEARCHING
        
        # 获取探索方向
        exploration_target = self.world_model.get_exploration_target()
        
        if exploration_target:
            # 有未探索区域，导航到那里
            current_pose = self.sensor_manager.get_current_pose_2d()
            if current_pose:
                dx = exploration_target[0] - current_pose.x
                dy = exploration_target[1] - current_pose.y
                target_angle = math.atan2(dy, dx)
                angle_diff = self._normalize_angle(target_angle - current_pose.theta)
                
                # 转向探索目标
                if abs(angle_diff) > 0.3:
                    if angle_diff > 0:
                        op = self.ugv_ops.rotate_left(min(abs(angle_diff), math.pi/4))
                    else:
                        op = self.ugv_ops.rotate_right(min(abs(angle_diff), math.pi/4))
                    await self.ugv_ops.execute(op)
                    self._operations_count += 1
        
        # 检查前方是否安全
        if perception.is_path_clear("front", self.config.obstacle_distance):
            # 前方安全，前进
            op = self.ugv_ops.explore_towards(
                "forward",
                duration=3.0,
                speed=self.config.exploration_speed
            )
            
            def obstacle_check():
                data = self.sensor_manager._latest_data
                return data and data.get_front_distance() < self.config.obstacle_distance
            
            await self.ugv_ops.execute(op, obstacle_check)
            self._operations_count += 1
            
        else:
            # 前方有障碍，选择转向
            if perception.is_path_clear("left"):
                op = self.ugv_ops.rotate_left(math.pi / 4)
            elif perception.is_path_clear("right"):
                op = self.ugv_ops.rotate_right(math.pi / 4)
            else:
                # 都不通，后退
                op = self.ugv_ops.move_backward(0.5, speed=0.2)
            
            await self.ugv_ops.execute(op)
            self._operations_count += 1
    
    async def _execute_vlm_suggestion(self, search_result: TargetSearchResult):
        """根据VLM建议执行动作"""
        suggestion = search_result.suggested_action.lower()
        
        if "左" in suggestion or "left" in suggestion:
            op = self.ugv_ops.rotate_left(math.pi / 6)
        elif "右" in suggestion or "right" in suggestion:
            op = self.ugv_ops.rotate_right(math.pi / 6)
        elif "前进" in suggestion or "forward" in suggestion:
            op = self.ugv_ops.move_forward(1.0, speed=self.config.exploration_speed)
        else:
            # 默认前进
            op = self.ugv_ops.move_forward(0.5, speed=self.config.exploration_speed)
        
        await self.ugv_ops.execute(op)
        self._operations_count += 1
    
    async def _handle_obstacle(self):
        """处理障碍物"""
        perception = await self.sensor_manager.get_fused_perception()
        
        # 决定绕行方向
        left_clear = perception.is_path_clear("left")
        right_clear = perception.is_path_clear("right")
        
        if left_clear:
            op = self.ugv_ops.avoid_obstacle("left")
        elif right_clear:
            op = self.ugv_ops.avoid_obstacle("right")
        else:
            # 两边都不通，后退
            op = self.ugv_ops.move_backward(1.0)
        
        await self.ugv_ops.execute(op)
        self._operations_count += 1
    
    def _generate_exploration_sequence(self) -> List[UGVOperation]:
        """生成探索操作序列"""
        operations = []
        
        if self.config.exploration_pattern == "spiral":
            # 螺旋探索模式
            for i in range(min(10, self.config.max_exploration_steps)):
                # 前进
                distance = 1.0 + i * 0.5
                operations.append(
                    self.ugv_ops.explore_towards("forward", duration=distance * 2)
                )
                # 转向
                operations.append(
                    self.ugv_ops.rotate_left(math.pi / 2)
                )
        
        elif self.config.exploration_pattern == "frontier":
            # 基于边界的探索
            for i in range(min(5, self.config.max_exploration_steps)):
                operations.append(
                    self.ugv_ops.explore_towards("forward", duration=5.0)
                )
                # 交替转向
                if i % 2 == 0:
                    operations.append(self.ugv_ops.rotate_left(math.pi / 3))
                else:
                    operations.append(self.ugv_ops.rotate_right(math.pi / 3))
        
        else:
            # 随机探索
            import random
            for _ in range(min(8, self.config.max_exploration_steps)):
                operations.append(
                    self.ugv_ops.explore_towards("forward", duration=3.0)
                )
                if random.random() > 0.5:
                    operations.append(self.ugv_ops.rotate_left(random.uniform(0.3, 1.0)))
                else:
                    operations.append(self.ugv_ops.rotate_right(random.uniform(0.3, 1.0)))
        
        return operations
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _report_progress(self, message: str):
        """报告进度"""
        logger.info(f"导航进度: {message}")
        if self._progress_callback:
            try:
                self._progress_callback(message, self.state)
            except Exception as e:
                logger.warning(f"进度回调失败: {e}")
    
    # === 控制方法 ===
    
    def abort(self):
        """中止探索"""
        self._abort_requested = True
        self.ugv_ops.abort()
        logger.info("请求中止探索导航")
    
    def pause(self):
        """暂停探索"""
        self._pause_requested = True
        logger.info("暂停探索导航")
    
    def resume(self):
        """恢复探索"""
        self._pause_requested = False
        logger.info("恢复探索导航")
    
    def get_state(self) -> NavigationState:
        """获取当前状态"""
        return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "state": self.state.value,
            "elapsed_time": elapsed,
            "operations_executed": self._operations_count,
            "vlm_calls": self._vlm_calls,
            "target_found": self.found_target is not None,
            "current_target": self.current_target_description
        }

