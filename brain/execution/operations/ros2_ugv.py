"""
ROS2 UGV原子操作 - ROS2 UGV Operations

定义无人地面车的所有原子操作，包括:
- 基本运动: 前进、后退、转向、停止
- 导航: 导航到点、跟随路径
- 探索: 向某方向探索、搜索目标
- 特殊: 接近目标、绕行障碍
"""

import asyncio
import math
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from loguru import logger

from brain.execution.operations.base import OperationType, OperationStatus
from brain.communication.ros2_interface import ROS2Interface, TwistCommand


class UGVOperationType(Enum):
    """UGV操作类型"""
    # 基本运动
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    STOP = "stop"
    
    # 导航
    NAVIGATE_TO = "navigate_to"
    FOLLOW_PATH = "follow_path"
    
    # 探索
    EXPLORE_FORWARD = "explore_forward"
    EXPLORE_LEFT = "explore_left"
    EXPLORE_RIGHT = "explore_right"
    SEARCH_TARGET = "search_target"
    
    # 目标交互
    APPROACH_TARGET = "approach_target"
    ALIGN_WITH_TARGET = "align_with_target"
    
    # 避障
    AVOID_OBSTACLE = "avoid_obstacle"
    BYPASS_LEFT = "bypass_left"
    BYPASS_RIGHT = "bypass_right"


@dataclass
class UGVOperation:
    """UGV操作"""
    # 基本信息
    name: str = "operation"
    type: OperationType = OperationType.MOVEMENT
    ugv_type: UGVOperationType = UGVOperationType.STOP
    description: str = ""
    
    # 运动参数
    distance: float = 0.0        # 距离 (米)
    angle: float = 0.0           # 角度 (弧度)
    linear_speed: float = 0.5    # 线速度 (m/s)
    angular_speed: float = 0.5   # 角速度 (rad/s)
    duration: float = 0.0        # 持续时间 (秒)
    
    # 目标位置
    target_x: float = 0.0
    target_y: float = 0.0
    target_yaw: float = 0.0
    
    # 路径
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    
    # 探索参数
    exploration_direction: str = "forward"
    target_description: str = ""
    
    # ROS2命令
    twist_cmd: Optional[TwistCommand] = None
    
    # 执行状态
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OperationStatus = OperationStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    
    def get_estimated_duration(self) -> float:
        """估计操作耗时"""
        if self.duration > 0:
            return self.duration
        
        if self.ugv_type in [UGVOperationType.MOVE_FORWARD, UGVOperationType.MOVE_BACKWARD]:
            return self.distance / max(self.linear_speed, 0.1)
        
        if self.ugv_type in [UGVOperationType.TURN_LEFT, UGVOperationType.TURN_RIGHT,
                             UGVOperationType.ROTATE_LEFT, UGVOperationType.ROTATE_RIGHT]:
            return abs(self.angle) / max(self.angular_speed, 0.1)
        
        if self.ugv_type == UGVOperationType.NAVIGATE_TO:
            dist = math.sqrt(self.target_x**2 + self.target_y**2)
            return dist / max(self.linear_speed, 0.1) + 5.0  # 额外时间
        
        return 10.0  # 默认


class ROS2UGVOperations:
    """
    ROS2 UGV操作执行器
    
    负责将高级操作转换为ROS2命令并执行
    """
    
    def __init__(
        self,
        ros2_interface: ROS2Interface,
        config: Optional[Dict[str, Any]] = None
    ):
        self.ros2 = ros2_interface
        self.config = config or {}
        
        # 默认参数
        self.default_linear_speed = self.config.get("default_linear_speed", 0.5)
        self.default_angular_speed = self.config.get("default_angular_speed", 0.5)
        self.control_rate = self.config.get("control_rate", 10.0)  # Hz
        
        # 安全参数
        self.min_obstacle_dist = self.config.get("min_obstacle_dist", 0.5)
        self.emergency_stop_dist = self.config.get("emergency_stop_dist", 0.3)
        
        # 状态
        self._current_operation: Optional[UGVOperation] = None
        self._is_executing = False
        self._abort_requested = False
        
        # 回调
        self._obstacle_callback: Optional[Callable] = None
        
        logger.info("ROS2UGVOperations 初始化完成")
    
    # === 操作工厂方法 ===
    
    @staticmethod
    def move_forward(
        distance: float,
        speed: float = 0.5
    ) -> UGVOperation:
        """创建前进操作"""
        return UGVOperation(
            name="前进",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.MOVE_FORWARD,
            distance=distance,
            linear_speed=speed,
            twist_cmd=TwistCommand.forward(speed),
            description=f"前进 {distance:.1f} 米"
        )
    
    @staticmethod
    def move_backward(
        distance: float,
        speed: float = 0.3
    ) -> UGVOperation:
        """创建后退操作"""
        return UGVOperation(
            name="后退",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.MOVE_BACKWARD,
            distance=distance,
            linear_speed=speed,
            twist_cmd=TwistCommand.backward(speed),
            description=f"后退 {distance:.1f} 米"
        )
    
    @staticmethod
    def turn_left(
        angle: float = math.pi / 4,
        linear_speed: float = 0.3,
        angular_speed: float = 0.5
    ) -> UGVOperation:
        """创建左转操作"""
        return UGVOperation(
            name="左转",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.TURN_LEFT,
            angle=angle,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            twist_cmd=TwistCommand.turn_left(linear_speed, angular_speed),
            description=f"左转 {math.degrees(angle):.0f} 度"
        )
    
    @staticmethod
    def turn_right(
        angle: float = math.pi / 4,
        linear_speed: float = 0.3,
        angular_speed: float = 0.5
    ) -> UGVOperation:
        """创建右转操作"""
        return UGVOperation(
            name="右转",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.TURN_RIGHT,
            angle=angle,
            linear_speed=linear_speed,
            angular_speed=angular_speed,
            twist_cmd=TwistCommand.turn_right(linear_speed, angular_speed),
            description=f"右转 {math.degrees(angle):.0f} 度"
        )
    
    @staticmethod
    def rotate_left(
        angle: float = math.pi / 2,
        speed: float = 0.5
    ) -> UGVOperation:
        """创建原地左旋转操作"""
        return UGVOperation(
            name="原地左转",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.ROTATE_LEFT,
            angle=angle,
            angular_speed=speed,
            twist_cmd=TwistCommand.rotate_left(speed),
            description=f"原地左转 {math.degrees(angle):.0f} 度"
        )
    
    @staticmethod
    def rotate_right(
        angle: float = math.pi / 2,
        speed: float = 0.5
    ) -> UGVOperation:
        """创建原地右旋转操作"""
        return UGVOperation(
            name="原地右转",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.ROTATE_RIGHT,
            angle=angle,
            angular_speed=speed,
            twist_cmd=TwistCommand.rotate_right(speed),
            description=f"原地右转 {math.degrees(angle):.0f} 度"
        )
    
    @staticmethod
    def stop() -> UGVOperation:
        """创建停止操作"""
        return UGVOperation(
            name="停止",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.STOP,
            twist_cmd=TwistCommand.stop(),
            description="停止"
        )
    
    @staticmethod
    def navigate_to(
        x: float,
        y: float,
        speed: float = 0.5
    ) -> UGVOperation:
        """创建导航到点操作"""
        return UGVOperation(
            name="导航到目标",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.NAVIGATE_TO,
            target_x=x,
            target_y=y,
            linear_speed=speed,
            description=f"导航到 ({x:.1f}, {y:.1f})"
        )
    
    @staticmethod
    def explore_towards(
        direction: str = "forward",
        duration: float = 5.0,
        speed: float = 0.3
    ) -> UGVOperation:
        """创建探索操作"""
        op_type_map = {
            "forward": UGVOperationType.EXPLORE_FORWARD,
            "left": UGVOperationType.EXPLORE_LEFT,
            "right": UGVOperationType.EXPLORE_RIGHT
        }
        
        direction_name = {"forward": "前方", "left": "左侧", "right": "右侧"}
        
        return UGVOperation(
            name=f"探索{direction_name.get(direction, direction)}",
            type=OperationType.MOVEMENT,
            ugv_type=op_type_map.get(direction, UGVOperationType.EXPLORE_FORWARD),
            exploration_direction=direction,
            duration=duration,
            linear_speed=speed,
            description=f"向{direction_name.get(direction, direction)}探索 {duration:.0f} 秒"
        )
    
    @staticmethod
    def search_target(
        target_description: str,
        duration: float = 30.0
    ) -> UGVOperation:
        """创建搜索目标操作"""
        return UGVOperation(
            name="搜索目标",
            type=OperationType.PERCEPTION,
            ugv_type=UGVOperationType.SEARCH_TARGET,
            target_description=target_description,
            duration=duration,
            description=f"搜索: {target_description}"
        )
    
    @staticmethod
    def approach_target(
        target_x: float,
        target_y: float,
        stop_distance: float = 1.0,
        speed: float = 0.3
    ) -> UGVOperation:
        """创建接近目标操作"""
        return UGVOperation(
            name="接近目标",
            type=OperationType.MOVEMENT,
            ugv_type=UGVOperationType.APPROACH_TARGET,
            target_x=target_x,
            target_y=target_y,
            distance=stop_distance,
            linear_speed=speed,
            description=f"接近目标，停止距离 {stop_distance:.1f}m"
        )
    
    @staticmethod
    def avoid_obstacle(direction: str = "left") -> UGVOperation:
        """创建避障操作"""
        if direction == "left":
            return UGVOperation(
                name="左侧绕行",
                type=OperationType.MOVEMENT,
                ugv_type=UGVOperationType.BYPASS_LEFT,
                description="左侧绕行避障"
            )
        else:
            return UGVOperation(
                name="右侧绕行",
                type=OperationType.MOVEMENT,
                ugv_type=UGVOperationType.BYPASS_RIGHT,
                description="右侧绕行避障"
            )
    
    # === 执行方法 ===
    
    async def execute(
        self,
        operation: UGVOperation,
        obstacle_check: Callable[[], bool] = None
    ) -> bool:
        """
        执行UGV操作
        
        Args:
            operation: 要执行的操作
            obstacle_check: 障碍物检测回调（返回True表示有障碍）
            
        Returns:
            bool: 是否成功完成
        """
        self._current_operation = operation
        self._is_executing = True
        self._abort_requested = False
        self._obstacle_callback = obstacle_check
        
        operation.status = OperationStatus.EXECUTING
        operation.start_time = datetime.now()
        
        logger.info(f"执行操作: {operation.description}")
        
        try:
            success = await self._execute_by_type(operation)
            
            if success:
                operation.status = OperationStatus.SUCCESS
                logger.info(f"操作完成: {operation.name}")
            else:
                operation.status = OperationStatus.FAILED
                logger.warning(f"操作失败: {operation.name}")
            
            return success
            
        except Exception as e:
            operation.status = OperationStatus.FAILED
            operation.error = str(e)
            logger.error(f"操作执行异常: {e}")
            return False
            
        finally:
            self._is_executing = False
            self._current_operation = None
            # 确保停止
            await self.ros2.publish_twist(TwistCommand.stop())
    
    async def _execute_by_type(self, op: UGVOperation) -> bool:
        """根据类型执行操作"""
        
        if op.ugv_type == UGVOperationType.STOP:
            await self.ros2.publish_twist(TwistCommand.stop())
            return True
        
        elif op.ugv_type in [UGVOperationType.MOVE_FORWARD, UGVOperationType.MOVE_BACKWARD]:
            return await self._execute_move(op)
        
        elif op.ugv_type in [UGVOperationType.TURN_LEFT, UGVOperationType.TURN_RIGHT]:
            return await self._execute_turn(op)
        
        elif op.ugv_type in [UGVOperationType.ROTATE_LEFT, UGVOperationType.ROTATE_RIGHT]:
            return await self._execute_rotate(op)
        
        elif op.ugv_type == UGVOperationType.NAVIGATE_TO:
            return await self._execute_navigate(op)
        
        elif op.ugv_type in [UGVOperationType.EXPLORE_FORWARD, 
                             UGVOperationType.EXPLORE_LEFT,
                             UGVOperationType.EXPLORE_RIGHT]:
            return await self._execute_explore(op)
        
        elif op.ugv_type == UGVOperationType.APPROACH_TARGET:
            return await self._execute_approach(op)
        
        elif op.ugv_type in [UGVOperationType.BYPASS_LEFT, UGVOperationType.BYPASS_RIGHT]:
            return await self._execute_bypass(op)
        
        else:
            logger.warning(f"未知操作类型: {op.ugv_type}")
            return False
    
    async def _execute_move(self, op: UGVOperation) -> bool:
        """执行移动操作"""
        if op.distance <= 0:
            return True
        
        # 计算需要的时间
        duration = op.distance / op.linear_speed
        interval = 1.0 / self.control_rate
        elapsed = 0.0
        
        while elapsed < duration and not self._abort_requested:
            # 障碍物检测
            if self._obstacle_callback and self._obstacle_callback():
                logger.warning("检测到障碍物，停止移动")
                await self.ros2.publish_twist(TwistCommand.stop())
                return False
            
            # 发送速度命令
            await self.ros2.publish_twist(op.twist_cmd)
            
            await asyncio.sleep(interval)
            elapsed += interval
        
        # 停止
        await self.ros2.publish_twist(TwistCommand.stop())
        
        return not self._abort_requested
    
    async def _execute_turn(self, op: UGVOperation) -> bool:
        """执行转向操作"""
        if op.angle <= 0:
            return True
        
        # 计算需要的时间
        duration = op.angle / op.angular_speed
        interval = 1.0 / self.control_rate
        elapsed = 0.0
        
        while elapsed < duration and not self._abort_requested:
            await self.ros2.publish_twist(op.twist_cmd)
            await asyncio.sleep(interval)
            elapsed += interval
        
        await self.ros2.publish_twist(TwistCommand.stop())
        
        return not self._abort_requested
    
    async def _execute_rotate(self, op: UGVOperation) -> bool:
        """执行原地旋转操作"""
        if op.angle <= 0:
            return True
        
        duration = op.angle / op.angular_speed
        interval = 1.0 / self.control_rate
        elapsed = 0.0
        
        while elapsed < duration and not self._abort_requested:
            await self.ros2.publish_twist(op.twist_cmd)
            await asyncio.sleep(interval)
            elapsed += interval
        
        await self.ros2.publish_twist(TwistCommand.stop())
        
        return not self._abort_requested
    
    async def _execute_navigate(self, op: UGVOperation) -> bool:
        """执行导航操作"""
        interval = 1.0 / self.control_rate
        
        while not self._abort_requested:
            # 获取当前位姿
            current_pose = self.ros2.get_current_pose()
            x, y, yaw = current_pose
            
            # 计算到目标的距离和方向
            dx = op.target_x - x
            dy = op.target_y - y
            distance = math.sqrt(dx**2 + dy**2)
            
            # 到达判断
            if distance < 0.3:
                logger.info("到达目标点")
                await self.ros2.publish_twist(TwistCommand.stop())
                return True
            
            # 计算目标方向
            target_angle = math.atan2(dy, dx)
            angle_diff = self._normalize_angle(target_angle - yaw)
            
            # 障碍物检测
            if self._obstacle_callback and self._obstacle_callback():
                logger.warning("导航中检测到障碍物")
                await self.ros2.publish_twist(TwistCommand.stop())
                return False
            
            # 计算速度命令
            if abs(angle_diff) > 0.3:
                # 需要先转向
                angular = op.angular_speed if angle_diff > 0 else -op.angular_speed
                cmd = TwistCommand(linear_x=0.1, angular_z=angular)
            else:
                # 前进
                linear = min(op.linear_speed, distance)
                angular = angle_diff * 0.5  # P控制
                cmd = TwistCommand(linear_x=linear, angular_z=angular)
            
            await self.ros2.publish_twist(cmd)
            await asyncio.sleep(interval)
        
        await self.ros2.publish_twist(TwistCommand.stop())
        return False
    
    async def _execute_explore(self, op: UGVOperation) -> bool:
        """执行探索操作"""
        # 根据方向选择初始动作
        if op.exploration_direction == "left":
            # 先左转45度
            await self._execute_rotate(self.rotate_left(math.pi / 4))
        elif op.exploration_direction == "right":
            # 先右转45度
            await self._execute_rotate(self.rotate_right(math.pi / 4))
        
        # 然后前进探索
        interval = 1.0 / self.control_rate
        elapsed = 0.0
        
        while elapsed < op.duration and not self._abort_requested:
            # 障碍物检测
            if self._obstacle_callback and self._obstacle_callback():
                logger.info("探索中遇到障碍物，尝试绕行")
                # 简单的绕行策略
                await self._execute_rotate(self.rotate_left(math.pi / 4))
                continue
            
            cmd = TwistCommand(linear_x=op.linear_speed, angular_z=0)
            await self.ros2.publish_twist(cmd)
            
            await asyncio.sleep(interval)
            elapsed += interval
        
        await self.ros2.publish_twist(TwistCommand.stop())
        return True
    
    async def _execute_approach(self, op: UGVOperation) -> bool:
        """执行接近目标操作"""
        interval = 1.0 / self.control_rate
        stop_distance = op.distance if op.distance > 0 else 1.0
        
        while not self._abort_requested:
            current_pose = self.ros2.get_current_pose()
            x, y, yaw = current_pose
            
            dx = op.target_x - x
            dy = op.target_y - y
            distance = math.sqrt(dx**2 + dy**2)
            
            # 到达停止距离
            if distance < stop_distance:
                logger.info(f"到达目标附近 (距离: {distance:.2f}m)")
                await self.ros2.publish_twist(TwistCommand.stop())
                return True
            
            # 计算方向
            target_angle = math.atan2(dy, dx)
            angle_diff = self._normalize_angle(target_angle - yaw)
            
            # 慢速接近
            linear = min(op.linear_speed, (distance - stop_distance) * 0.5)
            linear = max(0.1, linear)
            angular = angle_diff * 0.3
            
            cmd = TwistCommand(linear_x=linear, angular_z=angular)
            await self.ros2.publish_twist(cmd)
            
            await asyncio.sleep(interval)
        
        await self.ros2.publish_twist(TwistCommand.stop())
        return False
    
    async def _execute_bypass(self, op: UGVOperation) -> bool:
        """执行绕行操作"""
        is_left = op.ugv_type == UGVOperationType.BYPASS_LEFT
        
        # 1. 原地转向
        rotate_op = self.rotate_left(math.pi / 2) if is_left else self.rotate_right(math.pi / 2)
        await self._execute_rotate(rotate_op)
        
        # 2. 侧向移动
        move_op = self.move_forward(1.0, speed=0.3)
        await self._execute_move(move_op)
        
        # 3. 转回原方向
        rotate_back = self.rotate_right(math.pi / 2) if is_left else self.rotate_left(math.pi / 2)
        await self._execute_rotate(rotate_back)
        
        # 4. 前进超过障碍物
        await self._execute_move(self.move_forward(1.5, speed=0.3))
        
        return True
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    # === 控制方法 ===
    
    def abort(self):
        """中止当前操作"""
        self._abort_requested = True
        logger.info("请求中止当前操作")
    
    def is_executing(self) -> bool:
        """检查是否正在执行"""
        return self._is_executing
    
    def get_current_operation(self) -> Optional[UGVOperation]:
        """获取当前操作"""
        return self._current_operation


# === 便捷函数 ===

def create_operation_sequence(
    commands: List[Dict[str, Any]]
) -> List[UGVOperation]:
    """
    从命令列表创建操作序列
    
    Args:
        commands: 命令列表，每个命令是字典
            {"type": "move_forward", "distance": 2.0, "speed": 0.5}
    
    Returns:
        UGVOperation列表
    """
    operations = []
    
    op_map = {
        "move_forward": ROS2UGVOperations.move_forward,
        "move_backward": ROS2UGVOperations.move_backward,
        "turn_left": ROS2UGVOperations.turn_left,
        "turn_right": ROS2UGVOperations.turn_right,
        "rotate_left": ROS2UGVOperations.rotate_left,
        "rotate_right": ROS2UGVOperations.rotate_right,
        "stop": ROS2UGVOperations.stop,
        "navigate_to": ROS2UGVOperations.navigate_to,
        "explore": ROS2UGVOperations.explore_towards,
        "search": ROS2UGVOperations.search_target,
        "approach": ROS2UGVOperations.approach_target
    }
    
    for cmd in commands:
        op_type = cmd.get("type")
        if op_type in op_map:
            # 提取参数
            params = {k: v for k, v in cmd.items() if k != "type"}
            try:
                op = op_map[op_type](**params)
                operations.append(op)
            except Exception as e:
                logger.warning(f"创建操作失败: {op_type}, 错误: {e}")
    
    return operations

