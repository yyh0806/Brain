"""
控制适配器 - Control Adapter

负责:
- 抽象不同平台的控制接口（Ackermann/Differential）
- 统一的速度/转角控制接口
- 根据平台能力生成对应的Twist命令
"""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from brain.communication.ros2_interface import TwistCommand, ROS2Interface


class PlatformType(Enum):
    """平台类型"""
    ACKERMANN = "ackermann"      # 阿克曼转向（如car3）
    DIFFERENTIAL = "differential"  # 差速驱动（如car0）


@dataclass
class PlatformCapabilities:
    """平台能力"""
    platform_type: PlatformType
    max_linear_speed: float = 1.0    # m/s
    max_angular_speed: float = 1.0  # rad/s
    max_acceleration: float = 0.5    # m/s²
    min_turn_radius: float = 0.0    # m (0 = 可原地转向)
    wheelbase: float = 0.0          # m (Ackermann用)
    track_width: float = 0.0        # m (Differential用)


class ControlAdapter:
    """
    控制适配器
    
    提供统一的控制接口，适配不同平台类型
    """
    
    def __init__(
        self,
        ros2_interface: ROS2Interface,
        platform_type: PlatformType,
        capabilities: Optional[PlatformCapabilities] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.ros2 = ros2_interface
        self.platform_type = platform_type
        self.config = config or {}
        
        # 平台能力
        if capabilities:
            self.capabilities = capabilities
        else:
            # 从配置创建默认能力
            self.capabilities = PlatformCapabilities(
                platform_type=platform_type,
                max_linear_speed=self.config.get("max_linear_speed", 1.0),
                max_angular_speed=self.config.get("max_angular_speed", 1.0),
                max_acceleration=self.config.get("max_acceleration", 0.5),
                min_turn_radius=self.config.get("min_turn_radius", 0.0),
                wheelbase=self.config.get("wheelbase", 2.0),
                track_width=self.config.get("track_width", 1.0)
            )
        
        logger.info(f"ControlAdapter 初始化: {platform_type.value}")
    
    async def set_velocity(
        self,
        linear: float,
        angular: float
    ):
        """
        设置速度（统一接口）
        
        Args:
            linear: 线速度 (m/s)
            angular: 角速度 (rad/s)
        """
        # 限制速度
        linear = max(-self.capabilities.max_linear_speed,
                    min(linear, self.capabilities.max_linear_speed))
        angular = max(-self.capabilities.max_angular_speed,
                     min(angular, self.capabilities.max_angular_speed))
        
        if self.platform_type == PlatformType.ACKERMANN:
            cmd = self._ackermann_to_twist(linear, angular)
        else:
            cmd = self._differential_to_twist(linear, angular)
        
        await self.ros2.publish_twist(cmd)
    
    async def set_velocity_continuous(
        self,
        linear: float,
        angular: float,
        duration: float = 0.0
    ):
        """
        持续设置速度（用于平滑执行）
        
        Args:
            linear: 线速度 (m/s)
            angular: 角速度 (rad/s)
            duration: 持续时间（秒），0表示持续直到被其他命令覆盖
        """
        await self.set_velocity(linear, angular)
    
    async def move_forward(
        self,
        speed: float = 0.5
    ):
        """前进"""
        await self.set_velocity(speed, 0.0)
    
    async def move_backward(
        self,
        speed: float = 0.3
    ):
        """后退"""
        await self.set_velocity(-speed, 0.0)
    
    async def turn_left(
        self,
        linear_speed: float = 0.3,
        angular_speed: float = 0.5
    ):
        """左转（前进+左转）"""
        await self.set_velocity(linear_speed, angular_speed)
    
    async def turn_right(
        self,
        linear_speed: float = 0.3,
        angular_speed: float = -0.5
    ):
        """右转（前进+右转）"""
        await self.set_velocity(linear_speed, -angular_speed)
    
    async def rotate_left(
        self,
        angular_speed: float = 0.5
    ):
        """原地左转"""
        await self.set_velocity(0.0, angular_speed)
    
    async def rotate_right(
        self,
        angular_speed: float = 0.5
    ):
        """原地右转"""
        await self.set_velocity(0.0, -angular_speed)
    
    async def stop(self):
        """停止"""
        await self.set_velocity(0.0, 0.0)
    
    def _ackermann_to_twist(
        self,
        linear: float,
        angular: float
    ) -> TwistCommand:
        """
        Ackermann平台：角速度直接对应转向角速度
        
        对于Ackermann模型，Twist的angular.z就是转向角速度
        """
        return TwistCommand(
            linear_x=linear,
            linear_y=0.0,
            linear_z=0.0,
            angular_x=0.0,
            angular_y=0.0,
            angular_z=angular
        )
    
    def _differential_to_twist(
        self,
        linear: float,
        angular: float
    ) -> TwistCommand:
        """
        Differential平台：角速度对应左右轮差速
        
        对于差速驱动，Twist的angular.z就是角速度
        """
        return TwistCommand(
            linear_x=linear,
            linear_y=0.0,
            linear_z=0.0,
            angular_x=0.0,
            angular_y=0.0,
            angular_z=angular
        )
    
    def compute_turn_radius(
        self,
        linear: float,
        angular: float
    ) -> float:
        """
        计算转弯半径
        
        Returns:
            转弯半径（米），如果直线运动则返回inf
        """
        if abs(angular) < 1e-6:
            return float('inf')
        
        if self.platform_type == PlatformType.ACKERMANN:
            # Ackermann: R = v / w
            return abs(linear / angular) if angular != 0 else float('inf')
        else:
            # Differential: R = v / w
            return abs(linear / angular) if angular != 0 else float('inf')
    
    def compute_turn_angle_for_distance(
        self,
        distance: float,
        turn_radius: float
    ) -> float:
        """
        计算转过指定距离需要的角度
        
        Args:
            distance: 弧长（米）
            turn_radius: 转弯半径（米）
            
        Returns:
            角度（弧度）
        """
        if turn_radius == 0 or turn_radius == float('inf'):
            return 0.0
        
        return distance / turn_radius
    
    def plan_turn_at_intersection(
        self,
        turn_direction: str,  # "left", "right", "straight"
        approach_distance: float = 3.0,  # 接近路口的距离
        turn_radius: float = 2.0,  # 转弯半径
        exit_distance: float = 3.0  # 转弯后前进距离
    ) -> List[Tuple[float, float]]:
        """
        规划路口转弯路径
        
        Returns:
            [(linear_speed, angular_speed), ...] 速度序列
        """
        plan = []
        
        if turn_direction == "straight":
            # 直行
            plan.append((0.5, 0.0))  # 前进
        elif turn_direction == "left":
            # 左转
            # 1. 接近路口
            plan.append((0.5, 0.0))
            # 2. 开始转弯（根据转弯半径计算角速度）
            linear = 0.3
            angular = linear / turn_radius  # 正角速度
            plan.append((linear, angular))
            # 3. 转弯后直行
            plan.append((0.5, 0.0))
        elif turn_direction == "right":
            # 右转
            # 1. 接近路口
            plan.append((0.5, 0.0))
            # 2. 开始转弯
            linear = 0.3
            angular = -linear / turn_radius  # 负角速度
            plan.append((linear, angular))
            # 3. 转弯后直行
            plan.append((0.5, 0.0))
        
        return plan
    
    def get_capabilities(self) -> PlatformCapabilities:
        """获取平台能力"""
        return self.capabilities

