"""
机器人能力模型 - Robot Capabilities

定义不同无人平台的能力、约束和特性
用于LLM任务规划时了解平台限制
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class PlatformType(Enum):
    """平台类型"""
    UGV = "ugv"       # 无人地面车
    DRONE = "drone"   # 无人机
    USV = "usv"       # 无人船
    ROBOT_ARM = "arm" # 机械臂


class SensorCapability(Enum):
    """传感器能力"""
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR_2D = "lidar_2d"
    LIDAR_3D = "lidar_3d"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    RADAR = "radar"
    THERMAL = "thermal"


class MotionCapability(Enum):
    """运动能力"""
    FORWARD = "forward"           # 前进
    BACKWARD = "backward"         # 后退
    STRAFE = "strafe"            # 侧移（横向移动）
    ROTATE = "rotate"            # 原地旋转
    CLIMB = "climb"              # 爬坡
    FLY = "fly"                  # 飞行
    HOVER = "hover"              # 悬停
    SWIM = "swim"                # 航行


@dataclass
class KinematicLimits:
    """运动学限制"""
    max_linear_speed: float = 1.0      # 最大线速度 m/s
    max_angular_speed: float = 1.0     # 最大角速度 rad/s
    max_acceleration: float = 0.5      # 最大加速度 m/s²
    max_climb_angle: float = 0.3       # 最大爬坡角度 rad
    min_turn_radius: float = 0.0       # 最小转弯半径 m (0=可原地转)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "max_linear_speed": self.max_linear_speed,
            "max_angular_speed": self.max_angular_speed,
            "max_acceleration": self.max_acceleration,
            "max_climb_angle": self.max_climb_angle,
            "min_turn_radius": self.min_turn_radius
        }


@dataclass
class PerceptionRange:
    """感知范围"""
    camera_range: float = 10.0        # 相机有效距离 m
    camera_fov: float = 1.57          # 相机视场角 rad (90度)
    lidar_range: float = 30.0         # 激光雷达范围 m
    lidar_fov: float = 6.28           # 激光雷达视场角 rad (360度)
    detection_confidence: float = 0.8  # 检测置信度阈值
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "camera_range": self.camera_range,
            "camera_fov": self.camera_fov,
            "lidar_range": self.lidar_range,
            "lidar_fov": self.lidar_fov,
            "detection_confidence": self.detection_confidence
        }


@dataclass
class SafetyConstraints:
    """安全约束"""
    min_obstacle_distance: float = 0.5   # 最小障碍物距离 m
    emergency_stop_distance: float = 0.3 # 紧急停止距离 m
    max_operation_time: float = 3600     # 最大运行时间 s
    low_battery_threshold: float = 0.2   # 低电量阈值 (20%)
    safe_speed_near_obstacle: float = 0.3 # 障碍物附近安全速度 m/s
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "min_obstacle_distance": self.min_obstacle_distance,
            "emergency_stop_distance": self.emergency_stop_distance,
            "max_operation_time": self.max_operation_time,
            "low_battery_threshold": self.low_battery_threshold,
            "safe_speed_near_obstacle": self.safe_speed_near_obstacle
        }


@dataclass
class RobotCapabilities:
    """
    机器人能力描述
    
    完整描述一个机器人平台的所有能力和约束
    """
    # 基本信息
    name: str = "robot"
    platform_type: PlatformType = PlatformType.UGV
    description: str = "通用机器人平台"
    
    # 传感器
    sensors: List[SensorCapability] = field(default_factory=list)
    
    # 运动能力
    motion_capabilities: List[MotionCapability] = field(default_factory=list)
    
    # 运动学限制
    kinematics: KinematicLimits = field(default_factory=KinematicLimits)
    
    # 感知范围
    perception: PerceptionRange = field(default_factory=PerceptionRange)
    
    # 安全约束
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    
    # 操控能力
    has_manipulator: bool = False
    manipulator_reach: float = 0.0  # 机械臂工作范围 m
    
    # 载荷能力
    max_payload: float = 0.0  # 最大载荷 kg
    
    # 自主能力
    can_explore: bool = True          # 能否自主探索
    can_follow_path: bool = True      # 能否跟随路径
    can_avoid_obstacles: bool = True  # 能否避障
    
    # ROS2话题配置
    ros2_topics: Dict[str, str] = field(default_factory=dict)
    
    def get_capabilities_prompt(self) -> str:
        """
        生成能力描述供LLM使用
        
        Returns:
            str: 结构化的能力描述
        """
        lines = [
            f"## 机器人平台: {self.name}",
            f"类型: {self.platform_type.value}",
            f"描述: {self.description}",
            "",
            "### 传感器能力:",
        ]
        
        for sensor in self.sensors:
            lines.append(f"  - {sensor.value}")
        
        lines.extend([
            "",
            "### 运动能力:",
        ])
        
        for motion in self.motion_capabilities:
            lines.append(f"  - {motion.value}")
        
        lines.extend([
            "",
            "### 运动学限制:",
            f"  - 最大线速度: {self.kinematics.max_linear_speed} m/s",
            f"  - 最大角速度: {self.kinematics.max_angular_speed} rad/s",
            f"  - 最大爬坡角度: {self.kinematics.max_climb_angle:.2f} rad",
            f"  - 最小转弯半径: {self.kinematics.min_turn_radius} m",
            "",
            "### 感知范围:",
            f"  - 相机有效距离: {self.perception.camera_range} m",
            f"  - 激光雷达范围: {self.perception.lidar_range} m",
            "",
            "### 安全约束:",
            f"  - 最小障碍物距离: {self.safety.min_obstacle_distance} m",
            f"  - 紧急停止距离: {self.safety.emergency_stop_distance} m",
        ])
        
        if self.has_manipulator:
            lines.append(f"  - 机械臂工作范围: {self.manipulator_reach} m")
        
        lines.extend([
            "",
            "### 自主能力:",
            f"  - 自主探索: {'是' if self.can_explore else '否'}",
            f"  - 路径跟踪: {'是' if self.can_follow_path else '否'}",
            f"  - 自动避障: {'是' if self.can_avoid_obstacles else '否'}",
        ])
        
        return "\n".join(lines)
    
    def get_available_actions(self) -> List[str]:
        """获取可用的动作列表"""
        actions = []
        
        if MotionCapability.FORWARD in self.motion_capabilities:
            actions.extend(["move_forward", "navigate_to"])
        if MotionCapability.BACKWARD in self.motion_capabilities:
            actions.append("move_backward")
        if MotionCapability.ROTATE in self.motion_capabilities:
            actions.extend(["turn_left", "turn_right", "rotate"])
        if MotionCapability.STRAFE in self.motion_capabilities:
            actions.extend(["strafe_left", "strafe_right"])
        if MotionCapability.FLY in self.motion_capabilities:
            actions.extend(["takeoff", "land", "fly_to"])
        if MotionCapability.HOVER in self.motion_capabilities:
            actions.append("hover")
        
        if self.has_manipulator:
            actions.extend(["grasp", "release", "move_arm"])
        
        if self.can_explore:
            actions.append("explore")
        
        actions.extend(["stop", "wait"])
        
        return actions
    
    def can_perform_action(self, action: str) -> bool:
        """检查是否能执行指定动作"""
        available = self.get_available_actions()
        return action in available
    
    def get_speed_for_situation(self, situation: str) -> float:
        """根据情况获取建议速度"""
        if situation == "normal":
            return self.kinematics.max_linear_speed * 0.7
        elif situation == "near_obstacle":
            return self.safety.safe_speed_near_obstacle
        elif situation == "exploring":
            return self.kinematics.max_linear_speed * 0.5
        elif situation == "approaching_target":
            return self.kinematics.max_linear_speed * 0.3
        else:
            return self.kinematics.max_linear_speed * 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "platform_type": self.platform_type.value,
            "description": self.description,
            "sensors": [s.value for s in self.sensors],
            "motion_capabilities": [m.value for m in self.motion_capabilities],
            "kinematics": self.kinematics.to_dict(),
            "perception": self.perception.to_dict(),
            "safety": self.safety.to_dict(),
            "has_manipulator": self.has_manipulator,
            "available_actions": self.get_available_actions()
        }


# === 预定义平台配置 ===

def create_ugv_capabilities(
    name: str = "UGV",
    cmd_vel_topic: str = "/car3/twist"
) -> RobotCapabilities:
    """创建UGV平台能力配置"""
    return RobotCapabilities(
        name=name,
        platform_type=PlatformType.UGV,
        description="四轮差速驱动无人地面车",
        sensors=[
            SensorCapability.RGB_CAMERA,
            SensorCapability.DEPTH_CAMERA,
            SensorCapability.LIDAR_2D,
            SensorCapability.IMU
        ],
        motion_capabilities=[
            MotionCapability.FORWARD,
            MotionCapability.BACKWARD,
            MotionCapability.ROTATE
        ],
        kinematics=KinematicLimits(
            max_linear_speed=1.0,
            max_angular_speed=1.0,
            max_acceleration=0.5,
            max_climb_angle=0.3,
            min_turn_radius=0.0
        ),
        perception=PerceptionRange(
            camera_range=10.0,
            camera_fov=1.57,
            lidar_range=30.0,
            lidar_fov=6.28
        ),
        safety=SafetyConstraints(
            min_obstacle_distance=0.5,
            emergency_stop_distance=0.3,
            safe_speed_near_obstacle=0.3
        ),
        can_explore=True,
        can_follow_path=True,
        can_avoid_obstacles=True,
        ros2_topics={
            "cmd_vel": cmd_vel_topic,
            "odom": "/odom",
            "scan": "/scan",
            "rgb": "/camera/rgb/image_raw",
            "depth": "/camera/depth/image_raw"
        }
    )


def create_drone_capabilities(name: str = "Drone") -> RobotCapabilities:
    """创建无人机平台能力配置"""
    return RobotCapabilities(
        name=name,
        platform_type=PlatformType.DRONE,
        description="四旋翼无人机",
        sensors=[
            SensorCapability.RGB_CAMERA,
            SensorCapability.DEPTH_CAMERA,
            SensorCapability.IMU,
            SensorCapability.GPS
        ],
        motion_capabilities=[
            MotionCapability.FORWARD,
            MotionCapability.BACKWARD,
            MotionCapability.STRAFE,
            MotionCapability.ROTATE,
            MotionCapability.FLY,
            MotionCapability.HOVER
        ],
        kinematics=KinematicLimits(
            max_linear_speed=5.0,
            max_angular_speed=2.0,
            max_acceleration=2.0,
            max_climb_angle=1.57,  # 可垂直
            min_turn_radius=0.0
        ),
        perception=PerceptionRange(
            camera_range=50.0,
            camera_fov=1.57,
            lidar_range=0.0,  # 无激光雷达
            lidar_fov=0.0
        ),
        safety=SafetyConstraints(
            min_obstacle_distance=2.0,
            emergency_stop_distance=1.0,
            max_operation_time=1800,  # 30分钟
            safe_speed_near_obstacle=1.0
        ),
        can_explore=True,
        can_follow_path=True,
        can_avoid_obstacles=True
    )


def create_usv_capabilities(name: str = "USV") -> RobotCapabilities:
    """创建无人船平台能力配置"""
    return RobotCapabilities(
        name=name,
        platform_type=PlatformType.USV,
        description="无人水面船",
        sensors=[
            SensorCapability.RGB_CAMERA,
            SensorCapability.LIDAR_2D,
            SensorCapability.IMU,
            SensorCapability.GPS,
            SensorCapability.RADAR
        ],
        motion_capabilities=[
            MotionCapability.FORWARD,
            MotionCapability.BACKWARD,
            MotionCapability.ROTATE,
            MotionCapability.SWIM
        ],
        kinematics=KinematicLimits(
            max_linear_speed=3.0,
            max_angular_speed=0.5,
            max_acceleration=0.3,
            max_climb_angle=0.0,  # 不能爬坡
            min_turn_radius=2.0   # 有转弯半径
        ),
        perception=PerceptionRange(
            camera_range=100.0,
            camera_fov=1.57,
            lidar_range=50.0,
            lidar_fov=6.28
        ),
        safety=SafetyConstraints(
            min_obstacle_distance=5.0,
            emergency_stop_distance=3.0,
            max_operation_time=7200,  # 2小时
            safe_speed_near_obstacle=0.5
        ),
        can_explore=True,
        can_follow_path=True,
        can_avoid_obstacles=True
    )

