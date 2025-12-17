"""
无人机操作集 - Drone Operations

定义无人机平台的原子操作
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from brain.execution.operations.base import (
    Operation,
    OperationType,
    OperationPriority,
    OperationBuilder,
    Precondition,
    Postcondition
)


@dataclass
class Position:
    """位置"""
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.lat, "lon": self.lon, "alt": self.alt}


class DroneOperations:
    """
    无人机操作集
    
    提供无人机平台的标准操作
    """
    
    PLATFORM = "drone"
    
    # ==================== 飞行操作 ====================
    
    @classmethod
    def takeoff(
        cls,
        altitude: float = 10.0,
        speed: Optional[float] = None
    ) -> Operation:
        """
        起飞操作
        
        Args:
            altitude: 目标高度(米)
            speed: 起飞速度(米/秒)
        """
        builder = (OperationBuilder("takeoff", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(altitude=altitude)
                   .with_duration(altitude / 2 + 5)  # 估算时间
                   .with_priority(OperationPriority.HIGH))
        
        if speed:
            builder.with_params(speed=speed)
        
        # 前置条件
        builder.with_precondition(
            name="on_ground",
            condition="robot.state.on_ground == True",
            description="无人机必须在地面上"
        )
        builder.with_precondition(
            name="motors_ready",
            condition="robot.motors.ready == True",
            description="电机必须就绪"
        )
        builder.with_precondition(
            name="battery_ok",
            condition="robot.battery > 20",
            description="电池电量必须大于20%"
        )
        
        # 后置条件
        builder.with_postcondition(
            name="airborne",
            expected_state="robot.state.airborne == True",
            description="无人机应在空中"
        )
        builder.with_postcondition(
            name="at_altitude",
            expected_state=f"robot.altitude >= {altitude * 0.9}",
            description=f"无人机应达到约{altitude}米高度"
        )
        
        return builder.build()
    
    @classmethod
    def land(
        cls,
        position: Optional[Dict[str, float]] = None,
        precision: bool = False
    ) -> Operation:
        """
        降落操作
        
        Args:
            position: 降落位置(可选)
            precision: 是否精确降落
        """
        params = {"precision": precision}
        if position:
            params["position"] = position
        
        builder = (OperationBuilder("land", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(20.0)
                   .with_priority(OperationPriority.HIGH))
        
        builder.with_precondition(
            name="airborne",
            condition="robot.state.airborne == True",
            description="无人机必须在空中"
        )
        
        builder.with_postcondition(
            name="on_ground",
            expected_state="robot.state.on_ground == True",
            description="无人机应在地面上"
        )
        
        return builder.build()
    
    @classmethod
    def goto(
        cls,
        position: Dict[str, float],
        speed: Optional[float] = None,
        heading: Optional[float] = None,
        altitude: Optional[float] = None
    ) -> Operation:
        """
        前往指定位置
        
        Args:
            position: 目标位置 {lat, lon} 或 {x, y, z}
            speed: 飞行速度(米/秒)
            heading: 航向角(度)
            altitude: 飞行高度(米)
        """
        params = {"position": position}
        if speed:
            params["speed"] = speed
        if heading is not None:
            params["heading"] = heading
        if altitude:
            params["altitude"] = altitude
        
        builder = (OperationBuilder("goto", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(30.0))  # 根据距离估算
        
        builder.with_precondition(
            name="airborne",
            condition="robot.state.airborne == True",
            description="无人机必须在空中"
        )
        builder.with_precondition(
            name="gps_available",
            condition="robot.gps.status == 'available'",
            description="GPS必须可用"
        )
        
        return builder.build()
    
    @classmethod
    def hover(
        cls,
        duration: float,
        position: Optional[Dict[str, float]] = None
    ) -> Operation:
        """
        悬停操作
        
        Args:
            duration: 悬停时长(秒)
            position: 悬停位置(可选)
        """
        params = {"duration": duration}
        if position:
            params["position"] = position
        
        builder = (OperationBuilder("hover", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(duration))
        
        builder.with_precondition(
            name="airborne",
            condition="robot.state.airborne == True",
            description="无人机必须在空中"
        )
        
        return builder.build()
    
    @classmethod
    def orbit(
        cls,
        center: Dict[str, float],
        radius: float,
        speed: float = 2.0,
        clockwise: bool = True,
        turns: float = 1.0
    ) -> Operation:
        """
        环绕飞行
        
        Args:
            center: 环绕中心点
            radius: 环绕半径(米)
            speed: 环绕速度(米/秒)
            clockwise: 是否顺时针
            turns: 圈数
        """
        # 计算预估时间: 周长 * 圈数 / 速度
        import math
        circumference = 2 * math.pi * radius
        duration = circumference * turns / speed
        
        builder = (OperationBuilder("orbit", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(
                       center=center,
                       radius=radius,
                       speed=speed,
                       clockwise=clockwise,
                       turns=turns
                   )
                   .with_duration(duration))
        
        builder.with_precondition(
            name="airborne",
            condition="robot.state.airborne == True",
            description="无人机必须在空中"
        )
        
        return builder.build()
    
    @classmethod
    def follow_path(
        cls,
        waypoints: List[Dict[str, float]],
        speed: Optional[float] = None,
        loop: bool = False
    ) -> Operation:
        """
        路径跟踪
        
        Args:
            waypoints: 航点列表
            speed: 飞行速度(米/秒)
            loop: 是否循环
        """
        params = {"waypoints": waypoints, "loop": loop}
        if speed:
            params["speed"] = speed
        
        # 估算时间
        duration = len(waypoints) * 30  # 简化估算
        
        builder = (OperationBuilder("follow_path", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(duration))
        
        builder.with_precondition(
            name="airborne",
            condition="robot.state.airborne == True",
            description="无人机必须在空中"
        )
        
        return builder.build()
    
    @classmethod
    def return_to_home(cls, altitude: Optional[float] = None) -> Operation:
        """
        返航
        
        Args:
            altitude: 返航高度(米)
        """
        params = {}
        if altitude:
            params["altitude"] = altitude
        
        builder = (OperationBuilder("return_to_home", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(60.0)
                   .with_priority(OperationPriority.HIGH))
        
        builder.with_precondition(
            name="home_set",
            condition="robot.home_position != None",
            description="必须设置起飞点"
        )
        
        return builder.build()
    
    # ==================== 感知操作 ====================
    
    @classmethod
    def capture_image(
        cls,
        target: Optional[Dict[str, float]] = None,
        zoom: float = 1.0,
        resolution: str = "high"
    ) -> Operation:
        """
        拍照
        
        Args:
            target: 拍摄目标位置
            zoom: 变焦倍数
            resolution: 分辨率 (low/medium/high)
        """
        params = {"zoom": zoom, "resolution": resolution}
        if target:
            params["target"] = target
        
        return (OperationBuilder("capture_image", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(**params)
                .with_duration(2.0)
                .build())
    
    @classmethod
    def record_video(
        cls,
        duration: float,
        quality: str = "1080p",
        target: Optional[Dict[str, float]] = None
    ) -> Operation:
        """
        录像
        
        Args:
            duration: 录制时长(秒)
            quality: 视频质量
            target: 拍摄目标
        """
        params = {"duration": duration, "quality": quality}
        if target:
            params["target"] = target
        
        return (OperationBuilder("record_video", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(**params)
                .with_duration(duration + 2)  # 加启停时间
                .build())
    
    @classmethod
    def scan_area(
        cls,
        area: Dict[str, Any],
        resolution: str = "medium",
        sensor: str = "camera"
    ) -> Operation:
        """
        区域扫描
        
        Args:
            area: 扫描区域定义
            resolution: 扫描分辨率
            sensor: 使用的传感器
        """
        return (OperationBuilder("scan_area", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(
                    area=area,
                    resolution=resolution,
                    sensor=sensor
                )
                .with_duration(60.0)
                .build())
    
    @classmethod
    def detect_objects(
        cls,
        object_types: List[str],
        area: Optional[Dict[str, Any]] = None
    ) -> Operation:
        """
        目标检测
        
        Args:
            object_types: 要检测的目标类型
            area: 检测区域
        """
        params = {"object_types": object_types}
        if area:
            params["area"] = area
        
        return (OperationBuilder("detect_objects", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(**params)
                .with_duration(5.0)
                .build())
    
    # ==================== 任务操作 ====================
    
    @classmethod
    def pickup(
        cls,
        object_id: str,
        approach_altitude: float = 5.0
    ) -> Operation:
        """
        拾取物体
        
        Args:
            object_id: 物体ID
            approach_altitude: 接近高度
        """
        return (OperationBuilder("pickup", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(
                    object_id=object_id,
                    approach_altitude=approach_altitude
                )
                .with_duration(30.0)
                .with_precondition(
                    name="gripper_ready",
                    condition="robot.gripper.ready == True",
                    description="抓取器必须就绪"
                )
                .build())
    
    @classmethod
    def dropoff(
        cls,
        position: Dict[str, float],
        release_altitude: float = 2.0
    ) -> Operation:
        """
        放下物体
        
        Args:
            position: 放置位置
            release_altitude: 释放高度
        """
        return (OperationBuilder("dropoff", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(
                    position=position,
                    release_altitude=release_altitude
                )
                .with_duration(20.0)
                .with_precondition(
                    name="carrying_payload",
                    condition="robot.payload != None",
                    description="必须携带有效载荷"
                )
                .build())
    
    @classmethod
    def spray(
        cls,
        area: Dict[str, Any],
        substance: str,
        amount: float
    ) -> Operation:
        """
        喷洒操作
        
        Args:
            area: 喷洒区域
            substance: 喷洒物质
            amount: 喷洒量
        """
        return (OperationBuilder("spray", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(
                    area=area,
                    substance=substance,
                    amount=amount
                )
                .with_duration(60.0)
                .with_precondition(
                    name="spray_tank_ok",
                    condition="robot.spray_tank.level > 10",
                    description="喷洒罐必须有足够液体"
                )
                .build())
    
    # ==================== 安全操作 ====================
    
    @classmethod
    def emergency_stop(cls) -> Operation:
        """紧急停止"""
        return (OperationBuilder("emergency_stop", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(1.0)
                .build())
    
    @classmethod
    def emergency_land(cls) -> Operation:
        """紧急降落"""
        return (OperationBuilder("emergency_land", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(30.0)
                .build())
    
    @classmethod
    def avoid_obstacle(
        cls,
        obstacle_position: Dict[str, float],
        safe_distance: float = 5.0
    ) -> Operation:
        """
        障碍物规避
        
        Args:
            obstacle_position: 障碍物位置
            safe_distance: 安全距离
        """
        return (OperationBuilder("avoid_obstacle", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_params(
                    obstacle_position=obstacle_position,
                    safe_distance=safe_distance
                )
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(10.0)
                .build())
    
    # ==================== 通信操作 ====================
    
    @classmethod
    def send_telemetry(cls) -> Operation:
        """发送遥测数据"""
        return (OperationBuilder("send_telemetry", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_duration(1.0)
                .build())
    
    @classmethod
    def upload_data(
        cls,
        data_type: str,
        destination: str
    ) -> Operation:
        """
        上传数据
        
        Args:
            data_type: 数据类型
            destination: 目标地址
        """
        return (OperationBuilder("upload_data", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_params(
                    data_type=data_type,
                    destination=destination
                )
                .with_duration(10.0)
                .build())

