"""
无人车操作集 - UGV (Unmanned Ground Vehicle) Operations

定义无人车平台的原子操作
"""

from typing import Dict, List, Any, Optional

from brain.execution.operations.base import (
    Operation,
    OperationType,
    OperationPriority,
    OperationBuilder,
    Precondition,
    Postcondition
)


class UGVOperations:
    """
    无人车操作集
    
    提供无人车平台的标准操作
    """
    
    PLATFORM = "ugv"
    
    # ==================== 移动操作 ====================
    
    @classmethod
    def start(cls) -> Operation:
        """启动车辆"""
        builder = (OperationBuilder("start", cls.PLATFORM)
                   .of_type(OperationType.CONTROL)
                   .with_duration(5.0)
                   .with_priority(OperationPriority.HIGH))
        
        builder.with_precondition(
            name="stopped",
            condition="robot.state.moving == False",
            description="车辆必须停止"
        )
        builder.with_precondition(
            name="battery_ok",
            condition="robot.battery > 15",
            description="电池电量必须大于15%"
        )
        
        builder.with_postcondition(
            name="ready",
            expected_state="robot.state.ready == True",
            description="车辆应准备就绪"
        )
        
        return builder.build()
    
    @classmethod
    def stop(cls) -> Operation:
        """停止车辆"""
        return (OperationBuilder("stop", cls.PLATFORM)
                .of_type(OperationType.CONTROL)
                .with_duration(3.0)
                .with_priority(OperationPriority.HIGH)
                .with_postcondition(
                    name="stopped",
                    expected_state="robot.state.moving == False",
                    description="车辆应停止"
                )
                .build())
    
    @classmethod
    def goto(
        cls,
        position: Dict[str, float],
        speed: Optional[float] = None,
        path_type: str = "optimal"
    ) -> Operation:
        """
        前往指定位置
        
        Args:
            position: 目标位置 {lat, lon} 或 {x, y}
            speed: 行驶速度(米/秒)
            path_type: 路径类型 (optimal/shortest/safest)
        """
        params = {"position": position, "path_type": path_type}
        if speed:
            params["speed"] = speed
        
        builder = (OperationBuilder("goto", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(60.0))
        
        builder.with_precondition(
            name="ready",
            condition="robot.state.ready == True",
            description="车辆必须准备就绪"
        )
        builder.with_precondition(
            name="gps_available",
            condition="robot.gps.status == 'available'",
            description="GPS必须可用"
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
            speed: 行驶速度(米/秒)
            loop: 是否循环
        """
        params = {"waypoints": waypoints, "loop": loop}
        if speed:
            params["speed"] = speed
        
        duration = len(waypoints) * 45  # 估算
        
        builder = (OperationBuilder("follow_path", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(duration))
        
        builder.with_precondition(
            name="ready",
            condition="robot.state.ready == True",
            description="车辆必须准备就绪"
        )
        
        return builder.build()
    
    @classmethod
    def follow_road(
        cls,
        destination: Dict[str, float],
        speed: Optional[float] = None
    ) -> Operation:
        """
        沿道路行驶
        
        Args:
            destination: 目的地
            speed: 行驶速度
        """
        params = {"destination": destination}
        if speed:
            params["speed"] = speed
        
        return (OperationBuilder("follow_road", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(120.0)
                .with_precondition(
                    name="on_road",
                    condition="robot.state.on_road == True",
                    description="车辆必须在道路上"
                )
                .build())
    
    @classmethod
    def turn(
        cls,
        angle: float,
        direction: str = "auto"
    ) -> Operation:
        """
        转向
        
        Args:
            angle: 转向角度(度)
            direction: 方向 (left/right/auto)
        """
        return (OperationBuilder("turn", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(angle=angle, direction=direction)
                .with_duration(5.0)
                .build())
    
    @classmethod
    def reverse(
        cls,
        distance: float,
        speed: Optional[float] = None
    ) -> Operation:
        """
        倒车
        
        Args:
            distance: 倒车距离(米)
            speed: 倒车速度
        """
        params = {"distance": distance}
        if speed:
            params["speed"] = speed
        
        return (OperationBuilder("reverse", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(distance / 0.5 + 5)  # 估算
                .build())
    
    @classmethod
    def park(cls, position: Optional[Dict[str, float]] = None) -> Operation:
        """
        停车
        
        Args:
            position: 停车位置(可选)
        """
        params = {}
        if position:
            params["position"] = position
        
        return (OperationBuilder("park", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(15.0)
                .with_postcondition(
                    name="parked",
                    expected_state="robot.state.parked == True",
                    description="车辆应停好"
                )
                .build())
    
    @classmethod
    def return_to_home(cls) -> Operation:
        """返回起点"""
        return (OperationBuilder("return_to_home", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_duration(120.0)
                .with_priority(OperationPriority.HIGH)
                .with_precondition(
                    name="home_set",
                    condition="robot.home_position != None",
                    description="必须设置起点"
                )
                .build())
    
    # ==================== 感知操作 ====================
    
    @classmethod
    def scan_surroundings(
        cls,
        range_m: float = 30.0,
        resolution: str = "medium"
    ) -> Operation:
        """
        扫描周围环境
        
        Args:
            range_m: 扫描范围(米)
            resolution: 分辨率
        """
        return (OperationBuilder("scan_surroundings", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(range=range_m, resolution=resolution)
                .with_duration(10.0)
                .build())
    
    @classmethod
    def capture_image(
        cls,
        direction: str = "front",
        zoom: float = 1.0
    ) -> Operation:
        """
        拍照
        
        Args:
            direction: 拍摄方向 (front/left/right/rear/all)
            zoom: 变焦倍数
        """
        return (OperationBuilder("capture_image", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(direction=direction, zoom=zoom)
                .with_duration(2.0)
                .build())
    
    @classmethod
    def record_video(
        cls,
        duration: float,
        direction: str = "front"
    ) -> Operation:
        """
        录像
        
        Args:
            duration: 录制时长(秒)
            direction: 录制方向
        """
        return (OperationBuilder("record_video", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(duration=duration, direction=direction)
                .with_duration(duration + 2)
                .build())
    
    @classmethod
    def detect_obstacles(
        cls,
        distance: float = 20.0
    ) -> Operation:
        """
        障碍物检测
        
        Args:
            distance: 检测距离(米)
        """
        return (OperationBuilder("detect_obstacles", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(distance=distance)
                .with_duration(3.0)
                .build())
    
    @classmethod
    def detect_lane(cls) -> Operation:
        """车道检测"""
        return (OperationBuilder("detect_lane", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_duration(2.0)
                .build())
    
    @classmethod
    def read_sign(cls) -> Operation:
        """读取交通标志"""
        return (OperationBuilder("read_sign", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_duration(3.0)
                .build())
    
    # ==================== 任务操作 ====================
    
    @classmethod
    def load_cargo(
        cls,
        cargo_id: str,
        position: Optional[Dict[str, float]] = None
    ) -> Operation:
        """
        装载货物
        
        Args:
            cargo_id: 货物ID
            position: 装载位置
        """
        params = {"cargo_id": cargo_id}
        if position:
            params["position"] = position
        
        return (OperationBuilder("load_cargo", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(**params)
                .with_duration(60.0)
                .with_precondition(
                    name="cargo_bay_empty",
                    condition="robot.cargo_bay.empty == True",
                    description="货舱必须为空"
                )
                .build())
    
    @classmethod
    def unload_cargo(
        cls,
        position: Optional[Dict[str, float]] = None
    ) -> Operation:
        """
        卸载货物
        
        Args:
            position: 卸载位置
        """
        params = {}
        if position:
            params["position"] = position
        
        return (OperationBuilder("unload_cargo", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(**params)
                .with_duration(45.0)
                .with_precondition(
                    name="has_cargo",
                    condition="robot.cargo_bay.empty == False",
                    description="必须有货物"
                )
                .build())
    
    @classmethod
    def pickup_passenger(
        cls,
        pickup_point: Dict[str, float]
    ) -> Operation:
        """
        接乘客
        
        Args:
            pickup_point: 接客点
        """
        return (OperationBuilder("pickup_passenger", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(pickup_point=pickup_point)
                .with_duration(120.0)
                .build())
    
    @classmethod
    def dropoff_passenger(
        cls,
        dropoff_point: Dict[str, float]
    ) -> Operation:
        """
        送乘客
        
        Args:
            dropoff_point: 下客点
        """
        return (OperationBuilder("dropoff_passenger", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(dropoff_point=dropoff_point)
                .with_duration(60.0)
                .build())
    
    # ==================== 安全操作 ====================
    
    @classmethod
    def emergency_stop(cls) -> Operation:
        """紧急停止"""
        return (OperationBuilder("emergency_stop", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(2.0)
                .build())
    
    @classmethod
    def avoid_obstacle(
        cls,
        obstacle_position: Dict[str, float],
        safe_distance: float = 3.0
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
                .with_duration(15.0)
                .build())
    
    @classmethod
    def pull_over(cls) -> Operation:
        """靠边停车"""
        return (OperationBuilder("pull_over", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_priority(OperationPriority.HIGH)
                .with_duration(20.0)
                .build())
    
    # ==================== 通信操作 ====================
    
    @classmethod
    def send_status(cls) -> Operation:
        """发送状态"""
        return (OperationBuilder("send_status", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_duration(1.0)
                .build())
    
    @classmethod
    def request_assistance(
        cls,
        reason: str
    ) -> Operation:
        """
        请求援助
        
        Args:
            reason: 原因
        """
        return (OperationBuilder("request_assistance", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_params(reason=reason)
                .with_priority(OperationPriority.HIGH)
                .with_duration(5.0)
                .build())

