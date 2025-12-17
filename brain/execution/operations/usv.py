"""
无人船操作集 - USV (Unmanned Surface Vehicle) Operations

定义无人船平台的原子操作
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


class USVOperations:
    """
    无人船操作集
    
    提供无人船平台的标准操作
    """
    
    PLATFORM = "usv"
    
    # ==================== 航行操作 ====================
    
    @classmethod
    def start_engine(cls) -> Operation:
        """启动引擎"""
        builder = (OperationBuilder("start_engine", cls.PLATFORM)
                   .of_type(OperationType.CONTROL)
                   .with_duration(10.0)
                   .with_priority(OperationPriority.HIGH))
        
        builder.with_precondition(
            name="engine_off",
            condition="robot.engine.running == False",
            description="引擎必须关闭"
        )
        builder.with_precondition(
            name="fuel_ok",
            condition="robot.fuel > 15",
            description="燃料必须大于15%"
        )
        
        builder.with_postcondition(
            name="engine_running",
            expected_state="robot.engine.running == True",
            description="引擎应该运行"
        )
        
        return builder.build()
    
    @classmethod
    def stop_engine(cls) -> Operation:
        """停止引擎"""
        return (OperationBuilder("stop_engine", cls.PLATFORM)
                .of_type(OperationType.CONTROL)
                .with_duration(5.0)
                .with_postcondition(
                    name="engine_stopped",
                    expected_state="robot.engine.running == False",
                    description="引擎应该停止"
                )
                .build())
    
    @classmethod
    def navigate_to(
        cls,
        position: Dict[str, float],
        speed: Optional[float] = None,
        heading: Optional[float] = None
    ) -> Operation:
        """
        航行到指定位置
        
        Args:
            position: 目标位置 {lat, lon}
            speed: 航速(节)
            heading: 航向(度)
        """
        params = {"position": position}
        if speed:
            params["speed"] = speed
        if heading is not None:
            params["heading"] = heading
        
        builder = (OperationBuilder("navigate_to", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(120.0))
        
        builder.with_precondition(
            name="engine_running",
            condition="robot.engine.running == True",
            description="引擎必须运行"
        )
        builder.with_precondition(
            name="gps_available",
            condition="robot.gps.status == 'available'",
            description="GPS必须可用"
        )
        
        return builder.build()
    
    @classmethod
    def follow_route(
        cls,
        waypoints: List[Dict[str, float]],
        speed: Optional[float] = None
    ) -> Operation:
        """
        跟踪航线
        
        Args:
            waypoints: 航点列表
            speed: 航速(节)
        """
        params = {"waypoints": waypoints}
        if speed:
            params["speed"] = speed
        
        duration = len(waypoints) * 60  # 估算
        
        builder = (OperationBuilder("follow_route", cls.PLATFORM)
                   .of_type(OperationType.MOVEMENT)
                   .with_params(**params)
                   .with_duration(duration))
        
        builder.with_precondition(
            name="engine_running",
            condition="robot.engine.running == True",
            description="引擎必须运行"
        )
        
        return builder.build()
    
    @classmethod
    def hold_position(
        cls,
        duration: float,
        position: Optional[Dict[str, float]] = None
    ) -> Operation:
        """
        保持位置(锚泊/动力定位)
        
        Args:
            duration: 保持时长(秒)
            position: 位置(可选)
        """
        params = {"duration": duration}
        if position:
            params["position"] = position
        
        return (OperationBuilder("hold_position", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(duration)
                .build())
    
    @classmethod
    def patrol_area(
        cls,
        area: Dict[str, Any],
        pattern: str = "lawnmower",
        speed: Optional[float] = None
    ) -> Operation:
        """
        区域巡逻
        
        Args:
            area: 巡逻区域
            pattern: 巡逻模式 (lawnmower/spiral/random)
            speed: 航速
        """
        params = {"area": area, "pattern": pattern}
        if speed:
            params["speed"] = speed
        
        return (OperationBuilder("patrol_area", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(300.0)
                .build())
    
    @classmethod
    def return_to_port(cls, port_position: Optional[Dict[str, float]] = None) -> Operation:
        """
        返回港口
        
        Args:
            port_position: 港口位置(可选，默认使用起点)
        """
        params = {}
        if port_position:
            params["port_position"] = port_position
        
        return (OperationBuilder("return_to_port", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(180.0)
                .with_priority(OperationPriority.HIGH)
                .build())
    
    @classmethod
    def dock(cls, dock_position: Dict[str, float]) -> Operation:
        """
        靠泊
        
        Args:
            dock_position: 码头位置
        """
        return (OperationBuilder("dock", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_params(dock_position=dock_position)
                .with_duration(120.0)
                .with_priority(OperationPriority.HIGH)
                .with_postcondition(
                    name="docked",
                    expected_state="robot.state.docked == True",
                    description="船应靠泊"
                )
                .build())
    
    @classmethod
    def undock(cls) -> Operation:
        """离泊"""
        return (OperationBuilder("undock", cls.PLATFORM)
                .of_type(OperationType.MOVEMENT)
                .with_duration(60.0)
                .with_precondition(
                    name="docked",
                    condition="robot.state.docked == True",
                    description="船必须处于靠泊状态"
                )
                .with_postcondition(
                    name="undocked",
                    expected_state="robot.state.docked == False",
                    description="船应离泊"
                )
                .build())
    
    # ==================== 感知操作 ====================
    
    @classmethod
    def scan_surface(
        cls,
        range_m: float = 500.0,
        resolution: str = "medium"
    ) -> Operation:
        """
        水面扫描
        
        Args:
            range_m: 扫描范围(米)
            resolution: 分辨率
        """
        return (OperationBuilder("scan_surface", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(range=range_m, resolution=resolution)
                .with_duration(30.0)
                .build())
    
    @classmethod
    def detect_vessels(
        cls,
        range_m: float = 2000.0
    ) -> Operation:
        """
        检测船只
        
        Args:
            range_m: 检测范围(米)
        """
        return (OperationBuilder("detect_vessels", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(range=range_m)
                .with_duration(10.0)
                .build())
    
    @classmethod
    def measure_depth(cls) -> Operation:
        """测量水深"""
        return (OperationBuilder("measure_depth", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_duration(5.0)
                .build())
    
    @classmethod
    def capture_image(
        cls,
        direction: str = "forward",
        zoom: float = 1.0
    ) -> Operation:
        """
        拍照
        
        Args:
            direction: 拍摄方向
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
        direction: str = "forward"
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
    def sonar_scan(
        cls,
        mode: str = "forward",
        range_m: float = 100.0
    ) -> Operation:
        """
        声纳扫描
        
        Args:
            mode: 扫描模式 (forward/side/down)
            range_m: 扫描范围
        """
        return (OperationBuilder("sonar_scan", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_params(mode=mode, range=range_m)
                .with_duration(20.0)
                .build())
    
    @classmethod
    def weather_check(cls) -> Operation:
        """检查天气"""
        return (OperationBuilder("weather_check", cls.PLATFORM)
                .of_type(OperationType.PERCEPTION)
                .with_duration(5.0)
                .build())
    
    # ==================== 任务操作 ====================
    
    @classmethod
    def collect_sample(
        cls,
        sample_type: str,
        depth: Optional[float] = None
    ) -> Operation:
        """
        采集样本
        
        Args:
            sample_type: 样本类型 (water/sediment/biological)
            depth: 采样深度(米)
        """
        params = {"sample_type": sample_type}
        if depth:
            params["depth"] = depth
        
        return (OperationBuilder("collect_sample", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(**params)
                .with_duration(60.0)
                .build())
    
    @classmethod
    def deploy_buoy(
        cls,
        position: Dict[str, float],
        buoy_type: str = "marker"
    ) -> Operation:
        """
        投放浮标
        
        Args:
            position: 投放位置
            buoy_type: 浮标类型
        """
        return (OperationBuilder("deploy_buoy", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(position=position, buoy_type=buoy_type)
                .with_duration(30.0)
                .build())
    
    @classmethod
    def retrieve_buoy(
        cls,
        buoy_id: str
    ) -> Operation:
        """
        回收浮标
        
        Args:
            buoy_id: 浮标ID
        """
        return (OperationBuilder("retrieve_buoy", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(buoy_id=buoy_id)
                .with_duration(60.0)
                .build())
    
    @classmethod
    def tow_object(
        cls,
        object_id: str,
        destination: Dict[str, float]
    ) -> Operation:
        """
        拖曳物体
        
        Args:
            object_id: 物体ID
            destination: 目的地
        """
        return (OperationBuilder("tow_object", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(object_id=object_id, destination=destination)
                .with_duration(300.0)
                .build())
    
    @classmethod
    def launch_drone(cls, drone_id: str) -> Operation:
        """
        释放无人机
        
        Args:
            drone_id: 无人机ID
        """
        return (OperationBuilder("launch_drone", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(drone_id=drone_id)
                .with_duration(30.0)
                .build())
    
    @classmethod
    def recover_drone(cls, drone_id: str) -> Operation:
        """
        回收无人机
        
        Args:
            drone_id: 无人机ID
        """
        return (OperationBuilder("recover_drone", cls.PLATFORM)
                .of_type(OperationType.MANIPULATION)
                .with_params(drone_id=drone_id)
                .with_duration(60.0)
                .build())
    
    # ==================== 安全操作 ====================
    
    @classmethod
    def emergency_stop(cls) -> Operation:
        """紧急停止"""
        return (OperationBuilder("emergency_stop", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(5.0)
                .build())
    
    @classmethod
    def collision_avoidance(
        cls,
        vessel_position: Dict[str, float],
        safe_distance: float = 100.0
    ) -> Operation:
        """
        碰撞规避
        
        Args:
            vessel_position: 船只位置
            safe_distance: 安全距离(米)
        """
        return (OperationBuilder("collision_avoidance", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_params(
                    vessel_position=vessel_position,
                    safe_distance=safe_distance
                )
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(30.0)
                .build())
    
    @classmethod
    def man_overboard_response(
        cls,
        last_known_position: Dict[str, float]
    ) -> Operation:
        """
        落水人员响应
        
        Args:
            last_known_position: 最后已知位置
        """
        return (OperationBuilder("man_overboard_response", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_params(last_known_position=last_known_position)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(60.0)
                .build())
    
    @classmethod
    def anchor_deploy(cls) -> Operation:
        """下锚"""
        return (OperationBuilder("anchor_deploy", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_duration(30.0)
                .with_postcondition(
                    name="anchored",
                    expected_state="robot.state.anchored == True",
                    description="船应抛锚"
                )
                .build())
    
    @classmethod
    def anchor_retrieve(cls) -> Operation:
        """起锚"""
        return (OperationBuilder("anchor_retrieve", cls.PLATFORM)
                .of_type(OperationType.SAFETY)
                .with_duration(45.0)
                .with_precondition(
                    name="anchored",
                    condition="robot.state.anchored == True",
                    description="船必须已抛锚"
                )
                .build())
    
    # ==================== 通信操作 ====================
    
    @classmethod
    def send_ais(cls) -> Operation:
        """发送AIS信息"""
        return (OperationBuilder("send_ais", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_duration(1.0)
                .build())
    
    @classmethod
    def broadcast_distress(
        cls,
        reason: str
    ) -> Operation:
        """
        发送求救信号
        
        Args:
            reason: 原因
        """
        return (OperationBuilder("broadcast_distress", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_params(reason=reason)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(5.0)
                .build())
    
    @classmethod
    def report_position(cls) -> Operation:
        """报告位置"""
        return (OperationBuilder("report_position", cls.PLATFORM)
                .of_type(OperationType.COMMUNICATION)
                .with_duration(2.0)
                .build())

