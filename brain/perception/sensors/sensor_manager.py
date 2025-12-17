"""
传感器管理器 - Sensor Manager

负责:
- 管理各类传感器
- 数据采集与融合
- 传感器状态监控
- 数据预处理
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from loguru import logger


class SensorType(Enum):
    """传感器类型"""
    CAMERA = "camera"
    LIDAR = "lidar"
    GPS = "gps"
    IMU = "imu"
    ULTRASONIC = "ultrasonic"
    RADAR = "radar"
    DEPTH_CAMERA = "depth_camera"
    THERMAL = "thermal"
    BAROMETER = "barometer"
    COMPASS = "compass"


class SensorStatus(Enum):
    """传感器状态"""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class SensorData:
    """传感器数据"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: float = 1.0  # 数据质量 0-1


@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_type: SensorType
    enabled: bool = True
    update_rate: float = 10.0  # Hz
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseSensor(ABC):
    """传感器基类"""
    
    def __init__(
        self, 
        sensor_id: str,
        sensor_type: SensorType,
        config: SensorConfig
    ):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config
        self.status = SensorStatus.UNKNOWN
        self.last_data: Optional[SensorData] = None
        self.error_count = 0
        
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化传感器"""
        pass
    
    @abstractmethod
    async def read(self) -> Optional[SensorData]:
        """读取传感器数据"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """关闭传感器"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取传感器状态"""
        return {
            "sensor_id": self.sensor_id,
            "type": self.sensor_type.value,
            "status": self.status.value,
            "error_count": self.error_count,
            "last_update": (
                self.last_data.timestamp.isoformat()
                if self.last_data else None
            )
        }


class CameraSensor(BaseSensor):
    """相机传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, SensorType.CAMERA, config)
        self.resolution = config.parameters.get("resolution", [1920, 1080])
        self.fps = config.parameters.get("fps", 30)
        
    async def initialize(self) -> bool:
        self.status = SensorStatus.INITIALIZING
        try:
            # 这里应该连接实际的相机设备
            # 示例中模拟初始化
            await asyncio.sleep(0.1)
            self.status = SensorStatus.READY
            logger.info(f"相机 [{self.sensor_id}] 初始化完成")
            return True
        except Exception as e:
            self.status = SensorStatus.ERROR
            logger.error(f"相机初始化失败: {e}")
            return False
    
    async def read(self) -> Optional[SensorData]:
        if self.status not in [SensorStatus.READY, SensorStatus.ACTIVE]:
            return None
            
        self.status = SensorStatus.ACTIVE
        
        try:
            # 这里应该从实际相机读取图像
            # 示例中返回模拟数据
            data = SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(),
                data={
                    "frame_id": uuid.uuid4().hex[:8],
                    "resolution": self.resolution,
                    "image": None  # 实际应为图像数据
                },
                metadata={"fps": self.fps}
            )
            self.last_data = data
            return data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"相机读取失败: {e}")
            return None
    
    async def shutdown(self):
        self.status = SensorStatus.DISCONNECTED
        logger.info(f"相机 [{self.sensor_id}] 已关闭")


class LidarSensor(BaseSensor):
    """激光雷达传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, SensorType.LIDAR, config)
        self.range = config.parameters.get("range", 100.0)
        self.points_per_second = config.parameters.get("points_per_second", 300000)
        
    async def initialize(self) -> bool:
        self.status = SensorStatus.INITIALIZING
        try:
            await asyncio.sleep(0.2)
            self.status = SensorStatus.READY
            logger.info(f"激光雷达 [{self.sensor_id}] 初始化完成")
            return True
        except Exception as e:
            self.status = SensorStatus.ERROR
            logger.error(f"激光雷达初始化失败: {e}")
            return False
    
    async def read(self) -> Optional[SensorData]:
        if self.status not in [SensorStatus.READY, SensorStatus.ACTIVE]:
            return None
            
        self.status = SensorStatus.ACTIVE
        
        try:
            # 模拟点云数据
            data = SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(),
                data={
                    "point_cloud": [],  # 实际应为点云数据
                    "num_points": 0
                },
                metadata={
                    "range": self.range,
                    "points_per_second": self.points_per_second
                }
            )
            self.last_data = data
            return data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"激光雷达读取失败: {e}")
            return None
    
    async def shutdown(self):
        self.status = SensorStatus.DISCONNECTED
        logger.info(f"激光雷达 [{self.sensor_id}] 已关闭")


class GPSSensor(BaseSensor):
    """GPS传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, SensorType.GPS, config)
        self.accuracy = config.parameters.get("accuracy", 0.01)
        
    async def initialize(self) -> bool:
        self.status = SensorStatus.INITIALIZING
        try:
            await asyncio.sleep(0.5)  # GPS初始化通常较慢
            self.status = SensorStatus.READY
            logger.info(f"GPS [{self.sensor_id}] 初始化完成")
            return True
        except Exception as e:
            self.status = SensorStatus.ERROR
            logger.error(f"GPS初始化失败: {e}")
            return False
    
    async def read(self) -> Optional[SensorData]:
        if self.status not in [SensorStatus.READY, SensorStatus.ACTIVE]:
            return None
            
        self.status = SensorStatus.ACTIVE
        
        try:
            # 模拟GPS数据
            data = SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(),
                data={
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "altitude": 0.0,
                    "accuracy": self.accuracy,
                    "satellites": 12,
                    "fix_type": "3D"
                }
            )
            self.last_data = data
            return data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"GPS读取失败: {e}")
            return None
    
    async def shutdown(self):
        self.status = SensorStatus.DISCONNECTED
        logger.info(f"GPS [{self.sensor_id}] 已关闭")


class IMUSensor(BaseSensor):
    """惯性测量单元传感器"""
    
    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, SensorType.IMU, config)
        self.rate = config.parameters.get("rate", 100)
        
    async def initialize(self) -> bool:
        self.status = SensorStatus.INITIALIZING
        try:
            await asyncio.sleep(0.1)
            self.status = SensorStatus.READY
            logger.info(f"IMU [{self.sensor_id}] 初始化完成")
            return True
        except Exception as e:
            self.status = SensorStatus.ERROR
            logger.error(f"IMU初始化失败: {e}")
            return False
    
    async def read(self) -> Optional[SensorData]:
        if self.status not in [SensorStatus.READY, SensorStatus.ACTIVE]:
            return None
            
        self.status = SensorStatus.ACTIVE
        
        try:
            # 模拟IMU数据
            data = SensorData(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.now(),
                data={
                    "acceleration": {"x": 0.0, "y": 0.0, "z": -9.8},
                    "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "magnetometer": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
                }
            )
            self.last_data = data
            return data
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"IMU读取失败: {e}")
            return None
    
    async def shutdown(self):
        self.status = SensorStatus.DISCONNECTED
        logger.info(f"IMU [{self.sensor_id}] 已关闭")


class SensorManager:
    """
    传感器管理器
    
    统一管理所有传感器，提供数据采集和融合功能
    """
    
    SENSOR_CLASSES = {
        SensorType.CAMERA: CameraSensor,
        SensorType.LIDAR: LidarSensor,
        SensorType.GPS: GPSSensor,
        SensorType.IMU: IMUSensor,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.sensors: Dict[str, BaseSensor] = {}
        self.data_callbacks: List[Callable[[SensorData], None]] = []
        
        # 数据缓存
        self.latest_data: Dict[str, SensorData] = {}
        
        # 采集任务
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
        logger.info("SensorManager 初始化")
    
    async def initialize(self):
        """初始化所有传感器"""
        sensors_config = self.config.get("sensors", {})
        
        for sensor_name, sensor_cfg in sensors_config.items():
            if not sensor_cfg.get("enabled", True):
                continue
                
            try:
                sensor_type = SensorType(sensor_name)
                sensor_class = self.SENSOR_CLASSES.get(sensor_type)
                
                if sensor_class:
                    config = SensorConfig(
                        sensor_type=sensor_type,
                        enabled=True,
                        parameters=sensor_cfg
                    )
                    
                    sensor = sensor_class(
                        sensor_id=f"{sensor_name}_0",
                        config=config
                    )
                    
                    if await sensor.initialize():
                        self.sensors[sensor.sensor_id] = sensor
                        logger.info(f"传感器 {sensor.sensor_id} 注册成功")
                    else:
                        logger.warning(f"传感器 {sensor_name} 初始化失败")
                        
            except ValueError:
                logger.warning(f"未知传感器类型: {sensor_name}")
    
    def register_sensor(self, sensor: BaseSensor):
        """注册传感器"""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"传感器 {sensor.sensor_id} 已注册")
    
    def unregister_sensor(self, sensor_id: str):
        """注销传感器"""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            logger.info(f"传感器 {sensor_id} 已注销")
    
    async def start_collection(self):
        """开始数据采集"""
        if self._running:
            return
            
        self._running = True
        
        for sensor_id, sensor in self.sensors.items():
            if sensor.config.enabled:
                task = asyncio.create_task(
                    self._collection_loop(sensor)
                )
                self._collection_tasks[sensor_id] = task
        
        logger.info("数据采集已启动")
    
    async def stop_collection(self):
        """停止数据采集"""
        self._running = False
        
        for task in self._collection_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._collection_tasks.clear()
        logger.info("数据采集已停止")
    
    async def _collection_loop(self, sensor: BaseSensor):
        """单个传感器的采集循环"""
        interval = 1.0 / sensor.config.update_rate
        
        while self._running:
            try:
                data = await sensor.read()
                
                if data:
                    self.latest_data[sensor.sensor_id] = data
                    
                    # 触发回调
                    for callback in self.data_callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"数据回调执行失败: {e}")
                            
            except Exception as e:
                logger.error(f"传感器 {sensor.sensor_id} 采集异常: {e}")
            
            await asyncio.sleep(interval)
    
    def on_data(self, callback: Callable[[SensorData], None]):
        """注册数据回调"""
        self.data_callbacks.append(callback)
    
    async def get_current_data(self) -> Dict[str, Any]:
        """获取当前所有传感器数据"""
        result = {}
        
        for sensor_id, data in self.latest_data.items():
            result[sensor_id] = {
                "type": data.sensor_type.value,
                "timestamp": data.timestamp.isoformat(),
                "data": data.data,
                "quality": data.quality
            }
        
        return result
    
    async def get_sensor_data(
        self, 
        sensor_type: SensorType
    ) -> Optional[SensorData]:
        """获取指定类型传感器的数据"""
        for sensor_id, data in self.latest_data.items():
            if data.sensor_type == sensor_type:
                return data
        return None
    
    def get_sensor_status(self) -> Dict[str, Any]:
        """获取所有传感器状态"""
        return {
            sensor_id: sensor.get_status()
            for sensor_id, sensor in self.sensors.items()
        }
    
    async def shutdown(self):
        """关闭所有传感器"""
        await self.stop_collection()
        
        for sensor in self.sensors.values():
            await sensor.shutdown()
        
        self.sensors.clear()
        logger.info("SensorManager 已关闭")


@dataclass
class FusedSensorData:
    """融合后的传感器数据"""
    timestamp: datetime
    position: Dict[str, float]  # lat, lon, alt
    velocity: Dict[str, float]  # vx, vy, vz
    orientation: Dict[str, float]  # roll, pitch, yaw
    obstacles: List[Dict[str, Any]]
    confidence: float = 1.0


class SensorFusion:
    """
    传感器数据融合
    
    融合多个传感器的数据，提供更准确的状态估计
    """
    
    def __init__(self, sensor_manager: SensorManager):
        self.sensor_manager = sensor_manager
        
    async def fuse(self) -> FusedSensorData:
        """执行数据融合"""
        all_data = await self.sensor_manager.get_current_data()
        
        # GPS位置
        gps_data = None
        for sid, data in all_data.items():
            if data["type"] == "gps":
                gps_data = data["data"]
                break
        
        position = {
            "lat": gps_data.get("latitude", 0.0) if gps_data else 0.0,
            "lon": gps_data.get("longitude", 0.0) if gps_data else 0.0,
            "alt": gps_data.get("altitude", 0.0) if gps_data else 0.0
        }
        
        # IMU姿态
        imu_data = None
        for sid, data in all_data.items():
            if data["type"] == "imu":
                imu_data = data["data"]
                break
        
        orientation = imu_data.get("orientation", {}) if imu_data else {}
        
        # 速度估计 (简化)
        velocity = {"vx": 0.0, "vy": 0.0, "vz": 0.0}
        
        # 障碍物检测 (来自激光雷达)
        obstacles = []
        
        return FusedSensorData(
            timestamp=datetime.now(),
            position=position,
            velocity=velocity,
            orientation=orientation,
            obstacles=obstacles,
            confidence=0.9
        )

