"""
传感器管理器 - Sensor Manager

负责:
- 管理各类传感器
- 数据采集与融合
- 传感器状态监控
- 数据预处理
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import time
import threading
from loguru import logger

# 导入缓存管理器
try:
    from brain.utils.cache_manager import get_cache_manager, cached_async
    CACHE_AVAILABLE = True
except ImportError:
    logger.warning("缓存管理器不可用，传感器管理器将运行在无缓存模式")
    CACHE_AVAILABLE = False


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


class AsyncBatchedSensorManager:
    """
    异步批量传感器管理器

    提供高性能的传感器数据采集和批处理功能
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
        self.data_callbacks: List[Callable[[List[SensorData]], None]] = []

        # 高性能数据缓存
        self.data_buffer: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=100))
        self.buffer_lock = asyncio.Lock()

        # 批处理配置
        self.batch_size = self.config.get("batch_size", 10)
        self.batch_timeout = self.config.get("batch_timeout", 0.05)  # 50ms

        # 缓存系统
        self.cache_manager = None
        self.sensor_cache = None
        if CACHE_AVAILABLE:
            self.cache_manager = get_cache_manager()
            self.sensor_cache = self.cache_manager.create_cache(
                name="sensor_data",
                cache_type="ttl",
                max_size=5000,
                ttl_seconds=1.0  # 1秒TTL
            )

        # 异步任务池
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._running = False

        # 性能监控
        self.metrics = {
            "total_readings": 0,
            "successful_reads": 0,
            "failed_reads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_read_time": 0.0,
            "batch_efficiency": 0.0,
            "buffer_utilization": 0.0
        }

        # 读取时间历史
        self.read_times: deque = deque(maxlen=1000)

        logger.info("AsyncBatchedSensorManager 初始化完成 (高性能模式)")
    
    async def initialize(self):
        """初始化所有传感器"""
        sensors_config = self.config.get("sensors", {})

        initialization_tasks = []
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

                    # 并行初始化传感器
                    task = asyncio.create_task(
                        self._initialize_sensor(sensor_class, sensor_name, config)
                    )
                    initialization_tasks.append(task)

            except ValueError:
                logger.warning(f"未知传感器类型: {sensor_name}")

        # 等待所有传感器初始化完成
        results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

        successful_sensors = sum(1 for r in results if r is True)
        total_sensors = len(initialization_tasks)

        logger.info(f"传感器初始化完成: {successful_sensors}/{total_sensors} 成功")

        # 更新缓冲区利用率
        await self._update_buffer_utilization()

    async def _initialize_sensor(
        self, sensor_class, sensor_name: str, config: SensorConfig
    ) -> bool:
        """初始化单个传感器"""
        try:
            sensor = sensor_class(
                sensor_id=f"{sensor_name}_0",
                config=config
            )

            if await sensor.initialize():
                self.sensors[sensor.sensor_id] = sensor
                logger.info(f"传感器 {sensor.sensor_id} 初始化成功")
                return True
            else:
                logger.warning(f"传感器 {sensor_name} 初始化失败")
                return False

        except Exception as e:
            logger.error(f"传感器 {sensor_name} 初始化异常: {e}")
            return False

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
        """启动异步批量数据采集"""
        if self._running:
            return

        self._running = True

        # 启动传感器采集循环
        for sensor_id, sensor in self.sensors.items():
            if sensor.config.enabled:
                task = asyncio.create_task(
                    self._optimized_collection_loop(sensor)
                )
                self._collection_tasks[sensor_id] = task

        # 启动批处理器
        self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())

        logger.info("异步数据采集已启动")

    async def stop_collection(self):
        """停止数据采集"""
        self._running = False

        # 停止采集任务
        for task in self._collection_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # 停止批处理器
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        self._collection_tasks.clear()
        logger.info("异步数据采集已停止")

    async def _optimized_collection_loop(self, sensor: BaseSensor):
        """优化的传感器采集循环"""
        interval = 1.0 / sensor.config.update_rate
        consecutive_failures = 0
        max_failures = 5

        while self._running:
            start_time = time.time()

            try:
                # 检查缓存
                cache_key = f"{sensor.sensor_id}_{int(start_time)}"
                cached_data = None

                if self.sensor_cache:
                    cached_data = self.sensor_cache.get(cache_key)

                if cached_data is not None:
                    # 缓存命中
                    await self._add_to_buffer(cached_data)
                    self.metrics["cache_hits"] += 1
                else:
                    # 缓存未命中，读取传感器
                    data = await self._read_sensor_with_timeout(sensor, interval * 0.8)

                    if data:
                        self.metrics["cache_misses"] += 1

                        # 存入缓存
                        if self.sensor_cache:
                            self.sensor_cache.set(cache_key, data, ttl_seconds=0.5)

                        # 添加到缓冲区
                        await self._add_to_buffer(data)

                        self.metrics["successful_reads"] += 1
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        self.metrics["failed_reads"] += 1

                self.metrics["total_reads"] += 1

                # 更新读取时间统计
                read_time = time.time() - start_time
                self.read_times.append(read_time)
                self._update_average_read_time()

                # 自适应调整读取间隔
                if consecutive_failures > 0:
                    # 失败后增加延迟
                    await asyncio.sleep(interval * (1 + consecutive_failures * 0.2))
                else:
                    # 正常情况下精确控制间隔
                    elapsed = time.time() - start_time
                    if elapsed < interval:
                        await asyncio.sleep(interval - elapsed)

            except asyncio.TimeoutError:
                self.metrics["failed_reads"] += 1
                consecutive_failures += 1
                logger.warning(f"传感器 {sensor.sensor_id} 读取超时")

            except Exception as e:
                self.metrics["failed_reads"] += 1
                consecutive_failures += 1
                logger.error(f"传感器 {sensor.sensor_id} 采集异常: {e}")

                # 连续失败过多时暂时跳过
                if consecutive_failures >= max_failures:
                    await asyncio.sleep(interval * 5)

    async def _read_sensor_with_timeout(self, sensor: BaseSensor, timeout: float) -> Optional[SensorData]:
        """带超时的传感器读取"""
        return await asyncio.wait_for(sensor.read(), timeout=timeout)

    async def _add_to_buffer(self, data: SensorData):
        """添加数据到缓冲区"""
        async with self.buffer_lock:
            self.data_buffer[data.sensor_id].append(data)
            self.data_buffer["all"].append(data)

    async def _batch_processor_loop(self):
        """批处理循环"""
        last_batch_time = time.time()

        while self._running:
            try:
                current_time = time.time()
                time_since_last_batch = current_time - last_batch_time

                # 检查是否应该处理批次
                should_process = False
                batch_data = []

                async with self.buffer_lock:
                    # 检查"all"缓冲区
                    if len(self.data_buffer["all"]) >= self.batch_size:
                        should_process = True
                        # 取出批次数据
                        for _ in range(min(self.batch_size, len(self.data_buffer["all"]))):
                            batch_data.append(self.data_buffer["all"].popleft())

                    elif time_since_last_batch >= self.batch_timeout:
                        should_process = True
                        # 取出所有数据
                        while self.data_buffer["all"]:
                            batch_data.append(self.data_buffer["all"].popleft())

                if should_process and batch_data:
                    # 处理批次数据
                    start_time = time.time()
                    await self._process_batch(batch_data)
                    processing_time = time.time() - start_time

                    # 更新批处理效率
                    batch_size = len(batch_data)
                    if batch_size > 0:
                        efficiency = min(1.0, (self.batch_size / batch_size) * (self.batch_timeout / processing_time))
                        self.metrics["batch_efficiency"] = (
                            (self.metrics["batch_efficiency"] * 0.7 + efficiency * 0.3)
                        )

                    last_batch_time = current_time
                    logger.debug(f"处理批次: {batch_size} 项, 耗时: {processing_time:.3f}s")
                else:
                    await asyncio.sleep(0.01)  # 短暂休眠

            except Exception as e:
                logger.error(f"批处理循环异常: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch_data: List[SensorData]):
        """处理批次数据"""
        if not batch_data:
            return

        try:
            # 触发批量回调
            for callback in self.data_callbacks:
                try:
                    await callback(batch_data)
                except Exception as e:
                    logger.error(f"批量数据回调执行失败: {e}")

        except Exception as e:
            logger.error(f"批次数据处理失败: {e}")

    def on_data(self, callback: Callable[[List[SensorData]], None]):
        """注册批量数据回调"""
        self.data_callbacks.append(callback)

    async def get_current_data(self, batch_mode: bool = True) -> Dict[str, Any]:
        """获取当前传感器数据"""
        if batch_mode:
            # 返回缓冲区中的所有数据
            async with self.buffer_lock:
                result = {}
                for sensor_id, buffer in self.data_buffer.items():
                    if sensor_id == "all":
                        continue
                    if buffer:
                        result[sensor_id] = [
                            {
                                "timestamp": data.timestamp.isoformat(),
                                "data": data.data,
                                "quality": data.quality,
                                "metadata": data.metadata
                            }
                            for data in list(buffer)
                        ]
                return result
        else:
            # 返回最新的数据
            async with self.buffer_lock:
                result = {}
                for sensor_id, buffer in self.data_buffer.items():
                    if sensor_id == "all":
                        continue
                    if buffer:
                        latest = buffer[-1]
                        result[sensor_id] = {
                            "type": latest.sensor_type.value,
                            "timestamp": latest.timestamp.isoformat(),
                            "data": latest.data,
                            "quality": latest.quality,
                            "metadata": latest.metadata
                        }
                return result

    async def get_sensor_data(
        self,
        sensor_type: SensorType,
        use_cache: bool = True
    ) -> Optional[List[SensorData]]:
        """获取指定类型传感器的数据（支持缓存）"""
        if use_cache and self.sensor_cache:
            # 从缓存获取最近的多个数据点
            recent_data = []
            for i in range(min(10, len(self.read_times))):  # 获取最近10个数据点
                timestamp = time.time() - i * 0.1  # 假设100ms间隔
                cache_key = f"{sensor_type.value}_{int(timestamp * 1000000)}"
                cached_data = self.sensor_cache.get(cache_key)
                if cached_data and cached_data.sensor_type == sensor_type:
                    recent_data.append(cached_data)

            if recent_data:
                return recent_data

        # 从缓冲区获取
        async with self.buffer_lock:
            if sensor_type.value in self.data_buffer:
                return list(self.data_buffer[sensor_type.value])

        return None

    def get_sensor_status(self) -> Dict[str, Any]:
        """获取所有传感器状态"""
        return {
            sensor_id: sensor.get_status()
            for sensor_id, sensor in self.sensors.items()
        }

    async def _update_buffer_utilization(self):
        """更新缓冲区利用率"""
        total_capacity = len(self.data_buffer) * 100
        current_usage = sum(len(buffer) for buffer in self.data_buffer.values())
        self.metrics["buffer_utilization"] = current_usage / total_capacity

    def _update_average_read_time(self):
        """更新平均读取时间"""
        if self.read_times:
            self.metrics["average_read_time"] = sum(self.read_times) / len(self.read_times)

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        cache_hit_rate = 0.0
        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            cache_hit_rate = self.metrics["cache_hits"] / (
                self.metrics["cache_hits"] + self.metrics["cache_misses"]
            )

        success_rate = 0.0
        if self.metrics["total_readings"] > 0:
            success_rate = self.metrics["successful_reads"] / self.metrics["total_readings"]

        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate,
            "active_sensors": len([s for s in self.metrics.values() if s.status in ["READY", "ACTIVE"]]),
            "total_sensors": len(self.sensors)
        }

    def get_buffer_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计"""
        return {
            sensor_id: {
                "size": len(buffer),
                "max_size": buffer.maxlen
            }
            for sensor_id, buffer in self.data_buffer.items()
        }

    async def clear_buffers(self):
        """清空所有缓冲区"""
        async with self.buffer_lock:
            for buffer in self.data_buffer.values():
                buffer.clear()
        logger.info("传感器数据缓冲区已清空")

    async def shutdown(self):
        """关闭所有传感器"""
        await self.stop_collection()

        # 关闭所有传感器
        shutdown_tasks = [
            asyncio.create_task(sensor.shutdown())
            for sensor in self.sensors.values()
        ]

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.sensors.clear()
        await self.clear_buffers()

        logger.info("AsyncBatchedSensorManager 已关闭")


# 向后兼容的SensorManager
class SensorManager(AsyncBatchedSensorManager):
    """
    向后兼容的传感器管理器
    """
    pass  # 继承自AsyncBatchedSensorManager
