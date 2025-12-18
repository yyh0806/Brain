"""
ROS2通信接口 - ROS2 Interface

负责:
- 管理ROS2节点生命周期
- 发布控制命令 (geometry_msgs/Twist)
- 订阅传感器数据 (Image, LaserScan, PointCloud2, Odometry, IMU等)
- 提供异步接口供Brain系统调用

支持两种运行模式:
1. 真实ROS2环境 - 需要安装rclpy
2. 模拟模式 - 用于测试，无需ROS2环境
"""

import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

# 导入命令队列系统
try:
    from .command_queue import CommandQueue, CommandPriority, CommandType
    COMMAND_QUEUE_AVAILABLE = True
except ImportError:
    logger.warning("命令队列系统不可用，将使用直接发布模式")
    COMMAND_QUEUE_AVAILABLE = False
    # 提供fallback定义
    class CommandPriority(Enum):
        NORMAL = "normal"
        HIGH = "high"
        LOW = "low"

    class CommandType(Enum):
        TWIST = "twist"
        STOP = "stop"

# 尝试导入ROS2，如果不可用则使用模拟模式
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image, CompressedImage, LaserScan, PointCloud2, Imu
    from nav_msgs.msg import Odometry, OccupancyGrid
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ROS2 (rclpy) 不可用，将使用模拟模式")


class ROS2Mode(Enum):
    """ROS2运行模式"""
    REAL = "real"           # 真实ROS2环境
    SIMULATION = "simulation"  # 模拟模式


@dataclass
class ROS2Config:
    """ROS2配置"""
    node_name: str = "brain_node"
    mode: ROS2Mode = ROS2Mode.SIMULATION
    
    # 话题配置
    topics: Dict[str, str] = field(default_factory=lambda: {
        "cmd_vel": "/car3/twist",
        "rgb_image": "/camera/rgb/image_raw",
        "depth_image": "/camera/depth/image_raw",
        "laser_scan": "/scan",
        "pointcloud": "/points",
        "odom": "/odom",
        "imu": "/imu/data",
        "map": "/map"
    })
    
    # QoS配置
    sensor_qos_depth: int = 10
    cmd_qos_depth: int = 10
    
    # 超时配置
    publish_timeout: float = 1.0
    subscribe_timeout: float = 5.0


@dataclass
class TwistCommand:
    """速度控制命令"""
    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "linear": {"x": self.linear_x, "y": self.linear_y, "z": self.linear_z},
            "angular": {"x": self.angular_x, "y": self.angular_y, "z": self.angular_z}
        }
    
    @classmethod
    def forward(cls, speed: float = 1.0) -> 'TwistCommand':
        """前进"""
        return cls(linear_x=speed)
    
    @classmethod
    def backward(cls, speed: float = 1.0) -> 'TwistCommand':
        """后退"""
        return cls(linear_x=-speed)
    
    @classmethod
    def turn_left(cls, linear_speed: float = 0.5, angular_speed: float = 1.0) -> 'TwistCommand':
        """左转"""
        return cls(linear_x=linear_speed, angular_z=angular_speed)
    
    @classmethod
    def turn_right(cls, linear_speed: float = 0.5, angular_speed: float = 1.0) -> 'TwistCommand':
        """右转"""
        return cls(linear_x=linear_speed, angular_z=-angular_speed)
    
    @classmethod
    def rotate_left(cls, angular_speed: float = 1.0) -> 'TwistCommand':
        """原地左旋转"""
        return cls(angular_z=angular_speed)
    
    @classmethod
    def rotate_right(cls, angular_speed: float = 1.0) -> 'TwistCommand':
        """原地右旋转"""
        return cls(angular_z=-angular_speed)
    
    @classmethod
    def stop(cls) -> 'TwistCommand':
        """停止"""
        return cls()


@dataclass
class SensorData:
    """传感器数据"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 图像数据
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    
    # 激光雷达
    laser_scan: Optional[Dict[str, Any]] = None  # ranges, angle_min, angle_max, etc.
    
    # 点云
    pointcloud: Optional[np.ndarray] = None  # Nx3 或 Nx4 数组
    
    # 里程计
    odometry: Optional[Dict[str, Any]] = None  # position, orientation, velocity
    
    # IMU
    imu: Optional[Dict[str, Any]] = None  # orientation, angular_velocity, linear_acceleration
    
    # 占据地图
    occupancy_grid: Optional[np.ndarray] = None
    map_info: Optional[Dict[str, Any]] = None  # resolution, width, height, origin
    
    def has_rgb(self) -> bool:
        return self.rgb_image is not None
    
    def has_depth(self) -> bool:
        return self.depth_image is not None
    
    def has_lidar(self) -> bool:
        return self.laser_scan is not None
    
    def has_odom(self) -> bool:
        return self.odometry is not None
    
    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """获取当前位置"""
        if self.odometry:
            pos = self.odometry.get("position", {})
            return (pos.get("x", 0), pos.get("y", 0), pos.get("z", 0))
        return None
    
    def get_orientation(self) -> Optional[Tuple[float, float, float, float]]:
        """获取当前姿态(四元数)"""
        if self.odometry:
            orient = self.odometry.get("orientation", {})
            return (orient.get("x", 0), orient.get("y", 0), 
                    orient.get("z", 0), orient.get("w", 1))
        return None


class ROS2Interface:
    """
    ROS2通信接口
    
    提供与ROS2系统的异步通信能力
    """
    
    def __init__(self, config: Optional[ROS2Config] = None):
        self.config = config or ROS2Config()

        # 确定运行模式
        if ROS2_AVAILABLE and self.config.mode == ROS2Mode.REAL:
            self.mode = ROS2Mode.REAL
        else:
            self.mode = ROS2Mode.SIMULATION

        # ROS2节点（仅在真实模式下使用）
        self._node: Optional[Any] = None
        self._executor: Optional[Any] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._executor_thread: Optional[threading.Thread] = None

        # 发布者和订阅者
        self._publishers: Dict[str, Any] = {}
        self._subscribers: Dict[str, Any] = {}

        # 最新传感器数据缓存
        self._sensor_data = SensorData()
        self._data_lock = threading.Lock()

        # 回调函数
        self._sensor_callbacks: Dict[str, List[Callable]] = {}

        # 运行状态
        self._running = False

        # 调试：回调计数
        self._callback_counts: Dict[str, int] = {}

        # 命令队列系统
        self.command_queue: Optional[CommandQueue] = None
        self.use_command_queue = COMMAND_QUEUE_AVAILABLE and self.config.get("use_command_queue", True)

        logger.info(f"ROS2Interface 初始化完成 (模式: {self.mode.value}, 命令队列: {self.use_command_queue})")
    
    async def initialize(self):
        """初始化ROS2接口"""
        if self.mode == ROS2Mode.REAL:
            await self._init_ros2()
            # 启动后台spin任务
            asyncio.create_task(self._spin_ros2_async())
        else:
            await self._init_simulation()

        # 初始化命令队列
        if self.use_command_queue:
            self.command_queue = CommandQueue(
                max_size=self.config.get("command_queue_size", 1000),
                batch_timeout=self.config.get("command_batch_timeout", 0.05),
                max_rate=self.config.get("command_max_rate", 30.0),
                enable_batching=self.config.get("enable_command_batching", True)
            )
            await self.command_queue.start()
            logger.info("命令队列已启动")

        self._running = True
        logger.info("ROS2接口初始化完成")
    
    async def _spin_ros2_async(self):
        """异步spin ROS2节点"""
        while self._running and rclpy.ok():
            if self._executor:
                self._executor.spin_once(timeout_sec=0.1)
            else:
                rclpy.spin_once(self._node, timeout_sec=0.1)
            await asyncio.sleep(0.01)  # 让出控制权
    
    async def _init_ros2(self):
        """初始化真实ROS2环境"""
        if not rclpy.ok():
            rclpy.init()
        
        self._node = rclpy.create_node(self.config.node_name)
        
        # 创建发布者
        self._publishers["cmd_vel"] = self._node.create_publisher(
            Twist,
            self.config.topics["cmd_vel"],
            self.config.cmd_qos_depth
        )
        
        # 创建订阅者 - 使用sensor_data QoS（适合图像数据）
        from rclpy.qos import qos_profile_sensor_data
        qos = qos_profile_sensor_data
        
        # RGB图像订阅 - 根据配置选择Image或CompressedImage
        # 默认尝试CompressedImage（更常见），如果失败再试Image
        use_compressed = self.config.topics.get("rgb_image_compressed", True)
        if use_compressed:
            try:
                self._subscribers["rgb_image"] = self._node.create_subscription(
                    CompressedImage,
                    self.config.topics["rgb_image"],
                    self._rgb_compressed_callback,
                    qos
                )
                logger.info(f"订阅压缩图像: {self.config.topics['rgb_image']}")
            except Exception as e:
                logger.warning(f"订阅CompressedImage失败，尝试Image: {e}")
                self._subscribers["rgb_image"] = self._node.create_subscription(
                    Image,
                    self.config.topics["rgb_image"],
                    self._rgb_callback,
                    qos
                )
        else:
            self._subscribers["rgb_image"] = self._node.create_subscription(
                Image,
                self.config.topics["rgb_image"],
                self._rgb_callback,
                qos
            )
        
        # 深度图像订阅
        self._subscribers["depth_image"] = self._node.create_subscription(
            Image,
            self.config.topics["depth_image"],
            self._depth_callback,
            qos
        )
        
        # 激光雷达订阅
        self._subscribers["laser_scan"] = self._node.create_subscription(
            LaserScan,
            self.config.topics["laser_scan"],
            self._laser_callback,
            qos
        )
        
        # 里程计订阅
        odom_topic = self.config.topics.get("odom", "/odom")
        if odom_topic:
            self._subscribers["odom"] = self._node.create_subscription(
                Odometry,
                odom_topic,
                self._odom_callback,
                qos
            )
            logger.info(f"订阅里程计话题: {odom_topic}")
        else:
            logger.warning("未配置里程计话题")
        
        # IMU订阅
        imu_topic = self.config.topics.get("imu")
        if imu_topic:
            self._subscribers["imu"] = self._node.create_subscription(
                Imu,
                imu_topic,
                self._imu_callback,
                qos
            )
        
        # 点云订阅
        pointcloud_topic = self.config.topics.get("pointcloud")
        if pointcloud_topic:
            self._subscribers["pointcloud"] = self._node.create_subscription(
                PointCloud2,
                pointcloud_topic,
                self._pointcloud_callback,
                qos
            )
            logger.info(f"订阅点云话题: {pointcloud_topic}")
        
        # 创建executor（spin将在异步任务中执行）
        from rclpy.executors import SingleThreadedExecutor
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        
        logger.info("ROS2节点和订阅者已创建")
    
    async def _init_simulation(self):
        """初始化模拟环境"""
        logger.info("使用模拟模式，无需ROS2环境")
        # 初始化模拟数据
        self._init_simulated_data()
    
    def _init_simulated_data(self):
        """初始化模拟传感器数据"""
        with self._data_lock:
            # 模拟RGB图像
            self._sensor_data.rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 模拟深度图像
            self._sensor_data.depth_image = np.ones((480, 640), dtype=np.float32) * 5.0
            
            # 模拟激光雷达
            self._sensor_data.laser_scan = {
                "ranges": [5.0] * 360,
                "angle_min": -np.pi,
                "angle_max": np.pi,
                "angle_increment": 2 * np.pi / 360,
                "range_min": 0.1,
                "range_max": 30.0
            }
            
            # 模拟里程计
            self._sensor_data.odometry = {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "linear_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}
            }
            
            # 模拟IMU
            self._sensor_data.imu = {
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81}
            }
    
    def _spin_ros2(self):
        """ROS2 spin线程"""
        while self._running and rclpy.ok():
            if self._executor:
                self._executor.spin_once(timeout_sec=0.1)
            else:
                rclpy.spin_once(self._node, timeout_sec=0.1)
    
    # === ROS2回调函数 ===
    
    def _rgb_callback(self, msg):
        """RGB图像回调（未压缩）"""
        with self._data_lock:
            # 将ROS Image消息转换为numpy数组
            self._sensor_data.rgb_image = self._image_msg_to_numpy(msg)
            self._sensor_data.timestamp = datetime.now()
        self._trigger_callbacks("rgb_image")
    
    def _rgb_compressed_callback(self, msg):
        """RGB压缩图像回调"""
        self._callback_counts["rgb_image"] = self._callback_counts.get("rgb_image", 0) + 1
        logger.debug(f"收到压缩图像消息 #{self._callback_counts['rgb_image']}, 数据大小: {len(msg.data)} bytes")
        try:
            import cv2
            # 解压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                # BGR转RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with self._data_lock:
                    self._sensor_data.rgb_image = img
                    self._sensor_data.timestamp = datetime.now()
                logger.debug(f"✓ 成功解码压缩图像: {img.shape}")
                self._trigger_callbacks("rgb_image")
            else:
                logger.warning("cv2.imdecode返回None")
        except ImportError:
            logger.error("需要安装opencv-python: pip install opencv-python")
        except Exception as e:
            logger.warning(f"压缩图像解码失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _depth_callback(self, msg):
        """深度图像回调"""
        with self._data_lock:
            self._sensor_data.depth_image = self._image_msg_to_numpy(msg, depth=True)
            self._sensor_data.timestamp = datetime.now()
        self._trigger_callbacks("depth_image")
    
    def _laser_callback(self, msg):
        """激光雷达回调"""
        with self._data_lock:
            self._sensor_data.laser_scan = {
                "ranges": list(msg.ranges),
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "range_min": msg.range_min,
                "range_max": msg.range_max
            }
            self._sensor_data.timestamp = datetime.now()
        self._trigger_callbacks("laser_scan")
    
    def _odom_callback(self, msg):
        """里程计回调"""
        with self._data_lock:
            self._sensor_data.odometry = {
                "position": {
                    "x": msg.pose.pose.position.x,
                    "y": msg.pose.pose.position.y,
                    "z": msg.pose.pose.position.z
                },
                "orientation": {
                    "x": msg.pose.pose.orientation.x,
                    "y": msg.pose.pose.orientation.y,
                    "z": msg.pose.pose.orientation.z,
                    "w": msg.pose.pose.orientation.w
                },
                "linear_velocity": {
                    "x": msg.twist.twist.linear.x,
                    "y": msg.twist.twist.linear.y,
                    "z": msg.twist.twist.linear.z
                },
                "angular_velocity": {
                    "x": msg.twist.twist.angular.x,
                    "y": msg.twist.twist.angular.y,
                    "z": msg.twist.twist.angular.z
                }
            }
            self._sensor_data.timestamp = datetime.now()
            self._callback_counts["odom"] = self._callback_counts.get("odom", 0) + 1
        self._trigger_callbacks("odom")
        logger.debug(f"收到里程计数据: pos=({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})")
    
    def _imu_callback(self, msg):
        """IMU回调"""
        with self._data_lock:
            self._sensor_data.imu = {
                "orientation": {
                    "x": msg.orientation.x,
                    "y": msg.orientation.y,
                    "z": msg.orientation.z,
                    "w": msg.orientation.w
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x,
                    "y": msg.angular_velocity.y,
                    "z": msg.angular_velocity.z
                },
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x,
                    "y": msg.linear_acceleration.y,
                    "z": msg.linear_acceleration.z
                }
            }
            self._sensor_data.timestamp = datetime.now()
        self._trigger_callbacks("imu")
    
    def _pointcloud_callback(self, msg):
        """点云回调"""
        try:
            # 将PointCloud2转换为numpy数组
            # 使用sensor_msgs_py库或手动解析
            import struct
            
            # 获取点云数据
            points = []
            point_step = msg.point_step
            data = msg.data
            
            # 解析每个点（假设格式为XYZ）
            for i in range(0, len(data), point_step):
                if i + point_step > len(data):
                    break
                point_data = data[i:i+point_step]
                
                # 提取XYZ（假设前12字节是XYZ float32）
                if len(point_data) >= 12:
                    x = struct.unpack('f', point_data[0:4])[0]
                    y = struct.unpack('f', point_data[4:8])[0]
                    z = struct.unpack('f', point_data[8:12])[0]
                    points.append([x, y, z])
            
            if points:
                with self._data_lock:
                    self._sensor_data.pointcloud = np.array(points, dtype=np.float32)
                    self._sensor_data.timestamp = datetime.now()
                    self._callback_counts["pointcloud"] = self._callback_counts.get("pointcloud", 0) + 1
                self._trigger_callbacks("pointcloud")
                logger.debug(f"收到点云数据: {len(points)} 个点")
        except Exception as e:
            logger.warning(f"点云数据解析异常: {e}")
    
    def _image_msg_to_numpy(self, msg, depth: bool = False) -> np.ndarray:
        """将ROS Image消息转换为numpy数组"""
        if depth:
            # 深度图像
            dtype = np.float32 if msg.encoding in ["32FC1"] else np.uint16
            data = np.frombuffer(msg.data, dtype=dtype)
            expected_size = msg.height * msg.width
            
            # 计算每个像素的字节数
            bytes_per_pixel = np.dtype(dtype).itemsize
            actual_pixels = len(msg.data) // bytes_per_pixel
            
            if actual_pixels != expected_size:
                # 尝试根据实际数据大小推断分辨率
                # 常见情况：深度图分辨率可能是RGB的一半
                if actual_pixels == expected_size // 2:
                    # 可能是宽度减半
                    inferred_width = msg.width // 2
                    if inferred_width > 0 and msg.height * inferred_width == actual_pixels:
                        logger.debug(f"深度图像分辨率推断: {msg.height}x{inferred_width} (原始声明: {msg.height}x{msg.width})")
                        return data.reshape(msg.height, inferred_width)
                    # 或高度减半
                    inferred_height = msg.height // 2
                    if inferred_height > 0 and inferred_height * msg.width == actual_pixels:
                        logger.debug(f"深度图像分辨率推断: {inferred_height}x{msg.width} (原始声明: {msg.height}x{msg.width})")
                        return data.reshape(inferred_height, msg.width)
                
                # 如果无法推断，尝试直接reshape实际像素数
                logger.debug(f"深度图像数据大小不匹配: 期望{expected_size}, 实际{actual_pixels}, 尝试自动推断分辨率")
                # 尝试常见的宽高比
                for h in [msg.height, msg.height//2]:
                    for w in [msg.width, msg.width//2]:
                        if h * w == actual_pixels and h > 0 and w > 0:
                            logger.info(f"使用推断分辨率: {h}x{w}")
                            return data.reshape(h, w)
                
                # 如果都不匹配，返回None
                logger.debug(f"无法推断深度图像分辨率，数据大小: {actual_pixels}, 声明: {msg.height}x{msg.width}")
                return None
            
            return data.reshape(msg.height, msg.width)
        else:
            # RGB图像
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    
    def _trigger_callbacks(self, sensor_type: str):
        """触发传感器回调"""
        callbacks = self._sensor_callbacks.get(sensor_type, [])
        for callback in callbacks:
            try:
                callback(self._sensor_data)
            except Exception as e:
                logger.error(f"传感器回调执行失败: {e}")
    
    # === 公共API ===
    
    async def publish_twist(self, cmd: TwistCommand, priority: CommandPriority = CommandPriority.NORMAL, use_queue: Optional[bool] = None):
        """
        发布速度控制命令（支持命令队列）

        Args:
            cmd: TwistCommand对象
            priority: 命令优先级
            use_queue: 是否使用命令队列，None表示使用默认配置
        """
        should_use_queue = use_queue if use_queue is not None else self.use_command_queue

        if should_use_queue and self.command_queue:
            # 使用命令队列
            command_id = await self.command_queue.enqueue_command(
                command=cmd,
                command_type=CommandType.MOVE,
                priority=priority
            )
            logger.debug(f"命令入队: {command_id} ({priority.name})")
            return command_id
        else:
            # 直接发布
            return await self._publish_twist_direct(cmd)

    async def _publish_twist_direct(self, cmd: TwistCommand):
        """直接发布速度控制命令"""
        if self.mode == ROS2Mode.REAL:
            twist_msg = Twist()
            # 显式转换为float，避免np.float/整型触发ROS类型校验错误
            twist_msg.linear.x = float(cmd.linear_x)
            twist_msg.linear.y = float(cmd.linear_y)
            twist_msg.linear.z = float(cmd.linear_z)
            twist_msg.angular.x = float(cmd.angular_x)
            twist_msg.angular.y = float(cmd.angular_y)
            twist_msg.angular.z = float(cmd.angular_z)

            self._publishers["cmd_vel"].publish(twist_msg)
            logger.debug(f"直接发布Twist命令: linear_x={cmd.linear_x}, angular_z={cmd.angular_z}")
        else:
            # 模拟模式：更新模拟的里程计
            await self._simulate_movement(cmd)
            logger.debug(f"模拟Twist命令: linear_x={cmd.linear_x}, angular_z={cmd.angular_z}")

    async def emergency_stop(self):
        """紧急停止（最高优先级）"""
        if self.command_queue:
            return await self.command_queue.enqueue_emergency_stop()
        else:
            return await self._publish_twist_direct(TwistCommand.stop())

    async def publish_movement(self, linear_x: float, angular_z: float = 0.0, priority: CommandPriority = CommandPriority.NORMAL):
        """发布运动命令（便捷方法）"""
        cmd = TwistCommand(linear_x=linear_x, angular_z=angular_z)
        return await self.publish_twist(cmd, priority)

    async def publish_turn(self, angular_z: float, linear_x: float = 0.0, priority: CommandPriority = CommandPriority.HIGH):
        """发布转向命令（便捷方法）"""
        cmd = TwistCommand(linear_x=linear_x, angular_z=angular_z)
        return await self.publish_twist(cmd, priority)
    
    async def _simulate_movement(self, cmd: TwistCommand, dt: float = 0.1):
        """模拟运动更新"""
        with self._data_lock:
            if self._sensor_data.odometry:
                odom = self._sensor_data.odometry
                
                # 简单的运动模型
                import math
                
                # 获取当前yaw角
                q = odom["orientation"]
                yaw = math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]),
                                 1 - 2 * (q["y"]**2 + q["z"]**2))
                
                # 更新位置
                dx = cmd.linear_x * math.cos(yaw) * dt
                dy = cmd.linear_x * math.sin(yaw) * dt
                dyaw = cmd.angular_z * dt
                
                odom["position"]["x"] += dx
                odom["position"]["y"] += dy
                
                # 更新姿态
                new_yaw = yaw + dyaw
                odom["orientation"]["z"] = math.sin(new_yaw / 2)
                odom["orientation"]["w"] = math.cos(new_yaw / 2)
                
                # 更新速度
                odom["linear_velocity"]["x"] = cmd.linear_x
                odom["angular_velocity"]["z"] = cmd.angular_z
    
    async def publish_twist_for_duration(
        self, 
        cmd: TwistCommand, 
        duration: float,
        rate: float = 10.0
    ):
        """
        在指定时间内持续发布速度命令
        
        Args:
            cmd: TwistCommand对象
            duration: 持续时间（秒）
            rate: 发布频率（Hz）
        """
        interval = 1.0 / rate
        elapsed = 0.0
        
        while elapsed < duration:
            await self.publish_twist(cmd)
            await asyncio.sleep(interval)
            elapsed += interval
        
        # 停止
        await self.publish_twist(TwistCommand.stop())
    
    def get_sensor_data(self) -> SensorData:
        """获取当前传感器数据"""
        with self._data_lock:
            return SensorData(
                timestamp=self._sensor_data.timestamp,
                rgb_image=self._sensor_data.rgb_image.copy() if self._sensor_data.rgb_image is not None else None,
                depth_image=self._sensor_data.depth_image.copy() if self._sensor_data.depth_image is not None else None,
                laser_scan=self._sensor_data.laser_scan.copy() if self._sensor_data.laser_scan else None,
                pointcloud=self._sensor_data.pointcloud.copy() if self._sensor_data.pointcloud is not None else None,
                odometry=self._sensor_data.odometry.copy() if self._sensor_data.odometry else None,
                imu=self._sensor_data.imu.copy() if self._sensor_data.imu else None,
                occupancy_grid=self._sensor_data.occupancy_grid.copy() if self._sensor_data.occupancy_grid is not None else None,
                map_info=self._sensor_data.map_info.copy() if self._sensor_data.map_info else None
            )
    
    def get_rgb_image(self) -> Optional[np.ndarray]:
        """获取最新RGB图像"""
        with self._data_lock:
            return self._sensor_data.rgb_image.copy() if self._sensor_data.rgb_image is not None else None
    
    def get_depth_image(self) -> Optional[np.ndarray]:
        """获取最新深度图像"""
        with self._data_lock:
            return self._sensor_data.depth_image.copy() if self._sensor_data.depth_image is not None else None
    
    def get_laser_scan(self) -> Optional[Dict[str, Any]]:
        """获取最新激光雷达数据"""
        with self._data_lock:
            return self._sensor_data.laser_scan.copy() if self._sensor_data.laser_scan else None
    
    def get_odometry(self) -> Optional[Dict[str, Any]]:
        """获取最新里程计数据"""
        with self._data_lock:
            return self._sensor_data.odometry.copy() if self._sensor_data.odometry else None
    
    def get_imu(self) -> Optional[Dict[str, Any]]:
        """获取最新IMU数据"""
        with self._data_lock:
            return self._sensor_data.imu.copy() if self._sensor_data.imu else None
    
    def get_current_pose(self) -> Tuple[float, float, float]:
        """获取当前位姿 (x, y, yaw)"""
        import math
        
        with self._data_lock:
            if self._sensor_data.odometry:
                odom = self._sensor_data.odometry
                x = odom["position"]["x"]
                y = odom["position"]["y"]
                
                q = odom["orientation"]
                yaw = math.atan2(2 * (q["w"] * q["z"] + q["x"] * q["y"]),
                                 1 - 2 * (q["y"]**2 + q["z"]**2))
                
                return (x, y, yaw)
        
        return (0.0, 0.0, 0.0)
    
    def register_sensor_callback(self, sensor_type: str, callback: Callable):
        """注册传感器数据回调"""
        if sensor_type not in self._sensor_callbacks:
            self._sensor_callbacks[sensor_type] = []
        self._sensor_callbacks[sensor_type].append(callback)
    
    def update_simulated_image(self, rgb_image: np.ndarray):
        """更新模拟的RGB图像（用于测试）"""
        with self._data_lock:
            self._sensor_data.rgb_image = rgb_image
            self._sensor_data.timestamp = datetime.now()
    
    def update_simulated_laser(self, ranges: List[float]):
        """更新模拟的激光雷达数据（用于测试）"""
        with self._data_lock:
            if self._sensor_data.laser_scan:
                self._sensor_data.laser_scan["ranges"] = ranges
                self._sensor_data.timestamp = datetime.now()
    
    async def shutdown(self):
        """关闭ROS2接口"""
        self._running = False

        # 停止命令队列
        if self.command_queue:
            await self.command_queue.stop()
            logger.info("命令队列已停止")

        if self.mode == ROS2Mode.REAL:
            if self._spin_thread:
                self._spin_thread.join(timeout=2.0)

            if self._node:
                self._node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()

        logger.info("ROS2接口已关闭")

    def get_command_queue_stats(self) -> Optional[Dict[str, Any]]:
        """获取命令队列统计信息"""
        if self.command_queue:
            return self.command_queue.get_stats()
        return None

    def get_command_queue_status(self) -> Optional[Dict[str, Any]]:
        """获取命令队列状态"""
        if self.command_queue:
            return self.command_queue.get_queue_status()
        return None
    
    def is_running(self) -> bool:
        """检查接口是否运行中"""
        return self._running
    
    def get_mode(self) -> ROS2Mode:
        """获取运行模式"""
        return self.mode


# === 便捷函数 ===

def create_twist(
    linear_x: float = 0.0,
    linear_y: float = 0.0,
    linear_z: float = 0.0,
    angular_x: float = 0.0,
    angular_y: float = 0.0,
    angular_z: float = 0.0
) -> TwistCommand:
    """创建Twist命令的便捷函数"""
    return TwistCommand(
        linear_x=linear_x,
        linear_y=linear_y,
        linear_z=linear_z,
        angular_x=angular_x,
        angular_y=angular_y,
        angular_z=angular_z
    )

