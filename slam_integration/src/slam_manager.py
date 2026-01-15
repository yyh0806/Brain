# -*- coding: utf-8 -*-
"""
SLAM管理器 - Brain项目SLAM集成核心模块

提供统一的SLAM接口，支持多种SLAM后端（FAST-LIVO, LIO-SAM等）
"""

import asyncio
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import threading

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from nav_msgs.msg import OccupancyGrid, Path
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from sensor_msgs.msg import PointCloud2, Image, Imu
    import tf2_ros
    from tf2_ros import TransformException
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("警告: ROS2未安装或未在PATH中，SLAM功能将不可用")

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SLAMConfig:
    """SLAM配置"""
    backend: str = "fast_livo"
    resolution: float = 0.1
    map_update_interval: float = 0.1
    zero_copy: bool = True
    update_frequency: float = 10.0


@dataclass
class MapMetadata:
    """地图元数据"""
    resolution: float
    width: int
    height: int
    origin: Tuple[float, float, float]  # x, y, yaw


class SLAMManager:
    """
    SLAM管理器 - 统一的SLAM接口

    职责：
    1. 订阅SLAM节点发布的地图和位姿
    2. 提供零拷贝的地图访问接口
    3. 管理坐标转换
    4. 支持场景自适应（室内/室外/混合）
    """

    def __init__(self, config: SLAMConfig):
        self.config = config
        self._slam_map: Optional[OccupancyGrid] = None
        self._slam_pose: Optional[PoseStamped] = None
        self._slam_path: Optional[Path] = None
        self._is_initialized = False

        # ROS2相关
        self._ros_node: Optional[Node] = None
        self._executor: Optional[MultiThreadedExecutor] = None
        self._ros_thread: Optional[threading.Thread] = None

        # 地图缓冲（用于异步更新）
        self._map_buffer = deque(maxlen=2)

        # 坐标转换器
        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None

        if not ROS2_AVAILABLE:
            logger.warning("ROS2不可用，SLAMManager将以模拟模式运行")
            return

        self._initialize_ros2()

    def _initialize_ros2(self):
        """初始化ROS2节点"""
        try:
            # 如果ROS2还未初始化
            if not rclpy.ok():
                rclpy.init()

            # 创建ROS2节点
            self._ros_node = Node('brain_slam_subscriber')
            self._executor = MultiThreadedExecutor(num_threads=2)

            # 订阅SLAM话题
            self._setup_subscribers()

            # TF2
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._ros_node)

            # 在独立线程中运行ROS2
            self._ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
            self._ros_thread.start()

            self._is_initialized = True
            logger.info("SLAM Manager ROS2初始化成功")

        except Exception as e:
            logger.error(f"SLAM Manager初始化失败: {e}")
            self._is_initialized = False

    def _setup_subscribers(self):
        """设置ROS2订阅者"""
        # 订阅地图
        self._ros_node.create_subscription(
            OccupancyGrid,
            '/map',
            self._on_map_received,
            10  # QoS depth
        )

        # 订阅位姿
        self._ros_node.create_subscription(
            PoseStamped,
            '/pose',
            self._on_pose_received,
            10
        )

        # 订阅路径（可选）
        self._ros_node.create_subscription(
            Path,
            '/path',
            self._on_path_received,
            10
        )

        logger.info("SLAM话题订阅已设置")

    def _on_map_received(self, msg: OccupancyGrid):
        """地图接收回调"""
        self._slam_map = msg
        self._map_buffer.append(msg)
        # logger.debug(f"SLAM地图更新: {msg.info.width}x{msg.info.height}")

    def _on_pose_received(self, msg: PoseStamped):
        """位姿接收回调"""
        self._slam_pose = msg
        # logger.debug(f"SLAM位姿更新: {msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}")

    def _on_path_received(self, msg: Path):
        """路径接收回调"""
        self._slam_path = msg

    def _spin_ros(self):
        """在独立线程中运行ROS2"""
        try:
            rclpy.spin(self._ros_node, executor=self._executor)
        except Exception as e:
            logger.error(f"ROS2 spin错误: {e}")

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._is_initialized

    @property
    def slam_map(self) -> Optional[OccupancyGrid]:
        """
        获取SLAM地图（零拷贝引用）

        返回的是ROS2消息对象，不进行数据复制
        """
        return self._slam_map

    @property
    def slam_pose(self) -> Optional[PoseStamped]:
        """获取SLAM位姿（零拷贝引用）"""
        return self._slam_pose

    @property
    def slam_path(self) -> Optional[Path]:
        """获取SLAM路径"""
        return self._slam_path

    def get_geometric_map(self) -> Optional[np.ndarray]:
        """
        获取几何地图（numpy数组格式）

        注意：这会进行数据转换，不是零拷贝
        建议优先使用slam_map属性获取零拷贝访问
        """
        if self._slam_map is None:
            return None

        # 将ROS消息转换为numpy数组
        return np.array(
            self._slam_map.data,
            dtype=np.int8
        ).reshape(
            self._slam_map.info.height,
            self._slam_map.info.width
        )

    def get_map_metadata(self) -> Optional[MapMetadata]:
        """获取地图元数据"""
        if self._slam_map is None:
            return None

        origin = self._slam_map.info.origin.position
        return MapMetadata(
            resolution=self._slam_map.info.resolution,
            width=self._slam_map.info.width,
            height=self._slam_map.info.height,
            origin=(origin.x, origin.y, 0.0)
        )

    def world_to_grid(self, world_position: Tuple[float, float]) -> Tuple[int, int]:
        """
        世界坐标 → 栅格坐标

        Args:
            world_position: 世界坐标 (x, y) 单位：米

        Returns:
            栅格坐标 (grid_x, grid_y)
        """
        if self._slam_map is None:
            raise ValueError("SLAM地图尚未可用")

        origin = self._slam_map.info.origin.position
        resolution = self._slam_map.info.resolution

        grid_x = int((world_position[0] - origin.x) / resolution)
        grid_y = int((world_position[1] - origin.y) / resolution)

        return (grid_x, grid_y)

    def grid_to_world(self, grid_position: Tuple[int, int]) -> Tuple[float, float]:
        """
        栅格坐标 → 世界坐标

        Args:
            grid_position: 栅格坐标 (grid_x, grid_y)

        Returns:
            世界坐标 (x, y) 单位：米
        """
        if self._slam_map is None:
            raise ValueError("SLAM地图尚未可用")

        origin = self._slam_map.info.origin.position
        resolution = self._slam_map.info.resolution

        world_x = origin.x + grid_position[0] * resolution
        world_y = origin.y + grid_position[1] * resolution

        return (world_x, world_y)

    async def wait_for_map(self, timeout: float = 5.0) -> bool:
        """
        等待SLAM地图可用

        Args:
            timeout: 超时时间(秒)

        Returns:
            是否成功获取地图
        """
        if not self._is_initialized:
            logger.warning("SLAM Manager未初始化")
            return False

        start_time = asyncio.get_event_loop().time()
        while self._slam_map is None:
            await asyncio.sleep(0.1)
            if asyncio.get_event_loop().time() - start_time > timeout:
                logger.warning(f"等待SLAM地图超时({timeout}秒)")
                return False

        logger.info("SLAM地图已就绪")
        return True

    def get_robot_position(self) -> Optional[Tuple[float, float, float]]:
        """
        获取机器人位置

        Returns:
            (x, y, yaw) 单位：米、弧度
        """
        if self._slam_pose is None:
            return None

        pose = self._slam_pose.pose
        return (
            pose.position.x,
            pose.position.y,
            0.0  # TODO: 从四元数计算yaw
        )

    def shutdown(self):
        """关闭SLAM Manager"""
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            logger.info("SLAM Manager已关闭")


class CoordinateTransformer:
    """
    坐标转换器 - 统一SLAM、感知、认知的坐标系

    坐标系说明：
    - map: SLAM全局坐标系
    - odom: 里程计坐标系
    - base_link: 机器人本体坐标系
    - camera_link: 相机坐标系
    """

    def __init__(self, slam_manager: SLAMManager):
        self.slam_manager = slam_manager

    def world_to_grid(self, world_position: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标 → 栅格坐标"""
        return self.slam_manager.world_to_grid(world_position)

    def grid_to_world(self, grid_position: Tuple[int, int]) -> Tuple[float, float]:
        """栅格坐标 → 世界坐标"""
        return self.slam_manager.grid_to_world(grid_position)

    async def transform_to_map_frame(self, pose: PoseStamped) -> Optional[PoseStamped]:
        """
        将任意坐标系的位姿转换到map坐标系

        Args:
            pose: 输入位姿

        Returns:
            map坐标系下的位姿
        """
        if self.slam_manager._tf_buffer is None:
            logger.warning("TF2缓冲区不可用")
            return None

        try:
            transform = self.slam_manager._tf_buffer.lookup_transform(
                "map",
                pose.header.frame_id,
                pose.header.stamp,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            # 应用变换
            transformed_pose = PoseStamped()
            transformed_pose.header.frame_id = "map"
            # TODO: 完整的变换计算
            return transformed_pose

        except TransformException as e:
            logger.warning(f"坐标转换失败: {e}")
            return None


# 模拟SLAM管理器（用于无ROS2环境的测试）
class MockSLAMManager(SLAMManager):
    """模拟SLAM管理器 - 用于开发和测试"""

    def __init__(self, config: SLAMConfig):
        super().__init__(config)
        # 创建模拟地图
        self._create_mock_map()

    def _create_mock_map(self, width=500, height=500, resolution=0.1):
        """创建模拟占据栅格地图"""
        self._mock_map_data = np.zeros((height, width), dtype=np.int8)

        # 添加一些障碍物
        # 墙壁
        self._mock_map_data[50:150, 100] = 100
        self._mock_map_data[50:150, 200] = 100

        logger.info("模拟SLAM地图已创建")

    @property
    def slam_map(self):
        """返回模拟地图"""
        # 创建模拟的OccupancyGrid消息
        if not hasattr(self, '_mock_slam_map'):
            # 这里简化处理，实际应该创建完整的ROS2消息
            class MockMap:
                info = type('obj', (object,), {
                    'resolution': 0.1,
                    'width': 500,
                    'height': 500,
                    'origin': type('obj', (object,), {'position': type('obj', (object,), {'x': 0.0, 'y': 0.0})})()
                })()
                data = np.zeros(500 * 500, dtype=np.int8)

            self._mock_slam_map = MockMap()

        return self._mock_slam_map

    def get_geometric_map(self):
        """获取模拟几何地图"""
        return self._mock_map_data


if __name__ == "__main__":
    # 测试代码
    async def test_slam_manager():
        config = SLAMConfig()
        manager = SLAMManager(config)

        if manager.is_initialized:
            logger.info("SLAM Manager初始化成功")

            # 等待地图
            if await manager.wait_for_map(timeout=10.0):
                logger.info("成功获取SLAM地图")

                # 测试坐标转换
                grid_pos = manager.world_to_grid((5.0, 3.0))
                logger.info(f"世界坐标(5.0, 3.0) → 栅格坐标{grid_pos}")

                world_pos = manager.grid_to_world(grid_pos)
                logger.info(f"栅格坐标{grid_pos} → 世界坐标{world_pos}")
        else:
            # 使用模拟模式
            logger.info("使用模拟SLAM Manager")
            mock_manager = MockSLAMManager(config)

            geo_map = mock_manager.get_geometric_map()
            if geo_map is not None:
                logger.info(f"模拟地图尺寸: {geo_map.shape}")

    # 运行测试
    asyncio.run(test_slam_manager())
