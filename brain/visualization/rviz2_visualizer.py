"""
RViz2 可视化器 - RViz2 Visualizer

通过发布ROS2话题，在RViz2中实时显示世界地图、机器人状态、语义物体等信息
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from loguru import logger

# 尝试导入ROS2消息类型
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
    from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
    from visualization_msgs.msg import Marker, MarkerArray
    from std_msgs.msg import Header, ColorRGBA
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # 创建占位类型，避免类型注解错误
    Marker = Any
    MarkerArray = Any
    OccupancyGrid = Any
    MapMetaData = Any
    Path = Any
    PoseStamped = Any
    Header = Any
    ColorRGBA = Any
    logger.warning("ROS2 (rclpy) 不可用，RViz2可视化将不可用")


@dataclass
class RViz2VisualizerConfig:
    """RViz2可视化配置"""
    enabled: bool = True
    update_rate: float = 2.0  # Hz
    map_topic: str = "/brain/map"
    path_topic: str = "/brain/robot_path"
    markers_topic: str = "/brain/visualization_markers"
    pose_topic: str = "/brain/robot_pose"
    frame_id: str = "map"  # 坐标系


class RViz2Visualizer:
    """
    RViz2可视化器
    
    通过发布ROS2话题，在RViz2中显示世界模型信息
    """
    
    def __init__(
        self,
        ros2_interface,
        world_model,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            ros2_interface: ROS2Interface 实例
            world_model: WorldModel 实例
            config: 配置字典
        """
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 (rclpy) is required for RViz2 visualization")
        
        self.ros2_interface = ros2_interface
        self.world_model = world_model
        
        # 解析配置
        if config:
            self.config = RViz2VisualizerConfig(**{k: v for k, v in config.items() 
                                                    if hasattr(RViz2VisualizerConfig, k)})
        else:
            self.config = RViz2VisualizerConfig()
        
        # 检查ROS2接口是否可用
        if not self.ros2_interface or self.ros2_interface.mode.value != "real":
            logger.warning("ROS2接口不可用或处于模拟模式，RViz2可视化可能无法工作")
        
        # 发布者（在ROS2节点中创建）
        self._map_publisher = None
        self._path_publisher = None
        self._markers_publisher = None
        self._pose_publisher = None
        
        # 状态
        self._running = False
        self._last_update = time.time()
        self._update_interval = 1.0 / self.config.update_rate
        
        # 轨迹点列表
        self._path_points: List[PoseStamped] = []
        self._max_path_points = 1000
        
        logger.info("RViz2Visualizer 初始化完成")
    
    def initialize(self):
        """初始化发布者（需要在ROS2节点创建后调用）"""
        if not self.config.enabled:
            return
        
        if not self.ros2_interface or not hasattr(self.ros2_interface, '_node'):
            logger.warning("ROS2节点未创建，无法初始化RViz2可视化器")
            return
        
        node = self.ros2_interface._node
        
        # QoS配置：地图使用持久化，其他使用默认
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        default_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 创建发布者
        self._map_publisher = node.create_publisher(
            OccupancyGrid,
            self.config.map_topic,
            map_qos
        )
        
        self._path_publisher = node.create_publisher(
            Path,
            self.config.path_topic,
            default_qos
        )
        
        self._markers_publisher = node.create_publisher(
            MarkerArray,
            self.config.markers_topic,
            default_qos
        )
        
        self._pose_publisher = node.create_publisher(
            PoseStamped,
            self.config.pose_topic,
            default_qos
        )
        
        self._running = True
        logger.info("RViz2可视化器发布者已创建")
    
    def update(self):
        """更新可视化（定期调用）"""
        if not self._running or not self.config.enabled:
            return
        
        if not self._map_publisher:
            # 发布者未初始化
            return
        
        current_time = time.time()
        if current_time - self._last_update < self._update_interval:
            return
        
        self._last_update = current_time
        
        try:
            # 发布地图（即使为空也发布一次，让RViz知道话题存在）
            self._publish_map()
            
            # 发布机器人位姿
            self._publish_robot_pose()
            
            # 发布轨迹
            self._publish_path()
            
            # 发布标记（语义物体、障碍物、探索边界等）
            self._publish_markers()
        
        except Exception as e:
            logger.error(f"更新RViz2可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _publish_map(self):
        """发布占据栅格地图"""
        if not self._map_publisher:
            return
        
        # 即使地图为空，也发布一个空地图，让RViz知道话题存在
        if self.world_model.current_map is None:
            # 创建一个默认大小的地图（100m x 100m，分辨率0.1m），以机器人位置为中心
            map_size = 100.0
            resolution = 0.1
            grid_size = int(map_size / resolution)
            map_data = np.full((grid_size, grid_size), -1, dtype=np.int8)  # 未知区域
            
            # 根据机器人当前位置设置地图原点，使机器人位置在地图中心
            robot_x = self.world_model.robot_position.get('x', 0.0)
            robot_y = self.world_model.robot_position.get('y', 0.0)
            origin_x = robot_x - map_size / 2
            origin_y = robot_y - map_size / 2
            logger.info(f"地图数据为空，发布默认空地图（100m x 100m），以机器人位置({robot_x:.2f}, {robot_y:.2f})为中心")
        else:
            map_data = self.world_model.current_map
            resolution = self.world_model.map_resolution
            map_origin = self.world_model.map_origin
            
            if isinstance(map_origin, tuple) and len(map_origin) >= 2:
                origin_x, origin_y = map_origin[0], map_origin[1]
            else:
                # 如果没有原点信息，根据机器人位置和地图大小计算原点
                robot_x = self.world_model.robot_position.get('x', 0.0)
                robot_y = self.world_model.robot_position.get('y', 0.0)
                height, width = map_data.shape
                map_size_x = width * resolution
                map_size_y = height * resolution
                origin_x = robot_x - map_size_x / 2
                origin_y = robot_y - map_size_y / 2
                logger.info(f"地图原点未设置，根据机器人位置({robot_x:.2f}, {robot_y:.2f})计算: ({origin_x:.2f}, {origin_y:.2f})")
        
        # 创建OccupancyGrid消息
        map_msg = OccupancyGrid()
        map_msg.header = Header()
        map_msg.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        map_msg.header.frame_id = self.config.frame_id
        
        # 地图元数据
        height, width = map_data.shape
        map_msg.info = MapMetaData()
        map_msg.info.resolution = float(resolution)
        map_msg.info.width = int(width)
        map_msg.info.height = int(height)
        map_msg.info.origin.position.x = float(origin_x)
        map_msg.info.origin.position.y = float(origin_y)
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # 转换地图数据：-1 -> -1, 0 -> 0, 100 -> 100
        # RViz期望值范围：-1(未知), 0(自由), 100(占据)
        map_data_flat = map_data.flatten().astype(np.int8)
        map_msg.data = map_data_flat.tolist()
        
        self._map_publisher.publish(map_msg)
        logger.info(f"发布地图: {width}x{height}, 分辨率={resolution}m, 原点=({origin_x:.2f}, {origin_y:.2f}), 地图范围=[{origin_x:.2f}, {origin_x + width * resolution:.2f}] x [{origin_y:.2f}, {origin_y + height * resolution:.2f}]")
    
    def _publish_robot_pose(self):
        """发布机器人当前位姿"""
        if not self._pose_publisher:
            return
        
        pos = self.world_model.robot_position
        heading = self.world_model.robot_heading
        
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.config.frame_id
        
        x = float(pos.get('x', 0.0))
        y = float(pos.get('y', 0.0))
        z = float(pos.get('z', 0.0))
        
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        
        # 将航向角转换为四元数
        qx, qy, qz, qw = self._yaw_to_quaternion(heading)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        
        self._pose_publisher.publish(pose_msg)
        logger.info(f"发布机器人位姿: ({x:.2f}, {y:.2f}, {z:.2f}), 航向={math.degrees(heading):.1f}°")
        
        # 添加到轨迹（创建副本，避免引用问题）
        pose_copy = PoseStamped()
        pose_copy.header = Header()
        pose_copy.header.stamp = pose_msg.header.stamp
        pose_copy.header.frame_id = pose_msg.header.frame_id
        pose_copy.pose = Pose()
        pose_copy.pose.position.x = x
        pose_copy.pose.position.y = y
        pose_copy.pose.position.z = z
        pose_copy.pose.orientation.x = qx
        pose_copy.pose.orientation.y = qy
        pose_copy.pose.orientation.z = qz
        pose_copy.pose.orientation.w = qw
        
        self._path_points.append(pose_copy)
        if len(self._path_points) > self._max_path_points:
            self._path_points.pop(0)
    
    def _publish_path(self):
        """发布机器人轨迹"""
        if not self._path_publisher:
            return
        
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        path_msg.header.frame_id = self.config.frame_id
        
        if not self._path_points:
            # 即使没有轨迹点，也发布空路径（让RViz知道话题存在）
            path_msg.poses = []
        else:
            # 限制轨迹点数量（避免消息过大）
            max_points = 500
            if len(self._path_points) > max_points:
                # 均匀采样
                indices = np.linspace(0, len(self._path_points) - 1, max_points, dtype=int)
                path_msg.poses = [self._path_points[i] for i in indices]
            else:
                path_msg.poses = self._path_points.copy()
        
        self._path_publisher.publish(path_msg)
        if len(path_msg.poses) > 0:
            logger.info(f"发布轨迹: {len(path_msg.poses)} 个点")
    
    def _publish_markers(self):
        """发布可视化标记（语义物体、障碍物、探索边界等）"""
        if not self._markers_publisher:
            return
        
        markers = MarkerArray()
        marker_id = 0
        
        # 1. 语义物体
        for obj_id, obj in self.world_model.semantic_objects.items():
            marker = self._create_object_marker(obj, marker_id)
            if marker:
                markers.markers.append(marker)
                marker_id += 1
        
        # 2. 障碍物
        for obs_id, obs in self.world_model.tracked_objects.items():
            marker = self._create_obstacle_marker(obs, marker_id)
            if marker:
                markers.markers.append(marker)
                marker_id += 1
        
        # 3. 探索边界
        for frontier in self.world_model.exploration_frontiers:
            marker = self._create_frontier_marker(frontier, marker_id)
            if marker:
                markers.markers.append(marker)
                marker_id += 1
        
        # 4. 删除标记（用于清理旧的标记）- 放在最前面
        delete_marker = Marker()
        delete_marker.header = Header()
        delete_marker.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        delete_marker.header.frame_id = self.config.frame_id
        delete_marker.action = Marker.DELETEALL
        markers.markers.insert(0, delete_marker)
        
        self._markers_publisher.publish(markers)
        if marker_id > 1:  # 除了删除标记外还有其他标记
            logger.info(f"发布 {marker_id - 1} 个可视化标记（语义物体: {len(self.world_model.semantic_objects)}, 障碍物: {len(self.world_model.tracked_objects)}, 边界: {len(self.world_model.exploration_frontiers)}）")
    
    def _create_object_marker(self, obj, marker_id: int) -> Optional[Marker]:
        """创建语义物体标记"""
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        marker.header.frame_id = self.config.frame_id
        
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # 位置
        x, y = obj.world_position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        # 大小
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 1.0
        
        # 颜色（根据物体类型）
        color = self._get_object_color(obj.label)
        marker.color = ColorRGBA(
            r=color[0],
            g=color[1],
            b=color[2],
            a=0.8
        )
        
        # 文本标签
        marker.text = obj.label
        
        return marker
    
    def _create_obstacle_marker(self, obs, marker_id: int) -> Optional[Marker]:
        """创建障碍物标记"""
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        marker.header.frame_id = self.config.frame_id
        
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # 位置
        pos = obs.position
        marker.pose.position.x = float(pos.get('x', 0.0))
        marker.pose.position.y = float(pos.get('y', 0.0))
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        # 大小
        size = obs.size
        marker.scale.x = float(size.get('width', 0.5))
        marker.scale.y = float(size.get('width', 0.5))
        marker.scale.z = float(size.get('height', 1.0))
        
        # 颜色（红色）
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)
        
        return marker
    
    def _create_frontier_marker(self, frontier, marker_id: int) -> Optional[Marker]:
        """创建探索边界标记"""
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = self.ros2_interface._node.get_clock().now().to_msg()
        marker.header.frame_id = self.config.frame_id
        
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 位置
        x, y = frontier.position
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        
        # 大小
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # 颜色（黄色）
        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7)
        
        return marker
    
    def _get_object_color(self, label: str) -> Tuple[float, float, float]:
        """根据物体标签获取颜色"""
        label_lower = label.lower()
        
        color_map = {
            'door': (1.0, 0.5, 0.0),      # 橙色
            'building': (0.5, 0.5, 0.5),  # 灰色
            'person': (0.0, 1.0, 1.0),    # 青色
            'vehicle': (0.5, 0.0, 0.5),   # 紫色
            'target': (1.0, 0.0, 1.0),    # 洋红色
        }
        
        for key, color in color_map.items():
            if key in label_lower:
                return color
        
        return (0.0, 0.8, 0.8)  # 默认青色
    
    def _yaw_to_quaternion(self, yaw: float) -> Tuple[float, float, float, float]:
        """将航向角转换为四元数"""
        qx = 0.0
        qy = 0.0
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return (qx, qy, qz, qw)
    
    def stop(self):
        """停止可视化"""
        self._running = False
        logger.info("RViz2可视化已停止")

