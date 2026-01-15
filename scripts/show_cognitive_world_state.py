#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知层世界状态可视化演示
使用真实ROS2传感器数据展示认知层世界状态可视化

Usage:
    # Terminal 1: 播放rosbag（可选，如果没有实时传感器）
    export ROS_DOMAIN_ID=42
    ros2 bag play <rosbag文件> --loop

    # Terminal 2: 启动可视化
    export ROS_DOMAIN_ID=42
    python3 scripts/show_cognitive_world_state.py

    # Terminal 3: 启动RViz查看可视化
    rviz2 -d config/rviz2/cognitive_world_model_correct.rviz
"""

import sys
import os
import time
import math
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from collections import deque
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor

# 设置ROS Domain ID
os.environ['ROS_DOMAIN_ID'] = '42'

# Add Brain to path
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

# ROS2消息类型
from sensor_msgs.msg import Image as SensorImage, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import StaticTransformBroadcaster

# Cognitive layer imports
from brain.cognitive.world_model.world_model import WorldModel
from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer

# VLM配置：使用Mock还是真实Ollama
USE_MOCK_VLM = False  # 🔧 使用真实Ollama VLM - 图像编码已修复

# 使用统一的VLM工厂函数
from brain.perception.vlm import get_vlm_client

if USE_MOCK_VLM:
    print("🎭 使用 Mock VLM 客户端（演示模式）")
else:
    print("🤖 使用真实 Ollama LLaVA 客户端")


def quaternion_to_yaw(quaternion) -> float:
    """从四元数计算航向角（yaw）"""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w

    # 计算yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw


def ros_image_to_numpy(img_msg: SensorImage) -> np.ndarray:
    """简单的ROS Image转numpy转换（不依赖cv_bridge）"""
    if img_msg.encoding in ['rgb8', 'bgr8', 'mono8']:
        dtype = np.uint8
    else:
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

    arr = np.frombuffer(img_msg.data, dtype=dtype)

    if img_msg.encoding in ['rgb8', 'bgr8']:
        n_channels = 3
    else:
        n_channels = 1

    if n_channels == 1:
        arr = arr.reshape((img_msg.height, img_msg.width))
    else:
        arr = arr.reshape((img_msg.height, img_msg.width, n_channels))

    if img_msg.encoding == 'bgr8':
        arr = arr[:, :, ::-1].copy()

    return arr


def parse_pointcloud(msg: PointCloud2) -> Optional[np.ndarray]:
    """解析点云数据"""
    try:
        # 获取点云字段
        fields = {}
        for field in msg.fields:
            fields[field.name] = field

        # 检查是否有xyz字段
        if 'x' not in fields or 'y' not in fields or 'z' not in fields:
            return None

        # 解析点云
        cloud_points = []
        data = msg.data
        point_step = msg.point_step

        # 遍历每个点
        for i in range(0, len(data), point_step):
            # 提取xyz坐标
            x_offset = fields['x'].offset
            y_offset = fields['y'].offset
            z_offset = fields['z'].offset

            if i + x_offset + 4 > len(data) or i + y_offset + 4 > len(data) or i + z_offset + 4 > len(data):
                continue

            x_bytes = data[i + x_offset : i + x_offset + 4]
            y_bytes = data[i + y_offset : i + y_offset + 4]
            z_bytes = data[i + z_offset : i + z_offset + 4]

            x = np.frombuffer(x_bytes, dtype=np.float32)[0]
            y = np.frombuffer(y_bytes, dtype=np.float32)[0]
            z = np.frombuffer(z_bytes, dtype=np.float32)[0]

            cloud_points.append([x, y, z])

        return np.array(cloud_points)

    except Exception:
        return None


class CognitiveWorldStateViz(Node):
    """认知世界状态可视化节点"""

    def __init__(self, duration_seconds: float = 600.0):
        super().__init__('cognitive_world_state_viz')

        # 配置
        self.duration_seconds = duration_seconds
        self.start_time = time.time()

        # 状态
        self.odom_count = 0
        self.pointcloud_count = 0
        self.rgb_count = 0
        self.vlm_count = 0
        self.last_display_time = 0
        self.display_interval = 5.0  # 每5秒显示一次状态

        # 位姿历史 (用于时间戳对齐)
        # 存储 (timestamp, pose_dict) 元组
        self.pose_history = deque(maxlen=1000)  # 保留最近1000个位姿
        self.pose_history_lock = Lock()

        # VLM客户端 (异步处理)
        self.vlm_client = get_vlm_client(use_mock=USE_MOCK_VLM)
        self.vlm_enabled = True
        self.vlm_processing_interval = 5.0  # 每5秒处理一次VLM (避免过载)
        self.last_vlm_time = 0
        self.vlm_processing = False  # 防止重复提交VLM任务

        # VLM线程池（限制并发，避免线程耗尽）
        self.vlm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm_worker")

        self.get_logger().info("=" * 80)
        self.get_logger().info("🎯 认知层世界状态可视化")
        self.get_logger().info("=" * 80)

        # 1. 初始化WorldModel
        self.get_logger().info("正在初始化WorldModel...")
        world_config = {
            'map_resolution': 0.1,  # 10cm per cell
            'map_size': 50.0,      # 50m x 50m
        }
        self.world_model = WorldModel(config=world_config)

        # 初始化固定大小的地图，防止尺寸跳变
        self._initialize_fixed_map()

        self.get_logger().info("✅ WorldModel初始化完成")
        self.get_logger().info(f"   地图分辨率: {self.world_model.map_resolution}m/cell")
        self.get_logger().info(f"   地图原点: {self.world_model.map_origin}")
        self.get_logger().info(f"   地图尺寸: {self.world_model.current_map.shape if self.world_model.current_map is not None else 'None'}")

        # 2. 初始化可视化器
        self.get_logger().info("正在初始化可视化器...")
        self.visualizer = WorldModelVisualizer(
            world_model=self.world_model,
            publish_rate=2.0  # 2Hz
        )
        self.get_logger().info("✅ 可视化器初始化完成")
        self.get_logger().info("   发布话题:")
        self.get_logger().info("     - /world_model/semantic_grid")
        self.get_logger().info("     - /world_model/semantic_markers")
        self.get_logger().info("     - /world_model/belief_markers")
        self.get_logger().info("     - /world_model/trajectory")
        self.get_logger().info("     - /world_model/frontiers")
        self.get_logger().info("     - /world_model/change_events")
        self.get_logger().info("     - /vlm/detections")

        # 3. 发布静态TF变换 (map -> odom)
        self._publish_static_tf()

        # 4. 订阅传感器话题
        self._setup_subscribers()

        self.get_logger().info("=" * 80)
        self.get_logger().info("✅ 认知世界状态可视化节点已启动")
        self.get_logger().info("=" * 80)

    def _publish_static_tf(self):
        """发布静态TF变换 (map -> odom)"""
        try:
            # 创建静态TF广播器
            self.tf_broadcaster = StaticTransformBroadcaster(self)

            # 创建变换消息
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "odom"

            # 设置变换为恒等变换（map和odom重合）
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            # 发送变换
            self.tf_broadcaster.sendTransform(t)

            self.get_logger().info("✅ 已发布静态TF: map -> odom")
        except Exception as e:
            self.get_logger().warning(f"发布静态TF失败: {e}")

    def _initialize_fixed_map(self):
        """初始化固定大小的地图，防止尺寸跳变"""
        try:
            # 创建固定大小的地图：500x500 (50m x 50m at 0.1m/cell)
            grid_size = 500
            self.world_model.current_map = np.full((grid_size, grid_size), -1, dtype=np.int8)
            self.world_model.map_origin = (-25.0, -25.0)  # 中心为(0,0)

            self.get_logger().info("✅ 已初始化固定大小地图")
            self.get_logger().info(f"   地图尺寸: {grid_size}x{grid_size}")
            self.get_logger().info(f"   地图范围: 50m x 50m")
        except Exception as e:
            self.get_logger().warning(f"初始化固定地图失败: {e}")

    def _sync_occupancy_map(self):
        """确保地图尺寸保持固定 - 每次调用都强制检查"""
        target_size = 500

        # 每次都强制检查，确保地图尺寸不会改变
        if (self.world_model.current_map is None or
            self.world_model.current_map.shape[0] != target_size or
            self.world_model.current_map.shape[1] != target_size):

            # 如果尺寸不对，立即修正
            if self.odom_count % 50 == 0:  # 偶尔打印日志
                self.get_logger().warning(f"修正地图尺寸: {self.world_model.current_map.shape if self.world_model.current_map is not None else 'None'} -> {target_size}x{target_size}")

            self.world_model.current_map = np.full((target_size, target_size), -1, dtype=np.int8)
            self.world_model.map_origin = (-25.0, -25.0)

    def _setup_subscribers(self):
        """设置ROS2订阅者"""
        self.odom_sub = self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            self.pointcloud_callback,
            10
        )
        self.rgb_sub = self.create_subscription(
            SensorImage,
            '/front_stereo_camera/left/image_raw',
            self.rgb_callback,
            10
        )

        self.get_logger().info("📡 已创建ROS2订阅者:")
        self.get_logger().info("   - /chassis/odom (Odometry)")
        self.get_logger().info("   - /front_3d_lidar/lidar_points (PointCloud2)")
        self.get_logger().info("   - /front_stereo_camera/left/image_raw (SensorImage)")

    def odom_callback(self, msg: Odometry):
        """里程计回调 - 更新WorldModel并记录位姿历史"""
        self.odom_count += 1

        try:
            # 提取位姿
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            yaw = quaternion_to_yaw(ori)

            # 构建当前位姿字典
            current_pose = {
                'x': pos.x,
                'y': pos.y,
                'z': pos.z,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': yaw
            }

            # 获取消息时间戳 (用于时间戳对齐)
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            pose_timestamp = datetime.fromtimestamp(msg_time) if msg_time > 0 else datetime.now()

            # 记录位姿历史 (线程安全)
            with self.pose_history_lock:
                self.pose_history.append((pose_timestamp, current_pose.copy()))

            # 构建感知数据
            perception_data = {
                'timestamp': datetime.now(),
                'pose': current_pose,
                'velocity': {
                    'linear_x': msg.twist.twist.linear.x,
                    'linear_y': msg.twist.twist.linear.y,
                    'linear_z': msg.twist.twist.linear.z,
                    'angular_x': msg.twist.twist.angular.x,
                    'angular_y': msg.twist.twist.angular.y,
                    'angular_z': msg.twist.twist.angular.z
                }
            }

            # 更新WorldModel
            self.world_model.update_from_perception(perception_data)

            # 同步OccupancyMapper的地图到WorldModel
            self._sync_occupancy_map()

            # 定期显示状态
            self._try_display_status()

            # 检查运行时长
            elapsed = time.time() - self.start_time
            if elapsed >= self.duration_seconds:
                self.get_logger().info("=" * 80)
                self.get_logger().info("✅ 演示完成")
                self.get_logger().info(f"   运行时长: {elapsed:.1f}秒")
                self.get_logger().info(f"   里程计更新: {self.odom_count}次")
                self.get_logger().info(f"   点云更新: {self.pointcloud_count}次")
                self.get_logger().info("=" * 80)
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"里程计回调错误: {e}")

    def pointcloud_callback(self, msg: PointCloud2):
        """点云回调 - 更新占据地图"""
        self.pointcloud_count += 1

        try:
            # 解析点云
            points = parse_pointcloud(msg)
            if points is None or len(points) == 0:
                return

            # 简单的占据地图更新（直接在固定地图上标记）
            if self.world_model.current_map is not None:
                self._update_occupancy_from_pointcloud(points)

        except Exception as e:
            # 只记录错误，不打印到终端避免刷屏
            pass

    def _update_occupancy_from_pointcloud(self, points: np.ndarray):
        """从点云更新占据地图（修复版）"""
        try:
            resolution = self.world_model.map_resolution
            origin_x, origin_y = self.world_model.map_origin
            grid = self.world_model.current_map

            # 过滤有效点（去除NaN和异常值）
            valid_mask = (
                np.isfinite(points[:, 0]) &
                np.isfinite(points[:, 1]) &
                np.isfinite(points[:, 2]) &
                (np.abs(points[:, 0]) < 50.0) &  # 限制范围50米
                (np.abs(points[:, 1]) < 50.0) &
                (points[:, 2] > -2.0) & (points[:, 2] < 5.0)  # 合理的高度范围
            )
            valid_points = points[valid_mask]

            if len(valid_points) == 0:
                return

            # 点云已经在正确的坐标系中（传感器坐标系或map坐标系）
            # 直接转换为栅格坐标
            world_x = valid_points[:, 0]
            world_y = valid_points[:, 1]

            # 转换到栅格坐标
            gx = ((world_x - origin_x) / resolution).astype(int)
            gy = ((world_y - origin_y) / resolution).astype(int)

            # 检查边界并标记占据
            height, width = grid.shape
            valid_mask = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
            gx_valid = gx[valid_mask]
            gy_valid = gy[valid_mask]

            # 标记为占据（直接覆盖，不只是未知区域）
            grid[gy_valid, gx_valid] = 100

        except Exception as e:
            # 静默失败，避免刷屏
            pass

    def rgb_callback(self, msg: SensorImage):
        """RGB回调 - VLM语义理解"""
        self.rgb_count += 1

        try:
            # 检查VLM是否启用和处理间隔
            current_time = time.time()
            if not self.vlm_enabled:
                return
            if (current_time - self.last_vlm_time) < self.vlm_processing_interval:
                return
            if self.vlm_processing:  # 如果正在处理，跳过
                return

            # 获取图像时间戳
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            image_timestamp = datetime.fromtimestamp(msg_time) if msg_time > 0 else datetime.now()

            # 🔑 关键：查找图像时刻对应的机器人位姿（时间戳对齐）
            robot_pose = self._find_pose_at_timestamp(image_timestamp)

            # 标记为处理中
            self.vlm_processing = True

            # 使用线程池异步处理VLM（避免阻塞回调）
            def process_vlm():
                try:
                    # 编码图像
                    image_data = self.vlm_client._encode_image_from_ros(msg)

                    # 调用VLM分析
                    self.get_logger().info(f"🔍 VLM分析图像 (时间戳: {image_timestamp.strftime('%H:%M:%S.%f')[:-3]})")

                    vlm_result = self.vlm_client.analyze_image(
                        image_data=image_data,
                        prompt=self._create_vlm_prompt(),
                        robot_pose=robot_pose,
                        timestamp=image_timestamp
                    )

                    # 更新WorldModel语义物体
                    if 'error' not in vlm_result:
                        self._update_semantic_objects(vlm_result, robot_pose)
                        self.vlm_count += 1
                        self.last_vlm_time = current_time

                        # 打印结果
                        objects = vlm_result.get('objects', [])
                        self.get_logger().info(f"📦 更新了 {len(objects)} 个语义物体到WorldModel")
                        self.get_logger().info(f"✅ VLM检测到 {len(objects)} 个物体")
                        for obj in objects[:5]:  # 只显示前5个
                            self.get_logger().info(f"   - {obj}")
                    else:
                        self.get_logger().warning(f"VLM错误: {vlm_result.get('error')}")

                except Exception as e:
                    self.get_logger().error(f"VLM处理异常: {e}")
                finally:
                    # 标记处理完成
                    self.vlm_processing = False

            # 提交到线程池（而不是创建新线程）
            self.vlm_executor.submit(process_vlm)

        except Exception as e:
            self.get_logger().error(f"RGB回调错误: {e}")
            self.vlm_processing = False  # 确保在出错时重置标志

    def _find_pose_at_timestamp(self, target_timestamp: datetime) -> Dict[str, float]:
        """
        查找给定时间戳对应的机器人位姿（时间戳对齐）

        Args:
            target_timestamp: 目标时间戳

        Returns:
            对应的机器人位姿字典，如果找不到则返回当前位姿
        """
        with self.pose_history_lock:
            if not self.pose_history:
                # 没有历史位姿，返回当前位姿
                return self.world_model.robot_position.copy() if hasattr(self.world_model, 'robot_position') else {'x': 0, 'y': 0, 'z': 0, 'yaw': 0}

            # 找到最接近的位姿
            min_diff = float('inf')
            closest_pose = None

            for pose_time, pose in self.pose_history:
                time_diff = abs((pose_time - target_timestamp).total_seconds())
                if time_diff < min_diff:
                    min_diff = time_diff
                    closest_pose = pose

            # 如果时间差太大（>1秒），可能有问题
            if min_diff > 1.0:
                self.get_logger().warning(
                    f"⚠️  位姿时间戳对齐差异较大: {min_diff:.2f}秒 "
                    f"(图像: {target_timestamp.strftime('%H:%M:%S.%f')[:-3]}, "
                    f"位姿: {closest_pose.get('x', 0):.2f}, {closest_pose.get('y', 0):.2f})"
                )

            return closest_pose or self.world_model.robot_position.copy()

    def _create_vlm_prompt(self) -> str:
        """创建VLM提示词"""
        return (
            "Analyze this image and identify all visible objects. "
            "For each object, provide: 1) Object name (door, person, building, car, obstacle, etc.), "
            "2) Relative position (left, center, right, far, near), "
            "3) Size estimate. "
            "Format your response as a JSON object: "
            "{\"objects\": [{\"name\": \"door\", \"position\": \"left\", \"size\": \"large\"}]}"
        )

    def _update_semantic_objects(self, vlm_result: Dict[str, Any], robot_pose: Dict[str, float]):
        """
        将VLM结果更新到WorldModel的语义物体

        Args:
            vlm_result: VLM分析结果
            robot_pose: 机器人位姿（图像时刻）
        """
        try:
            objects = vlm_result.get('objects', [])
            if not objects:
                return

            # 获取机器人位置
            robot_x = robot_pose.get('x', 0.0)
            robot_y = robot_pose.get('y', 0.0)
            robot_yaw = robot_pose.get('yaw', 0.0)

            for obj in objects:
                obj_name = obj.get('name', 'unknown').lower()
                position_hint = obj.get('position', 'center')
                size_hint = obj.get('size', 'medium')

                # 根据位置提示计算相对坐标
                offset_x, offset_y = self._calculate_position_offset(position_hint, size_hint)

                # 转换为世界坐标（考虑机器人朝向）
                world_x = robot_x + offset_x * math.cos(robot_yaw) - offset_y * math.sin(robot_yaw)
                world_y = robot_y + offset_x * math.sin(robot_yaw) + offset_y * math.cos(robot_yaw)

                # 创建或更新语义物体
                obj_id = f"vlm_{self.vlm_count}_{len(self.world_model.semantic_objects)}"

                # 使用SemanticObject类（如果存在）
                from brain.cognitive.world_model.semantic.semantic_object import SemanticObject, ObjectState

                semantic_obj = SemanticObject(
                    id=obj_id,
                    label=obj_name,
                    world_position=(world_x, world_y),
                    confidence=0.7,  # VLM默认置信度
                    state=ObjectState.DETECTED,
                    attributes={'source': 'vlm'}  # 🔑 标记为VLM检测，用于可视化过滤
                )

                # 添加到WorldModel
                self.world_model.semantic_objects[obj_id] = semantic_obj

            self.get_logger().info(f"📦 更新了 {len(objects)} 个语义物体到WorldModel")

        except Exception as e:
            self.get_logger().error(f"更新语义物体失败: {e}")

    def _calculate_position_offset(self, position_hint: str, size_hint: str) -> Tuple[float, float]:
        """
        根据位置提示计算相对偏移量

        Args:
            position_hint: 位置提示 (left, center, right, far, near)
            size_hint: 大小提示 (small, medium, large)

        Returns:
            (offset_x, offset_y) 相对机器人的偏移（米）
        """
        # 基础距离（根据大小）
        size_distance = {
            'small': 1.0,
            'medium': 2.0,
            'large': 4.0
        }.get(size_hint.lower(), 2.0)

        # 水平偏移（根据左右）
        horizontal_offset = {
            'left': -1.5,
            'center': 0.0,
            'right': 1.5
        }.get(position_hint.lower(), 0.0)

        # 垂直偏移（根据远近）
        vertical_offset = {
            'near': 0.5,
            'center': size_distance,
            'far': size_distance * 1.5
        }.get(position_hint.lower(), size_distance)

        return (horizontal_offset, vertical_offset)

    def _try_display_status(self):
        """尝试显示状态"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # 定期显示状态
        if elapsed - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            self._display_status()

    def _display_status(self):
        """显示当前状态"""
        print("\n" + "=" * 80)
        print(f"📊 认知世界状态可视化 (运行时长: {time.time() - self.start_time:.1f}秒)")
        print("=" * 80)

        # 1. 机器人状态
        robot_position = self.world_model.robot_position
        print(f"\n🤖 机器人位置:")
        print(f"   x: {robot_position.get('x', 0):.3f} m")
        print(f"   y: {robot_position.get('y', 0):.3f} m")
        print(f"   z: {robot_position.get('z', 0):.3f} m")
        print(f"   yaw: {self.world_model.robot_heading * 180 / math.pi:.1f}°")

        # 2. 占据栅格
        print(f"\n🗺️  占据栅格:")
        if self.world_model.current_map is not None:
            grid = self.world_model.current_map
            print(f"   形状: {grid.shape}")
            print(f"   分辨率: {self.world_model.map_resolution} m/cell")
            print(f"   总单元数: {grid.size:,}")
            print(f"   未知: {np.sum(grid == -1):,}")
            print(f"   空闲: {np.sum(grid == 0):,}")
            print(f"   占据: {np.sum(grid == 100):,}")
        else:
            print(f"   (栅格未初始化)")

        # 3. 语义物体
        semantic_count = len(self.world_model.semantic_objects)
        print(f"\n📦 语义物体: {semantic_count}")

        if semantic_count > 0:
            print(f"   物体列表:")
            for i, (obj_id, obj) in enumerate(list(self.world_model.semantic_objects.items())[:5]):
                print(f"   [{i+1}] {obj_id}: {obj.label}")
                if hasattr(obj, 'world_position') and obj.world_position:
                    wx, wy = obj.world_position
                    print(f"       位置: ({wx:.2f}, {wy:.2f}), 置信度: {obj.confidence:.2f}")

        # 4. 跟踪物体
        tracked_count = len(self.world_model.tracked_objects)
        print(f"\n🎯 跟踪物体: {tracked_count}")

        # 5. 探索前沿
        frontier_count = len(self.world_model.exploration_frontiers)
        print(f"\n🔍 探索前沿: {frontier_count}")

        # 6. 位姿历史
        pose_history_count = len(self.world_model.pose_history)
        print(f"\n📍 位姿历史: {pose_history_count} 个记录")

        # 7. 因果图统计（三模态融合 - 因果地图）
        if hasattr(self.world_model, 'causal_graph'):
            stats = self.world_model.get_causal_graph_statistics()
            print(f"\n🔗 因果图统计:")
            print(f"   节点数: {stats['num_nodes']}")
            print(f"   边数: {stats['num_edges']}")
            print(f"   高置信度边 (>0.7): {stats['high_confidence_edges']}")
            print(f"   平均置信度: {stats['avg_confidence']:.2f}")
            if stats['num_edges'] == 0:
                print(f"   💡 提示: 移动机器人或物体会触发因果检测")

        # 8. 统计信息
        print(f"\n📊 统计信息:")
        print(f"   里程计更新: {self.odom_count} 次")
        print(f"   点云更新: {self.pointcloud_count} 次")
        print(f"   RGB图像接收: {self.rgb_count} 次")
        print(f"   VLM分析: {self.vlm_count} 次")
        print(f"   更新频率: {self.odom_count / (time.time() - self.start_time):.2f} Hz")

        # VLM统计
        vlm_stats = self.vlm_client.get_statistics()
        if vlm_stats['total_requests'] > 0:
            print(f"\n🤖 VLM统计:")
            print(f"   总请求: {vlm_stats['total_requests']}")
            print(f"   成功: {vlm_stats['successful_requests']}")
            print(f"   失败: {vlm_stats['failed_requests']}")
            print(f"   成功率: {vlm_stats['success_rate']:.1%}")

        print("\n" + "=" * 80)
        print("💡 RViz2中应该看到:")
        print("   - 灰色/白色/黑色地图 (未知/空闲/占据)")
        print("   - 彩色物体标签 (门=蓝色, 人=红色, 建筑=绿色)")
        print("   - 绿色机器人轨迹线")
        print("   - 箭头+文字 (探索边界, 优先级+距离)")
        print("=" * 80 + "\n")


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    # 创建可视化节点
    viz_node = CognitiveWorldStateViz(duration_seconds=600.0)

    # 使用多线程执行器
    executor = MultiThreadedExecutor()
    executor.add_node(viz_node)
    executor.add_node(viz_node.visualizer)  # 添加可视化器节点到执行器

    print("\n" + "=" * 80)
    print("🚀 认知层世界状态可视化演示")
    print("=" * 80)
    print("\n📡 正在订阅ROS2话题:")
    print("   /chassis/odom")
    print("   /front_3d_lidar/lidar_points")
    print("   /front_stereo_camera/left/image_raw")
    print("\n📤 正在发布可视化话题:")
    print("   /world_model/semantic_grid")
    print("   /world_model/semantic_markers")
    print("   /world_model/belief_markers")
    print("   /world_model/trajectory")
    print("   /world_model/frontiers")
    print("   /world_model/change_events")
    print("   /vlm/detections")
    print("\n💡 在另一个终端启动RViz2查看可视化:")
    print("   rviz2 -d config/rviz2/cognitive_world_model_correct.rviz")
    print("\n⏱️  演示时长: 600秒 (10分钟)")
    print("📊 状态显示间隔: 5秒")
    print("\n💡 提示: 如果没有实时传感器，可以先播放rosbag:")
    print("   export ROS_DOMAIN_ID=42")
    print("   ros2 bag play <rosbag文件> --loop")
    print("\n按Ctrl+C停止\n")
    print("=" * 80 + "\n")

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    finally:
        # 清理VLM线程池
        print("\n🧹 清理资源...")
        if hasattr(viz_node, 'vlm_executor'):
            viz_node.vlm_executor.shutdown(wait=True)
            print("   ✅ VLM线程池已关闭")

        # 保存最终状态
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n✅ 已停止")
        print(f"   总里程计更新: {viz_node.odom_count}次")
        print(f"   总点云更新: {viz_node.pointcloud_count}次")
        print(f"   VLM检测次数: {viz_node.vlm_count}次")

        viz_node.destroy_node()

        # 优雅地关闭ROS2
        try:
            rclpy.shutdown()
        except (AttributeError, Exception):
            pass


if __name__ == '__main__':
    main()
