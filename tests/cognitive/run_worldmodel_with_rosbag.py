#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的WorldModel可视化Pipeline

从rosbag读取传感器数据 → 感知层处理 → 认知层WorldModel → 可视化 → RViz

Usage:
    # 确保rosbag在运行（ROS_DOMAIN_ID=42）
    ros2 bag play /home/yangyuhui/sim_data_bag --loop

    # 启动此脚本
    python3 tests/cognitive/run_worldmodel_with_rosbag.py

    # 在另一个终端启动RViz
    rviz2 -d rviz/semantic_worldmodel.rviz
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image as SensorImage, PointCloud2, CompressedImage
from nav_msgs.msg import Odometry
import struct
import cv2  # 用于解压缩CompressedImage

# Cognitive layer
from brain.cognitive.world_model.world_model import WorldModel
from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer
from brain.perception.utils.coordinates import quaternion_to_euler


class WorldModelPipeline(Node):
    """完整的WorldModel处理pipeline"""

    def __init__(self):
        super().__init__('worldmodel_pipeline')

        self.get_logger().info("=" * 80)
        self.get_logger().info("🚀 WorldModel完整Pipeline启动")
        self.get_logger().info("=" * 80)

        # 1. 初始化WorldModel
        self.get_logger().info("初始化WorldModel...")
        world_config = {
            'map_resolution': 0.1,  # 10cm per cell
            'map_size': 100.0,      # 100m x 100m
        }

        self.world_model = WorldModel(config=world_config)
        self.get_logger().info("✅ WorldModel初始化完成")

        # 2. 初始化可视化器
        self.get_logger().info("初始化可视化器...")
        self.visualizer = WorldModelVisualizer(
            world_model=self.world_model,
            publish_rate=2.0  # 2Hz
        )
        self.get_logger().info("✅ 可视化器初始化完成")

        # 3. 设置QoS（兼容rosbag）
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 4. 创建订阅者（支持仿真环境）
        self.get_logger().info("创建传感器订阅者...")

        # 根据环境检测订阅topics
        import os
        domain_id = os.environ.get('ROS_DOMAIN_ID', '42')
        self.get_logger().info(f"🔍 检测到的ROS_DOMAIN_ID: '{domain_id}' (类型: {type(domain_id)})")
        self.simulation_mode = (domain_id == '0')  # domain 0是仿真环境

        if self.simulation_mode:
            self.get_logger().info("检测到仿真环境 (Domain 0)")
            self.get_logger().info("使用仿真topics: /car3/*")
            odom_topic = '/car3/car_info'  # 仿真环境的真实odom话题
            rgb_topic = '/car3/rgbImage'
            lidar_topic = '/car3/lidar_points'
        else:
            self.get_logger().info("检测到rosbag环境 (Domain 42)")
            self.get_logger().info("使用rosbag topics")
            odom_topic = '/chassis/odom'
            rgb_topic = '/front_stereo_camera/left/image_raw'
            lidar_topic = '/front_3d_lidar/lidar_points'

        # 里程计
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )

        # RGB图像（用于VLM语义识别）
        # 仿真环境使用CompressedImage，rosbag使用普通Image
        rgb_msg_type = CompressedImage if self.simulation_mode else SensorImage
        self.rgb_sub = self.create_subscription(
            rgb_msg_type,
            rgb_topic,
            self.rgb_callback,
            qos_profile
        )

        # 点云（用于占据栅格）
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            lidar_topic,
            self.pointcloud_callback,
            qos_profile
        )

        self.get_logger().info("✅ 订阅者创建完成")
        if self.simulation_mode:
            self.get_logger().info(f"   - {odom_topic} (Odometry)")
            self.get_logger().info(f"   - {rgb_topic} (RGB)")
            self.get_logger().info(f"   - {lidar_topic} (PointCloud)")
            self.get_logger().info("   注意: 仿真环境使用静态odom")

        # 状态变量
        self.update_count = 0
        self.last_display_time = 0
        self.display_interval = 5.0  # 每5秒显示一次状态
        self.start_time = time.time()

        # 当前数据
        self.current_rgb = None
        self.current_pointcloud = None

        # 最新odom数据（用于点云转换）
        self.latest_odom_x = 0.0
        self.latest_odom_y = 0.0
        self.latest_odom_yaw = 0.0

        self.get_logger().info("=" * 80)
        self.get_logger().info("📡 Pipeline已启动，等待传感器数据...")
        self.get_logger().info("=" * 80)

    def odom_callback(self, msg: Odometry):
        """里程计回调"""
        # 提取位姿
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_euler((ori.x, ori.y, ori.z, ori.w))

        # 每100次打印一次位置
        if self.update_count % 100 == 0 and self.update_count > 0:
            self.get_logger().info(f"🤖 里程计 #{self.update_count}: 位置({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")

        # 保存最新odom值（供点云回调使用）
        self.latest_odom_x = pos.x
        self.latest_odom_y = pos.y
        self.latest_odom_yaw = yaw

        # 创建感知数据
        perception_data = {
            'timestamp': datetime.now(),
            'pose': {
                'x': pos.x,
                'y': pos.y,
                'z': pos.z,
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw
            },
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
        self.update_count += 1

        # 定期显示状态
        self._try_display_status()

    def rgb_callback(self, msg):
        """RGB图像回调 - 支持Image和CompressedImage"""
        try:
            # 处理CompressedImage (仿真环境)
            if isinstance(msg, CompressedImage):
                # 解压缩JPEG/PNG图像
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    # OpenCV默认是BGR，转换为RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.current_rgb = img

            # 处理普通Image (rosbag环境)
            elif isinstance(msg, SensorImage):
                if msg.encoding in ['rgb8', 'bgr8']:
                    dtype = np.uint8
                    arr = np.frombuffer(msg.data, dtype=dtype)
                    n_channels = 3
                    arr = arr.reshape((msg.height, msg.width, n_channels))
                    if msg.encoding == 'bgr8':
                        arr = arr[:, :, ::-1].copy()
                    self.current_rgb = arr

        except Exception as e:
            self.get_logger().error(f"RGB回调错误: {e}", throttle_duration_sec=5.0)

    def pointcloud_callback(self, msg: PointCloud2):
        """点云回调 - 更新占据栅格"""
        try:
            # 读取点云数据（手动解析，不使用sensor_msgs_py）
            points = self._read_pointcloud(msg)

            if len(points) == 0:
                return

            # 更新占据栅格
            self._update_occupancy_grid(points)

        except Exception as e:
            self.get_logger().error(f"点云回调错误: {e}", throttle_duration_sec=5.0)

    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        """读取点云数据（手动解析）"""
        # 将点云数据转换为字节数组
        point_step = msg.point_step
        row_step = msg.row_step

        # 查找x, y, z字段的偏移
        x_offset = y_offset = z_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset

        if x_offset is None or y_offset is None or z_offset is None:
            return np.array([])

        # 解析点云
        points = []
        data = msg.data

        for i in range(0, len(data), point_step):
            if i + point_step > len(data):
                break

            # 提取x, y, z（假设是float32）
            try:
                x_bytes = data[i + x_offset:i + x_offset + 4]
                y_bytes = data[i + y_offset:i + y_offset + 4]
                z_bytes = data[i + z_offset:i + z_offset + 4]

                x = struct.unpack('f', x_bytes)[0]
                y = struct.unpack('f', y_bytes)[0]
                z = struct.unpack('f', z_bytes)[0]

                # 过滤NaN和无限值
                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    if not (np.isinf(x) or np.isinf(y) or np.isinf(z)):
                        points.append([x, y, z])
            except:
                continue

        return np.array(points)

    def _initialize_map(self, grid_size: int, map_size: float):
        """初始化占据栅格地图 - 地图中心跟随机器人"""
        # ✅ 地图以机器人为中心，确保周围点云都能被包含
        robot_x = self.latest_odom_x
        robot_y = self.latest_odom_y

        self.world_model.map_origin = (
            robot_x - map_size / 2.0,  # 机器人左侧150米
            robot_y - map_size / 2.0   # 机器人下方150米
        )

        # 创建新的空地图（会丢失之前的占据数据）
        # TODO: 可以改进为保留旧数据，只扩展边界
        self.world_model.current_map = np.full(
            (grid_size, grid_size),
            -1,  # -1表示未知
            dtype=np.int8
        )

        self.get_logger().info(f"初始化占据栅格: {grid_size}x{grid_size}")
        self.get_logger().info(f"   地图原点设置为 ({self.world_model.map_origin[0]:.2f}, {self.world_model.map_origin[1]:.2f})")

    def _update_occupancy_grid(self, points: np.ndarray):
        """使用点云更新占据栅格"""
        map_size = 300.0  # 300m x 300m (仿真环境点云距离可达100m+)
        grid_size = int(map_size / self.world_model.map_resolution)

        # 初始化或重新调整地图
        if self.world_model.current_map is None:
            # 首次初始化
            self._initialize_map(grid_size, map_size)

        # 更新栅格（禁用边界重新初始化，因为地图已经足够大）
        origin_x, origin_y = self.world_model.map_origin
        resolution = self.world_model.map_resolution

        # 获取机器人当前位姿（用于坐标转换）
        # 使用最新odom值，而不是world_model中的值（可能有延迟）
        robot_x = self.latest_odom_x
        robot_y = self.latest_odom_y
        robot_yaw = self.latest_odom_yaw

        # 每50000个点云打印一次调试信息
        if hasattr(self, '_pointcloud_count'):
            self._pointcloud_count += len(points)
        else:
            self._pointcloud_count = len(points)

        # 统计过滤情况
        filtered_count = 0
        added_count = 0

        # 预计算旋转矩阵
        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        for point in points:
            x, y, z = point[0], point[1], point[2]

            # ✅ 过滤1: 高度过滤 - 使用绝对值，允许正负z值
            # 过滤过于接近地面的点（可能是地面噪声）
            if abs(z) < 0.1:
                filtered_count += 1
                continue

            # ✅ 过滤2: 高度限制 - 忽略过高的点（可能是不相关的物体）
            # 仿真环境的z值范围更大（-30到+30），所以使用更宽松的限制
            if abs(z) > 50.0:
                filtered_count += 1
                continue

            # ✅ 过滤3: 距离限制 - 仿真环境需要更远的距离
            dist = np.sqrt(x**2 + y**2)
            if dist > 150.0 or dist < 0.5:
                filtered_count += 1
                continue

            # 点云坐标在机器人坐标系，需要转换到世界坐标系
            # 2D旋转 + 平移
            point_world_x = robot_x + x * cos_yaw - y * sin_yaw
            point_world_y = robot_y + x * sin_yaw + y * cos_yaw

            # 计算相对于地图原点的坐标
            rel_x = point_world_x - origin_x
            rel_y = point_world_y - origin_y

            # 检查是否在地图范围内 (300m x 300m地图，边界为±150)
            if abs(rel_x) > 150 or abs(rel_y) > 150:
                filtered_count += 1
                continue

            # 世界坐标转栅格坐标
            gx = int(rel_x / resolution)
            gy = int(rel_y / resolution)

            # 检查边界
            if 0 <= gx < self.world_model.current_map.shape[1] and \
               0 <= gy < self.world_model.current_map.shape[0]:
                # ✅ 使用概率更新而不是直接覆盖
                current_val = self.world_model.current_map[gy, gx]
                if current_val == -1:  # 未知
                    self.world_model.current_map[gy, gx] = 50  # 初始值
                    added_count += 1
                elif current_val < 100:  # 未完全占据
                    # 增加占据概率
                    self.world_model.current_map[gy, gx] = min(100, current_val + 10)
                    added_count += 1
            else:
                filtered_count += 1

        self.world_model.last_update = datetime.now()

        # 打印调试信息
        if self._pointcloud_count % 50000 < len(points):
            self.get_logger().info(f"🔄 点云转换: 机器人({robot_x:.2f}, {robot_y:.2f}), 航向{np.degrees(robot_yaw):.1f}°, 点数{len(points)}")
            self.get_logger().info(f"   地图origin: ({origin_x:.2f}, {origin_y:.2f}), 机器人相对位置: ({robot_x - origin_x:.2f}, {robot_y - origin_y:.2f})")
            self.get_logger().info(f"   过滤: {filtered_count}, 添加到地图: {added_count}")

            # ✅ 采样前10个点云，查看其分布
            sample_size = min(10, len(points))
            for i in range(sample_size):
                x, y, z = points[i][0], points[i][1], points[i][2]
                dist = np.sqrt(x**2 + y**2)
                self.get_logger().info(f"   样本点{i}: x={x:.2f}, y={y:.2f}, z={z:.2f}, dist={dist:.2f}")

    def _try_display_status(self):
        """尝试显示状态"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        if elapsed - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            self._display_status()

    def _display_status(self):
        """显示当前状态"""
        print("\n" + "=" * 80)
        print(f"📊 WorldModel Pipeline状态 (运行: {time.time() - self.start_time:.1f}秒)")
        print("=" * 80)

        # 机器人状态
        robot_pos = self.world_model.robot_position
        print(f"\n🤖 机器人:")
        print(f"   位置: ({robot_pos.get('x', 0):.2f}, {robot_pos.get('y', 0):.2f}, {robot_pos.get('z', 0):.2f})")
        print(f"   航向: {self.world_model.robot_heading:.1f}°")

        # 占据栅格
        print(f"\n🗺️  占据栅格:")
        if self.world_model.current_map is not None:
            grid = self.world_model.current_map
            total = grid.size
            unknown = np.sum(grid == -1)
            free = np.sum(grid == 0)
            occupied = np.sum(grid == 100)

            print(f"   形状: {grid.shape}")
            print(f"   总单元: {total:,}")
            print(f"   未知: {unknown:,} ({100*unknown/total:.1f}%)")
            print(f"   空闲: {free:,} ({100*free/total:.1f}%)")
            print(f"   占据: {occupied:,} ({100*occupied/total:.1f}%)")
        else:
            print(f"   (未初始化)")

        # 语义物体
        print(f"\n📦 语义物体: {len(self.world_model.semantic_objects)}")

        # 探索前沿
        print(f"\n🔍 探索前沿: {len(self.world_model.exploration_frontiers)}")

        # 位姿历史
        print(f"\n📍 位姿历史: {len(self.world_model.pose_history)} 个记录")

        # 统计
        print(f"\n📊 统计:")
        print(f"   总更新: {self.update_count}")
        print(f"   频率: {self.update_count / (time.time() - self.start_time):.2f} Hz")

        print("\n" + "=" * 80)
        print("💡 RViz可视化话题:")
        print("   rviz2 -d rviz/semantic_worldmodel.rviz")
        print("=" * 80)


def main(args=None):
    """主函数"""
    # 设置ROS_DOMAIN_ID (只在没有设置时使用默认值)
    if 'ROS_DOMAIN_ID' not in os.environ:
        os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init(args=args)

    # 创建pipeline节点
    pipeline = WorldModelPipeline()
    visualizer = pipeline.visualizer  # 获取visualizer节点

    print("\n" + "=" * 80)
    print("🚀 WorldModel可视化Pipeline已启动")
    print("=" * 80)
    print("\n✅ 系统状态:")
    print("   • WorldModel: 运行中")
    print("   • Visualizer: 运行中 (2Hz)")
    print("   • 传感器订阅: 已连接")
    print("\n📡 发布话题:")
    print("   • /world_model/semantic_grid (OccupancyGrid)")
    print("   • /world_model/semantic_markers (MarkerArray)")
    print("   • /world_model/trajectory (Path)")
    print("   • /world_model/frontiers (MarkerArray)")
    print("\n💡 提示:")
    print("   在另一个终端运行:")
    print("   rviz2 -d rviz/semantic_worldmodel.rviz")
    print("=" * 80)

    # 使用MultiThreadedExecutor来spin多个节点
    executor = MultiThreadedExecutor()
    executor.add_node(pipeline)
    executor.add_node(visualizer)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，正在关闭...")
    finally:
        executor.shutdown()
        pipeline.destroy_node()
        visualizer.destroy_node()
        try:
            rclpy.shutdown()
        except:
            pass

        print("\n" + "=" * 80)
        print("✅ Pipeline已关闭")
        print("=" * 80)


if __name__ == '__main__':
    main()
