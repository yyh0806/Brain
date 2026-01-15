#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E Test: WorldModel Semantic Visualization

测试WorldModel语义可视化功能。

Usage:
    # Terminal 1: 播放rosbag
    export ROS_DOMAIN_ID=42
    ros2 bag play /home/yangyuhui/sim_data_bag --loop

    # Terminal 2: 启动可视化测试
    python tests/cognitive/test_visualize_semantic_worldmodel.py

    # Terminal 3: 启动RViz查看可视化
    rviz2 -d rviz/semantic_worldmodel.rviz

Expected Results:
    - RViz中显示语义占据栅格（不同颜色代表不同语义）
    - RViz中显示语义物体标签（3D文字）
    - RViz中显示机器人轨迹（绿色路径线）
    - RViz中显示探索前沿（绿色箭头）
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any

# Add Brain to path
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image as SensorImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

# Cognitive layer imports
from brain.cognitive.world_model.world_model import WorldModel
from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer
from brain.perception.utils.coordinates import quaternion_to_euler


def ros_image_to_numpy(img_msg):
    """Simple ROS Image to numpy conversion"""
    import numpy as np

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


class WorldModelVisualizationTest(Node):
    """WorldModel可视化测试节点"""

    def __init__(self, duration_seconds: float = 30.0):
        super().__init__('worldmodel_visualization_test')

        self.duration_seconds = duration_seconds
        self.start_time = time.time()

        # 1. 初始化WorldModel
        world_config = {
            'map_resolution': 0.1,  # 10cm per cell
            'map_size': 50.0,      # 50m x 50m
        }

        self.get_logger().info("=" * 80)
        self.get_logger().info("🎯 WorldModel语义可视化测试")
        self.get_logger().info("=" * 80)

        self.get_logger().info("正在初始化WorldModel...")
        self.world_model = WorldModel(config=world_config)

        self.get_logger().info("✅ WorldModel初始化完成")
        self.get_logger().info(f"   地图分辨率: {self.world_model.map_resolution}m/cell")
        self.get_logger().info(f"   地图原点: {self.world_model.map_origin}")

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
        self.get_logger().info("     - /world_model/trajectory")
        self.get_logger().info("     - /world_model/frontiers")

        # 传感器数据缓冲
        self.current_odometry = None
        self.current_rgb_image = None

        # 设置订阅者
        self._setup_subscribers()

        # 统计信息
        self.update_count = 0
        self.last_display_time = 0
        self.display_interval = 5.0  # 每5秒显示一次状态

    def _setup_subscribers(self):
        """设置ROS2订阅者"""
        self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )
        self.create_subscription(
            SensorImage,
            '/front_stereo_camera/left/image_raw',
            self.rgb_callback,
            10
        )

        self.get_logger().info("=" * 80)
        self.get_logger().info("📡 已创建ROS2订阅者:")
        self.get_logger().info("   - /chassis/odom (Odometry)")
        self.get_logger().info("   - /front_stereo_camera/left/image_raw (RGB Image)")
        self.get_logger().info("=" * 80)

    def odom_callback(self, msg: Odometry):
        """里程计回调 - 更新WorldModel"""
        self.current_odometry = msg

        # 更新WorldModel
        self._update_world_model_from_odometry(msg)

        # 显示状态
        self._try_display_status()

        # 检查运行时长
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration_seconds:
            self.get_logger().info("=" * 80)
            self.get_logger().info("✅ 测试完成")
            self.get_logger().info(f"   运行时长: {elapsed:.1f}秒")
            self.get_logger().info(f"   总更新次数: {self.update_count}")
            self.get_logger().info("=" * 80)
            rclpy.shutdown()

    def rgb_callback(self, msg: SensorImage):
        """RGB回调 - 存储图像"""
        try:
            self.current_rgb_image = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"RGB回调错误: {e}")

    def _update_world_model_from_odometry(self, msg: Odometry):
        """从里程计数据更新WorldModel"""
        # 提取位姿
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_euler((ori.x, ori.y, ori.z, ori.w))

        # 创建字典格式的感知数据
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
        print(f"📊 WorldModel可视化状态 (运行时长: {time.time() - self.start_time:.1f}秒)")
        print("=" * 80)

        # 1. 机器人状态
        robot_position = self.world_model.robot_position
        print(f"\n🤖 机器人位置:")
        print(f"   x: {robot_position.get('x', 0):.3f} m")
        print(f"   y: {robot_position.get('y', 0):.3f} m")
        print(f"   z: {robot_position.get('z', 0):.3f} m")

        # 2. 占据栅格
        print(f"\n🗺️  占据栅格:")
        if self.world_model.current_map is not None:
            import numpy as np
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
                    print(f"       位置: ({wx:.2f}, {wy:.2f})")

        # 4. 跟踪物体
        tracked_count = len(self.world_model.tracked_objects)
        print(f"\n🎯 跟踪物体: {tracked_count}")

        # 5. 探索前沿
        frontier_count = len(self.world_model.exploration_frontiers)
        print(f"\n🔍 探索前沿: {frontier_count}")

        # 6. 位姿历史
        pose_history_count = len(self.world_model.pose_history)
        print(f"\n📍 位姿历史: {pose_history_count} 个记录")

        # 7. 可视化统计
        print(f"\n📊 可视化统计:")
        print(f"   总更新次数: {self.update_count}")
        print(f"   更新频率: {self.update_count / (time.time() - self.start_time):.2f} Hz")

        print("\n" + "=" * 80)
        print("💡 提示: 在RViz中查看可视化结果:")
        print("   rviz2 -d rviz/semantic_worldmodel.rviz")
        print("=" * 80)


def main(args=None):
    """主函数"""
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init(args=args)

    # 创建测试节点
    test_node = WorldModelVisualizationTest(duration_seconds=30.0)

    # 使用多线程执行器
    executor = MultiThreadedExecutor()
    executor.add_node(test_node)

    print("\n" + "=" * 80)
    print("🚀 WorldModel语义可视化测试已启动")
    print("=" * 80)
    print("\n环境配置:")
    print("  • ROS_DOMAIN_ID: 42")
    print("  • rosbag: /home/yangyuhui/sim_data_bag")
    print("  • 测试时长: 30秒")
    print("\n正在收集数据并发布可视化...")
    print(f"显示间隔: {test_node.display_interval}秒")
    print(f"可视化发布频率: {test_node.visualizer.publish_rate} Hz")
    print("\n" + "=" * 80)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    finally:
        # 保存最终状态
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"/media/yangyuhui/CODES1/Brain/tests/cognitive/worldmodel_viz_state_{timestamp}.json"

        # 收集数据
        import numpy as np
        data = {
            "metadata": {
                "capture_time": datetime.now().isoformat(),
                "update_count": test_node.update_count,
                "duration": time.time() - test_node.start_time
            },
            "robot_state": {
                "position": test_node.world_model.robot_position,
                "velocity": test_node.world_model.robot_velocity,
                "heading": test_node.world_model.robot_heading,
                "battery": test_node.world_model.battery_level,
                "signal": test_node.world_model.signal_strength
            },
            "occupancy_grid": {
                "shape": test_node.world_model.current_map.shape if test_node.world_model.current_map is not None else None,
                "resolution": test_node.world_model.map_resolution,
                "origin": test_node.world_model.map_origin,
                "cell_stats": {
                    "total": test_node.world_model.current_map.size if test_node.world_model.current_map is not None else 0,
                    "unknown": int(np.sum(test_node.world_model.current_map == -1)) if test_node.world_model.current_map is not None else 0,
                    "free": int(np.sum(test_node.world_model.current_map == 0)) if test_node.world_model.current_map is not None else 0,
                    "occupied": int(np.sum(test_node.world_model.current_map == 100)) if test_node.world_model.current_map is not None else 0,
                }
            },
            "semantic_objects": {
                "count": len(test_node.world_model.semantic_objects),
            },
            "exploration": {
                "frontiers_count": len(test_node.world_model.exploration_frontiers),
                "max_frontiers": test_node.world_model.max_frontiers,
                "explored_count": len(test_node.world_model.explored_positions)
            },
            "history": {
                "pose_history_count": len(test_node.world_model.pose_history),
                "change_history_count": len(test_node.world_model.change_history)
            },
            "environment": test_node.world_model.weather
        }

        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print("\n" + "=" * 80)
        print("📊 测试完成")
        print("=" * 80)
        print(f"  • 总更新次数: {test_node.update_count}")
        print(f"  • 运行时长: {time.time() - test_node.start_time:.1f}秒")
        print(f"  • 输出文件: {json_file}")
        print("=" * 80)

        test_node.destroy_node()

        # 优雅地关闭ROS2（处理Galactic的shutdown bug）
        try:
            rclpy.shutdown()
        except (AttributeError, Exception) as e:
            # ROS2 Galactic有已知的shutdown bug，忽略
            pass


if __name__ == '__main__':
    main()
