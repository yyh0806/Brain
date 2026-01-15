#!/usr/bin/env python3
"""
E2E Test: Display WorldModel Internal State

This test displays the complete internal state of WorldModel,
including all fields and their actual values.

Usage:
    export ROS_DOMAIN_ID=42
    ros2 bag play /home/yangyuhui/sim_data_bag --loop
    ollama run llava:7b  # Optional, for semantic understanding
    python test_display_worldmodel.py
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Add Brain to path
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

# Cognitive layer imports
from brain.cognitive.world_model.world_model import WorldModel
from brain.perception.utils.coordinates import quaternion_to_euler


def ros_image_to_numpy(img_msg):
    """Simple ROS Image to numpy conversion"""
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


class WorldModelDisplay(Node):
    """Display WorldModel internal state"""

    def __init__(self, duration_seconds: float = 15.0):
        super().__init__('worldmodel_display')

        self.duration_seconds = duration_seconds
        self.start_time = time.time()

        # Initialize WorldModel
        world_config = {
            'map_resolution': 0.1,  # 10cm per cell
            'map_size': 50.0,      # 50m x 50m
        }
        self.world_model = WorldModel(config=world_config)

        self.get_logger().info("✅ WorldModel initialized")
        self.get_logger().info(f"   Map resolution: {self.world_model.map_resolution}m/cell")

        # Sensor data buffers
        self.current_odometry = None
        self.current_rgb_image = None

        # Setup subscribers
        self._setup_subscribers()

        # Display interval
        self.last_display_time = 0
        self.display_interval = 2.0  # Display every 2 seconds

        # Update counter
        self.update_count = 0

    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        self.create_subscription(Odometry, '/chassis/odom', self.odom_callback, 10)
        self.create_subscription(SensorImage, '/front_stereo_camera/left/image_raw',
                             self.rgb_callback, 10)

        self.get_logger().info("📡 Subscribers created:")
        self.get_logger().info("   - /chassis/odom (Odometry)")
        self.get_logger().info("   - /front_stereo_camera/left/image_raw (RGB Image)")

    def odom_callback(self, msg: Odometry):
        """Odometry callback - update WorldModel"""
        self.current_odometry = msg

        # Update WorldModel with odometry
        self._update_world_model_from_odometry(msg)

        # Try to display
        self._try_display()

    def rgb_callback(self, msg: SensorImage):
        """RGB callback - store image"""
        try:
            self.current_rgb_image = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def _update_world_model_from_odometry(self, msg: Odometry):
        """Update WorldModel from odometry data"""
        # Extract pose
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        roll, pitch, yaw = quaternion_to_euler((ori.x, ori.y, ori.z, ori.w))

        # Create dict format perception data (WorldModel expects dict)
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

        # Update WorldModel
        self.world_model.update_from_perception(perception_data)
        self.update_count += 1

    def _try_display(self):
        """Try to display WorldModel state"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Check if display time
        if elapsed - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            self._display_worldmodel_state()

        # Check duration
        if elapsed >= self.duration_seconds:
            self.get_logger().info("✅ Test completed")
            rclpy.shutdown()

    def _display_worldmodel_state(self):
        """Display complete WorldModel internal state"""
        print("\n" + "=" * 80)
        print(f"WorldModel 内部状态展示 (更新次数: {self.update_count})")
        print("=" * 80)
        print(f"显示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"运行时长: {time.time() - self.start_time:.1f}秒")

        # 1. Robot State
        print("\n" + "-" * 80)
        print("1. 🤖 机器人状态 (Robot State)")
        print("-" * 80)

        robot_position = self.world_model.robot_position
        print(f"位置 (Position):")
        print(f"  x: {robot_position.get('x', 0):.3f} m")
        print(f"  y: {robot_position.get('y', 0):.3f} m")
        print(f"  z: {robot_position.get('z', 0):.3f} m")
        print(f"  lat: {robot_position.get('lat', 0):.6f}°")
        print(f"  lon: {robot_position.get('lon', 0):.6f}°")
        print(f"  alt: {robot_position.get('alt', 0):.3f} m")

        robot_velocity = self.world_model.robot_velocity
        print(f"\n速度 (Velocity):")
        print(f"  vx: {robot_velocity.get('vx', 0):.3f} m/s")
        print(f"  vy: {robot_velocity.get('vy', 0):.3f} m/s")
        print(f"  vz: {robot_velocity.get('vz', 0):.3f} m/s")

        print(f"\n航向 (Heading): {self.world_model.robot_heading:.1f}°")
        print(f"电池 (Battery): {self.world_model.battery_level:.1f}%")
        print(f"信号强度 (Signal): {self.world_model.signal_strength:.1f}%")

        # 2. Occupancy Grid
        print("\n" + "-" * 80)
        print("2. 🗺️ 占据栅格 (Occupancy Grid)")
        print("-" * 80)

        if self.world_model.current_map is not None:
            grid = self.world_model.current_map
            print(f"栅格形状: {grid.shape}")
            print(f"分辨率: {self.world_model.map_resolution} m/cell")
            print(f"原点: ({self.world_model.map_origin[0]:.1f}, {self.world_model.map_origin[1]:.1f})")

            # Count cell states
            total_cells = grid.size
            unknown_cells = np.sum(grid == -1)
            free_cells = np.sum(grid == 0)
            occupied_cells = np.sum(grid == 100)

            print(f"\n栅格统计:")
            print(f"  总单元数: {total_cells:,}")
            print(f"  未知 (-1): {unknown_cells:,} ({100*unknown_cells/total_cells:.1f}%)")
            print(f"  空闲 (0): {free_cells:,} ({100*free_cells/total_cells:.1f}%)")
            print(f"  占据 (100): {occupied_cells:,} ({100*occupied_cells/total_cells:.1f}%)")

            # Sample occupancy at robot position
            rx, ry = robot_position.get('x', 0), robot_position.get('y', 0)
            is_occupied = self.world_model.is_occupied_at(rx, ry)
            is_free = self.world_model.is_free_at(rx, ry)
            print(f"\n机器人位置 ({rx:.2f}, {ry:.2f}) 状态:")
            print(f"  是否占据: {is_occupied}")
            print(f"  是否空闲: {is_free}")
        else:
            print("  (栅格未初始化)")

        # 3. Semantic Objects
        print("\n" + "-" * 80)
        print("3. 📦 语义物体 (Semantic Objects)")
        print("-" * 80)

        semantic_objects = self.world_model.semantic_objects
        print(f"语义物体数量: {len(semantic_objects)}")
        print(f"最大容量: {self.world_model.max_semantic_objects}")

        if semantic_objects:
            print(f"\n物体列表:")
            for i, (obj_id, obj) in enumerate(list(semantic_objects.items())[:10]):
                print(f"\n  [{i+1}] ID: {obj_id}")
                print(f"      标签: {obj.label}")
                if hasattr(obj, 'world_position'):
                    wx, wy = obj.world_position
                    print(f"      世界位置: ({wx:.2f}, {wy:.2f})")
                print(f"      状态: {obj.state}")
                print(f"      置信度: {obj.confidence:.2f}")
                print(f"      描述: {obj.description[:50]}..." if len(obj.description) > 50 else f"      描述: {obj.description}")
                if hasattr(obj, 'first_seen'):
                    print(f"      首次观测: {obj.first_seen.strftime('%H:%M:%S')}")
                    print(f"      最后观测: {obj.last_seen.strftime('%H:%M:%S')}")
                print(f"      观测次数: {obj.observation_count}")
                print(f"      是否目标: {obj.is_target}")
        else:
            print("  (暂无语义物体)")

        # 4. Tracked Objects
        print("\n" + "-" * 80)
        print("4. 🎯 跟踪物体 (Tracked Objects)")
        print("-" * 80)

        tracked_objects = self.world_model.tracked_objects
        print(f"跟踪物体数量: {len(tracked_objects)}")

        if tracked_objects:
            for obj_id, obj in list(tracked_objects.items())[:5]:
                print(f"  - {obj_id}: {obj}")
        else:
            print("  (暂无跟踪物体)")

        # 5. Exploration Frontiers
        print("\n" + "-" * 80)
        print("5. 🔍 探索前沿 (Exploration Frontiers)")
        print("-" * 80)

        frontiers = self.world_model.exploration_frontiers
        print(f"前沿数量: {len(frontiers)}")
        print(f"最大前沿数: {self.world_model.max_frontiers}")

        if frontiers:
            print(f"\n前沿点:")
            for i, frontier in enumerate(frontiers[:5]):
                print(f"  [{i+1}] ID: {frontier.id}")
                print(f"      位置: ({frontier.center_x:.1f}, {frontier.center_y:.1f})")
                print(f"      优先级: {frontier.priority}")
                print(f"      单元格数: {frontier.size}")
        else:
            print("  (暂无前沿)")

        # 6. Pose History
        print("\n" + "-" * 80)
        print("6. 📍 位姿历史 (Pose History)")
        print("-" * 80)

        pose_history = self.world_model.pose_history
        print(f"历史记录数: {len(pose_history)}")
        print(f"最大历史数: {self.world_model.max_pose_history}")

        if pose_history:
            print(f"\n最近轨迹:")
            for i, pose_entry in enumerate(pose_history[-10:]):
                timestamp = pose_entry.get('timestamp', 'N/A')
                x = pose_entry.get('x', 0)
                y = pose_entry.get('y', 0)
                print(f"  [{i+1}] {timestamp}: ({x:.2f}, {y:.2f})")

        # 7. Environment Info
        print("\n" + "-" * 80)
        print("7. 🌤️ 环境信息 (Environment)")
        print("-" * 80)

        weather = self.world_model.weather
        print(f"天气: {weather.get('condition', 'unknown')}")
        print(f"风速: {weather.get('wind_speed', 0):.1f} m/s")
        print(f"风向: {weather.get('wind_direction', 0):.1f}°")
        print(f"能见度: {weather.get('visibility', 'unknown')}")
        print(f"温度: {weather.get('temperature', 0):.1f}°C")

        # 8. Change History
        print("\n" + "-" * 80)
        print("8. 📝 变化历史 (Change History)")
        print("-" * 80)

        change_history = self.world_model.change_history
        print(f"变化记录数: {len(change_history)}")

        if change_history:
            print(f"\n最近变化:")
            for change in change_history[-5:]:
                print(f"  - {change}")

        # 9. Metadata
        print("\n" + "-" * 80)
        print("9. ⚙️ 元数据 (Metadata)")
        print("-" * 80)

        print(f"对象计数器: {self.world_model._object_counter}")
        print(f"前沿计数器: {self.world_model._frontier_counter}")
        print(f"已探索位置数: {len(self.world_model.explored_positions)}")

        if hasattr(self.world_model, 'resource_manager'):
            rm = self.world_model.resource_manager
            if rm:
                print(f"资源管理器: {rm}")

        print("\n" + "=" * 80)

    def save_to_json(self, filename: str):
        """Save WorldModel state to JSON"""
        data = {
            "metadata": {
                "capture_time": datetime.now().isoformat(),
                "update_count": self.update_count,
                "duration": time.time() - self.start_time
            },
            "robot_state": {
                "position": self.world_model.robot_position,
                "velocity": self.world_model.robot_velocity,
                "heading": self.world_model.robot_heading,
                "battery": self.world_model.battery_level,
                "signal": self.world_model.signal_strength
            },
            "occupancy_grid": {
                "shape": self.world_model.current_map.shape if self.world_model.current_map is not None else None,
                "resolution": self.world_model.map_resolution,
                "origin": self.world_model.map_origin,
                "cell_stats": {
                    "total": self.world_model.current_map.size if self.world_model.current_map is not None else 0,
                    "unknown": int(np.sum(self.world_model.current_map == -1)) if self.world_model.current_map is not None else 0,
                    "free": int(np.sum(self.world_model.current_map == 0)) if self.world_model.current_map is not None else 0,
                    "occupied": int(np.sum(self.world_model.current_map == 100)) if self.world_model.current_map is not None else 0,
                }
            },
            "semantic_objects": {
                "count": len(self.world_model.semantic_objects),
                "objects": [
                    {
                        "id": obj_id,
                        "label": obj.label,
                        "position": obj.world_position if hasattr(obj, 'world_position') else None,
                        "state": obj.state,
                        "confidence": obj.confidence,
                        "description": obj.description,
                        "first_seen": obj.first_seen.isoformat() if hasattr(obj, 'first_seen') else None,
                        "last_seen": obj.last_seen.isoformat() if hasattr(obj, 'last_seen') else None,
                        "observation_count": obj.observation_count,
                    }
                    for obj_id, obj in list(self.world_model.semantic_objects.items())[:20]
                ]
            },
            "exploration": {
                "frontiers_count": len(self.world_model.exploration_frontiers),
                "max_frontiers": self.world_model.max_frontiers,
                "explored_count": len(self.world_model.explored_positions)
            },
            "history": {
                "pose_history_count": len(self.world_model.pose_history),
                "change_history_count": len(self.world_model.change_history)
            },
            "environment": self.world_model.weather
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.get_logger().info(f"💾 Saved to {filename}")


def main(args=None):
    """Main function"""
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init(args=args)

    display = WorldModelDisplay(duration_seconds=15.0)

    print("\n" + "=" * 80)
    print("🎯 WorldModel 内部状态测试")
    print("=" * 80)
    print("\n环境配置:")
    print("  • ROS_DOMAIN_ID: 42")
    print("  • rosbag: /home/yangyuhui/sim_data_bag")
    print("\n正在收集数据并显示 WorldModel 状态...")
    print(f"显示间隔: {display.display_interval}秒")

    try:
        rclpy.spin(display)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    finally:
        # Save final state
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"/media/yangyuhui/CODES1/Brain/tests/perception/e2e/worldmodel_state_{timestamp}.json"
        display.save_to_json(json_file)

        print("\n" + "=" * 80)
        print("📊 测试完成")
        print("=" * 80)
        print(f"  • 总更新次数: {display.update_count}")
        print(f"  • 输出文件: {json_file}")
        print("=" * 80)

        display.destroy_node()


if __name__ == '__main__':
    main()
