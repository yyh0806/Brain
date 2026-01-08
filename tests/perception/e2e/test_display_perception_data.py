#!/usr/bin/env python3
"""
E2E Test: Display Complete PerceptionData Structure and Content

This test captures and displays the complete PerceptionData structure with all fields,
including real VLM scene understanding using ollama llava:7b.

Usage:
    export ROS_DOMAIN_ID=42
    ros2 bag play /home/yangyuhui/sim_data_bag --loop
    ollama run llava:7b  # In another terminal
    python test_display_perception_data.py

Requirements:
    - ollama run llava:7b (VLM model running)
    - rosbag playback with ROS_DOMAIN_ID=42
"""

import sys
import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

# Add Brain to path
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu as SensorImu
from nav_msgs.msg import Odometry

# Perception layer imports
from brain.perception.sensors.ros2_sensor_manager import PerceptionData
from brain.perception.understanding.vlm_perception import VLMPerception
from brain.perception.data.models import Pose3D, Velocity
from brain.perception.utils.math_utils import compute_laser_angles


def ros_image_to_numpy(img_msg: SensorImage) -> np.ndarray:
    """
    Convert ROS2 Image message to numpy array.
    Simple replacement for cv_bridge using only numpy.

    Supports: rgb8, bgr8, mono8
    """
    # Determine dtype from encoding
    if img_msg.encoding in ['rgb8', 'bgr8', 'mono8']:
        dtype = np.uint8
    elif img_msg.encoding == 'mono16':
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

    # Convert from buffer to numpy array
    arr = np.frombuffer(img_msg.data, dtype=dtype)

    # Determine number of channels
    if img_msg.encoding in ['rgb8', 'bgr8']:
        n_channels = 3
    elif img_msg.encoding in ['mono8', 'mono16']:
        n_channels = 1
    else:
        # Try to infer from image dimensions
        n_channels = arr.size // (img_msg.height * img_msg.width)

    # Reshape to image dimensions
    if n_channels == 1:
        arr = arr.reshape((img_msg.height, img_msg.width))
    else:
        arr = arr.reshape((img_msg.height, img_msg.width, n_channels))

    # Convert BGR to RGB if needed
    if img_msg.encoding == 'bgr8':
        arr = arr[:, :, ::-1].copy()  # BGR -> RGB

    return arr


class PerceptionDataDisplay(Node):
    """Display complete PerceptionData with VLM integration"""

    def __init__(self, num_frames: int = 3):
        super().__init__('perception_data_display')

        self.num_frames = num_frames
        self.frames_captured = 0
        self.start_time = time.time()

        # VLM service
        try:
            self.vlm = VLMPerception()
            self.get_logger().info("✅ VLM initialized (ollama llava:7b)")
        except Exception as e:
            self.get_logger().error(f"❌ VLM initialization failed: {e}")
            self.vlm = None

        # Sensor data buffers
        self.current_odometry = None
        self.current_rgb_image_msg = None  # Store ROS2 message
        self.current_rgb_image_np = None  # Store numpy array for VLM
        self.current_pointcloud = None
        self.current_imu = None

        # ROS2 subscribers
        self._setup_subscribers()

        # Storage for captured PerceptionData
        self.captured_data: List[Dict[str, Any]] = []

        self.get_logger().info("✅ PerceptionData display initialized")
        self.get_logger().info(f"📝 Will capture {self.num_frames} frames")

    def _setup_subscribers(self):
        """Setup ROS2 topic subscribers"""
        # Odometry subscriber
        self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )

        # RGB image subscriber
        self.create_subscription(
            SensorImage,
            '/front_stereo_camera/left/image_raw',
            self.rgb_callback,
            10
        )

        # Pointcloud subscriber
        self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            self.pointcloud_callback,
            10
        )

        # IMU subscriber
        self.create_subscription(
            SensorImu,
            '/chassis/imu',
            self.imu_callback,
            10
        )

        self.get_logger().info("📡 Subscribers created:")
        self.get_logger().info("   - /chassis/odom (Odometry)")
        self.get_logger().info("   - /front_stereo_camera/left/image_raw (RGB Image)")
        self.get_logger().info("   - /front_3d_lidar/lidar_points (Pointcloud)")
        self.get_logger().info("   - /chassis/imu (IMU)")

    def odom_callback(self, msg: Odometry):
        """Odometry callback"""
        self.current_odometry = msg
        self._try_capture_frame()

    def rgb_callback(self, msg: SensorImage):
        """RGB image callback"""
        try:
            self.current_rgb_image_msg = msg

            # Convert ROS2 Image to numpy array using our custom function
            self.current_rgb_image_np = ros_image_to_numpy(msg)

            # Ensure we have RGB format (3 channels)
            if len(self.current_rgb_image_np.shape) == 2:
                # Grayscale to RGB
                self.current_rgb_image_np = np.stack([self.current_rgb_image_np] * 3, axis=-1)

            self._try_capture_frame()
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def pointcloud_callback(self, msg: PointCloud2):
        """Pointcloud callback"""
        self.current_pointcloud = msg
        self._try_capture_frame()

    def imu_callback(self, msg: SensorImu):
        """IMU callback"""
        self.current_imu = msg
        self._try_capture_frame()

    def _try_capture_frame(self):
        """Try to capture a complete frame"""
        if self.frames_captured >= self.num_frames:
            return

        # Wait for at least odometry and RGB image
        if self.current_odometry is None or self.current_rgb_image_np is None:
            return

        self.frames_captured += 1
        self.get_logger().info(f"📸 Capturing frame {self.frames_captured}/{self.num_frames}...")

        # Create PerceptionData object
        perception_data = self._create_perception_data()

        # Display PerceptionData structure and content
        self._display_perception_data(perception_data, self.frames_captured)

        # Store for later
        self.captured_data.append(self._perception_to_dict(perception_data))

        # Clear buffers for next frame
        self.current_odometry = None
        self.current_rgb_image_msg = None
        self.current_rgb_image_np = None
        self.current_pointcloud = None
        self.current_imu = None

        # Wait a bit between frames
        if self.frames_captured < self.num_frames:
            time.sleep(2.0)

    def _create_perception_data(self) -> PerceptionData:
        """Create a PerceptionData object from current sensor data"""
        # Extract pose from odometry
        pose = None
        if self.current_odometry:
            pos = self.current_odometry.pose.pose.position
            ori = self.current_odometry.pose.pose.orientation

            # Convert quaternion to Euler angles
            from brain.perception.utils.coordinates import quaternion_to_euler
            roll, pitch, yaw = quaternion_to_euler((ori.x, ori.y, ori.z, ori.w))

            pose = Pose3D(
                x=pos.x,
                y=pos.y,
                z=pos.z,
                roll=roll,
                pitch=pitch,
                yaw=yaw
            )

        # Extract velocity from odometry
        velocity = None
        if self.current_odometry:
            twist = self.current_odometry.twist.twist
            velocity = Velocity(
                linear_x=twist.linear.x,
                linear_y=twist.linear.y,
                linear_z=twist.linear.z,
                angular_x=twist.angular.x,
                angular_y=twist.angular.y,
                angular_z=twist.angular.z
            )

        # RGB image (already converted to numpy array by ros2_numpy)
        rgb_image = self.current_rgb_image_np

        # Pointcloud (extract basic info)
        pointcloud = None
        laser_ranges = None
        laser_angles = None
        if self.current_pointcloud:
            # For simplicity, just store metadata
            num_points = self.current_pointcloud.width * self.current_pointcloud.height
            if self.current_pointcloud.height == 0:
                num_points = self.current_pointcloud.width
            # Could parse full pointcloud here, but for display purposes metadata is enough

        # IMU
        imu_data = None
        if self.current_imu:
            imu_data = {
                'linear_acceleration': {
                    'x': self.current_imu.linear_acceleration.x,
                    'y': self.current_imu.linear_acceleration.y,
                    'z': self.current_imu.linear_acceleration.z
                },
                'angular_velocity': {
                    'x': self.current_imu.angular_velocity.x,
                    'y': self.current_imu.angular_velocity.y,
                    'z': self.current_imu.angular_velocity.z
                }
            }

        # Create PerceptionData
        perception = PerceptionData(timestamp=datetime.now())

        # Set basic fields
        perception.pose = pose
        perception.velocity = velocity
        perception.rgb_image = rgb_image

        # Process VLM if available
        if self.vlm and rgb_image is not None:
            print("\n🤖 Triggering VLM scene analysis...")
            vlm_start = time.time()

            try:
                # Call VLM for scene understanding
                scene_result = self.vlm.understand_scene(rgb_image)

                vlm_time = time.time() - vlm_start
                print(f"   ✅ VLM analysis completed in {vlm_time:.2f} seconds")

                # Store VLM results
                if hasattr(scene_result, 'summary'):
                    perception.scene_description = scene_result

                if hasattr(scene_result, 'objects') and scene_result.objects:
                    perception.semantic_objects = scene_result.objects

                if hasattr(scene_result, 'spatial_relations'):
                    perception.spatial_relations = scene_result.spatial_relations

                if hasattr(scene_result, 'navigation_hints'):
                    perception.navigation_hints = scene_result.navigation_hints

            except Exception as e:
                print(f"   ⚠️  VLM analysis failed: {e}")

        # Sensor status
        perception.sensor_status = {
            'odometry': self.current_odometry is not None,
            'rgb_camera': self.current_rgb_image_np is not None,
            'pointcloud': self.current_pointcloud is not None,
            'imu': self.current_imu is not None,
            'vlm': self.vlm is not None
        }

        return perception

    def _display_perception_data(self, data: PerceptionData, frame_id: int):
        """Display complete PerceptionData structure and content"""
        print("\n" + "=" * 70)
        print(f"PerceptionData Frame #{frame_id}")
        print("=" * 70)
        print(f"Timestamp: {data.timestamp}")

        # Display structure first
        print("\n📋 PerceptionData字段列表:")
        fields = [
            ("timestamp", "datetime - 数据时间戳"),
            ("pose", "Pose3D - 3D位置和姿态"),
            ("velocity", "Velocity - 6自由度速度"),
            ("rgb_image", "np.ndarray - RGB图像 (H×W×3)"),
            ("rgb_image_right", "np.ndarray - 右RGB图像"),
            ("depth_image", "np.ndarray - 深度图像"),
            ("laser_ranges", "List[float] - 激光雷达距离测量"),
            ("laser_angles", "List[float] - 激光雷达角度"),
            ("pointcloud", "np.ndarray - 3D点云"),
            ("obstacles", "List[Dict] - 障碍物列表"),
            ("occupancy_grid", "np.ndarray - 占据栅格"),
            ("sensor_status", "Dict[str, bool] - 传感器状态"),
            ("semantic_objects", "List[DetectedObject] - VLM识别的物体"),
            ("scene_description", "SceneDescription - VLM场景描述"),
            ("spatial_relations", "List[Dict] - 空间关系"),
            ("navigation_hints", "List[str] - 导航提示"),
        ]
        for i, (field_name, field_desc) in enumerate(fields, 1):
            print(f"{i:2d}. {field_name:20s} - {field_desc}")

        print("\n" + "-" * 70)
        print("实际数据内容:")
        print("-" * 70)

        # Pose
        if data.pose:
            print(f"\n📍 位姿信息 (pose):")
            print(f"   位置: x={data.pose.x:.3f}, y={data.pose.y:.3f}, z={data.pose.z:.3f}")
            if hasattr(data.pose, 'roll'):
                print(f"   姿态: roll={data.pose.roll:.2f}, pitch={data.pose.pitch:.2f}, yaw={data.pose.yaw:.2f}")

        # Velocity
        if data.velocity:
            print(f"\n🚀 速度信息 (velocity):")
            print(f"   线速度: x={data.velocity.linear_x:.3f}, y={data.velocity.linear_y:.3f}, z={data.velocity.linear_z:.3f} m/s")
            print(f"   角速度: x={data.velocity.angular_x:.3f}, y={data.velocity.angular_y:.3f}, z={data.velocity.angular_z:.3f} rad/s")

        # RGB Image
        if data.rgb_image is not None:
            print(f"\n📷 RGB图像 (rgb_image):")
            print(f"   形状: {data.rgb_image.shape}")
            print(f"   大小: {data.rgb_image.nbytes / (1024*1024):.2f} MB")
            print(f"   数据类型: {data.rgb_image.dtype}")
        else:
            print(f"\n📷 RGB图像: ❌ None")

        # Laser/Pointcloud
        if data.pointcloud is not None:
            print(f"\n🔬 点云 (pointcloud):")
            print(f"   数据: {type(data.pointcloud)}")
        else:
            print(f"\n🔬 点云: ❌ None")

        if data.laser_ranges is not None:
            print(f"\n🔬 激光雷达 (laser_ranges):")
            print(f"   测量点数: {len(data.laser_ranges)}")
            print(f"   前方距离: {data.get_front_distance():.2f} m")
            print(f"   左侧距离: {data.get_left_distance():.2f} m")
            print(f"   右侧距离: {data.get_right_distance():.2f} m")
            print(f"   路径畅通: {data.is_path_clear('front', 1.0)}")

        # Obstacles
        if data.obstacles:
            print(f"\n⚠️  障碍物 (obstacles): {len(data.obstacles)} detected")
            for i, obs in enumerate(data.obstacles[:5]):
                print(f"   [{i}] {obs}")
        else:
            print(f"\n⚠️  障碍物: 无")

        # VLM Scene Understanding
        if data.scene_description:
            print(f"\n🤖 VLM场景理解 (scene_description):")
            if hasattr(data.scene_description, 'summary'):
                print(f"   场景描述: {data.scene_description.summary}")

        # Semantic Objects
        if data.semantic_objects:
            print(f"\n🤖 识别的物体 (semantic_objects): {len(data.semantic_objects)} 个")
            for i, obj in enumerate(data.semantic_objects[:5]):
                print(f"   [{i+1}] {obj.label} - 置信度: {obj.confidence:.2f}")
                if hasattr(obj, 'description') and obj.description:
                    print(f"       描述: {obj.description}")
                if hasattr(obj, 'position_description') and obj.position_description:
                    print(f"       位置: {obj.position_description}")
        else:
            print(f"\n🤖 识别的物体: 无")

        # Spatial Relations
        if data.spatial_relations:
            print(f"\n🤖 空间关系 (spatial_relations):")
            for rel in data.spatial_relations[:3]:
                print(f"   - {rel}")
        else:
            print(f"\n🤖 空间关系: 无")

        # Navigation Hints
        if data.navigation_hints:
            print(f"\n🤖 导航提示 (navigation_hints):")
            for hint in data.navigation_hints[:3]:
                print(f"   - {hint}")
        else:
            print(f"\n🤖 导航提示: 无")

        # Sensor Status
        print(f"\n📡 传感器状态 (sensor_status):")
        for sensor, status in data.sensor_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {sensor}")

        print("\n" + "=" * 70)

    def _perception_to_dict(self, data: PerceptionData) -> Dict[str, Any]:
        """Convert PerceptionData to dictionary for JSON serialization"""
        result = {
            "timestamp": data.timestamp.isoformat(),
        }

        # Pose
        if data.pose:
            result["pose"] = {
                "x": data.pose.x,
                "y": data.pose.y,
                "z": data.pose.z,
                "roll": data.pose.roll if hasattr(data.pose, 'roll') else 0.0,
                "pitch": data.pose.pitch if hasattr(data.pose, 'pitch') else 0.0,
                "yaw": data.pose.yaw if hasattr(data.pose, 'yaw') else 0.0,
            }

        # Velocity
        if data.velocity:
            result["velocity"] = {
                "linear_x": data.velocity.linear_x,
                "linear_y": data.velocity.linear_y,
                "linear_z": data.velocity.linear_z,
                "angular_x": data.velocity.angular_x,
                "angular_y": data.velocity.angular_y,
                "angular_z": data.velocity.angular_z,
            }

        # RGB Image
        if data.rgb_image is not None:
            result["rgb_image"] = {
                "shape": list(data.rgb_image.shape),
                "dtype": str(data.rgb_image.dtype),
                "size_mb": data.rgb_image.nbytes / (1024*1024),
            }

        # Semantic Objects
        if data.semantic_objects:
            result["semantic_objects"] = [
                {
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "description": obj.description if hasattr(obj, 'description') else "",
                    "position": obj.position_description if hasattr(obj, 'position_description') else "",
                }
                for obj in data.semantic_objects
            ]

        # Scene Description
        if data.scene_description and hasattr(data.scene_description, 'summary'):
            result["scene_description"] = {
                "summary": data.scene_description.summary
            }

        # Spatial Relations
        if data.spatial_relations:
            result["spatial_relations"] = data.spatial_relations

        # Navigation Hints
        if data.navigation_hints:
            result["navigation_hints"] = data.navigation_hints

        # Sensor Status
        result["sensor_status"] = data.sensor_status

        return result

    def save_to_json(self, filename: str):
        """Save captured data to JSON file"""
        data = {
            "metadata": {
                "capture_time": datetime.now().isoformat(),
                "num_frames": len(self.captured_data),
                "vlm_enabled": self.vlm is not None,
            },
            "frames": self.captured_data
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self.get_logger().info(f"💾 Saved to {filename}")

    def save_to_markdown(self, filename: str):
        """Save captured data to Markdown file"""
        lines = []

        # Header
        lines.append("# PerceptionData Display Report\n")
        lines.append(f"**Capture Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Total Frames**: {len(self.captured_data)}\n")
        lines.append(f"**VLM Enabled**: {self.vlm is not None}\n")
        lines.append("\n---\n")

        # Frames
        for i, frame_data in enumerate(self.captured_data):
            lines.append(f"## Frame {i+1}\n")
            lines.append(f"**Timestamp**: {frame_data['timestamp']}\n\n")

            # Add sections for each field
            if 'pose' in frame_data:
                pose = frame_data['pose']
                lines.append("### 📍 位姿信息\n")
                lines.append(f"- 位置: x={pose['x']:.3f}, y={pose['y']:.3f}, z={pose['z']:.3f}\n")
                lines.append(f"- 姿态: roll={pose['roll']:.2f}, pitch={pose['pitch']:.2f}, yaw={pose['yaw']:.2f}\n\n")

            if 'velocity' in frame_data:
                vel = frame_data['velocity']
                lines.append("### 🚀 速度信息\n")
                lines.append(f"- 线速度: x={vel['linear_x']:.3f}, y={vel['linear_y']:.3f}, z={vel['linear_z']:.3f} m/s\n")
                lines.append(f"- 角速度: z={vel['angular_z']:.3f} rad/s\n\n")

            if 'rgb_image' in frame_data:
                img = frame_data['rgb_image']
                lines.append("### 📷 RGB图像\n")
                lines.append(f"- 形状: {img['shape']}\n")
                lines.append(f"- 大小: {img['size_mb']:.2f} MB\n\n")

            if 'scene_description' in frame_data:
                lines.append("### 🤖 VLM场景描述\n")
                lines.append(f"{frame_data['scene_description']['summary']}\n\n")

            if 'semantic_objects' in frame_data:
                lines.append("### 🤖 识别的物体\n")
                for obj in frame_data['semantic_objects']:
                    lines.append(f"- **{obj['label']}** (置信度: {obj['confidence']:.2f})\n")
                    if obj['description']:
                        lines.append(f"  - {obj['description']}\n")
                lines.append("\n")

            if 'navigation_hints' in frame_data:
                lines.append("### 🤖 导航提示\n")
                for hint in frame_data['navigation_hints']:
                    lines.append(f"- {hint}\n")
                lines.append("\n")

            lines.append("---\n")

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        self.get_logger().info(f"💾 Saved to {filename}")


def main(args=None):
    """Main function"""
    # Set ROS_DOMAIN_ID
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init(args=args)

    # Create display node
    display = PerceptionDataDisplay(num_frames=3)

    print("\n" + "=" * 70)
    print("🎯 PerceptionData完整展示测试 (3帧)")
    print("=" * 70)
    print("\n环境配置:")
    print("  • ROS_DOMAIN_ID: 42")
    print("  • rosbag: /home/yangyuhui/sim_data_bag")
    print("  • VLM: ollama llava:7b")
    print("\n正在捕获数据...\n")

    try:
        # Spin until all frames captured
        while display.frames_captured < display.num_frames:
            rclpy.spin_once(display, timeout_sec=0.1)

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    finally:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"/media/yangyuhui/CODES1/Brain/tests/perception/e2e/perception_data_display_{timestamp}.json"
        md_file = f"/media/yangyuhui/CODES1/Brain/tests/perception/e2e/perception_data_display_{timestamp}.md"

        display.save_to_json(json_file)
        display.save_to_markdown(md_file)

        # Print summary
        print("\n" + "=" * 70)
        print("📊 测试完成")
        print("=" * 70)
        print(f"  • 捕获帧数: {display.frames_captured}")
        print(f"  • VLM已启用: {display.vlm is not None}")
        print(f"\n📁 输出文件:")
        print(f"  • {json_file}")
        print(f"  • {md_file}")
        print("=" * 70)

        # Cleanup
        display.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
