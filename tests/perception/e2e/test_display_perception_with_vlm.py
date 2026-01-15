#!/usr/bin/env python3
"""
E2E Test: Display Complete PerceptionData with Real VLM Analysis

This test captures and displays complete PerceptionData with real VLM scene understanding
using ollama llava:7b.

Usage:
    export ROS_DOMAIN_ID=42
    ros2 bag play /home/yangyuhui/sim_data_bag --loop
    ollama run llava:7b  # In another terminal
    python test_display_perception_with_vlm.py
"""

import sys
import os
import time
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

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
from brain.perception.utils.coordinates import quaternion_to_euler


def ros_image_to_numpy(img_msg: SensorImage) -> np.ndarray:
    """Convert ROS2 Image message to numpy array (simple replacement for cv_bridge)"""
    if img_msg.encoding in ['rgb8', 'bgr8', 'mono8']:
        dtype = np.uint8
    elif img_msg.encoding == 'mono16':
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported encoding: {img_msg.encoding}")

    arr = np.frombuffer(img_msg.data, dtype=dtype)

    if img_msg.encoding in ['rgb8', 'bgr8']:
        n_channels = 3
    elif img_msg.encoding in ['mono8', 'mono16']:
        n_channels = 1
    else:
        n_channels = arr.size // (img_msg.height * img_msg.width)

    if n_channels == 1:
        arr = arr.reshape((img_msg.height, img_msg.width))
    else:
        arr = arr.reshape((img_msg.height, img_msg.width, n_channels))

    if img_msg.encoding == 'bgr8':
        arr = arr[:, :, ::-1].copy()

    return arr


class PerceptionDataWithVLM(Node):
    """Capture and display PerceptionData with real VLM analysis"""

    def __init__(self, num_frames: int = 3):
        super().__init__('perception_data_vlm')

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
        self.current_rgb_image_msg = None
        self.current_rgb_image_np = None
        self.current_pointcloud = None
        self.current_imu = None

        # Storage for captured data
        self.captured_data: List[Dict[str, Any]] = []

        # ROS2 subscribers
        self._setup_subscribers()

        self.get_logger().info("✅ PerceptionData with VLM initialized")
        self.get_logger().info(f"📝 Will capture {self.num_frames} frames")

    def _setup_subscribers(self):
        """Setup ROS2 topic subscribers"""
        self.create_subscription(Odometry, '/chassis/odom', self.odom_callback, 10)
        self.create_subscription(SensorImage, '/front_stereo_camera/left/image_raw', self.rgb_callback, 10)
        self.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points', self.pointcloud_callback, 10)
        self.create_subscription(SensorImu, '/chassis/imu', self.imu_callback, 10)

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
            self.current_rgb_image_np = ros_image_to_numpy(msg)

            if len(self.current_rgb_image_np.shape) == 2:
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

        if self.current_odometry is None or self.current_rgb_image_np is None:
            return

        self.frames_captured += 1
        self.get_logger().info(f"📸 Capturing frame {self.frames_captured}/{self.num_frames}...")

        # Create PerceptionData and process VLM
        perception_data = self._create_perception_data()

        # Display PerceptionData structure and content
        self._display_perception_data(perception_data, self.frames_captured)

        # Store for later
        self.captured_data.append(self._perception_to_dict(perception_data))

        # Clear buffers
        self.current_odometry = None
        self.current_rgb_image_msg = None
        self.current_rgb_image_np = None
        self.current_pointcloud = None
        self.current_imu = None

        # Wait between frames
        if self.frames_captured < self.num_frames:
            time.sleep(2.0)

    def _create_perception_data(self) -> PerceptionData:
        """Create PerceptionData object from current sensor data"""
        # Extract pose from odometry
        pose = None
        if self.current_odometry:
            pos = self.current_odometry.pose.pose.position
            ori = self.current_odometry.pose.pose.orientation
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

        # Create PerceptionData
        perception = PerceptionData(timestamp=datetime.now())
        perception.pose = pose
        perception.velocity = velocity
        perception.rgb_image = self.current_rgb_image_np

        # Process VLM using direct ollama call (not async)
        if self.vlm and self.current_rgb_image_np is not None:
            print("\n🤖 Triggering VLM scene analysis...")
            vlm_start = time.time()

            try:
                # Use direct ollama API call for scene understanding
                import ollama

                # Resize image for VLM (optional, to speed up processing)
                img = self.current_rgb_image_np
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(img)

                # Save to temp file and send to ollama
                import io
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG')
                img_bytes = buf.getvalue()

                # Call ollama llava:7b
                response = ollama.generate(
                    model='llava:7b',
                    prompt="""请详细描述这张图像中的场景。
请用中文回答，包括：
1. 场景概述（室内/室外，环境类型）
2. 看到的物体（至少3个）
3. 空间关系（物体的相对位置）
4. 导航建议（是否可以通行，有什么需要注意的）""",
                    images=[img_bytes]
                )

                vlm_time = time.time() - vlm_start
                print(f"   ✅ VLM analysis completed in {vlm_time:.2f} seconds")
                print(f"   📝 VLM Response: {response['response'][:200]}...")

                # Store VLM result
                perception.scene_description = type('SceneDescription', (), {
                    'summary': response['response']
                })()

            except Exception as e:
                print(f"   ⚠️  VLM analysis failed: {e}")
                import traceback
                traceback.print_exc()

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
        """Display complete PerceptionData"""
        print("\n" + "=" * 70)
        print(f"PerceptionData Frame #{frame_id}")
        print("=" * 70)
        print(f"Timestamp: {data.timestamp}")

        print("\n📋 PerceptionData字段列表:")
        fields = [
            ("1.", "timestamp", "datetime - 数据时间戳"),
            ("2.", "pose", "Pose3D - 3D位置和姿态"),
            ("3.", "velocity", "Velocity - 6自由度速度"),
            ("4.", "rgb_image", "np.ndarray - RGB图像 (H×W×3)"),
            ("5.", "rgb_image_right", "np.ndarray - 右RGB图像"),
            ("6.", "depth_image", "np.ndarray - 深度图像"),
            ("7.", "laser_ranges", "List[float] - 激光雷达距离测量"),
            ("8.", "laser_angles", "List[float] - 激光雷达角度"),
            ("9.", "pointcloud", "np.ndarray - 3D点云"),
            ("10.", "obstacles", "List[Dict] - 障碍物列表"),
            ("11.", "occupancy_grid", "np.ndarray - 占据栅格"),
            ("12.", "sensor_status", "Dict[str, bool] - 传感器状态"),
            ("13.", "semantic_objects", "List[DetectedObject] - VLM识别的物体"),
            ("14.", "scene_description", "SceneDescription - VLM场景描述"),
            ("15.", "spatial_relations", "List[Dict] - 空间关系"),
            ("16.", "navigation_hints", "List[str] - 导航提示"),
        ]
        for num, field_name, field_desc in fields:
            print(f"{num:>3} {field_name:20s} - {field_desc}")

        print("\n" + "-" * 70)
        print("实际数据内容:")
        print("-" * 70)

        # Pose
        if data.pose:
            print(f"\n📍 位姿信息 (pose):")
            print(f"   位置: x={data.pose.x:.3f}, y={data.pose.y:.3f}, z={data.pose.z:.3f}")
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

        # VLM Scene Description
        if data.scene_description:
            print(f"\n🤖 VLM场景描述 (scene_description):")
            if hasattr(data.scene_description, 'summary'):
                summary = data.scene_description.summary
                # Truncate if too long
                if len(summary) > 500:
                    summary = summary[:500] + "..."
                print(f"   {summary}")

        # Sensor Status
        print(f"\n📡 传感器状态 (sensor_status):")
        for sensor, status in data.sensor_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {sensor}")

        print("\n" + "=" * 70)

    def _perception_to_dict(self, data: PerceptionData) -> Dict[str, Any]:
        """Convert PerceptionData to dictionary"""
        result = {"timestamp": data.timestamp.isoformat()}

        if data.pose:
            result["pose"] = {
                "x": data.pose.x,
                "y": data.pose.y,
                "z": data.pose.z,
                "roll": data.pose.roll,
                "pitch": data.pose.pitch,
                "yaw": data.pose.yaw,
            }

        if data.velocity:
            result["velocity"] = {
                "linear_x": data.velocity.linear_x,
                "linear_y": data.velocity.linear_y,
                "linear_z": data.velocity.linear_z,
                "angular_x": data.velocity.angular_x,
                "angular_y": data.velocity.angular_y,
                "angular_z": data.velocity.angular_z,
            }

        if data.rgb_image is not None:
            result["rgb_image"] = {
                "shape": list(data.rgb_image.shape),
                "dtype": str(data.rgb_image.dtype),
                "size_mb": data.rgb_image.nbytes / (1024*1024),
            }

        if data.scene_description and hasattr(data.scene_description, 'summary'):
            result["scene_description"] = {
                "summary": data.scene_description.summary
            }

        result["sensor_status"] = data.sensor_status

        return result

    def save_to_json(self, filename: str):
        """Save captured data to JSON"""
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
        """Save captured data to Markdown"""
        lines = []

        lines.append("# PerceptionData Display Report with VLM\n")
        lines.append(f"**Capture Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Total Frames**: {len(self.captured_data)}\n")
        lines.append(f"**VLM Enabled**: {self.vlm is not None}\n")
        lines.append("\n---\n")

        for i, frame_data in enumerate(self.captured_data):
            lines.append(f"## Frame {i+1}\n")
            lines.append(f"**Timestamp**: {frame_data['timestamp']}\n\n")

            if 'pose' in frame_data:
                pose = frame_data['pose']
                lines.append("### 📍 位姿信息\n")
                lines.append(f"- 位置: x={pose['x']:.3f}, y={pose['y']:.3f}, z={pose['z']:.3f}\n")
                lines.append(f"- 姿态: roll={pose['roll']:.2f}, pitch={pose['pitch']:.2f}, yaw={pose['yaw']:.2f}\n\n")

            if 'velocity' in frame_data:
                vel = frame_data['velocity']
                lines.append("### 🚀 速度信息\n")
                lines.append(f"- 线速度: x={vel['linear_x']:.3f}, y={vel['linear_y']:.3f} m/s\n")
                lines.append(f"- 角速度: z={vel['angular_z']:.3f} rad/s\n\n")

            if 'rgb_image' in frame_data:
                img = frame_data['rgb_image']
                lines.append("### 📷 RGB图像\n")
                lines.append(f"- 形状: {img['shape']}\n")
                lines.append(f"- 大小: {img['size_mb']:.2f} MB\n\n")

            if 'scene_description' in frame_data:
                lines.append("### 🤖 VLM场景描述\n")
                lines.append(f"{frame_data['scene_description']['summary']}\n\n")

            lines.append("---\n")

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        self.get_logger().info(f"💾 Saved to {filename}")


def main(args=None):
    """Main function"""
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init(args=args)

    # Create display node
    display = PerceptionDataWithVLM(num_frames=3)

    print("\n" + "=" * 70)
    print("🎯 PerceptionData完整展示测试 (含VLM场景理解) - 3帧")
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
        json_file = f"/media/yangyuhui/CODES1/Brain/tests/perception/e2e/perception_data_vlm_{timestamp}.json"
        md_file = f"/media/yangyuhui/CODES1/Brain/tests/perception/e2e/perception_data_vlm_{timestamp}.md"

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
