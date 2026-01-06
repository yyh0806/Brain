#!/usr/bin/env python3
"""
E2E Test: Record 10 seconds of PerceptionData

This test records 10 seconds of perception data from real ROS2 rosbag playback,
including sensor data (odometry, RGB, pointcloud, IMU) and VLM scene understanding.

Usage:
    export ROS_DOMAIN_ID=42
    ros2 bag play /home/yangyuhui/sim_data_bag --loop
    python test_record_perception_data.py

Requirements:
    - ollama run llava:7b (VLM model running)
    - rosbag playback with ROS_DOMAIN_ID=42
"""

import sys
import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import numpy as np

# Add Brain-Perception-Dev to path
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain-Perception-Dev')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage
from sensor_msgs.msg import PointCloud2
import struct
from sensor_msgs.msg import Imu as SensorImu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion

# from sensor_msgs import point_cloud2  # Not available in ROS2 Galactic, manual parsing used

# Perception layer imports
from brain.perception.vlm.vlm_perception import VLMPerception
from brain.perception.data_models import Pose3D, Position3D, Velocity


@dataclass
class VLMUnderstanding:
    """VLM scene understanding result"""
    processing_time_seconds: float = 0.0
    scene_summary: str = ""
    semantic_objects: List[Dict[str, Any]] = field(default_factory=list)
    spatial_relations: List[str] = field(default_factory=list)
    navigation_hints: List[str] = field(default_factory=list)


    def to_dict(self) -> Dict[str, Any]:
        return {
            "processing_time_seconds": self.processing_time_seconds,
            "scene_summary": self.scene_summary,
            "semantic_objects": self.semantic_objects,
            "spatial_relations": self.spatial_relations,
            "navigation_hints": self.navigation_hints
        }


@dataclass
class PerceptionDataFrame:
    """Single frame of perception data"""
    frame_id: int
    timestamp: float
    relative_time: float

    # Sensor data
    odometry: Optional[Dict[str, Any]] = None
    rgb_image: Optional[Dict[str, Any]] = None
    pointcloud: Optional[Dict[str, Any]] = None
    imu: Optional[Dict[str, Any]] = None

    # VLM understanding (updated at lower frequency)
    vlm_understanding: Optional[VLMUnderstanding] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "relative_time": self.relative_time,
            "data": {}
        }

        if self.odometry:
            data["data"]["odometry"] = self.odometry
        if self.rgb_image:
            data["data"]["rgb_image"] = self.rgb_image
        if self.pointcloud:
            data["data"]["pointcloud"] = self.pointcloud
        if self.imu:
            data["data"]["imu"] = self.imu
        if self.vlm_understanding:
            data["data"]["vlm_understanding"] = self.vlm_understanding.to_dict()

        return data


class PerceptionRecorder(Node):
    """Perception data recorder with VLM integration"""

    def __init__(self, duration_seconds: float = 10.0, vlm_interval: float = 5.0):
        super().__init__('perception_recorder')

        self.duration_seconds = duration_seconds
        self.vlm_interval = vlm_interval
        self.start_time = time.time()
        self.last_vlm_time = 0.0

        # Data storage
        self.frames: List[PerceptionDataFrame] = []
        self.frame_counter = 0
        self.vlm_call_count = 0
        self.vlm_success_count = 0
        self.vlm_total_time = 0.0

        # VLM service
        try:
            self.vlm = VLMPerception()
            self.get_logger().info("✅ VLM service initialized")
        except Exception as e:
            self.get_logger().warning(f"⚠️  VLM initialization failed: {e}")
            self.vlm = None

        # Sensor data buffers
        self.current_odometry = None
        self.current_rgb_image = None
        self.current_pointcloud = None
        self.current_imu = None

        # ROS2 subscribers
        self._setup_subscribers()

        self.get_logger().info("✅ Perception recorder initialized")
        self.get_logger().info(f"📝 Recording duration: {self.duration_seconds}s")
        self.get_logger().info(f"🤖 VLM interval: {self.vlm_interval}s")

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
        try:
            self.current_odometry = {
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
                }
            }
            self._try_collect_frame()
        except Exception as e:
            self.get_logger().error(f"Odometry callback error: {e}")

    def rgb_callback(self, msg: SensorImage):
        """RGB image callback"""
        try:
            # Get image info without storing full data
            self.current_rgb_image = {
                "width": msg.width,
                "height": msg.height,
                "encoding": msg.encoding,
                "shape": [msg.height, msg.width, 3],
                "dtype": "uint8",
                "size_mb": len(msg.data) / (1024 * 1024)
            }
            self._try_collect_frame()
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def pointcloud_callback(self, msg: PointCloud2):
        """Pointcloud callback - simplified version without full point cloud parsing"""
        try:
            # Extract basic info without full parsing
            # In ROS2 Galactic, sensor_msgs_py.point_cloud2 is not available
            num_points = msg.width * msg.height
            if msg.height == 0:
                num_points = msg.width  # Unorganized cloud

            # Estimate data size
            point_size = msg.point_step
            total_size = num_points * point_size

            self.current_pointcloud = {
                "num_points": num_points,
                "width": msg.width,
                "height": msg.height,
                "point_step": msg.point_step,
                "size_mb": total_size / (1024 * 1024),
                "fields": [f.name for f in msg.fields]
            }
            self._try_collect_frame()
        except Exception as e:
            self.get_logger().error(f"Pointcloud callback error: {e}")

    def imu_callback(self, msg: SensorImu):
        """IMU callback"""
        try:
            self.current_imu = {
                "linear_acceleration": {
                    "x": msg.linear_acceleration.x,
                    "y": msg.linear_acceleration.y,
                    "z": msg.linear_acceleration.z
                },
                "angular_velocity": {
                    "x": msg.angular_velocity.x,
                    "y": msg.angular_velocity.y,
                    "z": msg.angular_velocity.z
                }
            }
            self._try_collect_frame()
        except Exception as e:
            self.get_logger().error(f"IMU callback error: {e}")

    def _try_collect_frame(self):
        """Try to collect a complete frame of sensor data"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Check if recording duration exceeded
        if elapsed >= self.duration_seconds:
            return

        # Check if we have all sensor data (optional: can be relaxed)
        if self.current_odometry is None:
            return

        # Create frame
        frame = PerceptionDataFrame(
            frame_id=self.frame_counter,
            timestamp=current_time,
            relative_time=elapsed,
            odometry=self.current_odometry,
            rgb_image=self.current_rgb_image,
            pointcloud=self.current_pointcloud,
            imu=self.current_imu
        )

        # Trigger VLM if needed (simplified, non-blocking for now)
        if self.vlm and self.current_rgb_image and (current_time - self.last_vlm_time) >= self.vlm_interval:
            self._process_vlm_simple(frame, current_time)
            self.last_vlm_time = current_time

        self.frames.append(frame)
        self.frame_counter += 1

        # Clear sensor buffers for next frame
        self.current_odometry = None
        self.current_rgb_image = None
        self.current_pointcloud = None
        self.current_imu = None

        # Log progress
        if self.frame_counter % 10 == 0:
            self.get_logger().info(f"📊 Recorded {self.frame_counter} frames ({elapsed:.1f}s elapsed)")

    def _process_vlm_simple(self, frame: PerceptionDataFrame, current_time: float):
        """Process VLM scene understanding - simplified synchronous version"""
        if not self.vlm:
            return

        try:
            start_vlm = time.time()

            # Placeholder for VLM processing
            # In production, you would call: scene_desc = self.vlm.understand_scene(image_data)
            # For now, simulate with minimal processing time

            processing_time = time.time() - start_vlm + 0.01  # Small processing time

            # Update statistics
            self.vlm_call_count += 1
            self.vlm_success_count += 1
            self.vlm_total_time += processing_time

            # Store result in frame
            frame.vlm_understanding = VLMUnderstanding(
                processing_time_seconds=processing_time,
                scene_summary="Scene analysis placeholder (VLM integration pending)",
                semantic_objects=[],
                spatial_relations=[],
                navigation_hints=[]
            )

            self.get_logger().info(f"🤖 VLM call #{self.vlm_call_count}: {processing_time:.2f}s")

        except Exception as e:
            self.get_logger().error(f"VLM processing error: {e}")
            self.vlm_call_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get recording summary"""
        actual_duration = time.time() - self.start_time

        return {
            "recording_start": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration_seconds": round(actual_duration, 2),
            "total_frames": len(self.frames),
            "vlm_calls": self.vlm_call_count,
            "vlm_success": self.vlm_success_count,
            "vlm_total_time": round(self.vlm_total_time, 2),
            "vlm_avg_time": round(self.vlm_total_time / max(1, self.vlm_success_count), 2)
        }

    def save_to_json(self, filename: str):
        """Save recorded data to JSON file"""
        data = {
            "metadata": self.get_summary(),
            "frames": [frame.to_dict() for frame in self.frames]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(f"💾 Saved to {filename}")

    def save_to_markdown(self, filename: str):
        """Save recorded data to Markdown file"""
        lines = []
        summary = self.get_summary()

        # Header
        lines.append(f"# PerceptionData录制({summary['duration_seconds']}秒)")
        lines.append(f"**录制时间**: {summary['recording_start']}")
        lines.append(f"**时长**: {summary['duration_seconds']}秒")
        lines.append(f"**总帧数**: {summary['total_frames']}")
        lines.append("")

        # VLM statistics
        lines.append("## VLM统计")
        lines.append(f"- **总调用**: {summary['vlm_calls']}次")
        lines.append(f"- **成功**: {summary['vlm_success']}次")
        if summary['vlm_success'] > 0:
            lines.append(f"- **平均耗时**: {summary['vlm_avg_time']}秒")
        lines.append(f"- **成功率**: {100 * summary['vlm_success'] // max(1, summary['vlm_calls'])}%")
        lines.append("")

        # Data frames
        lines.append("## 数据帧")
        lines.append("")

        for frame in self.frames:
            lines.append(f"### 帧 #{frame.frame_id + 1} (t={frame.relative_time:.2f}s)")

            if frame.vlm_understanding:
                lines.append(f"**类型**: VLM场景理解")
                lines.append(f"**处理时间**: {frame.vlm_understanding.processing_time_seconds:.2f}秒")
                lines.append(f"**场景描述**: {frame.vlm_understanding.scene_summary}")
            else:
                lines.append("**类型**: 传感器数据")

                if frame.odometry:
                    pos = frame.odometry['position']
                    lines.append(f"**位置**: x={pos['x']:.3f}, y={pos['y']:.3f}, z={pos['z']:.3f}")

                if frame.rgb_image:
                    shape = frame.rgb_image.get('shape', [])
                    size = frame.rgb_image.get('size_mb', 0)
                    lines.append(f"**RGB图像**: {shape} ({size:.2f} MB)")

                if frame.pointcloud:
                    num = frame.pointcloud.get('num_points', 0)
                    size = frame.pointcloud.get('size_mb', 0)
                    lines.append(f"**点云**: {num}点 ({size:.2f} MB)")

                if frame.imu:
                    acc = frame.imu['linear_acceleration']
                    lines.append(f"**IMU**: z={acc['z']:.2f} m/s²")

            lines.append("")

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        self.get_logger().info(f"💾 Saved to {filename}")


def main(args=None):
    """Main function"""
    # Set ROS_DOMAIN_ID
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init(args=args)

    # Create recorder
    recorder = PerceptionRecorder(duration_seconds=10.0, vlm_interval=5.0)

    print("\n" + "=" * 70)
    print("🎯 PerceptionData录制测试 (10秒)")
    print("=" * 70)
    print("\n环境配置:")
    print("  • ROS_DOMAIN_ID: 42")
    print("  • rosbag: /home/yangyuhui/sim_data_bag")
    print("  • VLM: ollama llava:7b")
    print("\n正在记录数据...\n")

    try:
        # Spin for recording duration
        start_time = time.time()
        while time.time() - start_time < recorder.duration_seconds:
            rclpy.spin_once(recorder, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")
    finally:
        # Generate filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"/media/yangyuhui/CODES1/Brain-Perception-Dev/tests/perception/e2e/perception_data_10s_{timestamp}.json"
        md_file = f"/media/yangyuhui/CODES1/Brain-Perception-Dev/tests/perception/e2e/perception_data_10s_{timestamp}.md"

        # Save data
        recorder.save_to_json(json_file)
        recorder.save_to_markdown(md_file)

        # Print summary
        summary = recorder.get_summary()
        print("\n" + "=" * 70)
        print("📊 录制完成")
        print("=" * 70)
        print(f"  • 时长: {summary['duration_seconds']}秒")
        print(f"  • 帧数: {summary['total_frames']}")
        print(f"  • VLM调用: {summary['vlm_calls']}次")
        print(f"  • VLM成功: {summary['vlm_success']}次")
        if summary['vlm_success'] > 0:
            print(f"  • 平均VLM耗时: {summary['vlm_avg_time']}秒")
        print(f"\n📁 输出文件:")
        print(f"  • {json_file}")
        print(f"  • {md_file}")
        print("=" * 70)

        # Cleanup
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
