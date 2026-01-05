#!/usr/bin/env python3
"""
ROS2 Topics诊断工具

检查ROS2 topics的状态和数据可用性
"""

import sys
import os

# 设置环境
os.environ['ROS_DOMAIN_ID'] = '0'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time


class TopicDiagnostic(Node):
    """Topic诊断节点"""

    def __init__(self):
        super().__init__('topic_diagnostic')

        self.rgb_count = 0
        self.depth_count = 0
        self.last_rgb_time = None
        self.last_depth_time = None

        # 订阅topics
        self.rgb_sub = self.create_subscription(
            Image, '/rgb_test', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/depth_test', self.depth_callback, 10
        )

        self.get_logger().info("✅ 诊断节点已启动")
        self.get_logger().info("监听topics: /rgb_test, /depth_test")
        print("\n正在监听topics，等待10秒...\n")

    def rgb_callback(self, msg):
        """RGB回调"""
        self.rgb_count += 1
        self.last_rgb_time = time.time()
        if self.rgb_count == 1:
            print(f"✅ 收到RGB数据!")
            print(f"   尺寸: {msg.width}x{msg.height}")
            print(f"   编码: {msg.encoding}")
            print(f"   数据大小: {len(msg.data)} bytes")

    def depth_callback(self, msg):
        """深度回调"""
        self.depth_count += 1
        self.last_depth_time = time.time()
        if self.depth_count == 1:
            print(f"\n✅ 收到深度图数据!")
            print(f"   尺寸: {msg.width}x{msg.height}")
            print(f"   编码: {msg.encoding}")
            print(f"   数据大小: {len(msg.data)} bytes")

    def print_status(self):
        """打印状态"""
        print(f"\n{'='*60}")
        print(f"📊 Topic状态报告")
        print(f"{'='*60}")
        print(f"\nRGB (/rgb_test):")
        print(f"  接收帧数: {self.rgb_count}")
        if self.last_rgb_time:
            print(f"  最后接收: {time.time() - self.last_rgb_time:.1f}秒前")

        print(f"\n深度 (/depth_test):")
        print(f"  接收帧数: {self.depth_count}")
        if self.last_depth_time:
            print(f"  最后接收: {time.time() - self.last_depth_time:.1f}秒前")

        if self.rgb_count > 0 and self.depth_count > 0:
            print(f"\n✅ Topics正常发布数据")
            print(f"   可以进行感知测试")
        else:
            print(f"\n⚠️  Topics未发布数据")
            print(f"   请检查:")
            print(f"   1. rosbag是否在播放")
            print(f"   2. ROS_DOMAIN_ID是否正确")
            print(f"   3. topic名称是否匹配")

        print(f"\n{'='*60}\n")


def main():
    rclpy.init()

    diagnostic = TopicDiagnostic()

    try:
        # 运行10秒
        start_time = time.time()
        while time.time() - start_time < 10:
            rclpy.spin_once(diagnostic, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n诊断被中断")
    finally:
        diagnostic.print_status()
        diagnostic.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
