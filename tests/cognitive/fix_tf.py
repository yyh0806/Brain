#!/usr/bin/env python3
"""
发布静态TF变换：map -> odom
"""
import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

class StaticTFPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_publisher')

        # 创建静态TF广播器
        self.tf_broadcaster = StaticTransformBroadcaster(self)

        # 创建map -> odom的变换（单位变换）
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"

        # 单位变换（无旋转，无平移）
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        # 发送静态TF
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info("✅ 已发布静态TF: map -> odom")
        self.get_logger().info("   (单位变换，无旋转无平移)")

def main():
    rclpy.init()

    tf_publisher = StaticTFPublisher()

    print("\n" + "="*60)
    print("🔄 静态TF发布器")
    print("="*60)
    print("\n已发布TF变换:")
    print("  map -> odom (单位变换)")
    print("\n现在RViz应该不会再报frame[map]错误了")
    print("="*60 + "\n")

    try:
        rclpy.spin(tf_publisher)
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号")
    finally:
        tf_publisher.destroy_node()
        rclpy.shutdown()
        print("\n✅ TF发布器已关闭")

if __name__ == '__main__':
    main()
