#!/usr/bin/env python3
"""
简单的静态odom发布器
用于没有odom的仿真环境
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math

class StaticOdomPublisher(Node):
    def __init__(self):
        super().__init__('static_odom_publisher')

        self.odom_pub = self.create_publisher(Odometry, '/chassis/odom', 10)

        # 创建定时器，10Hz发布
        self.timer = self.create_timer(0.1, self.publish_odom)

        # 静态位姿（可以根据需要修改）
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

        self.get_logger().info('静态odom发布器已启动')
        self.get_logger().info(f'位姿: x={self.x}, y={self.y}, yaw={self.yaw}')

    def publish_odom(self):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # 位置
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z

        # 姿态（四元数）
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(self.yaw / 2.0)
        q.w = math.cos(self.yaw / 2.0)
        odom.pose.pose.orientation = q

        # 速度（都是0）
        odom.twist.twist.linear.x = 0.0
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = StaticOdomPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
