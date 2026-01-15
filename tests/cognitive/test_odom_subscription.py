#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的里程计订阅测试
"""
import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class OdomTest(Node):
    def __init__(self):
        super().__init__('odom_test')
        self.count = 0

        self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info("订阅者已创建，等待里程计数据...")

    def odom_callback(self, msg):
        self.count += 1
        if self.count % 10 == 0:
            pos = msg.pose.pose.position
            self.get_logger().info(f"收到 #{self.count}: 位置 ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

def main():
    import os
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init()

    test = OdomTest()

    print("运行10秒...")
    import time
    start = time.time()

    try:
        while time.time() - start < 10:
            rclpy.spin_once(test, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n总共收到 {test.count} 条里程计消息")
        test.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
