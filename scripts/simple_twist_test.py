#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的Twist测试 - 直接发布并查看里程计反馈
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time


class SimpleTwistTest(Node):
    def __init__(self):
        super().__init__('simple_twist_test')

        self.twist_pub = self.create_publisher(Twist, '/car3/twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/car3/local_odom',
            self.odom_callback,
            10
        )

        self.last_odom = None
        self.odom_count = 0

        print("等待里程计数据...")
        time.sleep(2)

    def odom_callback(self, msg):
        self.odom_count += 1
        self.last_odom = msg

        if self.odom_count % 10 == 0:  # 每10次打印一次
            print(f"\r里程计更新 #{self.odom_count}: "
                  f"pos=({msg.pose.pose.position.x:.3f}, {msg.pose.pose.position.y:.3f}), "
                  f"vel=({msg.twist.twist.linear.x:.3f}, {msg.twist.twist.linear.y:.3f}, "
                  f"{msg.twist.twist.angular.z:.3f})",
                  end='', flush=True)

    def test_command(self, linear_x, linear_y, angular_z, duration):
        print(f"\n\n{'='*60}")
        print(f"发送命令: linear.x={linear_x}, linear.y={linear_y}, angular.z={angular_z}")
        print(f"持续时间: {duration}秒")
        print('='*60)

        if self.last_odom:
            print(f"初始位置: ({self.last_odom.pose.pose.position.x:.3f}, "
                  f"{self.last_odom.pose.pose.position.y:.3f})")

        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z

        start_time = time.time()
        start_x = self.last_odom.pose.pose.position.x if self.last_odom else 0
        start_y = self.last_odom.pose.pose.position.y if self.last_odom else 0

        print("发布中...")
        while time.time() - start_time < duration:
            self.twist_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        # 停止
        stop_msg = Twist()
        self.twist_pub.publish(stop_msg)

        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.1)

        if self.last_odom:
            end_x = self.last_odom.pose.pose.position.x
            end_y = self.last_odom.pose.pose.position.y
            dx = end_x - start_x
            dy = end_y - start_y
            print(f"\n位置变化: Δx={dx:.4f}m, Δy={dy:.4f}m")
            print(f"最终速度: vx={self.last_odom.twist.twist.linear.x:.4f}, "
                  f"vy={self.last_odom.twist.twist.linear.y:.4f}, "
                  f"ω={self.last_odom.twist.twist.angular.z:.4f}")

        time.sleep(1)


def main():
    rclpy.init()

    try:
        tester = SimpleTwistTest()

        print("\n" + "="*60)
        print("开始测试...")
        print("="*60)

        # 测试1: 前进
        tester.test_command(0.3, 0.0, 0.0, 3.0)

        # 测试2: 后退
        tester.test_command(-0.3, 0.0, 0.0, 3.0)

        # 测试3: 原地左转
        tester.test_command(0.0, 0.0, 0.5, 3.0)

        # 测试4: 前进+左转
        tester.test_command(0.2, 0.0, 0.5, 3.0)

        # 测试5: 尝试Y轴
        tester.test_command(0.0, 0.3, 0.0, 3.0)

        print("\n" + "="*60)
        print("测试完成")
        print("="*60)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
