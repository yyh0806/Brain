#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车移动控制脚本
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class CarController(Node):
    """小车控制器"""

    def __init__(self):
        super().__init__('car_controller')
        self.publisher = self.create_publisher(Twist, '/car3/twist', 10)

    def move(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        """
        控制小车移动

        Args:
            linear_x: 线速度 (前进为正)
            angular_z: 角速度 (左转为正)
            duration: 持续时间（秒）
        """
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z

        start_time = time.time()
        rate = self.create_rate(10)  # 10Hz

        while time.time() - start_time < duration:
            self.publisher.publish(msg)
            rate.sleep()

        # 停止
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.publisher.publish(stop_msg)

    def run_demo(self):
        """运行演示动作序列"""
        print("=" * 60)
        print("小车移动演示")
        print("=" * 60)
        print()

        print("[1/6] 前进 3秒 (速度: 0.5 m/s)")
        self.move(linear_x=0.5, angular_z=0.0, duration=3.0)

        print("[2/6] 左转 2秒 (角速度: 0.5 rad/s)")
        self.move(linear_x=0.0, angular_z=0.5, duration=2.0)

        print("[3/6] 前进 3秒 (速度: 0.5 m/s)")
        self.move(linear_x=0.5, angular_z=0.0, duration=3.0)

        print("[4/6] 右转 2秒 (角速度: -0.5 rad/s)")
        self.move(linear_x=0.0, angular_z=-0.5, duration=2.0)

        print("[5/6] 前进 2秒 (速度: 0.3 m/s)")
        self.move(linear_x=0.3, angular_z=0.0, duration=2.0)

        print("[6/6] 停止")
        self.move(linear_x=0.0, angular_z=0.0, duration=1.0)

        print()
        print("=" * 60)
        print("演示完成！在RViz中应该能看到小车移动轨迹")
        print("=" * 60)


def main():
    """主函数"""
    rclpy.init()

    controller = CarController()

    try:
        controller.run_demo()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
