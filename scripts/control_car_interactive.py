#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车交互式控制脚本 - 全向机器人版本
适用于麦克纳姆轮/全向轮机器人

控制说明：
- W/S: 前进/后退 (linear.x)
- A/D: 左移/右移 (linear.y)
- Q/E: 左转/右转 (angular.z)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import tty
import termios
import time

class CarInteractiveController(Node):
    """小车交互式控制器 - 全向机器人"""

    def __init__(self):
        super().__init__('car_interactive_controller')
        self.publisher = self.create_publisher(Twist, '/car3/twist', 10)

        # 控制参数
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.5  # rad/s

        print("=" * 60)
        print("小车交互式控制（全向机器人）")
        print("=" * 60)
        print()
        print("控制键:")
        print("  W/↑ - 前进")
        print("  S/↓ - 后退")
        print("  A/← - 左移 (y轴)")
        print("  D/→ - 右移 (y轴)")
        print("  Q   - 左转 (原地)")
        print("  E   - 右转 (原地)")
        print("  空格 - 停止")
        print("  +/- - 调整速度")
        print("  ESC - 退出")
        print()
        print(f"当前速度: 线速度={self.linear_speed:.1f} m/s, 角速度={self.angular_speed:.1f} rad/s")
        print()
        print("开始控制...")
        print()

    def get_key(self):
        """获取按键"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def publish_twist(self, linear_x, linear_y, angular_z):
        """发布Twist消息"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y  # 使用y轴控制横向移动！
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z

        self.publisher.publish(msg)

    def run(self):
        """运行交互式控制循环"""
        try:
            while rclpy.ok():
                # 获取按键
                key = self.get_key()

                # 根据按键执行动作
                if key in ['w', 'W', '\x1b[A']:  # W或上箭头
                    print("前进 (W) - X轴正向")
                    self.publish_twist(self.linear_speed, 0.0, 0.0)

                elif key in ['s', 'S', '\x1b[B']:  # S或下箭头
                    print("后退 (S) - X轴负向")
                    self.publish_twist(-self.linear_speed, 0.0, 0.0)

                elif key in ['a', 'A', '\x1b[D']:  # A或左箭头
                    print("左移 (A) - Y轴负向")
                    self.publish_twist(0.0, -self.linear_speed, 0.0)

                elif key in ['d', 'D', '\x1b[C']:  # D或右箭头
                    print("右移 (D) - Y轴正向")
                    self.publish_twist(0.0, self.linear_speed, 0.0)

                elif key in ['q', 'Q']:  # Q - 左转
                    print("左转 (Q) - 原地逆时针")
                    self.publish_twist(0.0, 0.0, self.angular_speed)

                elif key in ['e', 'E']:  # E - 右转
                    print("右转 (E) - 原地顺时针")
                    self.publish_twist(0.0, 0.0, -self.angular_speed)

                elif key == ' ':  # 空格 - 停止
                    print("停止 (空格)")
                    self.publish_twist(0.0, 0.0, 0.0)

                elif key == '+':
                    self.linear_speed += 0.1
                    if self.linear_speed > 2.0:
                        self.linear_speed = 2.0
                    print(f"加速: 线速度={self.linear_speed:.1f} m/s")

                elif key == '-':
                    self.linear_speed -= 0.1
                    if self.linear_speed < 0.1:
                        self.linear_speed = 0.1
                    print(f"减速: 线速度={self.linear_speed:.1f} m/s")

                elif key == '\x1b':  # ESC
                    print("退出")
                    break

                elif key == '\x03':  # Ctrl+C
                    print("用户中断")
                    break

        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            # 停止小车
            self.publish_twist(0.0, 0.0, 0.0)
            print()
            print("小车已停止")


def main():
    """主函数"""
    rclpy.init()

    controller = CarInteractiveController()

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
