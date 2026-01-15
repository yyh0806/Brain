#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车控制诊断脚本
测试不同的控制方式，找出可用的控制方法
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class CarControlDiagnostics(Node):
    """小车控制诊断"""

    def __init__(self):
        super().__init__('car_control_diagnostics')
        self.publisher = self.create_publisher(Twist, '/car3/twist', 10)

        print("=" * 60)
        print("小车控制诊断")
        print("=" * 60)
        print()

    def test_control(self, linear_x, angular_z, duration, description):
        """测试一种控制方式"""
        print(f"测试: {description}")
        print(f"  命令: linear.x={linear_x}, angular.z={angular_z}")
        print(f"  持续时间: {duration}秒")

        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z

        start_time = time.time()
        rate = self.create_rate(10)

        while time.time() - start_time < duration:
            self.publisher.publish(msg)
            rate.sleep()

        # 停止
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.publisher.publish(stop_msg)

        print(f"  ✓ 完成")
        print()

    def run_diagnostics(self):
        """运行诊断测试"""
        print("开始诊断测试...")
        print()

        # 测试1: 纯前进
        print("【测试1】纯前进（baseline）")
        self.test_control(0.5, 0.0, 2.0, "前进 - 线速度0.5m/s")
        time.sleep(1)

        # 测试2: 纯角速度（原地转向）
        print("【测试2】纯角速度（原地转向）")
        self.test_control(0.0, 0.5, 2.0, "原地左转 - 角速度0.5rad/s")
        time.sleep(1)

        # 测试3: 前进+转向（差速驱动）
        print("【测试3】前进+转向（差速驱动）")
        self.test_control(0.2, 0.5, 2.0, "左转 - 线速度0.2m/s + 角速度0.5rad/s")
        time.sleep(1)

        # 测试4: 更大角速度
        print("【测试4】更大角速度")
        self.test_control(0.2, 1.0, 2.0, "快速左转 - 线速度0.2m/s + 角速度1.0rad/s")
        time.sleep(1)

        # 测试5: 负角速度（右转）
        print("【测试5】右转")
        self.test_control(0.2, -0.5, 2.0, "右转 - 线速度0.2m/s + 角速度-0.5rad/s")
        time.sleep(1)

        # 测试6: 尝试使用linear.y（全向移动）
        print("【测试6】尝试侧向移动（全向机器人）")
        self.test_control(0.0, 0.0, 2.0, "测试前...先设置为0")

        msg = Twist()
        msg.linear.y = 0.5  # 侧向
        msg.angular.z = 0.0

        print("  命令: linear.y=0.5 (侧向)")
        start_time = time.time()
        rate = self.create_rate(10)

        while time.time() - start_time < 2.0:
            self.publisher.publish(msg)
            rate.sleep()

        stop_msg = Twist()
        self.publisher.publish(stop_msg)
        print("  ✓ 完成")
        print()

        print("=" * 60)
        print("诊断完成！")
        print()
        print("请观察RViz中小车的反应：")
        print("  - 如果测试1有效，说明前进功能正常")
        print("  - 如果测试2-5都无效，说明小车可能不支持角速度控制")
        print("  - 如果测试6有效，说明是全向机器人")
        print("=" * 60)


def main():
    """主函数"""
    rclpy.init()

    diagnostics = CarControlDiagnostics()

    try:
        diagnostics.run_diagnostics()
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        diagnostics.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
