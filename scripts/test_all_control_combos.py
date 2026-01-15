#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整测试差速驱动所有控制组合
基于odom反馈验证哪些方式有效
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import math


class ComprehensiveControlTest(Node):
    """完整控制测试"""

    def __init__(self):
        super().__init__('comprehensive_control_test')

        # 控制发布者
        self.cmd_pub = self.create_publisher(Twist, '/car3/twist', 10)

        # 里程计订阅
        self.odom_sub = self.create_subscription(
            Odometry,
            '/car3/car_info',  # 使用仿真环境的正确话题
            self.odom_callback,
            10
        )

        self.odom_data = None
        self.odom_count = 0

        print("等待里程计数据...")
        for _ in range(50):  # 等待最多5秒
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.odom_data is not None:
                break

        if self.odom_data is None:
            print("⚠️ 警告: 未收到里程计数据")
        else:
            print("✅ 里程计已连接")

    def odom_callback(self, msg):
        self.odom_data = msg
        self.odom_count += 1

        # 每10次打印一次
        if self.odom_count % 10 == 0:
            pos = msg.pose.pose.position
            vel = msg.twist.twist
            print(f"\r[Odom #{self.odom_count}] 位置: ({pos.x:.3f}, {pos.y:.3f}) "
                  f"速度: vx={vel.linear.x:.3f}, vy={vel.linear.y:.3f}, ω={vel.angular.z:.3f}",
                  end='', flush=True)

    def get_yaw(self, orientation):
        """从四元数获取偏航角"""
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def test_control(self, linear_x, linear_y, angular_z, duration, test_name):
        """测试一种控制方式"""
        print(f"\n\n{'='*70}")
        print(f"测试: {test_name}")
        print(f"命令: linear.x={linear_x:.2f}, linear.y={linear_y:.2f}, angular.z={angular_z:.2f}")
        print(f"{'='*70}")

        if self.odom_data is None:
            print("❌ 无里程计数据，跳过测试")
            return False

        # 记录初始状态
        start_x = self.odom_data.pose.pose.position.x
        start_y = self.odom_data.pose.pose.position.y
        start_yaw = self.get_yaw(self.odom_data.pose.pose.orientation)

        print(f"初始: 位置=({start_x:.4f}, {start_y:.4f}), 航向={math.degrees(start_yaw):.2f}°")

        # 发布控制命令
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)

        start_time = time.time()
        rate = self.create_rate(20)  # 20Hz

        print("执行控制命令... ", end='', flush=True)

        while time.time() - start_time < duration:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        # 停止
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)

        print("完成")

        # 等待odom更新
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)

        if self.odom_data is None:
            print("❌ 里程计数据丢失")
            return False

        # 记录最终状态
        end_x = self.odom_data.pose.pose.position.x
        end_y = self.odom_data.pose.pose.position.y
        end_yaw = self.get_yaw(self.odom_data.pose.pose.orientation)

        dx = end_x - start_x
        dy = end_y - start_y
        dyaw = end_yaw - start_yaw
        distance = math.sqrt(dx**2 + dy**2)

        # 获取速度反馈
        avg_vx = self.odom_data.twist.twist.linear.x
        avg_vy = self.odom_data.twist.twist.linear.y
        avg_vz = self.odom_data.twist.twist.angular.z

        print(f"\n结果:")
        print(f"  位置变化: Δx={dx:.4f}m, Δy={dy:.4f}m, 距离={distance:.4f}m")
        print(f"  航向变化: Δyaw={math.degrees(dyaw):.2f}°")
        print(f"  当前速度: vx={avg_vx:.4f}, vy={avg_vy:.4f}, ω={avg_vz:.4f}")

        # 判断有效性
        is_valid = False
        movement_type = None

        if distance > 0.02:  # 移动超过2cm
            if abs(dyaw) < 0.05:  # 基本直线
                if abs(dx) > abs(dy):
                    movement_type = "前进/后退 (X轴)"
                else:
                    movement_type = "横移 (Y轴)"
            elif distance < 0.05 and abs(dyaw) > 0.05:  # 主要是转向
                movement_type = "原地转向"
            else:
                movement_type = "弧线运动"

            is_valid = True

        if is_valid:
            print(f"  ✅ 有效! 运动类型: {movement_type}")
        else:
            print(f"  ❌ 无效 (移动<{0.02}m)")

        time.sleep(0.5)
        return is_valid

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("差速驱动完整测试序列")
        print("="*70)

        results = []

        # ========== 测试组1: 纯X轴 ==========
        print("\n【组1】纯X轴运动 (前进/后退)")
        results.append(('纯前进', self.test_control(0.5, 0.0, 0.0, 3.0, "纯前进 - linear.x=0.5")))
        results.append(('纯后退', self.test_control(-0.5, 0.0, 0.0, 3.0, "纯后退 - linear.x=-0.5")))

        # ========== 测试组2: 纯转向 ==========
        print("\n【组2】纯转向 (原地)")
        results.append(('左转', self.test_control(0.0, 0.0, 0.5, 3.0, "原地左转 - angular.z=0.5")))
        results.append(('右转', self.test_control(0.0, 0.0, -0.5, 3.0, "原地右转 - angular.z=-0.5")))
        results.append(('快速左转', self.test_control(0.0, 0.0, 1.0, 3.0, "快速左转 - angular.z=1.0")))

        # ========== 测试组3: 前进+转向 ==========
        print("\n【组3】前进+转向组合 (差速驱动标准)")
        results.append(('前+左转', self.test_control(0.3, 0.0, 0.5, 3.0, "前进+左转 - x=0.3, ω=0.5")))
        results.append(('前+右转', self.test_control(0.3, 0.0, -0.5, 3.0, "前进+右转 - x=0.3, ω=-0.5")))
        results.append(('慢速前+快转', self.test_control(0.2, 0.0, 0.8, 3.0, "慢速前进+快速左转 - x=0.2, ω=0.8")))
        results.append(('快速前+慢转', self.test_control(0.5, 0.0, 0.3, 3.0, "快速前进+慢速左转 - x=0.5, ω=0.3")))

        # ========== 测试组4: 后退+转向 ==========
        print("\n【组4】后退+转向组合")
        results.append(('后+左转', self.test_control(-0.3, 0.0, 0.5, 3.0, "后退+左转 - x=-0.3, ω=0.5")))
        results.append(('后+右转', self.test_control(-0.3, 0.0, -0.5, 3.0, "后退+右转 - x=-0.3, ω=-0.5")))

        # ========== 测试组5: 尝试Y轴 ==========
        print("\n【组5】Y轴控制 (全向机器人测试)")
        results.append(('纯右移', self.test_control(0.0, 0.5, 0.0, 3.0, "纯右移 - linear.y=0.5")))
        results.append(('纯左移', self.test_control(0.0, -0.5, 0.0, 3.0, "纯左移 - linear.y=-0.5")))

        # ========== 打印总结 ==========
        self.print_summary(results)

    def print_summary(self, results):
        """打印测试总结"""
        print("\n" + "="*70)
        print("测试总结报告")
        print("="*70)

        valid_count = sum(1 for _, valid in results if valid)
        total_count = len(results)

        print(f"\n总测试: {total_count}")
        print(f"有效: {valid_count} ✅")
        print(f"无效: {total_count - valid_count} ❌")

        print(f"\n{'='*70}")
        print("有效控制方式:")
        print('='*70)

        for name, valid in results:
            if valid:
                print(f"  ✅ {name}")

        print(f"\n{'='*70}")
        print("无效控制方式:")
        print('='*70)

        for name, valid in results:
            if not valid:
                print(f"  ❌ {name}")

        # 分析机器人类型
        print(f"\n{'='*70}")
        print("机器人类型分析:")
        print('='*70)

        has_x = any(name in ['纯前进', '纯后退'] and valid for name, valid in results)
        has_turn = any(name in ['左转', '右转'] and valid for name, valid in results)
        has_curve = any(name in ['前+左转', '前+右转'] and valid for name, valid in results)
        has_y = any(name in ['纯右移', '纯左移'] and valid for name, valid in results)

        if has_x and has_turn and has_curve and not has_y:
            print("\n🔍 机器人类型: **标准差速驱动** (Differential Drive)")
            print("\n✅ 推荐控制方式:")
            print("  - 前进: linear.x > 0, angular.z = 0")
            print("  - 后退: linear.x < 0, angular.z = 0")
            print("  - 左转: linear.x > 0, angular.z > 0 (或原地: x=0, ω>0)")
            print("  - 右转: linear.x > 0, angular.z < 0 (或原地: x=0, ω<0)")

        elif has_x and has_y and has_turn:
            print("\n🔍 机器人类型: **全向机器人** (Omnidirectional)")
            print("\n✅ 推荐控制方式:")
            print("  - 前进: linear.x > 0")
            print("  - 后退: linear.x < 0")
            print("  - 左移: linear.y < 0")
            print("  - 右移: linear.y > 0")
            print("  - 原地转向: angular.z != 0")

        elif has_x and not has_turn:
            print("\n🔍 机器人类型: **简单X轴移动** (无转向能力)")
            print("\n✅ 推荐控制方式:")
            print("  - 前进: linear.x > 0")
            print("  - 后退: linear.x < 0")

        else:
            print("\n⚠️  未知机器人类型，请查看详细测试结果")

        # 找出最佳参数
        if has_curve:
            curve_tests = [(name, valid) for name, valid in results if '前+' in name and valid]
            if curve_tests:
                print(f"\n💡 最佳转向参数来自有效测试: {curve_tests[0][0]}")

        print("="*70)


def main():
    rclpy.init()

    try:
        tester = ComprehensiveControlTest()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
