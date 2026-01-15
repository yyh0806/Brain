#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
差速驱动自动测试脚本
自动尝试各种控制方式，通过里程计反馈找出有效方法
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
from collections import deque


class DifferentialDriveTester(Node):
    """差速驱动测试器"""

    def __init__(self):
        super().__init__('differential_drive_tester')

        # 发布者和订阅者
        self.cmd_pub = self.create_publisher(Twist, '/car3/twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/car3/local_odom',
            self.odom_callback,
            10
        )

        # 数据记录
        self.odom_data = Odometry()
        self.odom_updated = False
        self.vel_history = deque(maxlen=50)

        # 测试结果
        self.results = []

        print("=" * 70)
        print("差速驱动自动测试")
        print("=" * 70)
        print("\n正在初始化...")
        time.sleep(2)

    def odom_callback(self, msg):
        """里程计回调"""
        self.odom_data = msg
        self.odom_updated = True

        # 记录速度
        self.vel_history.append({
            'x': msg.twist.twist.linear.x,
            'y': msg.twist.twist.linear.y,
            'z': msg.twist.twist.angular.z,
            'time': time.time()
        })

    def wait_for_movement(self, timeout=3.0):
        """等待机器人开始移动"""
        start_time = time.time()
        initial_vel = []

        # 记录初始速度
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.odom_updated:
                initial_vel.append(abs(self.odom_data.twist.twist.linear.x) +
                                  abs(self.odom_data.twist.twist.linear.y) +
                                  abs(self.odom_data.twist.twist.angular.z))
                self.odom_updated = False

        if not initial_vel:
            return None

        baseline = sum(initial_vel) / len(initial_vel)

        # 等待速度变化
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.odom_updated:
                current_vel = (abs(self.odom_data.twist.twist.linear.x) +
                              abs(self.odom_data.twist.twist.linear.y) +
                              abs(self.odom_data.twist.twist.angular.z))

                if current_vel > baseline + 0.05:  # 速度显著增加
                    return True
                self.odom_updated = False

        return False

    def test_control(self, linear_x, linear_y, angular_z, duration, test_name):
        """测试一种控制方式"""
        print(f"\n{'='*70}")
        print(f"测试: {test_name}")
        print(f"命令: linear.x={linear_x:.2f}, linear.y={linear_y:.2f}, angular.z={angular_z:.2f}")
        print(f"{'='*70}")

        # 清空历史
        self.vel_history.clear()

        # 记录初始位置
        rclpy.spin_once(self, timeout_sec=0.1)
        start_x = self.odom_data.pose.pose.position.x
        start_y = self.odom_data.pose.pose.position.y
        start_yaw = self.get_yaw(self.odom_data.pose.pose.orientation)

        # 发送控制命令
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z

        start_time = time.time()
        rate = self.create_rate(20)

        print("执行中... ", end='', flush=True)

        while time.time() - start_time < duration:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

        print("完成")

        # 停止
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)

        # 等待稳定
        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.1)

        # 分析结果
        end_x = self.odom_data.pose.pose.position.x
        end_y = self.odom_data.pose.pose.position.y
        end_yaw = self.get_yaw(self.odom_data.pose.pose.orientation)

        dx = end_x - start_x
        dy = end_y - start_y
        dyaw = end_yaw - start_yaw

        # 计算平均速度
        if self.vel_history:
            avg_vx = sum(v['x'] for v in self.vel_history) / len(self.vel_history)
            avg_vy = sum(v['y'] for v in self.vel_history) / len(self.vel_history)
            avg_vz = sum(v['z'] for v in self.vel_history) / len(self.vel_history)
        else:
            avg_vx = avg_vy = avg_vz = 0.0

        # 计算移动距离
        distance = (dx**2 + dy**2)**0.5

        print(f"\n结果:")
        print(f"  位置变化: Δx={dx:.4f}m, Δy={dy:.4f}m, 移动距离={distance:.4f}m")
        print(f"  姿态变化: Δyaw={dyaw:.4f} rad ({dyaw*180/3.14159:.2f}°)")
        print(f"  平均速度: vx={avg_vx:.4f} m/s, vy={avg_vy:.4f} m/s, ω={avg_vz:.4f} rad/s")

        # 判断是否有效
        is_valid = False
        movement_type = None

        if distance > 0.05:  # 移动超过5cm
            is_valid = True
            if abs(dyaw) < 0.1:  # 姿态变化小
                if dx > 0:
                    movement_type = "前进"
                else:
                    movement_type = "后退"
            elif distance < 0.1 and abs(dyaw) > 0.1:  # 主要是转向
                movement_type = "原地转向"
            else:
                movement_type = "弧线运动"

        if is_valid:
            print(f"  ✓✓✓ 有效！运动类型: {movement_type}")
        else:
            print(f"  ✗✗✗ 无效（移动距离<{0.05}m）")

        result = {
            'test_name': test_name,
            'linear_x': linear_x,
            'linear_y': linear_y,
            'angular_z': angular_z,
            'distance': distance,
            'dyaw': dyaw,
            'avg_vx': avg_vx,
            'avg_vy': avg_vy,
            'avg_vz': avg_vz,
            'valid': is_valid,
            'movement_type': movement_type
        }

        self.results.append(result)
        time.sleep(1)

        return is_valid

    def get_yaw(self, orientation):
        """从四元数获取偏航角"""
        import math
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def run_all_tests(self):
        """运行所有测试"""
        print("\n开始自动测试序列...")
        print("=" * 70)

        # 测试组1: 纯前进/后退
        print("\n【测试组1】纯X轴运动")
        self.test_control(0.3, 0.0, 0.0, 2.0, "前进 - 低速")
        self.test_control(0.5, 0.0, 0.0, 2.0, "前进 - 中速")
        self.test_control(-0.3, 0.0, 0.0, 2.0, "后退")

        # 测试组2: 纯转向
        print("\n【测试组2】纯转向（原地）")
        self.test_control(0.0, 0.0, 0.5, 2.0, "左转 - 中速")
        self.test_control(0.0, 0.0, -0.5, 2.0, "右转 - 中速")
        self.test_control(0.0, 0.0, 1.0, 2.0, "左转 - 快速")

        # 测试组3: 前进+转向（差速驱动标准）
        print("\n【测试组3】前进+转向组合")
        self.test_control(0.2, 0.0, 0.3, 2.0, "前进+左转（低速）")
        self.test_control(0.2, 0.0, 0.5, 2.0, "前进+左转（中速）")
        self.test_control(0.2, 0.0, -0.3, 2.0, "前进+右转（低速）")
        self.test_control(0.3, 0.0, 0.5, 2.0, "前进+左转（快速前进）")
        self.test_control(0.1, 0.0, 0.8, 2.0, "前进+左转（快速转向）")

        # 测试组4: 后退+转向
        print("\n【测试组4】后退+转向组合")
        self.test_control(-0.2, 0.0, 0.3, 2.0, "后退+左转")
        self.test_control(-0.2, 0.0, -0.3, 2.0, "后退+右转")

        # 测试组5: 尝试Y轴（如果支持的话）
        print("\n【测试组5】尝试Y轴控制（全向机器人测试）")
        self.test_control(0.0, 0.3, 0.0, 2.0, "纯右移（Y轴）")
        self.test_control(0.0, -0.3, 0.0, 2.0, "纯左移（Y轴）")
        self.test_control(0.2, 0.2, 0.0, 2.0, "对角线（X+Y）")

        # 测试组6: 极端角速度
        print("\n【测试组6】极端角速度测试")
        self.test_control(0.0, 0.0, 2.0, 2.0, "超高速左转")
        self.test_control(0.0, 0.0, -2.0, 2.0, "超高速右转")

        # 打印总结
        self.print_summary()

    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)

        valid_tests = [r for r in self.results if r['valid']]
        invalid_tests = [r for r in self.results if not r['valid']]

        print(f"\n总测试数: {len(self.results)}")
        print(f"有效测试: {len(valid_tests)} ✓")
        print(f"无效测试: {len(invalid_tests)} ✗")

        if valid_tests:
            print(f"\n{'='*70}")
            print("有效控制方式:")
            print('='*70)

            # 按运动类型分组
            forward_tests = [r for r in valid_tests if r['movement_type'] == '前进']
            backward_tests = [r for r in valid_tests if r['movement_type'] == '后退']
            turn_tests = [r for r in valid_tests if r['movement_type'] == '原地转向']
            curve_tests = [r for r in valid_tests if r['movement_type'] == '弧线运动']

            if forward_tests:
                print(f"\n✓ 前进:")
                best = max(forward_tests, key=lambda x: x['distance'])
                print(f"  最佳: linear.x={best['linear_x']:.2f}, angular.z={best['angular_z']:.2f}")
                print(f"       移动距离={best['distance']:.4f}m")

            if backward_tests:
                print(f"\n✓ 后退:")
                best = max(backward_tests, key=lambda x: abs(x['distance']))
                print(f"  最佳: linear.x={best['linear_x']:.2f}, angular.z={best['angular_z']:.2f}")
                print(f"       移动距离={abs(best['distance']):.4f}m")

            if turn_tests:
                print(f"\n✓ 转向:")
                best = max(turn_tests, key=lambda x: abs(x['dyaw']))
                print(f"  最佳: linear.x={best['linear_x']:.2f}, angular.z={best['angular_z']:.2f}")
                print(f"       转向角度={abs(best['dyaw'])*180/3.14159:.2f}°")

            if curve_tests:
                print(f"\n✓ 弧线运动（前进+转向）:")
                best = max(curve_tests, key=lambda x: x['distance'])
                print(f"  最佳: linear.x={best['linear_x']:.2f}, angular.z={best['angular_z']:.2f}")
                print(f"       移动距离={best['distance']:.4f}m, 转向={best['dyaw']*180/3.14159:.2f}°")

        print(f"\n{'='*70}")
        print("推荐控制方式:")
        print('='*70)

        # 根据测试结果给出建议
        if not valid_tests:
            print("⚠ 警告: 没有找到任何有效的控制方式！")
            print("\n可能原因:")
            print("  1. 机器人未启动或未连接")
            print("  2. /car3/twist 话题未连接")
            print("  3. 里程计数据未发布")
        else:
            # 分析控制模式
            has_forward = any(r['movement_type'] == '前进' for r in valid_tests)
            has_turn = any(r['movement_type'] == '原地转向' for r in valid_tests)
            has_curve = any(r['movement_type'] == '弧线运动' for r in valid_tests)
            has_y_axis = any(abs(r['avg_vy']) > 0.01 for r in valid_tests)

            if has_forward and has_turn and has_curve:
                print("\n✓ 标准差速驱动模式")
                print("\n控制方式:")
                print("  - 前进: linear.x > 0, angular.z = 0")
                print("  - 后退: linear.x < 0, angular.z = 0")
                print("  - 左转: linear.x > 0, angular.z > 0 (或 linear.x=0, angular.z>0 原地)")
                print("  - 右转: linear.x > 0, angular.z < 0 (或 linear.x=0, angular.z<0 原地)")

                # 找出最佳参数
                if has_curve:
                    best_curve = max([r for r in valid_tests if r['movement_type'] == '弧线运动'],
                                    key=lambda x: x['distance'])
                    print(f"\n推荐参数（基于测试）:")
                    print(f"  前进速度: 0.2-0.5 m/s")
                    print(f"  转向速度: 0.3-0.8 rad/s")
                    print(f"  组合示例: linear.x={best_curve['linear_x']:.2f}, "
                          f"angular.z={best_curve['angular_z']:.2f}")

            elif has_y_axis:
                print("\n✓ 全向机器人模式（支持Y轴）")
                print("\n控制方式:")
                print("  - 前进: linear.x > 0")
                print("  - 后退: linear.x < 0")
                print("  - 左移: linear.y < 0")
                print("  - 右移: linear.y > 0")
                print("  - 原地转向: angular.z != 0")
            else:
                print("\n未知控制模式，请查看上方详细测试结果")

        print("="*70)


def main():
    """主函数"""
    rclpy.init()

    try:
        tester = DifferentialDriveTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            tester.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
