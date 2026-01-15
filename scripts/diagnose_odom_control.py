#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小车控制诊断脚本 - 基于里程计反馈
通过读取odometry验证控制命令是否生效，并检查参数服务器配置
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Odometry
import rclpy.parameter
import time
import math
from collections import deque


class OdomControlDiagnostic(Node):
    """小车控制诊断 - 基于里程计"""

    def __init__(self):
        super().__init__('odom_control_diagnostic')

        # 发布者和订阅者
        self.twist_pub = self.create_publisher(Twist, '/car3/twist', 10)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/car3/local_odom',
            self.odom_callback,
            10
        )

        # 里程计数据
        self.odom_data = Odometry()
        self.odom_history = deque(maxlen=100)  # 保存最近100帧
        self.odom_updated = False

        print("=" * 70)
        print("小车控制诊断 - 基于里程计反馈")
        print("=" * 70)
        print()

        # 等待连接
        print("等待ROS2连接...")
        time.sleep(2)
        self.check_robot_config()

    def check_robot_config(self):
        """检查机器人配置"""
        print("\n【步骤1】检查参数服务器配置")
        print("-" * 70)

        # 列出所有节点
        try:
            node_names = self.get_node_names()
            car_nodes = [n for n in node_names if 'car' in n.lower()]
            print(f"✓ 发现节点: {car_nodes}")
        except Exception as e:
            print(f"✗ 列出节点失败: {e}")

        # 检查话题
        try:
            topic_names = self.get_topic_names_and_types()
            car_topics = [(n, t) for n, t in topic_names if 'car3' in n]
            print(f"✓ car3相关话题:")
            for topic, topic_type in car_topics:
                print(f"    - {topic} ({topic_type[0]})")
        except Exception as e:
            print(f"✗ 列出话题失败: {e}")

        print()

    def odom_callback(self, msg):
        """里程计回调"""
        self.odom_data = msg
        self.odom_updated = True
        self.odom_history.append({
            'timestamp': time.time(),
            'position': {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            },
            'linear_velocity': {
                'x': msg.twist.twist.linear.x,
                'y': msg.twist.twist.linear.y,
                'z': msg.twist.twist.linear.z
            },
            'angular_velocity': {
                'x': msg.twist.twist.angular.x,
                'y': msg.twist.twist.angular.y,
                'z': msg.twist.twist.angular.z
            }
        })

    def get_current_odom(self):
        """获取当前里程计状态"""
        return {
            'position': {
                'x': self.odom_data.pose.pose.position.x,
                'y': self.odom_data.pose.pose.position.y,
                'z': self.odom_data.pose.pose.position.z
            },
            'linear_velocity': {
                'x': self.odom_data.twist.twist.linear.x,
                'y': self.odom_data.twist.twist.linear.y,
                'z': self.odom_data.twist.twist.linear.z
            },
            'angular_velocity': {
                'z': self.odom_data.twist.twist.angular.z
            }
        }

    def wait_for_odom_update(self, timeout=5.0):
        """等待里程计更新"""
        self.odom_updated = False
        start_time = time.time()

        while not self.odom_updated and (time.time() - start_time) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.odom_updated

    def test_control(self, linear_x, linear_y, angular_z, duration, test_name):
        """测试控制命令并验证里程计反馈"""
        print(f"\n【测试】{test_name}")
        print(f"命令: linear.x={linear_x:.2f}, linear.y={linear_y:.2f}, angular.z={angular_z:.2f}")
        print(f"持续时间: {duration}秒")
        print("-" * 70)

        # 记录初始状态
        initial_odom = self.get_current_odom()
        print(f"初始位置: x={initial_odom['position']['x']:.4f}, "
              f"y={initial_odom['position']['y']:.4f}")

        # 清空历史
        self.odom_history.clear()

        # 发布控制命令
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z

        start_time = time.time()
        rate = self.create_rate(20)  # 20Hz

        velocity_samples = []

        while time.time() - start_time < duration:
            self.twist_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.odom_updated:
                vel = self.odom_data.twist.twist
                velocity_samples.append({
                    'linear_x': vel.linear.x,
                    'linear_y': vel.linear.y,
                    'angular_z': vel.angular.z
                })
                self.odom_updated = False

        # 停止
        stop_msg = Twist()
        self.twist_pub.publish(stop_msg)

        # 等待最终里程计更新
        time.sleep(0.5)
        rclpy.spin_once(self, timeout_sec=0.1)

        # 分析结果
        final_odom = self.get_current_odom()

        dx = final_odom['position']['x'] - initial_odom['position']['x']
        dy = final_odom['position']['y'] - initial_odom['position']['y']

        print(f"\n结果分析:")
        print(f"  位置变化: Δx={dx:.4f}m, Δy={dy:.4f}m")

        if velocity_samples:
            avg_lin_x = sum(s['linear_x'] for s in velocity_samples) / len(velocity_samples)
            avg_lin_y = sum(s['linear_y'] for s in velocity_samples) / len(velocity_samples)
            avg_ang_z = sum(s['angular_z'] for s in velocity_samples) / len(velocity_samples)

            print(f"  平均速度反馈:")
            print(f"    linear.x = {avg_lin_x:.4f} m/s (期望: {linear_x:.2f})")
            print(f"    linear.y = {avg_lin_y:.4f} m/s (期望: {linear_y:.2f})")
            print(f"    angular.z = {avg_ang_z:.4f} rad/s (期望: {angular_z:.2f})")

            # 验证
            success = True
            if abs(linear_x) > 0.01 and abs(avg_lin_x) < 0.01:
                print(f"  ✗ X轴控制失败: 期望{linear_x:.2f}，实际{avg_lin_x:.4f}")
                success = False
            elif abs(linear_x) > 0.01:
                print(f"  ✓ X轴控制正常")

            if abs(linear_y) > 0.01 and abs(avg_lin_y) < 0.01:
                print(f"  ✗ Y轴控制失败: 期望{linear_y:.2f}，实际{avg_lin_y:.4f}")
                success = False
            elif abs(linear_y) > 0.01:
                print(f"  ✓ Y轴控制正常")

            if abs(angular_z) > 0.01 and abs(avg_ang_z) < 0.01:
                print(f"  ✗ 角速度控制失败: 期望{angular_z:.2f}，实际{avg_ang_z:.4f}")
                success = False
            elif abs(angular_z) > 0.01:
                print(f"  ✓ 角速度控制正常")

            if success:
                print(f"  ✓✓✓ 测试通过")
            else:
                print(f"  ✗✗✗ 测试失败")

        print()

        return True

    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("\n【步骤2】等待里程计数据...")
        print("-" * 70)

        # 等待里程计数据
        if not self.wait_for_odom_update(timeout=5.0):
            print("✗ 未收到里程计数据！请检查:")
            print("  1. /car3/local_odom 话题是否发布")
            print("  2. 仿真节点是否正常运行")
            return
        else:
            print("✓ 里程计数据接收正常")
            initial = self.get_current_odom()
            print(f"  初始位置: x={initial['position']['x']:.4f}, "
                  f"y={initial['position']['y']:.4f}, "
                  f"z={initial['position']['z']:.4f}")
            print(f"  初始速度: vx={initial['linear_velocity']['x']:.4f}, "
                  f"vy={initial['linear_velocity']['y']:.4f}, "
                  f"vz={initial['angular_velocity']['z']:.4f}")

        print("\n【步骤3】控制测试序列")
        print("=" * 70)

        # 测试1: X轴正向（前进）
        self.test_control(
            linear_x=0.3,
            linear_y=0.0,
            angular_z=0.0,
            duration=2.0,
            test_name="X轴正向 - 前进"
        )
        time.sleep(1)

        # 测试2: X轴负向（后退）
        self.test_control(
            linear_x=-0.3,
            linear_y=0.0,
            angular_z=0.0,
            duration=2.0,
            test_name="X轴负向 - 后退"
        )
        time.sleep(1)

        # 测试3: Y轴正向（右移）
        self.test_control(
            linear_x=0.0,
            linear_y=0.3,
            angular_z=0.0,
            duration=2.0,
            test_name="Y轴正向 - 右移（关键测试）"
        )
        time.sleep(1)

        # 测试4: Y轴负向（左移）
        self.test_control(
            linear_x=0.0,
            linear_y=-0.3,
            angular_z=0.0,
            duration=2.0,
            test_name="Y轴负向 - 左移（关键测试）"
        )
        time.sleep(1)

        # 测试5: 原地左转
        self.test_control(
            linear_x=0.0,
            linear_y=0.0,
            angular_z=0.5,
            duration=2.0,
            test_name="角速度 - 原地左转"
        )
        time.sleep(1)

        # 测试6: 组合运动（斜向）
        self.test_control(
            linear_x=0.2,
            linear_y=0.2,
            angular_z=0.0,
            duration=2.0,
            test_name="组合运动 - 右前方斜向移动"
        )

        print("\n" + "=" * 70)
        print("诊断完成！")
        print("=" * 70)
        print("\n总结:")
        print("  请查看上方各测试的结果分析:")
        print("  - 如果X轴测试通过，说明前进/后退正常")
        print("  - 如果Y轴测试失败，说明机器人不支持横向移动或配置错误")
        print("  - 如果角速度测试通过，说明转向功能正常")
        print("\n建议:")
        print("  1. 如果Y轴始终无响应，检查机器人URDF是否配置全向轮")
        print("  2. 检查控制器插件是否支持linear.y")
        print("  3. 查看RViz中的TF树，确认坐标系定义")
        print("=" * 70)


def main():
    """主函数"""
    rclpy.init()

    try:
        diagnostic = OdomControlDiagnostic()
        diagnostic.run_full_diagnosis()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            diagnostic.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
