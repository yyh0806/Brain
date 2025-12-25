#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isaac Sim 小车原地绕圈控制脚本

通过分析ROS2 topics并发布控制命令，使小车在Isaac Sim中原地绕圈
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class CircleMotionController(Node):
    """原地绕圈控制节点"""
    
    def __init__(self):
        super().__init__('circle_motion_controller')
        
        # 控制命令发布者
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 订阅里程计以监控位置
        self.odom_sub = self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )
        
        # 位置记录
        self.current_position = None
        self.initial_position = None
        
        print("✓ 控制节点初始化完成")
        print("✓ 已订阅 /chassis/odom (里程计)")
        print("✓ 已发布 /cmd_vel (控制命令)")
    
    def odom_callback(self, msg):
        """里程计回调，记录当前位置"""
        pos = msg.pose.pose.position
        self.current_position = (pos.x, pos.y, pos.z)
        
        if self.initial_position is None:
            self.initial_position = self.current_position
            print(f"\n✓ 记录初始位置: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")
    
    def publish_twist(self, linear_x, angular_z):
        """发布速度控制命令"""
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(angular_z)
        
        self.cmd_pub.publish(twist)
        self.get_logger().debug(
            f'发布命令: linear.x={linear_x:.3f}, angular.z={angular_z:.3f}'
        )
    
    def stop(self):
        """停止小车"""
        self.publish_twist(0.0, 0.0)
        print("\n✓ 已发送停止命令")


def analyze_topics():
    """分析可用的ROS2 topics"""
    print("\n" + "="*70)
    print("分析 ROS2 Topics")
    print("="*70)
    
    if not rclpy.ok():
        rclpy.init()
    
    node = rclpy.create_node('topic_analyzer')
    
    # 等待话题列表
    print("\n等待话题列表（3秒）...")
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.1)
    
    # 获取话题列表
    topic_names_and_types = node.get_topic_names_and_types()
    
    print(f"\n发现 {len(topic_names_and_types)} 个话题")
    
    # 查找关键话题
    key_topics = {
        '/cmd_vel': False,
        '/chassis/odom': False,
        '/chassis/imu': False,
        '/front_3d_lidar/lidar_points': False,
        '/front_stereo_camera/left/image_raw': False
    }
    
    print("\n关键话题检查:")
    for topic_name, topic_type in topic_names_and_types:
        # 处理tuple格式（某些ROS2版本）
        if isinstance(topic_name, tuple):
            topic_name = topic_name[0]
        
        for key_topic in key_topics.keys():
            if key_topic in topic_name:
                key_topics[key_topic] = True
                print(f"  ✓ {topic_name:50s} - {topic_type}")
                break
    
    # 检查缺失的话题
    missing = [topic for topic, found in key_topics.items() if not found]
    if missing:
        print(f"\n⚠️  缺失的关键话题: {', '.join(missing)}")
    else:
        print("\n✅ 所有关键话题都存在！")
    
    node.destroy_node()
    
    return key_topics


def control_circle_motion(linear_speed=0.3, angular_speed=0.5, duration=None, continuous=False):
    """
    控制小车原地绕圈
    
    Args:
        linear_speed: 线速度 (m/s)，控制前进速度
        angular_speed: 角速度 (rad/s)，正值为左转，负值为右转
        duration: 持续时间（秒），如果为None或continuous=True则持续运行
        continuous: 是否持续运行（不停止）
    """
    print("\n" + "="*70)
    print("开始原地绕圈控制")
    print("="*70)
    print(f"参数:")
    print(f"  线速度: {linear_speed} m/s")
    print(f"  角速度: {angular_speed} rad/s")
    if continuous or duration is None:
        print(f"  运行模式: 持续运行（不停止）")
    else:
        print(f"  持续时间: {duration} 秒")
    print("\n请观察 Isaac Sim 中的小车...")
    
    if not rclpy.ok():
        rclpy.init()
    
    controller = CircleMotionController()
    
    # 等待里程计数据
    print("\n等待里程计数据...")
    start_wait = time.time()
    while controller.initial_position is None and (time.time() - start_wait) < 5.0:
        rclpy.spin_once(controller, timeout_sec=0.1)
        time.sleep(0.1)
    
    if controller.initial_position is None:
        print("⚠️  未收到里程计数据，但将继续执行控制命令")
    else:
        print(f"✓ 初始位置已记录: {controller.initial_position}")
    
    # 开始绕圈
    print(f"\n开始绕圈运动...")
    if continuous or duration is None:
        print(f"持续运行中，按 Ctrl+C 停止\n")
    else:
        print(f"按 Ctrl+C 可提前停止\n")
    
    start_time = time.time()
    publish_rate = 10.0  # 10 Hz
    publish_interval = 1.0 / publish_rate
    
    try:
        last_publish_time = time.time()
        last_status_time = time.time()
        
        while True:
            current_time = time.time()
            
            # 检查是否达到持续时间（如果不是持续模式）
            if not continuous and duration is not None:
                if (current_time - start_time) >= duration:
                    break
            
            # 按固定频率发布命令
            if current_time - last_publish_time >= publish_interval:
                controller.publish_twist(linear_speed, angular_speed)
                last_publish_time = current_time
            
            # 每秒更新一次状态显示
            if current_time - last_status_time >= 1.0:
                elapsed = current_time - start_time
                if continuous or duration is None:
                    print(f"\r已运行: {elapsed:.1f}s (持续运行中，按 Ctrl+C 停止)", 
                          end='', flush=True)
                else:
                    remaining = duration - elapsed
                    print(f"\r已运行: {elapsed:.1f}s / {duration:.1f}s (剩余: {remaining:.1f}s)", 
                          end='', flush=True)
                last_status_time = current_time
            
            # 处理ROS2消息
            rclpy.spin_once(controller, timeout_sec=0.01)
            time.sleep(0.01)
        
        if not continuous and duration is not None:
            print("\n\n✓ 绕圈运动完成")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    
    finally:
        # 停止小车
        controller.stop()
        time.sleep(0.5)
        
        # 显示最终位置
        if controller.current_position and controller.initial_position:
            dx = controller.current_position[0] - controller.initial_position[0]
            dy = controller.current_position[1] - controller.initial_position[1]
            distance = (dx**2 + dy**2)**0.5
            elapsed_total = time.time() - start_time
            print(f"\n位置变化:")
            print(f"  初始位置: {controller.initial_position}")
            print(f"  最终位置: {controller.current_position}")
            print(f"  移动距离: {distance:.3f} m")
            print(f"  总运行时间: {elapsed_total:.1f} s")
        
        controller.destroy_node()
        rclpy.shutdown()


def main():
    """主函数"""
    print("="*70)
    print("Isaac Sim 小车原地绕圈控制")
    print("="*70)
    print("\n请确保:")
    print("  1. Isaac Sim 正在运行")
    print("  2. 已按下 Play 按钮")
    print("  3. ROS_DOMAIN_ID=42 已设置")
    print()
    
    # 检查ROS_DOMAIN_ID
    import os
    domain_id = os.environ.get('ROS_DOMAIN_ID', '未设置')
    print(f"当前 ROS_DOMAIN_ID: {domain_id}")
    
    if domain_id != '42':
        print("\n⚠️  警告: ROS_DOMAIN_ID 不是 42")
        response = input("是否继续? (y/n): ").strip().lower()
        if response != 'y':
            print("退出")
            return
    
    # 1. 分析topics
    print("\n步骤 1: 分析 ROS2 Topics")
    key_topics = analyze_topics()
    
    if not key_topics.get('/cmd_vel', False):
        print("\n✗ 错误: 未找到 /cmd_vel 话题，无法控制小车")
        print("请检查 Isaac Sim 是否正常运行")
        return
    
    # 2. 询问控制参数
    print("\n" + "="*70)
    print("步骤 2: 设置控制参数")
    print("="*70)
    
    try:
        # 询问运行模式
        mode_input = input("\n运行模式 [1=持续运行(不停止), 2=定时运行] [默认: 1]: ").strip()
        continuous = (mode_input == '' or mode_input == '1')
        
        linear_input = input("线速度 (m/s) [默认: 0.3]: ").strip()
        linear_speed = float(linear_input) if linear_input else 0.3
        
        angular_input = input("角速度 (rad/s, 正数=左转, 负数=右转) [默认: 0.5]: ").strip()
        angular_speed = float(angular_input) if angular_input else 0.5
        
        if not continuous:
            duration_input = input("持续时间 (秒) [默认: 10.0]: ").strip()
            duration = float(duration_input) if duration_input else 10.0
        else:
            duration = None
        
    except ValueError:
        print("⚠️  输入无效，使用默认值")
        linear_speed = 0.3
        angular_speed = 0.5
        continuous = True
        duration = None
    
    # 3. 执行绕圈控制
    print("\n步骤 3: 执行绕圈控制")
    control_circle_motion(
        linear_speed=linear_speed,
        angular_speed=angular_speed,
        duration=duration,
        continuous=continuous
    )
    
    print("\n" + "="*70)
    print("程序结束")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

