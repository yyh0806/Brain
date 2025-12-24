#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nova Carter ROS2 话题检查和测试脚本

验证 Isaac Sim Nova Carter 的 ROS2 话题是否正常工作
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from loguru import logger


def check_topics():
    """检查 Nova Carter 相关话题"""
    if not rclpy.ok():
        rclpy.init()
    
    print("="*70)
    print("Nova Carter ROS2 话题检查")
    print("="*70)
    
    node = rclpy.create_node('topic_checker')
    
    # 等待话题列表
    print("\n等待话题列表（3秒）...")
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.1)
    
    # 获取话题列表
    topic_names_and_types = node.get_topic_names_and_types()
    
    # Nova Carter 相关话题
    target_topics = {
        '/cmd_vel': '控制',
        '/chassis/odom': '里程计',
        '/chassis/imu': 'IMU',
        '/front_3d_lidar/lidar_points': '3D激光雷达（点云）',
        '/front_stereo_camera/left/image_raw': '左相机',
        '/front_stereo_camera/right/image_raw': '右相机',
        '/front_stereo_imu/imu': '前IMU',
        '/back_stereo_imu/imu': '后IMU',
        '/left_stereo_imu/imu': '左IMU',
        '/right_stereo_imu/imu': '右IMU'
    }
    
    print(f"\n发现 {len(topic_names_and_types)} 个话题")
    print("\nNova Carter 相关话题:")
    
    nova_topics = []
    for topic_name, topic_type in topic_names_and_types:
        # 转换为字符串（Galactic 返回 tuple）
        if isinstance(topic_name, tuple):
            topic_name = topic_name[0]
        
        # 检查是否在目标列表中
        for target, desc in target_topics.items():
            if target in topic_name:
                nova_topics.append({
                    'topic': topic_name,
                    'type': topic_type,
                    'desc': desc
                })
                print(f"  ✓ {topic_name:30s} - {desc}")
                break
    
    if not nova_topics:
        print("  ✗ 未发现预期的 Nova Carter 话题")
        print("\n所有话题:")
        for topic_name, topic_type in topic_names_and_types[:20]:
            if isinstance(topic_name, tuple):
                name = topic_name[0]
            else:
                name = topic_name
            print(f"  - {name:40s} ({topic_type})")
    else:
        print(f"\n总共发现 {len(nova_topics)} 个 Nova Carter 话题")
        
        # 关键话题检查
        critical_topics = ['/cmd_vel', '/chassis/odom', '/front_3d_lidar/lidar_points']
        missing = []
        for topic in critical_topics:
            if topic not in [t['topic'] for t in nova_topics]:
                missing.append(topic)
        
        if missing:
            print(f"\n✗ 缺少关键话题: {', '.join(missing)}")
        else:
            print("\n✅ 所有关键话题都存在！")
    
    node.destroy_node()
    
    return nova_topics


def test_topic_subscription():
    """测试话题订阅"""
    if not rclpy.ok():
        rclpy.init()
    
    print("\n" + "="*70)
    print("测试话题订阅")
    print("="*70)
    
    node = rclpy.create_node('subscription_tester')
    
    # 订阅者
    odom_received = False
    pointcloud_received = False
    image_received = False
    imu_received = False
    
    def odom_callback(msg):
        nonlocal odom_received
        odom_received = True
        print(f"  [里程计] x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}, z={msg.pose.pose.position.z:.3f}")
    
    def pointcloud_callback(msg):
        nonlocal pointcloud_received
        pointcloud_received = True
        num_points = len(msg.data) // 12  # PointCloud2 是 xyz + padding
        print(f"  [点云] {num_points} 个点, height={msg.height}, width={msg.width}")
    
    def image_callback(msg):
        nonlocal image_received
        image_received = True
        print(f"  [相机] {msg.width}x{msg.height}, 编码={msg.encoding}")
    
    def imu_callback(msg):
        nonlocal imu_received
        imu_received = True
        print(f"  [IMU] 加速度=({msg.linear_acceleration.x:.2f}, {msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f})")
    
    # 创建订阅者（Galactic 语法）
    try:
        odom_sub = node.create_subscription(Odometry, '/chassis/odom', odom_callback, 10)
        print("  订阅里程计: /chassis/odom")
    except Exception as e:
        print(f"  订阅里程计失败: {e}")
    
    try:
        pc_sub = node.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points', pointcloud_callback, 10)
        print("  订阅点云: /front_3d_lidar/lidar_points")
    except Exception as e:
        print(f"  订阅点云失败: {e}")
    
    try:
        img_sub = node.create_subscription(Image, '/front_stereo_camera/left/image_raw', image_callback, 10)
        print("  订阅相机: /front_stereo_camera/left/image_raw")
    except Exception as e:
        print(f"  订阅相机失败: {e}")
    
    try:
        imu_sub = node.create_subscription(Imu, '/chassis/imu', imu_callback, 10)
        print("  订阅IMU: /chassis/imu")
    except Exception as e:
        print(f"  订阅IMU失败: {e}")
    
    print("\n等待数据（最多 10 秒）...")
    print("如果看到数据输出，说明订阅成功！\n")
    
    # 等待数据
    start_time = time.time()
    
    while (time.time() - start_time) < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.05)
    
    # 清理
    node.destroy_node()
    
    # 总结
    print("\n" + "="*70)
    print("订阅测试结果")
    print("="*70)
    print(f"里程计: {'✓ 收到' if odom_received else '✗ 未收到'}")
    print(f"点云: {'✓ 收到' if pointcloud_received else '✗ 未收到'}")
    print(f"相机: {'✓ 收到' if image_received else '✗ 未收到'}")
    print(f"IMU: {'✓ 收到' if imu_received else '✗ 未收到'}")
    
    if any([odom_received, pointcloud_received, image_received, imu_received]):
        print("\n✅ 传感器数据订阅成功！")
    else:
        print("\n✗ 所有传感器订阅失败，可能原因:")
        print("  1. Isaac Sim 未按 Play")
        print("  2. ROS2 Bridge 未启用")
        print("  3. 话题名称不匹配")


def test_control():
    """测试控制命令"""
    if not rclpy.ok():
        rclpy.init()
    
    print("\n" + "="*70)
    print("测试控制命令")
    print("="*70)
    
    node = rclpy.create_node('control_tester')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    
    print("发布控制命令到 /cmd_vel")
    print("测试序列：")
    print("  1. 前进 0.5 m/s (1秒)")
    print("  2. 停止 (1秒)")
    print("  3. 左转 0.5 rad/s (1秒)")
    print("  4. 停止\n")
    
    try:
        rclpy.init()
        
        # 前进
        twist = Twist()
        twist.linear.x = 0.5
        twist.angular.z = 0.0
        pub.publish(twist)
        print("  ✓ 前进命令已发布")
        
        rclpy.spin_once(node, timeout_sec=1.0)
        
        # 停止
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)
        print("  ✓ 停止命令已发布")
        
        rclpy.spin_once(node, timeout_sec=1.0)
        
        # 左转
        twist.linear.x = 0.3
        twist.angular.z = 0.5
        pub.publish(twist)
        print("  ✓ 左转命令已发布")
        
        rclpy.spin_once(node, timeout_sec=1.0)
        
        # 最终停止
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        pub.publish(twist)
        print("  ✓ 最终停止命令已发布")
        
        rclpy.spin_once(node, timeout_sec=0.5)
        
        print("\n✅ 控制命令测试完成！")
        print("你应该在 Isaac Sim 中看到小车移动。")
        
    except Exception as e:
        print(f"✗ 控制测试失败: {e}")
    
    node.destroy_node()
    rclpy.shutdown()


def main():
    """主函数"""
    print("""
========================================================================
 Nova Carter ROS2 话题验证和测试
========================================================================

本脚本将：
1. 列出所有 ROS2 话题
2. 筛选 Nova Carter 相关话题
3. 测试传感器数据订阅
4. 测试控制命令发送

请确保 Isaac Sim 正在运行，并已按下 Play 按钮！
""")
    
    # 1. 检查话题
    nova_topics = check_topics()
    
    if nova_topics:
        # 2. 测试订阅
        test_topic_subscription()
        
        # 3. 测试控制
        response = input("\n是否测试控制命令? (y/n): ").strip().lower()
        if response == 'y':
            test_control()
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()


