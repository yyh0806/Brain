#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nova Carter 简化测试脚本

用于验证基本的 ROS2 连接和数据接收
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("检查依赖...")

# 检查 ROS2
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    print("✓ ROS2 (rclpy) 可用")
except ImportError as e:
    print(f"✗ ROS2 不可用: {e}")
    sys.exit(1)

# 检查 ROS2 消息
try:
    from sensor_msgs.msg import PointCloud2, Image, Imu
    from nav_msgs.msg import Odometry
    print("✓ ROS2 消息类型可用")
except ImportError as e:
    print(f"✗ ROS2 消息类型不可用: {e}")
    sys.exit(1)

# 检查 numpy
try:
    import numpy as np
    print("✓ NumPy 可用")
except ImportError:
    print("✗ NumPy 不可用")
    sys.exit(1)

print("\n尝试连接到 ROS2...")

# 检查话题
try:
    source /opt/ros/galactic/setup.bash
except Exception as e:
    print(f"✗ 无法 source ROS2: {e}")
    print("请手动运行: source /opt/ros/galactic/setup.bash")

print("\n检查可用的 ROS2 话题...")

import subprocess
result = subprocess.run(
    ["ros2", "topic", "list"],
    capture_output=True,
    text=True,
    timeout=5
)

topics = result.stdout.strip().split('\n')

# 过滤 Nova Carter 相关话题
nova_carter_topics = [
    t for t in topics if any(kw in t.lower() for kw in [
        'cmd_vel', 'odom', 'imu', 'camera', 'lidar', 'point'
    ])
]

print("\n发现的 Nova Carter 相关话题:")
for topic in nova_carter_topics:
    print(f"  - {topic}")

print("\n" + "="*70)
print("如果看到话题列表，说明 ROS2 连接正常！")
print("="*70)

# 初始化 ROS2
if not rclpy.ok():
    rclpy.init()

node = rclpy.create_node('nova_carter_test')

# 创建订阅者
odom_count = 0
pointcloud_count = 0
image_count = 0

def odom_callback(msg):
    global odom_count
    odom_count += 1
    print(f"[里程计] 帧 {odom_count}: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}")

def pointcloud_callback(msg):
    global pointcloud_count
    pointcloud_count += 1
    print(f"[点云] 帧 {pointcloud_count}: {len(msg.data) // 12} 个点")

def image_callback(msg):
    global image_count
    image_count += 1
    print(f"[相机] 帧 {image_count}: {msg.width}x{msg.height}")

# 订阅话题
try:
    odom_sub = node.create_subscription(Odometry, '/chassis/odom', odom_callback)
    pointcloud_sub = node.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points', pointcloud_callback)
    image_sub = node.create_subscription(Image, '/front_stereo_camera/left/image_raw', image_callback)
    
    print("\n已创建订阅:")
    print(f"  里程计: /chassis/odom")
    print(f"  点云: /front_3d_lidar/lidar_points")
    print(f"  相机: /front_stereo_camera/left/image_raw")
    
except Exception as e:
    print(f"订阅失败: {e}")
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(1)

print("\n等待数据 (最多 10 秒)...")

async def spin():
    # 异步循环
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        
        if pointcloud_count > 0 or odom_count > 0:
            break
    
    print(f"\n接收数据统计:")
    print(f"  里程计帧: {odom_count}")
    print(f"  点云帧: {pointcloud_count}")
    print(f"  图像帧: {image_count}")
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(spin())
    except KeyboardInterrupt:
        print("\n\n用户中断，正在清理...")
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print(f"\n错误: {e}")
        node.destroy_node()
        rclpy.shutdown()






