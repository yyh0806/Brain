#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nova Carter 可视化启动脚本

功能：
1. 启动 RViz2 并加载配置
2. 发布测试数据（如果需要）
3. 显示传感器数据统计

用法:
    python3 launch_nova_carter_viz.py
"""

import sys
import subprocess
import time
from pathlib import Path
import signal

print("""
================================================================================
 Nova Carter 可视化启动器
================================================================================

本脚本将：
1. 启动 RViz2 并预配置显示
2. 显示传感器数据统计
3. 提供可选的测试数据发布

显示内容：
  - 3D 点云（激光雷达）
  - 机器人轨迹（里程计）
  - RGB 相机画面
  - 机器人坐标系（TF）
  - 网格（参考坐标系）
""")

# 检查 RViz2 是否安装
print("\n检查 RViz2...")
try:
    result = subprocess.run(
        ["which", "rviz2"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✓ RViz2 找到: {result.stdout.strip()}")
    else:
        print("✗ RViz2 未找到")
        print("  请先安装: sudo apt install ros-galactic-rviz2")
        sys.exit(1)
except Exception as e:
    print(f"✗ 检查失败: {e}")
    sys.exit(1)

# RViz2 配置文件路径
rviz_config = Path(__file__).parent / "config/rviz2/nova_carter_fixed.rviz"

if not rviz_config.exists():
    print(f"\n✗ RViz 配置文件不存在: {rviz_config}")
    sys.exit(1)

print(f"✓ RViz 配置文件: {rviz_config}")

# 启动 RViz2
print("\n启动 RViz2...")
print("提示：使用鼠标操作视图：")
print("  - 左键拖动：旋转")
print("  - 中键拖动：平移")
print("  - 滚轮：缩放")
print("  - 右键：设置焦点\n")

rviz_process = subprocess.Popen(
    ["rviz2", "-d", str(rviz_config)],
    env={
        **subprocess.os.environ,
        "ROS_DOMAIN_ID": "0"
    }
)

print(f"✓ RViz2 进程 ID: {rviz_process.pid}")
print("\n等待 RViz2 加载（10秒）...\n")

# 等待 RViz2 启动
time.sleep(3)

# 检查是否有传感器数据
print("检查传感器话题...")
import rclpy
from rclpy.node import Node

if not rclpy.ok():
    rclpy.init()

node = rclpy.create_node('viz_monitor')

# 订阅器统计
stats = {
    'pointcloud': 0,
    'odom': 0,
    'image': 0
}

def pointcloud_callback(msg):
    stats['pointcloud'] += 1
    if stats['pointcloud'] % 10 == 0:
        print(f"[点云] 已接收 {stats['pointcloud']} 帧")

def odom_callback(msg):
    stats['odom'] += 1
    if stats['odom'] % 10 == 0:
        print(f"[里程计] 已接收 {stats['odom']} 帧")

def image_callback(msg):
    stats['image'] += 1
    if stats['image'] % 30 == 0:  # 相机帧率高，少打印
        print(f"[相机] 已接收 {stats['image']} 帧")

from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry

try:
    pc_sub = node.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points', pointcloud_callback, 10)
    odom_sub = node.create_subscription(Odometry, '/chassis/odom', odom_callback, 10)
    img_sub = node.create_subscription(Image, '/front_stereo_camera/left/image_raw', image_callback, 10)
    
    print("✓ 已订阅话题:")
    print("  - /front_3d_lidar/lidar_points")
    print("  - /chassis/odom")
    print("  - /front_stereo_camera/left/image_raw")
    
except Exception as e:
    print(f"✗ 订阅失败: {e}")
    node.destroy_node()
    rviz_process.terminate()
    sys.exit(1)

print("\n监控传感器数据（按 Ctrl+C 退出）...")
print("-"*70)

# 监控循环
try:
    start_time = time.time()
    last_print = start_time
    
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        
        current_time = time.time()
        
        # 每5秒打印统计
        if current_time - last_print > 5.0:
            elapsed = current_time - start_time
            rate_pointcloud = stats['pointcloud'] / elapsed
            rate_odom = stats['odom'] / elapsed
            rate_image = stats['image'] / elapsed
            
            print(f"\n[统计] 运行时间: {elapsed:.1f}s | "
                  f"点云: {stats['pointcloud']} ({rate_pointcloud:.1f} Hz) | "
                  f"里程计: {stats['odom']} ({rate_odom:.1f} Hz) | "
                  f"相机: {stats['image']} ({rate_image:.1f} Hz)")
            
            if stats['pointcloud'] > 0:
                print("  ✓ 点云数据正常")
            else:
                print("  ✗ 未收到点云数据")
            
            if stats['odom'] > 0:
                print("  ✓ 里程计数据正常")
            else:
                print("  ✗ 未收到里程计数据")
            
            if stats['image'] > 0:
                print("  ✓ 相机数据正常")
            else:
                print("  ✗ 未收到相机数据")
            
            print()
            last_print = current_time
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n停止监控...")

# 清理
node.destroy_node()
rclpy.shutdown()

print("\n清理进程...")
rviz_process.terminate()
rviz_process.wait()

print("\nRViz2 已关闭")
print("再见！")


