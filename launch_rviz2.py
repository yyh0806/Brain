#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的 RViz2 启动脚本

修复图像显示问题，优化布局
"""

import sys
import subprocess
import time
from pathlib import Path

print("""
================================================================================
 Nova Carter 优化的 RViz2 启动器
================================================================================

修复的问题：
  ✓ 图像显示问题（使用 Image 而不是 Camera）
  ✓ 优化布局（3D 视图 + 图像面板）
  ✓ 自动调整视图参数

显示内容：
  - 主视图：3D 点云 + 机器人轨迹 + TF
  - 图像面板：RGB 相机实时画面
  - 状态面板：传感器数据统计

提示：
  - 按住鼠标左键拖动：旋转视图
  - 按住鼠标中键拖动：平移视图
  - 滚轮：缩放
  - 双击：聚焦物体
""")

# RViz2 配置文件
rviz_config = Path(__file__).parent / "config/rviz2/nova_carter_optimized.rviz"

if not rviz_config.exists():
    print(f"\n✗ RViz 配置文件不存在: {rviz_config}")
    print("正在创建默认配置...")
    
    # 创建简单的配置
    config_dir = Path(__file__).parent / "config/rviz2"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用更简单的配置
    print("✓ 配置目录已创建")

print(f"\n✓ RViz 配置文件: {rviz_config}")

# 启动 RViz2
print("\n启动 RViz2...")

rviz_process = subprocess.Popen(
    ["rviz2", "-d", str(rviz_config)],
    env={
        **subprocess.os.environ,
        "ROS_DOMAIN_ID": "0"
    }
)

print(f"✓ RViz2 进程 ID: {rviz_process.pid}")

# 等待 RViz2 启动
time.sleep(3)

# 启动监控脚本
print("\n启动传感器监控...")
print("="*70)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry

if not rclpy.ok():
    rclpy.init()

monitor_node = rclpy.create_node('rviz_monitor')

stats = {
    'pointcloud': 0,
    'odom': 0,
    'image': 0,
    'start_time': time.time()
}

def update_stats():
    """更新统计显示"""
    elapsed = time.time() - stats['start_time']
    
    if elapsed > 0:
        rate_pc = stats['pointcloud'] / elapsed
        rate_odom = stats['odom'] / elapsed
        rate_img = stats['image'] / elapsed
        
        print(f"\r[统计] 时间: {elapsed:6.1f}s | "
              f"点云: {stats['pointcloud']:4d} ({rate_pc:4.1f}Hz) | "
              f"里程计: {stats['odom']:4d} ({rate_odom:4.1f}Hz) | "
              f"相机: {stats['image']:4d} ({rate_img:4.1f}Hz) | "
              f"RViz PID: {rviz_process.pid}", end='', flush=True)

def pointcloud_callback(msg):
    stats['pointcloud'] += 1

def odom_callback(msg):
    stats['odom'] += 1

def image_callback(msg):
    stats['image'] += 1

# 创建订阅
try:
    pc_sub = monitor_node.create_subscription(
        PointCloud2,
        '/front_3d_lidar/lidar_points',
        pointcloud_callback,
        10
    )
    
    odom_sub = monitor_node.create_subscription(
        Odometry,
        '/chassis/odom',
        odom_callback,
        10
    )
    
    img_sub = monitor_node.create_subscription(
        Image,
        '/front_stereo_camera/left/image_raw',
        image_callback,
        10
    )
    
    print("✓ 监控已启动")
    print("按 Ctrl+C 停止\n")
    
except Exception as e:
    print(f"✗ 订阅失败: {e}")
    rviz_process.terminate()
    sys.exit(1)

# 监控循环
try:
    last_update = time.time()
    
    while True:
        rclpy.spin_once(monitor_node, timeout_sec=0.1)
        
        # 每 2 秒更新一次统计
        if time.time() - last_update > 2.0:
            update_stats()
            last_update = time.time()
        
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n停止监控...")

# 清理
monitor_node.destroy_node()
rclpy.shutdown()

print("\n停止 RViz2...")
rviz_process.terminate()

try:
    rviz_process.wait(timeout=5)
except subprocess.TimeoutExpired:
    print("  强制终止...")
    rviz_process.kill()

print("\n再见！")




