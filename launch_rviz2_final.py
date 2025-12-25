#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RViz2 最终优化启动脚本

优化布局：小左侧面板，大图像显示
"""

import sys
import subprocess
import time
from pathlib import Path

print("""
================================================================================
 Nova Carter RViz2 最终优化版
================================================================================

布局优化：
  ✓ 左侧面板缩小（30%宽度）
  ✓ 图像面板增大（640x400）
  ✓ 简化显示列表
  
显示内容：
  - 3D 点云（彩色，按高度编码）
  - 机器人轨迹（箭头）
  - RGB 相机（右下角，640x400）
  - 坐标系（TF）
  - 参考网格

操作说明：
  - 拖动分割线：调整面板大小
  - 右键点击分割线：隐藏/显示面板
  
如果图像未显示，请：
  1. 在 RViz2 菜单点击 "Panels"
  2. 勾选 "RGB Camera"
""")

# RViz2 配置文件
rviz_config = Path(__file__).parent / "config/rviz2/nova_carter_final.rviz"

print(f"\n✓ 使用配置文件: {rviz_config}")

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
print("\n等待 RViz2 加载...")
time.sleep(3)

# 布局调整提示
print("\n" + "="*70)
print("布局调整提示")
print("="*70)
print("\n如果需要调整面板大小：")
print("  - 找到左侧面板和中间视图之间的分割线")
print("  - 鼠标左键拖动分割线来调整宽度")
print("  - 建议左侧面板宽度：200-300像素")
print("\n如果需要调整图像大小：")
print("  - 找到图像面板周围的分割线")
print("  - 拖动分割线调整图像区域大小")
print("\n隐藏/显示面板：")
print("  - 右键点击分割线")
print("  - 选择 'Hide Left Dock' 或 'Hide Right Dock'")

# 启动监控
print("\n" + "="*70)
print("启动传感器监控...")
print("="*70)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry

if not rclpy.ok():
    rclpy.init()

monitor = rclpy.create_node('rviz_monitor')

stats = {
    'pointcloud': 0,
    'odom': 0,
    'image': 0,
    'start_time': time.time()
}

def pointcloud_callback(msg):
    stats['pointcloud'] += 1

def odom_callback(msg):
    stats['odom'] += 1

def image_callback(msg):
    stats['image'] += 1
    if stats['image'] % 30 == 0:
        print(f"[相机] 已接收 {stats['image']} 帧 | 分辨率: {msg.width}x{msg.height}")

try:
    pc_sub = monitor.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points', pointcloud_callback, 10)
    odom_sub = monitor.create_subscription(Odometry, '/chassis/odom', odom_callback, 10)
    img_sub = monitor.create_subscription(Image, '/front_stereo_camera/left/image_raw', image_callback, 10)
    
    print("✓ 监控已启动")
    print("\n实时数据流：")
    print("  点云 | 里程计 | 相机\n")
    
except Exception as e:
    print(f"✗ 订阅失败: {e}")
    rviz_process.terminate()
    sys.exit(1)

try:
    last_update = time.time()
    
    while True:
        rclpy.spin_once(monitor, timeout_sec=0.05)
        
        # 每3秒更新一次统计
        if time.time() - last_update > 3.0:
            elapsed = time.time() - stats['start_time']
            if elapsed > 0:
                rate_pc = stats['pointcloud'] / elapsed
                rate_odom = stats['odom'] / elapsed
                rate_img = stats['image'] / elapsed
                
                print(f"\r[统计] 点云:{stats['pointcloud']:4d} ({rate_pc:4.1f}Hz) | "
                      f"里程计:{stats['odom']:4d} ({rate_odom:4.1f}Hz) | "
                      f"相机:{stats['image']:4d} ({rate_img:4.1f}Hz) | RViz:{rviz_process.pid}", 
                      end='', flush=True)
            
            last_update = time.time()
        
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n\n停止监控...")

monitor.destroy_node()
rclpy.shutdown()

print("\n停止 RViz2...")
rviz_process.terminate()

try:
    rviz_process.wait(timeout=5)
except subprocess.TimeoutExpired:
    print("  强制终止...")
    rviz_process.kill()

print("\n再见！")






