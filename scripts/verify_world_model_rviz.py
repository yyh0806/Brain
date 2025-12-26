#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
世界模型RViz验证助手

该脚本将在终端中显示关键的世界模型统计信息，
辅助你在RViz中验证世界模型的正确性。
"""

import os
import sys

# 设置ROS域ID为42（与rviz2.sh一致）
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
import numpy as np
from datetime import datetime
import time

class WorldModelMonitor(Node):
    def __init__(self):
        super().__init__('world_model_monitor')
        self.get_logger().info(f'ROS_DOMAIN_ID = 42')
        
        # 统计数据
        self.map_stats = {
            'occupied': 0,
            'free': 0,
            'unknown': 0,
            'total_cells': 0,
            'map_width': 0,
            'map_height': 0,
            'resolution': 0.0,
            'update_count': 0,
            'last_update': None
        }
        
        self.robot_pose = None
        self.last_pose_time = None
        self.pointcloud_count = 0
        
        # 订阅话题
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',  # 或其他位姿话题
            self.pose_callback,
            10
        )
        
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            self.pc_callback,
            10
        )
        
        # 订阅odometry获取机器人位姿
        from nav_msgs.msg import Odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )
        
        self.start_time = time.time()
        self.get_logger().info('世界模型监控器已启动')
    
    def map_callback(self, msg):
        """处理占据栅格地图"""
        self.map_stats['update_count'] += 1
        self.map_stats['last_update'] = datetime.now()
        
        # 更新地图元数据
        self.map_stats['map_width'] = msg.info.width
        self.map_stats['map_height'] = msg.info.height
        self.map_stats['resolution'] = msg.info.resolution
        self.map_stats['total_cells'] = len(msg.data)
        
        # 统计各种状态
        data = np.array(msg.data)
        self.map_stats['occupied'] = np.sum((data >= 70) & (data <= 100))  # 占据
        self.map_stats['free'] = np.sum((data >= 0) & (data <= 30))  # 自由
        self.map_stats['unknown'] = np.sum(data == -1)  # 未知
        
        self.display_stats()
    
    def odom_callback(self, msg):
        """处理里程计数据"""
        self.robot_pose = msg.pose.pose
        self.last_pose_time = datetime.now()
    
    def pose_callback(self, msg):
        """处理位姿数据"""
        self.robot_pose = msg.pose
        self.last_pose_time = datetime.now()
    
    def pc_callback(self, msg):
        """处理点云数据"""
        self.pointcloud_count += 1
    
    def display_stats(self):
        """显示统计信息"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(f"世界模型验证 - 运行时间: {elapsed:.1f}秒")
        print("="*80)
        
        # 地图统计
        print(f"\n【占据栅格地图】")
        print(f"  地图尺寸: {self.map_stats['map_width']} x {self.map_stats['map_height']} 栅格")
        print(f"  分辨率: {self.map_stats['resolution']:.3f} 米/栅格")
        print(f"  总栅格数: {self.map_stats['total_cells']:,}")
        print(f"  更新次数: {self.map_stats['update_count']}")
        print(f"  最后更新: {self.map_stats['last_update']}")
        print(f"\n  栅格分布:")
        print(f"    ■ 占据 (障碍物): {self.map_stats['occupied']:6,} ({self.get_percentage(self.map_stats['occupied']):5.1f}%)")
        print(f"    □ 自由 (可通行): {self.map_stats['free']:6,} ({self.get_percentage(self.map_stats['free']):5.1f}%)")
        print(f"    ? 未知 (未探索): {self.map_stats['unknown']:6,} ({self.get_percentage(self.map_stats['unknown']):5.1f}%)")
        
        # 持久化验证
        print(f"\n【持久化验证】")
        if self.map_stats['occupied'] > 0:
            print(f"  ✓ 占据区域存在 ({self.map_stats['occupied']} 个栅格)")
        else:
            print(f"  ✗ 未检测到占据区域")
        
        if self.map_stats['free'] > 0:
            print(f"  ✓ 自由空间存在 ({self.map_stats['free']} 个栅格)")
        else:
            print(f"  ✗ 未检测到自由空间")
        
        unknown_ratio = self.get_percentage(self.map_stats['unknown'])
        if unknown_ratio < 50:
            print(f"  ✓ 已探索超过50% ({100-unknown_ratio:.1f}%)")
        elif unknown_ratio < 80:
            print(f"  ~ 已探索 {100-unknown_ratio:.1f}%")
        else:
            print(f"  ⚠ 大部分区域未知 ({unknown_ratio:.1f}%)")
        
        # 机器人位姿
        print(f"\n【机器人状态】")
        if self.robot_pose:
            x = self.robot_pose.position.x
            y = self.robot_pose.position.y
            z = self.robot_pose.position.z
            print(f"  位置: ({x:7.3f}, {y:7.3f}, {z:7.3f})")
            print(f"  最后更新: {self.last_pose_time}")
        else:
            print(f"  ⚠ 未接收到位姿数据")
        
        # 点云统计
        if self.pointcloud_count > 0:
            rate = self.pointcloud_count / elapsed
            print(f"\n【点云数据】")
            print(f"  接收数量: {self.pointcloud_count:,}")
            print(f"  频率: {rate:.1f} Hz")
        
        # 验证建议
        print(f"\n【验证建议】")
        print(f"  在RViz中观察Occupancy Grid面板：")
        print(f"    1. 障碍物（黑色区域）是否稳定存在")
        print(f"    2. 自由空间（白色区域）是否逐渐扩展")
        print(f"    3. 已探索区域是否保持已知状态（不变回灰色）")
        print(f"    4. 机器人移动轨迹是否清晰可见")
        print(f"\n  在RViz中调整Occupancy Grid透明度（Alpha值）：")
        print(f"    - 设置Alpha=0.7，可以更清楚地看到地图")
        print(f"    - 如果地图更新过快，可调整RViz的Frame Rate")
        
        print("="*80)
    
    def get_percentage(self, count):
        """计算百分比"""
        if self.map_stats['total_cells'] == 0:
            return 0.0
        return (count / self.map_stats['total_cells']) * 100


def main():
    rclpy.init()
    
    monitor = WorldModelMonitor()
    
    print("""
================================================================================
世界模型RViz验证助手
================================================================================

该脚本将：
  1. 监听 /map 话题，统计占据栅格地图状态
  2. 显示占据、自由、未知栅格的数量和比例
  3. 提供持久化验证建议
  4. 辅助你在RViz中观察世界模型的正确性

使用方法：
  1. 保持此脚本运行
  2. 在另一个终端启动RViz: bash scripts/start_rviz_perception.sh
  3. 播放rosbag: ros2 bag play <rosbag文件>
  4. 观察RViz中的Occupancy Grid面板
  5. 参考本脚本输出的统计信息进行验证

按 Ctrl+C 停止
================================================================================
""")
    
    try:
        # 初始等待
        print("等待数据...")
        time.sleep(2)
        
        while rclpy.ok():
            rclpy.spin_once(monitor, timeout_sec=1.0)
            
            # 每5秒更新一次显示
            if monitor.map_stats['update_count'] % 5 == 0:
                pass  # 已经在回调中更新显示
    
    except KeyboardInterrupt:
        print("\n\n停止监控...")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

