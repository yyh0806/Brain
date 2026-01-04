#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RViz地图测试脚本

发布模拟的占据栅格地图到 /brain/map 话题，
用于测试RViz配置是否正确。

运行此脚本后，RViz应该能看到一个模拟的占据地图。
"""

import os
import sys

# 设置ROS域ID为42（与rviz2.sh一致）
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import numpy as np
import time

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher_test')
        
        # 确认ROS域ID
        domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        self.get_logger().info(f'✓ ROS_DOMAIN_ID = {domain_id}')
        
        # 发布地图到 /brain/map 话题
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/brain/map',
            10
        )
        
        # 地图参数
        self.map_size = 50.0  # 50米 x 50米
        self.resolution = 0.1   # 0.1米/栅格
        self.grid_size = int(self.map_size / self.resolution)  # 500x500
        
        # 创建一个模拟地图
        self.create_test_map()
        
        # 定时发布
        self.timer = self.create_timer(1.0, self.publish_map)
        self.publish_count = 0
        
        self.get_logger().info(f'地图发布器已启动')
        self.get_logger().info(f'地图大小: {self.map_size}m x {self.map_size}m')
        self.get_logger().info(f'分辨率: {self.resolution}m/栅格')
        self.get_logger().info(f'栅格数: {self.grid_size} x {self.grid_size}')
        self.get_logger().info(f'发布话题: /brain/map')
    
    def create_test_map(self):
        """创建一个模拟的占据地图"""
        # 初始化全为未知 (-1)
        self.map_data = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # 添加一些障碍物 (100 = 占据)
        center = self.grid_size // 2
        
        # 1. 在中心创建一个矩形障碍物
        self.map_data[center-50:center+50, center-50:center+50] = 100
        
        # 2. 在左上角创建一个L形障碍物
        self.map_data[50:150, 50:150] = 100
        self.map_data[50:150, 150:250] = 100
        
        # 3. 在右下角创建一个圆形障碍物
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i - 350)**2 + (j - 350)**2 < 2500:  # 半径50
                    self.map_data[i, j] = 100
        
        # 4. 创建自由空间 (0 = 自由)
        # 从中心向外的射线
        for angle in np.linspace(0, 2*np.pi, 360):
            for r in range(1, 180):
                x = center + int(r * np.cos(angle))
                y = center + int(r * np.sin(angle))
                
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # 如果遇到障碍物，停止射线
                    if self.map_data[x, y] == 100:
                        break
                    # 标记为自由空间
                    self.map_data[x, y] = 0
        
        self.get_logger().info(f'模拟地图已创建')
        self.get_logger().info(f'占据栅格数: {np.sum(self.map_data == 100)}')
        self.get_logger().info(f'自由栅格数: {np.sum(self.map_data == 0)}')
        self.get_logger().info(f'未知栅格数: {np.sum(self.map_data == -1)}')
    
    def publish_map(self):
        """发布地图"""
        map_msg = OccupancyGrid()
        
        # Header
        map_msg.header = Header()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'odom'
        
        # 地图元数据
        map_msg.info = MapMetaData()
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.grid_size
        map_msg.info.height = self.grid_size
        map_msg.info.origin.position.x = -self.map_size / 2
        map_msg.info.origin.position.y = -self.map_size / 2
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # 地图数据
        map_msg.data = self.map_data.flatten().tolist()
        
        # 发布
        self.map_pub.publish(map_msg)
        self.publish_count += 1
        
        if self.publish_count % 5 == 0:
            self.get_logger().info(f'已发布地图 {self.publish_count} 次')


def main():
    # 检查ROS域ID
    domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
    print(f'✓ 环境变量 ROS_DOMAIN_ID = {domain_id}')
    
    rclpy.init()
    
    # 再次确认（rclpy初始化后可能不同）
    print(f'✓ rclpy初始化完成')
    
    print("""
================================================================================
RViz 地图测试脚本
================================================================================

此脚本将发布模拟的占据栅格地图到 /brain/map 话题。

在RViz中你应该能看到：
  - 中心一个矩形障碍物（黑色）
  - 左上角一个L形障碍物（黑色）
  - 右下角一个圆形障碍物（黑色）
  - 中心向外的自由空间射线（白色）
  - 其他区域为未知（灰色）

运行此脚本后：
  1. 在RViz中应该能看到Occupancy Grid显示
  2. 地图会每秒更新一次

按 Ctrl+C 停止
================================================================================
""")
    
    map_publisher = MapPublisher()
    
    try:
        rclpy.spin(map_publisher)
    except KeyboardInterrupt:
        print('\n停止地图发布...')
    finally:
        map_publisher.destroy_node()
        rclpy.shutdown()
        print('已退出')


if __name__ == '__main__':
    main()

