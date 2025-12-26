#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实感知层处理脚本 V2 - 从rosbag读取真实数据并发布地图
修复了QoS参数问题
"""

import os
import sys

# 设置ROS域ID为42
os.environ['ROS_DOMAIN_ID'] = '42'

import asyncio
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import numpy as np
from datetime import datetime
from loguru import logger


class FixedPerceptionProcessor(Node):
    """真实感知处理器 - V2: 修复了QoS问题"""
    
    def __init__(self):
        super().__init__('fixed_perception_processor')
        
        # 确认ROS域ID
        domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        logger.info(f'✓ ROS_DOMAIN_ID = {domain_id}')
        
        # 状态
        self.odom_data = None
        self.pointcloud_data = None
        
        # 地图参数
        self.map_resolution = 0.1
        self.map_size = 50.0
        self.grid_size = int(self.map_size / self.map_resolution)
        self.map_origin_x = -25.0
        self.map_origin_y = -25.0
        
        # 初始化地图
        self.occupancy_map = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # 发布者 - 使用QoSProfile对象
        map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/brain/map',
            qos_profile=map_qos
        )
        
        # 订阅者 - 使用QoSProfile对象
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/chassis/odom',
            qos_profile=odom_qos,
            self.odom_callback
        )
        
        pointcloud_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            qos_profile=pointcloud_qos,
            self.pointcloud_callback
        )
        
        self.map_timer = self.create_timer(1.0, self.publish_map)
        self.publish_count = 0
        
        logger.info('固定感知处理器V2已启动')
        logger.info(f'地图大小: {self.map_size}m x {self.map_size}m')
        logger.info(f'分辨率: {self.map_resolution}m/栅格')
        logger.info(f'栅格数: {self.grid_size} x {self.grid_size}')
        logger.info(f'订阅话题: /chassis/odom, /front_3d_lidar/lidar_points')
        logger.info(f'发布话题: /brain/map')
    
    def odom_callback(self, msg: Odometry):
        self.odom_data = msg
        if self.publish_count % 10 == 0:
            logger.debug(f'接收里程计: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})')
    
    def pointcloud_callback(self, msg: PointCloud2):
        self.pointcloud_data = msg
        points = self.parse_pointcloud(msg)
        
        if points is not None and len(points) > 0:
            self.update_map_from_pointcloud(points)
            
            if self.publish_count % 10 == 0:
                logger.debug(f'接收点云: {len(points)} 个点')
    
    def parse_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        try:
            fields = {}
            for field in msg.fields:
                fields[field.name] = field
            
            if 'x' not in fields or 'y' not in fields or 'z' not in fields:
                logger.warning('点云数据中缺少xyz字段')
                return None
            
            cloud_points = []
            data = msg.data
            point_step = msg.point_step
            
            for i in range(0, len(data), point_step):
                x_offset = fields['x'].offset
                y_offset = fields['y'].offset
                z_offset = fields['z'].offset
                
                x_bytes = data[i + x_offset : i + x_offset + 4]
                y_bytes = data[i + y_offset : i + y_offset + 4]
                z_bytes = data[i + z_offset : i + z_offset + 4]
                
                if len(x_bytes) < 4 or len(y_bytes) < 4 or len(z_bytes) < 4:
                    continue
                
                x = np.frombuffer(x_bytes, dtype=np.float32)[0]
                y = np.frombuffer(y_bytes, dtype=np.float32)[0]
                z = np.frombuffer(z_bytes, dtype=np.float32)[0]
                
                cloud_points.append([x, y, z])
            
            return np.array(cloud_points)
            
        except Exception as e:
            logger.error(f'解析点云失败: {e}')
            return None
    
    def update_map_from_pointcloud(self, points: np.ndarray):
        if self.odom_data is None:
            logger.warning('里程计数据不可用，无法更新地图')
            return
        
        robot_x = self.odom_data.pose.pose.position.x
        robot_y = self.odom_data.pose.pose.position.y
        robot_z = self.odom_data.pose.pose.position.z
        
        robot_grid_x = int((robot_x - self.map_origin_x) / self.map_resolution)
        robot_grid_y = int((robot_y - self.map_origin_y) / self.map_resolution)
        
        occupied_count = 0
        
        for point in points:
            px = point[0]
            py = point[1]
            pz = point[2]
            
            if 0.2 < pz < 2.0:
                grid_x = int((px - self.map_origin_x) / self.map_resolution)
                grid_y = int((py - self.map_origin_y) / self.map_resolution)
                
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    if self.occupancy_map[grid_y, grid_x] == -1:
                        self.occupancy_map[grid_y, grid_x] = 100
                        occupied_count += 1
        
        self.update_free_space_raycast(robot_grid_x, robot_grid_y)
        
        if occupied_count > 0:
            logger.info(f'更新地图: 标记了 {occupied_count} 个占据栅格')
    
    def update_free_space_raycast(self, robot_grid_x, robot_grid_y):
        radius = 50
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = robot_grid_x + dx
                ny = robot_grid_y + dy
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist <= radius and self.occupancy_map[ny, nx] == -1:
                        has_obstacle = False
                        for tx in range(0, dx):
                            ty = int(dy * abs(tx) / abs(dx)) if dx != 0 else robot_grid_y + dy
                            if 0 <= nx + tx < self.grid_size and 0 <= ty < self.grid_size:
                                if self.occupancy_map[ty, nx + tx] == 100:
                                    has_obstacle = True
                                    break
                            if has_obstacle:
                                break
                        
                        if not has_obstacle:
                            self.occupancy_map[ny, nx] = 0
    
    def publish_map(self):
        self.publish_count += 1
        
        map_msg = OccupancyGrid()
        
        map_msg.header = Header()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'odom'
        
        map_msg.info = MapMetaData()
        map_msg.info.resolution = float(self.map_resolution)
        map_msg.info.width = self.grid_size
        map_msg.info.height = self.grid_size
        map_msg.info.origin.position.x = float(self.map_origin_x)
        map_msg.info.origin.position.y = float(self.map_origin_y)
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        map_msg.data = self.occupancy_map.flatten().tolist()
        
        self.map_publisher.publish(map_msg)
        
        if self.publish_count % 5 == 0:
            occupied = np.sum(self.occupancy_map == 100)
            free = np.sum(self.occupancy_map == 0)
            unknown = np.sum(self.occupancy_map == -1)
            logger.info(f'发布地图 {self.publish_count} 次: 占据={occupied}, 自由={free}, 未知={unknown}')


def main():
    rclpy.init()
    
    print("""
================================================================================
真实感知层处理器 V2 - 修复了QoS参数问题
================================================================================

此脚本将：
  1. 订阅rosbag中的真实传感器数据
  2. 处理点云、里程计等数据
  3. 构建占据栅格地图
  4. 发布地图到/brain/map

订阅话题：
  - /chassis/odom (里程计）
  - /front_3d_lidar/lidar_points (3D点云）

发布话题：
  - /brain/map (占据栅格地图）

使用方法：
  1. 确保rosbag已播放
  2. 在另一个终端运行此脚本：
     python3 scripts/perception_with_rosbag_v2.py
  3. 启动RViz：
     bash start_rviz2.sh

按 Ctrl+C 停止
================================================================================
""")
    
    processor = FixedPerceptionProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        print('\n停止感知处理器...')
    finally:
        processor.destroy_node()
        rclpy.shutdown()
        print('已退出')


if __name__ == '__main__':
    main()

