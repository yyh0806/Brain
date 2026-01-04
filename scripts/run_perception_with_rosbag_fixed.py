#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实感知层处理脚本 - 从rosbag读取真实数据并发布地图

该脚本将：
1. 订阅rosbag中的真实传感器数据
2. 处理点云、里程计等数据
3. 构建占据栅格地图
4. 发布地图到/brain/map
"""

import os
import sys

# 设置ROS域ID为42
os.environ['ROS_DOMAIN_ID'] = '42'

import asyncio
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from std_msgs.msg import Header
import numpy as np
from datetime import datetime
from loguru import logger


class RealPerceptionProcessor(Node):
    """真实感知处理器"""
    
    def __init__(self):
        super().__init__('real_perception_processor')
        
        # 确认ROS域ID
        domain_id = os.environ.get('ROS_DOMAIN_ID', '0')
        logger.info(f'✓ ROS_DOMAIN_ID = {domain_id}')
        
        # 状态
        self.odom_data = None
        self.pointcloud_data = None
        
        # 地图参数
        self.map_resolution = 0.1  # 0.1米/栅格
        self.map_size = 50.0  # 50米 x 50米
        self.grid_size = int(self.map_size / self.map_resolution)  # 500x500
        self.map_origin_x = -25.0
        self.map_origin_y = -25.0
        
        # 初始化地图（全为未知）
        self.occupancy_map = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # 发布者
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/brain/map',
            10
        )
        
        # 订阅者（使用简单的QoS depth参数）
        self.odom_sub = self.create_subscription(
            Odometry,
            '/chassis/odom',
            10,
            self.odom_callback
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            10,
            self.pointcloud_callback
        )
        
        # 定时发布地图（1Hz）
        self.map_timer = self.create_timer(1.0, self.publish_map)
        self.publish_count = 0
        
        logger.info('真实感知处理器已启动')
        logger.info(f'地图大小: {self.map_size}m x {self.map_size}m')
        logger.info(f'分辨率: {self.map_resolution}m/栅格')
        logger.info(f'栅格数: {self.grid_size} x {self.grid_size}')
        logger.info(f'订阅话题: /chassis/odom, /front_3d_lidar/lidar_points')
        logger.info(f'发布话题: /brain/map')
    
    def odom_callback(self, msg: Odometry):
        """里程计回调"""
        self.odom_data = msg
        # 每10次输出一次日志，避免日志过多
        if self.publish_count % 10 == 0:
            logger.debug(f'接收里程计: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """点云回调 - 处理并更新地图"""
        self.pointcloud_data = msg
        
        # 解析点云数据
        points = self.parse_pointcloud(msg)
        
        if points is not None and len(points) > 0:
            # 更新占据地图
            self.update_map_from_pointcloud(points)
            
            # 每10次输出一次日志
            if self.publish_count % 10 == 0:
                logger.debug(f'接收点云: {len(points)} 个点')
    
    def parse_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        """解析点云数据"""
        try:
            # 获取点云字段
            fields = {}
            for field in msg.fields:
                fields[field.name] = field
            
            # 检查是否有xyz字段
            if 'x' not in fields or 'y' not in fields or 'z' not in fields:
                logger.warning('点云数据中缺少xyz字段')
                return None
            
            # 解析点云
            cloud_points = []
            
            # PointCloud2数据格式
            data = msg.data
            point_step = msg.point_step
            
            # 遍历每个点
            for i in range(0, len(data), point_step):
                # 提取xyz坐标
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
        """从点云更新占据地图"""
        if self.odom_data is None:
            logger.warning('里程计数据不可用，无法更新地图')
            return
        
        # 获取机器人位姿
        robot_x = self.odom_data.pose.pose.position.x
        robot_y = self.odom_data.pose.pose.position.y
        robot_z = self.odom_data.pose.pose.position.z
        
        # 将机器人位置转换为地图栅格坐标
        robot_grid_x = int((robot_x - self.map_origin_x) / self.map_resolution)
        robot_grid_y = int((robot_y - self.map_origin_y) / self.map_resolution)
        
        # 在机器人位置周围的障碍物点标记为占据
        # 简化实现：将点云投影到2D平面，高z值的点标记为障碍物
        occupied_count = 0
        
        for point in points:
            px = point[0]  # x
            py = point[1]  # y
            pz = point[2]  # z
            
            # 只考虑地面的点（高度在0.2m到2.0m之间）
            if 0.2 < pz < 2.0:
                # 转换为地图栅格坐标
                grid_x = int((px - self.map_origin_x) / self.map_resolution)
                grid_y = int((py - self.map_origin_y) / self.map_resolution)
                
                # 检查是否在地图范围内
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # 标记为占据
                    self.occupancy_map[grid_y, grid_x] = 100  # 100 = 占据
                    occupied_count += 1
        
        # 使用射线填充算法更新自由空间
        # 从机器人位置向障碍物方向填充自由空间
        self.update_free_space_raycast(robot_grid_x, robot_grid_y)
        
        if occupied_count > 0:
            logger.info(f'更新地图: 标记了 {occupied_count} 个占据栅格')
    
    def update_free_space_raycast(self, robot_grid_x, robot_grid_y):
        """射线填充算法：从机器人位置向障碍物填充自由空间"""
        # 简化实现：在机器人周围一定范围内，未标记为占据的设为自由
        radius = 50  # 5米范围
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = robot_grid_x + dx
                ny = robot_grid_y + dy
                
                # 检查边界
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # 计算到机器人的距离
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # 如果在范围内且不是占据，设为自由
                    if dist <= radius and self.occupancy_map[ny, nx] == -1:
                        # 检查到机器人之间是否有障碍物
                        has_obstacle = False
                        for tx in range(0, dx):
                            ty = int(dy * abs(tx) / abs(dx)) if dx != 0 else robot_grid_y + dy
                            if 0 <= nx + tx < self.grid_size and 0 <= ty < self.grid_size:
                                if self.occupancy_map[ty, nx + tx] == 100:
                                    has_obstacle = True
                                    break
                            if has_obstacle:
                                break
                        
                        # 如果无障碍物，设为自由
                        if not has_obstacle:
                            self.occupancy_map[ny, nx] = 0  # 0 = 自由
    
    def publish_map(self):
        """发布占据栅格地图"""
        self.publish_count += 1
        
        # 创建地图消息
        map_msg = OccupancyGrid()
        
        # Header
        map_msg.header = Header()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'odom'
        
        # 地图元数据
        map_msg.info = MapMetaData()
        map_msg.info.resolution = float(self.map_resolution)
        map_msg.info.width = self.grid_size
        map_msg.info.height = self.grid_size
        map_msg.info.origin.position.x = float(self.map_origin_x)
        map_msg.info.origin.position.y = float(self.map_origin_y)
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0
        
        # 地图数据
        map_msg.data = self.occupancy_map.flatten().tolist()
        
        # 发布
        self.map_publisher.publish(map_msg)
        
        # 每5次输出统计信息
        if self.publish_count % 5 == 0:
            occupied = np.sum(self.occupancy_map == 100)
            free = np.sum(self.occupancy_map == 0)
            unknown = np.sum(self.occupancy_map == -1)
            logger.info(f'发布地图 {self.publish_count} 次: 占据={occupied}, 自由={free}, 未知={unknown}')


def main():
    """主函数"""
    rclpy.init()
    
    print("""
================================================================================
真实感知层处理器 - 处理rosbag中的真实数据
================================================================================

此脚本将：
  1. 订阅rosbag中的真实传感器数据
  2. 处理点云和里程计数据
  3. 构建占据栅格地图
  4. 发布地图到/brain/map

订阅话题：
  - /chassis/odom (里程计）
  - /front_3d_lidar/lidar_points (3D点云）

发布话题：
  - /brain/map (占据栅格地图）

使用方法：
  1. 确保rosbag已播放：
     ros2 bag play <rosbag文件>  
  2. 在另一个终端运行此脚本：
     python3 scripts/run_perception_with_rosbag_fixed.py
  
  3. 启动RViz：
     bash start_rviz2.sh

  4. 在RViz中验证地图显示

按 Ctrl+C 停止
================================================================================
""")
    
    processor = RealPerceptionProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        print('\\n停止感知处理器...')
    finally:
        processor.destroy_node()
        rclpy.shutdown()
        print('已退出')


if __name__ == '__main__':
    main()



