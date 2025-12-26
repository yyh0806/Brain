#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟传感器数据生成器

生成模拟的传感器数据,用于在RViz中查看世界模型:
- 3D点云
- 里程计
- RGB图像
- 占据地图构建
"""

import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, Pose, Point, Quaternion
from std_msgs.msg import Header
import struct
import math
import time

class SensorSimulator(Node):
    """传感器模拟器"""
    
    def __init__(self):
        super().__init__('sensor_simulator')
        
        # QoS配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST(10)
        )
        
        # 创建发布者
        self.pc_pub = self.create_publisher(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            qos
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            '/chassis/odom',
            qos
        )
        
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            qos
        )
        
        # 模拟状态
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.map_resolution = 0.1  # 0.1米/格
        self.map_size = 200  # 200x200格
        self.map_origin_x = -10.0
        self.map_origin_y = -10.0
        
        # 创建占据地图
        self.map_width = self.map_size
        self.map_height = self.map_size
        self.occupancy_map = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        
        # 添加一些静态障碍物
        self._add_static_obstacles()
        
        # 定时器
        self.timer = self.create_timer(0.1, self.update)  # 10Hz
        
        # 计数器
        self.count = 0
        
        self.get_logger().info('传感器模拟器已启动')
    
    def _add_static_obstacles(self):
        """添加静态障碍物到地图"""
        # 添加几个障碍物块
        
        # 障碍物1: 左边的墙
        for i in range(50, 70):
            for j in range(20, 30):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        # 障碍物2: 前方的墙
        for i in range(30, 40):
            for j in range(60, 100):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        # 障碍物3: 右边的柱子
        for i in range(70, 90):
            for j in range(130, 140):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        # 障碍物4: 中央的障碍物
        for i in range(50, 70):
            for j in range(80, 90):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        self.get_logger().info('静态障碍物已添加到地图')
    
    def update(self):
        """更新传感器数据"""
        self.count += 1
        
        # 1. 模拟机器人运动(圆形轨迹)
        t = self.count * 0.1
        radius = 5.0
        omega = 0.2  # 角速度
        
        self.robot_x = radius * math.cos(omega * t)
        self.robot_y = radius * math.sin(omega * t)
        self.robot_yaw = omega * t
        
        # 2. 更新占据地图(标记机器人周围的自由空间)
        self._update_free_space()
        
        # 3. 发布点云
        self._publish_pointcloud()
        
        # 4. 发布里程计
        self._publish_odometry()
        
        # 5. 发布占据地图(每5次更新一次)
        if self.count % 5 == 0:
            self._publish_occupancy_map()
        
        if self.count % 50 == 0:
            self.get_logger().info(f'更新次数: {self.count}, 机器人位置: ({self.robot_x:.2f}, {self.robot_y:.2f})')
    
    def _update_free_space(self):
        """更新自由空间"""
        # 将机器人周围标记为自由空间
        robot_grid_x = int((self.robot_x - self.map_origin_x) / self.map_resolution)
        robot_grid_y = int((self.robot_y - self.map_origin_y) / self.map_resolution)
        
        # 标记3米半径内的区域为自由
        free_radius = int(3.0 / self.map_resolution)
        
        for dy in range(-free_radius, free_radius + 1):
            for dx in range(-free_radius, free_radius + 1):
                gx = robot_grid_x + dx
                gy = robot_grid_y + dy
                
                if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                    # 检查是否在圆内
                    if dx*dx + dy*dy <= free_radius*free_radius:
                        # 如果不是占据区域,标记为自由
                        if self.occupancy_map[gy, gx] == -1:
                            self.occupancy_map[gy, gx] = 0
    
    def _publish_pointcloud(self):
        """发布点云数据"""
        # 生成模拟点云
        num_points = 1000
        points = []
        
        # 在机器人前方生成点
        for i in range(num_points):
            # 极坐标
            distance = np.random.uniform(1.0, 10.0)
            angle = np.random.uniform(-math.pi/3, math.pi/3)  # 前方60度范围
            
            # 转换到机器人坐标系
            local_x = distance * math.cos(angle)
            local_y = distance * math.sin(angle)
            local_z = np.random.uniform(0.0, 0.5)
            
            # 转换到世界坐标系
            world_x = self.robot_x + local_x * math.cos(self.robot_yaw) - local_y * math.sin(self.robot_yaw)
            world_y = self.robot_y + local_x * math.sin(self.robot_yaw) + local_y * math.cos(self.robot_yaw)
            
            # 检查是否在障碍物上(模拟障碍物表面)
            grid_x = int((world_x - self.map_origin_x) / self.map_resolution)
            grid_y = int((world_y - self.map_origin_y) / self.map_resolution)
            
            if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
                if self.occupancy_map[grid_y, grid_x] == 100:
                    # 在障碍物上,添加到点云
                    points.append([world_x, world_y, local_z])
        
        # 添加一些随机点(模拟噪声)
        for _ in range(100):
            dx = np.random.uniform(-5, 5)
            dy = np.random.uniform(-5, 5)
            dz = np.random.uniform(0, 1)
            points.append([self.robot_x + dx, self.robot_y + dy, dz])
        
        # 创建点云消息
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.point_step = 16  # 4 * 4 bytes (x, y, z, padding)
        msg.row_step = msg.point_step * msg.width
        
        # 创建字段
        msg.fields = [
            PointField(name="x", offset=0, datatype=7, count=1),  # FLOAT32
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
        ]
        
        # 转换点云为字节数组
        pc_array = np.zeros((len(points), 4), dtype=np.float32)
        for i, point in enumerate(points):
            pc_array[i, 0] = point[0]
            pc_array[i, 1] = point[1]
            pc_array[i, 2] = point[2]
            pc_array[i, 3] = 0.0  # padding
        
        msg.data = pc_array.tobytes()
        
        self.pc_pub.publish(msg)
    
    def _publish_odometry(self):
        """发布里程计"""
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        
        # 位姿
        msg.pose.pose.position = Point(x=self.robot_x, y=self.robot_y, z=0.0)
        msg.pose.pose.orientation = self._yaw_to_quaternion(self.robot_yaw)
        
        # 速度
        v = 0.5  # 线速度
        omega = 0.2  # 角速度
        msg.twist.twist.linear.x = v * math.cos(self.robot_yaw)
        msg.twist.twist.linear.y = v * math.sin(self.robot_yaw)
        msg.twist.twist.angular.z = omega
        
        self.odom_pub.publish(msg)
    
    def _publish_occupancy_map(self):
        """发布占据地图"""
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        # 地图信息
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # 地图数据
        msg.data = self.occupancy_map.flatten().tolist()
        
        self.map_pub.publish(msg)
    
    def _yaw_to_quaternion(self, yaw):
        """将偏航角转换为四元数"""
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        quaternion.x = 0.0
        quaternion.y = 0.0
        return quaternion


def main():
    """主函数"""
    rclpy.init()
    
    print("""
================================================================================
传感器模拟器
================================================================================

此脚本将生成模拟的传感器数据:
  - 3D点云
  - 里程计
  - 占据地图

机器人将沿着圆形轨迹移动,并:
  1. 持续标记周围区域为自由空间
  2. 发布障碍物表面的点云
  3. 更新占据地图

在RViz中观察:
  - OccupancyGrid: 黑色=障碍物, 白色=自由空间, 灰色=未知
  - PointCloud2: 传感器点云(红色)
  - Odometry: 机器人位姿和轨迹

按 Ctrl+C 停止
================================================================================
""")
    
    simulator = SensorSimulator()
    
    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        print('\n停止模拟器...')
    finally:
        simulator.destroy_node()
        rclpy.shutdown()
        print('模拟器已关闭')


if __name__ == '__main__':
    main()

