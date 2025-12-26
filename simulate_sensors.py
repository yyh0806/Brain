#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Data Simulator

Generate simulated sensor data for viewing in RViz:
- 3D Point Cloud
- Odometry
- Occupancy Grid
"""

import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Header
import math


class SensorSimulator(Node):
    """Sensor Simulator Node"""
    
    def __init__(self):
        super().__init__('sensor_simulator')
        
        # QoS Configuration
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST(10)
        )
        
        # Create publishers
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
        
        # Simulation state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.map_resolution = 0.1  # 0.1m per cell
        self.map_size = 200  # 200x200 cells
        self.map_origin_x = -10.0
        self.map_origin_y = -10.0
        
        # Create occupancy map
        self.map_width = self.map_size
        self.map_height = self.map_size
        self.occupancy_map = np.full((self.map_height, self.map_width), -1, dtype=np.int8)
        
        # Add some static obstacles
        self._add_static_obstacles()
        
        # Timer
        self.timer = self.create_timer(0.1, self.update)  # 10Hz
        
        # Counter
        self.count = 0
        
        self.get_logger().info('Sensor simulator started')
    
    def _add_static_obstacles(self):
        """Add static obstacles to map"""
        # Obstacle 1: Left wall
        for i in range(50, 70):
            for j in range(20, 30):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        # Obstacle 2: Front wall
        for i in range(30, 40):
            for j in range(60, 100):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        # Obstacle 3: Right pillar
        for i in range(70, 90):
            for j in range(130, 140):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        # Obstacle 4: Central obstacle
        for i in range(50, 70):
            for j in range(80, 90):
                if 0 <= i < self.map_height and 0 <= j < self.map_width:
                    self.occupancy_map[i, j] = 100
        
        self.get_logger().info('Static obstacles added to map')
    
    def update(self):
        """Update sensor data"""
        self.count += 1
        
        # 1. Simulate robot motion (circular trajectory)
        t = self.count * 0.1
        radius = 5.0
        omega = 0.2  # angular velocity
        
        self.robot_x = radius * math.cos(omega * t)
        self.robot_y = radius * math.sin(omega * t)
        self.robot_yaw = omega * t
        
        # 2. Update occupancy map (mark free space around robot)
        self._update_free_space()
        
        # 3. Publish point cloud
        self._publish_pointcloud()
        
        # 4. Publish odometry
        self._publish_odometry()
        
        # 5. Publish occupancy map (every 5 updates)
        if self.count % 5 == 0:
            self._publish_occupancy_map()
        
        if self.count % 50 == 0:
            self.get_logger().info(f'Update count: {self.count}, Robot position: ({self.robot_x:.2f}, {self.robot_y:.2f})')
    
    def _update_free_space(self):
        """Update free space"""
        # Mark area around robot as free space
        robot_grid_x = int((self.robot_x - self.map_origin_x) / self.map_resolution)
        robot_grid_y = int((self.robot_y - self.map_origin_y) / self.map_resolution)
        
        # Mark 3m radius area as free
        free_radius = int(3.0 / self.map_resolution)
        
        for dy in range(-free_radius, free_radius + 1):
            for dx in range(-free_radius, free_radius + 1):
                gx = robot_grid_x + dx
                gy = robot_grid_y + dy
                
                if 0 <= gx < self.map_width and 0 <= gy < self.map_height:
                    # Check if inside circle
                    if dx*dx + dy*dy <= free_radius*free_radius:
                        # If not occupied, mark as free
                        if self.occupancy_map[gy, gx] == -1:
                            self.occupancy_map[gy, gx] = 0
    
    def _publish_pointcloud(self):
        """Publish point cloud data"""
        # Generate simulated point cloud
        num_points = 1000
        points = []
        
        # Generate points in front of robot
        for i in range(num_points):
            # Polar coordinates
            distance = np.random.uniform(1.0, 10.0)
            angle = np.random.uniform(-math.pi/3, math.pi/3)  # Front 60 degrees
            
            # Convert to robot frame
            local_x = distance * math.cos(angle)
            local_y = distance * math.sin(angle)
            local_z = np.random.uniform(0.0, 0.5)
            
            # Convert to world frame
            world_x = self.robot_x + local_x * math.cos(self.robot_yaw) - local_y * math.sin(self.robot_yaw)
            world_y = self.robot_y + local_x * math.sin(self.robot_yaw) + local_y * math.cos(self.robot_yaw)
            
            # Check if on obstacle (simulate obstacle surface)
            grid_x = int((world_x - self.map_origin_x) / self.map_resolution)
            grid_y = int((world_y - self.map_origin_y) / self.map_resolution)
            
            if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
                if self.occupancy_map[grid_y, grid_x] == 100:
                    # On obstacle, add to point cloud
                    points.append([world_x, world_y, local_z])
        
        # Add some random points (simulate noise)
        for _ in range(100):
            dx = np.random.uniform(-5, 5)
            dy = np.random.uniform(-5, 5)
            dz = np.random.uniform(0, 1)
            points.append([self.robot_x + dx, self.robot_y + dy, dz])
        
        # Create point cloud message
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.point_step = 16  # 4 * 4 bytes (x, y, z, padding)
        msg.row_step = msg.point_step * msg.width
        
        # Create fields
        msg.fields = [
            PointField(name="x", offset=0, datatype=7, count=1),  # FLOAT32
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
        ]
        
        # Convert point cloud to bytes
        pc_array = np.zeros((len(points), 4), dtype=np.float32)
        for i, point in enumerate(points):
            pc_array[i, 0] = point[0]
            pc_array[i, 1] = point[1]
            pc_array[i, 2] = point[2]
            pc_array[i, 3] = 0.0  # padding
        
        msg.data = pc_array.tobytes()
        
        self.pc_pub.publish(msg)
    
    def _publish_odometry(self):
        """Publish odometry"""
        msg = Odometry()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        
        # Pose
        msg.pose.pose.position = Point(x=self.robot_x, y=self.robot_y, z=0.0)
        msg.pose.pose.orientation = self._yaw_to_quaternion(self.robot_yaw)
        
        # Velocity
        v = 0.5  # linear velocity
        omega = 0.2  # angular velocity
        msg.twist.twist.linear.x = v * math.cos(self.robot_yaw)
        msg.twist.twist.linear.y = v * math.sin(self.robot_yaw)
        msg.twist.twist.angular.z = omega
        
        self.odom_pub.publish(msg)
    
    def _publish_occupancy_map(self):
        """Publish occupancy map"""
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        # Map info
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Map data
        msg.data = self.occupancy_map.flatten().tolist()
        
        self.map_pub.publish(msg)
    
    def _yaw_to_quaternion(self, yaw):
        """Convert yaw to quaternion"""
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)
        quaternion.x = 0.0
        quaternion.y = 0.0
        return quaternion


def main():
    """Main function"""
    rclpy.init()
    
    print("""
================================================================================
Sensor Data Simulator
================================================================================

This script will generate simulated sensor data:
  - 3D Point Cloud
  - Odometry
  - Occupancy Grid

The robot will move along a circular trajectory and:
  1. Continuously mark surrounding area as free space
  2. Publish point cloud on obstacle surfaces
  3. Update occupancy map

View in RViz:
  - Occupancy Grid: Black=Obstacles, White=Free Space, Gray=Unknown
  - PointCloud2: Sensor point cloud (red)
  - Odometry: Robot pose and trajectory

Press Ctrl+C to stop
================================================================================
""")
    
    simulator = SensorSimulator()
    
    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        print('\nStopping simulator...')
    finally:
        simulator.destroy_node()
        rclpy.shutdown()
        print('Simulator shut down')


if __name__ == '__main__':
    main()

