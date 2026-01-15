#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""简单的可视化测试脚本"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray

import time
import json


class VisualizationTest(Node):
    """可视化测试节点"""

    def __init__(self):
        super().__init__('visualization_test')
        
        # 创建订阅者
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 订阅所有可视化话题
        self.subscription_semantic_grid = self.create_subscription(
            OccupancyGrid,
            '/world_model/semantic_grid',
            self.semantic_grid_callback,
            qos_profile
        )
        
        self.subscription_semantic_markers = self.create_subscription(
            MarkerArray,
            '/world_model/semantic_markers',
            self.semantic_markers_callback,
            qos_profile
        )
        
        self.subscription_belief_markers = self.create_subscription(
            MarkerArray,
            '/world_model/belief_markers',
            self.belief_markers_callback,
            qos_profile
        )
        
        self.subscription_trajectory = self.create_subscription(
            Path,
            '/world_model/trajectory',
            self.trajectory_callback,
            qos_profile
        )
        
        self.subscription_frontiers = self.create_subscription(
            MarkerArray,
            '/world_model/frontiers',
            self.frontiers_callback,
            qos_profile
        )
        
        self.subscription_change_events = self.create_subscription(
            MarkerArray,
            '/world_model/change_events',
            self.change_events_callback,
            qos_profile
        )
        
        self.subscription_vlm_detections = self.create_subscription(
            MarkerArray,
            '/vlm/detections',
            self.vlm_detections_callback,
            qos_profile
        )
        
        self.get_logger().info("✅ 可视化测试节点已启动")
        self.get_logger().info("   监听话题:")
        self.get_logger().info("     - /world_model/semantic_grid")
        self.get_logger().info("     - /world_model/semantic_markers")
        self.get_logger().info("     - /world_model/belief_markers")
        self.get_logger().info("     - /world_model/trajectory")
        self.get_logger().info("     - /world_model/frontiers")
        self.get_logger().info("     - /world_model/change_events")
        self.get_logger().info("     - /vlm/detections")
    
    def semantic_grid_callback(self, msg):
        """语义占据网格回调"""
        self.get_logger().info(f"📊 收到semantic_grid: {msg.info.width}x{msg.info.height}")
    
    def semantic_markers_callback(self, msg):
        """语义标记回调"""
        self.get_logger().info(f"🏷️  收到semantic_markers: {len(msg.markers)}个标记")
    
    def belief_markers_callback(self, msg):
        """信念标记回调"""
        self.get_logger().info(f"💭 收到belief_markers: {len(msg.markers)}个信念标记")
    
    def trajectory_callback(self, msg):
        """轨迹回调"""
        self.get_logger().info(f"🛤️  收到trajectory: {len(msg.poses)}个位姿")
    
    def frontiers_callback(self, msg):
        """探索边界回调"""
        self.get_logger().info(f"🧭 收到frontiers: {len(msg.markers)}个探索边界")
    
    def change_events_callback(self, msg):
        """变化事件回调"""
        self.get_logger().info(f"🔄 收到change_events: {len(msg.markers)}个变化事件")
    
    def vlm_detections_callback(self, msg):
        """VLM检测回调"""
        self.get_logger().info(f"👁️  收到vlm_detections: {len(msg.markers)}个VLM标记")


def main(args=None):
    rclpy.init(args=args)
    
    test_node = VisualizationTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

