#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试RViz能否接收数据"""

import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
import time
import json


class RVizReceptionTest(Node):
    """RViz数据接收测试"""

    def __init__(self):
        super().__init__('rviz_reception_test')
        
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 订阅所有话题
        self.sub_semantic_grid = self.create_subscription(
            OccupancyGrid,
            '/world_model/semantic_grid',
            self.grid_callback,
            qos_profile
        )
        
        self.sub_semantic_markers = self.create_subscription(
            MarkerArray,
            '/world_model/semantic_markers',
            self.markers_callback,
            qos_profile
        )
        
        self.sub_trajectory = self.create_subscription(
            Path,
            '/world_model/trajectory',
            self.traj_callback,
            qos_profile
        )
        
        self.sub_frontiers = self.create_subscription(
            MarkerArray,
            '/world_model/frontiers',
            self.frontiers_callback,
            qos_profile
        )
        
        self.sub_belief = self.create_subscription(
            MarkerArray,
            '/world_model/belief_markers',
            self.belief_callback,
            qos_profile
        )
        
        self.sub_changes = self.create_subscription(
            MarkerArray,
            '/world_model/change_events',
            self.changes_callback,
            qos_profile
        )
        
        self.sub_vlm = self.create_subscription(
            MarkerArray,
            '/vlm/detections',
            self.vlm_callback,
            qos_profile
        )
        
        # 计数器
        self.counts = {
            'semantic_grid': 0,
            'semantic_markers': 0,
            'trajectory': 0,
            'frontiers': 0,
            'belief_markers': 0,
            'change_events': 0,
            'vlm_detections': 0
        }
        
        # 打印摘要的定时器
        self.timer = self.create_timer(5.0, self.print_summary)
        
        self.get_logger().info("="*70)
        self.get_logger().info("RViz数据接收测试")
        self.get_logger().info("="*70)
        self.get_logger().info("监听话题:")
        for topic in ['/world_model/semantic_grid', '/world_model/semantic_markers', 
                    '/world_model/trajectory', '/world_model/frontiers',
                    '/world_model/belief_markers', '/world_model/change_events',
                    '/vlm/detections']:
            self.get_logger().info(f"  - {topic}")
        self.get_logger().info("")
        self.get_logger().info("每5秒打印接收数据摘要")
        self.get_logger().info("="*70)
    
    def grid_callback(self, msg):
        self.counts['semantic_grid'] += 1
        if self.counts['semantic_grid'] <= 3:
            self.get_logger().info(f"📊 [第{self.counts['semantic_grid']}次] 收到semantic_grid: {msg.info.width}x{msg.info.height}")
    
    def markers_callback(self, msg):
        self.counts['semantic_markers'] += 1
        if self.counts['semantic_markers'] <= 3:
            self.get_logger().info(f"🏷️  [第{self.counts['semantic_markers']}次] 收到semantic_markers: {len(msg.markers)}个")
    
    def traj_callback(self, msg):
        self.counts['trajectory'] += 1
        if self.counts['trajectory'] <= 3:
            self.get_logger().info(f"🛤️  [第{self.counts['trajectory']}次] 收到trajectory: {len(msg.poses)}个位姿")
    
    def frontiers_callback(self, msg):
        self.counts['frontiers'] += 1
        if self.counts['frontiers'] <= 3:
            self.get_logger().info(f"🧭  [第{self.counts['frontiers']}次] 收到frontiers: {len(msg.markers)}个")
    
    def belief_callback(self, msg):
        self.counts['belief_markers'] += 1
        if self.counts['belief_markers'] <= 3:
            self.get_logger().info(f"💭  [第{self.counts['belief_markers']}次] 收到belief_markers: {len(msg.markers)}个")
    
    def changes_callback(self, msg):
        self.counts['change_events'] += 1
        if self.counts['change_events'] <= 3:
            self.get_logger().info(f"🔄  [第{self.counts['change_events']}次] 收到change_events: {len(msg.markers)}个")
    
    def vlm_callback(self, msg):
        self.counts['vlm_detections'] += 1
        if self.counts['vlm_detections'] <= 3:
            self.get_logger().info(f"👁️  [第{self.counts['vlm_detections']}次] 收到vlm_detections: {len(msg.markers)}个")
    
    def print_summary(self):
        """打印接收摘要"""
        self.get_logger().info("")
        self.get_logger().info("="*70)
        self.get_logger().info("📊 数据接收摘要（运行中）")
        self.get_logger().info("="*70)
        for topic, count in self.counts.items():
            status = "✅" if count > 0 else "❌"
            self.get_logger().info(f"{status} {topic}: {count}条消息")
        
        total = sum(self.counts.values())
        self.get_logger().info("")
        self.get_logger().info(f"总计: {total}条消息")
        self.get_logger().info("="*70)
        self.get_logger().info("")
        
        # 写入日志文件
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "rviz_reception_summary",
                "timestamp": int(time.time() * 1000),
                "location": "test_rviz_reception.py:print_summary",
                "message": "RViz数据接收摘要",
                "data": self.counts,
                "sessionId": "debug-session",
                "hypothesisId": "E,F,G,H"
            }) + "\n")


def main():
    rclpy.init()
    
    test_node = RVizReceptionTest()
    
    try:
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

