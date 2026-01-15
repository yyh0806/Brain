#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证WorldModel可视化话题数据
"""
import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
import time

class TopicVerifier(Node):
    def __init__(self):
        super().__init__('topic_verifier')
        self.counts = {
            'semantic_grid': 0,
            'semantic_markers': 0,
            'trajectory': 0,
            'frontiers': 0
        }

        # 订阅所有话题
        self.create_subscription(OccupancyGrid, '/world_model/semantic_grid',
                                self.grid_callback, 10)
        self.create_subscription(MarkerArray, '/world_model/semantic_markers',
                                self.markers_callback, 10)
        self.create_subscription(Path, '/world_model/trajectory',
                                self.trajectory_callback, 10)
        self.create_subscription(MarkerArray, '/world_model/frontiers',
                                self.frontiers_callback, 10)

        self.get_logger().info("话题验证器已启动，等待数据...")

    def grid_callback(self, msg):
        self.counts['semantic_grid'] += 1
        if self.counts['semantic_grid'] == 1:
            self.get_logger().info(f"✅ semantic_grid: {msg.info.width}x{msg.info.height}, "
                                  f"resolution: {msg.info.resolution}, "
                                  f"data length: {len(msg.data)}")

    def markers_callback(self, msg):
        self.counts['semantic_markers'] += 1
        if self.counts['semantic_markers'] == 1:
            self.get_logger().info(f"✅ semantic_markers: {len(msg.markers)} markers")

    def trajectory_callback(self, msg):
        self.counts['trajectory'] += 1
        if self.counts['trajectory'] == 1:
            self.get_logger().info(f"✅ trajectory: {len(msg.poses)} poses")

    def frontiers_callback(self, msg):
        self.counts['frontiers'] += 1
        if self.counts['frontiers'] == 1:
            self.get_logger().info(f"✅ frontiers: {len(msg.markers)} markers")

def main():
    import os
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init()

    verifier = TopicVerifier()

    print("\n验证WorldModel可视化话题（运行10秒）...")
    print("="*60)

    start = time.time()
    try:
        while time.time() - start < 10:
            rclpy.spin_once(verifier, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + "="*60)
        print("话题接收统计:")
        print(f"  semantic_grid: {verifier.counts['semantic_grid']} 条消息")
        print(f"  semantic_markers: {verifier.counts['semantic_markers']} 条消息")
        print(f"  trajectory: {verifier.counts['trajectory']} 条消息")
        print(f"  frontiers: {verifier.counts['frontiers']} 条消息")
        print("="*60)

        if all(count > 0 for count in verifier.counts.values()):
            print("\n✅ 所有话题都在正常发布数据！")
        else:
            print("\n⚠️  部分话题没有收到数据")

        verifier.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
