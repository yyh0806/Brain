#!/usr/bin/env python3
import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray

class QuickVerify(Node):
    def __init__(self):
        super().__init__('quick_verify')
        self.received = {}

        self.create_subscription(OccupancyGrid, '/world_model/semantic_grid',
                                lambda msg: print(f"✅ semantic_grid: {msg.info.width}x{msg.info.height}, data_len={len(msg.data)}"), 10)
        self.create_subscription(Path, '/world_model/trajectory',
                                lambda msg: print(f"✅ trajectory: {len(msg.poses)} poses, latest=({msg.poses[-1].pose.position.x:.2f}, {msg.poses[-1].pose.position.y:.2f})"), 10)
        self.create_subscription(MarkerArray, '/world_model/semantic_markers',
                                lambda msg: print(f"✅ semantic_markers: {len(msg.markers)} markers"), 10)
        self.create_subscription(MarkerArray, '/world_model/frontiers',
                                lambda msg: print(f"✅ frontiers: {len(msg.markers)} markers"), 10)

        print("等待话题数据（5秒）...")
        print("="*60)

    def spin_once(self):
        rclpy.spin_once(self, timeout_sec=5.0)

def main():
    rclpy.init()
    verifier = QuickVerify()
    verifier.spin_once()
    verifier.destroy_node()
    rclpy.shutdown()
    print("="*60)
    print("验证完成！")

if __name__ == '__main__':
    main()
