#!/usr/bin/env python3
import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class MapDebugger(Node):
    def __init__(self):
        super().__init__('map_debugger')
        self.count = 0

        self.create_subscription(OccupancyGrid, '/world_model/semantic_grid',
                                self.map_callback, 10)

        self.get_logger().info("开始监听 /world_model/semantic_grid...")

    def map_callback(self, msg):
        self.count += 1

        print(f"\n{'='*60}")
        print(f"收到第 {self.count} 条消息")
        print(f"{'='*60}")
        print(f"Header:")
        print(f"  frame_id: '{msg.header.frame_id}'")
        print(f"  stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        print(f"\nInfo:")
        print(f"  map_load_time: {msg.info.map_load_time.sec}.{msg.info.map_load_time.nanosec}")
        print(f"  resolution: {msg.info.resolution}")
        print(f"  width: {msg.info.width}")
        print(f"  height: {msg.info.height}")
        print(f"  origin: ({msg.info.origin.position.x}, {msg.info.origin.position.y}, {msg.info.origin.position.z})")
        print(f"\nData:")
        print(f"  data length: {len(msg.data)}")
        if len(msg.data) > 0:
            print(f"  first 10 values: {msg.data[:10]}")
            print(f"  last 10 values: {msg.data[-10:]}")
            print(f"  unique values: {set(msg.data)}")
        else:
            print(f"  ⚠️  WARNING: data is empty!")

        if self.count >= 3:
            print(f"\n{'='*60}")
            print("已收到3条消息，退出监听")
            print(f"{'='*60}")
            raise SystemExit

def main():
    rclpy.init()
    debugger = MapDebugger()

    try:
        rclpy.spin(debugger)
    except SystemExit:
        pass
    finally:
        debugger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
