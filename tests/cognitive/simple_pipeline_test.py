#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的WorldModel可视化测试
直接发布数据，不订阅rosbag
"""
import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from rclpy.node import Node
import numpy as np
from datetime import datetime

from brain.cognitive.world_model.world_model import WorldModel
from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer

class SimpleTestPublisher(Node):
    def __init__(self):
        super().__init__('simple_test_publisher')

        # 创建WorldModel
        world_config = {'map_resolution': 0.1, 'map_size': 50.0}
        self.world_model = WorldModel(config=world_config)

        # 创建可视化器
        self.visualizer = WorldModelVisualizer(
            world_model=self.world_model,
            publish_rate=2.0
        )

        self.get_logger().info("✅ 简单测试发布器已启动")
        self.get_logger().info("   正在发布到 /world_model/* 话题")

        # 模拟一些数据更新
        self.counter = 0

        # 定时器：每秒更新一次
        self.timer = self.create_timer(1.0, self.update_world_model)

    def update_world_model(self):
        """模拟更新WorldModel"""
        self.counter += 1

        # 模拟机器人移动（绕圈）
        import math
        angle = self.counter * 0.1
        x = 5.0 * math.cos(angle)
        y = 5.0 * math.sin(angle)
        yaw = angle

        perception_data = {
            'timestamp': datetime.now(),
            'pose': {
                'x': x,
                'y': y,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': yaw
            },
            'velocity': {
                'linear_x': 0.5,
                'linear_y': 0.0,
                'linear_z': 0.0,
                'angular_x': 0.0,
                'angular_y': 0.0,
                'angular_z': 0.1
            }
        }

        # 更新WorldModel
        self.world_model.update_from_perception(perception_data)

        if self.counter % 5 == 0:
            self.get_logger().info(f"更新 #{self.counter}: 位置 ({x:.2f}, {y:.2f}), "
                                  f"位姿历史: {len(self.world_model.pose_history)}")

def main():
    import os
    os.environ['ROS_DOMAIN_ID'] = '42'

    rclpy.init()

    publisher = SimpleTestPublisher()
    visualizer = publisher.visualizer  # 获取visualizer节点

    print("\n" + "="*60)
    print("🚀 简单WorldModel可视化测试")
    print("="*60)
    print("\n正在发布模拟数据到WorldModel话题...")
    print("在RViz中应该能看到:")
    print("  - /world_model/semantic_grid (100x100空地图)")
    print("  - /world_model/trajectory (圆形轨迹)")
    print("="*60 + "\n")

    # 使用executor来spin多个节点
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(visualizer)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号")
    finally:
        executor.shutdown()
        publisher.destroy_node()
        visualizer.destroy_node()
        rclpy.shutdown()
        print("\n✅ 测试完成")

if __name__ == '__main__':
    main()
