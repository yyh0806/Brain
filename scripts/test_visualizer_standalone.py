#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立的WorldModelVisualizer测试"""

import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from brain.cognitive.world_model import WorldModel
from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer


def main():
    print("="*70)
    print("独立WorldModelVisualizer测试")
    print("="*70)
    
    # 初始化ROS2
    rclpy.init()
    
    # 创建WorldModel
    print("\n[1/3] 创建WorldModel...")
    world_model = WorldModel()
    print("✓ WorldModel已创建")
    
    # 添加一些测试数据
    print("\n[2/3] 添加测试数据...")
    
    # 添加机器人位姿历史
    world_model.update_robot_position(0.0, 0.0, 0.0)
    world_model.update_robot_position(1.0, 0.0, 0.0)
    world_model.update_robot_position(1.0, 1.0, 0.0)
    world_model.update_robot_position(2.0, 1.0, 0.0)
    print(f"✓ 机器人位姿历史: {len(world_model.pose_history)}个位姿")
    
    # 添加一些语义物体
    world_model.semantic_objects['obj_1'] = type('SemanticObject', (), {
        'label': '门',
        'world_position': [5.0, 5.0],
        'confidence': 0.9,
        'observation_count': 5,
        'state': type('ObjectState', (), {'value': 'CONFIRMED'}),
        'is_valid': lambda: True
    })()
    
    world_model.semantic_objects['obj_2'] = type('SemanticObject', (), {
        'label': '人',
        'world_position': [10.0, 10.0],
        'confidence': 0.85,
        'observation_count': 3,
        'state': type('ObjectState', (), {'value': 'DETECTED'}),
        'is_valid': lambda: True
    })()
    
    print(f"✓ 语义物体: {len(world_model.semantic_objects)}个")
    
    # 添加一些探索边界
    class Frontier:
        def __init__(self, center_x, center_y, priority):
            self.center_x = center_x
            self.center_y = center_y
            self.priority = priority
    
    world_model.exploration_frontiers = [
        Frontier(15.0, 15.0, 0.9),
        Frontier(20.0, 10.0, 0.6),
        Frontier(8.0, 20.0, 0.4)
    ]
    print(f"✓ 探索边界: {len(world_model.exploration_frontiers)}个")
    
    # 创建可视化器
    print("\n[3/3] 创建WorldModelVisualizer...")
    visualizer = WorldModelVisualizer(world_model, publish_rate=2.0)
    print("✓ WorldModelVisualizer已创建")
    print("\n✓ 可视化节点正在运行...")
    print("\n话题:")
    print("  - /world_model/semantic_grid")
    print("  - /world_model/semantic_markers")
    print("  - /world_model/belief_markers")
    print("  - /world_model/trajectory")
    print("  - /world_model/frontiers")
    print("  - /world_model/change_events")
    print("  - /vlm/detections")
    print("\n提示: 在另一个终端运行以下命令查看数据:")
    print("  ros2 topic echo /world_model/semantic_grid")
    print("  ros2 topic echo /world_model/semantic_markers")
    print("  ros2 topic echo /world_model/frontiers")
    print("\n按Ctrl+C停止\n")
    
    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("\n\n收到中断信号...")
    finally:
        print("停止可视化节点...")
        visualizer.destroy_node()
        rclpy.shutdown()
        print("✓ 已停止")


if __name__ == '__main__':
    main()

