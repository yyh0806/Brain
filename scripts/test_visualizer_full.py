#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整的可视化器测试（包含所有数据类型）"""

import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import rclpy
from brain.cognitive.world_model import WorldModel
from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer


def main():
    print("="*70)
    print("完整WorldModelVisualizer测试（包含所有数据）")
    print("="*70)
    
    # 初始化ROS2
    rclpy.init()
    
    # 创建WorldModel
    print("\n[1/3] 创建WorldModel...")
    world_model = WorldModel()
    print("✓ WorldModel已创建")
    
    # 添加机器人位姿历史
    world_model.robot_position['x'] = 0.0
    world_model.robot_position['y'] = 0.0
    world_model.robot_position['z'] = 0.0
    world_model.pose_history.append({'x': 0.0, 'y': 0.0, 'z': 0.0})
    
    world_model.robot_position['x'] = 1.0
    world_model.robot_position['y'] = 0.0
    world_model.robot_position['z'] = 0.0
    world_model.pose_history.append({'x': 1.0, 'y': 0.0, 'z': 0.0})
    
    world_model.robot_position['x'] = 1.0
    world_model.robot_position['y'] = 1.0
    world_model.robot_position['z'] = 0.0
    world_model.pose_history.append({'x': 1.0, 'y': 1.0, 'z': 0.0})
    
    world_model.robot_position['x'] = 2.0
    world_model.robot_position['y'] = 1.0
    world_model.robot_position['z'] = 0.0
    world_model.pose_history.append({'x': 2.0, 'y': 1.0, 'z': 0.0})
    
    print(f"✓ 机器人位姿历史: {len(world_model.pose_history)}个位姿")
    print(f"✓ 当前机器人位置: ({world_model.robot_position['x']}, {world_model.robot_position['y']}, {world_model.robot_position['z']})")
    
    # 添加语义物体
    class TestSemanticObject:
        def __init__(self, label, position, confidence, observation_count=1, source='test'):
            self.label = label
            self.world_position = position
            self.confidence = confidence
            self.observation_count = observation_count
            self.state = type('ObjectState', (), {'value': 'CONFIRMED'})()
            self.attributes = {'source': source}
        
        def is_valid(self):
            return True
    
    world_model.semantic_objects['obj_1'] = TestSemanticObject('门', [5.0, 5.0], 0.9, 5)
    world_model.semantic_objects['obj_2'] = TestSemanticObject('人', [10.0, 10.0], 0.85, 3)
    world_model.semantic_objects['obj_3'] = TestSemanticObject('障碍物', [8.0, 12.0], 0.95, 7)
    
    print(f"✓ 语义物体: {len(world_model.semantic_objects)}个")
    
    # 添加VLM检测的物体（source='vlm'）
    world_model.semantic_objects['vlm_obj_1'] = TestSemanticObject('门', [15.0, 15.0], 0.92, 2, 'vlm')
    world_model.semantic_objects['vlm_obj_2'] = TestSemanticObject('人', [20.0, 20.0], 0.88, 1, 'vlm')
    
    print(f"✓ VLM检测的物体: 2个")
    
    # 添加一些探索边界
    class TestFrontier:
        def __init__(self, center_x, center_y, priority=0.5):
            self.center_x = center_x
            self.center_y = center_y
            self.priority = priority
    
    world_model.exploration_frontiers = [
        TestFrontier(15.0, 15.0, 0.9),
        TestFrontier(20.0, 10.0, 0.6),
        TestFrontier(8.0, 20.0, 0.4)
    ]
    print(f"✓ 探索边界: {len(world_model.exploration_frontiers)}个")
    
    # 添加一些变化事件（模拟）
    class TestEnvironmentChange:
        def __init__(self, change_type, position):
            from brain.cognitive.world_model.environment_change import ChangeType
            self.change_type = change_type
            self.data = {'position': position}
    
    from brain.cognitive.world_model.environment_change import ChangeType
    
    world_model.pending_changes = [
        TestEnvironmentChange(ChangeType.NEW_OBSTACLE, {'x': 25.0, 'y': 25.0, 'z': 0.0}),
        TestEnvironmentChange(ChangeType.TARGET_MOVED, {'x': 30.0, 'y': 30.0, 'z': 0.0})
    ]
    print(f"✓ 变化事件: {len(world_model.pending_changes)}个")
    
    # 添加一些信念（如果有信念修正策略）
    if hasattr(world_model, 'belief_revision_policy') and world_model.belief_revision_policy:
        class TestBelief:
            def __init__(self, belief_id, confidence, falsified=False, position=None):
                self.belief_id = belief_id
                self.confidence = confidence
                self.falsified = falsified
                self.metadata = {'position': position} if position else {}
        
        world_model.belief_revision_policy.beliefs['belief_1'] = TestBelief('belief_1', 0.9, False, {'x': 5.0, 'y': 5.0, 'z': 0.5})
        world_model.belief_revision_policy.beliefs['belief_2'] = TestBelief('belief_2', 0.7, False, {'x': 10.0, 'y': 10.0, 'z': 0.5})
        world_model.belief_revision_policy.beliefs['belief_3'] = TestBelief('belief_3', 0.4, False, {'x': 20.0, 'y': 20.0, 'z': 0.5})
        world_model.belief_revision_policy.beliefs['belief_4'] = TestBelief('belief_4', 0.9, True, {'x': 25.0, 'y': 25.0, 'z': 0.5})
        
        print(f"✓ 信念: {len(world_model.belief_revision_policy.beliefs)}个")
    
    # 创建可视化器
    print("\n[2/3] 创建WorldModelVisualizer...")
    visualizer = WorldModelVisualizer(world_model, publish_rate=2.0)
    print("✓ WorldModelVisualizer已创建")
    print("\n[3/3] 可视化节点正在运行...")
    print("\n话题:")
    print("  - /world_model/semantic_grid")
    print("  - /world_model/semantic_markers")
    print("  - /world_model/belief_markers")
    print("  - /world_model/trajectory")
    print("  - /world_model/frontiers")
    print("  - /world_model/change_events")
    print("  - /vlm/detections")
    print("\n请观察RViz，应该看到：")
    print("  ✓ 语义占据地图（100x100）")
    print("  ✓ 机器人轨迹（绿色，4个位姿）")
    print("  ✓ 语义物体标注（3个物体：门、人、障碍物）")
    print("  ✓ 探索边界（3个箭头，不同优先级）")
    print("  ✓ 信念标记（3个球体：绿色、黄色、红色）")
    print("  ✓ 变化事件（2个标记：橙色、紫色）")
    print("  ✓ VLM检测（2个带边界框的物体）")
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

