#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化系统调试测试脚本"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray
import time
import json


class VisualizationDebugger(Node):
    """可视化调试节点"""

    def __init__(self):
        super().__init__('visualization_debugger')
        
        # 创建订阅者
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        
        #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "debugger_init",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:__init__",
                "message": "可视化调试节点初始化",
                "data": {},
                "sessionId": "debug-session",
                "hypothesisId": "A,B,C,D"
            }) + "\n")
        #endregion
        
        # 订阅所有可视化话题
        self.subscription_semantic_grid = self.create_subscription(
            nav_msgs.msg.OccupancyGrid,
            '/world_model/semantic_grid',
            self.semantic_grid_callback,
            qos_profile
        )
        
        self.subscription_semantic_markers = self.create_subscription(
            visualization_msgs.msg.MarkerArray,
            '/world_model/semantic_markers',
            self.semantic_markers_callback,
            qos_profile
        )
        
        self.subscription_belief_markers = self.create_subscription(
            visualization_msgs.msg.MarkerArray,
            '/world_model/belief_markers',
            self.belief_markers_callback,
            qos_profile
        )
        
        self.subscription_trajectory = self.create_subscription(
            nav_msgs.msg.Path,
            '/world_model/trajectory',
            self.trajectory_callback,
            qos_profile
        )
        
        self.subscription_frontiers = self.create_subscription(
            visualization_msgs.msg.MarkerArray,
            '/world_model/frontiers',
            self.frontiers_callback,
            qos_profile
        )
        
        self.subscription_change_events = self.create_subscription(
            visualization_msgs.msg.MarkerArray,
            '/world_model/change_events',
            self.change_events_callback,
            qos_profile
        )
        
        self.subscription_vlm_detections = self.create_subscription(
            visualization_msgs.msg.MarkerArray,
            '/vlm/detections',
            self.vlm_detections_callback,
            qos_profile
        )
        
        self.get_logger().info("✅ 可视化调试节点已启动")
        self.get_logger().info("   监听话题:")
        self.get_logger().info("     - /world_model/semantic_grid")
        self.get_logger().info("     - /world_model/semantic_markers")
        self.get_logger().info("     - /world_model/belief_markers")
        self.get_logger().info("     - /world_model/trajectory")
        self.get_logger().info("     - /world_model/frontiers")
        self.get_logger().info("     - /world_model/change_events")
        self.get_logger().info("     - /vlm/detections")
        
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "debugger_subscriptions_created",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:__init__",
                "message": "已创建所有订阅者",
                "data": {},
                "sessionId": "debug-session",
                "hypothesisId": "D"
            }) + "\n")
        // #endregion
    
    def semantic_grid_callback(self, msg):
        """语义占据网格回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_semantic_grid",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:semantic_grid_callback",
                "message": "收到semantic_grid消息",
                "data": {"width": msg.info.width, "height": msg.info.height, "data_len": len(msg.data)},
                "sessionId": "debug-session",
                "hypothesisId": "A,B,C,D"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"📊 收到semantic_grid: {msg.info.width}x{msg.info.height}")
    
    def semantic_markers_callback(self, msg):
        """语义标记回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_semantic_markers",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:semantic_markers_callback",
                "message": "收到semantic_markers消息",
                "data": {"markers_count": len(msg.markers)},
                "sessionId": "debug-session",
                "hypothesisId": "A,B,C,D"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"🏷️  收到semantic_markers: {len(msg.markers)}个标记")
    
    def belief_markers_callback(self, msg):
        """信念标记回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_belief_markers",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:belief_markers_callback",
                "message": "收到belief_markers消息",
                "data": {"markers_count": len(msg.markers)},
                "sessionId": "debug-session",
                "hypothesisId": "B"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"💭 收到belief_markers: {len(msg.markers)}个信念标记")
    
    def trajectory_callback(self, msg):
        """轨迹回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_trajectory",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:trajectory_callback",
                "message": "收到trajectory消息",
                "data": {"poses_count": len(msg.poses)},
                "sessionId": "debug-session",
                "hypothesisId": "A,B,C,D"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"🛤️  收到trajectory: {len(msg.poses)}个位姿")
    
    def frontiers_callback(self, msg):
        """探索边界回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_frontiers",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:frontiers_callback",
                "message": "收到frontiers消息",
                "data": {"markers_count": len(msg.markers)},
                "sessionId": "debug-session",
                "hypothesisId": "A,B,C,D"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"🧭 收到frontiers: {len(msg.markers)}个探索边界")
    
    def change_events_callback(self, msg):
        """变化事件回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_change_events",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:change_events_callback",
                "message": "收到change_events消息",
                "data": {"markers_count": len(msg.markers)},
                "sessionId": "debug-session",
                "hypothesisId": "A,B,C"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"🔄 收到change_events: {len(msg.markers)}个变化事件")
    
    def vlm_detections_callback(self, msg):
        """VLM检测回调"""
        // #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                "id": "recv_vlm_detections",
                "timestamp": int(time.time() * 1000),
                "location": "test_visualization_debug.py:vlm_detections_callback",
                "message": "收到vlm_detections消息",
                "data": {"markers_count": len(msg.markers)},
                "sessionId": "debug-session",
                "hypothesisId": "B,C"
            }) + "\n")
        // #endregion
        self.get_logger().info(f"👁️ 收到vlm_detections: {len(msg.markers)}个VLM标记")


def main(args=None):
    rclpy.init(args=args)
    
    // #region agent log
    with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({
            "id": "debugger_main_start",
            "timestamp": int(time.time() * 1000),
            "location": "test_visualization_debug.py:main",
            "message": "调试节点主函数开始",
            "data": {},
            "sessionId": "debug-session",
            "hypothesisId": "A,B,C,D"
        }) + "\n")
    // #endregion
    
    debugger = VisualizationDebugger()
    
    try:
        rclpy.spin(debugger)
    except KeyboardInterrupt:
        pass
    finally:
        debugger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

