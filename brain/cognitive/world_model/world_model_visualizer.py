#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldModel语义可视化节点

将WorldModel的语义信息发布到RViz进行可视化，
使用占据栅格+语义颜色编码。
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import math
from typing import Tuple, Dict, Optional
from datetime import datetime


class WorldModelVisualizer(Node):
    """WorldModel语义可视化节点"""

    # 语义占据栅格值编码
    SEMANTIC_UNKNOWN = -1       # 未知 → 灰色
    SEMANTIC_FREE = 0           # 空闲 → 白色
    SEMANTIC_OCCUPIED = 100     # 普通占据 → 黑色

    # 语义扩展值 (101-199)
    SEMANTIC_DOOR = 101         # 门 → 蓝色
    SEMANTIC_PERSON = 102       # 人 → 红色
    SEMANTIC_BUILDING = 103     # 建筑 → 绿色
    SEMANTIC_OBSTACLE = 104     # 障碍物 → 橙色
    SEMANTIC_TARGET = 105       # 目标 → 紫色
    SEMANTIC_POI = 106          # 兴趣点 → 黄色

    def __init__(self, world_model, publish_rate: float = 2.0):
        """
        初始化可视化节点

        Args:
            world_model: WorldModel实例
            publish_rate: 发布频率 (Hz)
        """
        super().__init__('world_model_visualizer')

        self.world_model = world_model
        self.publish_rate = publish_rate

        # 发布者
        self.semantic_grid_pub = self.create_publisher(
            OccupancyGrid,
            '/world_model/semantic_grid',
            10
        )
        self.semantic_markers_pub = self.create_publisher(
            MarkerArray,
            '/world_model/semantic_markers',
            10
        )
        self.trajectory_pub = self.create_publisher(
            Path,
            '/world_model/trajectory',
            10
        )
        self.frontiers_pub = self.create_publisher(
            MarkerArray,
            '/world_model/frontiers',
            10
        )
        # ✨ 新增：信念状态发布者
        self.belief_markers_pub = self.create_publisher(
            MarkerArray,
            '/world_model/belief_markers',
            10
        )
        # ✨ 新增：变化事件发布者
        self.change_events_pub = self.create_publisher(
            MarkerArray,
            '/world_model/change_events',
            10
        )
        # ✨ 新增：VLM检测发布者
        self.vlm_markers_pub = self.create_publisher(
            MarkerArray,
            '/vlm/detections',
            10
        )
        # ✨ 新增：因果图发布者（三模态融合 - 因果地图模态）
        self.causal_graph_pub = self.create_publisher(
            MarkerArray,
            '/world_model/causal_graph',
            10
        )

        self.get_logger().info("✅ WorldModelVisualizer initialized")
        self.get_logger().info(f"   发布频率: {publish_rate} Hz")
        self.get_logger().info("   发布的话题:")
        self.get_logger().info("     - /world_model/semantic_grid (OccupancyGrid)")
        self.get_logger().info("     - /world_model/semantic_markers (MarkerArray)")
        self.get_logger().info("     - /world_model/belief_markers (MarkerArray) ✨ 新增")
        self.get_logger().info("     - /world_model/trajectory (Path)")
        self.get_logger().info("     - /world_model/frontiers (MarkerArray)")
        self.get_logger().info("     - /world_model/change_events (MarkerArray) ✨ 新增")
        self.get_logger().info("     - /vlm/detections (MarkerArray) ✨ 新增")
        self.get_logger().info("     - /world_model/causal_graph (MarkerArray) ✨ 新增三模态融合")

        # 定时器：定时发布可视化数据
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_visualization
        )

        # Marker ID计数器（用于删除旧marker）
        self.marker_id_counter = 0
        self._last_processed_changes = set()  # ✨ 新增：跟踪已处理的变化

    def publish_visualization(self):
        """发布所有可视化数据"""
        try:
            #region agent log
            import json
            with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "viz_publish_start",
                    "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                    "location": "world_model_visualizer.py:publish_visualization",
                    "message": "开始发布可视化数据",
                    "data": {"marker_id_counter": self.marker_id_counter},
                    "sessionId": "debug-session",
                    "hypothesisId": "A,B,C,D"
                }) + "\n")
            #endregion
            
            #region agent log
            # 1. 生成并发布语义占据栅格
            try:
                semantic_grid = self._generate_semantic_grid()
                if semantic_grid is not None:
                    self.semantic_grid_pub.publish(semantic_grid)
                    with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "id": "viz_semantic_grid_ok",
                            "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                            "location": "world_model_visualizer.py:publish_visualization",
                            "message": "成功发布semantic_grid",
                            "data": {
                                "width": semantic_grid.info.width,
                                "height": semantic_grid.info.height,
                                "data_len": len(semantic_grid.data),
                                "origin": [semantic_grid.info.origin.position.x, semantic_grid.info.origin.position.y],
                                "resolution": semantic_grid.info.resolution
                            },
                            "sessionId": "debug-session",
                            "hypothesisId": "A"
                        }) + "\n")
                    if self.marker_id_counter % 10 == 0:  # 每10次打印一次
                        self.get_logger().info(f"✅ 发布semantic_grid: {semantic_grid.info.width}x{semantic_grid.info.height}, data_len={len(semantic_grid.data)}")
                else:
                    with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "id": "viz_semantic_grid_none",
                            "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                            "location": "world_model_visualizer.py:publish_visualization",
                            "message": "semantic_grid为None",
                            "data": {},
                            "sessionId": "debug-session",
                            "hypothesisId": "B"
                        }) + "\n")
            except Exception as e:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_semantic_grid_error",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": f"semantic_grid生成/发布错误: {e}",
                        "data": {"error_type": type(e).__name__, "error_msg": str(e)},
                        "sessionId": "debug-session",
                        "hypothesisId": "A"
                    }) + "\n")
                self.get_logger().error(f"semantic_grid错误: {e}", throttle_duration_sec=5.0)
            #endregion

            #region agent log
            # 2. 生成并发布语义物体标注
            markers = self._generate_semantic_markers()
            if markers is not None:
                self.semantic_markers_pub.publish(markers)
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_semantic_markers_ok",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "成功发布semantic_markers",
                        "data": {"count": len(markers.markers)},
                        "sessionId": "debug-session",
                        "hypothesisId": "C"
                    }) + "\n")
            else:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_semantic_markers_none",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "semantic_markers为None",
                        "data": {"semantic_objects_count": len(self.world_model.semantic_objects) if hasattr(self.world_model, 'semantic_objects') else 0},
                        "sessionId": "debug-session",
                        "hypothesisId": "C"
                    }) + "\n")
            #endregion

            #region agent log
            # 1. 生成并发布语义占据栅格
            try:
                semantic_grid = self._generate_semantic_grid()
                if semantic_grid is not None:
                    self.semantic_grid_pub.publish(semantic_grid)
                    with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "id": "viz_semantic_grid_ok",
                            "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                            "location": "world_model_visualizer.py:publish_visualization",
                            "message": "成功发布semantic_grid",
                            "data": {
                                "width": semantic_grid.info.width,
                                "height": semantic_grid.info.height,
                                "data_len": len(semantic_grid.data),
                                "origin": [semantic_grid.info.origin.position.x, semantic_grid.info.origin.position.y],
                                "resolution": semantic_grid.info.resolution
                            },
                            "sessionId": "debug-session",
                            "hypothesisId": "A"
                        }) + "\n")
                    if self.marker_id_counter % 10 == 0:
                        self.get_logger().info(f"✅ 发布semantic_grid: {semantic_grid.info.width}x{semantic_grid.info.height}, data_len={len(semantic_grid.data)}")
                else:
                    with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                        f.write(json.dumps({
                            "id": "viz_semantic_grid_none",
                            "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                            "location": "world_model_visualizer.py:publish_visualization",
                            "message": "semantic_grid为None",
                            "data": {},
                            "sessionId": "debug-session",
                            "hypothesisId": "B"
                        }) + "\n")
            except Exception as e:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_semantic_grid_error",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": f"semantic_grid生成/发布错误: {e}",
                        "data": {"error_type": type(e).__name__, "error_msg": str(e)},
                        "sessionId": "debug-session",
                        "hypothesisId": "A"
                    }) + "\n")
                self.get_logger().error(f"semantic_grid错误: {e}", throttle_duration_sec=5.0)
            #endregion

            #region agent log
            # 4. 生成并发布探索前沿
            frontiers = self._generate_frontier_markers()
            if frontiers is not None:
                self.frontiers_pub.publish(frontiers)
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_frontiers_ok",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "成功发布frontiers",
                        "data": {"count": len(frontiers.markers)},
                        "sessionId": "debug-session",
                        "hypothesisId": "C"
                    }) + "\n")
            else:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_frontiers_none",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "frontiers为None",
                        "data": {"frontiers_count": len(self.world_model.exploration_frontiers) if hasattr(self.world_model, 'exploration_frontiers') else 0},
                        "sessionId": "debug-session",
                        "hypothesisId": "B"
                    }) + "\n")
            #endregion

            #region agent log
            # 5. ✨ 新增：生成并发布信念状态标记
            belief_markers = self._generate_belief_markers()
            if belief_markers is not None:
                self.belief_markers_pub.publish(belief_markers)
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_belief_markers_ok",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "成功发布belief_markers",
                        "data": {"count": len(belief_markers.markers)},
                        "sessionId": "debug-session",
                        "hypothesisId": "B,C"
                    }) + "\n")
            else:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_belief_markers_none",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "belief_markers为None",
                        "data": {"has_belief_policy": hasattr(self.world_model, 'belief_revision_policy')},
                        "sessionId": "debug-session",
                        "hypothesisId": "B"
                    }) + "\n")
            #endregion

            #region agent log
            # 6. ✨ 新增：生成并发布变化事件
            change_markers = self._generate_change_markers()
            if change_markers is not None:
                self.change_events_pub.publish(change_markers)
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_change_markers_ok",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "成功发布change_markers",
                        "data": {"count": len(change_markers.markers)},
                        "sessionId": "debug-session",
                        "hypothesisId": "A,C"
                    }) + "\n")
            else:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_change_markers_none",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "change_markers为None",
                        "data": {"has_pending_changes": hasattr(self.world_model, 'pending_changes'), "changes_count": len(self.world_model.pending_changes) if hasattr(self.world_model, 'pending_changes') else 0},
                        "sessionId": "debug-session",
                        "hypothesisId": "B,C"
                    }) + "\n")
            #endregion

            #region agent log
            # 7. ✨ 新增：生成并发布VLM检测
            vlm_markers = self._generate_vlm_markers()
            if vlm_markers is not None:
                self.vlm_markers_pub.publish(vlm_markers)
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_vlm_markers_ok",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "成功发布vlm_markers",
                        "data": {"count": len(vlm_markers.markers)},
                        "sessionId": "debug-session",
                        "hypothesisId": "B,C"
                    }) + "\n")
            else:
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_vlm_markers_none",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "vlm_markers为None",
                        "data": {"has_semantic_objects": hasattr(self.world_model, 'semantic_objects'), "objects_count": len(self.world_model.semantic_objects) if hasattr(self.world_model, 'semantic_objects') else 0},
                        "sessionId": "debug-session",
                        "hypothesisId": "B,C"
                    }) + "\n")
            #endregion

            #region agent log
            # 8. ✨ 新增：生成并发布因果图（三模态融合 - 因果地图模态）
            causal_markers = self._generate_causal_graph_markers()
            if causal_markers is not None:
                self.causal_graph_pub.publish(causal_markers)
                with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        "id": "viz_causal_graph_ok",
                        "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                        "location": "world_model_visualizer.py:publish_visualization",
                        "message": "成功发布causal_graph_markers",
                        "data": {"count": len(causal_markers.markers)},
                        "sessionId": "debug-session",
                        "hypothesisId": "C"
                    }) + "\n")
            #endregion

            #region agent log
            self.marker_id_counter += 1
            with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "viz_publish_complete",
                    "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                    "location": "world_model_visualizer.py:publish_visualization",
                    "message": "发布可视化数据完成",
                    "data": {"new_counter": self.marker_id_counter},
                    "sessionId": "debug-session",
                    "hypothesisId": "A"
                }) + "\n")
            #endregion

        except Exception as e:
            #region agent log
            with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    "id": "viz_exception",
                    "timestamp": int(self.get_clock().now().nanoseconds / 1000000),
                    "location": "world_model_visualizer.py:publish_visualization",
                    "message": f"发布可视化数据时出错: {e}",
                    "data": {"error_type": type(e).__name__, "error_msg": str(e)},
                    "sessionId": "debug-session",
                    "hypothesisId": "A"
                }) + "\n")
            #endregion
            self.get_logger().error(f"发布可视化数据时出错: {e}", throttle_duration_sec=5.0)

    def _generate_semantic_grid(self) -> Optional[OccupancyGrid]:
        """
        生成语义占据栅格

        Returns:
            OccupancyGrid消息
        """
        # ✅ 修复：不再强制固定尺寸，使用实际地图尺寸
        # 这样可以完整显示1000x1000的地图
        if self.world_model.current_map is None:
            # 如果栅格未初始化，创建一个默认的500x500栅格
            grid = np.full((500, 500), -1, dtype=np.int8)
        else:
            # 使用实际的地图尺寸（可能是1000x1000）
            grid = self.world_model.current_map

        # 调试：每10次发布打印一次地图信息
        if self.marker_id_counter % 10 == 0:
            occupied = np.sum(grid == 100)
            self.get_logger().info(f"🗺️  发布地图: {grid.shape}, 占据={occupied}")

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = "map"  # 使用map frame作为世界坐标系

        # 设置栅格元数据
        grid_msg.info.resolution = self.world_model.map_resolution
        grid_msg.info.width = grid.shape[1]
        grid_msg.info.height = grid.shape[0]

        # 设置原点位置
        grid_msg.info.origin.position.x = self.world_model.map_origin[0]
        grid_msg.info.origin.position.y = self.world_model.map_origin[1]
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # 复制并增强栅格（添加语义颜色）
        semantic_grid = self._enhance_grid_with_semantics(grid)
        grid_msg.data = semantic_grid.flatten().astype(np.int8).tolist()

        return grid_msg

    def _enhance_grid_with_semantics(self, grid: np.ndarray) -> np.ndarray:
        """
        将语义信息编码到栅格中

        Args:
            grid: 原始占据栅格

        Returns:
            增强后的语义栅格
        """
        semantic_grid = grid.copy()

        # 1. 标记VLM识别的物体
        for obj_id, obj in self.world_model.semantic_objects.items():
            if not hasattr(obj, 'is_valid') or not obj.is_valid():
                continue

            if not hasattr(obj, 'world_position') or obj.world_position is None:
                continue

            gx, gy = self._world_to_grid(obj.world_position)
            if self._is_valid_grid(gx, gy, semantic_grid.shape):
                # 根据标签设置语义值
                semantic_value = self._get_semantic_value(obj.label)
                semantic_grid[gy, gx] = semantic_value

        # 2. 标记追踪的障碍物
        for obj_id, obj in self.world_model.tracked_objects.items():
            if not hasattr(obj, 'position'):
                continue

            position = obj.position
            if hasattr(position, 'x') and hasattr(position, 'y'):
                gx, gy = self._world_to_grid((position.x, position.y))
                if self._is_valid_grid(gx, gy, semantic_grid.shape):
                    semantic_grid[gy, gx] = self.SEMANTIC_OBSTACLE

        # 3. 标记目标（扩展区域）
        for obj_id, obj in self.world_model.semantic_objects.items():
            if not hasattr(obj, 'is_valid') or not obj.is_valid():
                continue

            if not hasattr(obj, 'is_target') or not obj.is_target:
                continue

            if not hasattr(obj, 'world_position') or obj.world_position is None:
                continue

            gx, gy = self._world_to_grid(obj.world_position)
            # 扩展目标区域（3x3）
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = gx + dx, gy + dy
                    if self._is_valid_grid(nx, ny, semantic_grid.shape):
                        semantic_grid[ny, nx] = self.SEMANTIC_TARGET

        return semantic_grid

    def _get_semantic_value(self, label: str) -> int:
        """
        根据标签返回语义值

        Args:
            label: 物体标签

        Returns:
            语义占据栅格值
        """
        label_lower = label.lower()

        semantic_map = {
            '门': self.SEMANTIC_DOOR,
            'door': self.SEMANTIC_DOOR,
            '入口': self.SEMANTIC_DOOR,
            'entrance': self.SEMANTIC_DOOR,
            '门禁': self.SEMANTIC_DOOR,
            'gate': self.SEMANTIC_DOOR,

            '人': self.SEMANTIC_PERSON,
            'person': self.SEMANTIC_PERSON,
            '行人': self.SEMANTIC_PERSON,
            'pedestrian': self.SEMANTIC_PERSON,
            '人影': self.SEMANTIC_PERSON,

            '建筑': self.SEMANTIC_BUILDING,
            'building': self.SEMANTIC_BUILDING,
            '房子': self.SEMANTIC_BUILDING,
            'house': self.SEMANTIC_BUILDING,
            '房屋': self.SEMANTIC_BUILDING,
            '房间': self.SEMANTIC_BUILDING,
            'room': self.SEMANTIC_BUILDING,
        }

        return semantic_map.get(label_lower, self.SEMANTIC_OCCUPIED)

    def _generate_semantic_markers(self) -> Optional[MarkerArray]:
        """
        生成语义物体标注（3D文字）

        Returns:
            MarkerArray消息
        """
        markers = MarkerArray()
        marker_id = 0

        for obj_id, obj in self.world_model.semantic_objects.items():
            if not hasattr(obj, 'is_valid') or not obj.is_valid():
                continue

            if not hasattr(obj, 'world_position') or obj.world_position is None:
                continue

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "semantic_labels"
            marker.id = marker_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # 位置（在物体上方1米处）
            marker.pose.position.x = obj.world_position[0]
            marker.pose.position.y = obj.world_position[1]
            marker.pose.position.z = 1.0
            marker.pose.orientation.w = 1.0

            # 文字标签
            marker.text = f"{obj.label}"
            marker.scale.z = 0.3  # 文字高度

            # 颜色
            color = self._get_label_color(obj.label)
            marker.color.r = color[0] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[2] / 255.0
            marker.color.a = 1.0

            # 生命周期（自动删除）
            marker.lifetime.sec = 1  # 1秒后自动删除

            markers.markers.append(marker)
            marker_id += 1

        return markers if markers.markers else None

    def _generate_trajectory(self) -> Optional[Path]:
        """
        生成机器人轨迹

        Returns:
            Path消息
        """
        pose_history = self.world_model.pose_history
        if not pose_history:
            return None

        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()

        # 从pose_history提取最近100个轨迹点
        recent_poses = pose_history[-100:] if len(pose_history) > 100 else pose_history

        for pose_entry in recent_poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose.position.x = pose_entry.get('x', 0.0)
            pose_stamped.pose.position.y = pose_entry.get('y', 0.0)
            pose_stamped.pose.position.z = 0.0

            # 设置航向
            heading = pose_entry.get('heading', 0.0)
            # 简单的航向转四元数（只考虑yaw）
            import math
            pose_stamped.pose.orientation.z = math.sin(heading / 2.0)
            pose_stamped.pose.orientation.w = math.cos(heading / 2.0)

            path.poses.append(pose_stamped)

        return path

    def _generate_frontier_markers(self) -> Optional[MarkerArray]:
        """
        生成探索前沿标记（增强版：按优先级可视化）
        
        Returns:
            MarkerArray: 包含探索边界的Markers
        """
        frontiers = self.world_model.exploration_frontiers
        if not frontiers:
            return None

        markers = MarkerArray()

        for i, frontier in enumerate(frontiers):
            # 1. 主箭头Marker
            arrow_marker = Marker()
            arrow_marker.header.frame_id = "map"
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = "frontiers"
            arrow_marker.id = i
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD

            # 位置
            arrow_marker.pose.position.x = frontier.position[0]
            arrow_marker.pose.position.y = frontier.position[1]
            arrow_marker.pose.position.z = 0.5
            arrow_marker.pose.orientation.w = 1.0

            # 根据优先级设置大小和颜色
            priority = getattr(frontier, 'priority', 0.5)
            size, color = self._get_frontier_properties(priority)
            
            arrow_marker.scale.x = size['length']  # 箭头长度
            arrow_marker.scale.y = size['width']   # 箭头宽度
            arrow_marker.scale.z = size['height']  # 箭头高度
            
            arrow_marker.color.r = color[0]
            arrow_marker.color.g = color[1]
            arrow_marker.color.b = color[2]
            arrow_marker.color.a = 0.9

            # 生命周期
            arrow_marker.lifetime.sec = 1

            markers.markers.append(arrow_marker)

            # 2. 文字标签Marker（优先级和距离）
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "frontier_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = frontier.position[0]
            text_marker.pose.position.y = frontier.position[1]
            text_marker.pose.position.z = 1.5
            text_marker.pose.orientation.w = 1.0

            # 计算距离
            dx = frontier.position[0] - self.world_model.robot_position.get('x', 0)
            dy = frontier.position[1] - self.world_model.robot_position.get('y', 0)
            distance = math.sqrt(dx * dx + dy * dy)

            # 文本内容：优先级和距离
            text_marker.text = f"P:{priority:.2f}\nD:{distance:.1f}m"
            text_marker.scale.z = 0.2

            # 颜色与箭头相同
            text_marker.color.r = color[0]
            text_marker.color.g = color[1]
            text_marker.color.b = color[2]
            text_marker.color.a = 0.9

            text_marker.lifetime.sec = 1

            markers.markers.append(text_marker)

        return markers

    def _get_frontier_properties(self, priority: float):
        """
        根据优先级获取探索边界的属性
        
        Args:
            priority: 优先级 0-1
        
        Returns:
            (size_dict, color_tuple)
        """
        if priority > 0.8:
            # 高优先级：亮绿色大箭头
            return (
                {'length': 1.5, 'width': 0.3, 'height': 0.3},
                (0.0, 1.0, 0.0)
            )
        elif priority > 0.5:
            # 中等优先级：黄色中箭头
            return (
                {'length': 1.0, 'width': 0.2, 'height': 0.2},
                (1.0, 1.0, 0.0)
            )
        else:
            # 低优先级：灰色小箭头
            return (
                {'length': 0.5, 'width': 0.1, 'height': 0.1},
                (0.5, 0.5, 0.5)
            )

    def _world_to_grid(self, world_position: Tuple[float, float]) -> Tuple[int, int]:
        """
        将世界坐标转换为栅格坐标

        Args:
            world_position: 世界坐标 (x, y)

        Returns:
            栅格坐标 (gx, gy)
        """
        wx, wy = world_position

        # 使用WorldModel的map_origin和map_resolution
        resolution = self.world_model.map_resolution
        origin_x, origin_y = self.world_model.map_origin

        gx = int((wx - origin_x) / resolution)
        gy = int((wy - origin_y) / resolution)

        return gx, gy

    def _is_valid_grid(self, gx: int, gy: int, shape: Tuple[int, int]) -> bool:
        """
        检查栅格坐标是否有效

        Args:
            gx: 栅格x坐标
            gy: 栅格y坐标
            shape: 栅格形状 (height, width)

        Returns:
            是否有效
        """
        height, width = shape
        return 0 <= gx < width and 0 <= gy < height

    def _get_label_color(self, label: str) -> Tuple[int, int, int]:
        """
        获取标签对应的RGB颜色

        Args:
            label: 物体标签

        Returns:
            RGB颜色元组 (r, g, b)，范围0-255
        """
        label_lower = label.lower()

        color_map = {
            '门': (0, 0, 255),           # 蓝色
            'door': (0, 0, 255),
            '入口': (0, 0, 255),

            '人': (255, 0, 0),           # 红色
            'person': (255, 0, 0),
            '行人': (255, 0, 0),

            '建筑': (0, 128, 0),         # 绿色
            'building': (0, 128, 0),
            '房子': (0, 128, 0),
            '房间': (0, 128, 0),

            '障碍': (255, 165, 0),       # 橙色
            'obstacle': (255, 165, 0),
            '墙': (255, 165, 0),
            'wall': (255, 165, 0),
        }

        return color_map.get(label_lower, (128, 128, 128))  # 默认灰色

    # ========== 信念状态可视化（新增） ==========

    def _generate_belief_markers(self) -> Optional[MarkerArray]:
        """
        生成信念状态标记
        
        Returns:
            MarkerArray: 包含所有信念的球形Marker
        """
        markers = MarkerArray()
        
        # 从WorldModel获取信念修正策略
        if not hasattr(self.world_model, 'belief_revision_policy'):
            return None
        
        belief_policy = self.world_model.belief_revision_policy
        if not belief_policy or not hasattr(belief_policy, 'beliefs'):
            return None
        
        marker_id = 0
        
        # 遍历所有信念
        for belief_id, belief in belief_policy.beliefs.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "belief_markers"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 设置位置（如果有位置信息）
            if hasattr(belief, 'metadata') and belief.metadata:
                position = belief.metadata.get('position', {})
                marker.pose.position.x = float(position.get('x', 0.0))
                marker.pose.position.y = float(position.get('y', 0.0))
                marker.pose.position.z = float(position.get('z', 0.0))
                marker.pose.orientation.w = 1.0
            else:
                # 如果没有位置信息，在机器人位置显示
                marker.pose.position.x = self.world_model.robot_position.get('x', 0.0)
                marker.pose.position.y = self.world_model.robot_position.get('y', 0.0)
                marker.pose.position.z = 0.5
                marker.pose.orientation.w = 1.0
            
            # 根据置信度设置大小和颜色
            confidence = belief.confidence if hasattr(belief, 'confidence') else 0.5
            marker.scale.x = 0.1 + confidence * 0.2  # 0.1-0.3米
            marker.scale.y = 0.1 + confidence * 0.2
            marker.scale.z = 0.1 + confidence * 0.2
            
            # 颜色编码
            color = self._get_belief_color(confidence, belief.falsified if hasattr(belief, 'falsified') else False)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.7
            
            # 生命周期
            marker.lifetime.sec = 1
            
            markers.markers.append(marker)
            marker_id += 1
        
        return markers if markers.markers else None

    def _get_belief_color(self, confidence: float, falsified: bool) -> Tuple[float, float, float]:
        """
        获取信念对应的颜色
        
        Args:
            confidence: 置信度 0-1
            falsified: 是否已证伪
        
        Returns:
            RGB颜色元组
        """
        if falsified:
            return (0.5, 0.5, 0.5)  # 灰色
        elif confidence > 0.8:
            return (0.0, 1.0, 0.0)  # 绿色
        elif confidence > 0.5:
            return (1.0, 1.0, 0.0)  # 黄色
        else:
            return (1.0, 0.0, 0.0)  # 红色

    # ========== VLM检测可视化（新增） ==========

    def _generate_vlm_markers(self) -> Optional[MarkerArray]:
        """
        生成VLM检测标记
        
        Returns:
            MarkerArray: 包含VLM检测到的物体的Markers
        """
        markers = MarkerArray()
        
        if not hasattr(self.world_model, 'semantic_objects'):
            return None
        
        marker_id = 0
        
        for obj_id, obj in self.world_model.semantic_objects.items():
            # 只处理VLM检测的物体
            if not hasattr(obj, 'attributes'):
                continue
            if obj.attributes.get('source') != 'vlm':
                continue
            
            if not obj.is_valid() or obj.world_position is None:
                continue
            
            # 1. 创建边界框Marker
            bbox_marker = self._create_vlm_bbox_marker(obj, marker_id)
            if bbox_marker:
                markers.markers.append(bbox_marker)
                marker_id += 1
            
            # 2. 创建标签Marker
            label_marker = self._create_vlm_label_marker(obj, marker_id)
            if label_marker:
                markers.markers.append(label_marker)
                marker_id += 1
            
            # 3. 创建置信度Marker
            conf_marker = self._create_vlm_confidence_marker(obj, marker_id)
            if conf_marker:
                markers.markers.append(conf_marker)
                marker_id += 1
        
        return markers if markers.markers else None

    def _create_vlm_bbox_marker(self, obj, marker_id: int) -> Optional[Marker]:
        """创建VLM边界框Marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vlm_bboxes"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 位置
        marker.pose.position.x = obj.world_position[0]
        marker.pose.position.y = obj.world_position[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # 边界框大小（假设1x1米）
        size = 1.0
        half_size = size / 2.0
        
        # 定义矩形顶点
        points = [
            Point(x=obj.world_position[0] - half_size, y=obj.world_position[1] - half_size, z=0.0),
            Point(x=obj.world_position[0] + half_size, y=obj.world_position[1] - half_size, z=0.0),
            Point(x=obj.world_position[0] + half_size, y=obj.world_position[1] + half_size, z=0.0),
            Point(x=obj.world_position[0] - half_size, y=obj.world_position[1] + half_size, z=0.0),
            Point(x=obj.world_position[0] - half_size, y=obj.world_position[1] - half_size, z=0.0)  # 闭合
        ]
        
        # 添加点到Marker
        marker.points = points
        
        # 颜色
        color = self._get_vlm_color(obj.label)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8
        
        # 线宽
        marker.scale.x = 0.05  # 线宽
        
        # 生命周期
        marker.lifetime.sec = 1
        
        return marker

    def _create_vlm_label_marker(self, obj, marker_id: int) -> Optional[Marker]:
        """创建VLM标签Marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vlm_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 位置（在物体上方）
        marker.pose.position.x = obj.world_position[0]
        marker.pose.position.y = obj.world_position[1]
        marker.pose.position.z = 1.5
        marker.pose.orientation.w = 1.0
        
        # 标签内容
        label_text = f"{obj.label}"
        if hasattr(obj, 'state'):
            label_text += f"\n[{obj.state.value}]"
        if hasattr(obj, 'observation_count'):
            label_text += f"\nobs:{obj.observation_count}"
        
        marker.text = label_text
        marker.scale.z = 0.3  # 文字高度
        
        # 颜色
        color = self._get_vlm_color(obj.label)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        
        # 生命周期
        marker.lifetime.sec = 1
        
        return marker

    def _create_vlm_confidence_marker(self, obj, marker_id: int) -> Optional[Marker]:
        """创建VLM置信度Marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vlm_confidence"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 位置（在标签下方）
        marker.pose.position.x = obj.world_position[0]
        marker.pose.position.y = obj.world_position[1]
        marker.pose.position.z = 1.2
        marker.pose.orientation.w = 1.0
        
        # 置信度内容
        marker.text = f"conf:{obj.confidence:.2f}"
        marker.scale.z = 0.2  # 文字高度
        
        # 白色
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.9
        
        # 生命周期
        marker.lifetime.sec = 1
        
        return marker

    def _get_vlm_color(self, label: str) -> Tuple[float, float, float]:
        """获取VLM标签对应的颜色"""
        label_lower = label.lower()
        
        color_map = {
            '门': (0.0, 0.0, 1.0),           # 蓝色
            'door': (0.0, 0.0, 1.0),
            '入口': (0.0, 0.0, 1.0),
            'entrance': (0.0, 0.0, 1.0),
            
            '人': (1.0, 0.0, 0.0),           # 红色
            'person': (1.0, 0.0, 0.0),
            '行人': (1.0, 0.0, 0.0),
            'pedestrian': (1.0, 0.0, 0.0),
            
            '建筑': (0.0, 0.5, 0.0),         # 绿色
            'building': (0.0, 0.5, 0.0),
            '房子': (0.0, 0.5, 0.0),
            'house': (0.0, 0.5, 0.0),
            '房间': (0.0, 0.5, 0.0),
            'room': (0.0, 0.5, 0.0),
            
            '障碍': (1.0, 0.5, 0.0),         # 橙色
            'obstacle': (1.0, 0.5, 0.0),
            '墙': (1.0, 0.5, 0.0),
            'wall': (1.0, 0.5, 0.0),
            
            '目标': (0.5, 0.0, 0.5),         # 紫色
            'target': (0.5, 0.0, 0.5),
        }
        
        return color_map.get(label_lower, (0.5, 0.5, 0.5))  # 默认灰色

    # ========== 变化事件可视化（新增） ==========

    def _generate_change_markers(self) -> Optional[MarkerArray]:
        """
        生成环境变化事件标记
        
        Returns:
            MarkerArray: 包含变化事件的临时Markers
        """
        markers = MarkerArray()
        
        # 获取待处理的变化
        if not hasattr(self.world_model, 'pending_changes'):
            return None
        
        pending_changes = self.world_model.pending_changes
        if not pending_changes:
            return None
        
        marker_id = 0
        
        # 跟踪已处理的变化
        current_changes = set()
        
        for i, change in enumerate(pending_changes):
            # 跳过已处理的变化
            change_key = f"{change.change_type.value}_{i}"
            if change_key in self._last_processed_changes:
                continue
            
            current_changes.add(change_key)
            
            marker = self._create_change_marker(change, marker_id)
            if marker:
                markers.markers.append(marker)
                marker_id += 1
        
        # 更新已处理的变化列表
        self._last_processed_changes = current_changes
        
        return markers if markers.markers else None

    def _create_change_marker(self, change, marker_id: int) -> Optional[Marker]:
        """创建变化事件Marker"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "change_events"
        marker.id = marker_id
        marker.action = Marker.ADD
        
        # 根据变化类型设置Marker类型和属性
        marker_type, color, scale, position = self._get_change_marker_properties(change)
        
        marker.type = marker_type
        marker.pose.position.x = float(position.get('x', 0.0))
        marker.pose.position.y = float(position.get('y', 0.0))
        marker.pose.position.z = float(position.get('z', 0.0))
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8
        
        # 临时标记，5秒后自动消失
        marker.lifetime.sec = 5
        
        return marker

    def _get_change_marker_properties(self, change):
        """获取变化Marker的属性"""
        from brain.cognitive.world_model.environment_change import ChangeType
        
        change_type = change.change_type
        
        # 默认位置
        position = change.data.get('position', {})
        if not position:
            # 如果没有位置，使用机器人位置
            position = {
                'x': self.world_model.robot_position.get('x', 0.0),
                'y': self.world_model.robot_position.get('y', 0.0),
                'z': 0.5
            }
        
        if change_type == ChangeType.NEW_OBSTACLE:
            # 新障碍物：橙色圆柱体
            return (
                Marker.CYLINDER,
                (1.0, 0.5, 0.0),  # 橙色
                (0.5, 0.5, 0.2),  # 尺寸
                position
            )
        
        elif change_type == ChangeType.TARGET_MOVED:
            # 目标移动：紫色圆柱体
            return (
                Marker.CYLINDER,
                (0.5, 0.0, 0.5),  # 紫色
                (0.5, 0.5, 0.2),
                position
            )
        
        elif change_type == ChangeType.PATH_BLOCKED:
            # 路径阻塞：红色X（LINE_LIST）
            return (
                Marker.LINE_LIST,
                (1.0, 0.0, 0.0),  # 红色
                (0.2, 0.2, 0.2),
                position
            )
        
        elif change_type == ChangeType.OBSTACLE_MOVED:
            # 障碍物移动：黄色圆柱体
            return (
                Marker.CYLINDER,
                (1.0, 1.0, 0.0),  # 黄色
                (0.4, 0.4, 0.2),
                position
            )
        
        elif change_type == ChangeType.TARGET_APPEARED:
            # 目标出现：绿色圆柱体
            return (
                Marker.CYLINDER,
                (0.0, 1.0, 0.0),  # 绿色
                (0.5, 0.5, 0.2),
                position
            )
        
        elif change_type == ChangeType.OBSTACLE_REMOVED:
            # 障碍物移除：蓝色虚线框（LINE_STRIP）
            return (
                Marker.LINE_STRIP,
                (0.0, 0.0, 1.0),  # 蓝色
                (0.3, 0.3, 0.3),
                position
            )
        
        else:
            # 默认：灰色立方体
            return (
                Marker.CUBE,
                (0.5, 0.5, 0.5),
                (0.2, 0.2, 0.2),
                position
            )

    # ============ 三模态融合 - 因果地图可视化方法 ============

    def _generate_causal_graph_markers(self) -> Optional[MarkerArray]:
        """生成因果图可视化标记（三模态融合 - 因果地图模态）

        使用箭头标记显示因果关系：
        - 箭头起点：原因节点
        - 箭头终点：效果节点
        - 颜色：绿色=高置信度，红色=低置信度
        - 粗细：置信度越高越粗

        Returns:
            MarkerArray containing causal graph visualization
        """
        if not hasattr(self.world_model, 'causal_graph'):
            return None

        # 获取因果图统计
        stats = self.world_model.causal_graph.get_statistics()
        if stats['num_edges'] == 0:
            # 没有因果关系，不发布markers
            return None

        marker_array = MarkerArray()
        marker_id = 0

        # 遍历因果边（降低置信度阈值到0.3，更容易看到）
        for (cause_id, effect_id), edge in self.world_model.causal_graph.edges.items():
            if edge.confidence < 0.3:  # 降低阈值，更容易显示
                continue

            # 获取节点位置
            cause_pos = self._get_causal_node_position(cause_id)
            effect_pos = self._get_causal_node_position(effect_id)

            if not cause_pos or not effect_pos:
                continue

            # 创建箭头标记
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "causal_graph"
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # 箭头起点和终点
            start = Point()
            start.x = float(cause_pos[0])
            start.y = float(cause_pos[1])
            start.z = 0.5

            end = Point()
            end.x = float(effect_pos[0])
            end.y = float(effect_pos[1])
            end.z = 0.5

            marker.points = [start, end]

            # 颜色：绿色=高置信度，红色=低置信度
            marker.color.r = 1.0 - edge.confidence
            marker.color.g = edge.confidence
            marker.color.b = 0.0
            marker.color.a = 0.8

            # 粗细：置信度越高越粗
            marker.scale.x = 0.05 * edge.confidence  # 轴直径
            marker.scale.y = 0.1 * edge.confidence   # 头部直径
            marker.scale.z = 0.15 * edge.confidence  # 头部长度

            # 持续时间
            marker.lifetime.sec = 2
            marker.lifetime.nanosec = 0

            marker_array.markers.append(marker)
            marker_id += 1

            # 添加文本标签显示关系类型
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.ns = "causal_graph_labels"
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            # 文本位置：箭头中点
            mid_x = (start.x + end.x) / 2
            mid_y = (start.y + end.y) / 2
            text_marker.pose.position.x = mid_x
            text_marker.pose.position.y = mid_y
            text_marker.pose.position.z = 0.7

            # 文本内容
            cause_node = self.world_model.causal_graph.nodes.get(cause_id)
            effect_node = self.world_model.causal_graph.nodes.get(effect_id)
            cause_label = cause_node.label if cause_node else cause_id
            effect_label = effect_node.label if effect_node else effect_id

            text_marker.text = f"{edge.relation_type.value}\n{edge.confidence:.0%}"
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8

            text_marker.scale.z = 0.2  # 文本高度

            text_marker.lifetime.sec = 2
            text_marker.lifetime.nanosec = 0

            marker_array.markers.append(text_marker)
            marker_id += 1

        return marker_array if len(marker_array.markers) > 0 else None

    def _get_causal_node_position(self, node_id: str) -> Optional[Tuple[float, float]]:
        """获取因果图节点的位置

        Args:
            node_id: 节点ID

        Returns:
            (x, y) 位置元组，如果未找到返回None
        """
        # 如果是robot节点，使用机器人位置
        if node_id == "robot":
            return (
                self.world_model.robot_position.get('x', 0),
                self.world_model.robot_position.get('y', 0)
            )

        # 尝试从语义物体获取位置
        if hasattr(self.world_model, 'semantic_objects'):
            if node_id in self.world_model.semantic_objects:
                obj = self.world_model.semantic_objects[node_id]
                return obj.world_position

        # 尝试从跟踪物体获取位置
        if hasattr(self.world_model, 'tracked_objects'):
            if node_id in self.world_model.tracked_objects:
                obj = self.world_model.tracked_objects[node_id]
                pos = obj.position
                return (pos.get('x', 0), pos.get('y', 0))

        # 尝试解析ID（格式如 object_123）
        if node_id.startswith("object_"):
            # 搜索所有语义物体，找到匹配的
            if hasattr(self.world_model, 'semantic_objects'):
                for obj_id, obj in self.world_model.semantic_objects.items():
                    if node_id in obj_id or obj_id in node_id:
                        return obj.world_position

        return None


def main(args=None):
    """主函数 - 用于独立运行可视化节点"""
    rclpy.init(args=args)

    # 注意：独立运行需要先创建WorldModel实例
    # 这里只是示例，实际使用时应该从外部传入WorldModel
    print("WorldModelVisualizer需要与WorldModel实例一起使用")
    print("请参见测试脚本: tests/cognitive/test_visualize_semantic_worldmodel.py")

    rclpy.shutdown()


if __name__ == '__main__':
    main()
