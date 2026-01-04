#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerceptionData 可视化工具
支持实时可视化感知数据，特别是占据地图
"""

import os
import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from typing import Optional
from pathlib import Path
import sys
import warnings

# 设置ROS2域ID（必须在导入ROS2相关模块之前设置）
if 'ROS_DOMAIN_ID' not in os.environ:
    os.environ['ROS_DOMAIN_ID'] = '42'
    print(f"Set ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}")
else:
    print(f"Using ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}")

# 忽略matplotlib字体警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')

# 设置matplotlib使用DejaVu Sans字体（支持更多字符）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brain.core.brain import Brain


class PerceptionDataVisualizer:
    """PerceptionData可视化器"""
    
    def __init__(self, figsize=(16, 10), vlm_enabled=False, sensor_manager=None):
        self.fig = plt.figure(figsize=figsize)
        self.fig.suptitle('PerceptionData Real-time Visualization', fontsize=16, fontweight='bold')
        
        # 创建子图布局（固定大小，避免动态变化）
        # 使用GridSpec来固定子图大小
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 4, figure=self.fig, 
                     width_ratios=[1, 1, 1, 1], 
                     height_ratios=[1, 1],
                     hspace=0.3, wspace=0.3)
        
        self.ax_rgb_left = self.fig.add_subplot(gs[0, 0])  # 左眼RGB
        self.ax_rgb_right = self.fig.add_subplot(gs[0, 1])  # 右眼RGB
        self.ax_lidar = self.fig.add_subplot(gs[0, 2])
        self.ax_pose = self.fig.add_subplot(gs[0, 3])
        self.ax_occupancy = self.fig.add_subplot(gs[1, :2])  # 占据地图（大图，占两列）
        self.ax_obstacles = self.fig.add_subplot(gs[1, 2])
        self.ax_semantic = self.fig.add_subplot(gs[1, 3])
        
        # 存储colorbar引用，避免重复创建
        self.occupancy_cbar = None
        
        # VLM状态和sensor_manager引用
        self._vlm_enabled = vlm_enabled
        self._sensor_manager = sensor_manager  # 用于检查VLM服务状态
        
        plt.ion()  # 开启交互模式
        plt.tight_layout()
    
    def visualize(self, perception_data, frame_count: int = 0):
        """可视化PerceptionData"""
        # #region agent log
        import json
        visualize_start = time.time()
        rgb_right_check = getattr(perception_data, 'rgb_image_right', None)
        rgb_right_shape = list(rgb_right_check.shape) if rgb_right_check is not None and hasattr(rgb_right_check, 'shape') else None
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"visualize_perception.py:visualize:72","message":"visualize entry","data":{"frame_count":frame_count,"rgb_right_exists":rgb_right_check is not None,"rgb_right_shape":rgb_right_shape,"hasattr_rgb_right":hasattr(perception_data, 'rgb_image_right')},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        
        self.fig.suptitle(f'PerceptionData Visualization - Frame #{frame_count}', 
                         fontsize=16, fontweight='bold')
        
        # 1. 左眼RGB图像
        draw_left_start = time.time()
        self._draw_rgb_image(perception_data.rgb_image, self.ax_rgb_left, "Left RGB")
        draw_left_duration = (time.time() - draw_left_start) * 1000
        # #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"G","location":"visualize_perception.py:visualize:79","message":"draw_left_rgb duration","data":{"duration_ms":draw_left_duration},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        
        # 2. 右眼RGB图像
        draw_right_start = time.time()
        rgb_right = getattr(perception_data, 'rgb_image_right', None)
        if rgb_right is not None and rgb_right.size > 0:
            self._draw_rgb_image(rgb_right, self.ax_rgb_right, "Right RGB")
        else:
            # 显示调试信息
            debug_info = f"No Right RGB Data"
            if hasattr(perception_data, 'rgb_image_right'):
                debug_info += f"\n(Attribute exists but is None or empty)"
            else:
                debug_info += f"\n(Attribute not found)"
            self.ax_rgb_right.clear()
            self.ax_rgb_right.text(0.5, 0.5, debug_info,
                                   ha='center', va='center', transform=self.ax_rgb_right.transAxes,
                                   fontsize=10, color='orange')
            self.ax_rgb_right.set_title("Right RGB", fontsize=8)
            self.ax_rgb_right.axis('off')
        draw_right_duration = (time.time() - draw_right_start) * 1000
        # #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"visualize_perception.py:visualize:97","message":"draw_right_rgb result","data":{"rgb_right_was_none":rgb_right is None,"rgb_right_size":rgb_right.size if rgb_right is not None else 0,"duration_ms":draw_right_duration},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
        
        # 3. 激光雷达数据
        self._draw_lidar(perception_data.laser_ranges, perception_data.laser_angles)
        
        # 4. 机器人位姿
        self._draw_pose(perception_data.pose, perception_data.velocity)
        
        # 5. 占据栅格地图（重点 - 支持全局地图）
        self._draw_occupancy_grid(
            perception_data.occupancy_grid,
            perception_data.grid_resolution,
            perception_data.grid_origin,
            perception_data.pose,
            perception_data  # 传递perception_data以支持全局地图显示
        )
        
        # 6. 障碍物
        self._draw_obstacles(perception_data.obstacles, perception_data.pose, perception_data)
        
        # 7. 语义信息
        self._draw_semantic_info(perception_data)
        
        # 使用固定布局，不每次调用tight_layout
        # plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)
        draw_start = time.time()
        plt.draw()
        draw_duration = (time.time() - draw_start) * 1000
        pause_start = time.time()
        plt.pause(0.05)  # 增加暂停时间以改善显示
        pause_duration = (time.time() - pause_start) * 1000
        visualize_total = (time.time() - visualize_start) * 1000
        # #region agent log
        with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"visualize_perception.py:visualize:123","message":"plt.draw and pause duration","data":{"draw_duration_ms":draw_duration,"pause_duration_ms":pause_duration,"visualize_total_ms":visualize_total},"timestamp":int(time.time()*1000)})+'\n')
        # #endregion
    
    def _draw_rgb_image(self, rgb_image: Optional[np.ndarray], ax, title: str = "RGB Image"):
        """绘制RGB图像"""
        ax.clear()
        
        if rgb_image is not None and rgb_image.size > 0 and len(rgb_image.shape) == 3:
            # 直接显示图像，不做多余的"全黑"检测
            # 图像处理已在ros2_interface中完成，这里只负责显示
            # 确保数据类型正确
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)
            
            # 直接显示图像，让matplotlib自动设置坐标轴范围
            ax.imshow(rgb_image, aspect='auto', interpolation='nearest')
            img_min, img_max = rgb_image.min(), rgb_image.max()
            ax.set_title(f'{title}\nShape: {rgb_image.shape}\nRange: [{img_min}, {img_max}]', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No {title}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
    
    def _draw_lidar(self, ranges: Optional[list], angles: Optional[list]):
        """绘制激光雷达数据"""
        self.ax_lidar.clear()
        if ranges and angles and len(ranges) == len(angles):
            x = np.array(ranges) * np.cos(np.array(angles))
            y = np.array(ranges) * np.sin(np.array(angles))
            self.ax_lidar.plot(x, y, 'r.', markersize=2, alpha=0.6)
            self.ax_lidar.plot(0, 0, 'go', markersize=10, label='机器人')
            self.ax_lidar.set_title(f'Lidar\n{len(ranges)} points')
            self.ax_lidar.set_xlabel('X (m)')
            self.ax_lidar.set_ylabel('Y (m)')
            self.ax_lidar.axis('equal')
            self.ax_lidar.grid(True, alpha=0.3)
            self.ax_lidar.legend()
        else:
            self.ax_lidar.text(0.5, 0.5, 'No Lidar Data',
                             ha='center', va='center', transform=self.ax_lidar.transAxes)
    
    def _draw_pose(self, pose, velocity):
        """绘制机器人位姿"""
        self.ax_pose.clear()
        # 固定坐标轴范围
        self.ax_pose.set_xlim(-5, 5)
        self.ax_pose.set_ylim(-5, 5)
        
        if pose:
            # 绘制位置
            self.ax_pose.scatter(pose.x, pose.y, s=200, c='red', marker='o', 
                               label='Robot Position', zorder=3)
            
            # 绘制朝向箭头
            arrow_length = 0.5
            dx = arrow_length * np.cos(pose.yaw)
            dy = arrow_length * np.sin(pose.yaw)
            self.ax_pose.arrow(pose.x, pose.y, dx, dy,
                             head_width=0.1, head_length=0.1,
                             fc='red', ec='red', zorder=2)
            
            # 显示信息
            info = f'Position: ({pose.x:.2f}, {pose.y:.2f})\n'
            info += f'Heading: {np.degrees(pose.yaw):.1f} deg'
            if velocity:
                info += f'\nVelocity: {velocity.linear_x:.2f} m/s'
            self.ax_pose.text(0.05, 0.95, info, transform=self.ax_pose.transAxes,
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            self.ax_pose.set_title('Robot Pose', fontsize=9)
            self.ax_pose.set_xlabel('X (m)', fontsize=8)
            self.ax_pose.set_ylabel('Y (m)', fontsize=8)
            self.ax_pose.axis('equal')
            self.ax_pose.grid(True, alpha=0.3)
            self.ax_pose.legend(fontsize=7)
        else:
            self.ax_pose.text(0.5, 0.5, 'No Pose Data',
                             ha='center', va='center', transform=self.ax_pose.transAxes, fontsize=10)
    
    def _draw_occupancy_grid(self, grid: Optional[np.ndarray], 
                           resolution: float, origin: tuple, pose, perception_data=None):
        """绘制占据栅格地图（改进版 - 支持全局地图）"""
        self.ax_occupancy.clear()
        
        # 优先显示全局地图，否则显示局部占据栅格
        if perception_data is not None and hasattr(perception_data, 'global_map') and perception_data.global_map is not None:
            grid = perception_data.global_map
            map_type = "Global"
        elif grid is not None:
            map_type = "Local"
        else:
            self.ax_occupancy.text(0.5, 0.5, 'Occupancy Map Not Generated',
                                 ha='center', va='center',
                                 transform=self.ax_occupancy.transAxes)
            self.ax_occupancy.set_title('Occupancy Grid Map')
            return
        
        # 检查grid的形状和数据类型
        if len(grid.shape) != 2:
            self.ax_occupancy.text(0.5, 0.5, f'Invalid grid shape: {grid.shape}',
                                 ha='center', va='center',
                                 transform=self.ax_occupancy.transAxes)
            return
        
        # 检查grid的数据类型和值范围
        unique_values = np.unique(grid)
        min_val, max_val = grid.min(), grid.max()
        
        # 创建自定义颜色映射
        # -1=未知(灰色), 0=自由(白色), 50=未知(浅灰), 100=占据(黑色)
        colors = ['lightgray', 'white', 'lightgray', 'black']
        cmap = mcolors.ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 50, 100.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 计算地图范围（世界坐标）
        # grid.shape是(height, width)，即(行数, 列数)
        height, width = grid.shape
        
        # origin是(grid_origin_x, grid_origin_y)
        origin_x, origin_y = origin[0], origin[1]
        
        # 计算世界坐标范围
        # 注意：imshow的extent是[left, right, bottom, top]
        # 对于origin='lower'，y轴从下往上
        # 栅格坐标(0,0)对应世界坐标(origin_x, origin_y)
        # 栅格坐标(width-1, height-1)对应世界坐标(origin_x + (width-1)*resolution, origin_y + (height-1)*resolution)
        x_min = origin_x
        x_max = origin_x + (width - 1) * resolution + resolution  # 包含最后一个栅格
        y_min = origin_y
        y_max = origin_y + (height - 1) * resolution + resolution  # 包含最后一个栅格
        
        # 绘制地图
        # 注意：grid.data的索引是[gy, gx]，即[行, 列]
        # imshow期望的是[行, 列]，所以直接使用grid即可
        im = self.ax_occupancy.imshow(
            grid,
            cmap=cmap,
            norm=norm,
            origin='lower',  # 使用'lower'：y轴从下往上，匹配标准坐标系
            extent=[x_min, x_max, y_min, y_max],  # [left, right, bottom, top]
            interpolation='nearest',
            alpha=0.8,
            aspect='equal'  # 保持纵横比
        )
        
        # 绘制机器人位置（如果已知）
        if pose:
            # 检查机器人位置是否在地图范围内
            if x_min <= pose.x <= x_max and y_min <= pose.y <= y_max:
                self.ax_occupancy.plot(pose.x, pose.y, 'ro', markersize=10, 
                                     label='Robot', zorder=10)
                # 绘制朝向
                arrow_length = 0.3
                dx = arrow_length * np.cos(pose.yaw)
                dy = arrow_length * np.sin(pose.yaw)
                self.ax_occupancy.arrow(pose.x, pose.y, dx, dy,
                                      head_width=0.15, head_length=0.1,
                                      fc='red', ec='red', zorder=10)
            else:
                # 机器人位置超出地图范围，显示警告
                self.ax_occupancy.text(0.02, 0.98, 
                                     f'Robot outside map\nPose: ({pose.x:.2f}, {pose.y:.2f})',
                                     transform=self.ax_occupancy.transAxes,
                                     verticalalignment='top',
                                     fontsize=8,
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 添加或更新颜色条（避免重复创建）
        if self.occupancy_cbar is None:
            self.occupancy_cbar = plt.colorbar(im, ax=self.ax_occupancy, fraction=0.046, pad=0.04)
            self.occupancy_cbar.set_label('Occupancy State', rotation=270, labelpad=15)
            self.occupancy_cbar.set_ticks([-1, 0, 50, 100])
            self.occupancy_cbar.set_ticklabels(['Unknown', 'Free', 'Unknown', 'Occupied'])
        else:
            # 更新现有colorbar
            self.occupancy_cbar.update_normal(im)
        
        # 计算统计信息
        occupied_cells = np.sum(grid == 100)
        free_cells = np.sum(grid == 0)
        unknown_cells = np.sum(grid == -1)
        total_cells = width * height
        
        # 设置标题和标签（包含调试信息和地图类型）
        title = f'{map_type} Occupancy Map\n'
        title += f'Resolution: {resolution:.3f}m, Grid: {width}x{height}\n'
        title += f'Occupied: {occupied_cells} | Free: {free_cells} | Unknown: {unknown_cells}\n'
        title += f'Black=Occupied, White=Free, Gray=Unknown'
        self.ax_occupancy.set_title(title, fontsize=9)
        self.ax_occupancy.set_xlabel('X (m)')
        self.ax_occupancy.set_ylabel('Y (m)')
        self.ax_occupancy.grid(True, alpha=0.3, linestyle='--')
        if pose and (x_min <= pose.x <= x_max and y_min <= pose.y <= y_max):
            self.ax_occupancy.legend(loc='upper right')
        self.ax_occupancy.set_aspect('equal', adjustable='box')
    
    def _draw_obstacles(self, obstacles: list, pose, perception_data=None):
        """绘制障碍物"""
        self.ax_obstacles.clear()
        # 固定坐标轴范围
        self.ax_obstacles.set_xlim(-10, 10)
        self.ax_obstacles.set_ylim(-10, 10)
        
        if not obstacles:
            # 检查是否有激光雷达数据
            if perception_data and hasattr(perception_data, 'laser_ranges') and perception_data.laser_ranges:
                self.ax_obstacles.text(0.5, 0.5, 
                    f'No Obstacles Detected\nLidar: {len(perception_data.laser_ranges)} points',
                    ha='center', va='center', transform=self.ax_obstacles.transAxes)
            else:
                self.ax_obstacles.text(0.5, 0.5, 
                    'No Lidar Data\nCannot detect obstacles',
                    ha='center', va='center', transform=self.ax_obstacles.transAxes)
            self.ax_obstacles.set_title('Obstacles')
            return
        
        # 绘制机器人位置
        if pose:
            self.ax_obstacles.plot(0, 0, 'go', markersize=10, label='Robot')
        
        # 绘制障碍物
        for obs in obstacles:
            if 'local_position' in obs:
                x = obs['local_position']['x']
                y = obs['local_position']['y']
                size = obs.get('size', 0.5)
                distance = obs.get('distance', 0)
                
                # 绘制障碍物圆圈
                circle = Circle((x, y), size/2, color='red', alpha=0.5, 
                              label='Obstacle' if obs == obstacles[0] else '')
                self.ax_obstacles.add_patch(circle)
                
                # 添加标签
                self.ax_obstacles.text(x, y, f"{obs.get('id', '?')}\n{distance:.1f}m",
                                     ha='center', va='center', fontsize=7)
        
        self.ax_obstacles.set_title(f'Obstacles ({len(obstacles)})', fontsize=9)
        self.ax_obstacles.set_xlabel('X (m)', fontsize=8)
        self.ax_obstacles.set_ylabel('Y (m)', fontsize=8)
        self.ax_obstacles.axis('equal')
        self.ax_obstacles.grid(True, alpha=0.3)
        self.ax_obstacles.legend(fontsize=7)
        # 坐标轴范围已在函数开头设置
    
    def _draw_semantic_info(self, perception_data):
        """绘制语义信息"""
        self.ax_semantic.clear()
        self.ax_semantic.axis('off')
        
        info_lines = []
        
        # 语义物体
        if perception_data.semantic_objects:
            info_lines.append(f"Semantic Objects: {len(perception_data.semantic_objects)}")
            for i, obj in enumerate(perception_data.semantic_objects[:5]):  # 最多显示5个
                label = obj.label if hasattr(obj, 'label') else str(obj)
                confidence = obj.confidence if hasattr(obj, 'confidence') else 0.0
                distance = ""
                if hasattr(obj, 'estimated_distance') and obj.estimated_distance:
                    distance = f" @ {obj.estimated_distance:.1f}m"
                info_lines.append(f"  • {label} ({confidence:.2f}){distance}")
            if len(perception_data.semantic_objects) > 5:
                info_lines.append(f"  ... {len(perception_data.semantic_objects) - 5} more")
        else:
            info_lines.append("Semantic Objects: None")
            info_lines.append("(VLM may not be running or")
            info_lines.append(" no objects detected yet)")
        
        # 数据接收状态
        info_lines.append(f"\nData Status:")
        info_lines.append(f"  RGB: {'✓' if perception_data.rgb_image is not None else '✗'}")
        rgb_right = getattr(perception_data, 'rgb_image_right', None)
        if rgb_right is not None:
            rgb_right_min = rgb_right.min() if rgb_right.size > 0 else 0
            rgb_right_max = rgb_right.max() if rgb_right.size > 0 else 0
            info_lines.append(f"  RGB Right: ✓ (Range: [{rgb_right_min}, {rgb_right_max}])")
        else:
            info_lines.append(f"  RGB Right: ✗ (No data)")
        info_lines.append(f"  Lidar: {'✓' if perception_data.laser_ranges else '✗'}")
        info_lines.append(f"  PointCloud: {'✓' if perception_data.pointcloud is not None else '✗'}")
        info_lines.append(f"  Pose: {'✓' if perception_data.pose else '✗'}")
        
        # VLM状态和全局地图信息
        # 检查VLM是否真正可用（优先检查sensor_manager的_vlm_service）
        vlm_available = False
        if self._sensor_manager and hasattr(self._sensor_manager, '_vlm_service'):
            vlm_available = self._sensor_manager._vlm_service is not None
        elif hasattr(self, '_vlm_enabled') and self._vlm_enabled:
            vlm_available = True
        elif hasattr(perception_data, 'semantic_objects') and perception_data.semantic_objects:
            vlm_available = True
        elif hasattr(perception_data, 'scene_description') and perception_data.scene_description:
            vlm_available = True
        
        vlm_status = "Enabled" if vlm_available else "Disabled"
        info_lines.append(f"\nVLM Status: {vlm_status}")
        if vlm_available:
            if perception_data.rgb_image is None:
                info_lines.append("  Waiting for RGB image...")
            else:
                info_lines.append("  Ready to analyze")
            
            # 显示全局地图信息
            if hasattr(perception_data, 'global_map') and perception_data.global_map is not None:
                info_lines.append(f"\nGlobal Map: ✓")
                if hasattr(perception_data, 'world_metadata') and perception_data.world_metadata:
                    meta = perception_data.world_metadata
                    if 'update_count' in meta:
                        info_lines.append(f"  Updates: {meta['update_count']}")
                    if 'confidence' in meta:
                        info_lines.append(f"  Confidence: {meta['confidence']:.2f}")
            else:
                info_lines.append(f"\nGlobal Map: ✗ (Using Local)")
        
        # 场景描述
        if perception_data.scene_description:
            summary = perception_data.scene_description.summary
            if len(summary) > 50:
                summary = summary[:50] + "..."
            info_lines.append(f"\nScene: {summary}")
        
        # 导航提示
        if perception_data.navigation_hints:
            info_lines.append(f"\nNavigation Hints:")
            for hint in perception_data.navigation_hints[:3]:
                info_lines.append(f"  • {hint}")
        
        # 空间关系
        if perception_data.spatial_relations:
            info_lines.append(f"\nSpatial Relations: {len(perception_data.spatial_relations)}")
        
        if not info_lines:
            info_lines.append("No Semantic Info")
        
        self.ax_semantic.text(0.05, 0.95, '\n'.join(info_lines),
                            transform=self.ax_semantic.transAxes,
                            verticalalignment='top',
                            fontsize=9,
                            family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        self.ax_semantic.set_title('Semantic Info')


async def visualize_perception_loop(brain: Brain, update_interval: float = 0.5):
    """持续可视化感知数据"""
    # 检查VLM是否启用（检查_vlm_service而非vlm）
    vlm_enabled = (
        hasattr(brain.sensor_manager, '_vlm_service') and 
        brain.sensor_manager._vlm_service is not None
    )
    visualizer = PerceptionDataVisualizer(vlm_enabled=vlm_enabled, sensor_manager=brain.sensor_manager)
    frame_count = 0
    no_data_count = 0
    
    try:
        while True:
            # #region agent log
            import json
            loop_iter_start = time.time()
            # #endregion
            
            # 获取感知数据（参考L2测试：多次调用以确保VLM分析完成）
            # 第一次调用可能触发VLM分析
            get_data_start = time.time()
            perception_data = await brain.sensor_manager.get_fused_perception()
            get_data_duration = (time.time() - get_data_start) * 1000
            # #region agent log
            with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"visualize_perception.py:visualize_perception_loop:486","message":"get_fused_perception duration","data":{"duration_ms":get_data_duration},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
            
            if perception_data is None:
                no_data_count += 1
                if no_data_count > 10:
                    logger.warning("连续10次未获取到感知数据，可能存在问题")
                await asyncio.sleep(update_interval)
                continue
            
            no_data_count = 0
            frame_count += 1
            
            # 可视化数据
            visualize_start = time.time()
            visualizer.visualize(perception_data, frame_count)
            visualize_duration = (time.time() - visualize_start) * 1000
            
            # #region agent log
            loop_iter_duration = (time.time() - loop_iter_start) * 1000
            with open('/media/yangyuhui/CODES1/Brain/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"visualize_perception.py:visualize_perception_loop:505","message":"full loop iteration duration","data":{"frame_count":frame_count,"get_data_ms":get_data_duration,"visualize_ms":visualize_duration,"total_ms":loop_iter_duration},"timestamp":int(time.time()*1000)})+'\n')
            # #endregion
            
            # 安全地获取位姿信息
            pose_str = "N/A"
            if perception_data.pose:
                pose_str = f"{perception_data.pose.x:.2f}, {perception_data.pose.y:.2f}"
            
            # 显示数据状态
            data_status = []
            if perception_data.pose:
                data_status.append("Pose")
            if perception_data.rgb_image is not None:
                data_status.append("RGB")
            if perception_data.pointcloud is not None:
                data_status.append("PointCloud")
            if perception_data.laser_ranges:
                data_status.append("Lidar")
            if perception_data.obstacles:
                data_status.append(f"{len(perception_data.obstacles)}Obs")
            
            print(f"\rFrame #{frame_count} - "
                  f"Pose: {pose_str} | "
                  f"Data: {', '.join(data_status)} | "
                  f"Semantic: {len(perception_data.semantic_objects)}", end='')
            
            # 控制更新频率（30fps = 33.3ms per frame）
            target_frame_time = 1.0 / 30.0  # 30fps
            elapsed = time.time() - loop_iter_start
            sleep_time = max(0, target_frame_time - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n\nVisualization stopped")
    finally:
        plt.ioff()
        plt.close('all')


async def main():
    """主函数"""
    config_path = "config/environments/isaac_sim/nova_carter.yaml"
    
    print("Initializing Brain...")
    brain = Brain(config_path=config_path)
    
    # 初始化ROS2接口
    print("Initializing ROS2 interface...")
    await brain.ros2.initialize()
    
    # 等待传感器数据
    print("Waiting for sensor data...")
    print("This may take a few seconds...")
    
    sensor_ready = await brain.sensor_manager.wait_for_sensors(timeout=15.0)
    
    if not sensor_ready:
        print("WARNING: Sensors not ready, but continuing anyway...")
        print("Make sure ROS2 topics are publishing data:")
        print("  - /chassis/odom (odometry)")
        print("  - /front_3d_lidar/lidar_points (pointcloud)")
        print("  - /front_stereo_camera/left/image_raw (RGB image)")
        print("\nWaiting 3 more seconds for data collection...")
        await asyncio.sleep(3.0)
    else:
        print("Sensors ready!")
    
    # 显示传感器状态
    sensor_health = brain.sensor_manager.get_sensor_health()
    print("\nSensor Status:")
    for sensor_name, is_healthy in sensor_health.items():
        status = "✓" if is_healthy else "✗"
        print(f"  {status} {sensor_name}: {'Healthy' if is_healthy else 'Not Ready'}")
    
    print("\nStarting perception data visualization...")
    print("Press Ctrl+C to stop\n")
    
    await visualize_perception_loop(brain, update_interval=0.5)


if __name__ == "__main__":
    asyncio.run(main())

