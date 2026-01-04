"""
世界模型可视化工具 - World Model Visualization

负责可视化 WorldModel 的状态，包括：
- 占据网格地图（2D热图）
- 机器人位置和轨迹
- 检测到的物体
- 探索区域
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class WorldModelVizConfig:
    """世界模型可视化配置"""
    show_occupancy_map: bool = True
    show_robot_pose: bool = True
    show_trajectory: bool = True
    show_detected_objects: bool = True
    show_exploration_areas: bool = False
    max_trajectory_points: int = 1000
    heatmap_cmap: str = "viridis"
    background_color: str = "black"


class WorldModelVisualizer:
    """
    世界模型可视化器
    
    提供 WorldModel 状态的 2D 可视化
    """
    
    def __init__(self, config: Optional[WorldModelVizConfig] = None):
        self.config = config or WorldModelVizConfig()
        
        # 创建图形
        self.fig = plt.figure(figsize=(14, 10))
        
        # 创建子图网格（2x2）
        # 左上：占据地图
        # 右上：机器人轨迹
        # 左下：检测物体
        # 右下：探索区域
        
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_traj = self.fig.add_subplot(gs[0, 1])
        self.ax_objects = self.fig.add_subplot(gs[1, 0])
        self.ax_exploration = self.fig.add_subplot(gs[1, 1])
        
        # 配置背景色
        self.fig.patch.set_facecolor(self.config.background_color)
        for ax in [self.ax_map, self.ax_traj, self.ax_objects, self.ax_exploration]:
            ax.set_facecolor(self.config.background_color)
        
        # 初始化数据缓存
        self.occupancy_grid = None
        self.robot_trajectory = []
        self.detected_objects = {}
        self.exploration_frontiers = []
        self.current_robot_pose = {"x": 0, "y": 0, "z": 0, "yaw": 0}
        
        # 自动更新控制
        self.auto_refresh = True
        self.last_refresh_time = datetime.now()
        
        logger.info("世界模型可视化器初始化完成")
    
    def update_occupancy_map(
        self, 
        grid: np.ndarray,
        resolution: float = 0.1,
        origin: Tuple[float, float] = (0.0, 0.0)
    ):
        """
        更新占据网格地图
        
        Args:
            grid: 占据网格 (H x W)
            resolution: 分辨率（米/格）
            origin: 原点 (x, y)
        """
        self.occupancy_grid = grid
        self.grid_resolution = resolution
        self.grid_origin = origin
        
        # 绘制占据地图
        self._draw_occupancy_map()
        
        if self.auto_refresh:
            self._auto_refresh_plot()
    
    def update_robot_pose(
        self,
        position: Dict[str, float],
        yaw: float,
        velocity: Optional[Dict[str, float]] = None
    ):
        """
        更新机器人位姿
        
        Args:
            position: 位置字典 {"x", "y", "z"}
            yaw: 偏航角（度）
            velocity: 速度字典 {"vx", "vy", "vz"}
        """
        self.current_robot_pose = {
            "position": position,
            "yaw": yaw,
            "velocity": velocity or {}
        }
        
        # 添加到轨迹
        self.robot_trajectory.append({
            "x": position.get("x", 0),
            "y": position.get("y", 0),
            "yaw": yaw,
            "timestamp": datetime.now()
        })
        
        # 限制轨迹长度
        if len(self.robot_trajectory) > self.config.max_trajectory_points:
            self.robot_trajectory = self.robot_trajectory[-self.config.max_trajectory_points:]
        
        # 绘制
        self._draw_robot_trajectory()
        self._draw_robot_current_pose()
        
        if self.auto_refresh:
            self._auto_refresh_plot()
    
    def update_detected_objects(self, objects: Dict[str, Any]):
        """
        更新检测到的物体
        
        Args:
            objects: 物体字典 {id: object_data}
        """
        self.detected_objects = objects
        
        # 绘制物体
        self._draw_detected_objects()
        
        if self.auto_refresh:
            self._auto_refresh_plot()
    
    def update_exploration_frontiers(self, frontiers: List[Dict[str, Any]]):
        """
        更新探索区域
        
        Args:
            frontiers: 探索边界列表
        """
        self.exploration_frontiers = frontiers
        
        # 绘制探索区域
        self._draw_exploration_areas()
        
        if self.auto_refresh:
            self._auto_refresh_plot()
    
    def _draw_occupancy_map(self):
        """绘制占据网格地图"""
        if self.occupancy_grid is None:
            self.ax_map.clear()
            self.ax_map.set_title("占据网格地图 (未初始化)")
            return
        
        # 清除之前的绘图
        self.ax_map.clear()
        
        height, width = self.occupancy_grid.shape
        
        # 使用热图显示占据概率
        # 0=未知(灰), 0-0.5=可能(黄), 0.5-1.0=占用(绿), 1=占用(红)
        im = self.ax_map.imshow(
            self.occupancy_grid,
            cmap=self.config.heatmap_cmap,
            origin='upper',
            extent=[
                self.grid_origin[0],
                self.grid_origin[0] + width * self.grid_resolution,
                self.grid_origin[1] + height * self.grid_resolution,
                self.grid_origin[1]
            ],
            vmin=0,
            vmax=1,
            alpha=0.8
        )
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=self.ax_map, fraction=0.046, pad=0.04)
        cbar.set_label('占据概率')
        
        # 设置标题和标签
        self.ax_map.set_title(f"占据网格地图 (分辨率: {self.grid_resolution}m)")
        self.ax_map.set_xlabel('X (米)')
        self.ax_map.set_ylabel('Y (米)')
    
    def _draw_robot_trajectory(self):
        """绘制机器人运动轨迹"""
        self.ax_traj.clear()
        
        if not self.config.show_trajectory or len(self.robot_trajectory) < 2:
            self.ax_traj.set_title("机器人轨迹 (无数据)")
            return
        
        # 提取轨迹点
        x = [p["x"] for p in self.robot_trajectory]
        y = [p["y"] for p in self.robot_trajectory]
        
        # 绘制轨迹（使用颜色编码时间）
        if len(x) > 0:
            # 创建时间数组用于颜色映射
            timestamps = [(p["timestamp"] - self.robot_trajectory[0]["timestamp"]).total_seconds() 
                         for p in self.robot_trajectory]
            max_time = max(timestamps) if timestamps else 1
            
            scatter = self.ax_traj.scatter(
                x, y,
                c=timestamps,
                cmap='cool',
                s=10,
                alpha=0.7
            )
            
            # 绘制当前位置
            current_pos = self.current_robot_pose.get("position", {})
            self.ax_traj.scatter(
                [current_pos.get("x", 0)],
                [current_pos.get("y", 0)],
                c=['red'],
                s=50,
                marker='o',
                edgecolors='white',
                linewidth=2
            )
            
            # 添加时间颜色条
            if len(x) > 1:
                sc_traj = self.ax_traj.scatter(x, y, c=timestamps, cmap='cool', s=0, alpha=0)
                cbar_traj = plt.colorbar(sc_traj, ax=self.ax_traj, fraction=0.046, pad=0.04)
                cbar_traj.set_label('时间 (秒)')
        
        # 设置标题
        self.ax_traj.set_title(f"机器人轨迹 (点数: {len(self.robot_trajectory)})")
        self.ax_traj.set_xlabel('X (米)')
        self.ax_traj.set_ylabel('Y (米)')
        self.ax_traj.grid(True, alpha=0.3)
        
        # 设置坐标范围
        if self.occupancy_grid is not None:
            self.ax_traj.set_xlim(
                self.grid_origin[0],
                self.grid_origin[0] + self.occupancy_grid.shape[1] * self.grid_resolution
            )
            self.ax_traj.set_ylim(
                self.grid_origin[1],
                self.grid_origin[1] + self.occupancy_grid.shape[0] * self.grid_resolution
            )
    
    def _draw_robot_current_pose(self):
        """绘制机器人当前位姿"""
        if not self.config.show_robot_pose:
            return
        
        pos = self.current_robot_pose.get("position", {})
        yaw = self.current_robot_pose.get("yaw", 0)
        
        # 绘制箭头表示朝向
        arrow_length = 2.0
        arrow_dx = arrow_length * np.cos(np.radians(yaw))
        arrow_dy = arrow_length * np.sin(np.radians(yaw))
        
        self.ax_traj.arrow(
            pos.get("x", 0),
            pos.get("y", 0),
            arrow_dx,
            arrow_dy,
            head_width=0.4,
            head_length=0.6,
            fc='red',
            ec='white',
            linewidth=2
        )
    
    def _draw_detected_objects(self):
        """绘制检测到的物体"""
        self.ax_objects.clear()
        
        if not self.config.show_detected_objects or not self.detected_objects:
            self.ax_objects.set_title("检测到的物体 (无数据)")
            return
        
        # 按物体类型分类显示
        obstacle_objs = []
        target_objs = []
        other_objs = []
        
        for obj_id, obj in self.detected_objects.items():
            obj_type = obj.get("type", "unknown").lower()
            pos = obj.get("position", {"x": 0, "y": 0})
            
            if "obstacle" in obj_type or "wall" in obj_type:
                obstacle_objs.append((obj_id, obj, pos))
            elif "target" in obj_type:
                target_objs.append((obj_id, obj, pos))
            else:
                other_objs.append((obj_id, obj, pos))
        
        # 绘制障碍物（红色方块）
        if obstacle_objs:
            ox, oy = zip(*[(o[2]["x"], o[2]["y"]) for o in obstacle_objs])
            self.ax_objects.scatter(
                ox, oy,
                c='red',
                s=30,
                marker='s',
                edgecolors='white',
                label='障碍物'
            )
        
        # 绘制目标（绿色圆圈）
        if target_objs:
            tx, ty = zip(*[(t[2]["x"], t[2]["y"]) for t in target_objs])
            self.ax_objects.scatter(
                tx, ty,
                c='green',
                s=40,
                marker='o',
                edgecolors='white',
                linewidth=2,
                label='目标'
            )
        
        # 绘制其他物体（黄色三角形）
        if other_objs:
            ox, oy = zip(*[(o[2]["x"], o[2]["y"]) for o in other_objs])
            self.ax_objects.scatter(
                ox, oy,
                c='yellow',
                s=20,
                marker='^',
                edgecolors='white',
                linewidth=2,
                label='其他'
            )
        
        # 添加图例
        if obstacle_objs or target_objs or other_objs:
            self.ax_objects.legend()
            self.ax_objects.grid(True, alpha=0.3)
        
        # 设置标题
        self.ax_objects.set_title(f"检测到的物体 (总计: {len(self.detected_objects)})")
        
        # 设置坐标范围
        if self.occupancy_grid is not None:
            self.ax_objects.set_xlim(
                self.grid_origin[0],
                self.grid_origin[0] + self.occupancy_grid.shape[1] * self.grid_resolution
            )
            self.ax_objects.set_ylim(
                self.grid_origin[1],
                self.grid_origin[1] + self.occupancy_grid.shape[0] * self.grid_resolution
            )
    
    def _draw_exploration_areas(self):
        """绘制探索区域"""
        self.ax_exploration.clear()
        
        if not self.config.show_exploration_areas or not self.exploration_frontiers:
            self.ax_exploration.set_title("探索区域 (无数据)")
            return
        
        # 绘制探索边界（蓝色多边形）
        for frontier in self.exploration_frontiers:
            pos = frontier.get("position", {"x": 0, "y": 0})
            self.ax_exploration.add_patch(
                patches.Polygon(
                    [(pos["x"] - 1, pos["y"] - 1),
                     (pos["x"] + 1, pos["y"] - 1),
                     (pos["x"] + 1, pos["y"] + 1)],
                    closed=True,
                    fill=True,
                    alpha=0.3,
                    edgecolor='blue',
                    facecolor='lightblue',
                    linewidth=2
                )
            )
        
        self.ax_exploration.set_title(f"探索区域 (边界数: {len(self.exploration_frontiers)})")
        
        # 设置坐标范围
        if self.occupancy_grid is not None:
            self.ax_exploration.set_xlim(
                self.grid_origin[0],
                self.grid_origin[0] + self.occupancy_grid.shape[1] * self.grid_resolution
            )
            self.ax_exploration.set_ylim(
                self.grid_origin[1],
                self.grid_origin[1] + self.occupancy_grid.shape[0] * self.grid_resolution
            )
        
        self.ax_exploration.grid(True, alpha=0.3)
        self.ax_exploration.set_xlabel('X (米)')
        self.ax_exploration.set_ylabel('Y (米)')
    
    def _auto_refresh_plot(self):
        """自动刷新显示"""
        now = datetime.now()
        if (now - self.last_refresh_time).total_seconds() < 0.1:
            return
        
        self.last_refresh_time = now
        self.fig.canvas.draw()
    
    def save_screenshot(self, filename: Optional[str] = None):
        """
        保存可视化截图
        
        Args:
            filename: 文件名，默认为带时间戳的文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"world_model_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"世界模型可视化截图已保存: {filename}")
        
        return filename
    
    def show(self):
        """显示可视化窗口"""
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """关闭可视化器"""
        plt.close(self.fig)
        logger.info("世界模型可视化器已关闭")


# 便捷函数
def create_world_model_viz(config: Optional[WorldModelVizConfig] = None) -> WorldModelVisualizer:
    """
    创建世界模型可视化器
    
    Args:
        config: 可视化配置
        
    Returns:
        WorldModelVisualizer: 可视化器实例
    """
    return WorldModelVisualizer(config)


# 使用示例
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Qt5Agg')
    
    # 创建可视化器
    viz = create_world_model_viz(WorldModelVizConfig())
    
    # 生成示例占据地图
    grid_size = 100
    resolution = 0.2
    
    # 创建示例占据地图（中心区域占用，四周空闲）
    example_grid = np.full((grid_size, grid_size), 0.1, dtype=np.float32)
    center = grid_size // 2
    
    # 添加一些障碍物（红色区域，值=1.0）
    example_grid[center-5:center+5, center-5:center+5] = 1.0
    example_grid[center-3:center+3, center-3:center+3] = 0.8
    example_grid[center-2:center+2, center-2:center+2] = 0.6
    example_grid[center-1:center+1, center-1:center+1] = 0.4
    example_grid[center:center+1, center:center+1] = 0.2
    example_grid[center:center+2, center:center+2] = 0.6
    
    # 更新占据地图
    viz.update_occupancy_map(
        grid=example_grid,
        resolution=resolution,
        origin=(-grid_size * resolution / 2, -grid_size * resolution / 2)
    )
    
    # 添加示例机器人轨迹
    for i in range(20):
        angle = i * 36 / 20
        x = 5 * np.cos(np.radians(angle))
        y = 5 * np.sin(np.radians(angle))
        viz.update_robot_pose(
            position={"x": x, "y": y},
            yaw=angle
        )
    
    # 添加示例物体
    viz.update_detected_objects({
        "obstacle_1": {"type": "obstacle", "position": {"x": 8, "y": 3}},
        "obstacle_2": {"type": "obstacle", "position": {"x": 2, "y": 8}},
        "target_1": {"type": "target", "position": {"x": -5, "y": 5}},
        "wall_1": {"type": "wall", "position": {"x": 0, "y": -8}},
        "wall_2": {"type": "wall", "position": {"x": 10, "y": 0}}
    })
    
    print("世界模型可视化器已启动")
    print("控制方式:")
    print("  h : 显示/隐藏占据地图")
    print("  t : 显示/隐藏机器人轨迹")
    print("  o : 显示/隐藏检测物体")
    print("  e : 显示/隐藏探索区域")
    print("  r : 刷新显示")
    print("  s : 保存截图")
    print("  q : 退出")
    
    # 键盘控制
    from matplotlib.widgets import Button
    
    def on_key(event):
        if event.key == 'h':
            viz.config.show_occupancy_map = not viz.config.show_occupancy_map
            if viz.config.show_occupancy_map:
                viz.update_occupancy_map(
                    grid=viz.occupancy_grid,
                    resolution=viz.grid_resolution,
                    origin=viz.grid_origin
                )
        elif event.key == 't':
            viz.config.show_trajectory = not viz.config.show_trajectory
            viz._draw_robot_trajectory()
        elif event.key == 'o':
            viz.config.show_detected_objects = not viz.config.show_detected_objects
            viz._draw_detected_objects()
        elif event.key == 'e':
            viz.config.show_exploration_areas = not viz.config.show_exploration_areas
            viz._draw_exploration_areas()
        elif event.key == 'r':
            viz.fig.canvas.draw()
        elif event.key == 's':
            viz.save_screenshot()
        elif event.key == 'q':
            plt.close(viz.fig)
            print("退出可视化器")
            return
        
        # 刷新显示
        viz.fig.canvas.draw()
    
    viz.fig.canvas.mpl_connect('key_press_event', on_key)
    viz.show()








