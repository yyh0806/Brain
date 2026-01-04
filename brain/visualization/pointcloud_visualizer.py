"""
点云可视化器 - Point Cloud Visualizer

负责实时显示 Nova Carter 的 3D 激光雷达点云数据
功能：
- 实时显示点云（3D 散点图）
- 支持多种颜色编码（高度、距离、强度）
- 交互式旋转和缩放
- 保存点云截图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from mpl_toolkits.mplot3d import Axes3D
import threading
import os


@dataclass
class PointCloudVizState:
    """点云可视化状态"""
    points: np.ndarray = None
    colors: np.ndarray = None
    view_elevation: float = 30.0  # 仰角（度）
    view_azimuth: float = 45.0  # 方位角（度）
    zoom: float = 1.0  # 缩放因子
    color_mode: str = "height"  # height, distance, intensity
    auto_rotate: bool = False
    rotation_speed: float = 1.0  # 旋转速度（度/秒）


class PointCloudVisualizer:
    """
    3D 点云可视化器
    
    使用 matplotlib 3D 散点图实时显示点云
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 窗口和画布
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 可视化状态
        self.state = PointCloudVizState()
        
        # 保存路径
        self.save_dir = self.config.get("save_dir", "data/pointcloud")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 动画控制
        self.animation_running = False
        self.animation_thread = None
        
        # 配置显示选项
        self.show_grid = self.config.get("show_grid", False)
        self.show_robot_pose = self.config.get("show_robot_pose", True)
        self.background_color = self.config.get("background_color", "black")
        
        # 初始化图形
        self._init_plot()
        
        logger.info("点云可视化器初始化完成")
    
    def _init_plot(self):
        """初始化图形界面"""
        self.ax.set_xlim3d([-10, 10])
        self.ax.set_ylim3d([-10, 10])
        self.ax.set_zlim3d([0, 10])
        
        self.ax.set_xlabel('X (米)')
        self.ax.set_ylabel('Y (米)')
        self.ax.set_zlabel('Z (米)')
        self.ax.set_title('Nova Carter 点云实时显示')
        
        # 设置背景
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_facecolor(self.background_color)
        
        # 添加网格
        if self.show_grid:
            self.ax.grid(True, alpha=0.3)
        
        # 初始化空点云
        self._update_plot_title()
        
        plt.tight_layout()
    
    def _update_plot_title(self):
        """更新图表标题"""
        if self.state.points is not None:
            num_points = len(self.state.points)
            self.ax.set_title(f'Nova Carter 点云 | 点数: {num_points} | 缩放: {self.state.zoom:.1f}x')
        else:
            self.ax.set_title('Nova Carter 点云 | 等待数据...')
    
    def update_pointcloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        更新点云数据
        
        Args:
            points: 点云数组 (N, 3)
            colors: 可选的颜色数组 (N, 3) 或 (N, 4)
        """
        # 保存新数据
        self.state.points = points
        self.state.colors = colors
        
        # 清除旧数据
        self.ax.clear()
        
        if len(points) == 0:
            self._update_plot_title()
            self.fig.canvas.draw()
            return
        
        # 根据缩放和旋转调整点云
        transformed_points = self._transform_points(points)
        
        # 确定颜色
        if colors is None:
            colors = self._compute_colors(transformed_points)
        elif colors.shape[1] == 3:
            # 如果传入的是 (N, 3)，转换为 (N, 4)
            alpha = np.ones((len(colors), 1))
            colors = np.hstack([colors, alpha])
        
        # 绘制点云
        self.ax.scatter(
            transformed_points[:, 0],
            transformed_points[:, 1],
            transformed_points[:, 2],
            c=colors,
            s=1,  # 点大小
            marker='.',
            alpha=0.6
        )
        
        # 显示机器人位置（如果有）
        if self.show_robot_pose:
            self._draw_robot_pose()
        
        # 更新标题
        self._update_plot_title()
        
        # 刷新显示
        self.fig.canvas.draw()
    
    def _transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        根据视图参数变换点云
        
        Args:
            points: 原始点云 (N, 3)
            
        Returns:
            transformed: 变换后的点云
        """
        if len(points) == 0:
            return points
        
        # 1. 应用旋转（绕 Z 轴）
        angle_rad = np.radians(self.state.view_azimuth)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x_rot = points[:, 0] * cos_a - points[:, 1] * sin_a
        y_rot = points[:, 0] * sin_a + points[:, 1] * cos_a
        z_rot = points[:, 2]
        
        # 2. 应用仰角（绕 X 轴）
        elevation_rad = np.radians(self.state.view_elevation)
        cos_e = np.cos(elevation_rad)
        sin_e = np.sin(elevation_rad)
        
        x_final = x_rot
        y_final = y_rot * cos_e - z_rot * sin_e
        z_final = y_rot * sin_e + z_rot * cos_e
        
        # 3. 应用缩放（绕原点）
        if self.state.zoom != 1.0:
            x_final = x_final * self.state.zoom
            y_final = y_final * self.state.zoom
            z_final = z_final * self.state.zoom
        
        transformed = np.stack([x_final, y_final, z_final], axis=1)
        return transformed
    
    def _compute_colors(self, points: np.ndarray) -> np.ndarray:
        """
        根据当前颜色模式计算颜色
        
        Args:
            points: 点云数组
            
        Returns:
            colors: RGB 颜色数组
        """
        num_points = len(points)
        
        if self.state.color_mode == "height":
            # 基于 Z 坐标编码颜色
            z_values = points[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            z_range = z_max - z_min + 0.001
            
            # 归一化到 0-1
            normalized = (z_values - z_min) / z_range
            
            # 颜色：蓝(低) -> 绿(中) -> 红(高)
            colors = np.zeros((num_points, 3), dtype=np.float32)
            colors[:, 0] = 1.0 - normalized  # 红
            colors[:, 1] = normalized  # 绿色
            colors[:, 2] = normalized * 0.5  # 蓝色
            
        elif self.state.color_mode == "distance":
            # 基于到原点距离编码颜色
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
            d_min, d_max = np.min(distances), np.max(distances)
            d_range = d_max - d_min + 0.001
            
            normalized = (distances - d_min) / d_range
            
            colors = np.zeros((num_points, 3), dtype=np.float32)
            colors[:, 0] = 1.0 - normalized  # 红(近)
            colors[:, 1] = normalized  # 绿色(中)
            colors[:, 2] = normalized * 0.5  # 蓝色(远)
            
        else:  # intensity 模式
            # 使用固定颜色
            colors = np.ones((num_points, 3), dtype=np.float32) * 0.7  # 灰色
        
        return colors
    
    def _draw_robot_pose(self, robot_position: Optional[Dict[str, float]] = None):
        """
        在点云图上绘制机器人位置
        
        Args:
            robot_position: 机器人位置字典 {"x", "y", "z", "yaw"}
        """
        if robot_position is None:
            return
        
        x = robot_position.get("x", 0)
        y = robot_position.get("y", 0)
        z = robot_position.get("z", 0)
        
        # 绘制机器人图标（红色箭头）
        self.ax.quiver(
            x, y, z,
            1, 0, 0,  # 方向
            length=2.0,
            color='red',
            linewidth=3,
            arrow_length_ratio=0.3
        )
        
        # 绘制位置标签
        label = f'Robot\n({x:.1f}, {y:.1f}, {z:.1f})'
        self.ax.text(x, y, z + 0.5, label, color='white', fontsize=9)
    
    def start_auto_rotation(self, speed: float = 1.0):
        """
        开始自动旋转
        
        Args:
            speed: 旋转速度（度/秒）
        """
        self.state.auto_rotate = True
        self.state.rotation_speed = speed
        logger.info(f"开始自动旋转: {speed}°/秒")
    
    def stop_auto_rotation(self):
        """停止自动旋转"""
        self.state.auto_rotate = False
        logger.info("停止自动旋转")
    
    def rotate_view(self, delta_angle: float):
        """
        手动旋转视图
        
        Args:
            delta_angle: 旋转角度（度），正数顺时针，负数逆时针
        """
        self.state.view_azimuth = (self.state.view_azimuth + delta_angle) % 360
        if self.state.points is not None:
            self.update_pointcloud(self.state.points, self.state.colors)
        logger.info(f"旋转视图到: {self.state.view_azimuth}°")
    
    def set_elevation(self, elevation: float):
        """
        设置俯仰角
        
        Args:
            elevation: 俯仰角（度），0-90度
        """
        self.state.view_elevation = np.clip(elevation, -90, 90)
        if self.state.points is not None:
            self.update_pointcloud(self.state.points, self.state.colors)
        logger.info(f"设置俯仰角到: {self.state.view_elevation}°")
    
    def set_zoom(self, zoom: float):
        """
        设置缩放因子
        
        Args:
            zoom: 缩放因子，1.0=原始大小
        """
        self.state.zoom = np.clip(zoom, 0.5, 3.0)
        if self.state.points is not None:
            self.update_pointcloud(self.state.points, self.state.colors)
        logger.info(f"设置缩放到: {self.state.zoom:.1f}x")
    
    def set_color_mode(self, mode: str):
        """
        设置颜色编码模式
        
        Args:
            mode: 颜色模式 ("height", "distance", "intensity")
        """
        self.state.color_mode = mode
        if self.state.points is not None:
            self.update_pointcloud(self.state.points, self.state.colors)
        logger.info(f"颜色模式: {mode}")
    
    def reset_view(self):
        """重置视图参数"""
        self.state.view_azimuth = 45.0
        self.state.view_elevation = 30.0
        self.state.zoom = 1.0
        if self.state.points is not None:
            self.update_pointcloud(self.state.points, self.state.colors)
        logger.info("视图已重置")
    
    def save_screenshot(self, filename: Optional[str] = None):
        """
        保存点云截图
        
        Args:
            filename: 文件名，默认为带时间戳的文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointcloud_{timestamp}.png"
        
        filepath = os.path.join(self.save_dir, filename)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"截图已保存: {filepath}")
        
        return filepath
    
    def show(self):
        """显示可视化窗口"""
        plt.show()
    
    def close(self):
        """关闭可视化器"""
        plt.close(self.fig)
        logger.info("点云可视化器已关闭")


# 便捷函数
def create_visualizer(config: Optional[Dict[str, Any]] = None) -> PointCloudVisualizer:
    """
    创建点云可视化器
    
    Args:
        config: 配置字典
        
    Returns:
        PointCloudVisualizer: 可视化器实例
    """
    return PointCloudVisualizer(config)


# 使用示例
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Qt5Agg')  # 使用 Qt 后端（推荐）
    
    # 创建可视化器
    viz = create_visualizer({
        "save_dir": "data/pointcloud",
        "show_grid": True,
        "show_robot_pose": True,
        "background_color": "black"
    })
    
    # 生成模拟点云数据（用于测试）
    np.random.seed(42)
    
    # 生成地面点
    ground_points = np.random.randn(2000, 3) * [5, 5, 0.02]
    ground_points[:, 2] += np.random.randn(2000) * 0.01
    
    # 生成障碍物（盒子）
    for i in range(5):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z_base = np.random.uniform(0.5, 2.0)
        
        # 生成盒子的 6 个面
        for _ in range(100):
            px = x + np.random.uniform(-0.4, 0.4)
            py = y + np.random.uniform(-0.4, 0.4)
            pz = z_base + np.random.uniform(0, 0.5)
            ground_points = np.vstack([ground_points, [px, py, pz]])
    
    # 显示点云
    colors = viz._compute_colors(ground_points)
    viz.update_pointcloud(ground_points, colors)
    
    # 显示窗口
    print("点云可视化器已启动")
    print("控制方式:")
    print("  -/ : 缩小")
    print("  + : 放大")
    print("  ← : 左旋转")
    print("  → : 右旋转")
    print("  ↑ : 上仰")
    print("  ↓ : 下俯")
    print("  h : 高度颜色模式")
    print("  d : 距离颜色模式")
    print("  i : 强度颜色模式")
    print("  r : 重置视图")
    print("  s : 保存截图")
    print("  q : 退出")
    
    # 键盘控制
    from matplotlib.widgets import Button
    
    def on_key(event):
        if event.key == '/':
            viz.set_zoom(viz.state.zoom / 1.2)
        elif event.key == '+':
            viz.set_zoom(viz.state.zoom * 1.2)
        elif event.key == 'left':
            viz.rotate_view(-5)
        elif event.key == 'right':
            viz.rotate_view(5)
        elif event.key == 'up':
            viz.set_elevation(viz.state.view_elevation + 10)
        elif event.key == 'down':
            viz.set_elevation(viz.state.view_elevation - 10)
        elif event.key == 'h':
            viz.set_color_mode("height")
        elif event.key == 'd':
            viz.set_color_mode("distance")
        elif event.key == 'i':
            viz.set_color_mode("intensity")
        elif event.key == 'r':
            viz.reset_view()
        elif event.key == 's':
            viz.save_screenshot()
        elif event.key == 'q':
            plt.close(viz.fig)
            print("退出可视化器")
            return
    
    viz.fig.canvas.mpl_connect('key_press_event', on_key)
    viz.show()








