#!/usr/bin/env python3
"""
Nova Carter 完整集成测试脚本

集成功能：
1. ROS2 接口（Nova Carter 话题）
2. 点云数据处理
3. 点云可视化（3D）
4. 世界模型可视化（2D 占据地图）
5. 交互式控制
6. 统计信息显示
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, Image, Imu
from nav_msgs.msg import Odometry
import numpy as np
from loguru import logger

# 导入自定义模块
from brain.communication.ros2_interface import ROS2Interface, ROS2Config, ROS2Mode, TwistCommand
from brain.perception.lidar_processing import LidarProcessor, PointCloudData
from brain.visualization.pointcloud_visualizer import PointCloudVisualizer, create_visualizer
from brain.visualization.world_model_viz import WorldModelVisualizer, create_world_model_viz, WorldModelVizConfig
from brain.cognitive.world_model import WorldModel
from brain.perception.sensors.sensor_manager import MultiSensorManager
from typing import Dict, List, Any, Optional


class NovaCarterTester:
    """Nova Carter 完整测试器"""
    
    def __init__(self):
        # ROS2 接口
        self.ros2_interface: Optional[ROS2Interface] = None
        
        # 点云处理器
        self.lidar_processor: Optional[LidarProcessor] = None
        
        # 可视化器
        self.pointcloud_viz: Optional[PointCloudVisualizer] = None
        self.world_model_viz: Optional[WorldModelVisualizer] = None
        
        # 世界模型
        self.world_model: Optional[WorldModel] = None
        
        # 传感器管理器
        self.sensor_manager: Optional[MultiSensorManager] = None
        
        # 运行状态
        self.running = False
        
        # 统计信息
        self.stats = {
            "start_time": None,
            "frame_count": 0,
            "pointcloud_count": 0,
            "odom_count": 0,
            "image_count": 0
        }
    
    async def initialize(self):
        """初始化所有组件"""
        print("\n========================================================================")
        print(" Nova Carter 完整集成测试")
        print("========================================================================")
        
        # 1. 配置 ROS2 接口
        print("\n[1/7] 初始化 ROS2 接口...")
        ros2_config = ROS2Config(
            node_name="brain_nova_carter",
            mode=ROS2Mode.REAL,
            topics={
                "cmd_vel": "/cmd_vel",
                "odom": "/chassis/odom",
                "rgb_image": "/front_stereo_camera/left/image_raw",
                "imu": "/chassis/imu",
                "pointcloud": "/front_3d_lidar/lidar_points"
            }
        )
        ros2_config.topics["rgb_image_compressed"] = False
        
        self.ros2_interface = ROS2Interface(ros2_config)
        await self.ros2_interface.initialize()
        
        if self.ros2_interface.get_mode() == ROS2Mode.REAL:
            print("      ✓ ROS2 接口已连接")
        else:
            print("      ⚠ ROS2 接口处于模拟模式")
        
        # 2. 初始化点云处理器
        print("\n[2/7] 初始化点云处理器...")
        self.lidar_processor = LidarProcessor({
            "max_points": 10000,
            "downsample_factor": 2,
            "ground_z_threshold": 0.05,
            "noise_filter": True
        })
        print("      ✓ 点云处理器已创建")
        
        # 3. 初始化点云可视化器
        print("\n[3/7] 初始化点云可视化器...")
        self.pointcloud_viz = create_visualizer({
            "save_dir": "data/pointcloud",
            "show_grid": False,
            "show_robot_pose": False,
            "background_color": "black"
        })
        print("      ✓ 点云可视化器已创建")
        
        # 4. 初始化世界模型可视化器
        print("\n[4/7] 初始化世界模型可视化器...")
        self.world_model_viz = create_world_model_viz(WorldModelVizConfig(
            "show_occupancy_map": True,
            "show_robot_pose": True,
            "show_trajectory": True,
            "show_detected_objects": False,
            "show_exploration_areas": False,
            "max_trajectory_points": 500,
            "background_color": "black"
        ))
        print("      ✓ 世界模型可视化器已创建")
        
        # 5. 初始化世界模型
        print("\n[5/7] 初始化世界模型...")
        self.world_model = WorldModel({})
        print("      ✓ 世界模型已创建")
        
        # 6. 初始化传感器管理器
        print("\n[6/7] 初始化传感器管理器...")
        self.sensor_manager = MultiSensorManager({})
        print("      ✓ 传感器管理器已创建")
        
        self.running = True
        self.stats["start_time"] = time.time()
        
        # 注册回调
        self._register_callbacks()
        
        print("\n" + "-"*74)
        print("初始化完成！等待传感器数据...")
        print("-"*74)
    
    def _register_callbacks(self):
        """注册传感器回调"""
        # 点云回调
        async def on_pointcloud(pointcloud_data: PointCloudData):
            if pointcloud_data.points is not None and len(pointcloud_data.points) > 0:
                self.stats["pointcloud_count"] += 1
                
                # 处理点云
                processed_data = self.lidar_processor.process(pointcloud_data, return_colors=True)
                
                # 更新点云可视化
                self.pointcloud_viz.update_pointcloud(
                    processed_data.points,
                    processed_data.colors
                )
                
                # 生成占据地图
                occupancy_grid, info = self.lidar_processor.to_occupancy_grid(
                    processed_data.points,
                    resolution=0.1,
                    origin=(0.0, 0.0)
                )
                
                # 更新世界模型
                self.world_model.current_map = occupancy_grid
                self.world_model.map_resolution = info["resolution"]
                self.world_model.map_origin = info["origin"]
                
                logger.debug(f"点云处理完成: {len(processed_data.points)} 点, 占据地图 {occupancy_grid.shape}")
        
        self.ros2_interface.register_sensor_callback("pointcloud", on_pointcloud)
        
        # 里程计回调
        async def on_odometry(odom_data):
            self.stats["odom_count"] += 1
            
            # 提取位置和姿态
            if "pose" in odom_data:
                pos = odom_data["pose"]
                x = pos.get("x", 0)
                y = pos.get("y", 0)
                z = pos.get("z", 0)
                yaw = pos.get("yaw", 0)
            else:
                x = odom_data.get("position", {}).get("x", 0)
                y = odom_data.get("position", {}).get("y", 0)
                z = odom_data.get("position", {}).get("z", 0)
                yaw = odom_data.get("orientation", {}).get("yaw", 0)
            
            # 更新机器人位置
            self.world_model.robot_position.update({"x": x, "y": y, "z": z})
            self.world_model.robot_heading = yaw
            
            # 更新世界模型可视化
            self.world_model_viz.update_robot_pose(
                position=self.world_model.robot_position,
                yaw=yaw,
                velocity=odom_data.get("velocity", {})
            )
            
            logger.debug(f"里程计更新: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        
        self.ros2_interface.register_sensor_callback("odom", on_odometry)
        
        # IMU 回调
        async def on_imu(imu_data):
            # 更新世界模型的 IMU 数据
            pass
        
        self.ros2_interface.register_sensor_callback("imu", on_imu)
        
        # 图像回调
        async def on_image(image_data):
            self.stats["image_count"] += 1
            logger.debug(f"图像接收: {image_data.shape}")
        
        self.ros2_interface.register_sensor_callback("rgb_image", on_image)
    
    async def run(self):
        """主运行循环"""
        print("\n" + "="*74)
        print("开始传感器数据采集和处理...")
        print("="*74)
        print("\n传感器数据流：")
        print("  - 点云 (3D 激光雷达): /front_3d_lidar/lidar_points")
        print("  - 里程计: /chassis/odom")
        print("  - IMU: /chassis/imu")
        print("  - RGB 相机: /front_stereo_camera/left/image_raw")
        print("\n" + "-"*74)
        print("可视化功能：")
        print("  - 点云 (3D 视图): matplotlib")
        print("  - 占据地图 (2D 热图): matplotlib")
        print("  - 机器人位置和轨迹")
        print("\n" + "-"*74)
        print("控制说明：")
        print("  w/W - 前进")
        print("  s/S - 后退")
        print("  a/A - 左转")
        print("  d/D - 右转")
        print("  q/Q - 左旋转")
        print("  e/E - 右旋转")
        print("  space - 停止")
        print("  +/- - 缩小/放大")
        print("  h/d/i - 切换颜色模式")
        print("  r - 重置视图")
        print("  p - 保存点云截图")
        print("  o - 保存世界模型截图")
        print("  c - 打印统计")
        print("  q - 退出")
        print("-"*74)
        
        # 启动可视化的交互式更新
        viz_update_task = asyncio.create_task(self._update_visualizations_loop())
        
        # 控制循环（使用键盘）
        control_task = asyncio.create_task(self._control_loop())
        
        # 等待用户输入
        print("\n按任意键开始控制...")
        
        try:
            import termios
            import tty
            import select
            
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        print("\n退出程序...")
                        break
                    
                    elif key == 'w':
                        await self.ros2_interface.publish_twist(TwistCommand.forward(speed=0.5))
                        print("      -> 前进 0.5 m/s")
                    
                    elif key == 's':
                        await self.ros2_interface.publish_twist(TwistCommand.backward(speed=0.5))
                        print("      -> 后退 0.5 m/s")
                    
                    elif key == 'a':
                        await self.ros2_interface.publish_twist(TwistCommand.turn_left(linear_speed=0.3, angular_speed=0.5))
                        print("      -> 左转 (0.3 m/s, 0.5 rad/s)")
                    
                    elif key == 'd':
                        await self.ros2_interface.publish_twist(TwistCommand.turn_right(linear_speed=0.3, angular_speed=0.5))
                        print("      -> 右转 (0.3 m/s, 0.5 rad/s)")
                    
                    elif key == 'q':
                        await self.ros2_interface.publish_twist(TwistCommand.rotate_left(angular_speed=0.5))
                        print("      -> 左旋转 0.5 rad/s")
                    
                    elif key == 'e':
                        await self.ros2_interface.publish_twist(TwistCommand.rotate_right(angular_speed=0.5))
                        print("      -> 右旋转 0.5 rad/s")
                    
                    elif key == ' ':
                        await self.ros2_interface.publish_twist(TwistCommand.stop())
                        print("      -> 停止")
                    
                    elif key == '+':
                        self.pointcloud_viz.set_zoom(self.pointcloud_viz.state.zoom * 1.2)
                        print(f"      -> 放大 (缩放: {self.pointcloud_viz.state.zoom * 1.2:.1f}x)")
                    
                    elif key == '-':
                        zoom = max(0.5, self.pointcloud_viz.state.zoom / 1.2)
                        self.pointcloud_viz.set_zoom(zoom)
                        print(f"      -> 缩小 (缩放: {self.pointcloud_viz.state.zoom / 1.2:.1f}x)")
                    
                    elif key == 'h':
                        self.pointcloud_viz.set_elevation(self.pointcloud_viz.state.view_elevation + 10)
                        print(f"      -> 上仰 +10 度")
                    
                    elif key == 'd':
                        self.pointcloud_viz.set_elevation(self.pointcloud_viz.state.view_elevation - 10)
                        print(f"      -> 下俯 -10 度")
                    
                    elif key == 'r':
                        self.pointcloud_viz.rotate_view(-15)
                        print("      -> 左旋转 -15 度")
                    
                    elif key == 'l':
                        self.pointcloud_viz.rotate_view(15)
                        print("      -> 右旋转 +15 度")
                    
                    elif key == 'p':
                        await self._print_statistics()
                    
                    elif key == 'o':
                        self.pointcloud_viz.save_screenshot()
                        print("      -> 点云截图已保存")
                    
                    elif key == 'c':
                        await self._print_statistics()
                    
                await asyncio.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            logger.error(f"运行异常: {e}")
            import traceback
            traceback.print_exc()
        
        await self.cleanup()
    
    async def _update_visualizations_loop(self):
        """定期更新可视化"""
        while self.running:
            try:
                # 更新点云可视化
                if self.pointcloud_viz.state.points is not None:
                    self.pointcloud_viz.fig.canvas.draw()
                
                # 更新世界模型可视化
                if self.world_model.current_map is not None:
                    self.world_model_viz.fig.canvas.draw()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"可视化更新异常: {e}")
    
    async def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*74)
        print("系统统计信息")
        print("="*74)
        
        elapsed = time.time() - self.stats["start_time"]
        
        print(f"\n运行时间: {elapsed:.1f} 秒")
        print(f"接收帧数: {self.stats['frame_count']}")
        print(f"点云数: {self.stats['pointcloud_count']}")
        print(f"里程计数: {self.stats['odom_count']}")
        print(f"图像数: {self.stats['image_count']}")
        
        # 点云统计
        lidar_stats = self.lidar_processor.get_stats() if self.lidar_processor else {}
        if lidar_stats:
            print(f"\n点云处理统计:")
            print(f"  平均点数/帧: {lidar_stats.get('avg_points_per_msg', 0):.0f}")
            print(f"  过滤后点数: {lidar_stats.get('avg_filtered', 0):.0f}")
            print(f"  过滤率: {lidar_stats.get('avg_filtered', 0) * 100 / (lidar_stats.get('avg_points_per_msg', 0) + 1e-6):.1f}%")
        
        # 世界模型统计
        if self.world_model:
            print(f"\n世界模型统计:")
            print(f"  追踪物体数: {len(self.world_model.tracked_objects)}")
            print(f"  语义物体数: {len(self.world_model.semantic_objects)}")
        
        print("\n" + "-"*74)
    
    async def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        self.running = False
        
        # 停止可视化
        if self.pointcloud_viz:
            self.pointcloud_viz.close()
            print("  ✓ 点云可视化器已关闭")
        
        if self.world_model_viz:
            self.world_model_viz.close()
            print("  ✓ 世界模型可视化器已关闭")
        
        # 关闭 ROS2
        if self.ros2_interface:
            await self.ros2_interface.shutdown()
            print("  ✓ ROS2 接口已关闭")
        
        print("资源清理完成")
        self.stats["end_time"] = time.time()


async def main():
    """主函数"""
    tester = NovaCarterTester()
    
    try:
        # 初始化
        await tester.initialize()
        
        # 等待数据
        print("\n等待传感器数据（最多 10 秒）...")
        data_received = False
        
        for i in range(100):
            await asyncio.sleep(0.1)
            
            if tester.stats["pointcloud_count"] > 0:
                print("  ✓ 点云数据已接收")
                data_received = True
                break
            
            if tester.stats["odom_count"] > 0:
                print("  ✓ 里程计数据已接收")
                data_received = True
                break
        
        if data_received:
            print("\n传感器数据连接成功！")
            print("\n现在可以：")
            print("  1. 实时查看点云可视化（3D）")
            print("  2. 观察占据地图（2D 热图）")
            print("  3. 查看机器人轨迹")
            print("  4. 使用键盘控制小车")
            print("\n输入 'c' 查看统计信息")
            
            # 运行主循环
            await tester.run()
        
        else:
            print("\n警告：10 秒内未收到传感器数据")
            print("\n故障排查指南:")
            print("  1. 确认 Isaac Sim 正在运行")
            print("  2. 确认已按下 Play 按钮")
            print("  3. 确认场景中已添加 Nova Carter with Sensors")
            print("  4. 确认已添加 nova_carter_ROS 扩展")
            print("  5. 确认 ROS2 Bridge 已启用")
            print("  6. 在终端运行以下命令验证:")
            print("     export ROS_DOMAIN_ID=0")
            print("     source /opt/ros/galactic/setup.bash")
            print("     ros2 topic list")
            print("\n预期话题:")
            print("  - /cmd_vel")
            print("  - /chassis/odom")
            print("  - /front_3d_lidar/lidar_points")
            print("  - /front_stereo_camera/left/image_raw")
            print("  - /chassis/imu")
    
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        await tester.cleanup()
    
    except Exception as e:
        logger.error(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
        await tester.cleanup()


if __name__ == "__main__":
    print("""
================================================================================
 Nova Carter 完整集成测试
================================================================================

本脚本将：
1. 连接到 Isaac Sim Nova Carter 的 ROS2 话题
2. 接收并处理 3D 点云数据
3. 实时显示点云（3D 可视化）
4. 动态生成 2D 占据地图
5. 显示世界模型状态（机器人位置、轨迹、检测物体）
6. 提供交互式控制命令

确保 Isaac Sim 正在运行，并已按下 Play 按钮！
""")
    
    print("按 Ctrl+C 退出程序\n")
    
    asyncio.run(main())
'EOFSCRIPT

