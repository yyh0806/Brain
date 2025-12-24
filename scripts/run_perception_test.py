#!/usr/bin/env python3
"""
运行Isaac Sim感知测试

这个脚本用于运行Isaac Sim环境中的感知测试，启动Brain感知模块并展示结果。
"""

import asyncio
import argparse
import sys
import time
import signal
from pathlib import Path

try:
    import numpy as np
    print("NumPy模块加载成功")
except ImportError:
    print("错误: 需要安装NumPy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("Matplotlib模块加载成功")
except ImportError:
    print("警告: Matplotlib未安装，将跳过可视化")
    plt = None

# 全局变量
running = True
perception_data_history = []
update_count = 0


def signal_handler(sig, frame):
    """处理终止信号"""
    global running
    print("正在终止程序...")
    running = False


async def run_perception_test(config):
    """运行感知测试"""
    global perception_data_history, update_count, running
    
    print("启动Isaac Sim感知测试...")
    
    # 模拟ROS2接口
    # 在实际使用中，这里应该是真实的ROS2接口
    from unittest.mock import AsyncMock, Mock
    
    class MockROS2Interface:
        def __init__(self):
            self.timestamp_counter = 0
            self.pose_counter = 0
        
        async def get_sensor_data(self):
            # 模拟随时间变化的传感器数据
            self.timestamp_counter += 1
            self.pose_counter += 0.05
            
            # 模拟机器人移动
            pose_angle = (self.pose_counter % (2 * math.pi))
            pose_x = 2.0 * math.cos(pose_angle)
            pose_y = 2.0 * math.sin(pose_angle)
            
            # 模拟障碍物
            obstacle_angle = (self.timestamp_counter % (2 * math.pi))
            obstacle_distance = 3.0 + 2.0 * math.sin(obstacle_angle)
            
            return Mock(
                timestamp=self.timestamp_counter,
                rgb_image=np.random.randint(0, 256, (480, 640, 3)),
                depth_image=np.random.rand(480, 640) * 5.0,
                laser_scan={
                    "ranges": [obstacle_distance] * 180 + [10.0] * 180,
                    "angles": [i * 0.017 for i in range(360)]
                },
                imu={
                    "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
                    "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.1},
                    "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
                },
                odometry={
                    "position": {"x": pose_x, "y": pose_y, "z": 0.0},
                    "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                    "linear_velocity": {"x": -0.2, "y": 0.5, "z": 0.0},
                    "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
                }
            )
        
        def get_rgb_image(self):
            return np.random.randint(0, 256, (480, 640, 3))
        
        def get_depth_image(self):
            return np.random.rand(480, 640) * 5.0
        
        def get_laser_scan(self):
            return {
                "ranges": [3.0 + 2.0 * math.sin(time.time())] * 180 + [10.0] * 180,
                "angles": [i * 0.017 for i in range(360)]
            }
    
    # 模拟ROS2接口
    ros2_interface = MockROS2Interface()
    
    # 尝试导入Brain感知模块
    try:
        from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
        from brain.perception.object_detector import ObjectDetector
        from brain.perception.vlm.vlm_perception import VLMPerception
        
        # 创建传感器管理器
        sensor_manager = ROS2SensorManager(
            ros2_interface=ros2_interface,
            config={
                "sensors": {
                    "rgb_camera": {"enabled": True},
                    "depth_camera": {"enabled": True},
                    "lidar": {"enabled": True},
                    "imu": {"enabled": True}
                },
                "grid_resolution": 0.1,
                "map_size": 20.0,
                "pose_filter_alpha": 0.8,
                "obstacle_threshold": 0.5,
                "min_obstacle_size": 0.1
            }
        )
        
        # 创建目标检测器
        detector = ObjectDetector({
            "mode": "tracking",
            "confidence_threshold": 0.5
        })
        
        # 创建VLM感知器
        vlm = VLMPerception(
            model="llava:latest",
            ollama_host="http://localhost:11434",
            use_yolo=False
        )
        
        print("Brain感知模块初始化成功")
        
    except ImportError as e:
        print(f"错误: 无法导入Brain感知模块: {e}")
        print("请确保Brain项目路径正确并且依赖已安装")
        return
    
    # 启动感知循环
    print("启动感知循环...")
    last_vlm_time = 0
    
    try:
        while running:
            update_count += 1
            start_time = time.time()
            
            # 获取融合感知数据
            perception_data = await sensor_manager.get_fused_perception()
            
            if perception_data:
                # 记录感知数据历史
                perception_data_history.append(perception_data)
                if len(perception_data_history) > 100:
                    perception_data_history.pop(0)
                
                # 打印基本信息
                print(f"\\n[{update_count}] 感知数据更新")
                print(f"  时间戳: {perception_data.timestamp:.2f}")
                
                if perception_data.pose:
                    print(f"  机器人位姿: x={perception_data.pose.x:.2f}, y={perception_data.pose.y:.2f}, yaw={perception_data.pose.yaw:.2f}")
                
                # 打印传感器状态
                for sensor, is_healthy in perception_data.sensor_status.items():
                    status = "健康" if is_healthy else "不健康"
                    print(f"  {sensor}: {status}")
                
                # 打印障碍物信息
                if perception_data.obstacles:
                    print(f"  检测到 {len(perception_data.obstacles)} 个障碍物")
                    nearest_obstacle = sensor_manager.get_nearest_obstacle()
                    if nearest_obstacle:
                        print(f"  最近障碍物距离: {nearest_obstacle['distance']:.2f}米")
                
                # 每5次循环执行一次VLM分析
                if update_count % 5 == 0 and perception_data.rgb_image is not None:
                    try:
                        scene = await vlm.describe_scene(perception_data.rgb_image)
                        last_vlm_time = time.time()
                        
                        print(f"\\n  VLM场景分析:")
                        print(f"    概要: {scene.summary}")
                        print(f"    物体数量: {len(scene.objects)}")
                        if scene.objects:
                            for i, obj in enumerate(scene.objects):
                                print(f"    物体{i+1}: {obj.label} - {obj.position_description}")
                        print(f"    导航提示: {', '.join(scene.navigation_hints) if scene.navigation_hints else '无'}")
                        
                        # 如果有潜在目标，进行目标搜索
                        if scene.potential_targets and len(scene.potential_targets) > 0:
                            target = scene.potential_targets[0]
                            search_result = await vlm.find_target(perception_data.rgb_image, target)
                            
                            print(f"\\n  目标搜索 - {target}:")
                            if search_result.found:
                                print(f"    找到: 是")
                                print(f"    位置: {search_result.best_match.position_description}")
                                print(f"    建议: {search_result.suggested_action}")
                            else:
                                print(f"    找到: 否")
                                print(f"    说明: {search_result.explanation}")
                        
                    except Exception as e:
                        print(f"  VLM分析失败: {e}")
                elif update_count - last_vlm_time < 5:
                    print(f"  上次VLM分析时间: {time.time() - last_vlm_time:.1f}秒前")
                
                # 可视化
                if config.visualize and plt:
                    try:
                        visualize_perception_data(perception_data, update_count)
                    except Exception as e:
                        print(f"  可视化失败: {e}")
            else:
                print("  感知数据为空")
            
            # 控制循环频率
            elapsed = time.time() - start_time
            target_period = 1.0 / config.update_rate
            sleep_time = max(0, target_period - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\\n用户中断，正在关闭...")
    except Exception as e:
        print(f"\\n感知测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\\n感知测试结束")


def visualize_perception_data(perception_data, update_count):
    """可视化感知数据"""
    plt.figure(figsize=(15, 10))
    
    # RGB图像
    if perception_data.rgb_image is not None:
        plt.subplot(2, 3, 1)
        plt.imshow(perception_data.rgb_image)
        plt.title("RGB图像")
        plt.axis('off')
    
    # 深度图像
    if perception_data.depth_image is not None:
        plt.subplot(2, 3, 2)
        plt.imshow(perception_data.depth_image, cmap='jet')
        plt.title("深度图像")
        plt.colorbar()
        plt.axis('off')
    
    # 激光雷达数据
    if perception_data.laser_ranges is not None:
        plt.subplot(2, 3, 3)
        angles = np.linspace(-np.pi, np.pi, len(perception_data.laser_ranges))
        x = perception_data.laser_ranges * np.cos(angles)
        y = perception_data.laser_ranges * np.sin(angles)
        plt.plot(x, y, 'r.')
        plt.title("激光雷达数据")
        plt.axis('equal')
        plt.grid(True)
    
    # 机器人位姿
    if perception_data.pose:
        plt.subplot(2, 3, 4)
        plt.scatter(perception_data.pose.x, perception_data.pose.y, s=100, c='red', marker='o')
        plt.title(f"机器人位姿: ({perception_data.pose.x:.2f}, {perception_data.pose.y:.2f})")
        plt.axis('equal')
        plt.grid(True)
        
        # 绘制朝向
        arrow_length = 1.0
        arrow_x = perception_data.pose.x + arrow_length * np.cos(perception_data.pose.yaw)
        arrow_y = perception_data.pose.y + arrow_length * np.sin(perception_data.pose.yaw)
        plt.arrow(perception_data.pose.x, perception_data.pose.y,
                  arrow_x - perception_data.pose.x, arrow_y - perception_data.pose.y,
                  head_width=0.2, head_length=0.3, fc='red', ec='red')
    
    # 占据栅格地图
    if perception_data.occupancy_grid is not None:
        plt.subplot(2, 3, 5)
        plt.imshow(perception_data.occupancy_grid, cmap='gray')
        plt.title("占据栅格地图")
        plt.colorbar()
        plt.axis('off')
    
    # 障碍物
    if perception_data.obstacles:
        plt.subplot(2, 3, 6)
        for obs in perception_data.obstacles:
            if 'local_position' in obs:
                x = obs['local_position']['x']
                y = obs['local_position']['y']
                size = obs['size']
                circle = plt.Circle((x, y), size/2, color='red', alpha=0.5)
                plt.gca().add_patch(circle)
                plt.text(x, y, obs['id'], ha='center', va='center')
        
        plt.title("障碍物")
        plt.axis('equal')
        plt.grid(True)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
    
    plt.suptitle(f"感知数据可视化 - 更新 #{update_count}")
    plt.tight_layout()
    plt.pause(0.1)
    plt.clf()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行Isaac Sim感知测试")
    parser.add_argument("--update-rate", type=float, default=10.0,
                       help="更新频率 (Hz)")
    parser.add_argument("--visualize", action="store_true",
                       help="启用可视化")
    parser.add_argument("--duration", type=int, default=60,
                       help="测试持续时间 (秒)")
    args = parser.parse_args()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"开始Isaac Sim感知测试...")
    print(f"  更新频率: {args.update_rate} Hz")
    print(f"  可视化: {'启用' if args.visualize else '禁用'}")
    print(f"  持续时间: {args.duration} 秒")
    
    # 创建配置对象
    class Config:
        def __init__(self, update_rate, visualize):
            self.update_rate = update_rate
            self.visualize = visualize
    
    config = Config(args.update_rate, args.visualize)
    
    # 运行指定时间的测试
    try:
        await asyncio.wait_for(
            run_perception_test(config),
            timeout=args.duration
        )
    except asyncio.TimeoutError:
        print(f"\\n测试在 {args.duration} 秒后结束")
    except KeyboardInterrupt:
        print(f"\\n用户中断")
    
    print("\\n测试结束")


if __name__ == "__main__":
    asyncio.run(main())


