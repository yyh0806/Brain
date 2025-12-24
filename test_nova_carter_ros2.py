#!/usr/bin/env python3
"""
Nova Carter ROS2 连接测试和控制脚本
使用 Isaac Sim 官方的 nova_carter_ROS 示例
"""

import sys
import asyncio
import time
from loguru import logger
from pathlib import Path
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from brain.communication.ros2_interface import (
    ROS2Interface,
    ROS2Config,
    ROS2Mode,
    TwistCommand
)


# Nova Carter 官方示例的话题映射
NOVA_CARTER_TOPICS = {
    "cmd_vel": "/cmd_vel",
    "odom": "/chassis/odom",
    # 注意：Nova Carter 使用的是 3D 点云，不是传统激光雷达
    # "scan": "/front_3d_lidar/lidar_points",  # 3D 激光雷达（点云）
    "imu": "/chassis/imu",  # 底盘 IMU
    "rgb_image": "/front_stereo_camera/left/image_raw",  # 左相机（立体相机左）
    # Nova Carter 使用的是 Image 类型，不是 CompressedImage
    # "rgb_image_right": "/front_stereo_camera/right/image_raw",  # 右相机（立体相机右）
    # "imu_front": "/front_stereo_imu/imu",
    # "imu_back": "/back_stereo_imu/imu",
    # "imu_left": "/left_stereo_imu/imu",
    # "imu_right": "/right_stereo_imu/imu",
}


def print_header(text):
    """打印标题"""
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)


def print_success(text):
    """打印成功消息"""
    print(f"✓ {text}")


def print_warning(text):
    """打印警告消息"""
    print(f"⚠ {text}")


def print_error(text):
    """打印错误消息"""
    print(f"✗ {text}")


def print_info(text):
    """打印信息"""
    print(f"ℹ {text}")


async def test_connection():
    """测试 ROS2 连接"""
    print_header("Nova Carter ROS2 连接测试")
    
    # 配置 ROS2 接口
    config = ROS2Config(
        node_name="nova_carter_test",
        mode=ROS2Mode.REAL,
        topics=NOVA_CARTER_TOPICS
    )
    
    # 移除不需要的深度图像话题（Nova Carter 没有）
    config.topics.pop("depth_image", None)
    config.topics.pop("pointcloud", None)
    config.topics["rgb_image_compressed"] = False  # 使用非压缩图像
    
    print_info("初始化 ROS2 接口...")
    ros2_interface = ROS2Interface(config)
    
    try:
        await ros2_interface.initialize()
        
        # 检查运行状态
        if not ros2_interface.is_running():
            print_error("ROS2 接口未运行")
            return False
        
        mode = ros2_interface.get_mode()
        if mode == ROS2Mode.SIMULATION:
            print_warning("当前处于模拟模式，未连接到真实 ROS2")
            return False
        
        print_success("已连接到 ROS2")
        print_info(f"运行模式: {mode.value}")
        
        return True
        
    except Exception as e:
        print_error(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_sensor_data(ros2_interface):
    """测试传感器数据接收"""
    print_header("测试传感器数据接收")
    
    # 等待数据
    print_info("等待传感器数据 (5秒)...")
    await asyncio.sleep(5.0)
    
    # 检查里程计
    print("\n1. 里程计数据:")
    odom = ros2_interface.get_odometry()
    if odom:
        pos = odom.get("position", {})
        vel = odom.get("linear_velocity", {})
        print_success(f"   位置: x={pos.get('x', 0):.3f}, y={pos.get('y', 0):.3f}, z={pos.get('z', 0):.3f}")
        print_success(f"   速度: {vel.get('x', 0):.3f} m/s")
    else:
        print_warning("   未收到里程计数据")
    
    # 检查 RGB 图像
    print("\n2. RGB 图像数据:")
    rgb_image = ros2_interface.get_rgb_image()
    if rgb_image is not None:
        print_success(f"   图像尺寸: {rgb_image.shape}")
        print_success(f"   数据类型: {rgb_image.dtype}")
    else:
        print_warning("   未收到 RGB 图像数据")
    
    # 检查 IMU
    print("\n3. IMU 数据:")
    imu = ros2_interface.get_imu()
    if imu:
        ang_vel = imu.get("angular_velocity", {})
        print_success(f"   角速度: x={ang_vel.get('x', 0):.3f}, y={ang_vel.get('y', 0):.3f}, z={ang_vel.get('z', 0):.3f} rad/s")
    else:
        print_warning("   未收到 IMU 数据")
    
    # 检查激光雷达
    print("\n4. 激光雷达数据:")
    laser = ros2_interface.get_laser_scan()
    if laser:
        ranges = laser.get("ranges", [])
        print_success(f"   扫描点数: {len(ranges)}")
    else:
        print_warning("   未收到激光雷达数据")


async def test_movement(ros2_interface):
    """测试运动控制"""
    print_header("测试运动控制")
    
    print_info("请观察 Isaac Sim 中的小车...")
    
    # 测试 1: 前进
    print("\n测试 1: 前进 (2秒)")
    cmd = TwistCommand.forward(speed=0.5)
    await ros2_interface.publish_twist(cmd)
    await asyncio.sleep(2.0)
    await ros2_interface.publish_twist(TwistCommand.stop())
    print_success("   前进测试完成")
    
    await asyncio.sleep(1.0)
    
    # 测试 2: 后退
    print("\n测试 2: 后退 (2秒)")
    cmd = TwistCommand.backward(speed=0.5)
    await ros2_interface.publish_twist(cmd)
    await asyncio.sleep(2.0)
    await ros2_interface.publish_twist(TwistCommand.stop())
    print_success("   后退测试完成")
    
    await asyncio.sleep(1.0)
    
    # 测试 3: 左转
    print("\n测试 3: 左转 (2秒)")
    cmd = TwistCommand.turn_left(linear_speed=0.3, angular_speed=0.5)
    await ros2_interface.publish_twist(cmd)
    await asyncio.sleep(2.0)
    await ros2_interface.publish_twist(TwistCommand.stop())
    print_success("   左转测试完成")
    
    await asyncio.sleep(1.0)
    
    # 测试 4: 右转
    print("\n测试 4: 右转 (2秒)")
    cmd = TwistCommand.turn_right(linear_speed=0.3, angular_speed=0.5)
    await ros2_interface.publish_twist(cmd)
    await asyncio.sleep(2.0)
    await ros2_interface.publish_twist(TwistCommand.stop())
    print_success("   右转测试完成")
    
    await asyncio.sleep(1.0)
    
    # 测试 5: 原地左旋转
    print("\n测试 5: 原地左旋转 (2秒)")
    cmd = TwistCommand.rotate_left(angular_speed=0.5)
    await ros2_interface.publish_twist(cmd)
    await asyncio.sleep(2.0)
    await ros2_interface.publish_twist(TwistCommand.stop())
    print_success("   原地左旋转测试完成")
    
    print("\n✓ 所有运动测试完成")


async def interactive_control(ros2_interface):
    """交互式控制"""
    print_header("交互式控制模式")
    print("\n控制说明:")
    print("  w/W - 前进")
    print("  s/S - 后退")
    print("  a/A - 左转")
    print("  d/D - 右转")
    print("  q/Q - 原地左旋转")
    print("  e/E - 原地右旋转")
    print("  空格 - 停止")
    print("  p/P - 打印当前位置")
    print("  x/X - 退出")
    print("\n按键控制（按住持续运动，松开停止）：")
    
    try:
        import termios
        import tty
        import select
        
        # 设置终端
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        
        running = True
        
        while running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == 'x':
                    print("\n退出控制模式")
                    running = False
                    break
                
                elif key == ' ':
                    await ros2_interface.publish_twist(TwistCommand.stop())
                    print("\r停止", end='', flush=True)
                
                elif key == 'p':
                    pose = ros2_interface.get_current_pose()
                    print(f"\n位置: x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={pose[2]:.3f}")
                
                elif key in ['w', 's', 'a', 'd', 'q', 'e']:
                    # 持续发送命令
                    if key == 'w':
                        cmd = TwistCommand.forward(speed=0.5)
                        label = "前进"
                    elif key == 's':
                        cmd = TwistCommand.backward(speed=0.5)
                        label = "后退"
                    elif key == 'a':
                        cmd = TwistCommand.turn_left(linear_speed=0.3, angular_speed=0.5)
                        label = "左转"
                    elif key == 'd':
                        cmd = TwistCommand.turn_right(linear_speed=0.3, angular_speed=0.5)
                        label = "右转"
                    elif key == 'q':
                        cmd = TwistCommand.rotate_left(angular_speed=0.5)
                        label = "左旋转"
                    elif key == 'e':
                        cmd = TwistCommand.rotate_right(angular_speed=0.5)
                        label = "右旋转"
                    
                    # 发送命令
                    await ros2_interface.publish_twist(cmd)
                    print(f"\r{label}... (按空格停止)", end='', flush=True)
            
            # 显示当前位置
            pose = ros2_interface.get_current_pose()
            print(f"\r位置: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}", end='', flush=True)
        
        # 恢复终端
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
    except ImportError:
        print_warning("终端控制库不可用，使用简化模式")
        
        # 简化输入
        while True:
            print("\n\n请输入命令:")
            choice = input("w=前进, s=后退, a=左转, d=右转, q=左旋转, e=右旋转, space=停止, x=退出: ").strip().lower()
            
            if choice == 'x':
                break
            elif choice == 'w':
                await ros2_interface.publish_twist(TwistCommand.forward(speed=0.5))
            elif choice == 's':
                await ros2_interface.publish_twist(TwistCommand.backward(speed=0.5))
            elif choice == 'a':
                await ros2_interface.publish_twist(TwistCommand.turn_left(linear_speed=0.3, angular_speed=0.5))
            elif choice == 'd':
                await ros2_interface.publish_twist(TwistCommand.turn_right(linear_speed=0.3, angular_speed=0.5))
            elif choice == 'q':
                await ros2_interface.publish_twist(TwistCommand.rotate_left(angular_speed=0.5))
            elif choice == 'e':
                await ros2_interface.publish_twist(TwistCommand.rotate_right(angular_speed=0.5))
            elif choice == ' ' or choice == '':
                await ros2_interface.publish_twist(TwistCommand.stop())
            
            # 显示位置
            pose = ros2_interface.get_current_pose()
            print(f"位置: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}")
    
    # 确保停止
    await ros2_interface.publish_twist(TwistCommand.stop())


async def main():
    """主函数"""
    # 配置日志
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    
    # 测试连接
    connected = await test_connection()
    
    if not connected:
        print_error("连接失败！请检查:")
        print("  1. Isaac Sim 是否在运行？")
        print("  2. 是否按下了 Play 按钮？")
        print("  3. nova_carter_ROS 是否在场景中？")
        print("  4. ROS2 Bridge 是否启用？")
        return
    
    # 测试传感器
    await test_sensor_data()
    
    # 询问是否测试运动
    print("\n" + "="*70)
    response = input("是否测试运动控制? (y/n): ").strip().lower()
    if response == 'y':
        await test_movement()
    
    # 询问是否进入交互模式
    print("\n" + "="*70)
    response = input("是否进入交互式控制模式? (y/n): ").strip().lower()
    if response == 'y':
        await interactive_control()
    
    print_header("测试完成")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")

