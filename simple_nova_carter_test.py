#!/usr/bin/env python3
"""
Nova Carter 简化测试脚本 - 直接测试 ROS2 话题
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, PointCloud2
import time
import sys
import threading


class NovaCarterTester(Node):
    """Nova Carter 测试节点"""
    
    def __init__(self):
        super().__init__('nova_carter_test')
        
        # 控制命令发布者
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 订阅传感器话题
        self.odom_sub = self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )
        
        self.rgb_sub = self.create_subscription(
            Image,
            '/front_stereo_camera/left/image_raw',
            self.rgb_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/chassis/imu',
            self.imu_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            self.lidar_callback,
            10
        )
        
        # 计数器
        self.odom_count = 0
        self.rgb_count = 0
        self.imu_count = 0
        self.lidar_count = 0
        
        print("✓ 节点初始化完成")
        print("✓ 已订阅以下话题:")
        print("  - /cmd_vel (控制)")
        print("  - /chassis/odom (里程计)")
        print("  - /front_stereo_camera/left/image_raw (RGB相机)")
        print("  - /chassis/imu (IMU)")
        print("  - /front_3d_lidar/lidar_points (3D激光雷达)")
    
    def odom_callback(self, msg):
        """里程计回调"""
        self.odom_count += 1
        if self.odom_count == 1:
            print(f"\n✓ 收到里程计数据!")
            print(f"  位置: x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}, z={msg.pose.pose.position.z:.3f}")
    
    def rgb_callback(self, msg):
        """RGB 图像回调"""
        self.rgb_count += 1
        if self.rgb_count == 1:
            print(f"\n✓ 收到 RGB 图像!")
            print(f"  尺寸: {msg.height}x{msg.width}")
            print(f"  编码: {msg.encoding}")
    
    def imu_callback(self, msg):
        """IMU 回调"""
        self.imu_count += 1
        if self.imu_count == 1:
            print(f"\n✓ 收到 IMU 数据!")
            print(f"  角速度: x={msg.angular_velocity.x:.3f}, y={msg.angular_velocity.y:.3f}, z={msg.angular_velocity.z:.3f}")
    
    def lidar_callback(self, msg):
        """激光雷达回调"""
        self.lidar_count += 1
        if self.lidar_count == 1:
            print(f"\n✓ 收到激光雷达数据!")
            print(f"  点数: {msg.width * msg.height}")
    
    def publish_twist(self, linear_x, angular_z):
        """发布速度命令"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)


def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("控制菜单:")
    print("="*60)
    print("w/W - 前进")
    print("s/S - 后退")
    print("a/A - 左转")
    print("d/D - 右转")
    print("q/Q - 左旋转")
    print("e/E - 右旋转")
    print("space - 停止")
    print("p/P - 打印当前状态")
    print("x/X - 退出")
    print("="*60)


def main():
    """主函数"""
    print("="*60)
    print("Nova Carter ROS2 简化测试")
    print("="*60)
    print()
    
    # 初始化 ROS2
    print("初始化 ROS2...")
    rclpy.init()
    
    # 创建测试节点
    tester = NovaCarterTester()
    
    # 在后台运行
    spin_thread = threading.Thread(target=rclpy.spin, args=(tester,), daemon=True)
    spin_thread.start()
    
    # 等待数据
    print("\n等待传感器数据 (5秒)...")
    time.sleep(5.0)
    
    # 打印状态
    print("\n" + "="*60)
    print("传感器数据接收状态:")
    print("="*60)
    print(f"里程计: {tester.odom_count} 条消息")
    print(f"RGB 图像: {tester.rgb_count} 条消息")
    print(f"IMU: {tester.imu_count} 条消息")
    print(f"激光雷达: {tester.lidar_count} 条消息")
    
    if tester.odom_count == 0:
        print("\n⚠️  未收到里程计数据!")
        print("可能原因:")
        print("  - Isaac Sim 未按 Play 按钮")
        print("  - ROS2 Bridge 未启用")
    else:
        print("\n✅ ROS2 连接正常！")
        print("可以开始控制小车了。")
    
    # 询问是否测试控制
    response = input("\n是否测试运动控制? (y/n): ").strip().lower()
    if response == 'y':
        print("\n自动运动测试...")
        print("请观察 Isaac Sim 中的小车...")
        
        # 前进
        print("\n1. 前进 (2秒)")
        tester.publish_twist(0.5, 0.0)
        time.sleep(2.0)
        tester.publish_twist(0.0, 0.0)
        print("✓ 完成")
        
        # 后退
        print("\n2. 后退 (2秒)")
        tester.publish_twist(-0.5, 0.0)
        time.sleep(2.0)
        tester.publish_twist(0.0, 0.0)
        print("✓ 完成")
        
        # 左转
        print("\n3. 左转 (2秒)")
        tester.publish_twist(0.3, 0.5)
        time.sleep(2.0)
        tester.publish_twist(0.0, 0.0)
        print("✓ 完成")
        
        # 右转
        print("\n4. 右转 (2秒)")
        tester.publish_twist(0.3, -0.5)
        time.sleep(2.0)
        tester.publish_twist(0.0, 0.0)
        print("✓ 完成")
    
    # 交互式控制
    print_menu()
    print("\n进入交互式控制模式...")
    print("按键: w=前进, s=后退, a=左转, d=右转, q=左旋转, e=右旋转, space=停止, p=状态, x=退出")
    
    try:
        import termios
        import tty
        import select
        
        # 设置终端
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == 'x':
                    print("\n退出")
                    break
                
                elif key == ' ':
                    tester.publish_twist(0.0, 0.0)
                    print("\r停止", end='', flush=True)
                
                elif key == 'p':
                    print(f"\n里程计: {tester.odom_count}, RGB: {tester.rgb_count}, IMU: {tester.imu_count}, LiDAR: {tester.lidar_count}")
                
                elif key == 'w':
                    tester.publish_twist(0.5, 0.0)
                    print(f"\r前进... (按空格停止)", end='', flush=True)
                elif key == 's':
                    tester.publish_twist(-0.5, 0.0)
                    print(f"\r后退... (按空格停止)", end='', flush=True)
                elif key == 'a':
                    tester.publish_twist(0.3, 0.5)
                    print(f"\r左转... (按空格停止)", end='', flush=True)
                elif key == 'd':
                    tester.publish_twist(0.3, -0.5)
                    print(f"\r右转... (按空格停止)", end='', flush=True)
                elif key == 'q':
                    tester.publish_twist(0.0, 0.5)
                    print(f"\r左旋转... (按空格停止)", end='', flush=True)
                elif key == 'e':
                    tester.publish_twist(0.0, -0.5)
                    print(f"\r右旋转... (按空格停止)", end='', flush=True)
        
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
    except ImportError:
        print("\n简化输入模式")
        while True:
            choice = input("\n命令: ").strip().lower()
            
            if choice == 'x':
                break
            elif choice == 'w':
                tester.publish_twist(0.5, 0.0)
            elif choice == 's':
                tester.publish_twist(-0.5, 0.0)
            elif choice == 'a':
                tester.publish_twist(0.3, 0.5)
            elif choice == 'd':
                tester.publish_twist(0.3, -0.5)
            elif choice == 'q':
                tester.publish_twist(0.0, 0.5)
            elif choice == 'e':
                tester.publish_twist(0.0, -0.5)
            elif choice in [' ', '']:
                tester.publish_twist(0.0, 0.0)
    
    # 清理
    tester.publish_twist(0.0, 0.0)
    tester.destroy_node()
    rclpy.shutdown()
    print("\n程序结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")




