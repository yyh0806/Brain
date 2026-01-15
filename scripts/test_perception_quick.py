#!/usr/bin/env python3
"""
快速测试感知模块 - 简化版

功能:
1. 连接到 ROS2 (ROS_DOMAIN_ID=42)
2. 订阅传感器话题
3. 测试基本感知功能
4. 快速验证系统可用性

Author: Brain Development Team
Date: 2025-01-08
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
import json

# 设置 ROS_DOMAIN_ID
os.environ['ROS_DOMAIN_ID'] = '42'

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

# 感知模块导入
from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
from brain.perception.understanding.vlm_perception import VLMPerception


class QuickTestNode(Node):
    """快速测试节点"""

    def __init__(self):
        super().__init__('perception_quick_test')
        self.cv_bridge = CvBridge()
        self.data_received = {
            'rgb': False,
            'pointcloud': False,
            'imu': False,
            'odom': False
        }
        self.last_data = {}

        # 订阅话题
        self.create_subscription(Image, '/front_stereo_camera/left/image_raw', self.rgb_cb, 10)
        self.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points', self.pc_cb, 10)
        self.create_subscription(Imu, '/chassis/imu', self.imu_cb, 10)
        self.create_subscription(Odometry, '/chassis/odom', self.odom_cb, 10)

        logger.info("快速测试节点已启动")

    def rgb_cb(self, msg):
        if not self.data_received['rgb']:
            self.data_received['rgb'] = True
            logger.success(f"✓ 收到 RGB 图像: {msg.width}x{msg.height}")

    def pc_cb(self, msg):
        if not self.data_received['pointcloud']:
            self.data_received['pointcloud'] = True
            points = len(msg.data) // 16  # 4 floats * 4 bytes
            logger.success(f"✓ 收到点云: {points} 个点")

    def imu_cb(self, msg):
        if not self.data_received['imu']:
            self.data_received['imu'] = True
            logger.success("✓ 收到 IMU 数据")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_data['odom'] = (x, y)
        if not self.data_received['odom']:
            self.data_received['odom'] = True
            logger.success(f"✓ 收到里程计: ({x:.3f}, {y:.3f})")


async def test_vlm_quick():
    """快速测试 VLM"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 VLM (llava:7b)")
    logger.info("=" * 60)

    try:
        vlm = VLMPerception(model="llava:7b")
        logger.success("VLM 初始化成功")

        # 创建一个简单的测试图像（如果没有真实图像）
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        logger.info("测试场景描述...")
        description = await vlm.describe_scene(test_image)
        logger.info(f"场景描述: {description[:100]}...")

        logger.success("VLM 测试通过")
        return True

    except Exception as e:
        logger.error(f"VLM 测试失败: {e}")
        return False


def main():
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("=" * 60)
    logger.info("感知模块快速测试")
    logger.info("=" * 60)
    logger.info(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', 'default')}")

    # 初始化 ROS2
    rclpy.init()
    
    try:
        # 创建节点
        node = QuickTestNode()

        # 等待数据 (10秒)
        logger.info("\n等待传感器数据 (10秒)...")
        start = time.time()

        while time.time() - start < 10:
            rclpy.spin_once(node, timeout_sec=0.1)

        # 检查结果
        logger.info("\n" + "=" * 60)
        logger.info("数据接收状态")
        logger.info("=" * 60)

        all_received = True
        for sensor, received in node.data_received.items():
            status = "✓" if received else "✗"
            logger.info(f"{status} {sensor.upper()}: {'收到' if received else '未收到'}")
            if not received:
                all_received = False

        # 测试 VLM
        if all_received:
            logger.info("\n所有传感器数据已接收，测试 VLM...")
            
            # 测试 VLM（需要 asyncio）
            import asyncio
            try:
                vlm_ok = asyncio.run(test_vlm_quick())
            except KeyboardInterrupt:
                logger.info("\nVLM 测试被中断")
                vlm_ok = False
        else:
            logger.warning("\n部分传感器数据缺失，跳过 VLM 测试")
            vlm_ok = None

        # 总结
        logger.info("\n" + "=" * 60)
        logger.info("测试总结")
        logger.info("=" * 60)

        if all_received:
            logger.success("✓ 传感器数据接收: 通过")
        else:
            logger.warning("✗ 传感器数据接收: 部分缺失")

        if vlm_ok:
            logger.success("✓ VLM 功能: 通过")
        elif vlm_ok is False:
            logger.error("✗ VLM 功能: 失败")
        else:
            logger.info("- VLM 功能: 跳过")

        # 保存结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', 'default'),
            'sensor_data_received': node.data_received,
            'last_odom': node.last_data.get('odom'),
            'vlm_test': 'passed' if vlm_ok else ('failed' if vlm_ok is False else 'skipped')
        }

        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.success(f"结果已保存到: {result_file}")

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

