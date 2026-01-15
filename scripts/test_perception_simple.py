#!/usr/bin/env python3
"""
简化版感知测试 - 不依赖 cv_bridge

功能:
1. 连接到 ROS2 (ROS_DOMAIN_ID=42)
2. 订阅传感器话题（验证连接）
3. 测试 VLM 场景理解 (llava:7b)
4. 测试基本感知功能

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


def test_vlm_direct():
    """直接测试 VLM（不依赖 ROS2）"""
    logger.info("=" * 80)
    logger.info("测试 VLM (llava:7b) - 直接测试")
    logger.info("=" * 80)

    try:
        from brain.perception.understanding.vlm_perception import VLMPerception
        import numpy as np
        import asyncio

        # 创建 VLM 实例
        logger.info("初始化 VLM (llava:7b)...")
        vlm = VLMPerception(model="llava:7b")
        logger.success("✓ VLM 初始化成功")

        # 创建测试图像（使用简单图案）
        logger.info("创建测试图像...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # 创建一个简单的场景
        test_image[100:380, 100:540] = [200, 200, 200]  # 白色背景
        test_image[150:200, 200:280] = [255, 0, 0]      # 红色方块
        test_image[250:300, 350:430] = [0, 255, 0]      # 绿色方块
        test_image[300:350, 200:280] = [0, 0, 255]      # 蓝色方块

        # 保存测试图像
        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        test_image_path = output_dir / "test_image.png"
        cv2.imwrite(str(test_image_path), test_image)
        logger.success(f"✓ 测试图像已保存: {test_image_path}")

        # 异步测试 VLM
        async def run_vlm_tests():
            results = {}

            # 测试 1: 场景描述
            logger.info("\n测试 1: 场景描述...")
            try:
                scene_desc = await vlm.describe_scene(test_image)
                logger.success(f"✓ 场景描述: {scene_desc[:200]}...")
                results['scene_description'] = scene_desc
            except Exception as e:
                logger.error(f"✗ 场景描述失败: {e}")
                results['scene_description'] = f"Error: {str(e)}"

            # 测试 2: 查找物体
            logger.info("\n测试 2: 查找物体...")
            try:
                objects = await vlm.find_object(test_image, "box")
                logger.success(f"✓ 检测到 {len(objects)} 个盒子")
                for i, obj in enumerate(objects, 1):
                    logger.info(f"  物体 {i}: {obj.label} (置信度: {obj.confidence:.2f})")
                    logger.info(f"    位置: {obj.position_description}")
                    logger.info(f"    描述: {obj.description}")
                results['detected_objects'] = [
                    {
                        'label': obj.label,
                        'confidence': obj.confidence,
                        'position': obj.position_description,
                        'description': obj.description
                    }
                    for obj in objects
                ]
            except Exception as e:
                logger.error(f"✗ 查找物体失败: {e}")
                results['detected_objects'] = f"Error: {str(e)}"

            # 测试 3: 回答问题
            logger.info("\n测试 3: 回答问题...")
            try:
                answer = await vlm.ask_question(
                    test_image,
                    "What objects do you see in this image? Describe their colors and positions."
                )
                logger.success(f"✓ 回答: {answer[:300]}...")
                results['qa_answer'] = answer
            except Exception as e:
                logger.error(f"✗ 回答问题失败: {e}")
                results['qa_answer'] = f"Error: {str(e)}"

            return results

        # 运行测试
        results = asyncio.run(run_vlm_tests())

        # 保存结果
        result_file = output_dir / f"vlm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.success(f"✓ VLM 测试结果已保存: {result_file}")

        # 打印总结
        logger.info("\n" + "=" * 80)
        logger.info("VLM 测试总结")
        logger.info("=" * 80)
        logger.success("✓ VLM 初始化: 成功")
        logger.success("✓ 场景描述: 成功" if 'Error' not in results.get('scene_description', '') else "✗ 场景描述: 失败")
        logger.success("✓ 查找物体: 成功" if isinstance(results.get('detected_objects'), list) else "✗ 查找物体: 失败")
        logger.success("✓ 回答问题: 成功" if 'Error' not in results.get('qa_answer', '') else "✗ 回答问题: 失败")

        return True

    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.info("请确保已安装所需依赖:")
        logger.info("  pip install opencv-python ollama")
        return False
    except Exception as e:
        logger.error(f"VLM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sensor_subscriptions():
    """测试 ROS2 传感器订阅"""
    logger.info("=" * 80)
    logger.info("测试 ROS2 传感器订阅")
    logger.info("=" * 80)

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
        from nav_msgs.msg import Odometry

        # 初始化 ROS2
        rclpy.init()

        # 创建测试节点
        node = Node('perception_sensor_test')

        # 数据接收标志
        data_received = {
            'rgb': False,
            'pointcloud': False,
            'imu': False,
            'odom': False,
            'laser': False
        }

        # 统计
        message_count = {
            'rgb': 0,
            'pointcloud': 0,
            'imu': 0,
            'odom': 0,
            'laser': 0
        }

        # 创建回调函数
        def make_callback(sensor_name):
            def callback(msg):
                if not data_received[sensor_name]:
                    data_received[sensor_name] = True
                    logger.success(f"✓ {sensor_name}: 首次接收")
                message_count[sensor_name] += 1
            return callback

        # 订阅话题
        node.create_subscription(Image, '/front_stereo_camera/left/image_raw',
                             make_callback('rgb'), 10)
        logger.info("订阅: /front_stereo_camera/left/image_raw")

        node.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points',
                             make_callback('pointcloud'), 10)
        logger.info("订阅: /front_3d_lidar/lidar_points")

        node.create_subscription(Imu, '/chassis/imu',
                             make_callback('imu'), 10)
        logger.info("订阅: /chassis/imu")

        node.create_subscription(Odometry, '/chassis/odom',
                             make_callback('odom'), 10)
        logger.info("订阅: /chassis/odom")

        node.create_subscription(LaserScan, '/scan',
                             make_callback('laser'), 10)
        logger.info("订阅: /scan")

        # 等待数据
        logger.info("\n等待传感器数据 (15秒)...")
        start_time = time.time()

        while time.time() - start_time < 15:
            rclpy.spin_once(node, timeout_sec=0.1)

        # 打印结果
        logger.info("\n" + "=" * 80)
        logger.info("传感器订阅测试结果")
        logger.info("=" * 80)

        for sensor, received in data_received.items():
            status = "✓" if received else "✗"
            count = message_count[sensor]
            logger.info(f"{status} {sensor.upper()}: {'已接收' if received else '未接收'} ({count} 条消息)")

        all_received = all(data_received.values())
        any_received = any(data_received.values())

        # 保存结果
        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            'timestamp': datetime.now().isoformat(),
            'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', 'default'),
            'data_received': data_received,
            'message_count': message_count,
            'summary': {
                'all_sensors_available': all_received,
                'any_sensor_available': any_received,
                'available_sensors': [k for k, v in data_received.items() if v]
            }
        }

        result_file = output_dir / f"sensor_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.success(f"结果已保存: {result_file}")

        # 清理
        rclpy.shutdown()

        return any_received

    except Exception as e:
        logger.error(f"传感器订阅测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_perception_components():
    """测试感知组件"""
    logger.info("=" * 80)
    logger.info("测试感知组件")
    logger.info("=" * 80)

    results = {}

    try:
        # 测试 1: 占据栅格地图
        logger.info("\n测试 1: 占据栅格地图...")
        try:
            from brain.perception.mapping.occupancy_mapper import OccupancyMapper
            import numpy as np

            mapper = OccupancyMapper(resolution=0.1, map_size=50.0)
            logger.success("✓ 占据栅格地图初始化成功")

            # 模拟深度图更新
            depth_image = np.random.rand(480, 640) * 5.0
            camera_pose = {'x': 0, 'y': 0, 'z': 0.5, 'roll': 0, 'pitch': 0, 'yaw': 0}
            mapper.update_from_depth(depth_image, camera_pose)
            logger.success("✓ 从深度图更新成功")

            grid = mapper.get_map()
            occupied = np.sum(grid.data == 100)
            logger.info(f"  占据单元格: {occupied}")
            results['occupancy_map'] = 'success'
        except Exception as e:
            logger.error(f"✗ 占据栅格地图测试失败: {e}")
            results['occupancy_map'] = f'failed: {str(e)}'

        # 测试 2: 传感器融合
        logger.info("\n测试 2: 传感器融合...")
        try:
            from brain.perception.sensors.fusion import EKFPoseFusion, ObstacleDetector

            fusion = EKFPoseFusion()
            logger.success("✓ 位姿融合初始化成功")

            # 模拟数据
            odom_data = {'x': 0, 'y': 0, 'z': 0, 'qx': 0, 'qy': 0, 'qz': 0, 'qw': 1}
            imu_data = {'linear_acceleration': [0, 0, 9.8], 'angular_velocity': [0, 0, 0]}

            fusion.update_odom(odom_data)
            fusion.update_imu(imu_data)
            pose = fusion.get_pose()
            logger.success(f"✓ 位姿融合成功: ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")

            detector = ObstacleDetector()
            logger.success("✓ 障碍物检测器初始化成功")
            results['fusion'] = 'success'
        except Exception as e:
            logger.error(f"✗ 传感器融合测试失败: {e}")
            results['fusion'] = f'failed: {str(e)}'

        # 测试 3: 数据类型
        logger.info("\n测试 3: 数据类型...")
        try:
            from brain.perception.core.types import (
                Position3D, Pose3D, Velocity, BoundingBox
            )

            pos = Position3D(x=1.0, y=2.0, z=3.0)
            pose = Pose3D(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3)
            vel = Velocity(linear_x=1.0, linear_y=0.0, linear_z=0.0)

            logger.success("✓ 数据类型测试成功")
            results['data_types'] = 'success'
        except Exception as e:
            logger.error(f"✗ 数据类型测试失败: {e}")
            results['data_types'] = f'failed: {str(e)}'

        return results

    except Exception as e:
        logger.error(f"感知组件测试失败: {e}")
        return {}


def main():
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("=" * 80)
    logger.info("感知模块简化测试")
    logger.info("=" * 80)
    logger.info(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', 'default')}")
    logger.info(f"Ollama 模型: llava:7b")
    logger.info("=" * 80)

    # 创建输出目录
    output_dir = Path("/tmp/perception_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 测试结果汇总
    all_results = {}

    # 测试 1: VLM 直接测试
    logger.info("\n" + "▶" * 40)
    logger.info("开始测试 1: VLM 直接测试")
    logger.info("▶" * 40 + "\n")
    vlm_ok = test_vlm_direct()
    all_results['vlm_test'] = 'passed' if vlm_ok else 'failed'

    # 测试 2: ROS2 传感器订阅
    logger.info("\n" + "▶" * 40)
    logger.info("开始测试 2: ROS2 传感器订阅")
    logger.info("▶" * 40 + "\n")
    sensor_ok = test_sensor_subscriptions()
    all_results['sensor_test'] = 'passed' if sensor_ok else 'failed'

    # 测试 3: 感知组件
    logger.info("\n" + "▶" * 40)
    logger.info("开始测试 3: 感知组件")
    logger.info("▶" * 40 + "\n")
    component_results = test_perception_components()
    all_results['component_test'] = component_results

    # 打印最终总结
    logger.info("\n" + "=" * 80)
    logger.info("最终测试总结")
    logger.info("=" * 80)

    status_icon = lambda passed: "✅" if passed else "❌"

    logger.info(f"{status_icon(vlm_ok)} VLM 测试: {'通过' if vlm_ok else '失败'}")
    logger.info(f"{status_icon(sensor_ok)} 传感器订阅: {'通过' if sensor_ok else '失败'}")

    for component, result in component_results.items():
        passed = result == 'success'
        logger.info(f"{status_icon(passed)} {component}: {'通过' if passed else '失败'}")

    # 保存汇总结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', 'default'),
        'vlm_model': 'llava:7b',
        'test_results': all_results,
        'output_dir': str(output_dir)
    }

    summary_file = output_dir / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.success(f"\n测试汇总已保存: {summary_file}")
    logger.success(f"详细结果目录: {output_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试异常: {e}")
        import traceback
        traceback.print_exc()

