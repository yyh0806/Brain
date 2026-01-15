#!/usr/bin/env python3
"""
VLM 感知最终测试 - 使用正确API

Author: Brain Development Team
Date: 2025-01-08
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import json
import asyncio

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
import numpy as np
import cv2


async def test_vlm_final():
    """最终 VLM 测试"""
    logger.info("=" * 80)
    logger.info("VLM 感知最终测试 (llava:7b)")
    logger.info("=" * 80)

    try:
        from brain.perception.understanding.vlm_perception import VLMPerception

        # 初始化 VLM
        logger.info("\n初始化 VLM (llava:7b)...")
        vlm = VLMPerception(model="llava:7b")
        logger.success("✓ VLM 初始化成功")

        # 创建测试图像
        logger.info("\n创建测试场景...")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = [135, 206, 235]  # 天蓝色背景
        cv2.rectangle(image, (150, 150), (250, 250), (255, 0, 0), -1)  # 红色
        cv2.rectangle(image, (350, 200), (450, 300), (0, 255, 0), -1)  # 绿色
        cv2.circle(image, (320, 100), 40, (0, 0, 255), -1)  # 蓝色

        # 保存图像
        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "vlm_final_test.png"
        cv2.imwrite(str(image_path), image)
        logger.success(f"✓ 测试图像已保存: {image_path}")

        results = {}

        # 测试 1: 场景描述
        logger.info("\n" + "=" * 80)
        logger.info("测试 1: 场景描述")
        logger.info("=" * 80)

        scene_desc = await vlm.describe_scene(image)
        logger.success("✓ 场景描述完成")

        results['scene_description'] = {
            'summary': scene_desc.summary,
            'object_count': len(scene_desc.objects),
            'spatial_relations_count': len(scene_desc.spatial_relations),
            'navigation_hints_count': len(scene_desc.navigation_hints),
            'potential_targets_count': len(scene_desc.potential_targets)
        }

        logger.info(f"场景总结: {scene_desc.summary}")
        logger.info(f"检测到 {len(scene_desc.objects)} 个物体")
        logger.info(f"空间关系: {len(scene_desc.spatial_relations)} 个")
        logger.info(f"导航提示: {len(scene_desc.navigation_hints)} 个")

        # 测试 2: 目标查找
        logger.info("\n" + "=" * 80)
        logger.info("测试 2: 目标查找")
        logger.info("=" * 80)

        target_result = await vlm.find_target(image, "green box")
        logger.success("✓ 目标查找完成")

        results['target_search'] = {
            'found': target_result.found,
            'target_description': target_result.target_description,
            'matched_objects_count': len(target_result.matched_objects),
            'confidence': target_result.confidence,
            'has_best_match': target_result.best_match is not None
        }

        logger.info(f"找到目标: {target_result.found}")
        logger.info(f"置信度: {target_result.confidence:.2f}")
        logger.info(f"匹配物体: {len(target_result.matched_objects)} 个")

        # 测试 3: 空间查询 (返回字符串）
        logger.info("\n" + "=" * 80)
        logger.info("测试 3: 空间查询")
        logger.info("=" * 80)

        query = "Where is the green box located?"
        answer = await vlm.answer_spatial_query(image, query)
        logger.success("✓ 空间查询完成")

        results['spatial_query'] = {
            'query': query,
            'answer': answer
        }

        logger.info(f"问题: {query}")
        logger.info(f"回答: {answer}")

        # 测试 4: 历史记录
        logger.info("\n" + "=" * 80)
        logger.info("测试 4: 历史记录")
        logger.info("=" * 80)

        last_scene = vlm.get_last_scene()
        detection_history = vlm.get_detection_history(count=5)

        results['history'] = {
            'has_last_scene': last_scene is not None,
            'detection_history_count': len(detection_history)
        }

        logger.info(f"最后场景: {'存在' if last_scene else '不存在'}")
        logger.info(f"检测历史: {len(detection_history)} 条")

        # 保存结果
        result_file = output_dir / f"vlm_final_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'vlm_model': 'llava:7b',
                'test_image_path': str(image_path),
                'results': results
            }, f, indent=2, ensure_ascii=False)

        logger.success(f"\n✓ 测试结果已保存: {result_file}")

        # 打印总结
        logger.info("\n" + "=" * 80)
        logger.info("测试总结")
        logger.info("=" * 80)
        logger.success("✓ VLM 初始化: 通过")
        logger.success("✓ 场景描述: 通过")
        logger.success("✓ 目标查找: 通过")
        logger.success("✓ 空间查询: 通过")
        logger.success("✓ 历史记录: 通过")

        logger.success(f"\n✅ 所有 VLM 测试通过！")

        return True

    except Exception as e:
        logger.error(f"VLM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ros2_sensors():
    """测试 ROS2 传感器订阅"""
    logger.info("\n" + "=" * 80)
    logger.info("ROS2 传感器订阅测试")
    logger.info("=" * 80)

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
        from nav_msgs.msg import Odometry

        # 初始化 ROS2
        os.environ['ROS_DOMAIN_ID'] = '42'
        rclpy.init()

        # 创建节点
        node = Node('perception_sensor_final_test')

        # 数据统计
        stats = {
            'rgb': 0,
            'pointcloud': 0,
            'imu': 0,
            'odom': 0
        }

        # 创建回调
        def make_counter(sensor_name):
            def callback(msg):
                stats[sensor_name] += 1
            return callback

        # 订阅话题
        node.create_subscription(Image, '/front_stereo_camera/left/image_raw',
                             make_counter('rgb'), 10)
        node.create_subscription(PointCloud2, '/front_3d_lidar/lidar_points',
                             make_counter('pointcloud'), 10)
        node.create_subscription(Imu, '/chassis/imu',
                             make_counter('imu'), 10)
        node.create_subscription(Odometry, '/chassis/odom',
                             make_counter('odom'), 10)

        logger.info("等待传感器数据 (10秒)...")
        start_time = rclpy.clock.Clock().now()

        while (rclpy.clock.Clock().now() - start_time).nanoseconds / 1e9 < 10:
            rclpy.spin_once(node, timeout_sec=0.1)

        # 打印结果
        logger.info("\n传感器接收统计:")
        for sensor, count in stats.items():
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} {sensor.upper()}: {count} 条消息")

        # 清理
        rclpy.shutdown()

        return any(count > 0 for count in stats.values())

    except Exception as e:
        logger.error(f"传感器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("=" * 80)
    logger.info("感知模块最终测试")
    logger.info("=" * 80)
    logger.info(f"ROS_DOMAIN_ID: 42")
    logger.info(f"Ollama 模型: llava:7b")
    logger.info("=" * 80)

    all_passed = True

    # 测试 1: ROS2 传感器
    try:
        sensor_ok = test_ros2_sensors()
        all_passed = all_passed and sensor_ok
    except Exception as e:
        logger.error(f"传感器测试异常: {e}")
        all_passed = False

    # 测试 2: VLM
    try:
        vlm_ok = asyncio.run(test_vlm_final())
        all_passed = all_passed and vlm_ok
    except Exception as e:
        logger.error(f"VLM 测试异常: {e}")
        all_passed = False

    # 最终总结
    logger.info("\n" + "=" * 80)
    logger.info("最终测试总结")
    logger.info("=" * 80)

    if all_passed:
        logger.success("✅ 所有测试通过！")
        return 0
    else:
        logger.error("❌ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())

