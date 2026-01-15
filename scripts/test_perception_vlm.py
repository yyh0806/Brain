#!/usr/bin/env python3
"""
VLM 感知测试 - 修正版

使用正确的 API 测试 VLM 场景理解功能

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


async def test_vlm_with_real_image():
    """使用真实 ROS2 图像测试 VLM"""
    logger.info("=" * 80)
    logger.info("VLM 场景理解测试")
    logger.info("=" * 80)

    try:
        from brain.perception.understanding.vlm_perception import VLMPerception
        import cv2

        # 初始化 VLM
        logger.info("初始化 VLM (llava:7b)...")
        vlm = VLMPerception(model="llava:7b")
        logger.success("✓ VLM 初始化成功")

        # 尝试使用 ROS2 图像
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        # 初始化 ROS2
        os.environ['ROS_DOMAIN_ID'] = '42'
        rclpy.init()

        # 创建节点
        node = Node('vlm_test_node')
        cv_bridge = CvBridge()

        # 存储最新图像
        latest_image = None

        def image_callback(msg):
            nonlocal latest_image
            try:
                cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                latest_image = cv_image
                logger.success(f"✓ 收到图像: {msg.width}x{msg.height}")
            except Exception as e:
                logger.error(f"图像转换失败: {e}")

        # 订阅话题
        node.create_subscription(
            Image,
            '/front_stereo_camera/left/image_raw',
            image_callback,
            10
        )
        logger.info("订阅话题: /front_stereo_camera/left/image_raw")

        # 等待图像 (10秒)
        logger.info("\n等待图像 (10秒)...")
        start_time = rclpy.clock.Clock().now()

        while (rclpy.clock.Clock().now() - start_time).nanoseconds / 1e9 < 10:
            rclpy.spin_once(node, timeout_sec=0.1)
            if latest_image is not None:
                break

        # 使用图像或测试图像
        if latest_image is not None:
            test_image = latest_image
            logger.success("使用真实 ROS2 图像")
        else:
            # 创建测试图像
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            test_image[100:380, 100:540] = [200, 200, 200]  # 白色背景
            test_image[150:200, 200:280] = [255, 0, 0]      # 红色方块
            test_image[250:300, 350:430] = [0, 255, 0]      # 绿色方块
            test_image[300:350, 200:280] = [0, 0, 255]      # 蓝色方块
            logger.warning("使用测试图像（未收到 ROS2 图像）")

        # 保存测试图像
        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        test_image_path = output_dir / "vlm_test_image.png"
        cv2.imwrite(str(test_image_path), test_image)
        logger.success(f"测试图像已保存: {test_image_path}")

        # 测试 1: 场景描述
        logger.info("\n" + "=" * 80)
        logger.info("测试 1: 场景描述")
        logger.info("=" * 80)

        scene_desc = await vlm.describe_scene(test_image)
        logger.success("✓ 场景描述完成")
        logger.info(f"场景总结: {scene_desc.summary}")
        logger.info(f"检测到 {len(scene_desc.objects)} 个物体")
        logger.info(f"空间关系: {scene_desc.spatial_relations}")
        logger.info(f"导航提示: {scene_desc.navigation_hints}")

        # 测试 2: 查找目标
        logger.info("\n" + "=" * 80)
        logger.info("测试 2: 查找目标")
        logger.info("=" * 80)

        target_result = await vlm.find_target(
            test_image,
            target_description="red box",
            context="Find a red colored object in the image"
        )

        logger.success("✓ 目标查找完成")
        logger.info(f"找到目标: {target_result.found}")
        logger.info(f"目标描述: {target_result.target_description}")
        logger.info(f"匹配物体数: {len(target_result.matched_objects)}")

        if target_result.best_match:
            obj = target_result.best_match
            logger.info(f"最佳匹配:")
            logger.info(f"  - 标签: {obj.label}")
            logger.info(f"  - 置信度: {obj.confidence:.2f}")
            logger.info(f"  - 位置: {obj.position_description}")
            logger.info(f"  - 距离: {obj.estimated_distance}")

        # 测试 3: 空间查询
        logger.info("\n" + "=" * 80)
        logger.info("测试 3: 空间查询")
        logger.info("=" * 80)

        query_result = await vlm.answer_spatial_query(
            test_image,
            query="Where is the red box located in the image?"
        )

        logger.success("✓ 空间查询完成")
        logger.info(f"回答: {query_result.answer}")
        logger.info(f"置信度: {query_result.confidence:.2f}")
        if query_result.referenced_objects:
            logger.info(f"引用物体: {len(query_result.referenced_objects)} 个")

        # 清理 ROS2
        rclpy.shutdown()

        # 保存结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'vlm_model': 'llava:7b',
            'test_image_path': str(test_image_path),
            'scene_description': {
                'summary': scene_desc.summary,
                'objects': [obj.to_dict() for obj in scene_desc.objects],
                'spatial_relations': scene_desc.spatial_relations,
                'navigation_hints': scene_desc.navigation_hints
            },
            'target_search': {
                'found': target_result.found,
                'target_description': target_result.target_description,
                'matched_objects': len(target_result.matched_objects),
                'suggested_action': target_result.suggested_action,
                'confidence': target_result.confidence
            },
            'spatial_query': {
                'answer': query_result.answer,
                'confidence': query_result.confidence,
                'referenced_objects': len(query_result.referenced_objects or [])
            }
        }

        result_file = output_dir / f"vlm_detailed_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.success(f"详细结果已保存: {result_file}")

        # 打印总结
        logger.info("\n" + "=" * 80)
        logger.info("VLM 测试总结")
        logger.info("=" * 80)
        logger.success("✓ VLM 初始化: 成功")
        logger.success("✓ 场景描述: 成功")
        logger.success("✓ 目标查找: 成功")
        logger.success("✓ 空间查询: 成功")

        return True

    except ImportError as e:
        logger.error(f"导入失败: {e}")
        logger.info("请确保已安装所需依赖:")
        logger.info("  pip install opencv-python ollama rclpy sensor-msgs cv-bridge")
        return False
    except Exception as e:
        logger.error(f"VLM 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("=" * 80)
    logger.info("VLM 感知测试 - 修正版")
    logger.info("=" * 80)
    logger.info(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', 'default')}")
    logger.info(f"Ollama 模型: llava:7b")
    logger.info("=" * 80)

    try:
        # 运行异步测试
        success = asyncio.run(test_vlm_with_real_image())

        if success:
            logger.success("\n✅ 所有测试通过！")
        else:
            logger.error("\n❌ 部分测试失败")

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
        return 130
    except Exception as e:
        logger.error(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

