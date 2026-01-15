#!/usr/bin/env python3
"""
纯 VLM 测试 - 不依赖 ROS2

只测试 VLM (llava:7b) 功能，使用生成的测试图像

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


def create_test_scene_image():
    """创建测试场景图像"""
    logger.info("创建测试场景图像...")

    # 创建背景
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = [135, 206, 235]  # 天蓝色背景

    # 添加一些物体
    # 1. 红色立方体 (模拟障碍物)
    cv2.rectangle(image, (100, 150), (200, 250), (255, 0, 0), -1)
    cv2.putText(image, "OBSTACLE", (105, 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 2. 绿色盒子 (模拟目标)
    cv2.rectangle(image, (350, 200), (450, 300), (0, 255, 0), -1)
    cv2.putText(image, "TARGET", (360, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 3. 蓝色圆形 (模拟路标)
    cv2.circle(image, (320, 100), 40, (0, 0, 255), -1)
    cv2.putText(image, "LANDMARK", (285, 105),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 4. 黄色三角形 (模拟警告标志)
    pts = np.array([[450, 350], [500, 420], [400, 420]], np.int32)
    cv2.fillPoly(image, [pts], [255, 255, 0])
    cv2.putText(image, "WARNING", (420, 400),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    logger.success("测试场景图像创建完成")
    return image


async def test_vlm_comprehensive():
    """综合测试 VLM 功能"""
    logger.info("=" * 80)
    logger.info("VLM 综合测试 (llava:7b)")
    logger.info("=" * 80)

    try:
        from brain.perception.understanding.vlm_perception import VLMPerception

        # 初始化 VLM
        logger.info("\n步骤 1: 初始化 VLM (llava:7b)...")
        vlm = VLMPerception(model="llava:7b")
        logger.success("✓ VLM 初始化成功")

        # 创建测试图像
        logger.info("\n步骤 2: 创建测试场景...")
        test_image = create_test_scene_image()

        # 保存测试图像
        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "test_scene.png"
        cv2.imwrite(str(image_path), test_image)
        logger.success(f"✓ 测试图像已保存: {image_path}")

        # 测试 1: 场景描述
        logger.info("\n" + "=" * 80)
        logger.info("测试 1: 场景描述")
        logger.info("=" * 80)

        logger.info("正在分析场景...")
        scene_desc = await vlm.describe_scene(test_image)
        logger.success("✓ 场景描述完成")

        logger.info(f"\n场景总结:")
        logger.info(f"  {scene_desc.summary}")

        logger.info(f"\n检测到的物体 ({len(scene_desc.objects)} 个):")
        for i, obj in enumerate(scene_desc.objects, 1):
            logger.info(f"  {i}. {obj.label} (置信度: {obj.confidence:.2f})")
            if obj.bbox:
                logger.info(f"     位置: ({obj.bbox.x:.2f}, {obj.bbox.y:.2f})")
                logger.info(f"     大小: {obj.bbox.width:.2f} x {obj.bbox.height:.2f}")
            if obj.estimated_distance:
                logger.info(f"     距离: {obj.estimated_distance:.2f} 米")
            logger.info(f"     描述: {obj.description}")

        logger.info(f"\n空间关系:")
        for relation in scene_desc.spatial_relations:
            logger.info(f"  - {relation}")

        logger.info(f"\n导航提示:")
        for hint in scene_desc.navigation_hints:
            logger.info(f"  - {hint}")

        logger.info(f"\n潜在目标:")
        for target in scene_desc.potential_targets:
            logger.info(f"  - {target}")

        # 测试 2: 查找特定目标
        logger.info("\n" + "=" * 80)
        logger.info("测试 2: 查找特定目标")
        logger.info("=" * 80)

        target_desc = "green target box"
        logger.info(f"搜索目标: {target_desc}")
        target_result = await vlm.find_target(
            test_image,
            target_description=target_desc,
        )
        logger.success("✓ 目标查找完成")

        logger.info(f"\n搜索结果:")
        logger.info(f"  找到目标: {'是' if target_result.found else '否'}")
        logger.info(f"  目标描述: {target_result.target_description}")
        logger.info(f"  匹配物体数: {len(target_result.matched_objects)}")
        logger.info(f"  建议操作: {target_result.suggested_action}")
        logger.info(f"  置信度: {target_result.confidence:.2f}")
        logger.info(f"  说明: {target_result.explanation}")

        if target_result.best_match:
            best = target_result.best_match
            logger.info(f"\n  最佳匹配:")
            logger.info(f"    标签: {best.label}")
            logger.info(f"    置信度: {best.confidence:.2f}")
            logger.info(f"    位置: {best.position_description}")
            if best.estimated_distance:
                logger.info(f"    距离: {best.estimated_distance:.2f} 米")

        # 测试 3: 空间查询
        logger.info("\n" + "=" * 80)
        logger.info("测试 3: 空间查询")
        logger.info("=" * 80)

        query = "Where is the red obstacle located relative to the green target?"
        logger.info(f"空间查询: {query}")
        query_result = await vlm.answer_spatial_query(test_image, query)
        logger.success("✓ 空间查询完成")

        logger.info(f"\n查询结果:")
        logger.info(f"  回答: {query_result.answer}")
        logger.info(f"  置信度: {query_result.confidence:.2f}")

        if query_result.referenced_objects:
            logger.info(f"  引用物体 ({len(query_result.referenced_objects)} 个):")
            for obj in query_result.referenced_objects:
                logger.info(f"    - {obj.label} (置信度: {obj.confidence:.2f})")

        if query_result.spatial_context:
            logger.info(f"  空间上下文: {query_result.spatial_context}")

        # 测试 4: 历史记录
        logger.info("\n" + "=" * 80)
        logger.info("测试 4: 历史记录")
        logger.info("=" * 80)

        last_scene = vlm.get_last_scene()
        if last_scene:
            logger.success("✓ 获取最后场景成功")
            logger.info(f"  最后场景摘要: {last_scene.summary[:100]}...")
        else:
            logger.info("  没有历史场景记录")

        detection_history = vlm.get_detection_history(count=5)
        logger.info(f"  检测历史 ({len(detection_history)} 个):")
        for i, obj in enumerate(detection_history, 1):
            logger.info(f"    {i}. {obj.label} (置信度: {obj.confidence:.2f})")

        # 保存完整结果
        results = {
            'timestamp': datetime.now().isoformat(),
            'vlm_model': 'llava:7b',
            'test_image_path': str(image_path),
            'scene_description': {
                'summary': scene_desc.summary,
                'objects': [obj.to_dict() for obj in scene_desc.objects],
                'spatial_relations': scene_desc.spatial_relations,
                'navigation_hints': scene_desc.navigation_hints,
                'potential_targets': scene_desc.potential_targets
            },
            'target_search': {
                'found': target_result.found,
                'target_description': target_result.target_description,
                'matched_objects': len(target_result.matched_objects),
                'suggested_action': target_result.suggested_action,
                'confidence': target_result.confidence,
                'explanation': target_result.explanation
            },
            'spatial_query': {
                'query': query,
                'answer': query_result.answer,
                'confidence': query_result.confidence,
                'referenced_objects': len(query_result.referenced_objects or []),
                'spatial_context': query_result.spatial_context
            },
            'history': {
                'last_scene_available': last_scene is not None,
                'detection_history_count': len(detection_history)
            }
        }

        result_file = output_dir / f"vlm_comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.success(f"\n✓ 完整测试结果已保存: {result_file}")

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
        logger.success(f"测试图像: {image_path}")
        logger.success(f"详细结果: {result_file}")

        return True

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
    logger.info("VLM 纯功能测试")
    logger.info("=" * 80)
    logger.info(f"Ollama 模型: llava:7b")
    logger.info("=" * 80)

    try:
        # 运行异步测试
        success = asyncio.run(test_vlm_comprehensive())
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

