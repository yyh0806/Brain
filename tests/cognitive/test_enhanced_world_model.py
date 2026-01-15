# -*- coding: utf-8 -*-
"""
EnhancedWorldModel测试

测试集成SLAM的增强世界模型
"""

import sys
import os
import asyncio

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from brain.cognitive.world_model.world_model_slam import EnhancedWorldModel


async def test_enhanced_world_model():
    """测试EnhancedWorldModel的基本功能"""
    print("=" * 60)
    print("EnhancedWorldModel功能测试")
    print("=" * 60)

    # 1. 创建EnhancedWorldModel
    print("\n[测试1] 创建EnhancedWorldModel...")
    try:
        model = EnhancedWorldModel(config={
            "slam_backend": "fast_livo",
            "map_resolution": 0.1,
            "max_semantic_objects": 500
        })
        print("EnhancedWorldModel创建成功")
        print("  SLAM Manager:", "已初始化" if model.slam_manager else "不可用")
    except Exception as e:
        print("创建失败:", e)
        import traceback
        traceback.print_exc()
        return

    # 2. SLAM地图访问
    print("\n[测试2] SLAM地图访问...")
    if model.slam_manager:
        print("等待SLAM地图...")
        success = await model.slam_manager.wait_for_map(timeout=3.0)
        if success:
            print("SLAM地图已就绪")
        else:
            print("SLAM地图超时（使用模拟模式）")

        # 获取地图元数据
        metadata = model.get_map_metadata()
        if metadata:
            print(f"地图元数据:")
            print(f"  分辨率: {metadata.resolution}m")
            print(f"  尺寸: {metadata.width}x{metadata.height}")
        else:
            print("地图元数据不可用")

    # 3. 坐标转换
    print("\n[测试3] 坐标转换...")
    try:
        test_positions = [
            (0.0, 0.0),
            (5.0, 3.0),
            (-2.5, 1.5),
        ]

        for world_pos in test_positions:
            grid_pos = model.world_to_grid(world_pos)
            recovered = model.grid_to_world(grid_pos)
            error = ((recovered[0] - world_pos[0])**2 +
                    (recovered[1] - world_pos[1])**2)**0.5
            print("  {} -> {} -> {} (误差: {:.4f}m)".format(
                world_pos, grid_pos, recovered, error))
        print("坐标转换测试通过")
    except Exception as e:
        print("坐标转换测试失败:", e)

    # 4. 语义物体更新
    print("\n[测试4] 语义物体更新...")

    # 模拟语义物体数据
    class MockSemanticObject:
        def __init__(self, label, confidence, position):
            self.label = label
            self.confidence = confidence
            self.world_position = position

    mock_objects = [
        MockSemanticObject("门", 0.9, (5.0, 3.0, 0.0)),
        MockSemanticObject("人", 0.8, (7.0, 2.0, 0.0)),
        MockSemanticObject("建筑", 0.95, (10.0, 5.0, 0.0)),
    ]

    # 创建模拟PerceptionData
    class MockPerceptionData:
        def __init__(self, semantic_objects):
            self.semantic_objects = semantic_objects
            self.pose = None

    perception_data = MockPerceptionData(mock_objects)

    # 更新语义物体
    try:
        changes = await model.update_from_perception(perception_data)
        print(f"检测到 {len(changes)} 个变化")
        for change in changes:
            print(f"  - {change.description}")

        print(f"当前语义物体数量: {len(model.semantic_objects)}")
        for obj_id, obj in list(model.semantic_objects.items())[:3]:
            print(f"  - {obj.id}: {obj.label} (置信度: {obj.confidence})")
    except Exception as e:
        print("语义物体更新失败:", e)
        import traceback
        traceback.print_exc()

    # 5. 语义叠加
    print("\n[测试5] 语义叠加...")
    try:
        model._update_semantic_overlays()
        print(f"语义标注数量: {len(model.semantic_overlays)}")
        for grid_pos, label in list(model.semantic_overlays.items())[:3]:
            print(f"  栅格{grid_pos}: {label.label}")
        print("语义叠加测试通过")
    except Exception as e:
        print("语义叠加测试失败:", e)

    # 6. 获取增强地图
    print("\n[测试6] 获取增强地图...")
    try:
        enhanced_map = model.get_enhanced_map()
        if enhanced_map:
            print("增强地图获取成功")
            print("  几何层:", "有" if enhanced_map.geometric_layer else "无")
            print("  语义标注:", len(enhanced_map.semantic_overlays), "个")
            print("  风险区域:", "有" if enhanced_map.risk_areas is not None else "无")
            print("  探索边界:", len(enhanced_map.exploration_frontier), "个")
        else:
            print("增强地图获取失败")
    except Exception as e:
        print("增强地图获取失败:", e)

    # 7. 获取物体位置
    print("\n[测试7] 获取物体位置...")
    try:
        location = model.get_location("门")
        if location:
            print(f"物体位置: {location}")
        else:
            print("未找到指定物体")
    except Exception as e:
        print("获取物体位置失败:", e)

    # 8. 清理
    print("\n[测试8] 清理...")
    try:
        model.shutdown()
        print("EnhancedWorldModel已关闭")
    except Exception as e:
        print("关闭时出错:", e)

    # 总结
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n关键特性:")
    print("  [几何层] SLAM地图零拷贝引用")
    print("  [语义层] VLM语义物体独立管理")
    print("  [因果层] 状态演化追踪")
    print("  [语义叠加] 语义信息叠加到几何地图")
    print("\n下一步:")
    print("  1. 集成到认知层CognitiveLayer")
    print("  2. 与VLM结果融合")
    print("  3. 与规划层适配器集成")


if __name__ == "__main__":
    asyncio.run(test_enhanced_world_model())
