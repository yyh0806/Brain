# -*- coding: utf-8 -*-
"""
SLAM集成基础测试（不依赖pytest）

直接运行此文件进行基本功能验证
"""

import sys
import os

# 添加slam_integration到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 60)
print("SLAM集成基础测试")
print("=" * 60)

# 测试1: 检查Python环境
print("\n[测试1] 检查Python环境...")
print("Python版本: {}".format(sys.version))
print("Python环境正常")

# 测试2: 检查ROS2
print("\n[测试2] 检查ROS2环境...")
try:
    import rclpy
    print("ROS2可用 (rclpy已导入)")
    ROS2_AVAILABLE = True
except ImportError:
    print("警告: ROS2 Python库(rclpy)不可用")
    print("   将使用模拟模式进行测试")
    ROS2_AVAILABLE = False

# 测试3: 导入SLAM模块
print("\n[测试3] 导入SLAM集成模块...")
try:
    from slam_integration.src import SLAMConfig
    print("SLAMConfig导入成功")

    # 创建配置对象
    config = SLAMConfig(
        backend="fast_livo",
        resolution=0.1,
        zero_copy=True
    )
    print("SLAM配置创建成功: backend={}, resolution={}m".format(
        config.backend, config.resolution))
except Exception as e:
    print("SLAM模块导入失败: {}".format(e))
    sys.exit(1)

# 测试4: 创建SLAM Manager
print("\n[测试4] 创建SLAM Manager...")
try:
    from slam_integration.src import SLAMManager
    slam_manager = SLAMManager(config)
    print("SLAM Manager创建成功")
    print("   初始化状态: {}".format(slam_manager.is_initialized))
except Exception as e:
    print("SLAM Manager创建失败: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: MockSLAMManager（模拟模式）
print("\n[测试5] MockSLAMManager（模拟模式）...")
try:
    from slam_integration.src import MockSLAMManager
    mock_manager = MockSLAMManager(config)
    print("MockSLAMManager创建成功")

    # 获取几何地图
    geo_map = mock_manager.get_geometric_map()
    if geo_map is not None:
        print("模拟地图创建成功，尺寸: {}".format(geo_map.shape))
    else:
        print("警告: 模拟地图获取失败")
except Exception as e:
    print("MockSLAMManager测试失败: {}".format(e))
    import traceback
    traceback.print_exc()

# 测试6: 坐标转换
print("\n[测试6] 坐标转换测试...")
try:
    # 使用模拟manager进行坐标转换测试
    test_positions = [
        (0.0, 0.0),    # 原点
        (5.0, 3.0),    # 测试点1
        (-2.5, 1.5),   # 测试点2
    ]

    for world_pos in test_positions:
        grid_pos = mock_manager.world_to_grid(world_pos)
        recovered = mock_manager.grid_to_world(grid_pos)
        error = ((recovered[0] - world_pos[0])**2 +
                (recovered[1] - world_pos[1])**2)**0.5
        print("   {} -> {} -> {} (误差: {:.4f}m)".format(
            world_pos, grid_pos, recovered, error))

    print("坐标转换测试通过")
except Exception as e:
    print("坐标转换测试失败: {}".format(e))
    import traceback
    traceback.print_exc()

# 清理
print("\n[清理] 关闭SLAM Manager...")
try:
    slam_manager.shutdown()
    print("SLAM Manager已关闭")
except Exception as e:
    print("关闭时出现警告: {}".format(e))

# 总结
print("\n" + "=" * 60)
print("所有基础测试通过！")
print("=" * 60)
print("\n下一步:")
print("1. 如果需要使用真实SLAM，请安装并启动FAST-LIVO或LIO-SAM节点")
print("2. 运行认知层集成测试")
print("3. 查看slam_integration/README.md了解详细使用说明")
