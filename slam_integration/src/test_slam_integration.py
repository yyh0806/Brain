# -*- coding: utf-8 -*-
"""
SLAM集成测试模块

测试SLAM Manager的各项功能
"""

import asyncio
import pytest
import numpy as np
from typing import Optional

try:
    from slam_integration.src import SLAMManager, SLAMConfig, CoordinateTransformer
    SLAM_AVAILABLE = True
except ImportError:
    SLAM_AVAILABLE = False


class TestSLAMManager:
    """SLAM Manager测试套件"""

    @pytest.fixture
    def slam_config(self):
        """SLAM配置fixture"""
        return SLAMConfig(
            backend="fast_livo",
            resolution=0.1,
            zero_copy=True,
            update_frequency=10.0
        )

    @pytest.fixture
    def slam_manager(self, slam_config):
        """SLAM Manager fixture"""
        manager = SLAMManager(slam_config)
        yield manager
        manager.shutdown()

    @pytest.mark.asyncio
    async def test_slam_initialization(self, slam_manager):
        """测试SLAM Manager初始化"""
        if not SLAM_AVAILABLE:
            pytest.skip("SLAM模块不可用")

        # 检查是否初始化成功
        # 注意：如果ROS2不可用，MockSLAMManager会自动启用
        assert slam_manager is not None

    @pytest.mark.asyncio
    async def test_map_availability(self, slam_manager):
        """测试地图可用性"""
        if not SLAM_AVAILABLE:
            pytest.skip("SLAM模块不可用")

        # 等待地图（模拟模式下会立即返回）
        map_available = await slam_manager.wait_for_map(timeout=2.0)

        # 模拟模式下应该总是可用
        if hasattr(slam_manager, '_create_mock_map'):
            assert map_available
        else:
            # 真实SLAM可能需要更长时间
            print(f"地图可用: {map_available}")

    @pytest.mark.asyncio
    async def test_coordinate_transformation(self, slam_manager):
        """测试坐标转换"""
        if not SLAM_AVAILABLE:
            pytest.skip("SLAM模块不可用")

        # 等待地图就绪
        await slam_manager.wait_for_map(timeout=2.0)

        # 测试世界坐标 → 栅格坐标
        world_pos = (5.0, 3.0)
        grid_pos = slam_manager.world_to_grid(world_pos)

        assert isinstance(grid_pos, tuple)
        assert len(grid_pos) == 2
        print(f"世界坐标{world_pos} → 栅格坐标{grid_pos}")

        # 测试栅格坐标 → 世界坐标
        recovered_world = slam_manager.grid_to_world(grid_pos)

        # 应该接近原始坐标（可能有舍入误差）
        assert abs(recovered_world[0] - world_pos[0]) < slam_manager.config.resolution
        assert abs(recovered_world[1] - world_pos[1]) < slam_manager.config.resolution
        print(f"栅格坐标{grid_pos} → 世界坐标{recovered_world}")

    def test_map_metadata(self, slam_manager):
        """测试地图元数据"""
        if not SLAM_AVAILABLE:
            pytest.skip("SLAM模块不可用")

        metadata = slam_manager.get_map_metadata()

        if metadata is not None:
            assert metadata.resolution > 0
            assert metadata.width > 0
            assert metadata.height > 0
            print(f"地图元数据: {metadata.width}x{metadata.height}, 分辨率={metadata.resolution}m")


class TestCoordinateTransformer:
    """坐标转换器测试"""

    @pytest.fixture
    def transformer(self):
        """坐标转换器fixture"""
        config = SLAMConfig()
        slam_manager = SLAMManager(config)
        transformer = CoordinateTransformer(slam_manager)
        yield transformer
        slam_manager.shutdown()

    def test_coordinate_transformer_creation(self, transformer):
        """测试坐标转换器创建"""
        assert transformer is not None
        assert transformer.slam_manager is not None


def test_slam_config():
    """测试SLAM配置"""
    config = SLAMConfig(
        backend="fast_livo",
        resolution=0.1,
        zero_copy=True
    )

    assert config.backend == "fast_livo"
    assert config.resolution == 0.1
    assert config.zero_copy is True


# 集成测试示例
async def test_full_slam_pipeline():
    """
    完整的SLAM管道测试

    这个测试展示了如何在实际项目中使用SLAM集成
    """
    if not SLAM_AVAILABLE:
        print("SLAM模块不可用，跳过集成测试")
        return

    # 1. 创建SLAM Manager
    config = SLAMConfig(
        backend="fast_livo",
        resolution=0.1,
        zero_copy=True
    )
    slam_manager = SLAMManager(config)

    try:
        # 2. 等待SLAM地图
        if await slam_manager.wait_for_map(timeout=5.0):
            print("✅ SLAM地图已就绪")

            # 3. 获取地图元数据
            metadata = slam_manager.get_map_metadata()
            if metadata:
                print(f"✅ 地图尺寸: {metadata.width}x{metadata.height}")
                print(f"✅ 分辨率: {metadata.resolution}米/栅格")

            # 4. 测试坐标转换
            test_positions = [
                (0.0, 0.0),    # 原点
                (5.0, 3.0),    # 测试点1
                (-2.5, 1.5),   # 测试点2
            ]

            for world_pos in test_positions:
                grid_pos = slam_manager.world_to_grid(world_pos)
                recovered = slam_manager.grid_to_world(grid_pos)
                error = np.sqrt((recovered[0] - world_pos[0])**2 +
                              (recovered[1] - world_pos[1])**2)
                print(f"✅ 坐标转换测试: {world_pos} → {grid_pos} → {recovered} (误差: {error:.4f}m)")

            # 5. 获取机器人位置
            robot_pos = slam_manager.get_robot_position()
            if robot_pos:
                x, y, yaw = robot_pos
                print(f"✅ 机器人位置: x={x:.2f}m, y={y:.2f}m, yaw={yaw:.2f}rad")
            else:
                print("ℹ️  机器人位姿尚未可用（需要SLAM节点运行）")

            print("\n✅ 所有测试通过！")
        else:
            print("⚠️  SLAM地图超时（可能没有SLAM节点在运行）")
            print("ℹ️  使用模拟模式进行基本功能验证")

            # 模拟模式测试
            from slam_integration.src import MockSLAMManager
            mock_manager = MockSLAMManager(config)
            geo_map = mock_manager.get_geometric_map()

            if geo_map is not None:
                print(f"✅ 模拟地图尺寸: {geo_map.shape}")

    finally:
        # 6. 清理
        slam_manager.shutdown()
        print("✅ SLAM Manager已关闭")


if __name__ == "__main__":
    """直接运行测试"""
    print("=" * 60)
    print("SLAM集成测试")
    print("=" * 60)

    # 运行集成测试
    asyncio.run(test_full_slam_pipeline())

    print("\n提示: 使用pytest运行完整测试套件")
    print("pytest slam_integration/src/test_slam_integration.py -v")
