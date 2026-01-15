# -*- coding: utf-8 -*-
"""
认知层性能基准测试

运行方式：
    python3 tests/performance/benchmark_cognitive.py

输出：
    - 性能指标报告
    - 性能对比分析
"""

import asyncio
import time
import sys
import os
import numpy as np
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from brain.cognitive.world_model.modular_world_model import ModularWorldModel
from brain.cognitive.reasoning.async_cot_engine import AsyncCoTEngine
from brain.cognitive.world_model.risk_calculator import RiskAreaCalculator
from brain.cognitive.world_model.change_detector import SemanticObjectChangeDetector
from brain.cognitive.world_model.memory_manager import SemanticObjectManager


class MockSemanticObject:
    """模拟语义物体"""
    def __init__(self, obj_id, label, position, confidence=0.8):
        self.id = obj_id
        self.label = label
        self.world_position = position
        self.confidence = confidence


class MockPose:
    """模拟位姿"""
    def __init__(self, x, y, z, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw


class MockPerceptionData:
    """模拟感知数据"""
    def __init__(self, semantic_objects=None, pose=None):
        self.semantic_objects = semantic_objects or []
        self.pose = pose or MockPose(0, 0, 0, 0)


class PerformanceBenchmark:
    """性能基准测试"""

    def __init__(self):
        self.results = {}

    def record(self, name: str, value: float, unit: str = "ms"):
        """记录性能指标"""
        if name not in self.results:
            self.results[name] = []
        self.results[name].append({"value": value, "unit": unit})

    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 70)
        print("认知层性能基准测试报告")
        print("=" * 70)

        for name, measurements in self.results.items():
            values = [m["value"] for m in measurements]
            unit = measurements[0]["unit"]

            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)

            print(f"\n{name}:")
            print(f"  平均: {avg:.2f} {unit}")
            print(f"  最小: {min_val:.2f} {unit}")
            print(f"  最大: {max_val:.2f} {unit}")
            print(f"  样本数: {len(values)}")

        print("\n" + "=" * 70)

    async def benchmark_world_model_update(self, num_objects: int = 100):
        """基准测试：世界模型更新"""
        print(f"\n[测试] 世界模型更新（{num_objects}个物体）...")

        model = ModularWorldModel()
        await model.initialize()

        semantic_objects = [
            MockSemanticObject(f"obj{i}", f"object{i}", (float(i), float(i)))
            for i in range(num_objects)
        ]

        perception_data = MockPerceptionData(
            semantic_objects=semantic_objects,
            pose=MockPose(0, 0, 0, 0)
        )

        # 预热
        await model.update_from_perception(perception_data)

        # 正式测试
        times = []
        for _ in range(10):
            start = time.time()
            await model.update_from_perception(perception_data)
            duration = (time.time() - start) * 1000  # ms
            times.append(duration)

        avg_time = sum(times) / len(times)
        self.record(f"WorldModel更新（{num_objects}对象）", avg_time, "ms")

        print(f"  平均更新时间: {avg_time:.2f} ms")
        print(f"  目标: <10 ms")
        print(f"  状态: {'✓ 通过' if avg_time < 10 else '✗ 未达标'}")

        return avg_time < 10

    async def benchmark_reasoning_cache_hit_rate(self):
        """基准测试：推理缓存命中率"""
        print(f"\n[测试] 推理缓存命中率...")

        engine = AsyncCoTEngine()
        engine.start()

        try:
            # 100个唯一查询
            unique_queries = [f"查询{i}" for i in range(100)]

            # 每个查询执行3次（1次原始，2次缓存）
            for query in unique_queries:
                await engine.reason(query, {}, "default")
                await engine.reason(query, {}, "default")
                await engine.reason(query, {}, "default")

            stats = engine.get_statistics()
            hit_rate = stats["cache_hit_rate"]

            self.record("推理缓存命中率", hit_rate * 100, "%")

            print(f"  缓存命中率: {hit_rate * 100:.1f}%")
            print(f"  目标: >70%")
            print(f"  状态: {'✓ 通过' if hit_rate > 0.7 else '✗ 未达标'}")

            return hit_rate > 0.7

        finally:
            engine.stop()

    async def benchmark_memory_usage(self):
        """基准测试：内存使用"""
        print(f"\n[测试] 内存使用稳定性...")

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            model = ModularWorldModel()
            await model.initialize()

            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 执行1000次更新
            for i in range(1000):
                perception_data = MockPerceptionData(
                    semantic_objects=[
                        MockSemanticObject(f"obj{i % 100}", "object", (float(i), float(i)))
                    ],
                    pose=MockPose(float(i), float(i), 0, 0)
                )
                await model.update_from_perception(perception_data)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory

            self.record("内存增长（1000次更新）", memory_growth, "MB")

            print(f"  内存增长: {memory_growth:.1f} MB")
            print(f"  目标: <100 MB")
            print(f"  状态: {'✓ 通过' if memory_growth < 100 else '✗ 未达标'}")

            return memory_growth < 100

        except ImportError:
            print("  跳过（psutil未安装）")
            return True

    async def benchmark_change_detector(self, num_objects: int = 1000):
        """基准测试：变化检测器"""
        print(f"\n[测试] 增量变化检测（{num_objects}个物体）...")

        detector = SemanticObjectChangeDetector()

        # 创建初始对象
        semantic_objects = {
            f"obj{i}": MockSemanticObject(f"obj{i}", f"object{i}", (float(i), float(i)))
            for i in range(num_objects)
        }

        # 初始更新
        detector.update_semantic_objects(semantic_objects)

        # 修改10%的对象
        for i in range(0, num_objects, 10):
            obj = MockSemanticObject(f"obj{i}", f"object{i}", (float(i) + 0.1, float(i)))
            semantic_objects[f"obj{i}"] = obj

        # 测量增量更新时间
        start = time.time()
        new, changed, removed = detector.update_semantic_objects(semantic_objects)
        duration = (time.time() - start) * 1000  # ms

        self.record(f"增量变化检测（{num_objects}对象）", duration, "ms")

        print(f"  检测时间: {duration:.2f} ms")
        print(f"  新增: {len(new)}, 变化: {len(changed)}, 移除: {len(removed)}")
        print(f"  目标: <50 ms")
        print(f"  状态: {'✓ 通过' if duration < 50 else '✗ 未达标'}")

        return duration < 50

    async def benchmark_memory_manager(self, num_objects: int = 10000):
        """基准测试：内存管理器"""
        print(f"\n[测试] 内存管理器（{num_objects}个对象）...")

        manager = SemanticObjectManager(
            max_objects=num_objects,
            object_ttl=300.0,
            position_threshold=2.0
        )

        # 添加对象
        start = time.time()
        for i in range(num_objects):
            obj = MockSemanticObject(f"obj{i}", "object", (float(i), float(i)))
            manager.add_or_update(obj)
        add_duration = (time.time() - start) * 1000  # ms

        # 查询对象
        start = time.time()
        for i in range(num_objects):
            manager.get(f"obj{i}")
        get_duration = (time.time() - start) * 1000  # ms

        self.record(f"内存管理器添加（{num_objects}对象）", add_duration, "ms")
        self.record(f"内存管理器查询（{num_objects}对象）", get_duration, "ms")

        print(f"  添加时间: {add_duration:.2f} ms")
        print(f"  查询时间: {get_duration:.2f} ms")
        print(f"  目标: 查询 <100 ms")
        print(f"  状态: {'✓ 通过' if get_duration < 100 else '✗ 未达标'}")

        return get_duration < 100

    async def benchmark_risk_calculation(self, map_size: int = 100):
        """基准测试：风险计算"""
        print(f"\n[测试] 风险地图计算（{map_size}x{map_size}）...")

        calculator = RiskAreaCalculator()

        geometric_map = np.zeros((map_size, map_size), dtype=np.int8)
        geometric_map[20:80, 20:80] = -1  # 未知区域

        semantic_objects = {
            "obj1": MockSemanticObject("person", (50.0, 50.0)),
            "obj2": MockSemanticObject("car", (60.0, 60.0)),
        }

        start = time.time()
        risk_map = calculator.compute_risk_map(
            geometric_map,
            semantic_objects,
            (50.0, 50.0)
        )
        duration = (time.time() - start) * 1000  # ms

        self.record(f"风险地图计算（{map_size}x{map_size}）", duration, "ms")

        print(f"  计算时间: {duration:.2f} ms")
        print(f"  目标: <100 ms")
        print(f"  状态: {'✓ 通过' if duration < 100 else '✗ 未达标'}")

        return duration < 100

    async def run_all(self):
        """运行所有基准测试"""
        print("\n开始认知层性能基准测试...")
        print("=" * 70)

        results = {}

        # 运行各项测试
        results["world_model_small"] = await self.benchmark_world_model_update(100)
        results["world_model_large"] = await self.benchmark_world_model_update(1000)
        results["reasoning_cache"] = await self.benchmark_reasoning_cache_hit_rate()
        results["memory_usage"] = await self.benchmark_memory_usage()
        results["change_detector"] = await self.benchmark_change_detector(1000)
        results["memory_manager"] = await self.benchmark_memory_manager(10000)
        results["risk_calculation"] = await self.benchmark_risk_calculation(100)

        # 打印摘要
        self.print_summary()

        # 总结
        print("\n测试总结:")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"  通过: {passed}/{total}")
        print(f"  通过率: {passed / total * 100:.0f}%")

        if passed == total:
            print("\n✓ 所有性能测试通过！")
        else:
            print("\n✗ 部分性能测试未达标，需要优化")

        return passed == total


async def main():
    """主函数"""
    benchmark = PerformanceBenchmark()
    success = await benchmark.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
