# -*- coding: utf-8 -*-
"""
异步CoT推理引擎测试
"""

import pytest
import asyncio
import time
from brain.cognitive.reasoning.async_cot_engine import (
    ReasoningRequest,
    ReasoningResult,
    AsyncCoTEngine
)


@pytest.mark.asyncio
class TestReasoningResult:
    """测试推理结果"""

    def test_creation(self):
        """测试创建推理结果"""
        result = ReasoningResult(
            query="测试查询",
            chain=["步骤1", "步骤2"],
            conclusion="结论",
            confidence=0.8,
            mode="default"
        )

        assert result.query == "测试查询"
        assert len(result.chain) == 2
        assert result.conclusion == "结论"
        assert result.confidence == 0.8
        assert result.mode == "default"
        assert result.from_cache is False


@pytest.mark.asyncio
class TestAsyncCoTEngine:
    """测试异步CoT推理引擎"""

    async def test_initialization(self):
        """测试初始化"""
        engine = AsyncCoTEngine()

        assert engine.max_queue_size == 10
        assert engine._total_requests == 0
        assert len(engine._queue) == 0

    async def test_start_and_stop(self):
        """测试启动和停止"""
        engine = AsyncCoTEngine()
        engine.start()

        assert engine._running is True
        assert engine._worker_thread is not None

        engine.stop()

        assert engine._running is False

    async def test_basic_reasoning(self):
        """测试基本推理"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            result = await engine.reason(
                query="门在哪里？",
                context={"objects": ["门", "人"]},
                mode="location"
            )

            assert result is not None
            assert result.query == "门在哪里？"
            assert len(result.chain) > 0
            assert result.confidence > 0
            assert result.from_cache is False
        finally:
            engine.stop()

    async def test_caching(self):
        """测试缓存功能"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            # 第一次推理
            result1 = await engine.reason(
                query="门在哪里？",
                context={"objects": ["门"]},
                mode="location"
            )

            assert result1.from_cache is False

            # 第二次推理（相同查询）
            result2 = await engine.reason(
                query="门在哪里？",
                context={"objects": ["门"]},
                mode="location"
            )

            assert result2.from_cache is True
            assert result2.query == result1.query

        finally:
            engine.stop()

    async def test_cache_hit_rate(self):
        """测试缓存命中率"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            # 发送重复查询
            for _ in range(3):
                await engine.reason("查询1", {}, "default")

            # 发送不同查询
            await engine.reason("查询2", {}, "default")

            stats = engine.get_statistics()
            assert stats["total_requests"] == 4
            # 3次缓存命中（后两次对查询1的请求）
            assert stats["cache_hits"] >= 2

        finally:
            engine.stop()

    async def test_priority_queue(self):
        """测试优先级队列"""
        engine = AsyncCoTEngine(max_queue_size=10, num_workers=1)
        engine.start()

        try:
            # 发送不同优先级的请求
            tasks = []
            for i in range(5):
                priority = i % 3  # 0, 1, 2, 0, 1
                task = engine.reason(
                    query=f"查询{i}",
                    context={},
                    mode="default",
                    priority=priority
                )
                tasks.append(task)

            # 等待所有任务完成
            results = await asyncio.gather(*tasks)

            assert len(results) == 5

        finally:
            engine.stop()

    async def test_concurrent_reasoning(self):
        """测试并发推理"""
        engine = AsyncCoTEngine(max_queue_size=20, num_workers=2)
        engine.start()

        try:
            # 并发发送多个推理请求
            tasks = [
                engine.reason(f"查询{i}", {}, "default")
                for i in range(10)
            ]

            start = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start

            assert len(results) == 10
            # 并发处理应该比串行快
            # 注意：这里只是验证能并发处理，不严格要求时间

        finally:
            engine.stop()

    async def test_queue_overflow(self):
        """测试队列溢出"""
        engine = AsyncCoTEngine(max_queue_size=3, num_workers=1)
        engine.start()

        try:
            # 发送超过队列大小的请求
            # 由于队列是deque(maxlen=3)，旧请求会被丢弃
            tasks = []
            for i in range(10):
                task = engine.reason(f"查询{i}", {}, "default", priority=i)
                tasks.append(task)

            # 等待所有任务完成或超时
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 部分任务可能失败
            successful = [r for r in results if isinstance(r, ReasoningResult)]
            assert len(successful) > 0

        finally:
            engine.stop()

    async def test_get_statistics(self):
        """测试获取统计信息"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            # 执行一些推理
            await engine.reason("查询1", {}, "default")
            await engine.reason("查询1", {}, "default")  # 缓存命中

            stats = engine.get_statistics()

            assert stats["total_requests"] == 2
            assert stats["queue_size"] >= 0
            assert stats["cache_size"] >= 1
            assert stats["cache_hits"] >= 1

        finally:
            engine.stop()

    async def test_reasoning_chain_generation(self):
        """测试推理链生成"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            # 测试位置查询
            result = await engine.reason(
                query="门在哪里？",
                context={"objects": ["门", "人", "建筑"]},
                mode="location"
            )

            assert len(result.chain) > 0
            assert "位置" in result.chain[0] or "分析" in result.chain[0]

            # 测试障碍物查询
            result = await engine.reason(
                query="检测障碍物",
                context={},
                mode="obstacle"
            )

            assert len(result.chain) > 0

        finally:
            engine.stop()


@pytest.mark.asyncio
class TestCacheKeyComputation:
    """测试缓存键计算"""

    async def test_different_queries_different_keys(self):
        """测试不同查询产生不同缓存键"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            key1 = engine._compute_cache_key("查询1", "default", {})
            key2 = engine._compute_cache_key("查询2", "default", {})

            assert key1 != key2

        finally:
            engine.stop()

    async def test_same_query_same_key(self):
        """测试相同查询产生相同缓存键"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            context = {"objects": ["门", "人"]}
            key1 = engine._compute_cache_key("查询", "default", context)
            key2 = engine._compute_cache_key("查询", "default", context)

            assert key1 == key2

        finally:
            engine.stop()

    async def test_different_modes_different_keys(self):
        """测试不同模式产生不同缓存键"""
        engine = AsyncCoTEngine()
        engine.start()

        try:
            context = {}
            key1 = engine._compute_cache_key("查询", "mode1", context)
            key2 = engine._compute_cache_key("查询", "mode2", context)

            assert key1 != key2

        finally:
            engine.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
