# -*- coding: utf-8 -*-
"""
内存管理器测试
"""

import pytest
import time
from brain.cognitive.world_model.memory_manager import (
    CacheEntry,
    LRUCache,
    MemoryManagedDict,
    SemanticObjectManager
)


class MockSemanticObject:
    """模拟语义物体"""
    def __init__(self, label, position):
        self.label = label
        self.world_position = position


class TestCacheEntry:
    """测试缓存条目"""

    def test_creation(self):
        """测试创建缓存条目"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            size_bytes=100,
            ttl_seconds=10.0
        )

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.size_bytes == 100
        assert entry.ttl_seconds == 10.0
        assert entry.access_count == 0
        assert entry.timestamp is not None


class TestLRUCache:
    """测试LRU缓存"""

    def test_initialization(self):
        """测试初始化"""
        cache = LRUCache(max_size=10)

        assert cache.max_size == 10
        assert cache.size() == 0
        assert cache._hits == 0
        assert cache._misses == 0

    def test_put_and_get(self):
        """测试存取操作"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        assert cache.size() == 3
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_lru_eviction(self):
        """测试LRU淘汰"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # 访问key1，使其成为最近使用
        cache.get("key1")

        # 添加第4个元素，应该淘汰最旧的key2
        cache.put("key4", "value4")

        assert cache.size() == 3
        assert cache.get("key1") == "value1"  # 仍然存在
        assert cache.get("key2") is None  # 被淘汰
        assert cache.get("key3") == "value3"  # 仍然存在
        assert cache.get("key4") == "value4"  # 新添加的

    def test_update_existing_key(self):
        """测试更新已存在的键"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key1", "value1_updated")

        assert cache.size() == 1
        assert cache.get("key1") == "value1_updated"

    def test_hit_and_miss_counters(self):
        """测试命中和未命中计数"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")

        # 命中
        cache.get("key1")
        assert cache._hits == 1
        assert cache._misses == 0

        # 未命中
        cache.get("key_nonexistent")
        assert cache._hits == 1
        assert cache._misses == 1

    def test_remove(self):
        """测试移除操作"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        removed = cache.remove("key1")
        assert removed is True
        assert cache.get("key1") is None

        # 移除不存在的键
        removed = cache.remove("key_nonexistent")
        assert removed is False

    def test_cleanup_expired(self):
        """测试清理过期条目"""
        cache = LRUCache(max_size=10)

        cache.put("temp1", "value", ttl_seconds=0.5)
        cache.put("temp2", "value", ttl_seconds=1.0)
        cache.put("permanent", "value", ttl_seconds=None)

        time.sleep(0.6)

        expired_count = cache.cleanup_expired()
        assert expired_count == 1
        assert cache.get("temp1") is None
        assert cache.get("temp2") == "value"
        assert cache.get("permanent") == "value"

    def test_clear(self):
        """测试清空缓存"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        assert cache.size() == 2

        cache.clear()
        assert cache.size() == 0
        assert cache._hits == 0
        assert cache._misses == 0

    def test_get_statistics(self):
        """测试获取统计信息"""
        cache = LRUCache(max_size=3)

        cache.put("key1", "value1")
        cache.get("key1")  # 命中
        cache.get("key2")  # 未命中

        stats = cache.get_statistics()

        assert stats["size"] == 1
        assert stats["max_size"] == 3
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["utilization"] == 1.0 / 3.0


class TestMemoryManagedDict:
    """测试内存管理字典"""

    def test_basic_operations(self):
        """测试基本操作"""
        mdict = MemoryManagedDict(max_size=5, default_ttl=10.0)

        mdict.put("key1", "value1")
        mdict.put("key2", "value2")

        assert mdict.size() == 2
        assert mdict.get("key1") == "value1"
        assert mdict.get("key2") == "value2"

    def test_ttl_cleanup(self):
        """测试TTL自动清理"""
        mdict = MemoryManagedDict(max_size=10, default_ttl=0.5)

        mdict.put("temp", "value")

        time.sleep(0.6)

        # 访问时应该触发清理
        value = mdict.get("temp")
        assert value is None

    def test_custom_ttl(self):
        """测试自定义TTL"""
        mdict = MemoryManagedDict(max_size=10, default_ttl=10.0)

        mdict.put("key1", "value", ttl_seconds=0.5)

        time.sleep(0.6)

        assert mdict.get("key1") is None


class TestSemanticObjectManager:
    """测试语义物体管理器"""

    def test_add_object(self):
        """测试添加物体"""
        manager = SemanticObjectManager(max_objects=10, object_ttl=300.0)

        obj = MockSemanticObject("door", (5.0, 3.0))
        obj_id = manager.add_or_update(obj)

        assert obj_id.startswith("semantic_")
        assert manager.size() == 1
        assert manager.get(obj_id) == obj

    def test_update_object(self):
        """测试更新物体"""
        manager = SemanticObjectManager(max_objects=10, object_ttl=300.0)

        obj1 = MockSemanticObject("door", (5.0, 3.0))
        obj_id = manager.add_or_update(obj1)

        obj2 = MockSemanticObject("door", (5.1, 3.1))  # 相似物体
        obj_id2 = manager.add_or_update(obj2)

        # 应该匹配到已有物体
        assert obj_id == obj_id2
        assert manager.size() == 1

    def test_object_matching(self):
        """测试物体匹配"""
        manager = SemanticObjectManager(
            max_objects=10,
            object_ttl=300.0,
            position_threshold=2.0
        )

        obj1 = MockSemanticObject("door", (5.0, 3.0))
        obj_id1 = manager.add_or_update(obj1)

        obj2 = MockSemanticObject("door", (5.1, 3.1))  # 距离 < 2.0
        obj_id2 = manager.add_or_update(obj2)

        # 应该匹配
        assert obj_id1 == obj_id2

        obj3 = MockSemanticObject("door", (10.0, 10.0))  # 距离 > 2.0
        obj_id3 = manager.add_or_update(obj3)

        # 不应该匹配
        assert obj_id3 != obj_id1

    def test_remove_object(self):
        """测试移除物体"""
        manager = SemanticObjectManager(max_objects=10)

        obj = MockSemanticObject("door", (5.0, 3.0))
        obj_id = manager.add_or_update(obj)

        removed = manager.remove(obj_id)
        assert removed is True
        assert manager.size() == 0

    def test_get_all(self):
        """测试获取所有物体"""
        manager = SemanticObjectManager(max_objects=10)

        obj1 = MockSemanticObject("door", (5.0, 3.0))
        obj2 = MockSemanticObject("person", (7.0, 2.0))

        manager.add_or_update(obj1)
        manager.add_or_update(obj2)

        all_objects = manager.get_all()
        assert len(all_objects) == 2

    def test_cleanup_expired(self):
        """测试清理过期物体"""
        manager = SemanticObjectManager(max_objects=10, object_ttl=0.5)

        obj = MockSemanticObject("door", (5.0, 3.0))
        manager.add_or_update(obj)

        time.sleep(0.6)

        expired_count = manager.cleanup_expired()
        assert expired_count == 1
        assert manager.size() == 0

    def test_max_objects_limit(self):
        """测试最大物体数量限制"""
        manager = SemanticObjectManager(max_objects=3, object_ttl=300.0)

        # 添加4个物体
        for i in range(4):
            obj = MockSemanticObject(f"object{i}", (float(i), float(i)))
            manager.add_or_update(obj)

        # 由于LRU，应该保留最后3个
        assert manager.size() == 3

    def test_get_statistics(self):
        """测试获取统计信息"""
        manager = SemanticObjectManager(
            max_objects=100,
            object_ttl=300.0,
            position_threshold=2.0
        )

        obj = MockSemanticObject("door", (5.0, 3.0))
        manager.add_or_update(obj)

        stats = manager.get_statistics()
        assert stats["max_objects"] == 100
        assert stats["object_ttl"] == 300.0
        assert stats["size"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
