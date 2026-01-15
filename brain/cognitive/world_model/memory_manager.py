# -*- coding: utf-8 -*-
"""
内存管理器 - Memory Manager

实现智能内存管理策略：
- LRU缓存：自动清理最少使用的对象
- TTL过期：自动清理过期对象
- 内存监控：跟踪内存使用情况
- 清理策略：多种策略组合使用
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None


class LRUCache:
    """
    LRU缓存实现

    自动清理最少使用的对象，防止内存无限增长
    """

    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在返回None
        """
        with self._lock:
            if key in self._cache:
                # 移到末尾（表示最近使用）
                entry = self._cache.pop(key)
                entry.access_count += 1
                entry.timestamp = datetime.now()
                self._cache[key] = entry
                self._hits += 1
                return entry.value
            else:
                self._misses += 1
                return None

    def put(self, key: str, value: Any, size_bytes: int = 0, ttl_seconds: Optional[float] = None):
        """
        添加缓存值

        Args:
            key: 缓存键
            value: 缓存值
            size_bytes: 值的大小（字节）
            ttl_seconds: TTL（秒）
        """
        with self._lock:
            # 如果已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds
            )

            self._cache[key] = entry

            # 检查是否超过最大大小
            if len(self._cache) > self.max_size:
                # 删除最少使用的（最旧的）
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"LRU清理: 删除缓存条目 {oldest_key}")

    def remove(self, key: str) -> bool:
        """
        删除缓存条目

        Args:
            key: 缓存键

        Returns:
            是否删除成功
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def cleanup_expired(self) -> int:
        """
        清理过期的缓存条目

        Returns:
            清理的条目数量
        """
        with self._lock:
            now = datetime.now()
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.ttl_seconds is not None:
                    age = (now - entry.timestamp).total_seconds()
                    if age > entry.ttl_seconds:
                        expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                logger.debug(f"清理过期缓存: {len(expired_keys)}个条目")

            return len(expired_keys)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def size(self) -> int:
        """获取当前缓存大小"""
        return len(self._cache)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "utilization": len(self._cache) / self.max_size
            }


class MemoryManagedDict:
    """
    内存管理字典

    结合LRU和TTL的智能内存管理字典
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        cleanup_threshold: float = 0.8
    ):
        """
        Args:
            max_size: 最大条目数
            default_ttl: 默认TTL（秒）
            cleanup_threshold: 触发清理的使用率阈值
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_threshold = cleanup_threshold

        self._lru_cache = LRUCache(max_size=max_size)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """获取值"""
        # 先清理过期条目
        self._cleanup_if_needed()

        value = self._lru_cache.get(key)
        return value

    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """添加值"""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl

        self._lru_cache.put(key, value, ttl_seconds=ttl_seconds)
        self._cleanup_if_needed()

    def remove(self, key: str) -> bool:
        """删除值"""
        return self._lru_cache.remove(key)

    def _cleanup_if_needed(self):
        """根据使用率触发清理"""
        stats = self._lru_cache.get_statistics()
        if stats['utilization'] > self.cleanup_threshold:
            # 清理过期条目
            expired_count = self._lru_cache.cleanup_expired()
            if expired_count == 0 and stats['utilization'] > 0.9:
                # 如果没有过期条目但使用率仍然很高，强制清理
                self._force_cleanup()

    def _force_cleanup(self):
        """强制清理（删除最旧的20%）"""
        with self._lru_cache._lock:
            num_to_remove = int(len(self._lru_cache._cache) * 0.2)
            keys_to_remove = list(self._lru_cache._cache.keys())[:num_to_remove]

            for key in keys_to_remove:
                del self._lru_cache._cache[key]

            logger.info(f"强制清理: 删除{num_to_remove}个缓存条目")

    def clear(self):
        """清空"""
        self._lru_cache.clear()

    def size(self) -> int:
        """获取当前大小"""
        return self._lru_cache.size()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._lru_cache.get_statistics()


class SemanticObjectManager:
    """
    语义物体管理器

    专门用于管理语义对象的内存和生命周期
    """

    def __init__(
        self,
        max_objects: int = 500,
        object_ttl: float = 300.0,  # 5分钟
        position_threshold: float = 2.0  # 匹配距离阈值
    ):
        """
        Args:
            max_objects: 最大物体数量
            object_ttl: 物体TTL（秒）
            position_threshold: 位置匹配阈值（米）
        """
        self.max_objects = max_objects
        self.object_ttl = object_ttl
        self.position_threshold = position_threshold

        # 使用内存管理字典
        self.objects = MemoryManagedDict(
            max_size=max_objects,
            default_ttl=object_ttl,
            cleanup_threshold=0.8
        )

        self._object_counter = 0

    def add_or_update(self, obj: Any) -> str:
        """
        添加或更新语义物体

        Args:
            obj: 语义对象（需要有label和world_position属性）

        Returns:
            物体ID
        """
        # 尝试匹配已有物体
        matched_id = self._match_object(obj)

        if matched_id:
            # 更新已有物体
            self.objects.put(matched_id, obj, ttl_seconds=self.object_ttl)
            logger.debug(f"更新语义物体: {matched_id}")
            return matched_id
        else:
            # 新物体
            obj_id = f"semantic_{self._object_counter}"
            self._object_counter += 1

            self.objects.put(obj_id, obj, ttl_seconds=self.object_ttl)
            logger.debug(f"新增语义物体: {obj_id}")
            return obj_id

    def get(self, object_id: str) -> Optional[Any]:
        """获取语义物体"""
        return self.objects.get(object_id)

    def remove(self, object_id: str) -> bool:
        """删除语义物体"""
        return self.objects.remove(object_id)

    def get_all(self) -> Dict[str, Any]:
        """获取所有语义物体"""
        # 注意：这会返回所有对象，但不会更新LRU
        with self.objects._lru_cache._lock:
            return {
                key: entry.value
                for key, entry in self.objects._lru_cache._cache.items()
            }

    def size(self) -> int:
        """获取当前语义物体数量"""
        return self.objects.size()

    def _match_object(self, obj: Any) -> Optional[str]:
        """
        匹配物体（基于标签和位置）

        Args:
            obj: 待匹配的物体

        Returns:
            匹配到的物体ID，如果没有匹配返回None
        """
        if not hasattr(obj, 'label'):
            return None

        all_objects = self.get_all()

        for obj_id, existing_obj in all_objects.items():
            # 检查标签相似度
            if hasattr(existing_obj, 'label'):
                if obj.label.lower() in existing_obj.label.lower() or \
                   existing_obj.label.lower() in obj.label.lower():

                    # 检查位置距离
                    if hasattr(obj, 'world_position') and hasattr(existing_obj, 'world_position'):
                        try:
                            obj_pos = obj.world_position
                            existing_pos = existing_obj.world_position

                            if len(obj_pos) >= 2 and len(existing_pos) >= 2:
                                dist = ((obj_pos[0] - existing_pos[0])**2 +
                                       (obj_pos[1] - existing_pos[1])**2)**0.5

                                if dist < self.position_threshold:
                                    return obj_id
                        except Exception:
                            pass

        return None

    def cleanup_expired(self) -> int:
        """清理过期的语义物体"""
        return self.objects._lru_cache.cleanup_expired()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.objects.get_statistics()
        stats['max_objects'] = self.max_objects
        stats['object_ttl'] = self.object_ttl
        return stats


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("内存管理器测试")
    print("=" * 60)

    # 测试1: LRU缓存
    print("\n[测试1] LRU缓存...")
    cache = LRUCache(max_size=3)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    print(f"缓存大小: {cache.size()}")

    cache.put("key4", "value4")  # 应该触发LRU清理
    print(f"添加key4后缓存大小: {cache.size()}")
    print(f"key1是否存在: {cache.get('key1') is None}")  # 应该被清理

    stats = cache.get_statistics()
    print(f"缓存统计: {stats}")

    # 测试2: TTL过期
    print("\n[测试2] TTL过期...")
    cache_with_ttl = LRUCache(max_size=10)

    cache_with_ttl.put("temp1", "value", ttl_seconds=1.0)
    print("添加TTL=1秒的缓存")

    import time
    time.sleep(1.1)

    expired_count = cache_with_ttl.cleanup_expired()
    print(f"清理过期缓存: {expired_count}个条目")
    print(f"temp1是否存在: {cache_with_ttl.get('temp1') is None}")

    # 测试3: 语义物体管理
    print("\n[测试3] 语义物体管理...")
    obj_manager = SemanticObjectManager(max_objects=5, object_ttl=2.0)

    class MockSemanticObject:
        def __init__(self, label, position):
            self.label = label
            self.world_position = position

    obj1 = MockSemanticObject("门", (5.0, 3.0))
    obj2 = MockSemanticObject("门", (5.1, 3.1))  # 相似物体
    obj3 = MockSemanticObject("人", (7.0, 2.0))

    id1 = obj_manager.add_or_update(obj1)
    print(f"添加obj1: {id1}")

    id2 = obj_manager.add_or_update(obj2)
    print(f"添加相似obj2: {id2} (应该匹配obj1: {id1 == id2})")

    id3 = obj_manager.add_or_update(obj3)
    print(f"添加obj3: {id3}")

    print(f"当前物体数量: {obj_manager.size()}")

    # 测试4: 内存管理统计
    print("\n[测试4] 统计信息...")
    stats = obj_manager.get_statistics()
    print(f"语义物体统计: {stats}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n内存管理特性:")
    print("  ✓ LRU缓存: 自动清理最少使用的对象")
    print("  ✓ TTL过期: 自动清理过期对象")
    print("  ✓ 内存监控: 跟踪内存使用情况")
    print("  ✓ 智能清理: 多种策略组合使用")
