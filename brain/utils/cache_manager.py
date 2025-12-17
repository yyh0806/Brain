"""
Comprehensive Caching Manager for Brain System

Provides intelligent caching with:
- LRU and TTL cache implementations
- Memory-based and disk-based caching
- Cache invalidation strategies
- Performance monitoring
- Automatic cleanup
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar
from loguru import logger

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry:
    """缓存条目"""
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self, ttl_seconds: Optional[float] = None) -> bool:
        """检查是否过期"""
        if ttl_seconds is None:
            return False
        return time.time() - self.created_at > ttl_seconds

    def touch(self):
        """更新访问时间和计数"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheInterface(ABC, Generic[K, V]):
    """缓存接口"""

    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        pass

    @abstractmethod
    def set(self, key: K, value: V, ttl_seconds: Optional[float] = None) -> None:
        """设置缓存值"""
        pass

    @abstractmethod
    def delete(self, key: K) -> bool:
        """删除缓存值"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass

    @abstractmethod
    def size(self) -> int:
        """缓存大小"""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """缓存统计信息"""
        pass


class LRUCache(CacheInterface[K, V]):
    """LRU (Least Recently Used) 缓存实现"""

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[K, CacheEntry] = OrderedDict()
        self._lock = RLock()

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._sets = 0

    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._misses += 1
                return None

            # 移到末尾（最近使用）
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl_seconds: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            # 检查是否需要更新现有条目
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.created_at = time.time()
                entry.touch()
                self._cache.move_to_end(key)
                self._sets += 1
                return

            # 计算对象大小（近似）
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0

            # 检查是否需要清理空间
            while len(self._cache) >= self.max_size:
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"LRU缓存清理: {oldest_key}")

            # 添加新条目
            entry = CacheEntry(value=value, size_bytes=size_bytes)
            if ttl_seconds is not None:
                entry.created_at = time.time()  # 重置创建时间以应用TTL

            self._cache[key] = entry
            self._sets += 1

    def delete(self, key: K) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            logger.info("LRU缓存已清空")

    def size(self) -> int:
        """缓存大小"""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """缓存统计信息"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        total_memory = sum(entry.size_bytes for entry in self._cache.values())

        return {
            "type": "LRU",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "sets": self._sets,
            "ttl_seconds": self.ttl_seconds,
            "total_memory_bytes": total_memory,
            "avg_entry_size": total_memory / len(self._cache) if self._cache else 0
        }


class TTLCache(CacheInterface[K, V]):
    """TTL (Time To Live) 缓存实现"""

    def __init__(self, default_ttl_seconds: float = 300.0, max_size: int = 10000):
        self.default_ttl_seconds = default_ttl_seconds
        self.max_size = max_size
        self._cache: Dict[K, CacheEntry] = {}
        self._lock = RLock()

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._expires = 0
        self._sets = 0

    def get(self, key: K) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired(self.default_ttl_seconds):
                del self._cache[key]
                self._expires += 1
                self._misses += 1
                return None

            entry.touch()
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, ttl_seconds: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            ttl = ttl_seconds or self.default_ttl_seconds

            # 检查是否需要清理空间
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cleanup_expired()
                if len(self._cache) >= self.max_size:
                    # 删除最旧的条目
                    oldest_key = min(self._cache.keys(),
                                   key=lambda k: self._cache[k].created_at)
                    del self._cache[oldest_key]

            # 计算对象大小（近似）
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0

            # 添加或更新条目
            entry = CacheEntry(value=value, size_bytes=size_bytes)
            self._cache[key] = entry
            self._sets += 1

    def delete(self, key: K) -> bool:
        """删除缓存值"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            logger.info("TTL缓存已清空")

    def size(self) -> int:
        """缓存大小"""
        return len(self._cache)

    def _cleanup_expired(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self._cache.items():
            if current_time - entry.created_at > self.default_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]
            self._expires += 1

        if expired_keys:
            logger.debug(f"TTL缓存清理过期条目: {len(expired_keys)}个")

    def stats(self) -> Dict[str, Any]:
        """缓存统计信息"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        total_memory = sum(entry.size_bytes for entry in self._cache.values())

        return {
            "type": "TTL",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "expires": self._expires,
            "sets": self._sets,
            "ttl_seconds": self.default_ttl_seconds,
            "total_memory_bytes": total_memory,
            "avg_entry_size": total_memory / len(self._cache) if self._cache else 0
        }


class DiskCache(CacheInterface[str, V]):
    """磁盘缓存实现"""

    def __init__(self, cache_dir: Path, max_size_mb: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = Lock()

        # 元数据文件
        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("磁盘缓存元数据损坏，重新创建")
        return {}

    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f)
        except Exception as e:
            logger.error(f"保存磁盘缓存元数据失败: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[V]:
        """获取缓存值"""
        with self._lock:
            cache_path = self._get_cache_path(key)

            if not cache_path.exists():
                self._misses += 1
                return None

            # 检查元数据
            if key not in self._metadata:
                cache_path.unlink(missing_ok=True)
                self._misses += 1
                return None

            meta = self._metadata[key]
            ttl_seconds = meta.get("ttl_seconds")

            # 检查是否过期
            if ttl_seconds and time.time() - meta["created_at"] > ttl_seconds:
                cache_path.unlink(missing_ok=True)
                del self._metadata[key]
                self._save_metadata()
                self._misses += 1
                return None

            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)

                # 更新访问信息
                meta["last_accessed"] = time.time()
                meta["access_count"] = meta.get("access_count", 0) + 1
                self._save_metadata()

                self._hits += 1
                return value
            except Exception as e:
                logger.error(f"读取磁盘缓存失败 {key}: {e}")
                cache_path.unlink(missing_ok=True)
                if key in self._metadata:
                    del self._metadata[key]
                    self._save_metadata()
                self._misses += 1
                return None

    def set(self, key: str, value: V, ttl_seconds: Optional[float] = None) -> None:
        """设置缓存值"""
        with self._lock:
            cache_path = self._get_cache_path(key)
            current_time = time.time()

            try:
                # 写入缓存文件
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)

                # 获取文件大小
                file_size = cache_path.stat().st_size

                # 更新元数据
                self._metadata[key] = {
                    "created_at": current_time,
                    "last_accessed": current_time,
                    "access_count": 0,
                    "ttl_seconds": ttl_seconds,
                    "size_bytes": file_size
                }

                # 检查是否需要清理空间
                self._cleanup_if_needed()

                self._save_metadata()
            except Exception as e:
                logger.error(f"写入磁盘缓存失败 {key}: {e}")
                cache_path.unlink(missing_ok=True)

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            cache_path = self._get_cache_path(key)

            if cache_path.exists():
                cache_path.unlink()

            if key in self._metadata:
                del self._metadata[key]
                self._save_metadata()
                return True

            return False

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            # 删除所有缓存文件
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink(missing_ok=True)

            # 清空元数据
            self._metadata.clear()
            self._save_metadata()
            logger.info("磁盘缓存已清空")

    def size(self) -> int:
        """缓存大小"""
        return len(self._metadata)

    def _cleanup_if_needed(self):
        """如果需要则清理空间"""
        total_size = sum(meta.get("size_bytes", 0) for meta in self._metadata.values())

        if total_size > self.max_size_bytes:
            # 按最后访问时间排序，删除最旧的条目
            sorted_keys = sorted(
                self._metadata.keys(),
                key=lambda k: self._metadata[k]["last_accessed"]
            )

            for key in sorted_keys:
                self.delete(key)
                self._evictions += 1

                # 重新计算大小
                total_size = sum(meta.get("size_bytes", 0) for meta in self._metadata.values())
                if total_size <= self.max_size_bytes * 0.8:  # 清理到80%
                    break

    def stats(self) -> Dict[str, Any]:
        """缓存统计信息"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        total_memory = sum(meta.get("size_bytes", 0) for meta in self._metadata.values())

        return {
            "type": "Disk",
            "size": len(self._metadata),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "total_memory_bytes": total_memory,
            "max_memory_bytes": self.max_size_bytes,
            "memory_usage_percent": (total_memory / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
        }


class CacheManager:
    """缓存管理器 - 提供多级缓存支持"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("/tmp/brain_cache")
        self.caches: Dict[str, CacheInterface] = {}
        self._lock = Lock()

        # 默认缓存配置
        self._setup_default_caches()

    def _setup_default_caches(self):
        """设置默认缓存"""
        # L1缓存：内存LRU缓存（热数据）
        self.caches["l1_memory"] = LRUCache(max_size=100, ttl_seconds=60)

        # L2缓存：内存TTL缓存（温数据）
        self.caches["l2_memory"] = TTLCache(default_ttl_seconds=300, max_size=1000)

        # L3缓存：磁盘缓存（冷数据）
        self.caches["l3_disk"] = DiskCache(cache_dir=self.cache_dir / "l3", max_size_mb=50)

        # 特殊用途缓存
        self.caches["reasoning"] = LRUCache(max_size=500, ttl_seconds=1800)  # 30分钟
        self.caches["sensor_data"] = TTLCache(default_ttl_seconds=10, max_size=1000)  # 10秒
        self.caches["llm_responses"] = LRUCache(max_size=200, ttl_seconds=3600)  # 1小时

    def get_cache(self, name: str) -> CacheInterface:
        """获取指定名称的缓存"""
        with self._lock:
            if name not in self.caches:
                raise ValueError(f"缓存 '{name}' 不存在")
            return self.caches[name]

    def create_cache(
        self,
        name: str,
        cache_type: str = "lru",
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
        disk_size_mb: int = 100
    ) -> CacheInterface:
        """创建新缓存"""
        with self._lock:
            if name in self.caches:
                logger.warning(f"缓存 '{name}' 已存在，将被替换")

            if cache_type.lower() == "lru":
                cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
            elif cache_type.lower() == "ttl":
                cache = TTLCache(default_ttl_seconds=ttl_seconds or 300, max_size=max_size)
            elif cache_type.lower() == "disk":
                cache = DiskCache(cache_dir=self.cache_dir / name, max_size_mb=disk_size_mb)
            else:
                raise ValueError(f"不支持的缓存类型: {cache_type}")

            self.caches[name] = cache
            logger.info(f"创建缓存 '{name}': {cache_type}, max_size={max_size}")

            return cache

    def get_multi_level(self, key: str, *cache_names: str) -> Optional[Any]:
        """多级缓存获取"""
        for cache_name in cache_names:
            if cache_name not in self.caches:
                continue

            cache = self.caches[cache_name]
            value = cache.get(key)

            if value is not None:
                # 将值提升到更高级别的缓存
                self._promote_to_higher_caches(key, value, cache_name, cache_names)
                return value

        return None

    def set_multi_level(self, key: str, value: Any, *cache_names: str, ttl_seconds: Optional[float] = None):
        """多级缓存设置"""
        for cache_name in cache_names:
            if cache_name in self.caches:
                self.caches[cache_name].set(key, value, ttl_seconds)

    def _promote_to_higher_caches(self, key: str, value: Any, found_in: str, cache_names: Tuple[str, ...]):
        """将值提升到更高级别的缓存"""
        current_index = cache_names.index(found_in)

        # 提升到所有更高级别的缓存
        for i in range(current_index):
            higher_cache_name = cache_names[i]
            if higher_cache_name in self.caches:
                self.caches[higher_cache_name].set(key, value)

    def invalidate_pattern(self, pattern: str, cache_name: Optional[str] = None):
        """按模式失效缓存"""
        caches_to_check = [self.caches[cache_name]] if cache_name else list(self.caches.values())

        for cache in caches_to_check:
            if hasattr(cache, '_cache'):
                # 对于内存缓存
                keys_to_delete = []
                for key in cache._cache.keys():
                    if isinstance(key, str) and pattern in key:
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    cache.delete(key)

                if keys_to_delete:
                    logger.debug(f"缓存失效模式 '{pattern}': {len(keys_to_delete)}个键")

    def cleanup_expired(self):
        """清理所有过期条目"""
        for name, cache in self.caches.items():
            if isinstance(cache, TTLCache):
                cache._cleanup_expired()
                logger.debug(f"清理过期缓存 '{name}'")

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有缓存的统计信息"""
        stats = {}

        for name, cache in self.caches.items():
            try:
                stats[name] = cache.stats()
            except Exception as e:
                logger.error(f"获取缓存统计失败 '{name}': {e}")
                stats[name] = {"error": str(e)}

        return stats

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_all_stats()

        print("\n" + "="*80)
        print("Brain 缓存系统统计信息")
        print("="*80)

        for name, stat in stats.items():
            if "error" in stat:
                print(f"\n{name}: ERROR - {stat['error']}")
                continue

            hit_rate = stat.get("hit_rate", 0) * 100
            size = stat.get("size", 0)
            memory_mb = stat.get("total_memory_bytes", 0) / (1024*1024)

            print(f"\n{name} ({stat.get('type', 'Unknown')}):")
            print(f"  大小: {size} 条目")
            print(f"  命中率: {hit_rate:.1f}%")
            print(f"  内存使用: {memory_mb:.1f} MB")
            print(f"  命中/未命中: {stat.get('hits', 0)}/{stat.get('misses', 0)}")

            if "evictions" in stat:
                print(f"  驱逐次数: {stat['evictions']}")
            if "expires" in stat:
                print(f"  过期次数: {stat['expires']}")

        print("\n" + "="*80)

    def clear_all(self):
        """清空所有缓存"""
        for name, cache in self.caches.items():
            cache.clear()
        logger.info("所有缓存已清空")


# 全局缓存管理器实例
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def cached(cache_name: str = "l1_memory", ttl_seconds: Optional[float] = None, key_func=None):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cache_name)

            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result

            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)

            return result

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._cached = True
        wrapper._cache_name = cache_name

        return wrapper

    return decorator


async def cached_async(cache_name: str = "l1_memory", ttl_seconds: Optional[float] = None, key_func=None):
    """异步缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cache_name)

            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                return result

            # 执行异步函数并缓存结果
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl_seconds)

            return result

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._cached = True
        wrapper._cache_name = cache_name

        return wrapper

    return decorator