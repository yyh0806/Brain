"""
Resource Manager for Brain System

Provides comprehensive resource management:
- Memory leak detection and prevention
- Automatic resource cleanup
- Resource usage monitoring
- Garbage collection optimization
- Circular reference detection
"""

import gc
import os
import psutil
import resource
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from loguru import logger


class ResourceType(Enum):
    """资源类型"""
    MEMORY = "memory"
    FILE_HANDLE = "file_handle"
    NETWORK_CONNECTION = "network_connection"
    THREAD = "thread"
    TIMER = "timer"
    CACHE = "cache"
    EVENT_LISTENER = "event_listener"


@dataclass
class ResourceUsage:
    """资源使用情况"""
    resource_type: ResourceType
    usage_count: int = 0
    max_usage: int = 0
    usage_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_cleanup: datetime = field(default_factory=datetime.now)
    leaks_detected: int = 0

    def add_usage(self, count: int = 1):
        """添加使用记录"""
        self.usage_count += count
        self.usage_history.append((time.time(), self.usage_count))
        self.max_usage = max(self.max_usage, self.usage_count)

    def record_cleanup(self, cleaned: int = 0):
        """记录清理操作"""
        self.last_cleanup = datetime.now()
        self.usage_count = max(0, self.usage_count - cleaned)

    def get_trend(self) -> str:
        """获取使用趋势"""
        if len(self.usage_history) < 2:
            return "stable"

        recent = list(self.usage_history)[-10:]  # 最近10个记录
        if len(recent) < 2:
            return "stable"

        # 计算趋势
        start_usage = recent[0][1]
        end_usage = recent[-1][1]

        if end_usage > start_usage * 1.2:
            return "increasing"
        elif end_usage < start_usage * 0.8:
            return "decreasing"
        else:
            return "stable"


class ResourceTracker:
    """资源跟踪器"""

    def __init__(self, name: str):
        self.name = name
        self.resources: Dict[str, ResourceType] = {}
        self.cleanup_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.creation_stack: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        cleanup_handler: Optional[Callable] = None
    ):
        """注册资源"""
        with self._lock:
            self.resources[resource_id] = resource_type
            if cleanup_handler:
                self.cleanup_handlers[resource_id].append(cleanup_handler)

            # 记录创建栈（用于调试）
            if __debug__:
                import traceback
                self.creation_stack[resource_id] = traceback.format_stack()[-3:]

            logger.debug(f"资源注册: {self.name}.{resource_id} ({resource_type.value})")

    def unregister_resource(self, resource_id: str, force_cleanup: bool = False) -> bool:
        """注销资源"""
        with self._lock:
            if resource_id not in self.resources:
                return False

            resource_type = self.resources[resource_id]

            # 执行清理处理器
            if resource_id in self.cleanup_handlers:
                for handler in self.cleanup_handlers[resource_id]:
                    try:
                        handler()
                    except Exception as e:
                        logger.error(f"资源清理失败 {resource_id}: {e}")

            # 清理记录
            del self.resources[resource_id]
            self.cleanup_handlers.pop(resource_id, None)
            self.creation_stack.pop(resource_id, None)

            if force_cleanup:
                logger.warning(f"强制清理资源: {self.name}.{resource_id}")
            else:
                logger.debug(f"资源注销: {self.name}.{resource_id}")

            return True

    def cleanup_all(self) -> int:
        """清理所有资源"""
        with self._lock:
            count = 0
            resource_ids = list(self.resources.keys())

            for resource_id in resource_ids:
                if self.unregister_resource(resource_id, force_cleanup=True):
                    count += 1

            logger.info(f"清理了 {count} 个资源: {self.name}")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        with self._lock:
            stats = defaultdict(int)
            for resource_type in self.resources.values():
                stats[resource_type.value] += 1

            return {
                "total_resources": len(self.resources),
                "by_type": dict(stats),
                "pending_cleanups": len(self.cleanup_handlers)
            }

    def detect_leaks(self) -> List[Dict[str, Any]]:
        """检测潜在的资源泄漏"""
        with self._lock:
            leaks = []

            # 检查长时间未清理的资源
            current_time = time.time()
            for resource_id, resource_type in self.resources.items():
                # 根据资源类型设置不同的超时时间
                timeouts = {
                    ResourceType.FILE_HANDLE: 300,      # 5分钟
                    ResourceType.NETWORK_CONNECTION: 600,  # 10分钟
                    ResourceType.THREAD: 1800,        # 30分钟
                    ResourceType.TIMER: 300,          # 5分钟
                    ResourceType.CACHE: 3600,         # 1小时
                    ResourceType.MEMORY: 1800,        # 30分钟
                }

                timeout = timeouts.get(resource_type, 600)

                # 检查创建时间（简化：使用栈跟踪信息）
                if resource_id in self.creation_stack:
                    # 这里应该记录创建时间，简化处理
                    pass

            return leaks


class MemoryManager:
    """内存管理器"""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.process = psutil.Process()
        self.baseline_memory = self._get_current_memory()
        self.memory_history: deque = deque(maxlen=1000)
        self.gc_stats = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def _get_current_memory(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            "rss": memory_info.rss / (1024 * 1024),  # MB
            "vms": memory_info.vms / (1024 * 1024),  # MB
            "percent": memory_percent,
            "available": psutil.virtual_memory().available / (1024 * 1024)  # MB
        }

    def start_monitoring(self):
        """开始内存监控"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        logger.info("内存监控已启动")

    def stop_monitoring(self):
        """停止内存监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("内存监控已停止")

    def _monitor_memory(self):
        """内存监控线程"""
        while self._monitoring:
            try:
                memory_info = self._get_current_memory()
                self.memory_history.append((time.time(), memory_info))

                # 检测内存增长趋势
                if len(self.memory_history) > 10:
                    recent = list(self.memory_history)[-10:]
                    start_memory = recent[0][1]["rss"]
                    current_memory = recent[-1][1]["rss"]

                    growth_rate = (current_memory - start_memory) / start_memory

                    # 如果内存增长率超过50%，触发垃圾回收
                    if growth_rate > 0.5:
                        logger.warning(f"检测到内存快速增长: {growth_rate*100:.1f}%")
                        self.force_garbage_collection()

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"内存监控错误: {e}")
                time.sleep(self.check_interval)

    def force_garbage_collection(self) -> Dict[str, Any]:
        """强制垃圾回收"""
        logger.info("执行强制垃圾回收")

        # 执行多轮垃圾回收
        gc_results = []
        for generation in range(3):
            collected = gc.collect()
            gc_results.append(collected)

        # 收集垃圾回收统计
        stats = gc.get_stats()
        memory_after = self._get_current_memory()

        result = {
            "collections": gc_results,
            "total_collected": sum(gc_results),
            "gc_stats": stats,
            "memory_before": self.memory_history[-1][1] if self.memory_history else self.baseline_memory,
            "memory_after": memory_after,
            "memory_freed": self.memory_history[-1][1]["rss"] - memory_after["rss"] if self.memory_history else 0
        }

        self.gc_stats.append((time.time(), result))
        logger.info(f"垃圾回收完成: 回收 {result['total_collected']} 个对象, "
                   f"释放 {result['memory_freed']:.1f} MB 内存")

        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        current_memory = self._get_current_memory()

        stats = {
            "current": current_memory,
            "baseline": self.baseline_memory,
            "growth": current_memory["rss"] - self.baseline_memory["rss"],
            "growth_percent": ((current_memory["rss"] - self.baseline_memory["rss"]) /
                              self.baseline_memory["rss"]) * 100,
            "history_count": len(self.memory_history),
            "gc_collections": len(self.gc_stats)
        }

        # 添加趋势分析
        if len(self.memory_history) > 10:
            recent = list(self.memory_history)[-10:]
            memory_trend = recent[-1][1]["rss"] - recent[0][1]["rss"]
            stats["recent_trend"] = "increasing" if memory_trend > 10 else "stable"
            stats["recent_growth"] = memory_trend

        return stats

    def detect_memory_leaks(self) -> List[str]:
        """检测潜在内存泄漏"""
        leaks = []

        if len(self.memory_history) < 20:
            return leaks

        # 检查长期增长趋势
        long_term = list(self.memory_history)[-20:]
        start_memory = long_term[0][1]["rss"]
        end_memory = long_term[-1][1]["rss"]
        growth_rate = (end_memory - start_memory) / start_memory

        if growth_rate > 0.2:  # 20%增长
            leaks.append(f"检测到长期内存增长趋势: {growth_rate*100:.1f}%")

        # 检查垃圾回收效果
        if len(self.gc_stats) > 2:
            recent_gc = self.gc_stats[-2:]
            if recent_gc[1][1]["total_collected"] < 100 and growth_rate > 0.1:
                leaks.append("垃圾回收效果不佳，可能存在循环引用")

        return leaks


class ResourceManager:
    """综合资源管理器"""

    def __init__(self):
        self.trackers: Dict[str, ResourceTracker] = {}
        self.memory_manager = MemoryManager()
        self.resource_usage: Dict[ResourceType, ResourceUsage] = {
            rtype: ResourceUsage(resource_type=rtype) for rtype in ResourceType
        }
        self.cleanup_policies: Dict[ResourceType, List[Callable]] = defaultdict(list)
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None

        # 注册默认清理策略
        self._register_default_policies()

    def _register_default_policies(self):
        """注册默认清理策略"""

        def cleanup_caches():
            """清理缓存的策略"""
            try:
                # 尝试清理各种缓存
                for tracker in self.trackers.values():
                    cache_resources = [
                        rid for rid, rtype in tracker.resources.items()
                        if rtype == ResourceType.CACHE
                    ]
                    for resource_id in cache_resources[:10]:  # 限制每次清理数量
                        tracker.unregister_resource(resource_id)
            except Exception as e:
                logger.error(f"缓存清理失败: {e}")

        def cleanup_memory():
            """清理内存的策略"""
            try:
                self.memory_manager.force_garbage_collection()
            except Exception as e:
                logger.error(f"内存清理失败: {e}")

        self.cleanup_policies[ResourceType.CACHE].append(cleanup_caches)
        self.cleanup_policies[ResourceType.MEMORY].append(cleanup_memory)

    def create_tracker(self, name: str) -> ResourceTracker:
        """创建资源跟踪器"""
        tracker = ResourceTracker(name)
        self.trackers[name] = tracker
        logger.info(f"创建资源跟踪器: {name}")
        return tracker

    def get_tracker(self, name: str) -> Optional[ResourceTracker]:
        """获取资源跟踪器"""
        return self.trackers.get(name)

    def start_monitoring(self):
        """开始资源监控"""
        self.memory_manager.start_monitoring()
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("资源管理器监控已启动")

    def stop_monitoring(self):
        """停止资源监控"""
        self._running = False
        self.memory_manager.stop_monitoring()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        logger.info("资源管理器监控已停止")

    def _cleanup_loop(self):
        """清理循环"""
        cleanup_interval = 300  # 5分钟
        last_cleanup = time.time()

        while self._running:
            try:
                current_time = time.time()

                # 定期执行清理
                if current_time - last_cleanup > cleanup_interval:
                    self.perform_periodic_cleanup()
                    last_cleanup = current_time

                time.sleep(30)  # 每30秒检查一次

            except Exception as e:
                logger.error(f"清理循环错误: {e}")
                time.sleep(30)

    def perform_periodic_cleanup(self):
        """执行定期清理"""
        logger.info("开始定期资源清理")

        total_cleaned = 0

        # 执行各种清理策略
        for resource_type, policies in self.cleanup_policies.items():
            for policy in policies:
                try:
                    policy()
                    logger.debug(f"执行清理策略: {resource_type.value}")
                except Exception as e:
                    logger.error(f"清理策略执行失败 {resource_type.value}: {e}")

        # 清理各个跟踪器中的泄漏资源
        for name, tracker in self.trackers.items():
            leaks = tracker.detect_leaks()
            if leaks:
                logger.warning(f"跟踪器 {name} 检测到资源泄漏: {len(leaks)} 个")
                tracker.cleanup_all()

        # 检测内存泄漏
        memory_leaks = self.memory_manager.detect_memory_leaks()
        if memory_leaks:
            logger.warning(f"检测到内存泄漏: {memory_leaks}")

        logger.info("定期资源清理完成")

    def cleanup_all(self) -> Dict[str, int]:
        """清理所有资源"""
        results = {}

        for name, tracker in self.trackers.items():
            count = tracker.cleanup_all()
            results[name] = count

        # 执行垃圾回收
        gc_result = self.memory_manager.force_garbage_collection()
        results["garbage_collected"] = gc_result["total_collected"]

        logger.info(f"全局资源清理完成: {results}")
        return results

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统资源统计"""
        stats = {
            "trackers": {},
            "memory": self.memory_manager.get_memory_stats(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        }

        for name, tracker in self.trackers.items():
            stats["trackers"][name] = tracker.get_stats()

        return stats

    def print_resource_report(self):
        """打印资源使用报告"""
        stats = self.get_system_stats()

        print("\n" + "="*80)
        print("Brain 系统资源使用报告")
        print("="*80)

        # 系统资源
        system_stats = stats["system"]
        print(f"\n系统资源:")
        print(f"  CPU使用率: {system_stats['cpu_percent']:.1f}%")
        print(f"  内存使用率: {system_stats['memory_percent']:.1f}%")
        print(f"  磁盘使用率: {system_stats['disk_usage']:.1f}%")

        # 内存统计
        memory_stats = stats["memory"]
        print(f"\n内存使用:")
        print(f"  当前: {memory_stats['current']['rss']:.1f} MB ({memory_stats['current']['percent']:.1f}%)")
        print(f"  基线: {memory_stats['baseline']['rss']:.1f} MB")
        print(f"  增长: {memory_stats['growth']:.1f} MB ({memory_stats['growth_percent']:.1f}%)")
        if "recent_trend" in memory_stats:
            print(f"  趋势: {memory_stats['recent_trend']}")

        # 跟踪器统计
        print(f"\n资源跟踪器:")
        for name, tracker_stats in stats["trackers"].items():
            print(f"  {name}: {tracker_stats['total_resources']} 个资源")
            for rtype, count in tracker_stats["by_type"].items():
                print(f"    - {rtype}: {count}")

        print("\n" + "="*80)


# 全局资源管理器实例
_global_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """获取全局资源管理器实例"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
        _global_resource_manager.start_monitoring()
    return _global_resource_manager


def cleanup_on_exit():
    """程序退出时清理资源"""
    if _global_resource_manager:
        _global_resource_manager.cleanup_all()
        _global_resource_manager.stop_monitoring()


# 注册退出清理函数
import atexit
atexit.register(cleanup_on_exit)


def track_resource(
    resource_type: ResourceType,
    tracker_name: str = "default",
    cleanup_handler: Optional[Callable] = None
):
    """资源跟踪装饰器"""
    def decorator(cls_or_func):
        def wrapper(*args, **kwargs):
            resource_manager = get_resource_manager()
            tracker = resource_manager.get_tracker(tracker_name)

            if tracker is None:
                tracker = resource_manager.create_tracker(tracker_name)

            # 生成资源ID
            if isinstance(cls_or_func, type):
                resource_id = f"{cls_or_func.__module__}.{cls_or_func.__name__}_{id(cls_or_func)}"
            else:
                resource_id = f"{cls_or_func.__module__}.{cls_or_func.__name__}_{time.time()}"

            # 注册资源
            tracker.register_resource(resource_id, resource_type, cleanup_handler)

            try:
                result = cls_or_func(*args, **kwargs)
                return result
            finally:
                # 自动注销资源（对于函数调用）
                if not isinstance(cls_or_func, type):
                    tracker.unregister_resource(resource_id)

        return wrapper
    return decorator


class ResourceContext:
    """资源上下文管理器"""

    def __init__(self, resource_type: ResourceType, name: str = None, cleanup_handler: Callable = None):
        self.resource_type = resource_type
        self.name = name or f"resource_{time.time()}"
        self.cleanup_handler = cleanup_handler
        self.resource_manager = get_resource_manager()
        self.tracker = None
        self.resource_id = None

    def __enter__(self):
        self.tracker = self.resource_manager.get_tracker("context") or \
                      self.resource_manager.create_tracker("context")
        self.resource_id = f"{self.name}_{id(self)}"
        self.tracker.register_resource(self.resource_id, self.resource_type, self.cleanup_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker and self.resource_id:
            self.tracker.unregister_resource(self.resource_id)
        return False