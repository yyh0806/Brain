"""
性能监控器 - Performance Monitor

监控感知层的性能指标，包括：
- 内存使用
- 处理时间
- 缓存命中率
- 吞吐量
"""

import time
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from loguru import logger


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 内存指标
    memory_used_mb: float = 0.0
    memory_percent: float = 0.0
    
    # 处理时间指标
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    min_processing_time_ms: float = 0.0
    
    # 吞吐量指标
    events_per_second: float = 0.0
    data_points_per_second: float = 0.0
    
    # 缓存指标
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    
    # 错误率
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory_used_mb": self.memory_used_mb,
            "memory_percent": self.memory_percent,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "max_processing_time_ms": self.max_processing_time_ms,
            "min_processing_time_ms": self.min_processing_time_ms,
            "events_per_second": self.events_per_second,
            "data_points_per_second": self.data_points_per_second,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_size": self.cache_size,
            "error_rate": self.error_rate
        }


class PerformanceMonitor:
    """
    性能监控器
    
    收集和报告感知层的性能指标
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # 处理时间历史
        self.processing_times: deque = deque(maxlen=window_size)
        
        # 事件计数
        self.event_counts: deque = deque(maxlen=window_size)
        self.data_point_counts: deque = deque(maxlen=window_size)
        
        # 错误计数
        self.error_counts: deque = deque(maxlen=window_size)
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 进程信息
        self.process = psutil.Process(os.getpid())
        
        # 上次更新时间
        self.last_update = datetime.now()
        
        logger.info("PerformanceMonitor 初始化完成")
    
    def record_processing_time(self, time_ms: float):
        """记录处理时间"""
        self.processing_times.append(time_ms)
        self.last_update = datetime.now()
    
    def record_event(self, count: int = 1):
        """记录事件"""
        self.event_counts.append((datetime.now(), count))
    
    def record_data_points(self, count: int):
        """记录数据点"""
        self.data_point_counts.append((datetime.now(), count))
    
    def record_error(self):
        """记录错误"""
        self.error_counts.append(datetime.now())
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.cache_misses += 1
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                "memory_used_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": memory_percent
            }
        except Exception as e:
            logger.warning(f"获取内存使用失败: {e}")
            return {"memory_used_mb": 0.0, "memory_percent": 0.0}
    
    def get_processing_time_stats(self) -> Dict[str, float]:
        """获取处理时间统计"""
        if not self.processing_times:
            return {
                "avg_ms": 0.0,
                "max_ms": 0.0,
                "min_ms": 0.0
            }
        
        times = list(self.processing_times)
        return {
            "avg_ms": sum(times) / len(times),
            "max_ms": max(times),
            "min_ms": min(times)
        }
    
    def get_throughput(self) -> Dict[str, float]:
        """获取吞吐量统计"""
        now = datetime.now()
        
        # 计算事件吞吐量
        events_per_second = 0.0
        if self.event_counts:
            recent_events = [
                (ts, count) for ts, count in self.event_counts
                if (now - ts).total_seconds() <= 1.0
            ]
            if recent_events:
                total_events = sum(count for _, count in recent_events)
                events_per_second = total_events
        
        # 计算数据点吞吐量
        data_points_per_second = 0.0
        if self.data_point_counts:
            recent_data = [
                (ts, count) for ts, count in self.data_point_counts
                if (now - ts).total_seconds() <= 1.0
            ]
            if recent_data:
                total_points = sum(count for _, count in recent_data)
                data_points_per_second = total_points
        
        return {
            "events_per_second": events_per_second,
            "data_points_per_second": data_points_per_second
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hit_rate": hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests
        }
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        now = datetime.now()
        recent_errors = [
            ts for ts in self.error_counts
            if (now - ts).total_seconds() <= 60.0  # 最近60秒
        ]
        
        if self.event_counts:
            recent_events = [
                (ts, count) for ts, count in self.event_counts
                if (now - ts).total_seconds() <= 60.0
            ]
            total_events = sum(count for _, count in recent_events)
            if total_events > 0:
                return len(recent_errors) / total_events
        
        return 0.0
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标"""
        memory = self.get_memory_usage()
        time_stats = self.get_processing_time_stats()
        throughput = self.get_throughput()
        cache_stats = self.get_cache_statistics()
        error_rate = self.get_error_rate()
        
        return PerformanceMetrics(
            memory_used_mb=memory["memory_used_mb"],
            memory_percent=memory["memory_percent"],
            avg_processing_time_ms=time_stats["avg_ms"],
            max_processing_time_ms=time_stats["max_ms"],
            min_processing_time_ms=time_stats["min_ms"],
            events_per_second=throughput["events_per_second"],
            data_points_per_second=throughput["data_points_per_second"],
            cache_hit_rate=cache_stats["cache_hit_rate"],
            cache_size=cache_stats.get("cache_size", 0),
            error_rate=error_rate
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        metrics = self.get_metrics()
        return {
            "metrics": metrics.to_dict(),
            "processing_times_count": len(self.processing_times),
            "event_counts_count": len(self.event_counts),
            "error_counts_count": len(self.error_counts)
        }
    
    def reset(self):
        """重置所有统计"""
        self.processing_times.clear()
        self.event_counts.clear()
        self.data_point_counts.clear()
        self.error_counts.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("性能监控统计已重置")


# 全局性能监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例（单例模式）"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def set_performance_monitor(monitor: PerformanceMonitor):
    """设置全局性能监控器实例"""
    global _global_monitor
    _global_monitor = monitor









