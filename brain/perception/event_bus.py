"""
感知事件总线 - Perception Event Bus

实现发布-订阅模式，解耦感知层各模块
支持异步事件处理和流式数据传递
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class PerceptionEventType(Enum):
    """感知事件类型"""
    SENSOR_DATA = "sensor_data"              # 传感器数据更新
    FUSION_COMPLETE = "fusion_complete"       # 融合完成
    OBJECT_DETECTED = "object_detected"       # 目标检测
    OBSTACLE_DETECTED = "obstacle_detected"   # 障碍物检测
    MAP_UPDATED = "map_updated"              # 地图更新
    VLM_ANALYSIS = "vlm_analysis"            # VLM分析完成
    SENSOR_ERROR = "sensor_error"            # 传感器错误
    FUSION_ERROR = "fusion_error"            # 融合错误


@dataclass
class PerceptionEvent:
    """感知事件"""
    event_type: PerceptionEventType
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # 事件来源组件
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata
        }


class PerceptionEventBus:
    """
    感知事件总线
    
    实现发布-订阅模式，允许模块间解耦通信
    """
    
    def __init__(self):
        # 订阅者字典：event_type -> [callbacks]
        self._subscribers: Dict[PerceptionEventType, List[Callable]] = {}
        
        # 事件历史（可选，用于调试）
        self._event_history: List[PerceptionEvent] = []
        self._max_history = 100
        
        # 统计信息
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "subscribers_count": 0
        }
        
        logger.info("PerceptionEventBus 初始化完成")
    
    def subscribe(
        self,
        event_type: PerceptionEventType,
        callback: Callable[[PerceptionEvent], Any],
        priority: int = 0
    ):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            callback: 回调函数，接收 PerceptionEvent 参数
            priority: 优先级（数字越大优先级越高，默认0）
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        # 按优先级插入
        self._subscribers[event_type].append((priority, callback))
        # 按优先级排序（降序）
        self._subscribers[event_type].sort(key=lambda x: x[0], reverse=True)
        
        self._stats["subscribers_count"] = sum(len(callbacks) for callbacks in self._subscribers.values())
        logger.debug(f"订阅事件: {event_type.value}, 优先级={priority}")
    
    def unsubscribe(
        self,
        event_type: PerceptionEventType,
        callback: Callable[[PerceptionEvent], Any]
    ):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            callback: 要移除的回调函数
        """
        if event_type not in self._subscribers:
            return
        
        # 移除匹配的回调
        self._subscribers[event_type] = [
            (p, cb) for p, cb in self._subscribers[event_type]
            if cb != callback
        ]
        
        self._stats["subscribers_count"] = sum(len(callbacks) for callbacks in self._subscribers.values())
        logger.debug(f"取消订阅: {event_type.value}")
    
    async def publish(
        self,
        event_type: PerceptionEventType,
        data: Any,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        发布事件（异步）
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件来源
            metadata: 元数据
        """
        event = PerceptionEvent(
            event_type=event_type,
            data=data,
            source=source,
            metadata=metadata or {}
        )
        
        # 记录事件历史
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        self._stats["events_published"] += 1
        
        # 通知所有订阅者
        if event_type in self._subscribers:
            callbacks = self._subscribers[event_type]
            
            # 并发执行所有回调
            tasks = []
            for priority, callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        tasks.append(callback(event))
                    else:
                        # 同步函数，在线程池中执行
                        tasks.append(asyncio.to_thread(callback, event))
                except Exception as e:
                    logger.error(f"创建回调任务失败: {e}")
            
            if tasks:
                # 等待所有回调完成（允许异常）
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 统计成功和失败
                for result in results:
                    if isinstance(result, Exception):
                        self._stats["events_failed"] += 1
                        logger.error(f"事件回调执行失败: {result}")
                    else:
                        self._stats["events_processed"] += 1
        else:
            logger.debug(f"事件 {event_type.value} 没有订阅者")
    
    def publish_sync(
        self,
        event_type: PerceptionEventType,
        data: Any,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        发布事件（同步版本）
        
        注意：这会在当前事件循环中执行，如果没有事件循环会创建新的
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建任务
                asyncio.create_task(self.publish(event_type, data, source, metadata))
            else:
                # 如果事件循环未运行，直接运行
                loop.run_until_complete(self.publish(event_type, data, source, metadata))
        except RuntimeError:
            # 没有事件循环，创建新的
            asyncio.run(self.publish(event_type, data, source, metadata))
    
    def get_subscribers(self, event_type: PerceptionEventType) -> List[Callable]:
        """获取指定事件类型的所有订阅者"""
        if event_type not in self._subscribers:
            return []
        return [cb for _, cb in self._subscribers[event_type]]
    
    def get_event_history(
        self,
        event_type: Optional[PerceptionEventType] = None,
        count: int = 10
    ) -> List[PerceptionEvent]:
        """
        获取事件历史
        
        Args:
            event_type: 过滤事件类型（可选）
            count: 返回数量
            
        Returns:
            List[PerceptionEvent]: 事件列表
        """
        events = self._event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "event_types": {
                et.value: len(callbacks)
                for et, callbacks in self._subscribers.items()
            },
            "total_event_types": len(self._subscribers)
        }
    
    def clear_history(self):
        """清空事件历史"""
        self._event_history.clear()
        logger.info("事件历史已清空")
    
    def reset_statistics(self):
        """重置统计信息"""
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "subscribers_count": 0
        }
        logger.info("统计信息已重置")


# 全局事件总线实例
_global_event_bus: Optional[PerceptionEventBus] = None


def get_event_bus() -> PerceptionEventBus:
    """获取全局事件总线实例（单例模式）"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = PerceptionEventBus()
    return _global_event_bus


def set_event_bus(event_bus: PerceptionEventBus):
    """设置全局事件总线实例"""
    global _global_event_bus
    _global_event_bus = event_bus
