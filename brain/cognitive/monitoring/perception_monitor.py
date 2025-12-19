"""
感知变化监控器 - Perception Monitor

负责：
- 持续监控感知数据，检测显著变化
- 评估变化是否需要触发重规划
- 根据变化类型和优先级触发相应回调
- 管理重规划触发策略
"""

from typing import Dict, List, Any, Optional, Callable, Awaitable, Set
from datetime import datetime, timedelta
import asyncio
from loguru import logger

from brain.cognitive.world_model import WorldModel, EnvironmentChange, ChangeType, ChangePriority

# 导入类型定义
from brain.cognitive.monitoring.monitor_events import (
    TriggerAction,
    ReplanTrigger,
    MonitorEvent
)


class PerceptionMonitor:
    """
    感知变化监控器
    
    持续监控感知数据，检测显著变化并触发相应动作
    """
    
    # 默认触发器配置
    DEFAULT_TRIGGERS = {
        ChangeType.NEW_OBSTACLE: ReplanTrigger(
            change_type=ChangeType.NEW_OBSTACLE,
            priority=ChangePriority.HIGH,
            action=TriggerAction.REPLAN,
            threshold=0.7,
            cooldown=5.0,
            description="检测到新障碍物"
        ),
        ChangeType.PATH_BLOCKED: ReplanTrigger(
            change_type=ChangeType.PATH_BLOCKED,
            priority=ChangePriority.CRITICAL,
            action=TriggerAction.CONFIRM_AND_REPLAN,
            threshold=0.9,
            cooldown=2.0,
            description="路径被阻塞"
        ),
        ChangeType.TARGET_MOVED: ReplanTrigger(
            change_type=ChangeType.TARGET_MOVED,
            priority=ChangePriority.MEDIUM,
            action=TriggerAction.REPLAN,
            threshold=0.6,
            cooldown=10.0,
            description="目标位置变化"
        ),
        ChangeType.TARGET_APPEARED: ReplanTrigger(
            change_type=ChangeType.TARGET_APPEARED,
            priority=ChangePriority.HIGH,
            action=TriggerAction.CONFIRM_AND_REPLAN,
            threshold=0.6,
            cooldown=5.0,
            description="发现新目标"
        ),
        ChangeType.TARGET_LOST: ReplanTrigger(
            change_type=ChangeType.TARGET_LOST,
            priority=ChangePriority.HIGH,
            action=TriggerAction.NOTIFY_ONLY,
            threshold=0.7,
            cooldown=5.0,
            description="目标丢失"
        ),
        ChangeType.WEATHER_CHANGED: ReplanTrigger(
            change_type=ChangeType.WEATHER_CHANGED,
            priority=ChangePriority.MEDIUM,
            action=TriggerAction.CONFIRM_AND_REPLAN,
            threshold=0.7,
            cooldown=30.0,
            description="天气变化"
        ),
        ChangeType.BATTERY_LOW: ReplanTrigger(
            change_type=ChangeType.BATTERY_LOW,
            priority=ChangePriority.CRITICAL,
            action=TriggerAction.REPLAN,
            threshold=0.95,
            cooldown=60.0,
            description="电池电量低"
        ),
        ChangeType.GEOFENCE_APPROACH: ReplanTrigger(
            change_type=ChangeType.GEOFENCE_APPROACH,
            priority=ChangePriority.CRITICAL,
            action=TriggerAction.PAUSE,
            threshold=0.9,
            cooldown=5.0,
            description="接近地理围栏"
        ),
    }
    
    def __init__(
        self,
        world_model: WorldModel,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            world_model: 世界模型实例
            config: 配置选项
        """
        self.world_model = world_model
        self.config = config or {}
        
        # 触发器
        self.triggers: Dict[ChangeType, ReplanTrigger] = dict(self.DEFAULT_TRIGGERS)
        
        # 回调函数
        self._replan_callback: Optional[Callable[[MonitorEvent], Awaitable[None]]] = None
        self._confirmation_callback: Optional[Callable[[MonitorEvent], Awaitable[bool]]] = None
        self._notification_callback: Optional[Callable[[MonitorEvent], Awaitable[None]]] = None
        
        # 监控状态
        self._running = False
        self._paused = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 事件队列
        self.pending_events: List[MonitorEvent] = []
        self.event_history: List[MonitorEvent] = []
        self.max_history = 100
        
        # 监控参数
        self.monitor_interval = self.config.get("monitor_interval", 0.5)  # 监控间隔（秒）
        self.change_buffer_time = self.config.get("change_buffer_time", 1.0)  # 变化缓冲时间
        
        # 临时忽略的变化类型
        self._ignored_types: Set[ChangeType] = set()
        
        logger.info("PerceptionMonitor 初始化完成")
    
    def set_replan_callback(self, callback: Callable[[MonitorEvent], Awaitable[None]]):
        """设置重规划回调"""
        self._replan_callback = callback
    
    def set_confirmation_callback(self, callback: Callable[[MonitorEvent], Awaitable[bool]]):
        """设置确认回调"""
        self._confirmation_callback = callback
    
    def set_notification_callback(self, callback: Callable[[MonitorEvent], Awaitable[None]]):
        """设置通知回调"""
        self._notification_callback = callback
    
    def configure_trigger(
        self,
        change_type: ChangeType,
        action: Optional[TriggerAction] = None,
        threshold: Optional[float] = None,
        cooldown: Optional[float] = None
    ):
        """配置触发器"""
        if change_type in self.triggers:
            trigger = self.triggers[change_type]
            if action is not None:
                trigger.action = action
            if threshold is not None:
                trigger.threshold = threshold
            if cooldown is not None:
                trigger.cooldown = cooldown
            logger.info(f"更新触发器 {change_type.value}: action={trigger.action.value}")
    
    def ignore_change_type(self, change_type: ChangeType):
        """临时忽略某种变化类型"""
        self._ignored_types.add(change_type)
    
    def unignore_change_type(self, change_type: ChangeType):
        """取消忽略某种变化类型"""
        self._ignored_types.discard(change_type)
    
    async def start_monitoring(self):
        """启动监控循环"""
        if self._running:
            logger.warning("监控已在运行中")
            return
        
        self._running = True
        self._paused = False
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("感知监控已启动")
    
    async def stop_monitoring(self):
        """停止监控循环"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("感知监控已停止")
    
    def pause_monitoring(self):
        """暂停监控"""
        self._paused = True
        logger.info("感知监控已暂停")
    
    def resume_monitoring(self):
        """恢复监控"""
        self._paused = False
        logger.info("感知监控已恢复")
    
    async def _monitor_loop(self):
        """监控循环"""
        logger.info("开始感知监控循环")
        
        while self._running:
            try:
                if not self._paused:
                    await self._check_for_changes()
                
                await asyncio.sleep(self.monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_for_changes(self):
        """检查感知变化"""
        # 获取显著变化
        changes = self.world_model.detect_significant_changes()
        
        for change in changes:
            # 检查是否被忽略
            if change.change_type in self._ignored_types:
                continue
            
            # 查找对应的触发器
            trigger = self.triggers.get(change.change_type)
            if not trigger:
                continue
            
            # 检查置信度是否达到阈值
            if change.confidence < trigger.threshold:
                continue
            
            # 检查冷却时间
            if not trigger.can_trigger():
                logger.debug(f"触发器 {change.change_type.value} 在冷却中")
                continue
            
            # 创建事件
            event = MonitorEvent(
                change=change,
                trigger=trigger,
                action=trigger.action
            )
            
            # 标记触发
            trigger.mark_triggered()
            
            # 处理事件
            await self._handle_event(event)
    
    async def _handle_event(self, event: MonitorEvent):
        """处理监控事件"""
        logger.info(f"处理感知事件: {event.change.change_type.value}, 动作: {event.action.value}")
        
        self.pending_events.append(event)
        
        try:
            if event.action == TriggerAction.REPLAN:
                if self._replan_callback:
                    await self._replan_callback(event)
                event.handled = True
                
            elif event.action == TriggerAction.CONFIRM_AND_REPLAN:
                # 先请求确认
                confirmed = True
                if self._confirmation_callback:
                    confirmed = await self._confirmation_callback(event)
                
                if confirmed and self._replan_callback:
                    await self._replan_callback(event)
                event.handled = True
                event.handler_result = {"confirmed": confirmed}
                
            elif event.action == TriggerAction.NOTIFY_ONLY:
                if self._notification_callback:
                    await self._notification_callback(event)
                event.handled = True
                
            elif event.action == TriggerAction.PAUSE:
                self._paused = True
                if self._notification_callback:
                    await self._notification_callback(event)
                event.handled = True
                
            elif event.action == TriggerAction.ABORT:
                # 中止操作需要特殊处理
                if self._notification_callback:
                    await self._notification_callback(event)
                event.handled = True
                
            elif event.action == TriggerAction.IGNORE:
                event.handled = True
            
        except Exception as e:
            logger.error(f"处理事件失败: {e}")
            event.handler_result = {"error": str(e)}
        
        # 移到历史
        self.pending_events.remove(event)
        self.event_history.append(event)
        
        # 限制历史长度
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    async def process_sensor_update(self, sensor_data: Dict[str, Any]) -> List[MonitorEvent]:
        """
        处理传感器更新并检测需要响应的变化
        
        这是一个更主动的方法，可以在获取新传感器数据时直接调用
        
        Args:
            sensor_data: 传感器数据
            
        Returns:
            触发的事件列表
        """
        # 更新世界模型
        changes = self.world_model.update_from_perception(sensor_data)
        
        events = []
        
        for change in changes:
            # 检查是否被忽略
            if change.change_type in self._ignored_types:
                continue
            
            # 查找对应的触发器
            trigger = self.triggers.get(change.change_type)
            if not trigger:
                continue
            
            # 检查置信度是否达到阈值
            if change.confidence < trigger.threshold:
                continue
            
            # 检查冷却时间
            if not trigger.can_trigger():
                continue
            
            # 创建事件
            event = MonitorEvent(
                change=change,
                trigger=trigger,
                action=trigger.action
            )
            
            trigger.mark_triggered()
            events.append(event)
        
        return events
    
    def evaluate_change_significance(
        self,
        change: EnvironmentChange,
        current_mission_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        评估变化的显著性
        
        Args:
            change: 环境变化
            current_mission_context: 当前任务上下文
            
        Returns:
            评估结果
        """
        trigger = self.triggers.get(change.change_type)
        
        result = {
            "is_significant": False,
            "priority": change.priority.value,
            "confidence": change.confidence,
            "recommended_action": TriggerAction.IGNORE.value,
            "reason": ""
        }
        
        if not trigger:
            result["reason"] = "无对应触发器"
            return result
        
        # 基于置信度
        if change.confidence < trigger.threshold:
            result["reason"] = f"置信度不足: {change.confidence:.2f} < {trigger.threshold:.2f}"
            return result
        
        # 基于优先级
        priority_values = {
            ChangePriority.CRITICAL: 4,
            ChangePriority.HIGH: 3,
            ChangePriority.MEDIUM: 2,
            ChangePriority.LOW: 1,
            ChangePriority.INFO: 0
        }
        
        if priority_values.get(change.priority, 0) >= 2:
            result["is_significant"] = True
            result["recommended_action"] = trigger.action.value
            result["reason"] = f"优先级{change.priority.value}，需要处理"
        
        # 考虑任务上下文
        if current_mission_context:
            # 如果变化影响当前任务目标
            if change.change_type == ChangeType.TARGET_LOST:
                current_target = current_mission_context.get("target_id")
                lost_target = change.data.get("object_id")
                if current_target == lost_target:
                    result["is_significant"] = True
                    result["recommended_action"] = TriggerAction.CONFIRM_AND_REPLAN.value
                    result["reason"] = "当前任务目标丢失"
            
            # 如果变化阻塞当前路径
            if change.change_type == ChangeType.PATH_BLOCKED:
                result["is_significant"] = True
                result["recommended_action"] = TriggerAction.REPLAN.value
                result["reason"] = "路径被阻塞，需要重规划"
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "running": self._running,
            "paused": self._paused,
            "pending_events": len(self.pending_events),
            "total_events_handled": len(self.event_history),
            "ignored_types": [t.value for t in self._ignored_types],
            "triggers": {
                k.value: {
                    "action": v.action.value,
                    "threshold": v.threshold,
                    "cooldown": v.cooldown,
                    "can_trigger": v.can_trigger()
                }
                for k, v in self.triggers.items()
            }
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的事件"""
        recent = self.event_history[-count:]
        return [
            {
                "change_type": e.change.change_type.value,
                "priority": e.change.priority.value,
                "action": e.action.value,
                "description": e.change.description,
                "timestamp": e.timestamp.isoformat(),
                "handled": e.handled
            }
            for e in recent
        ]

