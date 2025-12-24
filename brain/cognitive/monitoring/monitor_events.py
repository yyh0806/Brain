# -*- coding: utf-8 -*-
"""
监控事件类型定义 - Monitor Event Types

从 perception_monitor.py 拆分出来的类型定义，用于感知监控的输入输出。
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# 导入依赖类型
from brain.cognitive.world_model.world_model import EnvironmentChange, ChangeType, ChangePriority


class TriggerAction(Enum):
    """触发动作
    
    注意：这些动作只是通知类型，不直接触发执行。
    实际的动作触发由规划层决定。
    """
    REPLAN = "replan"                     # 重新规划（通知）
    CONFIRM_AND_REPLAN = "confirm_replan" # 确认后重规划（通知）
    NOTIFY_ONLY = "notify"                # 仅通知
    PAUSE = "pause"                       # 暂停执行（通知）
    ABORT = "abort"                       # 中止任务（通知）
    IGNORE = "ignore"                     # 忽略


@dataclass
class ReplanTrigger:
    """重规划触发器"""
    change_type: ChangeType
    priority: ChangePriority
    action: TriggerAction
    threshold: float  # 触发阈值
    cooldown: float   # 冷却时间（秒）
    description: str
    last_triggered: Optional[datetime] = None
    
    def can_trigger(self) -> bool:
        """检查是否可以触发（考虑冷却）"""
        if self.last_triggered is None:
            return True
        elapsed = (datetime.now() - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown
    
    def mark_triggered(self):
        """标记已触发"""
        self.last_triggered = datetime.now()


@dataclass
class MonitorEvent:
    """监控事件
    
    注意：这个事件只通知变化，不触发动作。
    实际的动作触发由规划层根据此事件决定。
    """
    change: EnvironmentChange
    trigger: ReplanTrigger
    action: TriggerAction
    timestamp: datetime = field(default_factory=datetime.now)
    handled: bool = False
    handler_result: Optional[Dict[str, Any]] = None






