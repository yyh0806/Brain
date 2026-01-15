# -*- coding: utf-8 -*-
"""
环境变化类型定义 - Environment Change Types

从 world_model.py 拆分出来的环境变化相关类型定义。
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ChangeType(Enum):
    """环境变化类型"""
    NEW_OBSTACLE = "new_obstacle"           # 新障碍物出现
    OBSTACLE_MOVED = "obstacle_moved"       # 障碍物移动
    OBSTACLE_REMOVED = "obstacle_removed"   # 障碍物消失
    TARGET_APPEARED = "target_appeared"     # 目标出现
    TARGET_MOVED = "target_moved"           # 目标移动
    TARGET_LOST = "target_lost"             # 目标丢失
    PATH_BLOCKED = "path_blocked"           # 路径被阻塞
    PATH_CLEARED = "path_cleared"           # 路径畅通
    WEATHER_CHANGED = "weather_changed"     # 天气变化
    BATTERY_LOW = "battery_low"             # 电池电量低
    SIGNAL_DEGRADED = "signal_degraded"     # 信号降级
    NEW_POI = "new_poi"                     # 新兴趣点
    GEOFENCE_APPROACH = "geofence_approach" # 接近地理围栏


class ChangePriority(Enum):
    """变化优先级"""
    CRITICAL = "critical"   # 必须立即处理
    HIGH = "high"           # 高优先级
    MEDIUM = "medium"       # 中等优先级
    LOW = "low"             # 低优先级
    INFO = "info"           # 仅信息


@dataclass
class EnvironmentChange:
    """环境变化记录"""
    change_type: ChangeType
    priority: ChangePriority
    description: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    requires_replan: bool = False
    requires_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.change_type.value,
            "priority": self.priority.value,
            "description": self.description,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "requires_replan": self.requires_replan,
            "requires_confirmation": self.requires_confirmation
        }











