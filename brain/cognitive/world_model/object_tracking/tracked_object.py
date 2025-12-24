# -*- coding: utf-8 -*-
"""
被跟踪的物体 - Tracked Object

从 world_model.py 拆分出来的物体追踪类型定义。
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TrackedObject:
    """被跟踪的物体"""
    id: str
    object_type: str
    position: Dict[str, float]
    velocity: Dict[str, float] = field(default_factory=lambda: {"vx": 0, "vy": 0, "vz": 0})
    size: Dict[str, float] = field(default_factory=lambda: {"width": 1, "height": 1, "depth": 1})
    confidence: float = 1.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    is_obstacle: bool = False
    is_target: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)
    position_history: List[Dict[str, float]] = field(default_factory=list)




