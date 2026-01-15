# -*- coding: utf-8 -*-
"""
语义物体类型定义 - Semantic Object Types

从 world_model.py 拆分出来的语义理解相关类型定义。
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# 导入 VLM 相关类型（可选）
try:
    from brain.perception.understanding.vlm_perception import BoundingBox
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    BoundingBox = None


class ObjectState(Enum):
    """物体状态（用于语义物体）"""
    DETECTED = "detected"       # 首次检测到
    TRACKED = "tracked"         # 持续追踪中
    LOST = "lost"              # 丢失
    CONFIRMED = "confirmed"     # 确认存在


@dataclass
class SemanticObject:
    """语义物体（从 SemanticWorldModel 迁移）"""
    id: str
    label: str
    
    # 位置信息
    world_position: Tuple[float, float] = (0.0, 0.0)  # 世界坐标
    local_position: Optional[Tuple[float, float]] = None  # 相对机器人坐标
    
    # 边界框（相对图像）
    bbox: Optional[BoundingBox] = None
    
    # 状态
    state: ObjectState = ObjectState.DETECTED
    confidence: float = 0.0
    
    # 描述
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    observation_count: int = 1
    
    # 是否为目标
    is_target: bool = False
    target_type: str = ""  # 如 "destination", "landmark", "obstacle"
    
    def update_observation(self, confidence: float, position: Tuple[float, float] = None):
        """更新观测"""
        self.last_seen = datetime.now()
        self.observation_count += 1
        self.confidence = min(1.0, self.confidence * 0.8 + confidence * 0.2)
        
        if position:
            # 平滑位置更新
            alpha = 0.7
            self.world_position = (
                alpha * self.world_position[0] + (1 - alpha) * position[0],
                alpha * self.world_position[1] + (1 - alpha) * position[1]
            )
        
        if self.state == ObjectState.LOST:
            self.state = ObjectState.TRACKED
        elif self.observation_count >= 3:
            self.state = ObjectState.CONFIRMED
    
    def mark_lost(self):
        """标记为丢失"""
        self.state = ObjectState.LOST
    
    def is_valid(self, max_age: float = 60.0) -> bool:
        """检查是否有效"""
        age = (datetime.now() - self.last_seen).total_seconds()
        return age < max_age and self.state != ObjectState.LOST
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "world_position": self.world_position,
            "state": self.state.value,
            "confidence": self.confidence,
            "description": self.description,
            "is_target": self.is_target
        }


@dataclass
class ExplorationFrontier:
    """探索边界（从 SemanticWorldModel 迁移）"""
    id: str
    position: Tuple[float, float]  # 边界位置
    direction: float  # 方向（弧度）
    
    # 探索优先级
    priority: float = 0.5
    
    # 状态
    explored: bool = False
    reachable: bool = True
    
    # 探索收益预估
    expected_info_gain: float = 0.5
    
    timestamp: datetime = field(default_factory=datetime.now)












