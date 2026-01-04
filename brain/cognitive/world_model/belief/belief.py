# -*- coding: utf-8 -*-
"""
信念定义 - Belief Definition

信念是认知层维护的关于世界的假设。
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Belief:
    """信念定义
    
    信念是认知层维护的关于世界的假设，如"杯子在厨房"。
    信念可以被证据支持或证伪。
    """
    id: str
    content: str  # 信念内容，如 "杯子在厨房"
    confidence: float  # 置信度 0-1
    evidence_count: int = 0  # 支持证据数量
    falsified: bool = False  # 是否已被证伪
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "falsified": self.falsified,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    def update_confidence(self, new_confidence: float):
        """更新置信度"""
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.last_updated = datetime.now()
    
    def add_evidence(self):
        """增加支持证据"""
        self.evidence_count += 1
        self.last_updated = datetime.now()
    
    def mark_falsified(self):
        """标记为已证伪"""
        self.falsified = True
        self.confidence = 0.0
        self.last_updated = datetime.now()










