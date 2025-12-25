# -*- coding: utf-8 -*-
"""
推理结果类型定义 - Reasoning Result Types

从 cot_engine.py 拆分出来的类型定义，用于推理引擎的输入输出。
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ReasoningMode(Enum):
    """推理模式"""
    PLANNING = "planning"               # 任务规划
    REPLANNING = "replanning"           # 重新规划
    EXCEPTION_HANDLING = "exception"    # 异常处理
    CLARIFICATION = "clarification"   # 指令澄清
    DECISION = "decision"               # 决策判断


class ComplexityLevel(Enum):
    """复杂度等级"""
    SIMPLE = "simple"       # 简单，直接执行
    MODERATE = "moderate"   # 中等，简单推理
    COMPLEX = "complex"     # 复杂，完整CoT
    CRITICAL = "critical"  # 关键，深度推理+验证


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    question: str
    analysis: str
    conclusion: str
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "question": self.question,
            "analysis": self.analysis,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence
        }
    
    def __str__(self) -> str:
        return f"步骤{self.step_number}: {self.question}\n分析: {self.analysis}\n结论: {self.conclusion}"


@dataclass
class ReasoningResult:
    """推理结果
    
    注意：decision 字段只包含解释性判断（"为什么变化"），不是动作指令。
    suggestion 字段是建议性提示，规划层决定是否采纳。
    """
    mode: ReasoningMode
    query: str
    context_summary: str
    complexity: ComplexityLevel
    chain: List[ReasoningStep]
    decision: str  # 解释性判断，不是动作指令
    suggestion: str  # 建议性提示，不是决策
    confidence: float
    raw_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "query": self.query,
            "complexity": self.complexity.value,
            "chain": [step.to_dict() for step in self.chain],
            "decision": self.decision,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_chain_summary(self) -> str:
        """获取推理链摘要"""
        if not self.chain:
            return "无推理步骤"
        
        lines = ["推理过程:"]
        for step in self.chain:
            lines.append(f"  {step.step_number}. {step.question}")
            lines.append(f"     → {step.conclusion}")
        lines.append(f"\n最终判断: {self.decision}")
        
        return "\n".join(lines)








