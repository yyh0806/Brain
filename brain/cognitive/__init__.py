# -*- coding: utf-8 -*-
"""
认知模块 - Cognitive Module

提供感知驱动的智能决策能力：
- WorldModel: 实时维护环境认知
- DialogueManager: 多轮对话管理
- CoTEngine: 链式思维推理引擎
- PerceptionMonitor: 感知变化监控
- CognitiveLayer: 统一接口（新增）

核心架构警句（设计准则）：
感知层看到世界，认知层相信世界，规划层改变世界。

认知层只输出：状态、变化、推理、建议，绝不输出行动决策。
"""

# 统一接口（推荐使用）
from brain.cognitive.interface import (
    CognitiveLayer,
    ObservationResult,
    ObservationStatus,
    Belief,
    BeliefUpdate,
    CognitiveOutput,
    DialogueResponse
)

# 核心类型
from brain.cognitive.world_model import (
    WorldModel,
    PlanningContext,
    EnvironmentChange,
    ChangeType,
    ChangePriority,
    TrackedObject,
    ObjectState,
    SemanticObject,
    ExplorationFrontier,
    Belief as WorldBelief,
    BeliefUpdatePolicy
)

from brain.cognitive.dialogue.dialogue_manager import (
    DialogueManager,
    DialogueContext,
    DialogueType,
    DialogueState,
    DialogueMessage
)

from brain.cognitive.reasoning.cot_engine import (
    CoTEngine,
    ReasoningResult,
    ReasoningMode,
    ComplexityLevel,
    ReasoningStep
)

from brain.cognitive.monitoring.perception_monitor import (
    PerceptionMonitor,
    MonitorEvent,
    TriggerAction,
    ReplanTrigger
)

__all__ = [
    # 统一接口（推荐）
    "CognitiveLayer",
    "ObservationResult",
    "ObservationStatus",
    "Belief",
    "BeliefUpdate",
    "CognitiveOutput",
    "DialogueResponse",
    
    # 世界模型
    "WorldModel",
    "PlanningContext",
    "EnvironmentChange",
    "ChangeType",
    "ChangePriority",
    "TrackedObject",
    "ObjectState",
    "SemanticObject",
    "ExplorationFrontier",
    "WorldBelief",
    "BeliefUpdatePolicy",
    
    # 对话管理
    "DialogueManager",
    "DialogueContext",
    "DialogueType",
    "DialogueState",
    "DialogueMessage",
    
    # 推理引擎
    "CoTEngine",
    "ReasoningResult",
    "ReasoningMode",
    "ComplexityLevel",
    "ReasoningStep",
    
    # 感知监控
    "PerceptionMonitor",
    "MonitorEvent",
    "TriggerAction",
    "ReplanTrigger",
]
