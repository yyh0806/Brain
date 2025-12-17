"""
认知模块 - Cognitive Module

提供感知驱动的智能决策能力：
- WorldModel: 实时维护环境认知
- DialogueManager: 多轮对话管理
- CoTEngine: 链式思维推理引擎
- PerceptionMonitor: 感知变化监控
"""

from brain.cognitive.world_model.world_model import WorldModel, EnvironmentChange, ChangeType
from brain.cognitive.dialogue.dialogue_manager import DialogueManager, DialogueContext, DialogueType
from brain.cognitive.reasoning.cot_engine import CoTEngine, ReasoningResult, ReasoningStep
from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor, ReplanTrigger

__all__ = [
    "WorldModel",
    "EnvironmentChange",
    "ChangeType",
    "DialogueManager",
    "DialogueContext",
    "DialogueType",
    "CoTEngine",
    "ReasoningResult",
    "ReasoningStep",
    "PerceptionMonitor",
    "ReplanTrigger",
]

