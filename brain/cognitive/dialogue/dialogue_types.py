# -*- coding: utf-8 -*-
"""
对话类型定义 - Dialogue Types

从 dialogue_manager.py 拆分出来的类型定义，用于对话管理的输入输出。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DialogueType(Enum):
    """对话类型"""
    CLARIFICATION = "clarification"       # 指令澄清
    CONFIRMATION = "confirmation"         # 执行确认
    PROGRESS_REPORT = "progress_report"   # 进度汇报
    ERROR_REPORT = "error_report"        # 错误汇报
    SUGGESTION = "suggestion"            # 建议
    INFORMATION = "information"          # 信息通知
    USER_INPUT = "user_input"           # 用户输入


class DialogueState(Enum):
    """对话状态"""
    IDLE = "idle"
    WAITING_USER_RESPONSE = "waiting_user_response"
    PROCESSING = "processing"
    COMPLETED = "completed"


@dataclass
class DialogueMessage:
    """对话消息"""
    role: str  # "system", "user", "assistant"
    content: str
    dialogue_type: DialogueType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "type": self.dialogue_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class DialogueContext:
    """对话上下文"""
    session_id: str
    mission_id: Optional[str] = None
    history: List[DialogueMessage] = field(default_factory=list)
    state: DialogueState = DialogueState.IDLE
    pending_question: Optional[str] = None
    pending_options: Optional[List[str]] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, message: DialogueMessage):
        """添加消息到历史"""
        self.history.append(message)
    
    def get_history_for_llm(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """获取用于LLM的对话历史"""
        recent = self.history[-max_turns * 2:] if len(self.history) > max_turns * 2 else self.history
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def get_summary(self) -> str:
        """获取对话摘要"""
        if not self.history:
            return "无对话历史"
        
        last_messages = self.history[-3:]
        summary = "\n".join([f"[{m.role}]: {m.content[:50]}..." for m in last_messages])
        return f"最近对话:\n{summary}"




