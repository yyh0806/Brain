# -*- coding: utf-8 -*-
"""
规划层输入输出接口定义 - Planning Layer I/O Interfaces

定义规划层与认知层、执行层之间的数据交换接口。
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# 导入认知层类型
from brain.cognitive.reasoning.reasoning_result import ReasoningResult, ReasoningMode
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief

# 导入规划层类型
from brain.planning.state import PlanState, PlanNode, NodeStatus


class PlanningStatus(Enum):
    """规划状态"""
    SUCCESS = "success"                    # 规划成功
    FAILURE = "failure"                    # 规划失败
    PARTIAL = "partial"                    # 部分规划（部分任务无法规划）
    CLARIFICATION_NEEDED = "clarification" # 需要澄清
    REJECTED = "rejected"                  # 拒绝执行（如违反约束）


@dataclass
class PlanningInput:
    """规划层输入

    来自认知层（L3 Cognitive Layer）的输入数据，包含：
    1. 任务指令
    2. 认知层的推理结果
    3. 世界状态（PlanningContext）
    4. 关于世界的信念集合
    """
    # 1. 任务指令
    command: str                           # 自然语言指令，如 "去厨房拿杯水"
    reasoning_result: Optional[ReasoningResult] = None  # 认知层推理结果

    # 2. 世界状态
    planning_context: Optional[PlanningContext] = None  # 环境上下文
    beliefs: List[Belief] = field(default_factory=list)  # 关于世界的信念集合

    # 3. 元信息
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_reasoning(self) -> bool:
        """是否有认知层推理结果"""
        return self.reasoning_result is not None

    def get_reasoning_mode(self) -> Optional[ReasoningMode]:
        """获取推理模式"""
        if self.reasoning_result:
            return self.reasoning_result.mode
        return None

    def get_high_confidence_beliefs(self, threshold: float = 0.7) -> List[Belief]:
        """获取高置信度信念"""
        return [b for b in self.beliefs if b.confidence >= threshold and not b.falsified]

    def to_summary(self) -> str:
        """生成输入摘要（用于日志）"""
        lines = [
            f"指令: {self.command}",
            f"时间: {self.timestamp.strftime('%H:%M:%S')}"
        ]
        if self.reasoning_result:
            lines.append(f"推理模式: {self.reasoning_result.mode.value}")
            lines.append(f"推理置信度: {self.reasoning_result.confidence:.2f}")
        if self.beliefs:
            lines.append(f"信念数量: {len(self.beliefs)}")
        return "\n".join(lines)


@dataclass
class PlanningOutput:
    """规划层输出

    返回给认知层（L3）和执行层（L5）的输出数据，包含：
    1. 执行计划（PlanState）
    2. 规划元信息
    3. 规划状态
    4. 澄清请求（如果需要）
    """
    # 1. 执行计划
    plan_state: PlanState                  # 任务树（HTN结构）

    # 2. 元信息
    planning_status: PlanningStatus        # 规划状态
    estimated_duration: float              # 预计执行时间（秒）
    resource_requirements: List[str] = field(default_factory=list)  # 资源需求
    success_rate: float = 1.0              # 预期成功率

    # 3. 反馈信息
    clarification_request: Optional[str] = None  # 需要用户澄清的问题
    rejection_reason: Optional[str] = None        # 拒绝执行的原因
    planning_log: List[str] = field(default_factory=list)  # 规划过程日志

    # 4. 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_successful(self) -> bool:
        """规划是否成功"""
        return self.planning_status == PlanningStatus.SUCCESS

    def needs_clarification(self) -> bool:
        """是否需要澄清"""
        return self.planning_status == PlanningStatus.CLARIFICATION_NEEDED

    def is_rejected(self) -> bool:
        """是否被拒绝"""
        return self.planning_status == PlanningStatus.REJECTED

    def get_plan_summary(self) -> str:
        """生成计划摘要"""
        lines = [
            f"规划状态: {self.planning_status.value}",
            f"节点数量: {len(self.plan_state.nodes)}",
            f"预计时长: {self.estimated_duration:.1f}秒",
            f"预期成功率: {self.success_rate:.1%}"
        ]
        if self.resource_requirements:
            lines.append(f"资源需求: {', '.join(self.resource_requirements)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "status": self.planning_status.value,
            "estimated_duration": self.estimated_duration,
            "success_rate": self.success_rate,
            "node_count": len(self.plan_state.nodes),
            "clarification_request": self.clarification_request,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ReplanningInput:
    """重规划输入

    触发重规划时的输入数据，包含：
    1. 当前计划状态
    2. 环境变化
    3. 重规划触发原因
    """
    # 1. 当前状态
    current_plan: PlanState                # 当前执行的计划
    current_node_id: Optional[str] = None  # 当前执行的节点ID

    # 2. 变化信息
    environment_changes: List[Dict[str, Any]] = field(default_factory=list)  # 环境变化列表
    failed_actions: List[str] = field(default_factory=list)  # 失败的动作列表
    new_beliefs: List[Belief] = field(default_factory=list)  # 新增/更新的信念

    # 3. 触发信息
    trigger_reason: str = ""                # 触发原因
    urgency: str = "normal"                 # 紧急程度: low, normal, high, critical

    # 4. 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplanningOutput:
    """重规划输出

    重规划的结果，包含：
    1. 新的计划
    2. 重规划类型
    3. 变化说明
    """
    # 1. 新计划
    new_plan: PlanState                    # 新的计划
    replanning_type: str                   # 重规划类型: repair, replan, adjust

    # 2. 变化说明
    modified_nodes: List[str] = field(default_factory=list)  # 修改的节点ID
    added_nodes: List[str] = field(default_factory=list)     # 新增的节点ID
    removed_nodes: List[str] = field(default_factory=list)   # 删除的节点ID

    # 3. 元信息
    success: bool = True                   # 是否成功
    reason: str = ""                       # 重规划原因说明
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_change_summary(self) -> str:
        """获取变化摘要"""
        lines = [
            f"重规划类型: {self.replanning_type}",
            f"修改节点: {len(self.modified_nodes)}",
            f"新增节点: {len(self.added_nodes)}",
            f"删除节点: {len(self.removed_nodes)}"
        ]
        return "\n".join(lines)
