# -*- coding: utf-8 -*-
"""
重规划管理器 - Replanning Manager

整合动态规划、计划修复和完全重规划功能。
支持认知层集成的环境变化检测和响应。
"""

from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from brain.planning.state import PlanState, PlanNode, NodeStatus
from brain.planning.interfaces import (
    ReplanningInput,
    ReplanningOutput,
    PlanningStatus,
    IWorldModel
)
from brain.planning.intelligent.dynamic_planner import DynamicPlanner
from brain.planning.intelligent.replanning_rules import ReplanningRules
from brain.planning.intelligent.plan_validator import PlanValidator
from brain.planning.intelligent.failure_types import FailureType


class ReplanningTrigger(Enum):
    """重规划触发原因"""
    ACTION_FAILURE = "action_failure"           # 动作执行失败
    ENVIRONMENT_CHANGE = "environment_change"   # 环境变化
    BELIEF_CONTRADICTION = "belief_contradiction"  # 信念矛盾
    PLAN_INVALID = "plan_invalid"               # 计划无效
    USER_INTERRUPT = "user_interrupt"           # 用户中断
    TIMEOUT = "timeout"                         # 超时
    OBSTACLE_DETECTED = "obstacle_detected"     # 检测到障碍
    GOAL_UNREACHABLE = "goal_unreachable"       # 目标不可达


class ReplanningStrategy(Enum):
    """重规划策略"""
    INSERT = "insert"       # 动态插入前置操作
    RETRY = "retry"         # 重试当前动作
    REPAIR = "repair"       # 修复计划（局部调整）
    REPLAN = "replan"       # 完全重新规划
    ABORT = "abort"         # 中止执行


@dataclass
class EnvironmentChange:
    """环境变化事件"""
    change_id: str
    change_type: str              # 变化类型
    description: str              # 变化描述
    severity: str                 # 严重程度: low, medium, high, critical
    affected_nodes: List[str]     # 受影响的节点ID列表
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplanningDecision:
    """重规划决策"""
    should_replan: bool           # 是否需要重规划
    strategy: ReplanningStrategy  # 使用的策略
    reason: str                   # 决策原因
    urgency: str = "normal"       # 紧急程度
    confidence: float = 1.0       # 决策置信度


class ReplanningManager:
    """
    重规划管理器

    职责：
    1. 监测环境变化和执行失败
    2. 判断是否需要重规划
    3. 选择合适的重规划策略
    4. 执行重规划并生成新计划
    5. 验证新计划的可行性

    架构：
    - 使用 DynamicPlanner 进行动态插入
    - 使用 PlanValidator 验证计划
    - 使用 ReplanningRules 判断重规划条件
    """

    def __init__(
        self,
        world_model: Optional[IWorldModel] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化重规划管理器

        Args:
            world_model: 世界模型接口
            config: 配置参数
        """
        self.world_model = world_model
        self.config = config or {}

        # 配置参数
        self.max_insertions = self.config.get("max_insertions", 3)
        self.max_retries = self.config.get("max_retries", 3)
        self.enable_plan_mending = self.config.get("enable_plan_mending", True)
        self.enable_full_replanning = self.config.get("enable_full_replanning", True)

        # 子组件
        self.dynamic_planner: Optional[DynamicPlanner] = None
        self.replanning_rules = ReplanningRules(
            max_insertions=self.max_insertions,
            max_retries=self.max_retries
        )
        self.plan_validator = PlanValidator()

        # 状态跟踪
        self.insertion_count = 0
        self.retry_count = 0
        self.replan_count = 0
        self.environment_changes: List[EnvironmentChange] = []

        # 初始化动态规划器（如果提供了世界模型）
        if world_model:
            self.dynamic_planner = DynamicPlanner(
                world_model=world_model,
                max_insertions=self.max_insertions
            )

        logger.info("ReplanningManager 初始化完成")

    # ============ 环境变化检测 ============

    def detect_environment_changes(
        self,
        current_plan: PlanState,
        current_context: Any,
        new_beliefs: List[Any],
        failed_actions: List[str]
    ) -> List[EnvironmentChange]:
        """
        检测环境变化

        Args:
            current_plan: 当前计划
            current_context: 当前上下文
            new_beliefs: 新增或更新的信念
            failed_actions: 失败的动作列表

        Returns:
            检测到的环境变化列表
        """
        changes = []

        # 1. 检测失败的节点
        for action_id in failed_actions:
            node = current_plan.get_node(action_id)
            if node:
                change = EnvironmentChange(
                    change_id=f"failure_{action_id}_{datetime.now().timestamp()}",
                    change_type=ReplanningTrigger.ACTION_FAILURE.value,
                    description=f"动作执行失败: {node.name}",
                    severity="high",
                    affected_nodes=[action_id]
                )
                changes.append(change)

        # 2. 检测信念矛盾
        for belief in new_beliefs:
            if hasattr(belief, 'falsified') and belief.falsified:
                # 找到受影响的节点
                affected = self._find_nodes_by_belief(current_plan, belief)
                change = EnvironmentChange(
                    change_id=f"belief_{belief.id}_{datetime.now().timestamp()}",
                    change_type=ReplanningTrigger.BELIEF_CONTRADICTION.value,
                    description=f"信念被证伪: {belief.content}",
                    severity="medium",
                    affected_nodes=affected
                )
                changes.append(change)

        # 3. 检测上下文变化（如果有认知层集成）
        if current_context and hasattr(current_context, 'recent_changes'):
            for ctx_change in current_context.recent_changes:
                # 根据变化类型判断严重程度
                severity = "low"
                if ctx_change.get('priority') == 'critical':
                    severity = "critical"
                elif ctx_change.get('priority') == 'high':
                    severity = "high"
                elif ctx_change.get('priority') == 'medium':
                    severity = "medium"

                # 找到受影响的节点
                affected = self._find_nodes_by_context(current_plan, ctx_change)

                change = EnvironmentChange(
                    change_id=f"context_{datetime.now().timestamp()}",
                    change_type=ReplanningTrigger.ENVIRONMENT_CHANGE.value,
                    description=ctx_change.get('description', '未知环境变化'),
                    severity=severity,
                    affected_nodes=affected,
                    metadata=ctx_change
                )
                changes.append(change)

        # 记录变化
        self.environment_changes.extend(changes)

        if changes:
            logger.info(f"检测到 {len(changes)} 个环境变化")

        return changes

    def _find_nodes_by_belief(
        self,
        plan: PlanState,
        belief: Any
    ) -> List[str]:
        """根据信念找到受影响的节点"""
        affected = []
        belief_content = belief.content if hasattr(belief, 'content') else str(belief)

        # 检查所有节点的前置条件和预期效果
        for node in plan.nodes.values():
            # 检查前置条件
            for precond in node.preconditions:
                if self._is_related_to_belief(precond, belief_content):
                    affected.append(node.id)
                    break

            # 检查预期效果
            for effect in node.expected_effects:
                if self._is_related_to_belief(effect, belief_content):
                    if node.id not in affected:
                        affected.append(node.id)
                    break

        return affected

    def _find_nodes_by_context(
        self,
        plan: PlanState,
        context_change: Dict[str, Any]
    ) -> List[str]:
        """根据上下文变化找到受影响的节点"""
        affected = []
        description = context_change.get('description', '').lower()
        change_type = context_change.get('type', '').lower()

        # 检查所有节点
        for node in plan.nodes.values():
            node_name = node.name.lower()
            node_action = (node.action or '').lower()

            # 简单匹配规则
            if 'obstacle' in change_type and 'move' in node_action:
                affected.append(node.id)
            elif 'door' in description and 'door' in node_name:
                affected.append(node.id)
            elif 'object' in description and 'search' in node_action:
                affected.append(node.id)

        return affected

    def _is_related_to_belief(self, condition: str, belief: str) -> bool:
        """判断条件是否与信念相关"""
        # 简化实现：检查是否有共同的关键词
        condition_lower = condition.lower()
        belief_lower = belief.lower()

        # 提取关键词（物体名、位置名等）
        keywords = ['door', 'kitchen', 'cup', 'water', 'table', 'living_room']
        for keyword in keywords:
            if keyword in condition_lower and keyword in belief_lower:
                return True

        return False

    # ============ 重规划决策 ============

    def make_replanning_decision(
        self,
        replanning_input: ReplanningInput,
        changes: List[EnvironmentChange]
    ) -> ReplanningDecision:
        """
        做出重规划决策

        Args:
            replanning_input: 重规划输入
            changes: 环境变化列表

        Returns:
            重规划决策
        """
        # 获取失败的节点
        failed_node = None
        if replanning_input.failed_actions:
            failed_node = replanning_input.current_plan.get_node(
                replanning_input.failed_actions[0]
            )

        # 检查严重程度
        has_critical = any(c.severity == "critical" for c in changes)
        has_high = any(c.severity == "high" for c in changes)

        # 根据触发原因和变化严重程度决定策略
        if has_critical:
            return ReplanningDecision(
                should_replan=True,
                strategy=ReplanningStrategy.REPLAN,
                reason="检测到严重环境变化，需要完全重新规划",
                urgency="critical",
                confidence=0.95
            )

        if has_high:
            # 检查是否可以修复
            if self.enable_plan_mending and len(changes) == 1:
                return ReplanningDecision(
                    should_replan=True,
                    strategy=ReplanningStrategy.REPAIR,
                    reason="检测到高优先级变化，尝试修复计划",
                    urgency="high",
                    confidence=0.85
                )
            else:
                return ReplanningDecision(
                    should_replan=True,
                    strategy=ReplanningStrategy.REPLAN,
                    reason="多个高优先级变化，完全重新规划",
                    urgency="high",
                    confidence=0.90
                )

        # 使用现有规则判断
        failure_type = self._infer_failure_type(changes, replanning_input)
        should_replan = self.replanning_rules.should_replan(
            failed_node=failed_node,
            failure_type=failure_type,
            insertion_count=self.insertion_count,
            retry_count=self.retry_count
        )

        if should_replan:
            return ReplanningDecision(
                should_replan=True,
                strategy=ReplanningStrategy.REPLAN,
                reason="触发重规划条件",
                urgency="normal",
                confidence=0.80
            )

        # 检查是否应该插入
        if self.replanning_rules.should_insert_precondition(
            failed_node=failed_node,
            failure_type=failure_type
        ):
            return ReplanningDecision(
                should_replan=False,
                strategy=ReplanningStrategy.INSERT,
                reason="插入前置操作来修复",
                urgency="low",
                confidence=0.75
            )

        # 默认：不重规划
        return ReplanningDecision(
            should_replan=False,
            strategy=ReplanningStrategy.RETRY,
            reason="重试当前动作",
            urgency="low",
            confidence=0.60
        )

    def _infer_failure_type(
        self,
        changes: List[EnvironmentChange],
        replanning_input: ReplanningInput
    ) -> FailureType:
        """推断失败类型"""
        for change in changes:
            if change.change_type == ReplanningTrigger.ENVIRONMENT_CHANGE.value:
                return FailureType.WORLD_STATE_CHANGED
            if change.change_type == ReplanningTrigger.ACTION_FAILURE.value:
                return FailureType.EXECUTION_FAILED
            if change.change_type == ReplanningTrigger.BELIEF_CONTRADICTION.value:
                return FailureType.PRECONDITION_FAILED

        return FailureType.EXECUTION_FAILED

    # ============ 重规划执行 ============

    def replan(
        self,
        replanning_input: ReplanningInput,
        decision: ReplanningDecision
    ) -> ReplanningOutput:
        """
        执行重规划

        Args:
            replanning_input: 重规划输入
            decision: 重规划决策

        Returns:
            重规划输出
        """
        logger.info(f"执行重规划: {decision.strategy.value}, 原因: {decision.reason}")

        current_plan = replanning_input.current_plan

        if decision.strategy == ReplanningStrategy.INSERT:
            return self._execute_insertion(replanning_input)
        elif decision.strategy == ReplanningStrategy.REPAIR:
            return self._execute_repair(replanning_input, decision)
        elif decision.strategy == ReplanningStrategy.REPLAN:
            return self._execute_full_replan(replanning_input, decision)
        elif decision.strategy == ReplanningStrategy.ABORT:
            return self._execute_abort(replanning_input)
        else:
            return self._execute_retry(replanning_input)

    def _execute_insertion(
        self,
        replanning_input: ReplanningInput
    ) -> ReplanningOutput:
        """执行动态插入"""
        if not self.dynamic_planner:
            logger.warning("动态规划器未初始化，无法执行插入")
            return self._create_failed_output("动态规划器未初始化")

        # 获取失败的节点
        failed_node_id = replanning_input.failed_actions[0] if replanning_input.failed_actions else None
        if not failed_node_id:
            return self._create_failed_output("没有失败的节点")

        failed_node = replanning_input.current_plan.get_node(failed_node_id)
        if not failed_node:
            return self._create_failed_output(f"找不到节点: {failed_node_id}")

        # 检查并插入前置条件
        modified, inserted_nodes = self.dynamic_planner.check_and_insert_preconditions(
            failed_node,
            list(replanning_input.current_plan.nodes.values())
        )

        if modified:
            self.insertion_count += len(inserted_nodes)

            # 将插入的节点添加到计划中
            for node in inserted_nodes:
                replanning_input.current_plan.add_node(node)

            return ReplanningOutput(
                new_plan=replanning_input.current_plan,
                replanning_type="insert",
                modified_nodes=[failed_node_id],
                added_nodes=[n.id for n in inserted_nodes],
                removed_nodes=[],
                success=True,
                reason=f"插入了 {len(inserted_nodes)} 个前置操作"
            )
        else:
            return self._create_failed_output("无法插入前置操作")

    def _execute_repair(
        self,
        replanning_input: ReplanningInput,
        decision: ReplanningDecision
    ) -> ReplanningOutput:
        """执行计划修复"""
        logger.info("执行计划修复（局部调整）")

        modified_nodes = []
        added_nodes = []
        removed_nodes = []

        # 简化实现：移除受影响的节点，标记为待重规划
        for change in self.environment_changes:
            for node_id in change.affected_nodes:
                node = replanning_input.current_plan.get_node(node_id)
                if node and node.status == NodeStatus.PENDING:
                    node.status = NodeStatus.FAILED
                    modified_nodes.append(node_id)

        return ReplanningOutput(
            new_plan=replanning_input.current_plan,
            replanning_type="repair",
            modified_nodes=modified_nodes,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            success=True,
            reason=f"修复了 {len(modified_nodes)} 个节点",
            timestamp=datetime.now()
        )

    def _execute_full_replan(
        self,
        replanning_input: ReplanningInput,
        decision: ReplanningDecision
    ) -> ReplanningOutput:
        """执行完全重新规划"""
        logger.info("执行完全重新规划")
        self.replan_count += 1

        # 完全重新规划需要调用规划器
        # 这里返回一个标记，表示需要完全重规划
        return ReplanningOutput(
            new_plan=replanning_input.current_plan,
            replanning_type="replan",
            modified_nodes=list(replanning_input.current_plan.nodes.keys()),
            added_nodes=[],
            removed_nodes=[],
            success=True,
            reason="需要完全重新规划（由上层规划器执行）",
            timestamp=datetime.now(),
            metadata={"requires_full_replanning": True}
        )

    def _execute_abort(self, replanning_input: ReplanningInput) -> ReplanningOutput:
        """中止执行"""
        logger.warning("中止计划执行")

        # 标记所有待执行节点为失败
        for node in replanning_input.current_plan.nodes.values():
            if node.status == NodeStatus.PENDING:
                node.status = NodeStatus.FAILED

        return ReplanningOutput(
            new_plan=replanning_input.current_plan,
            replanning_type="abort",
            modified_nodes=list(replanning_input.current_plan.nodes.keys()),
            added_nodes=[],
            removed_nodes=[],
            success=False,
            reason="执行被中止"
        )

    def _execute_retry(self, replanning_input: ReplanningInput) -> ReplanningOutput:
        """重试当前动作"""
        logger.info("重试当前动作")
        self.retry_count += 1

        # 不修改计划，只返回
        return ReplanningOutput(
            new_plan=replanning_input.current_plan,
            replanning_type="retry",
            modified_nodes=[],
            added_nodes=[],
            removed_nodes=[],
            success=True,
            reason=f"重试当前动作 (第 {self.retry_count} 次)"
        )

    def _create_failed_output(self, reason: str) -> ReplanningOutput:
        """创建失败的重规划输出"""
        return ReplanningOutput(
            new_plan=PlanState(),
            replanning_type="none",
            modified_nodes=[],
            added_nodes=[],
            removed_nodes=[],
            success=False,
            reason=reason
        )

    # ============ 工具方法 ============

    def validate_plan(self, plan: PlanState) -> Tuple[bool, List[str]]:
        """
        验证计划

        Returns:
            (是否有效, 问题列表)
        """
        validation = self.plan_validator.validate(plan)
        return validation.valid, validation.issues

    def reset_counters(self):
        """重置计数器"""
        self.insertion_count = 0
        self.retry_count = 0
        self.environment_changes = []
        if self.dynamic_planner:
            self.dynamic_planner.reset_insertion_count()
        logger.debug("重规划计数器已重置")

    def get_statistics(self) -> Dict[str, Any]:
        """获取重规划统计信息"""
        return {
            "insertion_count": self.insertion_count,
            "retry_count": self.retry_count,
            "replan_count": self.replan_count,
            "environment_changes": len(self.environment_changes)
        }
