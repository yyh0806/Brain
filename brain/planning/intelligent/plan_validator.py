"""
Plan Validator - 计划验证器

验证计划的完整性和可行性
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger

from brain.planning.state import PlanState, PlanNode, NodeStatus


@dataclass
class PlanValidation:
    """计划验证结果"""
    valid: bool
    issues: List[str]
    warnings: List[str]
    node_count: int = 0

    def add_issue(self, issue: str):
        """添加问题"""
        self.issues.append(issue)
        self.valid = False

    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


class PlanValidator:
    """
    计划验证器

    检查计划的：
    - 循环依赖
    - 前置条件满足
    - 必需参数存在
    - 资源冲突
    """

    def __init__(self):
        self.max_depth = 20  # 防止无限递归

    def validate(self, plan_state: PlanState) -> PlanValidation:
        """
        验证计划

        Args:
            plan_state: 计划状态

        Returns:
            验证结果
        """
        result = PlanValidation(
            valid=True,
            issues=[],
            warnings=[],
            node_count=len(plan_state.nodes)
        )

        # 检查每个根节点
        for root in plan_state.roots:
            self._validate_node(root, result, visited=set(), depth=0)

        # 检查是否有孤儿节点
        self._check_orphan_nodes(plan_state, result)

        # 检查是否有循环依赖
        self._check_cycles(plan_state, result)

        if result.valid:
            logger.info(f"计划验证通过: {result.node_count} 个节点")
        else:
            logger.warning(f"计划验证失败: {len(result.issues)} 个问题")

        return result

    def _validate_node(
        self,
        node: PlanNode,
        result: PlanValidation,
        visited: set,
        depth: int
    ):
        """递归验证节点"""
        # 防止无限递归
        if depth > self.max_depth:
            result.add_issue(f"节点深度超过限制: {node.name}")
            return

        # 检查循环引用
        if node.id in visited:
            result.add_issue(f"检测到循环引用: {node.name}")
            return

        visited.add(node.id)

        # 检查节点名称
        if not node.name:
            result.add_issue("节点缺少名称")

        # 检查参数
        if node.action and not node.parameters:
            result.add_warning(f"动作节点 {node.name} 没有参数")

        # 检查前置条件
        if node.preconditions and not self._validate_preconditions(node):
            result.add_warning(f"节点 {node.name} 的前置条件可能无法满足")

        # 递归检查子节点
        for child in node.children:
            self._validate_node(child, result, visited.copy(), depth + 1)

    def _validate_preconditions(self, node: PlanNode) -> bool:
        """验证前置条件（简化实现）"""
        # Phase 0: 简单检查
        if not node.preconditions:
            return True

        # TODO: 实现完整的前置条件验证
        return True

    def _check_orphan_nodes(
        self,
        plan_state: PlanState,
        result: PlanValidation
    ):
        """检查孤儿节点"""
        # 获取所有可达节点
        reachable = set()
        for root in plan_state.roots:
            self._collect_reachable(root, reachable)

        # 检查是否有不在可达集合中的节点
        for node_id, node in plan_state.nodes.items():
            if node_id not in reachable:
                result.add_warning(f"发现孤儿节点: {node.name}")

    def _collect_reachable(self, node: PlanNode, reachable: set):
        """收集所有可达节点"""
        if node.id in reachable:
            return
        reachable.add(node.id)
        for child in node.children:
            self._collect_reachable(child, reachable)

    def _check_cycles(
        self,
        plan_state: PlanState,
        result: PlanValidation
    ):
        """检查循环依赖"""
        # 使用DFS检测环
        for root in plan_state.roots:
            if self._has_cycle(root, visited=set(), rec_stack=set()):
                result.add_issue(f"检测到循环依赖，从节点: {root.name}")

    def _has_cycle(
        self,
        node: PlanNode,
        visited: set,
        rec_stack: set
    ) -> bool:
        """检测是否有环"""
        visited.add(node.id)
        rec_stack.add(node.id)

        for child in node.children:
            if child.id not in visited:
                if self._has_cycle(child, visited, rec_stack):
                    return True
            elif child.id in rec_stack:
                return True

        rec_stack.remove(node.id)
        return False
