"""
Plan State - 规划状态

管理PlanNode集合，支持状态更新和局部重算
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from .plan_node import PlanNode, NodeStatus


@dataclass
class PlanState:
    """
    规划状态
    
    管理PlanNode集合，支持状态更新、查询和局部重算
    """
    # 根节点列表
    roots: List[PlanNode] = field(default_factory=list)
    
    # 所有节点的索引（id -> node）
    nodes: Dict[str, PlanNode] = field(default_factory=dict)
    
    # 执行历史
    execution_history: List[Dict] = field(default_factory=list)
    
    # 元数据
    metadata: Dict = field(default_factory=dict)
    
    def add_root(self, root: PlanNode):
        """添加根节点"""
        self.roots.append(root)
        self._index_node(root)
    
    def _index_node(self, node: PlanNode, visited: Optional[set] = None):
        """索引节点及其所有后代

        Args:
            node: 要索引的节点
            visited: 已访问的节点ID集合（用于检测循环）
        """
        if visited is None:
            visited = set()

        # 检测循环引用
        if node.id in visited:
            logger.warning(f"检测到循环引用: 节点 {node.id}")
            return

        visited.add(node.id)
        self.nodes[node.id] = node

        for child in node.children:
            self._index_node(child, visited)
    
    def get_node(self, node_id: str) -> Optional[PlanNode]:
        """获取节点"""
        return self.nodes.get(node_id)
    
    def get_pending_nodes(self) -> List[PlanNode]:
        """获取所有待执行的节点"""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.PENDING
        ]
    
    def get_ready_nodes(self) -> List[PlanNode]:
        """获取所有就绪的节点（前置条件满足）"""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.READY
        ]
    
    def get_executing_nodes(self) -> List[PlanNode]:
        """获取所有执行中的节点"""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.EXECUTING
        ]
    
    def get_failed_nodes(self) -> List[PlanNode]:
        """获取所有失败的节点"""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.FAILED
        ]
    
    def get_successful_nodes(self) -> List[PlanNode]:
        """获取所有成功的节点"""
        return [
            node for node in self.nodes.values()
            if node.status == NodeStatus.SUCCESS
        ]
    
    def get_leaf_nodes(self) -> List[PlanNode]:
        """获取所有叶子节点"""
        return [node for node in self.nodes.values() if node.is_leaf]
    
    def get_nodes_by_status(self, status: NodeStatus) -> List[PlanNode]:
        """根据状态获取节点"""
        return [
            node for node in self.nodes.values()
            if node.status == status
        ]
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """更新节点状态"""
        node = self.get_node(node_id)
        if node:
            node.status = status
            self._record_execution(node, status)
    
    def _record_execution(self, node: PlanNode, status: NodeStatus):
        """记录执行历史"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "node_id": node.id,
            "node_name": node.name,
            "status": status.value,
            "level": node.level
        }
        self.execution_history.append(record)
    
    def get_execution_statistics(self) -> Dict:
        """获取执行统计"""
        total = len(self.nodes)
        pending = len(self.get_pending_nodes())
        ready = len(self.get_ready_nodes())
        executing = len(self.get_executing_nodes())
        successful = len(self.get_successful_nodes())
        failed = len(self.get_failed_nodes())

        # 计算成功率（只计算已完成的节点）
        completed = successful + failed
        success_rate = successful / completed if completed > 0 else 0.0

        return {
            "total": total,  # 主键
            "total_nodes": total,  # 别名，兼容性
            "pending": pending,
            "ready": ready,
            "executing": executing,
            "successful": successful,
            "success": successful,  # 别名，兼容性
            "failed": failed,
            "success_rate": success_rate
        }
    
    def can_safely_rollback(self, node_id: str = None) -> bool:
        """
        检查是否可以安全回滚节点

        Args:
            node_id: 节点ID（可选，如果不提供则检查所有失败节点）

        Phase 1: 基础实现
        Phase 2: 检查commit_level
        """
        # 如果没有提供node_id，检查是否有任何SOFT级别的失败节点可以回滚
        if node_id is None:
            from .plan_node import CommitLevel
            for node in self.nodes.values():
                if (node.status == NodeStatus.FAILED and
                    node.commit_level == CommitLevel.SOFT):
                    return True
            return False

        node = self.get_node(node_id)
        if not node:
            return False

        # Phase 1: 简单实现，所有节点都可以回滚
        # Phase 2: 检查commit_level == SOFT
        return True
    
    def get_nodes_to_rollback(self, failed_node_id: str = None) -> List[PlanNode]:
        """
        获取需要回滚的节点列表

        Args:
            failed_node_id: 失败节点ID（可选，如果不提供则返回所有EXECUTING和FAILED节点）

        从失败节点开始，向上查找所有需要回滚的节点
        """
        # 如果没有提供failed_node_id，返回所有EXECUTING和FAILED节点
        if failed_node_id is None:
            from .plan_node import NodeStatus
            return [
                node for node in self.nodes.values()
                if node.status in [NodeStatus.EXECUTING, NodeStatus.FAILED]
            ]

        failed_node = self.get_node(failed_node_id)
        if not failed_node:
            return []

        # Phase 1: 返回失败节点及其所有后代
        rollback_nodes = [failed_node]
        rollback_nodes.extend(failed_node.get_all_descendants())

        return rollback_nodes
    
    def get_nodes_for_replanning(self, failed_node_id: str) -> List[PlanNode]:
        """
        获取需要重规划的节点列表
        
        从失败节点开始，向上查找，直到找到可以局部重算的节点
        """
        failed_node = self.get_node(failed_node_id)
        if not failed_node:
            return []
        
        # Phase 1: 简单实现，返回失败节点及其父节点
        nodes = [failed_node]
        
        # 向上查找父节点
        current = failed_node
        while current.parent:
            nodes.append(current.parent)
            current = current.parent
        
        return nodes
    
    def clone(self) -> 'PlanState':
        """克隆PlanState（用于重规划）

        创建PlanState的深拷贝，保留所有节点和状态
        """
        import copy

        # 深拷贝所有节点
        cloned = PlanState()
        cloned.metadata = self.metadata.copy()
        cloned.execution_history = self.execution_history.copy()

        # 克隆根节点及其所有后代
        for root in self.roots:
            cloned_root = copy.deepcopy(root)
            cloned.add_root(cloned_root)

        return cloned

    def to_dict(self) -> Dict:
        """转换为字典"""
        stats = self.get_execution_statistics()
        return {
            "roots": [root.to_dict() for root in self.roots],
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "total": stats["total"],
            "total_nodes": stats["total_nodes"],
            "pending": stats["pending"],
            "ready": stats["ready"],
            "executing": stats["executing"],
            "successful": stats["successful"],
            "failed": stats["failed"],
            "success_rate": stats["success_rate"],
            "metadata": self.metadata
        }
