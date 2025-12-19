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
    
    def _index_node(self, node: PlanNode):
        """索引节点及其所有后代"""
        self.nodes[node.id] = node
        for child in node.children:
            self._index_node(child)
    
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
        
        return {
            "total_nodes": total,
            "pending": pending,
            "ready": ready,
            "executing": executing,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0
        }
    
    def can_safely_rollback(self, node_id: str) -> bool:
        """
        检查是否可以安全回滚节点
        
        Phase 2实现：检查commit_level
        """
        node = self.get_node(node_id)
        if not node:
            return False
        
        # Phase 0: 简单实现，所有节点都可以回滚
        # Phase 2: 检查commit_level == SOFT
        return True
    
    def get_nodes_to_rollback(self, failed_node_id: str) -> List[PlanNode]:
        """
        获取需要回滚的节点列表
        
        从失败节点开始，向上查找所有需要回滚的节点
        """
        failed_node = self.get_node(failed_node_id)
        if not failed_node:
            return []
        
        # Phase 0: 简单实现，只返回失败节点
        # Phase 2: 实现完整的回滚逻辑，考虑commit_level
        return [failed_node]
    
    def clone(self) -> 'PlanState':
        """克隆PlanState（用于重规划）"""
        # Phase 0: 简单实现
        # Phase 2: 实现深度克隆，保留已完成节点
        cloned = PlanState()
        # TODO: 实现深度克隆逻辑
        return cloned
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "roots": [root.to_dict() for root in self.roots],
            "total_nodes": len(self.nodes),
            "statistics": self.get_execution_statistics(),
            "metadata": self.metadata
        }
