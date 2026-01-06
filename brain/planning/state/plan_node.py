"""
Plan Node - 规划节点

表示规划中的一个节点，支持HTN结构和状态管理
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from loguru import logger


class NodeStatus(Enum):
    """节点状态"""
    PENDING = "pending"        # 待执行
    READY = "ready"            # 就绪（前置条件满足）
    EXECUTING = "executing"    # 执行中
    SUCCESS = "success"        # 成功
    FAILED = "failed"          # 失败
    SKIPPED = "skipped"        # 跳过
    CANCELLED = "cancelled"    # 取消


class CommitLevel(Enum):
    """提交级别"""
    HARD = "hard"  # 已对世界产生不可逆影响（如拿走杯子）
    SOFT = "soft"  # 感知/移动/查询，可安全回滚


@dataclass
class PlanNode:
    """
    规划节点
    
    表示规划中的一个节点，可以是任务、技能或操作
    支持HTN结构（层级任务网络）
    """
    # 基本信息
    id: str
    name: str
    
    # 层级信息（用于HTN结构）
    task: Optional[str] = None      # 所属任务
    skill: Optional[str] = None     # 所属技能
    action: Optional[str] = None   # 具体操作
    
    # 状态
    status: NodeStatus = NodeStatus.PENDING
    
    # 前置条件和预期效果
    preconditions: List[str] = field(default_factory=list)
    expected_effects: List[str] = field(default_factory=list)
    
    # 参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # HTN结构：子节点
    children: List['PlanNode'] = field(default_factory=list)
    parent: Optional['PlanNode'] = None
    
    # 不确定性（用于Skill/Task层）
    uncertainty: float = 0.0  # 0.0 = 确定, 1.0 = 完全不确定
    
    # 重试
    retry_count: int = 0
    max_retries: int = 3
    
    # 提交级别（Phase 2实现）
    commit_level: CommitLevel = CommitLevel.SOFT
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点（没有子节点）"""
        return len(self.children) == 0
    
    @property
    def is_root(self) -> bool:
        """是否为根节点（没有父节点）"""
        return self.parent is None
    
    @property
    def level(self) -> int:
        """节点层级（0为根节点）"""
        if self.is_root:
            return 0
        return self.parent.level + 1
    
    @property
    def execution_time(self) -> Optional[float]:
        """执行时间（秒）"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def add_child(self, child: 'PlanNode'):
        """添加子节点"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'PlanNode'):
        """移除子节点"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def get_all_descendants(self) -> List['PlanNode']:
        """获取所有后代节点"""
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.get_all_descendants())
        return result
    
    def find_node(self, node_id: str) -> Optional['PlanNode']:
        """查找节点（包括自身和所有后代）"""
        if self.id == node_id:
            return self
        
        for child in self.children:
            found = child.find_node(node_id)
            if found:
                return found
        
        return None
    
    def mark_started(self):
        """标记为开始执行"""
        self.status = NodeStatus.EXECUTING
        self.started_at = datetime.now()
    
    def mark_success(self):
        """标记为成功"""
        self.status = NodeStatus.SUCCESS
        self.completed_at = datetime.now()
    
    def mark_failed(self):
        """标记为失败"""
        self.status = NodeStatus.FAILED
        self.completed_at = datetime.now()
        self.retry_count += 1
    
    def can_retry(self) -> bool:
        """是否可以重试"""
        return self.retry_count < self.max_retries
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "task": self.task,
            "skill": self.skill,
            "action": self.action,
            "status": self.status.value,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "parameters": self.parameters,
            "uncertainty": self.uncertainty,
            "retry_count": self.retry_count,
            "commit_level": self.commit_level.value,
            "is_leaf": self.is_leaf,
            "level": self.level,
            "children_count": len(self.children),
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        return f"PlanNode(id={self.id}, name={self.name}, status={self.status.value})"
