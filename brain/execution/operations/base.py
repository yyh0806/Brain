"""
基础操作定义 - Base Operations

定义原子操作的基础类和接口
"""

from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

if TYPE_CHECKING:
    from brain.state.world_state import WorldState


class OperationType(Enum):
    """操作类型"""
    MOVEMENT = "movement"           # 移动操作
    PERCEPTION = "perception"       # 感知操作
    MANIPULATION = "manipulation"   # 操作/执行
    COMMUNICATION = "communication" # 通信操作
    CONTROL = "control"             # 控制操作
    SAFETY = "safety"               # 安全操作


class OperationStatus(Enum):
    """操作状态"""
    PENDING = "pending"             # 待执行
    QUEUED = "queued"               # 队列中
    EXECUTING = "executing"         # 执行中
    SUCCESS = "success"             # 成功
    FAILED = "failed"               # 失败
    CANCELLED = "cancelled"         # 取消
    TIMEOUT = "timeout"             # 超时
    SKIPPED = "skipped"             # 跳过


class OperationPriority(Enum):
    """操作优先级"""
    CRITICAL = 1    # 关键
    HIGH = 2        # 高
    NORMAL = 3      # 普通
    LOW = 4         # 低
    BACKGROUND = 5  # 后台


@dataclass
class Precondition:
    """前置条件"""
    name: str
    condition: str  # 条件表达式
    description: str = ""
    required: bool = True
    
    async def check(self, world_state: 'WorldState') -> bool:
        """检查条件是否满足"""
        try:
            # 简单的条件评估
            # 实际实现可能需要更复杂的表达式解析
            return world_state.evaluate_condition(self.condition)
        except Exception:
            return not self.required  # 非必需条件失败时返回True


@dataclass
class Postcondition:
    """后置条件"""
    name: str
    expected_state: str  # 期望状态表达式
    description: str = ""
    timeout: float = 30.0  # 等待超时
    
    async def verify(self, world_state: 'WorldState') -> bool:
        """验证条件是否达成"""
        try:
            return world_state.evaluate_condition(self.expected_state)
        except Exception:
            return False


@dataclass
class OperationResult:
    """操作结果"""
    status: OperationStatus
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retryable: bool = True
    
    @property
    def duration(self) -> Optional[float]:
        """执行时长(秒)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "data": self.data,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "duration": self.duration
        }


@dataclass
class Operation:
    """
    原子操作
    
    表示一个可执行的最小操作单元
    """
    id: str
    name: str
    type: OperationType
    platform: str  # drone, ugv, usv
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 时间估计
    estimated_duration: float = 5.0  # 秒
    timeout: Optional[float] = None
    
    # 优先级
    priority: OperationPriority = OperationPriority.NORMAL
    
    # 条件
    preconditions: List[Precondition] = field(default_factory=list)
    postconditions: List[Postcondition] = field(default_factory=list)
    
    # 回滚
    rollback_action: Optional['Operation'] = None
    is_rollback: bool = False
    
    # 来源
    source_task_id: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 执行状态
    status: OperationStatus = OperationStatus.PENDING
    result: Optional[OperationResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if self.timeout is None:
            self.timeout = self.estimated_duration * 3 + 30
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "platform": self.platform,
            "parameters": self.parameters,
            "estimated_duration": self.estimated_duration,
            "priority": self.priority.value,
            "status": self.status.value,
            "preconditions": [
                {"name": p.name, "condition": p.condition}
                for p in self.preconditions
            ],
            "postconditions": [
                {"name": p.name, "expected_state": p.expected_state}
                for p in self.postconditions
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        """从字典创建"""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=data["name"],
            type=OperationType(data.get("type", "control")),
            platform=data.get("platform", "drone"),
            parameters=data.get("parameters", {}),
            estimated_duration=data.get("estimated_duration", 5.0),
            priority=OperationPriority(data.get("priority", 3)),
            metadata=data.get("metadata", {})
        )
    
    def clone(self) -> 'Operation':
        """克隆操作"""
        return Operation(
            id=str(uuid.uuid4())[:8],
            name=self.name,
            type=self.type,
            platform=self.platform,
            parameters=dict(self.parameters),
            estimated_duration=self.estimated_duration,
            timeout=self.timeout,
            priority=self.priority,
            preconditions=list(self.preconditions),
            postconditions=list(self.postconditions),
            source_task_id=self.source_task_id,
            metadata=dict(self.metadata)
        )


class OperationBuilder:
    """操作构建器"""
    
    def __init__(self, name: str, platform: str):
        self._name = name
        self._platform = platform
        self._type = OperationType.CONTROL
        self._parameters: Dict[str, Any] = {}
        self._preconditions: List[Precondition] = []
        self._postconditions: List[Postcondition] = []
        self._duration = 5.0
        self._priority = OperationPriority.NORMAL
        self._metadata: Dict[str, Any] = {}
    
    def of_type(self, op_type: OperationType) -> 'OperationBuilder':
        """设置操作类型"""
        self._type = op_type
        return self
    
    def with_params(self, **params) -> 'OperationBuilder':
        """设置参数"""
        self._parameters.update(params)
        return self
    
    def with_precondition(
        self, 
        name: str, 
        condition: str,
        description: str = "",
        required: bool = True
    ) -> 'OperationBuilder':
        """添加前置条件"""
        self._preconditions.append(Precondition(
            name=name,
            condition=condition,
            description=description,
            required=required
        ))
        return self
    
    def with_postcondition(
        self,
        name: str,
        expected_state: str,
        description: str = ""
    ) -> 'OperationBuilder':
        """添加后置条件"""
        self._postconditions.append(Postcondition(
            name=name,
            expected_state=expected_state,
            description=description
        ))
        return self
    
    def with_duration(self, duration: float) -> 'OperationBuilder':
        """设置预计时长"""
        self._duration = duration
        return self
    
    def with_priority(self, priority: OperationPriority) -> 'OperationBuilder':
        """设置优先级"""
        self._priority = priority
        return self
    
    def with_metadata(self, **metadata) -> 'OperationBuilder':
        """设置元数据"""
        self._metadata.update(metadata)
        return self
    
    def build(self) -> Operation:
        """构建操作"""
        return Operation(
            id=str(uuid.uuid4())[:8],
            name=self._name,
            type=self._type,
            platform=self._platform,
            parameters=self._parameters,
            estimated_duration=self._duration,
            priority=self._priority,
            preconditions=self._preconditions,
            postconditions=self._postconditions,
            metadata=self._metadata
        )


class OperationFactory:
    """
    操作工厂
    
    提供创建标准操作的便捷方法
    """
    
    @staticmethod
    def create_movement(
        name: str,
        platform: str,
        target_position: Dict[str, float],
        speed: Optional[float] = None,
        **kwargs
    ) -> Operation:
        """创建移动操作"""
        params = {"position": target_position}
        if speed:
            params["speed"] = speed
        params.update(kwargs)
        
        return (OperationBuilder(name, platform)
                .of_type(OperationType.MOVEMENT)
                .with_params(**params)
                .with_duration(30.0)
                .build())
    
    @staticmethod
    def create_perception(
        name: str,
        platform: str,
        perception_type: str,
        **kwargs
    ) -> Operation:
        """创建感知操作"""
        return (OperationBuilder(name, platform)
                .of_type(OperationType.PERCEPTION)
                .with_params(perception_type=perception_type, **kwargs)
                .with_duration(5.0)
                .build())
    
    @staticmethod
    def create_wait(
        platform: str,
        duration: float
    ) -> Operation:
        """创建等待操作"""
        return (OperationBuilder("wait", platform)
                .of_type(OperationType.CONTROL)
                .with_params(duration=duration)
                .with_duration(duration)
                .build())
    
    @staticmethod
    def create_emergency_stop(platform: str) -> Operation:
        """创建紧急停止操作"""
        return (OperationBuilder("emergency_stop", platform)
                .of_type(OperationType.SAFETY)
                .with_priority(OperationPriority.CRITICAL)
                .with_duration(1.0)
                .build())

