"""
回滚管理器 - Rollback Manager

负责:
- 管理操作的回滚
- 状态恢复
- 安全回退执行
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

if TYPE_CHECKING:
    from brain.state.world_state import WorldState
    from brain.communication.robot_interface import RobotInterface

from brain.execution.operations.base import (
    Operation, 
    OperationResult, 
    OperationStatus,
    OperationType
)


class RollbackStatus(Enum):
    """回滚状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class RollbackStep:
    """回滚步骤"""
    operation: Operation
    rollback_action: Optional[Operation]
    executed: bool = False
    success: bool = False
    result: Optional[OperationResult] = None


@dataclass
class RollbackPlan:
    """回滚计划"""
    id: str
    mission_id: str
    steps: List[RollbackStep]
    target_index: int  # 回滚到的操作索引
    created_at: datetime = field(default_factory=datetime.now)
    status: RollbackStatus = RollbackStatus.PENDING
    
    @property
    def total_steps(self) -> int:
        return len(self.steps)
    
    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.executed)


@dataclass
class RollbackResult:
    """回滚结果"""
    success: bool
    status: RollbackStatus
    steps_executed: int
    steps_total: int
    restored_state: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


class RollbackManager:
    """
    回滚管理器
    
    管理任务的回滚操作，确保安全恢复到之前的状态
    """
    
    # 操作的默认回滚映射
    DEFAULT_ROLLBACK_MAP = {
        "takeoff": "land",
        "goto": "return_to_home",
        "pickup": "dropoff",
        "start_engine": "stop_engine",
        "undock": "dock",
        "arm": "disarm"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 最大回滚步数
        self.max_rollback_steps = config.get("max_rollback_steps", 10)
        
        # 回滚历史
        self.rollback_history: List[RollbackPlan] = []
        
        logger.info("RollbackManager 初始化完成")
    
    def create_rollback_plan(
        self,
        mission_id: str,
        operations: List[Operation],
        current_index: int,
        target_index: int = 0
    ) -> RollbackPlan:
        """
        创建回滚计划
        
        Args:
            mission_id: 任务ID
            operations: 操作列表
            current_index: 当前执行到的操作索引
            target_index: 目标回滚到的索引
            
        Returns:
            RollbackPlan: 回滚计划
        """
        # 限制回滚范围
        actual_target = max(
            target_index,
            current_index - self.max_rollback_steps
        )
        
        # 收集需要回滚的操作
        steps = []
        for i in range(current_index - 1, actual_target - 1, -1):
            if i < 0 or i >= len(operations):
                continue
            
            op = operations[i]
            rollback_action = self._get_rollback_action(op)
            
            steps.append(RollbackStep(
                operation=op,
                rollback_action=rollback_action
            ))
        
        plan = RollbackPlan(
            id=f"rollback_{mission_id}_{datetime.now().timestamp()}",
            mission_id=mission_id,
            steps=steps,
            target_index=actual_target
        )
        
        logger.info(f"创建回滚计划: {plan.id}, 步数={len(steps)}")
        
        return plan
    
    def _get_rollback_action(self, operation: Operation) -> Optional[Operation]:
        """获取操作的回滚动作"""
        # 首先检查操作自身是否定义了回滚
        if operation.rollback_action:
            return operation.rollback_action
        
        # 使用默认映射
        rollback_name = self.DEFAULT_ROLLBACK_MAP.get(operation.name)
        if rollback_name:
            return Operation.from_dict({
                "name": rollback_name,
                "type": operation.type.value,
                "platform": operation.platform,
                "parameters": {},
                "is_rollback": True,
                "metadata": {"rollback_for": operation.id}
            })
        
        # 某些操作不需要回滚
        no_rollback_operations = [
            "wait", "check_status", "capture_image", "record_video",
            "send_telemetry", "broadcast_status"
        ]
        if operation.name in no_rollback_operations:
            return None
        
        # 移动操作的通用回滚
        if operation.type == OperationType.MOVEMENT:
            if operation.name != "return_to_home":
                return Operation.from_dict({
                    "name": "return_to_home",
                    "type": "movement",
                    "platform": operation.platform,
                    "parameters": {},
                    "is_rollback": True
                })
        
        return None
    
    async def execute_rollback(
        self,
        plan: RollbackPlan,
        robot_interface: 'RobotInterface',
        world_state: 'WorldState'
    ) -> RollbackResult:
        """
        执行回滚计划
        
        Args:
            plan: 回滚计划
            robot_interface: 机器人接口
            world_state: 世界状态
            
        Returns:
            RollbackResult: 回滚结果
        """
        logger.info(f"开始执行回滚: {plan.id}")
        
        plan.status = RollbackStatus.IN_PROGRESS
        errors = []
        
        for step in plan.steps:
            if step.rollback_action is None:
                # 无需回滚此步骤
                step.executed = True
                step.success = True
                continue
            
            try:
                logger.info(f"执行回滚步骤: {step.rollback_action.name}")
                
                # 执行回滚操作
                result = await self._execute_rollback_step(
                    step.rollback_action,
                    robot_interface
                )
                
                step.executed = True
                step.result = result
                step.success = result.status == OperationStatus.SUCCESS
                
                if not step.success:
                    errors.append(
                        f"步骤 {step.operation.name} 回滚失败: {result.error_message}"
                    )
                    
                    # 检查是否可以继续
                    if self._is_critical_failure(step):
                        logger.error("关键回滚步骤失败，中止回滚")
                        break
                        
            except Exception as e:
                step.executed = True
                step.success = False
                errors.append(f"步骤 {step.operation.name} 回滚异常: {e}")
                logger.error(f"回滚步骤异常: {e}")
        
        # 确定最终状态
        if all(s.success for s in plan.steps if s.executed):
            plan.status = RollbackStatus.SUCCESS
        elif any(s.success for s in plan.steps if s.executed):
            plan.status = RollbackStatus.PARTIAL
        else:
            plan.status = RollbackStatus.FAILED
        
        # 记录历史
        self.rollback_history.append(plan)
        
        result = RollbackResult(
            success=plan.status == RollbackStatus.SUCCESS,
            status=plan.status,
            steps_executed=plan.completed_steps,
            steps_total=plan.total_steps,
            errors=errors
        )
        
        logger.info(f"回滚完成: 状态={plan.status.value}, 执行={plan.completed_steps}/{plan.total_steps}")
        
        return result
    
    async def _execute_rollback_step(
        self,
        operation: Operation,
        robot_interface: 'RobotInterface'
    ) -> OperationResult:
        """执行单个回滚步骤"""
        try:
            # 发送回滚命令
            response = await robot_interface.send_command(
                command=operation.name,
                parameters=operation.parameters
            )
            
            if response.success:
                # 等待完成
                completion = await robot_interface.wait_for_completion(
                    operation_id=operation.id,
                    timeout=operation.timeout or 60
                )
                
                if completion.completed:
                    return OperationResult(
                        status=OperationStatus.SUCCESS,
                        data=completion.data
                    )
                else:
                    return OperationResult(
                        status=OperationStatus.FAILED,
                        error_message=completion.error or "回滚操作未能完成"
                    )
            else:
                return OperationResult(
                    status=OperationStatus.FAILED,
                    error_message=response.error or "回滚命令发送失败"
                )
                
        except Exception as e:
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message=str(e)
            )
    
    def _is_critical_failure(self, step: RollbackStep) -> bool:
        """判断是否为关键失败"""
        # 安全类操作失败视为关键
        if step.operation.type == OperationType.SAFETY:
            return True
        
        # 起飞/降落失败视为关键
        critical_ops = ["takeoff", "land", "emergency_stop", "emergency_land"]
        if step.operation.name in critical_ops:
            return True
        
        return False
    
    def can_rollback(
        self,
        operations: List[Operation],
        current_index: int
    ) -> bool:
        """
        检查是否可以执行回滚
        
        Args:
            operations: 操作列表
            current_index: 当前索引
            
        Returns:
            bool: 是否可以回滚
        """
        if current_index <= 0:
            return False
        
        # 检查是否有可回滚的操作
        for i in range(current_index - 1, -1, -1):
            op = operations[i]
            if self._get_rollback_action(op) is not None:
                return True
        
        return False
    
    def estimate_rollback_time(
        self,
        plan: RollbackPlan
    ) -> float:
        """
        估算回滚时间
        
        Args:
            plan: 回滚计划
            
        Returns:
            float: 预计时间(秒)
        """
        total_time = 0.0
        
        for step in plan.steps:
            if step.rollback_action:
                total_time += step.rollback_action.estimated_duration
            else:
                total_time += 1.0  # 默认1秒
        
        return total_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取回滚统计"""
        status_counts = {}
        total_steps = 0
        successful_steps = 0
        
        for plan in self.rollback_history:
            status_counts[plan.status.value] = (
                status_counts.get(plan.status.value, 0) + 1
            )
            total_steps += plan.total_steps
            successful_steps += sum(1 for s in plan.steps if s.success)
        
        return {
            "total_rollbacks": len(self.rollback_history),
            "by_status": status_counts,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "success_rate": (
                successful_steps / total_steps if total_steps > 0 else 0
            )
        }

