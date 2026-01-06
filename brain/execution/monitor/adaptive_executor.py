"""
自适应执行器 - Adaptive Executor

集成Executor、ExecutionMonitor、DynamicPlanner和ReplanningRules
支持自动失败检测、动态插入和重规划
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from loguru import logger

from brain.planning.state import PlanNode, PlanState, NodeStatus
from brain.planning.action_level import WorldModelMock
from brain.execution.executor import Executor, ExecutionMode
from brain.execution.operations.base import Operation, OperationResult, OperationStatus
from brain.state.world_state import WorldState
from .execution_monitor import ExecutionMonitor
from .failure_classifier import FailureType
from brain.planning.intelligent import DynamicPlanner, ReplanningRules


class AdaptiveExecutor:
    """
    自适应执行器
    
    职责：
    - 执行PlanNode序列
    - 监控执行状态
    - 检测失败并分类
    - 根据失败类型选择策略（插入/重试/重规划）
    """
    
    def __init__(
        self,
        executor: Executor,
        world_model: WorldModelMock,
        world_state: WorldState,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化自适应执行器
        
        Args:
            executor: 基础执行器
            world_model: 世界模型
            world_state: 世界状态
            config: 配置
        """
        self.executor = executor
        self.world_model = world_model
        self.world_state = world_state
        self.config = config or {}
        
        # 组件
        self.monitor = ExecutionMonitor()
        self.dynamic_planner = DynamicPlanner(
            world_model=world_model,
            max_insertions=self.config.get("max_insertions", 3)
        )
        self.replanning_rules = ReplanningRules(
            max_insertions=self.config.get("max_insertions", 3),
            max_retries=self.config.get("max_retries", 3)
        )
        
        # 执行状态
        self.current_plan_state: Optional[PlanState] = None
        self.execution_history: List[Dict] = []
        
        logger.info("AdaptiveExecutor 初始化完成")
    
    async def execute_plan(
        self,
        plan_state: PlanState,
        robot_interface: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        执行计划
        
        Args:
            plan_state: 计划状态
            robot_interface: 机器人接口（可选，Phase 0使用None）
            
        Returns:
            执行结果
        """
        self.current_plan_state = plan_state
        self.dynamic_planner.reset_insertion_count()
        self.monitor.clear()
        
        logger.info(f"开始执行计划，共 {len(plan_state.nodes)} 个节点")
        
        # 获取所有叶子节点（实际执行的操作）
        leaf_nodes = plan_state.get_leaf_nodes()
        
        # 按顺序执行
        for node in leaf_nodes:
            if node.status in [NodeStatus.SUCCESS, NodeStatus.SKIPPED]:
                continue
            
            result = await self._execute_node(node, robot_interface)
            
            if result.status == OperationStatus.SUCCESS:
                self.monitor.record_success(node, result)
                # 更新世界模型
                self._update_world_model(node, result)
            else:
                # 处理失败
                failure_type = self.monitor.record_failure(node, result)
                recovery_action = await self._handle_failure(
                    node, failure_type, result
                )
                
                if recovery_action == "replan":
                    logger.warning("需要重规划，停止执行")
                    break
        
        # 返回执行结果
        stats = self.monitor.get_statistics()
        return {
            "success": len(plan_state.get_successful_nodes()) > 0,
            "statistics": stats,
            "failed_nodes": self.monitor.get_failed_nodes(),
            "plan_state": plan_state.to_dict()
        }
    
    async def _execute_node(
        self,
        node: PlanNode,
        robot_interface: Optional[Any]
    ) -> OperationResult:
        """
        执行单个节点
        
        Args:
            node: 节点
            robot_interface: 机器人接口
            
        Returns:
            操作结果
        """
        # 检查前置条件并插入必要操作
        inserted_nodes = self.dynamic_planner.check_and_insert_preconditions(
            node, node.task
        )
        
        # 如果有插入的节点，先执行它们
        for inserted_node in inserted_nodes:
            logger.info(f"执行插入的操作: {inserted_node.name}")
            inserted_result = await self._execute_node_action(inserted_node, robot_interface)
            
            if inserted_result.status != OperationStatus.SUCCESS:
                logger.warning(f"插入的操作失败: {inserted_node.name}")
                return inserted_result
            
            # 更新世界模型
            self._update_world_model(inserted_node, inserted_result)
        
        # 执行主节点
        self.monitor.start_execution(node)
        result = await self._execute_node_action(node, robot_interface)
        
        return result
    
    async def _execute_node_action(
        self,
        node: PlanNode,
        robot_interface: Optional[Any]
    ) -> OperationResult:
        """
        执行节点的实际操作
        
        Args:
            node: 节点
            robot_interface: 机器人接口
            
        Returns:
            操作结果
        """
        # Phase 0/1: 使用模拟执行
        # 将PlanNode转换为Operation
        operation = self._node_to_operation(node)
        
        # 使用Executor执行（Phase 0使用DRY_RUN模式）
        self.executor.set_mode(ExecutionMode.DRY_RUN)
        result = await self.executor.execute(operation, robot_interface)
        
        return result
    
    def _node_to_operation(self, node: PlanNode) -> Operation:
        """将PlanNode转换为Operation"""
        from brain.execution.operations.base import Operation, OperationType, OperationPriority
        
        # 映射操作类型
        type_mapping = {
            "movement": OperationType.MOVEMENT,
            "manipulation": OperationType.MANIPULATION,
            "perception": OperationType.PERCEPTION,
            "control": OperationType.CONTROL
        }
        
        # 从能力注册表获取类型（简化实现）
        op_type = OperationType.CONTROL
        if "move" in node.name or "goto" in node.name:
            op_type = OperationType.MOVEMENT
        elif "grasp" in node.name or "place" in node.name or "open" in node.name:
            op_type = OperationType.MANIPULATION
        elif "search" in node.name or "check" in node.name or "detect" in node.name:
            op_type = OperationType.PERCEPTION
        
        operation = Operation(
            id=node.id,
            name=node.action or node.name,
            type=op_type,
            platform="ugv",  # 默认UGV
            parameters=node.parameters,
            estimated_duration=5.0,
            priority=OperationPriority.NORMAL
        )
        
        return operation
    
    def _update_world_model(self, node: PlanNode, result: OperationResult):
        """更新世界模型"""
        if result.status != OperationStatus.SUCCESS:
            return
        
        # 根据操作类型更新世界模型
        if node.action == "open_door":
            door_id = node.parameters.get("door_id")
            if door_id:
                self.world_model.set_door_state(door_id, "open")
                logger.debug(f"世界模型更新: 门 {door_id} 已打开")
        
        elif node.action == "move_to":
            position = node.parameters.get("position")
            if position:
                self.world_model.set_robot_position(position)
                logger.debug(f"世界模型更新: 机器人位置 {position}")
    
    async def _handle_failure(
        self,
        node: PlanNode,
        failure_type: FailureType,
        result: OperationResult
    ) -> str:
        """
        处理失败
        
        Args:
            node: 失败的节点
            failure_type: 失败类型
            result: 操作结果
            
        Returns:
            恢复动作：'insert', 'retry', 'replan'
        """
        insertion_count = self.dynamic_planner.insertion_count
        retry_count = node.retry_count
        
        recovery_action = self.replanning_rules.get_recovery_action(
            node, failure_type, insertion_count, retry_count
        )
        
        logger.info(
            f"节点 {node.name} 失败，类型: {failure_type.value}, "
            f"恢复动作: {recovery_action}"
        )
        
        if recovery_action == "insert":
            # 动态插入已在_execute_node中处理
            pass
        elif recovery_action == "retry":
            # 重试（由Executor处理）
            pass
        elif recovery_action == "replan":
            # 需要重规划
            pass
        
        return recovery_action
