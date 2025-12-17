"""
任务规划器 - Task Planner

负责将解析后的任务转换为可执行的原子操作序列
支持:
- 感知驱动的规划（结合世界模型）
- 依赖分析
- 并行任务识别
- 资源约束检查
- 时序规划
- CoT推理集成
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from loguru import logger

from brain.execution.operations.base import (
    Operation, 
    OperationType,
    OperationPriority,
    Precondition,
    Postcondition
)
from brain.state.world_state import WorldState

if TYPE_CHECKING:
    from brain.cognitive.world_model import PlanningContext
    from brain.cognitive.cot_engine import ReasoningResult


class PlannerStrategy(Enum):
    """规划策略"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行
    HIERARCHICAL = "hierarchical"  # 分层规划
    REACTIVE = "reactive"          # 响应式
    PERCEPTION_DRIVEN = "perception_driven"  # 感知驱动


@dataclass
class TaskNode:
    """任务节点"""
    id: str
    name: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    children: List['TaskNode'] = field(default_factory=list)
    estimated_duration: float = 0.0  # seconds
    priority: int = 1


@dataclass
class PlanResult:
    """规划结果"""
    success: bool
    operations: List[Operation]
    estimated_total_time: float
    parallel_groups: List[List[str]]  # 可并行执行的操作组
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 新增：规划推理记录
    reasoning_summary: Optional[str] = None
    perception_context_used: bool = False


class TaskPlanner:
    """
    任务规划器
    
    将高层任务分解为原子操作序列，支持感知驱动的智能规划
    """
    
    def __init__(
        self, 
        world_state: WorldState,
        config: Optional[Dict[str, Any]] = None
    ):
        self.world_state = world_state
        self.config = config or {}
        
        # 操作库 - 定义每种操作类型的模板
        self.operation_templates = self._init_operation_templates()
        
        # 任务分解规则
        self.decomposition_rules = self._init_decomposition_rules()
        
        # 规划策略
        self.default_strategy = PlannerStrategy(
            self.config.get("strategy", "perception_driven")
        ) if self.config.get("strategy") else PlannerStrategy.PERCEPTION_DRIVEN
        
        logger.info(f"TaskPlanner 初始化完成 (策略: {self.default_strategy.value})")
    
    def _init_operation_templates(self) -> Dict[str, Dict]:
        """初始化操作模板"""
        return {
            # 通用操作
            "wait": {
                "type": OperationType.CONTROL,
                "params": ["duration"],
                "default_duration": 0.0
            },
            "check_status": {
                "type": OperationType.PERCEPTION,
                "params": [],
                "default_duration": 0.5
            },
            
            # 移动操作
            "takeoff": {
                "type": OperationType.MOVEMENT,
                "params": ["altitude"],
                "platform": ["drone"],
                "default_duration": 10.0
            },
            "land": {
                "type": OperationType.MOVEMENT,
                "params": ["position"],
                "platform": ["drone"],
                "default_duration": 15.0
            },
            "goto": {
                "type": OperationType.MOVEMENT,
                "params": ["position", "speed", "heading"],
                "platform": ["drone", "ugv", "usv"],
                "default_duration": 30.0
            },
            "hover": {
                "type": OperationType.MOVEMENT,
                "params": ["duration", "position"],
                "platform": ["drone"],
                "default_duration": 5.0
            },
            "orbit": {
                "type": OperationType.MOVEMENT,
                "params": ["center", "radius", "speed", "direction"],
                "platform": ["drone"],
                "default_duration": 60.0
            },
            "follow_path": {
                "type": OperationType.MOVEMENT,
                "params": ["waypoints", "speed"],
                "platform": ["drone", "ugv", "usv"],
                "default_duration": 120.0
            },
            "return_to_home": {
                "type": OperationType.MOVEMENT,
                "params": [],
                "platform": ["drone", "ugv", "usv"],
                "default_duration": 60.0
            },
            "avoid_obstacle": {
                "type": OperationType.MOVEMENT,
                "params": ["obstacle_id", "avoidance_strategy"],
                "platform": ["drone", "ugv", "usv"],
                "default_duration": 15.0
            },
            
            # 感知操作
            "scan_area": {
                "type": OperationType.PERCEPTION,
                "params": ["area", "resolution"],
                "default_duration": 30.0
            },
            "capture_image": {
                "type": OperationType.PERCEPTION,
                "params": ["target", "zoom"],
                "default_duration": 2.0
            },
            "record_video": {
                "type": OperationType.PERCEPTION,
                "params": ["duration", "quality"],
                "default_duration": 30.0
            },
            "detect_objects": {
                "type": OperationType.PERCEPTION,
                "params": ["object_types", "area"],
                "default_duration": 5.0
            },
            "measure_distance": {
                "type": OperationType.PERCEPTION,
                "params": ["target"],
                "default_duration": 1.0
            },
            "update_perception": {
                "type": OperationType.PERCEPTION,
                "params": [],
                "default_duration": 2.0
            },
            
            # 任务操作
            "pickup": {
                "type": OperationType.MANIPULATION,
                "params": ["object_id", "grip_force"],
                "default_duration": 10.0
            },
            "dropoff": {
                "type": OperationType.MANIPULATION,
                "params": ["position", "release_height"],
                "default_duration": 10.0
            },
            "spray": {
                "type": OperationType.MANIPULATION,
                "params": ["substance", "amount", "area"],
                "default_duration": 20.0
            },
            
            # 通信操作
            "send_data": {
                "type": OperationType.COMMUNICATION,
                "params": ["data", "destination"],
                "default_duration": 2.0
            },
            "receive_command": {
                "type": OperationType.COMMUNICATION,
                "params": ["timeout"],
                "default_duration": 5.0
            },
            "broadcast_status": {
                "type": OperationType.COMMUNICATION,
                "params": [],
                "default_duration": 1.0
            }
        }
    
    def _init_decomposition_rules(self) -> Dict[str, List[str]]:
        """初始化任务分解规则"""
        return {
            # 高层任务 -> 子任务序列
            "patrol": ["takeoff", "follow_path", "scan_area", "return_to_home", "land"],
            "survey": ["takeoff", "goto", "scan_area", "capture_image", "return_to_home", "land"],
            "delivery": ["takeoff", "goto:pickup", "pickup", "goto:destination", "dropoff", "return_to_home", "land"],
            "inspection": ["takeoff", "goto", "hover", "capture_image", "record_video", "return_to_home", "land"],
            "search_and_rescue": ["takeoff", "follow_path", "detect_objects", "orbit", "send_data", "return_to_home", "land"],
            "monitoring": ["takeoff", "goto", "hover", "record_video", "broadcast_status", "return_to_home", "land"],
            "mapping": ["takeoff", "follow_path", "scan_area", "capture_image", "send_data", "return_to_home", "land"],
            "tracking": ["takeoff", "detect_objects", "follow_path", "capture_image", "send_data", "return_to_home", "land"],
        }
    
    async def plan(
        self,
        parsed_task: Dict[str, Any],
        platform_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Operation]:
        """
        生成操作序列（基础版本）
        
        Args:
            parsed_task: 解析后的任务信息
            platform_type: 平台类型
            constraints: 安全约束
            
        Returns:
            List[Operation]: 操作序列
        """
        logger.info(f"开始规划任务: {parsed_task.get('task_type', 'unknown')}")
        
        constraints = constraints or {}
        
        # Step 1: 构建任务树
        task_tree = self._build_task_tree(parsed_task)
        
        # Step 2: 分解为原子操作
        raw_operations = await self._decompose_task_tree(task_tree, platform_type)
        
        # Step 3: 应用约束
        constrained_operations = self._apply_constraints(raw_operations, constraints)
        
        # Step 4: 优化操作序列
        optimized_operations = self._optimize_sequence(constrained_operations)
        
        # Step 5: 添加前置/后置条件
        final_operations = self._add_conditions(optimized_operations)
        
        # Step 6: 验证计划
        validation = await self._validate_plan(final_operations, platform_type)
        if not validation.valid:
            logger.warning(f"计划验证失败: {validation.issues}")
            final_operations = await self._fix_plan(final_operations, validation.issues)
        
        logger.info(f"规划完成, 生成 {len(final_operations)} 个操作")
        return final_operations
    
    async def plan_with_perception(
        self,
        parsed_task: Dict[str, Any],
        platform_type: str,
        planning_context: 'PlanningContext',
        cot_result: Optional['ReasoningResult'] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Operation]:
        """
        感知驱动的规划
        
        结合实时感知数据和CoT推理结果生成更智能的操作序列
        
        Args:
            parsed_task: 解析后的任务信息
            platform_type: 平台类型
            planning_context: 感知驱动的规划上下文
            cot_result: CoT推理结果
            constraints: 安全约束
            
        Returns:
            List[Operation]: 操作序列
        """
        logger.info(f"开始感知驱动规划: {parsed_task.get('task_type', 'unknown')}")
        
        constraints = constraints or {}
        
        # Step 1: 构建任务树
        task_tree = self._build_task_tree(parsed_task)
        
        # Step 2: 分解为原子操作
        raw_operations = await self._decompose_task_tree(task_tree, platform_type)
        
        # Step 3: 根据感知上下文调整操作
        perception_adjusted_ops = self._adjust_for_perception(
            raw_operations, 
            planning_context,
            platform_type
        )
        
        # Step 4: 根据CoT推理结果优化
        if cot_result:
            perception_adjusted_ops = self._apply_cot_suggestions(
                perception_adjusted_ops,
                cot_result
            )
        
        # Step 5: 应用约束
        constrained_operations = self._apply_constraints(perception_adjusted_ops, constraints)
        
        # Step 6: 优化操作序列
        optimized_operations = self._optimize_sequence(constrained_operations)
        
        # Step 7: 添加感知更新操作
        operations_with_perception = self._insert_perception_updates(
            optimized_operations,
            platform_type
        )
        
        # Step 8: 添加前置/后置条件
        final_operations = self._add_conditions(operations_with_perception)
        
        # Step 9: 验证计划
        validation = await self._validate_plan(final_operations, platform_type)
        if not validation.valid:
            logger.warning(f"计划验证失败: {validation.issues}")
            final_operations = await self._fix_plan(final_operations, validation.issues)
        
        logger.info(f"感知驱动规划完成, 生成 {len(final_operations)} 个操作")
        return final_operations
    
    def _adjust_for_perception(
        self,
        operations: List[Operation],
        planning_context: 'PlanningContext',
        platform_type: str
    ) -> List[Operation]:
        """根据感知上下文调整操作"""
        adjusted_ops = []
        
        for op in operations:
            adjusted_op = op
            
            # 根据障碍物调整路径
            if op.name in ["goto", "follow_path"] and planning_context.obstacles:
                adjusted_op = self._adjust_movement_for_obstacles(
                    op,
                    planning_context.obstacles,
                    platform_type
                )
            
            # 根据电池状态调整
            if planning_context.battery_level < 30:
                # 低电量时优先返航
                if op.name not in ["return_to_home", "land"]:
                    adjusted_op.metadata["low_battery_warning"] = True
            
            # 根据天气调整
            if planning_context.weather.get("wind_speed", 0) > 5:
                if op.name == "hover":
                    # 大风时增加悬停稳定时间
                    adjusted_op.parameters["duration"] = op.parameters.get("duration", 5) * 1.5
            
            adjusted_ops.append(adjusted_op)
        
        # 如果有需要避开的障碍物，插入避障操作
        critical_obstacles = [
            obs for obs in planning_context.obstacles 
            if obs.get("distance", float("inf")) < 10
        ]
        
        if critical_obstacles:
            # 在移动操作前插入障碍物检测
            new_ops = []
            for op in adjusted_ops:
                if op.type == OperationType.MOVEMENT and op.name != "land":
                    detect_op = self._create_operation(
                        "detect_objects",
                        {"object_types": ["obstacle"], "area": "front"},
                        platform_type,
                        TaskNode(id="auto", name="auto_detect", task_type="detect", parameters={})
                    )
                    if detect_op:
                        new_ops.append(detect_op)
                new_ops.append(op)
            adjusted_ops = new_ops
        
        return adjusted_ops
    
    def _adjust_movement_for_obstacles(
        self,
        operation: Operation,
        obstacles: List[Dict[str, Any]],
        platform_type: str
    ) -> Operation:
        """根据障碍物调整移动操作"""
        adjusted = operation
        
        # 检查目标路径上是否有障碍物
        target_pos = operation.parameters.get("position", {})
        
        for obstacle in obstacles:
            obs_pos = obstacle.get("position", {})
            obs_distance = obstacle.get("distance", float("inf"))
            
            # 如果障碍物在路径上且距离较近
            if obs_distance < 15:
                # 添加避障信息
                adjusted.metadata["obstacles_on_path"] = adjusted.metadata.get("obstacles_on_path", [])
                adjusted.metadata["obstacles_on_path"].append({
                    "id": obstacle.get("id"),
                    "distance": obs_distance,
                    "direction": obstacle.get("direction")
                })
                
                # 降低速度
                current_speed = adjusted.parameters.get("speed", 5.0)
                adjusted.parameters["speed"] = min(current_speed, 3.0)
                
                # 标记需要避障
                adjusted.metadata["requires_avoidance"] = True
        
        return adjusted
    
    def _apply_cot_suggestions(
        self,
        operations: List[Operation],
        cot_result: 'ReasoningResult'
    ) -> List[Operation]:
        """根据CoT推理结果应用建议"""
        # 解析CoT建议
        suggestion = cot_result.suggestion.lower()
        
        adjusted_ops = list(operations)
        
        # 根据建议调整操作
        if "安全" in suggestion or "谨慎" in suggestion:
            # 增加检查操作
            for i, op in enumerate(adjusted_ops):
                if op.type == OperationType.MOVEMENT:
                    op.parameters["speed"] = op.parameters.get("speed", 5) * 0.8
        
        if "快速" in suggestion or "高效" in suggestion:
            # 优化路径，减少等待
            adjusted_ops = [op for op in adjusted_ops if op.name != "wait"]
        
        # 记录CoT影响
        for op in adjusted_ops:
            op.metadata["cot_applied"] = True
            op.metadata["cot_confidence"] = cot_result.confidence
        
        return adjusted_ops
    
    def _insert_perception_updates(
        self,
        operations: List[Operation],
        platform_type: str
    ) -> List[Operation]:
        """在关键点插入感知更新操作"""
        result = []
        perception_interval = self.config.get("perception_update_interval", 3)
        
        for i, op in enumerate(operations):
            result.append(op)
            
            # 每隔几个操作插入感知更新
            if (i + 1) % perception_interval == 0 and op.name not in ["update_perception", "detect_objects"]:
                update_op = self._create_operation(
                    "update_perception",
                    {},
                    platform_type,
                    TaskNode(id="auto", name="perception_update", task_type="perception", parameters={})
                )
                if update_op:
                    update_op.metadata["auto_inserted"] = True
                    result.append(update_op)
        
        return result
    
    def _build_task_tree(self, parsed_task: Dict[str, Any]) -> TaskNode:
        """构建任务树"""
        task_type = parsed_task.get("task_type", "custom")
        
        root = TaskNode(
            id=str(uuid.uuid4())[:8],
            name=parsed_task.get("name", task_type),
            task_type=task_type,
            parameters=parsed_task.get("parameters", {}),
            priority=parsed_task.get("priority", 1)
        )
        
        # 添加子任务
        subtasks = parsed_task.get("subtasks", [])
        for subtask in subtasks:
            child = self._build_task_tree(subtask)
            child.dependencies = subtask.get("dependencies", [])
            root.children.append(child)
        
        return root
    
    async def _decompose_task_tree(
        self, 
        task_node: TaskNode,
        platform_type: str
    ) -> List[Operation]:
        """递归分解任务树"""
        operations = []
        
        if task_node.children:
            # 有子任务，递归分解
            for child in task_node.children:
                child_ops = await self._decompose_task_tree(child, platform_type)
                operations.extend(child_ops)
        else:
            # 叶子节点，转换为原子操作
            ops = self._task_to_operations(task_node, platform_type)
            operations.extend(ops)
        
        return operations
    
    def _task_to_operations(
        self, 
        task: TaskNode,
        platform_type: str
    ) -> List[Operation]:
        """将任务节点转换为操作序列"""
        operations = []
        
        # 检查是否有预定义的分解规则
        if task.task_type in self.decomposition_rules:
            rule = self.decomposition_rules[task.task_type]
            for op_name in rule:
                # 处理带参数的操作名 (如 "goto:pickup")
                if ":" in op_name:
                    base_name, param_key = op_name.split(":")
                    params = task.parameters.get(param_key, {})
                else:
                    base_name = op_name
                    params = task.parameters.get(base_name, {})
                
                op = self._create_operation(base_name, params, platform_type, task)
                if op:
                    operations.append(op)
        else:
            # 直接作为单个操作
            op = self._create_operation(
                task.task_type, 
                task.parameters, 
                platform_type, 
                task
            )
            if op:
                operations.append(op)
        
        return operations
    
    def _create_operation(
        self,
        op_name: str,
        params: Dict[str, Any],
        platform_type: str,
        source_task: TaskNode
    ) -> Optional[Operation]:
        """创建单个操作"""
        template = self.operation_templates.get(op_name)
        
        if not template:
            logger.warning(f"未知操作类型: {op_name}")
            return None
        
        # 检查平台兼容性
        supported_platforms = template.get("platform", ["drone", "ugv", "usv"])
        if platform_type not in supported_platforms:
            logger.warning(f"操作 {op_name} 不支持平台 {platform_type}")
            return None
        
        # 创建操作
        operation = Operation(
            id=str(uuid.uuid4())[:8],
            name=op_name,
            type=template["type"],
            platform=platform_type,
            parameters=params,
            estimated_duration=template.get("default_duration", 5.0),
            priority=OperationPriority.NORMAL,
            source_task_id=source_task.id,
            metadata={
                "source_task": source_task.name,
                "template": op_name
            }
        )
        
        return operation
    
    def _apply_constraints(
        self, 
        operations: List[Operation],
        constraints: Dict[str, Any]
    ) -> List[Operation]:
        """应用约束条件"""
        constrained_ops = []
        
        for op in operations:
            # 速度约束
            max_speed = constraints.get("max_speed")
            if max_speed and "speed" in op.parameters:
                op.parameters["speed"] = min(op.parameters["speed"], max_speed)
            
            # 安全距离约束
            safe_distance = constraints.get("safe_distance")
            if safe_distance:
                op.metadata["safe_distance"] = safe_distance
            
            # 地理围栏约束
            geofence = constraints.get("geofence", {})
            if geofence.get("enabled") and "position" in op.parameters:
                op.metadata["geofence_check"] = True
            
            # 禁飞区检查
            no_fly_zones = constraints.get("no_fly_zones", [])
            if no_fly_zones and "position" in op.parameters:
                op.metadata["no_fly_zones"] = no_fly_zones
            
            constrained_ops.append(op)
        
        return constrained_ops
    
    def _optimize_sequence(self, operations: List[Operation]) -> List[Operation]:
        """优化操作序列"""
        if not operations:
            return operations
        
        optimized = []
        
        # 合并连续的同类操作
        i = 0
        while i < len(operations):
            current = operations[i]
            
            # 检查是否可以与下一个操作合并
            if i + 1 < len(operations):
                next_op = operations[i + 1]
                
                # 合并连续的goto操作为路径
                if current.name == "goto" and next_op.name == "goto":
                    waypoints = [current.parameters.get("position")]
                    j = i + 1
                    while j < len(operations) and operations[j].name == "goto":
                        waypoints.append(operations[j].parameters.get("position"))
                        j += 1
                    
                    if len(waypoints) > 2:
                        # 合并为路径跟踪
                        path_op = Operation(
                            id=str(uuid.uuid4())[:8],
                            name="follow_path",
                            type=OperationType.MOVEMENT,
                            platform=current.platform,
                            parameters={
                                "waypoints": waypoints,
                                "speed": current.parameters.get("speed")
                            },
                            estimated_duration=sum(
                                op.estimated_duration 
                                for op in operations[i:j]
                            )
                        )
                        optimized.append(path_op)
                        i = j
                        continue
            
            optimized.append(current)
            i += 1
        
        return optimized
    
    def _add_conditions(self, operations: List[Operation]) -> List[Operation]:
        """添加前置和后置条件"""
        for i, op in enumerate(operations):
            # 根据操作类型添加条件
            if op.name == "takeoff":
                op.preconditions.append(
                    Precondition(
                        name="on_ground",
                        condition="robot.state.on_ground == True",
                        description="机器人必须在地面上"
                    )
                )
                op.preconditions.append(
                    Precondition(
                        name="battery_sufficient",
                        condition="robot.battery > 20",
                        description="电池电量必须大于20%"
                    )
                )
                op.postconditions.append(
                    Postcondition(
                        name="airborne",
                        expected_state="robot.state.airborne == True",
                        description="机器人应该在空中"
                    )
                )
            
            elif op.name == "land":
                op.preconditions.append(
                    Precondition(
                        name="airborne",
                        condition="robot.state.airborne == True",
                        description="机器人必须在空中"
                    )
                )
                op.postconditions.append(
                    Postcondition(
                        name="on_ground",
                        expected_state="robot.state.on_ground == True",
                        description="机器人应该在地面上"
                    )
                )
            
            elif op.name == "goto":
                op.preconditions.append(
                    Precondition(
                        name="ready_to_move",
                        condition="robot.state.ready == True",
                        description="机器人必须准备就绪"
                    )
                )
                if op.parameters.get("position"):
                    target = op.parameters["position"]
                    op.postconditions.append(
                        Postcondition(
                            name="at_target",
                            expected_state=f"robot.position.near({target})",
                            description=f"机器人应该在目标位置附近"
                        )
                    )
            
            # 添加回滚操作
            op.rollback_action = self._create_rollback_action(op)
        
        return operations
    
    def _create_rollback_action(self, operation: Operation) -> Optional[Operation]:
        """创建回滚操作"""
        rollback_map = {
            "takeoff": "land",
            "goto": "return_to_home",
            "pickup": "dropoff",
            "start_recording": "stop_recording"
        }
        
        rollback_name = rollback_map.get(operation.name)
        if rollback_name:
            return Operation(
                id=str(uuid.uuid4())[:8],
                name=rollback_name,
                type=operation.type,
                platform=operation.platform,
                parameters={},
                is_rollback=True,
                metadata={"rollback_for": operation.id}
            )
        
        return None
    
    async def _validate_plan(
        self, 
        operations: List[Operation],
        platform_type: str
    ) -> 'PlanValidation':
        """验证计划"""
        issues = []
        
        # 检查操作序列的逻辑一致性
        has_takeoff = False
        has_land = False
        
        for i, op in enumerate(operations):
            # 无人机特定检查
            if platform_type == "drone":
                if op.name == "takeoff":
                    has_takeoff = True
                if op.name == "land":
                    has_land = True
                
                # goto操作必须在takeoff之后
                if op.name == "goto" and not has_takeoff:
                    issues.append(f"操作 {i}: goto 必须在 takeoff 之后")
            
            # 检查参数完整性
            template = self.operation_templates.get(op.name)
            if template:
                for param in template.get("params", []):
                    if param not in op.parameters and param not in ["duration"]:
                        issues.append(f"操作 {i} ({op.name}): 缺少参数 {param}")
        
        # 无人机必须降落
        if platform_type == "drone" and has_takeoff and not has_land:
            issues.append("无人机任务缺少 land 操作")
        
        return PlanValidation(
            valid=len(issues) == 0,
            issues=issues
        )
    
    async def _fix_plan(
        self, 
        operations: List[Operation],
        issues: List[str]
    ) -> List[Operation]:
        """修复计划问题"""
        fixed = list(operations)
        
        for issue in issues:
            if "缺少 land 操作" in issue:
                # 添加降落操作
                land_op = self._create_operation(
                    "land",
                    {},
                    fixed[-1].platform if fixed else "drone",
                    TaskNode(id="auto", name="auto_land", task_type="land", parameters={})
                )
                if land_op:
                    fixed.append(land_op)
        
        return fixed
    
    def estimate_total_time(self, operations: List[Operation]) -> float:
        """估算总执行时间"""
        return sum(op.estimated_duration for op in operations)
    
    def find_parallel_groups(
        self, 
        operations: List[Operation]
    ) -> List[List[str]]:
        """识别可并行执行的操作组"""
        groups = []
        current_group = []
        
        for op in operations:
            # 移动操作通常不能并行
            if op.type == OperationType.MOVEMENT:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([op.id])
            else:
                current_group.append(op.id)
        
        if current_group:
            groups.append(current_group)
        
        return groups


@dataclass
class PlanValidation:
    """计划验证结果"""
    valid: bool
    issues: List[str] = field(default_factory=list)
