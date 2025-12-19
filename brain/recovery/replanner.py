"""
重规划器 - Replanner

负责:
- 感知驱动的动态重规划
- 环境变化响应式规划调整
- 基于LLM和CoT的智能规划
- 保持任务目标不变的前提下重新生成操作序列
"""

import json
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from brain.models.llm_interface import LLMInterface, LLMMessage
from brain.models.prompt_templates import PromptTemplates
from brain.execution.operations.base import Operation
from brain.execution.operations.drone import DroneOperations
from brain.execution.operations.ugv import UGVOperations
from brain.execution.operations.usv import USVOperations
from brain.state.world_state import WorldState

if TYPE_CHECKING:
    from brain.cognitive.world_model import PlanningContext, EnvironmentChange
    from brain.cognitive.reasoning.cot_engine import ReasoningResult


@dataclass
class ReplanResult:
    """重规划结果"""
    success: bool
    new_operations: List[Operation]
    strategy: str
    reason: str
    risk_assessment: str = ""
    alternative_plan: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 新增：感知相关信息
    perception_triggered: bool = False
    changes_addressed: List[str] = field(default_factory=list)
    cot_reasoning: Optional[Dict[str, Any]] = None


class Replanner:
    """
    重规划器
    
    在任务执行过程中，根据感知变化或执行失败动态调整计划
    """
    
    def __init__(
        self,
        planner: 'TaskPlanner',
        llm: LLMInterface,
        config: Optional[Dict[str, Any]] = None
    ):
        self.planner = planner
        self.llm = llm
        self.config = config or {}
        
        self.template = PromptTemplates.get_template("replanning")
        
        # 重规划历史
        self.replan_history: List[Dict[str, Any]] = []
        
        logger.info("Replanner 初始化完成 (感知驱动模式)")
    
    async def replan_with_perception(
        self,
        original_command: str,
        completed_operations: List[Operation],
        changes: List['EnvironmentChange'],
        planning_context: 'PlanningContext',
        cot_reasoning: Optional['ReasoningResult'] = None,
        platform_type: str = "drone"
    ) -> List[Operation]:
        """
        感知驱动的重规划
        
        根据环境变化和感知上下文重新生成操作序列
        
        Args:
            original_command: 原始任务指令
            completed_operations: 已完成的操作列表
            changes: 触发重规划的环境变化
            planning_context: 当前感知上下文
            cot_reasoning: CoT推理结果
            platform_type: 平台类型
            
        Returns:
            List[Operation]: 新的操作序列
        """
        logger.info(f"感知驱动重规划: 变化数={len(changes)}, 已完成操作数={len(completed_operations)}")
        
        # 构建变化描述
        changes_description = self._describe_changes(changes)
        
        # 构建感知上下文
        perception_context = planning_context.to_prompt_context()
        
        # 1. 尝试使用LLM进行智能重规划
        try:
            llm_result = await self._llm_perception_replan(
                original_command=original_command,
                completed_operations=completed_operations,
                changes_description=changes_description,
                perception_context=perception_context,
                cot_reasoning=cot_reasoning,
                platform_type=platform_type
            )
            
            if llm_result.success:
                self._record_replan(
                    method="llm_perception",
                    original_command=original_command,
                    completed_ops=completed_operations,
                    trigger="perception_change",
                    new_ops=llm_result.new_operations,
                    changes=[c.description for c in changes]
                )
                return llm_result.new_operations
                
        except Exception as e:
            logger.warning(f"LLM感知重规划失败: {e}, 使用规则重规划")
        
        # 2. 回退到基于规则的重规划
        rule_result = await self._rule_based_perception_replan(
            original_command=original_command,
            completed_operations=completed_operations,
            changes=changes,
            planning_context=planning_context,
            platform_type=platform_type
        )
        
        self._record_replan(
            method="rule_based_perception",
            original_command=original_command,
            completed_ops=completed_operations,
            trigger="perception_change",
            new_ops=rule_result.new_operations,
            changes=[c.description for c in changes]
        )
        
        return rule_result.new_operations
    
    async def replan(
        self,
        original_command: str,
        completed_operations: List[Operation],
        failed_operation: Operation,
        error: str,
        environment_state: Dict[str, Any],
        world_state: WorldState
    ) -> List[Operation]:
        """
        执行重规划（失败触发版本）
        
        Args:
            original_command: 原始任务指令
            completed_operations: 已完成的操作列表
            failed_operation: 失败的操作
            error: 错误信息
            environment_state: 当前环境状态
            world_state: 当前世界状态
            
        Returns:
            List[Operation]: 新的操作序列
        """
        logger.info(f"开始重规划: 原始指令='{original_command[:50]}...', 失败操作={failed_operation.name}")
        
        # 1. 尝试使用LLM进行智能重规划
        try:
            llm_result = await self._llm_replan(
                original_command=original_command,
                completed_operations=completed_operations,
                failed_operation=failed_operation,
                error=error,
                environment_state=environment_state,
                world_state=world_state
            )
            
            if llm_result.success:
                self._record_replan(
                    "llm",
                    original_command,
                    completed_operations,
                    "failure",
                    llm_result.new_operations
                )
                return llm_result.new_operations
                
        except Exception as e:
            logger.warning(f"LLM重规划失败: {e}, 使用规则重规划")
        
        # 2. 回退到基于规则的重规划
        rule_result = await self._rule_based_replan(
            original_command=original_command,
            completed_operations=completed_operations,
            failed_operation=failed_operation,
            error=error,
            world_state=world_state
        )
        
        self._record_replan(
            "rule_based",
            original_command,
            completed_operations,
            "failure",
            rule_result.new_operations
        )
        
        return rule_result.new_operations
    
    def _describe_changes(self, changes: List['EnvironmentChange']) -> str:
        """描述环境变化"""
        descriptions = []
        for change in changes:
            priority = change.priority.value
            desc = f"[{priority}] {change.description}"
            if change.data:
                details = ", ".join([f"{k}={v}" for k, v in list(change.data.items())[:3]])
                desc += f" ({details})"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    async def _llm_perception_replan(
        self,
        original_command: str,
        completed_operations: List[Operation],
        changes_description: str,
        perception_context: str,
        cot_reasoning: Optional['ReasoningResult'],
        platform_type: str
    ) -> ReplanResult:
        """使用LLM进行感知驱动重规划"""
        # 格式化已完成操作
        completed_ops_str = json.dumps(
            [
                {
                    "name": op.name,
                    "type": op.type.value,
                    "parameters": op.parameters
                }
                for op in completed_operations
            ],
            ensure_ascii=False,
            indent=2
        )
        
        # 构建CoT提示
        cot_context = ""
        if cot_reasoning:
            cot_context = f"""
## CoT推理结果
决策: {cot_reasoning.decision}
建议: {cot_reasoning.suggestion}
置信度: {cot_reasoning.confidence:.2f}
"""
        
        # 构建提示词
        user_prompt = f"""## 原始任务
{original_command}

## 已完成的操作
{completed_ops_str}

## 环境变化（触发重规划的原因）
{changes_description}

## 当前感知状态
{perception_context}
{cot_context}
## 要求
请根据环境变化调整剩余计划，确保：
1. 保持原任务目标不变
2. 适应新的环境条件
3. 规避新发现的障碍/风险
4. 生成可执行的操作序列

请输出JSON格式的新计划：
```json
{{
    "strategy": "规划策略描述",
    "replan_reason": "重规划原因",
    "operations": [
        {{"name": "操作名", "type": "操作类型", "parameters": {{}}}}
    ],
    "risk_assessment": "风险评估",
    "changes_addressed": ["处理的变化1", "处理的变化2"]
}}
```"""
        
        # 调用LLM
        messages = [
            LLMMessage(
                role="system",
                content="你是一个无人系统的智能重规划器。根据环境感知变化，动态调整任务计划以确保安全高效地完成任务。"
            ),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        response = await self.llm.chat(messages)
        
        # 解析响应
        result = self._parse_perception_replan_response(response.content, platform_type)
        result.perception_triggered = True
        
        return result
    
    def _parse_perception_replan_response(
        self, 
        content: str,
        platform: str
    ) -> ReplanResult:
        """解析感知重规划响应"""
        try:
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if not json_match:
                json_match = re.search(r'\{[\s\S]*\}', content)
            
            if not json_match:
                raise ValueError("无法找到JSON内容")
            
            json_str = json_match.group(1) if '```' in content else json_match.group()
            data = json.loads(json_str)
            
            # 转换操作列表
            operations = []
            for op_data in data.get("operations", []):
                op = Operation.from_dict({
                    **op_data,
                    "platform": platform
                })
                operations.append(op)
            
            return ReplanResult(
                success=len(operations) > 0,
                new_operations=operations,
                strategy=data.get("strategy", "perception_adaptive"),
                reason=data.get("replan_reason", "环境变化适应"),
                risk_assessment=data.get("risk_assessment", ""),
                changes_addressed=data.get("changes_addressed", []),
                perception_triggered=True
            )
            
        except Exception as e:
            logger.error(f"解析感知重规划响应失败: {e}")
            return ReplanResult(
                success=False,
                new_operations=[],
                strategy="failed",
                reason=str(e)
            )
    
    async def _rule_based_perception_replan(
        self,
        original_command: str,
        completed_operations: List[Operation],
        changes: List['EnvironmentChange'],
        planning_context: 'PlanningContext',
        platform_type: str
    ) -> ReplanResult:
        """基于规则的感知重规划"""
        new_operations = []
        strategy = "rule_based_perception"
        reason = "基于感知变化的规则重规划"
        changes_addressed = []
        
        from brain.cognitive.world_model import ChangeType
        
        for change in changes:
            if change.change_type == ChangeType.NEW_OBSTACLE:
                # 新障碍物：添加避障操作
                avoid_op = self._create_avoidance_operation(
                    change.data,
                    platform_type
                )
                if avoid_op:
                    new_operations.append(avoid_op)
                    changes_addressed.append(f"避开新障碍物")
            
            elif change.change_type == ChangeType.PATH_BLOCKED:
                # 路径阻塞：重新规划路径
                reroute_ops = self._create_reroute_operations(
                    change.data,
                    planning_context,
                    platform_type
                )
                new_operations.extend(reroute_ops)
                changes_addressed.append(f"绕过阻塞路径")
            
            elif change.change_type == ChangeType.TARGET_MOVED:
                # 目标移动：更新目标位置
                update_op = self._create_target_update_operation(
                    change.data,
                    platform_type
                )
                if update_op:
                    new_operations.append(update_op)
                    changes_addressed.append(f"追踪移动目标")
            
            elif change.change_type == ChangeType.BATTERY_LOW:
                # 低电量：优先返航
                return_ops = self._create_emergency_return_operations(
                    platform_type
                )
                new_operations.extend(return_ops)
                changes_addressed.append(f"低电量返航")
                strategy = "emergency_return"
                reason = "电池电量不足，执行紧急返航"
                break  # 低电量优先级最高
            
            elif change.change_type == ChangeType.WEATHER_CHANGED:
                # 天气变化：调整飞行参数
                adjust_ops = self._create_weather_adjustment_operations(
                    change.data,
                    platform_type
                )
                new_operations.extend(adjust_ops)
                changes_addressed.append(f"适应天气变化")
        
        # 如果没有特殊处理，继续原任务
        if not new_operations:
            # 简单地重新创建剩余操作
            new_operations = await self._regenerate_remaining_operations(
                original_command,
                completed_operations,
                platform_type
            )
            strategy = "continue_with_updates"
            reason = "环境变化不影响核心计划，继续执行"
        
        return ReplanResult(
            success=len(new_operations) > 0,
            new_operations=new_operations,
            strategy=strategy,
            reason=reason,
            changes_addressed=changes_addressed,
            perception_triggered=True
        )
    
    def _create_avoidance_operation(
        self,
        obstacle_data: Dict[str, Any],
        platform_type: str
    ) -> Optional[Operation]:
        """创建避障操作"""
        return Operation.from_dict({
            "name": "avoid_obstacle",
            "type": "movement",
            "platform": platform_type,
            "parameters": {
                "obstacle_id": obstacle_data.get("object_id"),
                "obstacle_position": obstacle_data.get("position"),
                "avoidance_strategy": "detour"  # 或 "altitude_change"
            }
        })
    
    def _create_reroute_operations(
        self,
        blockage_data: Dict[str, Any],
        planning_context: 'PlanningContext',
        platform_type: str
    ) -> List[Operation]:
        """创建绕路操作"""
        operations = []
        
        # 停止当前移动
        operations.append(Operation.from_dict({
            "name": "hover" if platform_type == "drone" else "wait",
            "type": "movement" if platform_type == "drone" else "control",
            "platform": platform_type,
            "parameters": {"duration": 2.0}
        }))
        
        # 检测替代路径
        operations.append(Operation.from_dict({
            "name": "detect_objects",
            "type": "perception",
            "platform": platform_type,
            "parameters": {"object_types": ["obstacle", "path"], "area": "surroundings"}
        }))
        
        # 如果是无人机，可以提高高度
        if platform_type == "drone":
            operations.append(Operation.from_dict({
                "name": "goto",
                "type": "movement",
                "platform": platform_type,
                "parameters": {
                    "position": {
                        "x": planning_context.current_position.get("x", 0),
                        "y": planning_context.current_position.get("y", 0),
                        "z": planning_context.current_position.get("z", 10) + 10  # 升高10米
                    },
                    "speed": 2.0
                }
            }))
        
        return operations
    
    def _create_target_update_operation(
        self,
        target_data: Dict[str, Any],
        platform_type: str
    ) -> Optional[Operation]:
        """创建目标追踪更新操作"""
        new_position = target_data.get("new_position")
        if not new_position:
            return None
        
        return Operation.from_dict({
            "name": "goto",
            "type": "movement",
            "platform": platform_type,
            "parameters": {
                "position": new_position,
                "speed": 5.0,  # 追踪时适当加速
                "tracking_mode": True
            }
        })
    
    def _create_emergency_return_operations(
        self,
        platform_type: str
    ) -> List[Operation]:
        """创建紧急返航操作"""
        operations = []
        
        # 广播紧急状态
        operations.append(Operation.from_dict({
            "name": "broadcast_status",
            "type": "communication",
            "platform": platform_type,
            "parameters": {"status": "emergency_return", "reason": "low_battery"}
        }))
        
        # 返航
        operations.append(Operation.from_dict({
            "name": "return_to_home",
            "type": "movement",
            "platform": platform_type,
            "parameters": {"emergency": True}
        }))
        
        # 降落
        if platform_type == "drone":
            operations.append(Operation.from_dict({
                "name": "land",
                "type": "movement",
                "platform": platform_type,
                "parameters": {}
            }))
        
        return operations
    
    def _create_weather_adjustment_operations(
        self,
        weather_data: Dict[str, Any],
        platform_type: str
    ) -> List[Operation]:
        """创建天气适应操作"""
        operations = []
        
        condition = weather_data.get("new", "unknown")
        
        if condition in ["heavy_rain", "storm"]:
            # 恶劣天气，降低高度并减速
            operations.append(Operation.from_dict({
                "name": "broadcast_status",
                "type": "communication",
                "platform": platform_type,
                "parameters": {"status": "weather_adaptation", "condition": condition}
            }))
        
        elif condition == "strong_wind":
            # 大风，增加稳定时间
            operations.append(Operation.from_dict({
                "name": "hover" if platform_type == "drone" else "wait",
                "type": "movement" if platform_type == "drone" else "control",
                "platform": platform_type,
                "parameters": {"duration": 5.0, "stabilize": True}
            }))
        
        return operations
    
    async def _regenerate_remaining_operations(
        self,
        original_command: str,
        completed_operations: List[Operation],
        platform_type: str
    ) -> List[Operation]:
        """重新生成剩余操作"""
        # 简化实现：返回基本的继续操作
        operations = []
        
        # 更新感知
        operations.append(Operation.from_dict({
            "name": "update_perception",
            "type": "perception",
            "platform": platform_type,
            "parameters": {}
        }))
        
        # 继续向目标移动（这里简化处理）
        operations.append(Operation.from_dict({
            "name": "check_status",
            "type": "perception",
            "platform": platform_type,
            "parameters": {}
        }))
        
        return operations
    
    async def _llm_replan(
        self,
        original_command: str,
        completed_operations: List[Operation],
        failed_operation: Operation,
        error: str,
        environment_state: Dict[str, Any],
        world_state: WorldState
    ) -> ReplanResult:
        """使用LLM进行重规划（失败触发版本）"""
        # 格式化已完成操作
        completed_ops_str = json.dumps(
            [
                {
                    "name": op.name,
                    "type": op.type.value,
                    "parameters": op.parameters,
                    "result": op.result.to_dict() if op.result else None
                }
                for op in completed_operations
            ],
            ensure_ascii=False,
            indent=2
        )
        
        # 格式化失败操作
        failed_op_str = json.dumps(
            {
                "name": failed_operation.name,
                "type": failed_operation.type.value,
                "parameters": failed_operation.parameters
            },
            ensure_ascii=False,
            indent=2
        )
        
        # 构建提示词
        user_prompt = self.template.format(
            original_command=original_command,
            completed_operations=completed_ops_str,
            failed_operation=failed_op_str,
            error=error,
            environment_state=json.dumps(environment_state, ensure_ascii=False, indent=2),
            world_state=json.dumps(world_state.to_dict(), ensure_ascii=False, indent=2)
        )
        
        # 调用LLM
        messages = [
            LLMMessage(role="system", content=self.template.system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        response = await self.llm.chat(messages)
        
        # 解析响应
        result = self._parse_llm_response(response.content, failed_operation.platform)
        
        return result
    
    def _parse_llm_response(
        self, 
        content: str,
        platform: str
    ) -> ReplanResult:
        """解析LLM响应"""
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError("无法找到JSON内容")
            
            data = json.loads(json_match.group())
            
            # 转换操作列表
            operations = []
            for op_data in data.get("operations", []):
                op = Operation.from_dict({
                    **op_data,
                    "platform": platform
                })
                operations.append(op)
            
            return ReplanResult(
                success=len(operations) > 0,
                new_operations=operations,
                strategy=data.get("strategy", "unknown"),
                reason=data.get("replan_reason", ""),
                risk_assessment=data.get("risk_assessment", ""),
                alternative_plan=data.get("alternative_plan")
            )
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
            return ReplanResult(
                success=False,
                new_operations=[],
                strategy="failed",
                reason=str(e)
            )
    
    async def _rule_based_replan(
        self,
        original_command: str,
        completed_operations: List[Operation],
        failed_operation: Operation,
        error: str,
        world_state: WorldState
    ) -> ReplanResult:
        """基于规则的重规划"""
        new_operations = []
        strategy = "rule_based"
        reason = f"因{failed_operation.name}失败而触发规则重规划"
        
        from brain.execution.operations.base import OperationType
        
        if failed_operation.type == OperationType.MOVEMENT:
            new_operations = self._replan_movement(
                failed_operation,
                completed_operations,
                world_state
            )
            strategy = "alternative_path"
            reason = "移动操作失败，尝试替代路径"
            
        elif failed_operation.type == OperationType.PERCEPTION:
            new_operations = self._replan_perception(
                failed_operation,
                completed_operations
            )
            strategy = "skip_or_retry"
            reason = "感知操作失败，调整执行策略"
            
        elif failed_operation.type == OperationType.MANIPULATION:
            new_operations = self._replan_manipulation(
                failed_operation,
                completed_operations,
                world_state
            )
            strategy = "retry_with_preparation"
            reason = "操作失败，添加准备步骤后重试"
            
        else:
            new_operations = [failed_operation.clone()]
            strategy = "simple_retry"
            reason = "尝试重新执行失败的操作"
        
        return ReplanResult(
            success=len(new_operations) > 0,
            new_operations=new_operations,
            strategy=strategy,
            reason=reason
        )
    
    def _replan_movement(
        self,
        failed_op: Operation,
        completed_ops: List[Operation],
        world_state: WorldState
    ) -> List[Operation]:
        """重规划移动操作"""
        operations = []
        
        current_pos = world_state.get("robot.position", {})
        target_pos = failed_op.parameters.get("position", {})
        
        if failed_op.name == "goto":
            
            platform = failed_op.platform
            
            if platform == "drone":
                ops_class = DroneOperations
                hover_op = ops_class.hover(duration=2)
                operations.append(hover_op)
                
                new_goto = ops_class.goto(
                    position=target_pos,
                    speed=failed_op.parameters.get("speed", 3.0) * 0.7
                )
                operations.append(new_goto)
                
            elif platform == "ugv":
                ops_class = UGVOperations
                reverse_op = ops_class.reverse(distance=2.0)
                operations.append(reverse_op)
                
                new_goto = ops_class.goto(
                    position=target_pos,
                    path_type="safest"
                )
                operations.append(new_goto)
                
            elif platform == "usv":
                ops_class = USVOperations
                hold_op = ops_class.hold_position(duration=5)
                operations.append(hold_op)
                
                new_nav = ops_class.navigate_to(
                    position=target_pos,
                    speed=failed_op.parameters.get("speed", 3.0) * 0.7
                )
                operations.append(new_nav)
        
        elif failed_op.name in ["takeoff", "land"]:
            check_op = Operation.from_dict({
                "name": "check_status",
                "type": "perception",
                "platform": failed_op.platform,
                "parameters": {}
            })
            operations.append(check_op)
            operations.append(failed_op.clone())
        
        else:
            operations.append(failed_op.clone())
        
        return operations
    
    def _replan_perception(
        self,
        failed_op: Operation,
        completed_ops: List[Operation]
    ) -> List[Operation]:
        """重规划感知操作"""
        operations = []
        
        wait_op = Operation.from_dict({
            "name": "wait",
            "type": "control",
            "platform": failed_op.platform,
            "parameters": {"duration": 2.0}
        })
        operations.append(wait_op)
        operations.append(failed_op.clone())
        
        return operations
    
    def _replan_manipulation(
        self,
        failed_op: Operation,
        completed_ops: List[Operation],
        world_state: WorldState
    ) -> List[Operation]:
        """重规划操作类任务"""
        operations = []
        
        if failed_op.name == "pickup":
            hover_op = Operation.from_dict({
                "name": "hover",
                "type": "movement",
                "platform": failed_op.platform,
                "parameters": {"duration": 3.0}
            })
            operations.append(hover_op)
        
        elif failed_op.name == "dropoff":
            scan_op = Operation.from_dict({
                "name": "scan_area",
                "type": "perception",
                "platform": failed_op.platform,
                "parameters": {"resolution": "high"}
            })
            operations.append(scan_op)
        
        operations.append(failed_op.clone())
        
        return operations
    
    def _record_replan(
        self,
        method: str,
        original_command: str,
        completed_ops: List[Operation],
        trigger: str,
        new_ops: List[Operation],
        changes: Optional[List[str]] = None
    ):
        """记录重规划"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "trigger": trigger,
            "original_command": original_command[:100],
            "completed_count": len(completed_ops),
            "new_operations_count": len(new_ops),
            "new_operations": [op.name for op in new_ops],
            "changes_addressed": changes or []
        }
        
        self.replan_history.append(record)
        
        max_history = self.config.get("max_history", 50)
        if len(self.replan_history) > max_history:
            self.replan_history = self.replan_history[-max_history:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取重规划统计"""
        method_counts = {}
        trigger_counts = {}
        
        for record in self.replan_history:
            method = record["method"]
            method_counts[method] = method_counts.get(method, 0) + 1
            
            trigger = record.get("trigger", "unknown")
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        return {
            "total_replans": len(self.replan_history),
            "by_method": method_counts,
            "by_trigger": trigger_counts,
            "perception_triggered": sum(1 for r in self.replan_history if r.get("trigger") == "perception_change"),
            "recent": self.replan_history[-5:]
        }


# 需要在文件顶部导入
from brain.planning.task.task_planner import TaskPlanner
