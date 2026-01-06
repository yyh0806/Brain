# -*- coding: utf-8 -*-
"""
Brain - 感知驱动的智能任务规划核心系统

这是整个系统的核心协调器，集成了认知模块，实现:
- 自然语言理解 -> 感知驱动的任务规划
- CoT推理 -> 可追溯的决策过程
- 多轮对话 -> 指令澄清/确认/汇报
- 感知监控 -> 响应式重规划
- 错误恢复 -> 智能回退与重试
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from loguru import logger

from brain.planning.orchestrator import PlanningOrchestrator
from brain.execution.executor import Executor
from brain.core.monitor import SystemMonitor
from brain.perception.sensors.sensor_manager import MultiSensorManager as SensorManager
from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
from brain.communication.ros2_interface import ROS2Interface, ROS2Config
# EnvironmentModel 已删除，功能合并到 WorldModel
from brain.models.llm_interface import LLMInterface
from brain.models.task_parser import TaskParser
from brain.execution.operations.base import Operation, OperationResult, OperationStatus
from brain.recovery.error_handler import ErrorHandler
from brain.recovery.replanner import Replanner
from brain.state.world_state import WorldState
from brain.state.mission_state import MissionState, MissionStatus
from brain.state.checkpoint import CheckpointManager
from brain.communication.robot_interface import RobotInterface
from brain.utils.config import ConfigManager

# 认知模块 - 使用统一接口
from brain.cognitive.interface import CognitiveLayer
from brain.cognitive.world_model import WorldModel, EnvironmentChange, ChangeType
from brain.cognitive.dialogue import DialogueManager, DialogueContext, DialogueType
from brain.cognitive.reasoning import CoTEngine, ReasoningResult, ReasoningMode
from brain.cognitive.monitoring import PerceptionMonitor, MonitorEvent, TriggerAction
from brain.models.cot_prompts import CoTPrompts


class BrainStatus(Enum):
    """Brain系统状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    PLANNING = "planning"
    EXECUTING = "executing"
    REPLANNING = "replanning"  # 新增：重规划状态
    RECOVERING = "recovering"
    PAUSED = "paused"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class Mission:
    """任务定义"""
    id: str
    natural_language_command: str
    platform_type: str  # drone, ugv, usv
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    operations: List[Operation] = field(default_factory=list)
    status: MissionStatus = MissionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 新增：推理链记录
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    # 新增：对话历史
    dialogue_history: List[Dict[str, Any]] = field(default_factory=list)


class Brain:
    """
    Brain - 感知驱动的无人系统任务分解大脑
    
    核心职责:
    1. 自然语言理解 -> 结合感知的任务分解
    2. CoT推理 -> 智能规划与决策
    3. 多轮对话 -> 用户交互
    4. 感知监控 -> 响应式重规划
    5. 错误恢复 -> 智能回退
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.id = str(uuid.uuid4())[:8]
        self.status = BrainStatus.INITIALIZING
        
        # 加载配置
        self.config = ConfigManager(config_path)
        
        # 初始化子系统
        self._init_subsystems()
        
        # 初始化认知模块
        self._init_cognitive_modules()
        
        # 任务队列
        self.missions: Dict[str, Mission] = {}
        self.current_mission: Optional[Mission] = None
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # 异步锁
        self._lock = asyncio.Lock()
        
        # 感知监控任务
        self._perception_monitor_task: Optional[asyncio.Task] = None
        self._running = True
        
        logger.info(f"Brain [{self.id}] 初始化完成 (感知驱动模式)")
        self.status = BrainStatus.READY
    
    def _init_subsystems(self):
        """初始化所有子系统"""
        # 状态管理
        self.world_state = WorldState()
        self.mission_state = MissionState()
        self.checkpoint_manager = CheckpointManager(
            self.config.get("state.checkpoint_path", "./data/checkpoints")
        )
        
        # LLM接口（需要先初始化）
        self.llm = LLMInterface(self.config.get("llm", {}))
        self.task_parser = TaskParser(self.llm)
        
        # 先初始化ROS2接口（同步初始化）
        comm_config = self.config.get("communication", {})
        # 检查是否有ros2_interface配置（用于Isaac Sim等环境）
        ros2_config_dict = self.config.get("ros2_interface", comm_config)
        
        # 解析mode配置
        mode_str = ros2_config_dict.get("mode", "simulation")
        if isinstance(mode_str, str):
            from brain.communication.ros2_interface import ROS2Mode
            if mode_str.lower() == "real":
                mode = ROS2Mode.REAL
            else:
                mode = ROS2Mode.SIMULATION
        else:
            mode = ROS2Mode.SIMULATION
        
        # 创建ROS2Config对象，过滤不支持的参数
        ros2_config = ROS2Config(
            node_name=ros2_config_dict.get("node_name", comm_config.get("node_name", "brain_node")),
            mode=mode,
            topics=ros2_config_dict.get("topics", comm_config.get("topics", {}))
        )
        self.ros2 = ROS2Interface(ros2_config)
        
        # 初始化VLM（如果配置启用）
        vlm = None
        vlm_config = self.config.get("perception", {}).get("vlm", {})
        if vlm_config.get("enabled", True):
            try:
                from brain.perception.vlm.vlm_perception import VLMPerception, OLLAMA_AVAILABLE
                if OLLAMA_AVAILABLE:
                    vlm = VLMPerception(
                        model=vlm_config.get("model", "llava:7b"),
                        ollama_host=vlm_config.get("ollama_host", "http://localhost:11434")
                    )
                    logger.info("VLM已初始化并传入感知层")
                else:
                    logger.warning("Ollama不可用，VLM功能将不可用")
            except Exception as e:
                logger.warning(f"VLM初始化失败: {e}")
        
        # 感知系统 - 使用ROS2SensorManager并传入ROS2接口和VLM
        self.sensor_manager = ROS2SensorManager(
            ros2_interface=self.ros2,
            config=self.config.get("perception", {}),
            vlm=vlm  # 传入VLM
        )
        
        # 规划与执行
        self.planner = PlanningOrchestrator(
            platform=self.config.get("platform", "ugv"),
            config=self.config.get("planning", {})
        )
        self.executor = Executor(
            world_state=self.world_state,
            config=self.config.get("execution", {})
        )
        
        # 错误恢复
        self.error_handler = ErrorHandler(
            config=self.config.get("recovery", {})
        )
        self.replanner = Replanner(
            planner=self.planner,
            llm=self.llm,
            config=self.config.get("recovery", {})
        )
        
        # 系统监控
        self.monitor = SystemMonitor(
            brain=self,
            config=self.config.get("system", {})
        )
        
        # 机器人通信接口
        self.robot_interface = RobotInterface(
            config=self.config.get("communication", {})
        )
    
    def _init_cognitive_modules(self):
        """初始化认知模块"""
        # 感知驱动的世界模型
        self.cognitive_world_model = WorldModel(
            config=self.config.get("cognitive.world_model", {})
        )
        
        # 多轮对话管理器
        self.dialogue = DialogueManager(
            llm_interface=self.llm,
            user_callback=None  # 稍后设置
        )
        
        # CoT推理引擎
        self.cot_prompts = CoTPrompts()
        self.cot_engine = CoTEngine(
            llm_interface=self.llm,
            cot_prompts=self.cot_prompts,
            default_complexity_threshold=self.config.get("cognitive.cot_threshold", 0.5)
        )
        
        # 感知变化监控器
        self.perception_monitor = PerceptionMonitor(
            world_model=self.cognitive_world_model,
            config=self.config.get("cognitive.monitor", {})
        )
        
        # 设置监控回调
        self.perception_monitor.set_replan_callback(self._on_replan_triggered)
        self.perception_monitor.set_confirmation_callback(self._on_confirmation_required)
        self.perception_monitor.set_notification_callback(self._on_notification)
        
        # 创建认知层统一接口
        self.cognitive_layer = CognitiveLayer(
            world_model=self.cognitive_world_model,
            cot_engine=self.cot_engine,
            dialogue_manager=self.dialogue,
            perception_monitor=self.perception_monitor,
            config=self.config.get("cognitive", {})
        )
        
        # 将ROS2SensorManager传入认知层
        self.cognitive_layer._sensor_manager = self.sensor_manager
        
        # 启动感知数据监控和更新任务
        self._perception_update_task = asyncio.create_task(
            self._update_perception_loop()
        )
        
        logger.info("认知模块初始化完成（使用统一接口）")
    
    def set_user_callback(self, callback: Callable[[str, List[str]], Awaitable[str]]):
        """设置用户交互回调"""
        self.dialogue.set_user_callback(callback)
    
    def set_auto_confirm(self, enabled: bool, delay: float = 0.5):
        """设置自动确认模式（用于测试）"""
        self.dialogue.set_auto_confirm(enabled, delay)
    
    async def process_command(
        self, 
        command: str, 
        platform_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Mission:
        """
        处理自然语言指令 - 感知驱动版本
        
        Args:
            command: 自然语言指令
            platform_type: 平台类型 (drone/ugv/usv)
            context: 额外上下文信息
            
        Returns:
            Mission: 创建的任务对象
        """
        async with self._lock:
            logger.info(f"收到指令: {command} (平台: {platform_type})")
            
            # 开始对话会话
            session_id = f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.dialogue.start_session(session_id)
            
            # 创建任务
            mission = Mission(
                id=str(uuid.uuid4())[:8],
                natural_language_command=command,
                platform_type=platform_type,
                metadata=context or {}
            )
            
            self.missions[mission.id] = mission
            self.status = BrainStatus.PLANNING
            
            try:
                # Step 1: 获取感知数据并更新世界模型
                sensor_data = await self.sensor_manager.get_current_data()
                self.cognitive_world_model.update_from_perception(sensor_data)
                
                # Step 2: 获取规划上下文
                planning_context = self.cognitive_world_model.get_context_for_planning()
                
                # Step 3: 检测指令是否模糊，需要澄清
                clarification_result = await self._check_and_clarify(
                    command, platform_type, planning_context
                )
                
                if clarification_result.get("clarified"):
                    command = clarification_result["clarified_command"]
                    mission.dialogue_history.append({
                        "type": "clarification",
                        "original": mission.natural_language_command,
                        "clarified": command
                    })
                
                # Step 4: 使用CoT进行智能规划
                reasoning = await self.cot_engine.reason(
                    query=f"规划任务: {command}",
                    context={
                        "perception": planning_context.to_prompt_context(),
                        "platform": platform_type,
                        "constraints": self._get_safety_constraints(platform_type)
                    },
                    mode=ReasoningMode.PLANNING
                )
                
                # 记录推理链
                mission.reasoning_chain.append(reasoning.to_dict())
                
                logger.info(f"CoT推理完成: 置信度={reasoning.confidence:.2f}")
                
                # Step 5: 使用LLM解析指令
                env_state = planning_context.to_prompt_context()
                
                parsed_task = await self.task_parser.parse(
                    command=command,
                    platform_type=platform_type,
                    environment_state={"context": env_state},
                    world_state=self.world_state.to_dict()
                )
                
                logger.info(f"任务解析结果: {parsed_task}")
                
                # Step 6: 生成操作序列（结合感知上下文）
                operations = await self.planner.plan_with_perception(
                    parsed_task=parsed_task,
                    platform_type=platform_type,
                    planning_context=planning_context,
                    cot_result=reasoning,
                    constraints=self._get_safety_constraints(platform_type)
                )
                
                mission.operations = operations
                mission.status = MissionStatus.PLANNED
                
                # Step 7: 创建检查点
                await self.checkpoint_manager.create_checkpoint(
                    mission_id=mission.id,
                    stage="planned",
                    data={
                        "operations": [op.to_dict() for op in operations],
                        "world_state": self.world_state.to_dict(),
                        "reasoning": reasoning.to_dict()
                    }
                )
                
                # Step 8: 汇报规划结果
                await self.dialogue.send_information(
                    f"✅ 任务规划完成\n"
                    f"- 任务ID: {mission.id}\n"
                    f"- 操作数量: {len(operations)}\n"
                    f"- 预计时长: {self.planner.estimate_total_time(operations):.0f}秒\n"
                    f"- 规划置信度: {reasoning.confidence:.0%}"
                )
                
                logger.info(f"任务 [{mission.id}] 规划完成, 共 {len(operations)} 个操作")
                
                # 触发事件
                await self._emit_event("mission_planned", mission)
                
                return mission
                
            except Exception as e:
                logger.error(f"任务规划失败: {e}")
                mission.status = MissionStatus.FAILED
                mission.metadata["error"] = str(e)
                self.status = BrainStatus.READY
                raise
    
    async def _check_and_clarify(
        self,
        command: str,
        platform_type: str,
        planning_context
    ) -> Dict[str, Any]:
        """检查指令是否需要澄清"""
        # 检测模糊词
        ambiguous_words = ["那边", "那里", "这边", "附近", "差不多", "大概"]
        detected = [word for word in ambiguous_words if word in command]
        
        if not detected:
            return {"clarified": False}
        
        # 构建澄清上下文
        context_str = planning_context.to_prompt_context()
        
        # 检测模糊点
        ambiguities = []
        if any(word in command for word in ["那边", "那里", "这边"]):
            # 根据环境确定可能的目标
            options = []
            for poi in planning_context.points_of_interest[:3]:
                options.append(poi.get("description", poi.get("type", "未知")))
            for obs in planning_context.obstacles[:2]:
                direction = obs.get("direction", "未知方向")
                options.append(f"{direction}的{obs.get('type', '物体')}")
            
            if not options:
                options = ["东边", "西边", "南边", "北边"]
            
            ambiguities.append({
                "aspect": "位置",
                "question": "具体指哪个方向或目标？",
                "options": options
            })
        
        if any(word in command for word in ["附近", "差不多", "大概"]):
            ambiguities.append({
                "aspect": "精度",
                "question": "需要精确位置还是大致范围？",
                "options": ["精确位置", "5米范围内", "10米范围内"]
            })
        
        if ambiguities:
            result = await self.dialogue.clarify_ambiguous_command(
                command=command,
                ambiguities=ambiguities,
                world_context=context_str
            )
            return {
                "clarified": True,
                "clarified_command": result.get("clarified_command", command)
            }
        
        return {"clarified": False}
    
    async def execute_mission_with_perception(
        self, 
        mission_id: str,
        auto_recovery: bool = True
    ) -> MissionStatus:
        """
        感知驱动的任务执行
        
        主循环中持续监控感知变化，必要时触发重规划
        """
        mission = self.missions.get(mission_id)
        if not mission:
            raise ValueError(f"任务 [{mission_id}] 不存在")
        
        if mission.status not in [MissionStatus.PLANNED, MissionStatus.PAUSED]:
            raise ValueError(f"任务 [{mission_id}] 状态不允许执行: {mission.status}")
        
        self.current_mission = mission
        mission.status = MissionStatus.EXECUTING
        self.status = BrainStatus.EXECUTING
        
        # 开始感知监控
        await self.perception_monitor.start_monitoring()
        
        logger.info(f"开始执行任务 [{mission_id}] (感知驱动模式)")
        
        # 汇报开始执行
        await self.dialogue.send_information(
            f"🚀 开始执行任务 [{mission_id}]\n"
            f"总操作数: {len(mission.operations)}"
        )
        
        operation_index = mission.metadata.get("resume_from", 0)
        consecutive_failures = 0
        max_failures = self.config.get("recovery.error_thresholds.critical", 5)
        
        try:
            while operation_index < len(mission.operations):
                operation = mission.operations[operation_index]
                
                # === 感知驱动的核心：每次操作前检查环境变化 ===
                sensor_data = await self.sensor_manager.get_current_data()
                changes = self.cognitive_world_model.update_from_perception(sensor_data)
                
                # 检测显著变化
                significant_changes = self.cognitive_world_model.detect_significant_changes()
                
                if significant_changes:
                    # 使用CoT推理决定如何处理变化
                    replan_decision = await self._evaluate_changes_for_replan(
                        changes=significant_changes,
                        current_operation=operation,
                        remaining_operations=mission.operations[operation_index:],
                        mission=mission
                    )
                    
                    if replan_decision["action"] == "replan":
                        # 执行重规划
                        new_ops = await self._perception_driven_replan(
                            mission=mission,
                            changes=significant_changes,
                            operation_index=operation_index,
                            replan_decision=replan_decision
                        )
                        
                        if new_ops:
                            mission.operations = mission.operations[:operation_index] + new_ops
                            # 不增加operation_index，从当前位置继续
                            continue
                    
                    elif replan_decision["action"] == "pause":
                        mission.status = MissionStatus.PAUSED
                        mission.metadata["resume_from"] = operation_index
                        await self.dialogue.send_information(
                            f"⏸️ 任务暂停: {replan_decision.get('reason', '环境变化')}"
                        )
                        break
                
                # 检查是否需要保存检查点
                if operation_index % self.config.get("planning.checkpoint_interval", 5) == 0:
                    await self.checkpoint_manager.create_checkpoint(
                        mission_id=mission_id,
                        stage=f"executing_{operation_index}",
                        data={
                            "operation_index": operation_index,
                            "world_state": self.world_state.to_dict(),
                            "perception_state": self.cognitive_world_model.get_summary()
                        }
                    )
                
                # 执行操作
                progress = (operation_index + 1) / len(mission.operations) * 100
                logger.info(f"执行操作 [{operation_index + 1}/{len(mission.operations)}]: {operation.name}")
                
                # 定期汇报进度
                if operation_index % 3 == 0:
                    adjustment = await self.dialogue.report_progress(
                        status="执行中",
                        progress_percent=progress,
                        current_operation=operation.name,
                        world_state_summary=self.cognitive_world_model.get_summary().__str__(),
                        allow_adjustment=(operation_index > 0)
                    )
                    
                    if adjustment and adjustment not in ["继续", ""]:
                        # 用户请求调整
                        logger.info(f"用户请求调整: {adjustment}")
                        # 这里可以处理用户的调整请求
                
                result = await self._execute_operation(operation)
                
                if result.status == OperationStatus.SUCCESS:
                    consecutive_failures = 0
                    operation_index += 1
                    
                    # 更新世界状态
                    await self._update_world_state(operation, result)
                    
                elif result.status == OperationStatus.FAILED:
                    consecutive_failures += 1
                    logger.warning(f"操作失败: {result.error_message}")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("达到最大失败次数，任务中止")
                        mission.status = MissionStatus.FAILED
                        
                        await self.dialogue.report_error(
                            error=f"连续失败{consecutive_failures}次",
                            operation=operation.name,
                            suggestions=["中止任务", "手动接管"],
                            allow_choice=False
                        )
                        break
                    
                    if auto_recovery:
                        # 使用CoT进行智能恢复
                        recovery_result = await self._intelligent_failure_recovery(
                            mission=mission,
                            operation=operation,
                            operation_index=operation_index,
                            error=result.error_message
                        )
                        
                        if recovery_result.success:
                            if recovery_result.replanned:
                                mission.operations = recovery_result.new_operations
                                operation_index = recovery_result.resume_index
                            else:
                                continue
                        else:
                            mission.status = MissionStatus.FAILED
                            break
                    else:
                        mission.status = MissionStatus.FAILED
                        break
                
                # 检查系统状态
                if self.status == BrainStatus.EMERGENCY:
                    logger.warning("系统进入紧急状态，暂停任务")
                    mission.status = MissionStatus.PAUSED
                    mission.metadata["resume_from"] = operation_index
                    break
            
            # 任务完成
            if operation_index >= len(mission.operations):
                mission.status = MissionStatus.COMPLETED
                logger.info(f"任务 [{mission_id}] 执行完成")
                
                await self.dialogue.send_information(
                    f"✅ 任务 [{mission_id}] 执行完成!\n"
                    f"总操作: {len(mission.operations)}\n"
                    f"重规划次数: {mission.metadata.get('replan_count', 0)}"
                )
            
            await self._emit_event("mission_completed", mission)
            
        except Exception as e:
            logger.error(f"任务执行异常: {e}")
            mission.status = MissionStatus.FAILED
            mission.metadata["error"] = str(e)
            
            await self.dialogue.report_error(
                error=str(e),
                operation="任务执行",
                suggestions=["查看日志", "重试任务"],
                allow_choice=False
            )
            
        finally:
            # 停止感知监控
            await self.perception_monitor.stop_monitoring()
            self.current_mission = None
            self.status = BrainStatus.READY
            self.dialogue.end_session()
        
        return mission.status
    
    async def _evaluate_changes_for_replan(
        self,
        changes: List[EnvironmentChange],
        current_operation: Operation,
        remaining_operations: List[Operation],
        mission: Mission
    ) -> Dict[str, Any]:
        """使用CoT评估变化是否需要重规划"""
        # 构建变化描述
        changes_desc = "\n".join([
            f"- [{c.priority.value}] {c.description}"
            for c in changes
        ])
        
        # 构建当前计划描述
        remaining_ops_desc = "\n".join([
            f"  {i+1}. {op.name}"
            for i, op in enumerate(remaining_operations[:5])
        ])
        
        # 使用CoT推理
        reasoning = await self.cot_engine.reason(
            query="环境发生变化，是否需要调整计划？",
            context={
                "changes": changes_desc,
                "current_operation": current_operation.name,
                "remaining_plan": remaining_ops_desc,
                "original_task": mission.natural_language_command
            },
            mode=ReasoningMode.REPLANNING
        )
        
        # 记录推理
        mission.reasoning_chain.append({
            "type": "replan_evaluation",
            "reasoning": reasoning.to_dict()
        })
        
        # 解析决策
        decision = reasoning.decision.lower()
        
        if "replan" in decision or "重规划" in decision or "调整" in decision:
            # 需要确认
            if reasoning.confidence < 0.8:
                confirmed = await self.dialogue.report_and_confirm(
                    message=f"检测到环境变化:\n{changes_desc}",
                    suggestion=reasoning.suggestion,
                    details={"confidence": f"{reasoning.confidence:.0%}"}
                )
                if not confirmed:
                    return {"action": "continue", "reason": "用户拒绝重规划"}
            
            return {
                "action": "replan",
                "reason": reasoning.suggestion,
                "reasoning": reasoning
            }
        
        elif "pause" in decision or "暂停" in decision:
            return {
                "action": "pause",
                "reason": reasoning.suggestion
            }
        
        return {"action": "continue", "reason": "变化不影响当前计划"}
    
    async def _perception_driven_replan(
        self,
        mission: Mission,
        changes: List[EnvironmentChange],
        operation_index: int,
        replan_decision: Dict[str, Any]
    ) -> Optional[List[Operation]]:
        """感知驱动的重规划"""
        self.status = BrainStatus.REPLANNING
        
        logger.info("执行感知驱动重规划...")
        
        # 获取最新的规划上下文
        planning_context = self.cognitive_world_model.get_context_for_planning()
        
        # 已完成的操作
        completed_ops = mission.operations[:operation_index]
        
        try:
            # 使用增强的重规划器
            new_ops = await self.replanner.replan_with_perception(
                original_command=mission.natural_language_command,
                completed_operations=completed_ops,
                changes=changes,
                planning_context=planning_context,
                cot_reasoning=replan_decision.get("reasoning"),
                platform_type=mission.platform_type
            )
            
            if new_ops:
                # 记录重规划
                mission.metadata["replan_count"] = mission.metadata.get("replan_count", 0) + 1
                mission.metadata["last_replan"] = {
                    "timestamp": datetime.now().isoformat(),
                    "changes": [c.description for c in changes],
                    "new_ops_count": len(new_ops)
                }
                
                # 汇报重规划结果
                await self.dialogue.send_information(
                    f"🔄 任务已重规划\n"
                    f"- 原因: {replan_decision.get('reason', '环境变化')}\n"
                    f"- 新操作数: {len(new_ops)}"
                )
                
                logger.info(f"重规划完成，新操作数: {len(new_ops)}")
                return new_ops
                
        except Exception as e:
            logger.error(f"重规划失败: {e}")
            
            # 报告错误
            choice = await self.dialogue.report_error(
                error=str(e),
                operation="重规划",
                suggestions=["继续原计划", "暂停任务", "中止任务"],
                allow_choice=True
            )
            
            if "暂停" in choice:
                mission.status = MissionStatus.PAUSED
            elif "中止" in choice:
                mission.status = MissionStatus.FAILED
        
        finally:
            self.status = BrainStatus.EXECUTING
        
        return None
    
    async def _intelligent_failure_recovery(
        self,
        mission: Mission,
        operation: Operation,
        operation_index: int,
        error: str
    ) -> 'RecoveryResult':
        """智能失败恢复（使用CoT）"""
        self.status = BrainStatus.RECOVERING
        
        logger.info(f"智能错误恢复: {error}")
        
        # 获取当前环境
        planning_context = self.cognitive_world_model.get_context_for_planning()
        
        # 使用CoT分析异常
        reasoning = await self.cot_engine.reason(
            query=f"操作失败: {operation.name}，错误: {error}",
            context={
                "failed_operation": operation.name,
                "operation_params": operation.parameters,
                "error": error,
                "environment": planning_context.to_prompt_context()
            },
            mode=ReasoningMode.EXCEPTION_HANDLING
        )
        
        # 记录推理
        mission.reasoning_chain.append({
            "type": "failure_recovery",
            "reasoning": reasoning.to_dict()
        })
        
        # 分析错误
        error_analysis = await self.error_handler.analyze(
            operation=operation,
            error=error,
            world_state=self.world_state
        )
        
        # 根据推理结果决定恢复策略
        if "retry" in reasoning.decision.lower() or "重试" in reasoning.decision:
            return RecoveryResult(
                success=True,
                replanned=False,
                resume_index=operation_index
            )
        
        elif "replan" in reasoning.decision.lower() or "重规划" in reasoning.decision:
            # 重规划
            completed_ops = mission.operations[:operation_index]
            env_state = await self._get_environment_state()
            
            new_ops = await self.replanner.replan(
                original_command=mission.natural_language_command,
                completed_operations=completed_ops,
                failed_operation=operation,
                error=error,
                environment_state=env_state,
                world_state=self.world_state
            )
            
            return RecoveryResult(
                success=True,
                replanned=True,
                new_operations=completed_ops + new_ops,
                resume_index=operation_index
            )
        
        elif error_analysis.can_rollback:
            # 回滚
            await self._rollback(mission, operation_index)
            return RecoveryResult(success=False)
        
        return RecoveryResult(success=False)
    
    # === 感知监控回调 ===
    
    async def _on_replan_triggered(self, event: MonitorEvent):
        """重规划触发回调"""
        logger.info(f"感知监控触发重规划: {event.change.description}")
        
        if self.current_mission and self.status == BrainStatus.EXECUTING:
            # 标记需要重规划，在主循环中处理
            self.current_mission.metadata["pending_replan"] = {
                "change": event.change.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _on_confirmation_required(self, event: MonitorEvent) -> bool:
        """确认请求回调"""
        logger.info(f"感知监控请求确认: {event.change.description}")
        
        return await self.dialogue.request_confirmation(
            action=f"处理: {event.change.description}",
            reason=f"检测到{event.change.change_type.value}",
            details=event.change.data
        )
    
    async def _on_notification(self, event: MonitorEvent):
        """通知回调"""
        await self.dialogue.send_information(
            f"📢 感知通知: {event.change.description}",
            metadata={"event_type": event.change.change_type.value}
        )
    
    # === 原有方法（保持兼容性） ===
    
    async def execute_mission(
        self, 
        mission_id: str,
        auto_recovery: bool = True
    ) -> MissionStatus:
        """
        执行任务（兼容旧接口，内部调用感知驱动版本）
        """
        return await self.execute_mission_with_perception(
            mission_id=mission_id,
            auto_recovery=auto_recovery
        )
    
    async def _execute_operation(self, operation: Operation) -> OperationResult:
        """执行单个操作"""
        # 前置条件检查
        if not await self._check_preconditions(operation):
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message="前置条件不满足"
            )
        
        # 安全检查
        safety_check = await self._safety_check(operation)
        if not safety_check.passed:
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message=f"安全检查失败: {safety_check.reason}"
            )
        
        # 执行操作
        result = await self.executor.execute(
            operation=operation,
            robot_interface=self.robot_interface
        )
        
        return result
    
    async def _handle_failure(
        self,
        mission: Mission,
        operation: Operation,
        operation_index: int,
        error: str
    ):
        """处理操作失败（兼容旧接口）"""
        return await self._intelligent_failure_recovery(
            mission=mission,
            operation=operation,
            operation_index=operation_index,
            error=error
        )
    
    async def _rollback(self, mission: Mission, to_index: int):
        """回滚到指定操作"""
        logger.info(f"回滚任务到操作 {to_index}")
        
        checkpoint = await self.checkpoint_manager.get_nearest_checkpoint(
            mission_id=mission.id,
            target_index=to_index
        )
        
        if checkpoint:
            self.world_state.restore(checkpoint.data.get("world_state", {}))
            
            rollback_index = checkpoint.data.get("operation_index", 0)
            for i in range(to_index - 1, rollback_index - 1, -1):
                op = mission.operations[i]
                if op.rollback_action:
                    await self.executor.execute(
                        operation=op.rollback_action,
                        robot_interface=self.robot_interface
                    )
            
            logger.info(f"回滚完成，恢复到检查点: {checkpoint.stage}")
    
    async def _get_environment_state(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        # 使用 WorldModel 获取环境状态
        context = self.cognitive_world_model.get_context_for_planning()
        return {
            "robot_position": context.current_position,
            "robot_heading": context.current_heading,
            "obstacles": context.obstacles,
            "targets": context.targets,
            "points_of_interest": context.points_of_interest,
            "weather": context.weather,
            "battery_level": context.battery_level,
            "signal_strength": context.signal_strength,
            "constraints": context.constraints,
            "recent_changes": context.recent_changes
        }
    
    async def _update_world_state(self, operation: Operation, result: OperationResult):
        """更新世界状态"""
        self.world_state.update(
            operation=operation,
            result=result,
            timestamp=datetime.now()
        )
    
    async def _check_preconditions(self, operation: Operation) -> bool:
        """检查操作前置条件"""
        for precondition in operation.preconditions:
            if not await precondition.check(self.world_state):
                return False
        return True
    
    async def _safety_check(self, operation: Operation):
        """安全检查"""
        return await self.monitor.safety_check(operation)
    
    def _get_safety_constraints(self, platform_type: str) -> Dict[str, Any]:
        """获取平台安全约束"""
        platform_config = self.config.get(f"platforms.{platform_type}", {})
        safety_config = self.config.get("safety", {})
        
        return {
            "max_speed": platform_config.get("max_speed"),
            "safe_distance": platform_config.get("safe_distance"),
            "battery_warning": platform_config.get("battery_warning"),
            "geofence": safety_config.get("geofence", {}),
            "no_fly_zones": safety_config.get("no_fly_zones", [])
        }
    
    async def _emit_event(self, event_type: str, data: Any):
        """触发事件"""
        callbacks = self.event_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"事件回调执行失败: {e}")
    
    def on(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def pause_mission(self, mission_id: str):
        """暂停任务"""
        mission = self.missions.get(mission_id)
        if mission and mission.status == MissionStatus.EXECUTING:
            mission.status = MissionStatus.PAUSED
            logger.info(f"任务 [{mission_id}] 已暂停")
    
    async def resume_mission(self, mission_id: str):
        """恢复任务"""
        mission = self.missions.get(mission_id)
        if mission and mission.status == MissionStatus.PAUSED:
            return await self.execute_mission(mission_id)
    
    async def cancel_mission(self, mission_id: str):
        """取消任务"""
        mission = self.missions.get(mission_id)
        if mission:
            mission.status = MissionStatus.CANCELLED
            if self.current_mission and self.current_mission.id == mission_id:
                self.current_mission = None
            logger.info(f"任务 [{mission_id}] 已取消")
    
    async def emergency_stop(self):
        """紧急停止"""
        logger.warning("触发紧急停止!")
        self.status = BrainStatus.EMERGENCY
        
        await self.robot_interface.emergency_stop()
        
        if self.current_mission:
            self.current_mission.status = MissionStatus.PAUSED
        
        await self.perception_monitor.stop_monitoring()
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "brain_id": self.id,
            "status": self.status.value,
            "current_mission": self.current_mission.id if self.current_mission else None,
            "total_missions": len(self.missions),
            "world_state": self.world_state.summary(),
            "cognitive_world_model": self.cognitive_world_model.get_summary(),
            "perception_monitor": self.perception_monitor.get_status(),
            "dialogue_history_count": len(self.dialogue.get_conversation_history()),
            "monitor": self.monitor.get_metrics()
        }
    
    def get_reasoning_history(self, mission_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取推理历史"""
        if mission_id:
            mission = self.missions.get(mission_id)
            if mission:
                return mission.reasoning_chain
            return []
        
        # 返回所有任务的推理历史
        all_history = []
        for mission in self.missions.values():
            all_history.extend(mission.reasoning_chain)
        return all_history
    
    async def _update_perception_loop(self):
        """持续更新感知数据到认知层"""
        from brain.perception.vlm.vlm_perception import OLLAMA_AVAILABLE
        
        while self._running:
            try:
                # 从ROS2SensorManager获取融合数据
                perception_data = await self.sensor_manager.get_fused_perception()
                
                # 更新WorldModel - 机器人位置
                if perception_data and perception_data.pose:
                    self.cognitive_world_model.robot_position = {
                        "x": perception_data.pose.x if hasattr(perception_data.pose, 'x') else 0.0,
                        "y": perception_data.pose.y if hasattr(perception_data.pose, 'y') else 0.0,
                        "z": perception_data.pose.z if hasattr(perception_data.pose, 'z') else 0.0,
                        "lat": 0.0,
                        "lon": 0.0,
                        "alt": 0.0
                    }
                
                # 更新占据栅格地图
                if perception_data and perception_data.occupancy_grid is not None:
                    self.cognitive_world_model.current_map = perception_data.occupancy_grid
                    self.cognitive_world_model.map_resolution = perception_data.grid_resolution
                    self.cognitive_world_model.map_origin = perception_data.grid_origin
                    logger.debug(f"更新占据地图: shape={perception_data.occupancy_grid.shape}")
                
                # VLM场景理解（如果有RGB图像且VLM可用）
                if perception_data and perception_data.rgb_image is not None:
                    # 检查认知层是否有VLM
                    if hasattr(self.cognitive_layer, 'vlm') and self.cognitive_layer.vlm is not None and OLLAMA_AVAILABLE:
                        try:
                            import numpy as np
                            # 确保图像是numpy数组
                            rgb_image = perception_data.rgb_image
                            if not isinstance(rgb_image, np.ndarray):
                                logger.warning("RGB图像不是numpy数组，跳过VLM分析")
                            else:
                                scene = await self.cognitive_layer.vlm.analyze_scene(rgb_image)
                                # 更新语义对象到WorldModel
                                if hasattr(scene, 'objects') and scene.objects:
                                    for obj in scene.objects:
                                        self.cognitive_world_model.add_tracked_object(obj)
                        except Exception as e:
                            logger.warning(f"VLM场景分析失败: {e}")
                
                # 等待一段时间再更新（避免占用过多CPU）
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"感知数据更新异常: {e}")
                await asyncio.sleep(1.0)
    
    async def shutdown(self):
        """关闭系统"""
        logger.info("Brain 系统关闭中...")
        self.status = BrainStatus.SHUTDOWN
        
        # 停止感知监控任务
        self._running = False
        if self._perception_update_task is not None:
            self._perception_update_task.cancel()
            try:
                await self._perception_update_task
            except asyncio.CancelledError:
                pass
        
        # 停止感知监控
        await self.perception_monitor.stop_monitoring()
        
        # 结束对话会话
        self.dialogue.end_session()
        
        # 保存状态
        await self.checkpoint_manager.save_all()
        
        # 关闭连接
        await self.robot_interface.disconnect()
        
        # 关闭ROS2接口
        if hasattr(self, 'ros2') and self.ros2 is not None:
            await self.ros2.shutdown()
        
        logger.info("Brain 系统已关闭")


@dataclass
class RecoveryResult:
    """恢复结果"""
    success: bool
    replanned: bool = False
    new_operations: List[Operation] = field(default_factory=list)
    resume_index: int = 0
