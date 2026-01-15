"""
规划编排器 - Planning Orchestrator

统一规划与执行接口，协调三层规划器
支持认知层集成和动态重规划
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from brain.planning.state import PlanState
from brain.planning.capability import CapabilityRegistry, PlatformAdapter
from brain.planning.action_level import WorldModelMock
from brain.planning.planners import TaskLevelPlanner, SkillLevelPlanner, ActionLevelPlanner
from brain.planning.interfaces import (
    PlanningInput,
    PlanningOutput,
    ReplanningInput,
    ReplanningOutput,
    PlanningStatus,
    CognitiveWorldAdapter,
    IWorldModel
)
from brain.planning.intelligent import ReplanningManager
from brain.execution.monitor import AdaptiveExecutor
from brain.execution.executor import Executor
from brain.state.world_state import WorldState


class PlanningOrchestrator:
    """
    规划编排器

    提供统一的规划与执行接口
    协调Task-level、Skill-level、Action-level三层规划器

    支持两种模式：
    1. 独立模式：使用 WorldModelMock，接收简单的字符串指令
    2. 集成模式：使用 CognitiveWorldAdapter，接收来自认知层的 PlanningInput
    """

    def __init__(
        self,
        platform: str = "ugv",
        config: Optional[Dict[str, Any]] = None,
        use_cognitive_layer: bool = False,
        enable_replanning: bool = True
    ):
        """
        初始化规划编排器

        Args:
            platform: 平台类型（drone, ugv, usv）
            config: 配置
            use_cognitive_layer: 是否使用认知层集成模式
            enable_replanning: 是否启用动态重规划
        """
        self.platform = platform
        self.config = config or {}
        self.use_cognitive_layer = use_cognitive_layer
        self.enable_replanning = enable_replanning

        # 初始化组件
        self.capability_registry = CapabilityRegistry()
        self.platform_adapter = PlatformAdapter(self.capability_registry)

        # 世界模型（支持两种模式）
        if use_cognitive_layer:
            self.world_model: Optional[IWorldModel] = None  # 将通过 update_context 设置
            self.cognitive_adapter: Optional[CognitiveWorldAdapter] = None
        else:
            self.world_model = WorldModelMock()
            self.cognitive_adapter = None

        self.world_state = WorldState()

        # 三层规划器
        # 注意：action_planner 需要在设置世界模型后初始化
        self.action_planner: Optional[ActionLevelPlanner] = None
        self.skill_planner: Optional[SkillLevelPlanner] = None
        self.task_planner = TaskLevelPlanner()

        # 延迟初始化规划器（在设置世界模型后）
        self._init_planners()

        # 执行器
        self.executor = Executor(
            world_state=self.world_state,
            config=config
        )

        self.adaptive_executor = AdaptiveExecutor(
            executor=self.executor,
            world_model=self.world_model,
            world_state=self.world_state,
            config=config
        )

        # 重规划管理器
        self.replanning_manager: Optional[ReplanningManager] = None
        if enable_replanning:
            self.replanning_manager = ReplanningManager(
                world_model=self.world_model,
                config=config
            )

        logger.info(
            f"PlanningOrchestrator 初始化完成 "
            f"(平台: {platform}, 认知层模式: {use_cognitive_layer}, "
            f"重规划: {enable_replanning})"
        )

    def _init_planners(self):
        """初始化三层规划器"""
        if self.world_model is not None:
            self.action_planner = ActionLevelPlanner(
                capability_registry=self.capability_registry,
                platform_adapter=self.platform_adapter,
                world_model=self.world_model,
                platform=self.platform
            )

            self.skill_planner = SkillLevelPlanner(
                action_planner=self.action_planner
            )

    def update_context(
        self,
        planning_context,
        beliefs: Optional[List] = None
    ):
        """
        更新认知上下文（认知层集成模式）

        Args:
            planning_context: PlanningContext 对象
            beliefs: 信念列表
        """
        if not self.use_cognitive_layer:
            logger.warning("未启用认知层集成模式，忽略上下文更新")
            return

        # 创建或更新适配器
        if self.cognitive_adapter is None:
            self.cognitive_adapter = CognitiveWorldAdapter(
                planning_context=planning_context,
                beliefs=beliefs
            )
            self.world_model = self.cognitive_adapter
            self._init_planners()
            self.adaptive_executor.world_model = self.world_model
        else:
            self.cognitive_adapter.update_context(planning_context, beliefs)

        logger.debug("认知上下文已更新")

    # ============ 新接口：认知层集成 ============

    def plan(
        self,
        planning_input: PlanningInput
    ) -> PlanningOutput:
        """
        规划（新接口，支持认知层集成）

        完整流程：
        1. 检查输入有效性
        2. 处理认知层推理结果（如果有）
        3. Task-level: 解析自然语言指令
        4. Skill-level: 分解为技能序列
        5. Action-level: 转换为操作序列
        6. 生成 PlanningOutput

        Args:
            planning_input: 规划输入

        Returns:
            PlanningOutput: 规划输出
        """
        logger.info(f"开始规划: {planning_input.command}")

        # 1. 检查输入有效性
        if not planning_input.command:
            return PlanningOutput(
                plan_state=PlanState(),
                planning_status=PlanningStatus.FAILURE,
                estimated_duration=0.0,
                success_rate=0.0,
                rejection_reason="指令为空"
            )

        # 2. 更新认知上下文（如果有）
        if self.use_cognitive_layer and planning_input.planning_context:
            self.update_context(
                planning_input.planning_context,
                planning_input.beliefs
            )

        # 3. 处理认知层推理结果
        if planning_input.has_reasoning():
            reasoning_result = planning_input.reasoning_result
            logger.info(f"推理模式: {reasoning_result.mode.value}, 置信度: {reasoning_result.confidence:.2f}")

            # 检查推理结果是否建议拒绝执行
            if reasoning_result.confidence < 0.3:
                return PlanningOutput(
                    plan_state=PlanState(),
                    planning_status=PlanningStatus.REJECTED,
                    estimated_duration=0.0,
                    success_rate=0.0,
                    rejection_reason=f"推理置信度过低: {reasoning_result.confidence:.2f}"
                )

        # 4. Task-level规划
        try:
            task_info = self.task_planner.parse_command(planning_input.command)
            task_node = self.task_planner.create_task_node(task_info)
        except Exception as e:
            logger.error(f"Task-level规划失败: {e}")
            return PlanningOutput(
                plan_state=PlanState(),
                planning_status=PlanningStatus.FAILURE,
                estimated_duration=0.0,
                success_rate=0.0,
                rejection_reason=f"任务解析失败: {str(e)}"
            )

        # 5. Skill-level规划
        try:
            skills = task_info.get("skills", [])
            task_node = self.skill_planner.decompose_task(task_node, skills)
        except Exception as e:
            logger.error(f"Skill-level规划失败: {e}")
            return PlanningOutput(
                plan_state=PlanState(),
                planning_status=PlanningStatus.PARTIAL,
                estimated_duration=0.0,
                success_rate=0.5,
                rejection_reason=f"任务分解失败: {str(e)}"
            )

        # 6. 创建PlanState
        plan_state = PlanState()
        plan_state.add_root(task_node)

        logger.info(f"规划完成，共 {len(plan_state.nodes)} 个节点")

        # 7. 估算执行时间和成功率
        estimated_duration = self._estimate_duration(plan_state)
        success_rate = self._estimate_success_rate(plan_state, planning_input)

        # 8. 生成输出
        return PlanningOutput(
            plan_state=plan_state,
            planning_status=PlanningStatus.SUCCESS,
            estimated_duration=estimated_duration,
            success_rate=success_rate,
            planning_log=["规划成功", f"节点数: {len(plan_state.nodes)}"]
        )

    async def plan_and_execute_async(
        self,
        planning_input: PlanningInput,
        robot_interface: Optional[Any] = None
    ) -> PlanningOutput:
        """
        规划并执行（新接口，异步）

        Args:
            planning_input: 规划输入
            robot_interface: 机器人接口（可选）

        Returns:
            PlanningOutput: 包含执行结果
        """
        # 1. 先规划
        output = self.plan(planning_input)

        if not output.is_successful():
            return output

        # 2. 执行计划
        try:
            result = await self.adaptive_executor.execute_plan(
                plan_state=output.plan_state,
                robot_interface=robot_interface
            )

            # 3. 更新输出（添加执行结果）
            output.metadata["execution_result"] = result
            output.planning_log.append("执行完成")

        except Exception as e:
            logger.error(f"执行失败: {e}")
            output.planning_status = PlanningStatus.FAILURE
            output.success_rate = 0.0
            output.planning_log.append(f"执行失败: {str(e)}")

        return output

    # ============ 原有接口：向后兼容 ============

    async def plan_and_execute(
        self,
        command: str,
        robot_interface: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        规划并执行（原有接口，向后兼容）

        完整流程：
        1. Task-level: 解析自然语言指令
        2. Skill-level: 分解为技能序列
        3. Action-level: 转换为操作序列
        4. 执行计划

        Args:
            command: 自然语言指令
            robot_interface: 机器人接口（可选）

        Returns:
            执行结果
        """
        logger.info(f"开始规划并执行: {command}")

        # Step 1: Task-level规划
        task_info = self.task_planner.parse_command(command)
        task_node = self.task_planner.create_task_node(task_info)

        # Step 2: Skill-level规划
        skills = task_info.get("skills", [])
        task_node = self.skill_planner.decompose_task(task_node, skills)

        # Step 3: 创建PlanState
        plan_state = PlanState()
        plan_state.add_root(task_node)

        logger.info(f"规划完成，共 {len(plan_state.nodes)} 个节点")

        # Step 4: 执行计划
        result = await self.adaptive_executor.execute_plan(
            plan_state=plan_state,
            robot_interface=robot_interface
        )

        return result

    def get_plan(
        self,
        command: str
    ) -> PlanState:
        """
        只规划，不执行（原有接口，向后兼容）

        Args:
            command: 自然语言指令

        Returns:
            计划状态
        """
        logger.info(f"规划: {command}")

        # Task-level规划
        task_info = self.task_planner.parse_command(command)
        task_node = self.task_planner.create_task_node(task_info)

        # Skill-level规划
        skills = task_info.get("skills", [])
        task_node = self.skill_planner.decompose_task(task_node, skills)

        # 创建PlanState
        plan_state = PlanState()
        plan_state.add_root(task_node)

        return plan_state

    def get_capabilities(self) -> Dict[str, Any]:
        """获取平台能力信息"""
        return self.platform_adapter.get_capability_info(self.platform)

    # ============ 辅助方法 ============

    def _estimate_duration(self, plan_state: PlanState) -> float:
        """估算计划执行时间"""
        total_time = 0.0
        for node in plan_state.nodes:
            # 根据节点类型估算时间
            if hasattr(node, 'action_type'):
                # 从能力注册表获取默认时间
                capability = self.capability_registry.get_capability(node.action_type)
                if capability:
                    total_time += capability.default_duration
                else:
                    total_time += 1.0  # 默认1秒
            else:
                total_time += 1.0
        return total_time

    def _estimate_success_rate(
        self,
        plan_state: PlanState,
        planning_input: PlanningInput
    ) -> float:
        """估算计划成功率"""
        base_rate = 0.9

        # 基于推理置信度调整
        if planning_input.has_reasoning():
            base_rate = planning_input.reasoning_result.confidence

        # 基于信念数量调整
        if planning_input.beliefs:
            high_conf_beliefs = len(planning_input.get_high_confidence_beliefs())
            base_rate = base_rate * (0.8 + 0.2 * min(high_conf_beliefs / 5, 1.0))

        return max(0.0, min(1.0, base_rate))

    # ============ 动态重规划接口 ============

    def replan(
        self,
        replanning_input: ReplanningInput
    ) -> ReplanningOutput:
        """
        执行重规划

        Args:
            replanning_input: 重规划输入

        Returns:
            ReplanningOutput: 重规划输出
        """
        if not self.replanning_manager:
            logger.warning("重规划管理器未启用")
            return ReplanningOutput(
                new_plan=replanning_input.current_plan,
                replanning_type="none",
                success=False,
                reason="重规划功能未启用"
            )

        logger.info(f"开始重规划: {replanning_input.trigger_reason}")

        # 1. 检测环境变化
        changes = self.replanning_manager.detect_environment_changes(
            current_plan=replanning_input.current_plan,
            current_context=replanning_input.metadata.get('context'),
            new_beliefs=replanning_input.new_beliefs,
            failed_actions=replanning_input.failed_actions
        )

        # 2. 做出重规划决策
        decision = self.replanning_manager.make_replanning_decision(
            replanning_input=replanning_input,
            changes=changes
        )

        logger.info(
            f"重规划决策: {decision.strategy.value}, "
            f"置信度: {decision.confidence:.2f}, "
            f"原因: {decision.reason}"
        )

        # 3. 执行重规划
        if decision.should_replan or decision.strategy != ReplanningStrategy.RETRY:
            output = self.replanning_manager.replan(replanning_input, decision)

            # 4. 如果需要完全重规划，调用规划器
            if output.metadata.get('requires_full_replanning'):
                logger.info("需要完全重新规划，调用规划器")
                # 这里可以根据需要重新执行完整的规划流程
                # 简化实现：返回标记
                return output

            return output
        else:
            # 不需要重规划
            return ReplanningOutput(
                new_plan=replanning_input.current_plan,
                replanning_type="none",
                success=True,
                reason="无需重规划"
            )

    def check_and_replan(
        self,
        current_plan: PlanState,
        failed_actions: List[str],
        current_context: Any = None,
        new_beliefs: Optional[List] = None
    ) -> ReplanningOutput:
        """
        检查并执行重规划（便捷方法）

        Args:
            current_plan: 当前计划
            failed_actions: 失败的动作列表
            current_context: 当前上下文（可选）
            new_beliefs: 新的信念列表（可选）

        Returns:
            ReplanningOutput: 重规划输出
        """
        replanning_input = ReplanningInput(
            current_plan=current_plan,
            failed_actions=failed_actions,
            environment_changes=[],
            new_beliefs=new_beliefs or [],
            trigger_reason="执行失败或环境变化",
            metadata={'context': current_context}
        )

        return self.replan(replanning_input)

    def get_replanning_statistics(self) -> Dict[str, Any]:
        """获取重规划统计信息"""
        if self.replanning_manager:
            return self.replanning_manager.get_statistics()
        return {}
