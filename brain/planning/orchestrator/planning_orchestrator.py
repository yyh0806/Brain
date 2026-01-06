"""
规划编排器 - Planning Orchestrator

统一规划与执行接口，协调三层规划器
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from brain.planning.state import PlanState
from brain.planning.capability import CapabilityRegistry, PlatformAdapter
from brain.planning.action_level import WorldModelMock
from brain.planning.planners import TaskLevelPlanner, SkillLevelPlanner, ActionLevelPlanner
from brain.execution.monitor import AdaptiveExecutor
from brain.execution.executor import Executor
from brain.state.world_state import WorldState


class PlanningOrchestrator:
    """
    规划编排器
    
    提供统一的规划与执行接口
    协调Task-level、Skill-level、Action-level三层规划器
    """
    
    def __init__(
        self,
        platform: str = "ugv",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化规划编排器
        
        Args:
            platform: 平台类型（drone, ugv, usv）
            config: 配置
        """
        self.platform = platform
        self.config = config or {}
        
        # 初始化组件
        self.capability_registry = CapabilityRegistry()
        self.platform_adapter = PlatformAdapter(self.capability_registry)
        self.world_model = WorldModelMock()
        self.world_state = WorldState()
        
        # 三层规划器
        self.action_planner = ActionLevelPlanner(
            capability_registry=self.capability_registry,
            platform_adapter=self.platform_adapter,
            world_model=self.world_model,
            platform=platform
        )
        
        self.skill_planner = SkillLevelPlanner(
            action_planner=self.action_planner
        )
        
        self.task_planner = TaskLevelPlanner()
        
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
        
        logger.info(f"PlanningOrchestrator 初始化完成 (平台: {platform})")
    
    async def plan_and_execute(
        self,
        command: str,
        robot_interface: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        规划并执行
        
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
        只规划，不执行
        
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
