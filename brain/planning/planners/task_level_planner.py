"""
Task-level规划器

使用大模型解析自然语言，生成任务意图和HTN结构
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from brain.planning.state import PlanNode


@dataclass
class TaskGoal:
    """任务目标与成功判定条件"""
    conditions: List[str] = field(default_factory=list)  # 硬条件
    soft_constraints: List[str] = field(default_factory=list)  # 软约束
    partial_success: Optional[Dict[str, List[str]]] = None  # 部分成功条件
    
    def check_success(self, world_state: Any) -> Tuple[bool, float]:
        """
        检查是否成功
        
        Returns:
            (是否成功, 完成度0-1)
        """
        # Phase 2: 简单实现
        # Phase 3: 完整实现
        return False, 0.0


class TaskLevelPlanner:
    """
    Task-level规划器
    
    职责：理解"要干什么"
    使用大模型进行语义解析
    """
    
    def __init__(self, llm_interface: Optional[Any] = None):
        """
        初始化Task-level规划器
        
        Args:
            llm_interface: LLM接口（Phase 2使用简单规则，Phase 3集成LLM）
        """
        self.llm_interface = llm_interface
        
        logger.info("TaskLevelPlanner 初始化完成")
    
    def parse_command(
        self,
        command: str
    ) -> Dict[str, Any]:
        """
        解析自然语言指令
        
        Phase 2: 使用简单规则解析
        Phase 3: 使用LLM解析
        
        Args:
            command: 自然语言指令
            
        Returns:
            解析后的任务信息
        """
        # Phase 2: 简单规则解析
        # 示例："去厨房拿杯水"
        
        task_info = {
            "task_name": "去厨房拿杯水",
            "task_type": "fetch",
            "goal": TaskGoal(
                conditions=[
                    "robot.holding(cup)",
                    "robot.position.near(start_position)"
                ]
            ),
            "skills": self._extract_skills(command)
        }
        
        logger.info(f"解析指令: {command} -> {task_info['task_name']}")
        
        return task_info
    
    def _extract_skills(self, command: str) -> List[str]:
        """
        从指令中提取技能序列
        
        Phase 2: 简单规则
        """
        # 简单规则：根据关键词提取技能
        skills = []
        
        if "去" in command or "到" in command:
            # 提取位置
            if "厨房" in command:
                skills.append("去厨房")
            elif "桌子" in command:
                skills.append("去桌子")
            else:
                skills.append("去位置")
        
        if "拿" in command or "取" in command:
            if "杯" in command:
                skills.append("拿杯子")
            else:
                skills.append("拿物体")
        
        if "放" in command or "放置" in command:
            if "桌子" in command:
                skills.append("放桌子")
            else:
                skills.append("放物体")
        
        if "回来" in command or "返回" in command:
            skills.append("回来")
        
        return skills
    
    def create_task_node(
        self,
        task_info: Dict[str, Any]
    ) -> PlanNode:
        """
        创建任务根节点
        
        Args:
            task_info: 任务信息
            
        Returns:
            任务根节点
        """
        task_node = PlanNode(
            id=f"task_{task_info['task_name']}",
            name=task_info['task_name'],
            task=task_info['task_name'],
            preconditions=[],
            expected_effects=task_info['goal'].conditions
        )
        
        return task_node
