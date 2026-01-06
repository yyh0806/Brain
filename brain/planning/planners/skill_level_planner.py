"""
Skill-level规划器

将任务分解为技能序列，支持HTN分解
可混合使用大模型和规则
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from brain.planning.state import PlanNode
from brain.planning.planners.action_level_planner import ActionLevelPlanner


class SkillLevelPlanner:
    """
    Skill-level规划器
    
    职责：确定"用哪些能力"
    将任务分解为技能序列
    """
    
    def __init__(
        self,
        action_planner: ActionLevelPlanner
    ):
        """
        初始化Skill-level规划器
        
        Args:
            action_planner: Action-level规划器
        """
        self.action_planner = action_planner
        
        logger.info("SkillLevelPlanner 初始化完成")
    
    def decompose_task(
        self,
        task_node: PlanNode,
        skills: List[str]
    ) -> PlanNode:
        """
        将任务分解为技能序列
        
        Args:
            task_node: 任务节点
            skills: 技能列表
            
        Returns:
            包含技能子节点的任务节点
        """
        logger.info(f"分解任务: {task_node.name} -> {len(skills)} 个技能")
        
        for i, skill_name in enumerate(skills):
            # 创建技能节点
            skill_node = PlanNode(
                id=f"skill_{skill_name}_{i}",
                name=skill_name,
                skill=skill_name,
                task=task_node.name,
                preconditions=[],
                expected_effects=[]
            )
            
            # 使用Action-level规划器生成操作
            # 简化：使用默认参数
            action_nodes = self.action_planner.plan_skill(
                skill_name=skill_name,
                parameters=self._get_skill_parameters(skill_name),
                task_name=task_node.name
            )
            
            # 添加操作节点为技能节点的子节点
            for action_node in action_nodes:
                skill_node.add_child(action_node)
            
            # 添加技能节点为任务节点的子节点
            task_node.add_child(skill_node)
        
        return task_node
    
    def _get_skill_parameters(self, skill_name: str) -> Dict[str, Any]:
        """获取技能参数（简化实现）"""
        params_map = {
            "去厨房": {"location": "kitchen"},
            "拿杯子": {"object": "cup"},
            "放桌子": {"location": "table"},
            "回来": {},
        }
        
        return params_map.get(skill_name, {})
