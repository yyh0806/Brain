"""
动态规划器 - Dynamic Planner

检查前置条件，动态插入必要操作
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from brain.planning.state import PlanNode, NodeStatus
from brain.planning.action_level import WorldModelMock
from brain.execution.monitor import FailureType


class DynamicPlanner:
    """
    动态规划器
    
    职责：
    - 检查前置条件
    - 插入必要的前置操作
    - 规则：最多插入3次，避免死循环
    """
    
    def __init__(
        self,
        world_model: WorldModelMock,
        max_insertions: int = 3
    ):
        """
        初始化动态规划器
        
        Args:
            world_model: 世界模型
            max_insertions: 最大插入次数
        """
        self.world_model = world_model
        self.max_insertions = max_insertions
        self.insertion_count = 0
        
        logger.info(f"DynamicPlanner 初始化完成 (最大插入次数: {max_insertions})")
    
    def check_and_insert_preconditions(
        self,
        node: PlanNode,
        plan_nodes: Optional[List[PlanNode]] = None
    ) -> tuple:
        """
        检查前置条件并插入必要操作

        Args:
            node: 要检查的节点
            plan_nodes: 计划节点列表（未使用，保留用于兼容性）

        Returns:
            (modified: bool, inserted_nodes: List[PlanNode]) 元组
            - modified: 是否进行了修改
            - inserted_nodes: 插入的操作节点列表
        """
        if plan_nodes is None:
            plan_nodes = []

        if self.insertion_count >= self.max_insertions:
            logger.warning(f"已达到最大插入次数 ({self.max_insertions})，停止插入")
            return False, []

        inserted_nodes = []

        # 检查前置条件
        for precondition in node.preconditions:
            inserted = self._handle_precondition(node, precondition, node.task)
            if inserted:
                inserted_nodes.extend(inserted)
                self.insertion_count += len(inserted)

        if inserted_nodes:
            logger.info(f"为节点 {node.name} 插入了 {len(inserted_nodes)} 个前置操作")

        return len(inserted_nodes) > 0, inserted_nodes
    
    def _handle_precondition(
        self,
        node: PlanNode,
        precondition: str,
        task_name: Optional[str] = None
    ) -> List[PlanNode]:
        """
        处理单个前置条件

        Args:
            node: 节点
            precondition: 前置条件字符串
            task_name: 任务名称（可选）

        Returns:
            插入的节点列表
        """
        if task_name is None:
            task_name = node.task

        inserted = []

        # 检查门状态相关的前置条件
        if "door.state" in precondition or "door_open" in precondition:
            door_name = self._extract_door_name(precondition)
            if door_name:
                door_state = self.world_model.get_door_state(door_name)
                if door_state == "closed":
                    # 插入检查门操作
                    check_node = PlanNode(
                        id=f"check_door_{door_name}_{self.insertion_count}",
                        name="check_door_status",
                        action="check_door_status",
                        skill=node.skill,
                        task=task_name or node.task,
                        parameters={"door_id": door_name},
                        preconditions=[],
                        expected_effects=[f"door_status({door_name}).known"]
                    )
                    inserted.append(check_node)
                    
                    # 插入开门操作
                    open_node = PlanNode(
                        id=f"open_door_{door_name}_{self.insertion_count}",
                        name="open_door",
                        action="open_door",
                        skill=node.skill,
                        task=task_name or node.task,
                        parameters={"door_id": door_name},
                        preconditions=[f"door.state({door_name}) == 'closed'"],
                        expected_effects=[f"door.state({door_name}) == 'open'"]
                    )
                    inserted.append(open_node)
        
        # 检查物体可见性相关的前置条件
        elif "object.visible" in precondition or "object_location" in precondition:
            object_name = self._extract_object_name(precondition)
            if object_name:
                object_location = self.world_model.get_object_location(object_name)
                if not object_location:
                    # 插入搜索操作
                    search_node = PlanNode(
                        id=f"search_{object_name}_{self.insertion_count}",
                        name="search_object",
                        action="search_object",
                        skill=node.skill,
                        task=task_name or node.task,
                        parameters={"object_type": object_name},
                        preconditions=[],
                        expected_effects=[f"object_location({object_name}).known"]
                    )
                    inserted.append(search_node)
        
        return inserted
    
    def _extract_door_name(self, precondition: str) -> Optional[str]:
        """从前置条件中提取门名称

        支持的格式：
        - door.door_name == value
        - door.state(door_name) == value
        - door_door_name
        """
        import re
        # 匹配多种格式
        # 1. door.door_name
        match = re.search(r'door\.(\w+)', precondition)
        if match:
            return match.group(1)

        # 2. door.state(door_name)
        match = re.search(r'door\.state\((\w+)\)', precondition)
        if match:
            return match.group(1)

        # 3. door_door_name
        match = re.search(r'door_(\w+)', precondition)
        if match:
            return match.group(2)  # 返回door_name部分

        return None

    def _extract_object_name(self, precondition: str) -> Optional[str]:
        """从前置条件中提取物体名称

        支持的格式：
        - object.object_name.property == value
        - object.visible(object_name)
        - object_location(object_name)
        """
        import re
        # 匹配多种格式
        # 1. object.object_name.xxx
        match = re.search(r'object\.(\w+)\.', precondition)
        if match:
            object_name = match.group(1)
            # 过滤掉常见的属性名
            if object_name not in ['visible', 'location', 'state', 'type']:
                return object_name

        # 2. object.visible(object_name)
        match = re.search(r'object\.visible\((\w+)\)', precondition)
        if match:
            return match.group(1)

        # 3. object_location(object_name)
        match = re.search(r'object_location\((\w+)\)', precondition)
        if match:
            return match.group(2)  # 返回object_name部分

        return None
    
    def reset_insertion_count(self):
        """重置插入计数（用于新的规划周期）"""
        self.insertion_count = 0
        logger.debug("插入计数已重置")
