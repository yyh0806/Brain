"""
Action-level规划器

纯规则驱动的规划器，将技能转换为具体操作
不使用LLM，不使用概率，只使用确定值
"""

from typing import Dict, List, Any, Optional
from loguru import logger

from brain.planning.state import PlanNode, NodeStatus
from brain.planning.capability import CapabilityRegistry, PlatformAdapter
from brain.planning.action_level import WorldModelMock


class ActionLevelPlanner:
    """
    Action-level规划器
    
    职责：将技能转换为具体操作序列
    特点：
    - 纯规则驱动
    - 不使用LLM
    - 不使用概率，只使用确定值
    - 检查前置条件并插入必要操作
    """
    
    def __init__(
        self,
        capability_registry: CapabilityRegistry,
        platform_adapter: PlatformAdapter,
        world_model: WorldModelMock,
        platform: str = "ugv"
    ):
        """
        初始化Action-level规划器
        
        Args:
            capability_registry: 能力注册表
            platform_adapter: 平台适配器
            world_model: 世界模型（Phase 0使用mock）
            platform: 平台类型（drone, ugv, usv）
        """
        self.registry = capability_registry
        self.capability_registry = capability_registry  # 别名，兼容测试
        self.platform_adapter = platform_adapter
        self.world_model = world_model
        self.platform = platform
        
        logger.info(f"ActionLevelPlanner 初始化完成 (平台: {platform})")
    
    def plan_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        task_name: Optional[str] = None
    ) -> List[PlanNode]:
        """
        将技能转换为操作序列
        
        Args:
            skill_name: 技能名称（如"去厨房"、"拿杯子"）
            parameters: 技能参数
            task_name: 所属任务名称
            
        Returns:
            操作节点列表
        """
        logger.info(f"规划技能: {skill_name}")
        
        # Phase 0: 使用简单的规则映射
        skill_mapping = {
            "去厨房": self._plan_go_to_location,
            "去位置": self._plan_go_to_location,
            "拿杯子": self._plan_grasp_object,
            "拿物体": self._plan_grasp_object,
            "放杯子": self._plan_place_object,
            "放物体": self._plan_place_object,
            "回来": self._plan_return,
            "开门": self._plan_open_door,
            "检查门": self._plan_check_door,
        }
        
        planner = skill_mapping.get(skill_name)
        if planner:
            return planner(parameters, task_name, skill_name)
        else:
            logger.warning(f"未知技能: {skill_name}，使用默认规划")
            return self._plan_default_skill(skill_name, parameters, task_name)
    
    def _plan_go_to_location(
        self,
        parameters: Dict[str, Any],
        task_name: Optional[str],
        skill_name: str
    ) -> List[PlanNode]:
        """规划'去位置'技能"""
        location_name = parameters.get("location", parameters.get("target"))
        if not location_name:
            logger.error("缺少location参数")
            return []
        
        # 获取位置信息
        location = self.world_model.get_location(location_name)
        if not location:
            logger.error(f"未知位置: {location_name}")
            return []
        
        nodes = []
        
        # 检查是否需要开门
        door_name = f"{location_name}_door"
        door_state = self.world_model.get_door_state(door_name)
        
        if door_state == "closed":
            # 插入检查门操作
            check_node = PlanNode(
                id=f"check_door_{location_name}",
                name="check_door_status",
                action="check_door_status",
                skill=skill_name,
                task=task_name,
                parameters={"door_id": door_name},
                preconditions=[],
                expected_effects=[f"door_status({door_name}).known"]
            )
            nodes.append(check_node)
            
            # 插入开门操作
            open_node = PlanNode(
                id=f"open_door_{location_name}",
                name="open_door",
                action="open_door",
                skill=skill_name,
                task=task_name,
                parameters={"door_id": door_name},
                preconditions=[f"door.state({door_name}) == 'closed'"],
                expected_effects=[f"door.state({door_name}) == 'open'"]
            )
            nodes.append(open_node)
        
        # 移动到目标位置
        move_node = PlanNode(
            id=f"move_to_{location_name}",
            name="move_to",
            action="move_to",
            skill=skill_name,
            task=task_name,
            parameters={"position": location.position},
            preconditions=[f"door.state({door_name}) == 'open'" if door_state == "closed" else "robot.ready == True"],
            expected_effects=[f"robot.position.near({location_name})"]
        )
        nodes.append(move_node)
        
        return nodes
    
    def _plan_grasp_object(
        self,
        parameters: Dict[str, Any],
        task_name: Optional[str],
        skill_name: str
    ) -> List[PlanNode]:
        """规划'拿物体'技能"""
        object_name = parameters.get("object", parameters.get("object_id"))
        if not object_name:
            logger.error("缺少object参数")
            return []
        
        nodes = []
        
        # 查询物体位置
        object_location = self.world_model.get_object_location(object_name)
        
        if not object_location:
            # 物体位置未知，需要搜索
            search_node = PlanNode(
                id=f"search_{object_name}",
                name="search_object",
                action="search_object",
                skill=skill_name,
                task=task_name,
                parameters={"object_type": object_name},
                preconditions=[],
                expected_effects=[f"object_location({object_name}).known"]
            )
            nodes.append(search_node)
        else:
            # 移动到物体位置
            location = self.world_model.get_location(object_location)
            if location:
                move_node = PlanNode(
                    id=f"move_to_{object_name}",
                    name="move_to",
                    action="move_to",
                    skill=skill_name,
                    task=task_name,
                    parameters={"position": location.position},
                    preconditions=["robot.ready == True"],
                    expected_effects=[f"robot.position.near({object_name})"]
                )
                nodes.append(move_node)
        
        # 抓取物体
        grasp_node = PlanNode(
            id=f"grasp_{object_name}",
            name="grasp",
            action="grasp",
            skill=skill_name,
            task=task_name,
            parameters={"object_id": object_name},
            preconditions=[
                f"object.visible({object_name}) == True",
                f"robot.position.near({object_name})"
            ],
            expected_effects=[f"robot.holding({object_name})"]
        )
        nodes.append(grasp_node)
        
        return nodes
    
    def _plan_place_object(
        self,
        parameters: Dict[str, Any],
        task_name: Optional[str],
        skill_name: str
    ) -> List[PlanNode]:
        """规划'放物体'技能"""
        target_location = parameters.get("location", parameters.get("target"))
        if not target_location:
            logger.error("缺少location参数")
            return []
        
        nodes = []
        
        # 获取目标位置
        location = self.world_model.get_location(target_location)
        if not location:
            logger.error(f"未知位置: {target_location}")
            return []
        
        # 移动到目标位置
        move_node = PlanNode(
            id=f"move_to_{target_location}",
            name="move_to",
            action="move_to",
            skill=skill_name,
            task=task_name,
            parameters={"position": location.position},
            preconditions=["robot.holding(object_id)"],
            expected_effects=[f"robot.position.near({target_location})"]
        )
        nodes.append(move_node)
        
        # 放置物体
        place_node = PlanNode(
            id=f"place_{target_location}",
            name="place",
            action="place",
            skill=skill_name,
            task=task_name,
            parameters={"position": location.position},
            preconditions=[
                "robot.holding(object_id)",
                f"robot.position.near({target_location})"
            ],
            expected_effects=["robot.holding == None"]
        )
        nodes.append(place_node)
        
        return nodes
    
    def _plan_return(
        self,
        parameters: Dict[str, Any],
        task_name: Optional[str],
        skill_name: str
    ) -> List[PlanNode]:
        """规划'回来'技能"""
        start_position = self.world_model.get_robot_position()
        
        return_node = PlanNode(
            id="return_to_start",
            name="move_to",
            action="move_to",
            skill=skill_name,
            task=task_name,
            parameters={"position": start_position},
            preconditions=["robot.ready == True"],
            expected_effects=["robot.position.near(start_position)"]
        )
        
        return [return_node]
    
    def _plan_open_door(
        self,
        parameters: Dict[str, Any],
        task_name: Optional[str],
        skill_name: str
    ) -> List[PlanNode]:
        """规划'开门'技能"""
        door_name = parameters.get("door", parameters.get("door_id"))
        if not door_name:
            logger.error("缺少door参数")
            return []
        
        # 检查门状态
        check_node = PlanNode(
            id=f"check_{door_name}",
            name="check_door_status",
            action="check_door_status",
            skill=skill_name,
            task=task_name,
            parameters={"door_id": door_name},
            preconditions=[],
            expected_effects=[f"door_status({door_name}).known"]
        )
        
        # 开门
        open_node = PlanNode(
            id=f"open_{door_name}",
            name="open_door",
            action="open_door",
            skill=skill_name,
            task=task_name,
            parameters={"door_id": door_name},
            preconditions=[f"door.state({door_name}) == 'closed'"],
            expected_effects=[f"door.state({door_name}) == 'open'"]
        )
        
        return [check_node, open_node]
    
    def _plan_check_door(
        self,
        parameters: Dict[str, Any],
        task_name: Optional[str],
        skill_name: str
    ) -> List[PlanNode]:
        """规划'检查门'技能"""
        door_name = parameters.get("door", parameters.get("door_id"))
        if not door_name:
            logger.error("缺少door参数")
            return []
        
        check_node = PlanNode(
            id=f"check_{door_name}",
            name="check_door_status",
            action="check_door_status",
            skill=skill_name,
            task=task_name,
            parameters={"door_id": door_name},
            preconditions=[],
            expected_effects=[f"door_status({door_name}).known"]
        )
        
        return [check_node]
    
    def _plan_default_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        task_name: Optional[str]
    ) -> List[PlanNode]:
        """默认技能规划（未知技能）"""
        logger.warning(f"使用默认规划 for {skill_name}")
        
        # 创建一个占位节点
        node = PlanNode(
            id=f"skill_{skill_name}",
            name=skill_name,
            skill=skill_name,
            task=task_name,
            parameters=parameters,
            preconditions=[],
            expected_effects=[]
        )
        
        return [node]
