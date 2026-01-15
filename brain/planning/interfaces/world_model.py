"""
IWorldModel Interface

世界模型接口定义

提供规划层与世界模型之间的抽象接口，实现解耦
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from brain.planning.models import Location, Door
from brain.cognitive.world_model.planning_context import PlanningContext


class IWorldModel(ABC):
    """
    世界模型接口

    定义世界模型的抽象接口，规划层通过此接口访问世界状态
    支持多种实现：WorldModelMock（测试用）、WorldModel（真实环境）

    所有方法都是抽象的，具体实现由子类提供
    """

    @abstractmethod
    def get_location(self, location_name: str) -> Optional[Location]:
        """
        获取位置信息

        Args:
            location_name: 位置名称

        Returns:
            Location对象，如果不存在返回None
        """
        pass

    @abstractmethod
    def get_object_location(self, object_name: str) -> Optional[str]:
        """
        获取物体所在位置

        Args:
            object_name: 物体名称

        Returns:
            位置名称，如果未找到返回None
        """
        pass

    @abstractmethod
    def get_door_state(self, door_name: str) -> Optional[str]:
        """
        获取门的状态

        Args:
            door_name: 门名称

        Returns:
            门状态 ('open' 或 'closed')，如果不存在返回None
        """
        pass

    @abstractmethod
    def set_door_state(self, door_name: str, state: str):
        """
        设置门的状态

        Args:
            door_name: 门名称
            state: 门状态 ('open' 或 'closed')
        """
        pass

    @abstractmethod
    def get_robot_position(self) -> Dict[str, float]:
        """
        获取机器人位置

        Returns:
            位置坐标 {x, y, z, yaw}
        """
        pass

    @abstractmethod
    def set_robot_position(self, position: Dict[str, float]):
        """
        设置机器人位置

        Args:
            position: 位置坐标 {x, y, z, yaw}
        """
        pass

    @abstractmethod
    def is_object_visible(self, object_name: str) -> bool:
        """
        检查物体是否可见

        Args:
            object_name: 物体名称

        Returns:
            是否可见
        """
        pass

    @abstractmethod
    def is_door_open(self, door_name: str) -> bool:
        """
        检查门是否打开

        Args:
            door_name: 门名称

        Returns:
            是否打开
        """
        pass

    @abstractmethod
    def is_at_location(self, location_name: str, tolerance: float = 1.0) -> bool:
        """
        检查机器人是否在指定位置

        Args:
            location_name: 位置名称
            tolerance: 容差距离（米）

        Returns:
            是否在位置
        """
        pass

    @abstractmethod
    def get_available_locations(self) -> List[str]:
        """
        获取所有可用位置列表

        Returns:
            位置名称列表
        """
        pass

    @abstractmethod
    def get_available_objects(self) -> List[str]:
        """
        获取所有已知物体列表

        Returns:
            物体名称列表
        """
        pass

    @abstractmethod
    def get_semantic_objects(
        self,
        category: Optional[str] = None,
        position_filter: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        查询语义物体（直接访问原始数据）

        Args:
            category: 可选的类别过滤（如"door", "chair"）
            position_filter: 可选的位置过滤 {"x": 0, "y": 0, "radius": 5.0}

        Returns:
            语义物体列表，每个物体包含 id, label, position, confidence, state, attributes
        """
        pass

    @abstractmethod
    def get_fused_context(self) -> PlanningContext:
        """
        获取融合后的规划上下文

        Returns:
            包含三模态数据的PlanningContext：
            - semantic_objects: 语义物体列表
            - causal_graph: 因果关系数据
            - state_predictions: 状态预测
        """
        pass

    def get_object_affordances(self, object_id: str) -> List[str]:
        """
        获取物体可执行的操作（功能推理）

        Args:
            object_id: 物体ID

        Returns:
            可执行的操作列表（如["open", "close", "pass_through"]）
        """
        # 默认实现：返回通用操作
        return ["approach", "observe", "point_at"]
