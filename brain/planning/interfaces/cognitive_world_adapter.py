# -*- coding: utf-8 -*-
"""
认知世界适配器 - Cognitive World Adapter

将认知层的输出（PlanningContext、Belief等）转换为规划层需要的 IWorldModel 接口。
实现认知层（L3）到规划层（L4）的数据适配。
"""

from typing import Dict, List, Any, Optional
from loguru import logger
import math

from brain.planning.interfaces.world_model import IWorldModel, Location
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief


class CognitiveWorldAdapter(IWorldModel):
    """
    认知世界适配器

    将认知层的 PlanningContext 和 Belief 集合转换为 IWorldModel 接口。
    作为认知层和规划层之间的适配器，实现解耦。

    架构说明：
    - 认知层（L3）维护关于世界的信念和推理结果
    - 规划层（L4）需要世界状态信息来进行规划
    - 本适配器作为中间层，将认知层的认知状态转换为规划层可用的接口
    """

    def __init__(
        self,
        planning_context: Optional[PlanningContext] = None,
        beliefs: Optional[List[Belief]] = None
    ):
        """
        初始化认知世界适配器

        Args:
            planning_context: 认知层提供的规划上下文
            beliefs: 认知层维护的信念集合
        """
        self._planning_context = planning_context
        self._beliefs = beliefs or []

        # 内部状态
        self._robot_position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._robot_yaw: float = 0.0

        # 初始化机器人位置
        if planning_context and planning_context.current_position:
            self._robot_position = planning_context.current_position.copy()
            self._robot_yaw = planning_context.current_heading

        logger.info("CognitiveWorldAdapter 初始化完成")

    def update_context(
        self,
        planning_context: PlanningContext,
        beliefs: Optional[List[Belief]] = None
    ):
        """
        更新认知上下文

        当认知层的认知状态更新时调用此方法。

        Args:
            planning_context: 新的规划上下文
            beliefs: 更新的信念集合（可选）
        """
        self._planning_context = planning_context
        if beliefs is not None:
            self._beliefs = beliefs

        # 更新机器人位置
        if planning_context.current_position:
            self._robot_position = planning_context.current_position.copy()
        if planning_context.current_heading is not None:
            self._robot_yaw = planning_context.current_heading

        logger.debug("CognitiveWorldAdapter 上下文已更新")

    # ============ IWorldModel 接口实现 ============

    def get_location(self, location_name: str) -> Optional[Location]:
        """
        获取位置信息

        从 PlanningContext 的 points_of_interest 中查找位置。
        优先查找兴趣点，其次查找障碍物和目标。

        Args:
            location_name: 位置名称

        Returns:
            Location对象，如果不存在返回None
        """
        if not self._planning_context:
            return None

        # 在兴趣点中查找
        for poi in self._planning_context.points_of_interest:
            if poi.get("name") == location_name or poi.get("type") == location_name:
                pos = poi.get("position", {})
                return Location(
                    name=location_name,
                    position={
                        "x": pos.get("x", 0.0),
                        "y": pos.get("y", 0.0),
                        "z": pos.get("z", 0.0)
                    },
                    type=poi.get("type", "unknown")
                )

        # 在目标中查找
        for target in self._planning_context.targets:
            if target.get("name") == location_name or target.get("type") == location_name:
                pos = target.get("position", {})
                return Location(
                    name=location_name,
                    position={
                        "x": pos.get("x", 0.0),
                        "y": pos.get("y", 0.0),
                        "z": pos.get("z", 0.0)
                    },
                    type="target"
                )

        return None

    def get_object_location(self, object_name: str) -> Optional[str]:
        """
        获取物体所在位置

        从信念集合中查找物体位置信念。

        Args:
            object_name: 物体名称

        Returns:
            位置名称，如果未找到返回None
        """
        # 在信念中查找
        for belief in self._beliefs:
            if belief.falsified:
                continue
            # 查找形如 "X在Y" 的信念
            if f"{object_name}在" in belief.content or f"{object_name} at" in belief.content.lower():
                # 提取位置名称（简化实现）
                parts = belief.content.split("在")
                if len(parts) > 1:
                    return parts[1].strip().rstrip("。").strip()

        # 在兴趣点中查找
        if self._planning_context:
            for poi in self._planning_context.points_of_interest:
                if poi.get("name") == object_name:
                    # 返回位置描述
                    pos = poi.get("position", {})
                    return f"({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f})"

        return None

    def get_door_state(self, door_name: str) -> Optional[str]:
        """
        获取门的状态

        从信念集合中查找门状态信念。

        Args:
            door_name: 门名称

        Returns:
            门状态 ('open' 或 'closed')，如果不存在返回None
        """
        for belief in self._beliefs:
            if belief.falsified:
                continue
            if door_name in belief.content and ("门" in belief.content or "door" in belief.content.lower()):
                if "开" in belief.content or "open" in belief.content.lower():
                    return "open"
                elif "关" in belief.content or "closed" in belief.content.lower():
                    return "closed"

        return None

    def set_door_state(self, door_name: str, state: str):
        """
        设置门的状态

        注意：这是一个虚拟设置，实际状态由认知层管理。
        此方法主要用于与现有接口兼容。

        Args:
            door_name: 门名称
            state: 门状态 ('open' 或 'closed')
        """
        logger.debug(f"设置门状态（虚拟）: {door_name} -> {state}")
        # 实际实现中，这里应该通知认知层更新信念
        # 当前为简化实现，仅记录日志

    def get_robot_position(self) -> Dict[str, float]:
        """
        获取机器人位置

        从 PlanningContext 获取当前机器人位置。

        Returns:
            位置坐标 {x, y, z, yaw}
        """
        pos = self._robot_position.copy()
        pos["yaw"] = self._robot_yaw
        return pos

    def set_robot_position(self, position: Dict[str, float]):
        """
        设置机器人位置

        注意：这是一个虚拟设置，实际位置由认知层管理。

        Args:
            position: 位置坐标 {x, y, z, yaw}
        """
        self._robot_position = {
            "x": position.get("x", 0.0),
            "y": position.get("y", 0.0),
            "z": position.get("z", 0.0)
        }
        self._robot_yaw = position.get("yaw", 0.0)
        logger.debug(f"设置机器人位置（虚拟）: {position}")

    def is_object_visible(self, object_name: str) -> bool:
        """
        检查物体是否可见

        基于障碍物和目标信息判断物体是否可见。

        Args:
            object_name: 物体名称

        Returns:
            是否可见
        """
        if not self._planning_context:
            return False

        # 检查是否在兴趣点中
        for poi in self._planning_context.points_of_interest:
            if poi.get("name") == object_name:
                # 简化判断：如果兴趣点存在，则认为可见
                return True

        # 检查是否在目标中
        for target in self._planning_context.targets:
            if target.get("name") == object_name or target.get("type") == object_name:
                return True

        return False

    def is_door_open(self, door_name: str) -> bool:
        """
        检查门是否打开

        Args:
            door_name: 门名称

        Returns:
            是否打开
        """
        state = self.get_door_state(door_name)
        return state == "open"

    def is_at_location(self, location_name: str, tolerance: float = 1.0) -> bool:
        """
        检查机器人是否在指定位置

        Args:
            location_name: 位置名称
            tolerance: 容差距离（米）

        Returns:
            是否在位置
        """
        location = self.get_location(location_name)
        if not location:
            return False

        loc_pos = location.position
        robot_pos = self._robot_position

        # 计算距离
        dx = robot_pos.get("x", 0) - loc_pos.get("x", 0)
        dy = robot_pos.get("y", 0) - loc_pos.get("y", 0)
        dz = robot_pos.get("z", 0) - loc_pos.get("z", 0)
        distance = (dx**2 + dy**2 + dz**2) ** 0.5

        return distance <= tolerance

    def get_available_locations(self) -> List[str]:
        """
        获取所有可用位置列表

        Returns:
            位置名称列表
        """
        locations = set()

        if self._planning_context:
            # 从兴趣点收集
            for poi in self._planning_context.points_of_interest:
                name = poi.get("name")
                if name:
                    locations.add(name)

            # 从目标收集
            for target in self._planning_context.targets:
                name = target.get("name")
                if name:
                    locations.add(name)
                type_name = target.get("type")
                if type_name:
                    locations.add(type_name)

        return list(locations)

    def get_available_objects(self) -> List[str]:
        """
        获取所有已知物体列表

        Returns:
            物体名称列表
        """
        objects = set()

        if self._planning_context:
            # 从兴趣点收集
            for poi in self._planning_context.points_of_interest:
                if poi.get("type") == "object":
                    name = poi.get("name")
                    if name:
                        objects.add(name)

            # 从目标收集
            for target in self._planning_context.targets:
                name = target.get("name")
                if name:
                    objects.add(name)

        # 从信念中收集
        for belief in self._beliefs:
            if not belief.falsified:
                # 简单提取物体名称（如"杯子在厨房"中的"杯子"）
                if "在" in belief.content:
                    obj_name = belief.content.split("在")[0].strip()
                    if obj_name:
                        objects.add(obj_name)

        return list(objects)

    # ============ 扩展方法 ============

    def get_obstacles(self) -> List[Dict[str, Any]]:
        """获取所有障碍物"""
        if self._planning_context:
            return self._planning_context.obstacles
        return []

    def get_targets(self) -> List[Dict[str, Any]]:
        """获取所有目标"""
        if self._planning_context:
            return self._planning_context.targets
        return []

    def get_constraints(self) -> List[str]:
        """获取约束条件"""
        if self._planning_context:
            return self._planning_context.constraints
        return []

    def get_battery_level(self) -> float:
        """获取电池电量"""
        if self._planning_context:
            return self._planning_context.battery_level
        return 100.0

    def get_recent_changes(self) -> List[Dict[str, Any]]:
        """获取最近的环境变化"""
        if self._planning_context:
            return self._planning_context.recent_changes
        return []

    # ============ 三模态融合接口 ============

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
            语义物体列表
        """
        if not self._planning_context:
            return []

        objects = self._planning_context.semantic_objects

        # 按类别过滤
        if category:
            objects = [obj for obj in objects
                      if category.lower() in obj.get("label", "").lower()]

        # 按位置过滤
        if position_filter:
            x = position_filter.get("x", 0)
            y = position_filter.get("y", 0)
            radius = position_filter.get("radius", 5.0)

            objects = [obj for obj in objects
                      if self._is_near_position(obj, x, y, radius)]

        return objects

    def get_fused_context(self) -> PlanningContext:
        """
        获取融合后的规划上下文

        Returns:
            包含三模态数据的PlanningContext
        """
        if not self._planning_context:
            # 返回一个空的PlanningContext
            return PlanningContext(
                current_position=self._robot_position,
                current_heading=self._robot_yaw,
                obstacles=[],
                targets=[],
                points_of_interest=[],
                weather={},
                battery_level=100.0,
                signal_strength=100.0,
                available_paths=[],
                constraints=[],
                recent_changes=[],
                risk_areas=[],
                semantic_objects=[],
                causal_graph={},
                state_predictions=[]
            )

        return self._planning_context

    def get_object_affordances(self, object_id: str) -> List[str]:
        """
        获取物体可执行的操作（功能推理）

        根据物体类型推断可执行的操作。这是实现物体功能推理的关键方法。

        Args:
            object_id: 物体ID

        Returns:
            可执行的操作列表（如["open", "close", "pass_through"]）
        """
        # 在语义物体中查找
        if self._planning_context and self._planning_context.semantic_objects:
            for obj in self._planning_context.semantic_objects:
                if obj.get("id") == object_id:
                    label = obj.get("label", "").lower()

                    # 功能推理映射表
                    affordances_map = {
                        "door": ["open", "close", "lock", "unlock", "pass_through", "approach"],
                        "button": ["press", "hold", "approach"],
                        "cup": ["pick_up", "place", "pour", "drink_from", "approach"],
                        "table": ["place_on", "sit_at", "work_on", "approach"],
                        "chair": ["sit_on", "move", "stand_on", "approach"],
                        "car": ["enter", "exit", "drive", "approach"],
                        "box": ["open", "close", "pick_up", "carry", "approach"],
                        "screen": ["touch", "read", "interact", "approach"],
                        "switch": ["flip", "press", "toggle", "approach"],
                        "person": ["approach", "communicate", "follow", "observe"],
                        "building": ["enter", "exit", "approach"],
                        "wall": ["approach", "observe"],
                        "obstacle": ["avoid", "approach", "observe"]
                    }

                    # 查找匹配的功能
                    for key, actions in affordances_map.items():
                        if key in label:
                            return actions

                    # 默认操作
                    return ["approach", "observe", "point_at"]

        # 未找到物体，返回默认操作
        return ["approach", "observe", "point_at"]

    # ============ 私有辅助方法 ============

    def _is_near_position(
        self,
        obj: Dict[str, Any],
        x: float,
        y: float,
        radius: float
    ) -> bool:
        """检查物体是否在指定位置附近"""
        pos = obj.get("position", ())
        if len(pos) >= 2:
            distance = math.sqrt((pos[0] - x)**2 + (pos[1] - y)**2)
            return distance <= radius
        return False
