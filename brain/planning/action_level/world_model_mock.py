"""
WorldModel Mock - 世界模型模拟

Phase 0: 简单实现，返回固定值
用于测试和开发，不依赖真实的世界模型
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Location:
    """位置信息"""
    name: str
    position: Dict[str, float]  # x, y, z
    type: str = "room"  # room, door, object


class WorldModelMock:
    """
    世界模型模拟
    
    Phase 0: 返回固定值，用于测试
    """
    
    def __init__(self):
        """初始化模拟世界模型"""
        # 预定义的位置
        self.locations: Dict[str, Location] = {
            "kitchen": Location(
                name="kitchen",
                position={"x": 5.0, "y": 3.0, "z": 0.0},
                type="room"
            ),
            "living_room": Location(
                name="living_room",
                position={"x": 0.0, "y": 0.0, "z": 0.0},
                type="room"
            ),
            "dining_room": Location(
                name="dining_room",
                position={"x": 3.0, "y": 0.0, "z": 0.0},
                type="room"
            ),
            "table": Location(
                name="table",
                position={"x": 2.0, "y": 0.0, "z": 0.8},
                type="object"
            ),
            "kitchen_door": Location(
                name="kitchen_door",
                position={"x": 4.0, "y": 1.5, "z": 0.0},
                type="door"
            )
        }
        
        # 预定义的物体位置
        self.object_locations: Dict[str, str] = {
            "cup": "kitchen",  # 杯子在厨房
            "water": "kitchen"  # 水在厨房
        }
        
        # 门状态
        self.door_states: Dict[str, str] = {
            "kitchen_door": "closed"  # 默认关闭
        }
        
        # 机器人当前位置
        self.robot_position: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        logger.info("WorldModelMock 初始化完成")
    
    def get_location(self, location_name: str) -> Optional[Location]:
        """
        获取位置信息
        
        Args:
            location_name: 位置名称（如"kitchen"）
            
        Returns:
            Location对象，如果不存在则返回None
        """
        return self.locations.get(location_name)
    
    def get_object_location(self, object_name: str) -> Optional[str]:
        """
        获取物体位置
        
        Args:
            object_name: 物体名称（如"cup"）
            
        Returns:
            位置名称，如果不存在则返回None
        """
        return self.object_locations.get(object_name)
    
    def get_door_state(self, door_name: str) -> Optional[str]:
        """
        获取门状态
        
        Args:
            door_name: 门名称（如"kitchen_door"）
            
        Returns:
            门状态（"open"或"closed"），如果不存在则返回None
        """
        return self.door_states.get(door_name)
    
    def set_door_state(self, door_name: str, state: str):
        """
        设置门状态
        
        Args:
            door_name: 门名称
            state: 状态（"open"或"closed"）
        """
        if door_name in self.door_states:
            self.door_states[door_name] = state
            logger.info(f"门 {door_name} 状态更新为: {state}")
    
    def get_robot_position(self) -> Dict[str, float]:
        """获取机器人当前位置"""
        return self.robot_position.copy()
    
    def set_robot_position(self, position: Dict[str, float]):
        """设置机器人位置"""
        self.robot_position = position.copy()
        logger.debug(f"机器人位置更新: {position}")
    
    def is_object_visible(self, object_name: str) -> bool:
        """
        检查物体是否可见
        
        Phase 0: 简单实现，如果物体在已知位置则可见
        """
        location = self.get_object_location(object_name)
        if not location:
            return False
        
        # 简单判断：如果物体位置已知，则认为可见
        return True
    
    def is_door_open(self, door_name: str) -> bool:
        """检查门是否打开"""
        state = self.get_door_state(door_name)
        return state == "open"
    
    def is_at_location(self, location_name: str, tolerance: float = 1.0) -> bool:
        """
        检查机器人是否在指定位置
        
        Args:
            location_name: 位置名称
            tolerance: 容差（米）
        """
        location = self.get_location(location_name)
        if not location:
            return False
        
        loc_pos = location.position
        robot_pos = self.robot_position
        
        # 计算距离
        dx = robot_pos.get("x", 0) - loc_pos.get("x", 0)
        dy = robot_pos.get("y", 0) - loc_pos.get("y", 0)
        dz = robot_pos.get("z", 0) - loc_pos.get("z", 0)
        distance = (dx**2 + dy**2 + dz**2) ** 0.5
        
        return distance <= tolerance
