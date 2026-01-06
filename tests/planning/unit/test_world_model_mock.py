"""
Unit Tests for WorldModelMock

WorldModelMock 单元测试
"""

import pytest

from brain.planning.action_level import WorldModelMock
from brain.planning.interfaces import IWorldModel


@pytest.mark.unit
class TestWorldModelMock:
    """WorldModelMock 测试类"""

    def test_world_model_implements_interface(self, world_model_mock):
        """测试WorldModelMock实现IWorldModel接口"""
        assert isinstance(world_model_mock, IWorldModel)

    def test_initialization(self, world_model_mock):
        """测试初始化"""
        assert world_model_mock is not None
        assert len(world_model_mock.locations) > 0
        assert len(world_model_mock.object_locations) > 0

    def test_get_location(self, world_model_mock):
        """测试获取位置"""
        kitchen = world_model_mock.get_location("kitchen")
        assert kitchen is not None
        assert kitchen.name == "kitchen"
        assert kitchen.position["x"] == 5.0
        assert kitchen.position["y"] == 3.0

    def test_get_location_not_found(self, world_model_mock):
        """测试获取不存在的位置"""
        result = world_model_mock.get_location("nonexistent")
        assert result is None

    def test_get_object_location(self, world_model_mock):
        """测试获取物体位置"""
        cup_location = world_model_mock.get_object_location("cup")
        assert cup_location == "kitchen"

    def test_get_object_location_not_found(self, world_model_mock):
        """测试获取不存在的物体位置"""
        result = world_model_mock.get_object_location("nonexistent")
        assert result is None

    def test_get_door_state(self, world_model_mock):
        """测试获取门状态"""
        door_state = world_model_mock.get_door_state("kitchen_door")
        assert door_state == "closed"  # 默认关闭

    def test_set_door_state(self, world_model_mock):
        """测试设置门状态"""
        world_model_mock.set_door_state("kitchen_door", "open")
        assert world_model_mock.get_door_state("kitchen_door") == "open"

        world_model_mock.set_door_state("kitchen_door", "closed")
        assert world_model_mock.get_door_state("kitchen_door") == "closed"

    def test_get_robot_position(self, world_model_mock):
        """测试获取机器人位置"""
        position = world_model_mock.get_robot_position()
        assert isinstance(position, dict)
        assert "x" in position
        assert "y" in position
        assert "z" in position

    def test_set_robot_position(self, world_model_mock):
        """测试设置机器人位置"""
        new_position = {"x": 10.0, "y": 20.0, "z": 0.0}
        world_model_mock.set_robot_position(new_position)

        retrieved = world_model_mock.get_robot_position()
        assert retrieved["x"] == 10.0
        assert retrieved["y"] == 20.0

    def test_is_object_visible(self, world_model_mock):
        """测试检查物体可见性"""
        # 已知物体应该可见
        assert world_model_mock.is_object_visible("cup") is True

        # 未知物体不可见
        assert world_model_mock.is_object_visible("nonexistent") is False

    def test_is_door_open(self, world_model_mock):
        """测试检查门是否打开"""
        # 默认关闭
        assert world_model_mock.is_door_open("kitchen_door") is False

        # 打开门
        world_model_mock.set_door_state("kitchen_door", "open")
        assert world_model_mock.is_door_open("kitchen_door") is True

    def test_is_at_location(self, world_model_mock):
        """测试检查机器人是否在指定位置"""
        # 机器人初始在 (0, 0, 0)
        assert world_model_mock.is_at_location("living_room") is True

        # 不在其他位置
        assert world_model_mock.is_at_location("kitchen") is False

    def test_is_at_location_with_tolerance(self, world_model_mock):
        """测试带容差的位置检查"""
        # 移动机器人接近厨房
        near_kitchen = {"x": 4.5, "y": 3.2, "z": 0.0}
        world_model_mock.set_robot_position(near_kitchen)

        # 在默认容差内
        assert world_model_mock.is_at_location("kitchen", tolerance=1.0) is True

        # 不在更小的容差内
        assert world_model_mock.is_at_location("kitchen", tolerance=0.1) is False

    def test_get_available_locations(self, world_model_mock):
        """测试获取所有可用位置"""
        locations = world_model_mock.get_available_locations()
        assert isinstance(locations, list)
        assert "kitchen" in locations
        assert "living_room" in locations
        assert "table" in locations

    def test_get_available_objects(self, world_model_mock):
        """测试获取所有已知物体"""
        objects = world_model_mock.get_available_objects()
        assert isinstance(objects, list)
        assert "cup" in objects
        assert "water" in objects

    def test_predefined_locations(self, world_model_mock):
        """测试预定义位置"""
        # kitchen
        kitchen = world_model_mock.get_location("kitchen")
        assert kitchen is not None
        assert kitchen.type == "room"

        # living_room
        living_room = world_model_mock.get_location("living_room")
        assert living_room is not None

        # table
        table = world_model_mock.get_location("table")
        assert table is not None
        assert table.type == "object"

    def test_predefined_objects(self, world_model_mock):
        """测试预定义物体"""
        assert world_model_mock.get_object_location("cup") == "kitchen"
        assert world_model_mock.get_object_location("water") == "kitchen"
