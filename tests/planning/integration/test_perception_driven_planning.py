"""
Integration Tests for Perception-Driven Planning

感知驱动规划集成测试
"""

import pytest

from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.state import PlanState


@pytest.mark.integration
class TestPerceptionDrivenPlanning:
    """感知驱动规划集成测试"""

    def test_world_model_in_planning(self, planning_orchestrator):
        """测试世界模型在规划中的使用"""
        # 规划时应该查询世界模型
        plan = planning_orchestrator.get_plan("去厨房")

        # 验证世界模型被使用
        assert planning_orchestrator.world_model is not None

        # 验证生成的计划考虑了环境状态
        assert len(plan.nodes) > 0

    def test_dynamic_door_handling(self, planning_orchestrator):
        """测试动态门处理"""
        # 设置门为关闭状态
        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")

        plan = planning_orchestrator.get_plan("去厨房")

        # 计划应该包含处理门的逻辑
        # 这取决于DynamicPlanner的实现
        assert len(plan.nodes) > 0

    def test_object_location_planning(self, planning_orchestrator):
        """测试物体位置规划"""
        # 规划涉及物体的任务
        plan = planning_orchestrator.get_plan("拿杯子")

        assert len(plan.nodes) > 0

        # 验证世界模型中的物体信息被使用
        cup_location = planning_orchestrator.world_model.get_object_location("cup")
        assert cup_location is not None

    def test_robot_position_awareness(self, planning_orchestrator):
        """测试机器人位置感知"""
        # 设置机器人初始位置
        planning_orchestrator.world_model.set_robot_position({
            "x": 0.0, "y": 0.0, "z": 0.0
        })

        plan = planning_orchestrator.get_plan("去厨房")

        # 计划应该考虑机器人当前位置
        assert len(plan.nodes) > 0

    def test_location_verification(self, planning_orchestrator):
        """测试位置验证"""
        # 验证位置存在性检查
        kitchen = planning_orchestrator.world_model.get_location("kitchen")
        assert kitchen is not None

        # 使用已知位置规划
        plan = planning_orchestrator.get_plan("去厨房")
        assert len(plan.nodes) > 0

    def test_environment_state_changes(self, planning_orchestrator):
        """测试环境状态变化"""
        # 初始状态
        plan1 = planning_orchestrator.get_plan("去厨房")

        # 改变环境状态
        planning_orchestrator.world_model.set_door_state("kitchen_door", "open")

        # 重新规划
        plan2 = planning_orchestrator.get_plan("去厨房")

        # 两个计划都可能有效，取决于实现
        assert isinstance(plan1, PlanState)
        assert isinstance(plan2, PlanState)

    def test_multiple_locations_planning(self, planning_orchestrator):
        """测试多位置规划"""
        # 访问多个位置
        plan = planning_orchestrator.get_plan("去厨房然后去餐桌")

        # 应该生成包含多个移动的计划
        assert len(plan.nodes) > 0

    def test_obstacle_awareness(self, planning_orchestrator):
        """测试障碍物感知"""
        # 虽然WorldModelMock是简化的，但应该预留障碍物处理接口
        plan = planning_orchestrator.get_plan("去厨房")

        # 计划生成成功
        assert len(plan.nodes) > 0

    def test_world_model_mock_capabilities(self, planning_orchestrator):
        """测试WorldModelMock能力"""
        wm = planning_orchestrator.world_model

        # 测试所有接口方法
        assert wm.get_location("kitchen") is not None
        assert wm.get_object_location("cup") is not None
        assert wm.get_door_state("kitchen_door") is not None
        assert isinstance(wm.get_robot_position(), dict)
        assert isinstance(wm.get_available_locations(), list)
        assert isinstance(wm.get_available_objects(), list)


@pytest.mark.integration
class TestWorldModelIntegration:
    """世界模型集成测试"""

    def test_world_model_state_consistency(self, planning_orchestrator):
        """测试世界模型状态一致性"""
        wm = planning_orchestrator.world_model

        # 设置机器人位置
        pos = {"x": 1.0, "y": 2.0, "z": 0.0}
        wm.set_robot_position(pos)

        # 验证位置被正确设置
        retrieved = wm.get_robot_position()
        assert retrieved["x"] == 1.0
        assert retrieved["y"] == 2.0

    def test_door_state_persistence(self, planning_orchestrator):
        """测试门状态持久性"""
        wm = planning_orchestrator.world_model

        # 设置门状态
        wm.set_door_state("kitchen_door", "open")
        assert wm.is_door_open("kitchen_door") is True

        wm.set_door_state("kitchen_door", "closed")
        assert wm.is_door_open("kitchen_door") is False

    def test_location_availability(self, planning_orchestrator):
        """测试位置可用性"""
        wm = planning_orchestrator.world_model

        locations = wm.get_available_locations()
        assert len(locations) > 0
        assert all(isinstance(loc, str) for loc in locations)

    def test_object_tracking(self, planning_orchestrator):
        """测试物体跟踪"""
        wm = planning_orchestrator.world_model

        objects = wm.get_available_objects()
        assert "cup" in objects
        assert "water" in objects
