"""
Unit Tests for ActionLevelPlanner

ActionLevelPlanner 单元测试
"""

import pytest

from brain.planning.planners import ActionLevelPlanner
from brain.planning.state import NodeStatus


@pytest.mark.unit
class TestActionLevelPlanner:
    """ActionLevelPlanner 测试类"""

    def test_planner_initialization(self, action_level_planner):
        """测试规划器初始化"""
        assert action_level_planner is not None
        assert action_level_planner.platform == "ugv"

    def test_plan_go_to_location(self, action_level_planner):
        """测试规划'去厨房'技能"""
        nodes = action_level_planner.plan_skill(
            skill_name="去厨房",
            parameters={"location": "kitchen"},
            task_name="test_task"
        )

        assert len(nodes) > 0
        # 验证生成的节点包含基本操作
        actions = [node.action for node in nodes if node.action]
        assert "goto" in actions or "move_to" in actions

    def test_plan_grasp_object(self, action_level_planner):
        """测试规划'拿杯子'技能"""
        nodes = action_level_planner.plan_skill(
            skill_name="拿杯子",
            parameters={"object": "cup"},
            task_name="test_task"
        )

        assert len(nodes) > 0
        # 验证包含抓取操作
        actions = [node.action for node in nodes if node.action]
        assert "grasp" in actions or "pickup" in actions

    def test_plan_place_object(self, action_level_planner):
        """测试规划'放物体'技能"""
        nodes = action_level_planner.plan_skill(
            skill_name="放物体",
            parameters={"location": "table", "object": "cup"},
            task_name="test_task"
        )

        assert len(nodes) > 0
        # 验证包含放置操作
        actions = [node.action for node in nodes if node.action]
        assert "place" in actions or "dropoff" in actions or "put" in actions

    def test_plan_return(self, action_level_planner):
        """测试规划'回来'技能"""
        nodes = action_level_planner.plan_skill(
            skill_name="回来",
            parameters={},
            task_name="test_task"
        )

        assert len(nodes) > 0

    def test_plan_unknown_skill(self, action_level_planner):
        """测试规划未知技能"""
        nodes = action_level_planner.plan_skill(
            skill_name="unknown_skill",
            parameters={},
            task_name="test_task"
        )

        # 未知技能应该返回空列表或使用默认规划器
        assert isinstance(nodes, list)

    def test_generated_nodes_have_valid_structure(self, action_level_planner):
        """测试生成的节点具有有效结构"""
        nodes = action_level_planner.plan_skill(
            skill_name="去厨房",
            parameters={"location": "kitchen"},
            task_name="test_task"
        )

        for node in nodes:
            assert node.id is not None
            assert node.name is not None
            assert node.task == "test_task"
            assert node.status == NodeStatus.PENDING
            # 至少有一个层级标识（task, skill, 或 action）
            assert node.task or node.skill or node.action

    def test_door_handling(self, action_level_planner, world_model_mock):
        """测试门处理逻辑"""
        # 设置门为关闭状态
        world_model_mock.set_door_state("kitchen_door", "closed")

        nodes = action_level_planner.plan_skill(
            skill_name="去厨房",
            parameters={"location": "kitchen"},
            task_name="test_task"
        )

        # 验证可能包含开门或检查门的操作
        actions = [node.action for node in nodes if node.action]
        # 这个测试依赖于具体的实现逻辑

    def test_planner_uses_capability_registry(self, action_level_planner):
        """测试规划器使用能力注册表"""
        assert action_level_planner.capability_registry is not None
        assert action_level_planner.platform_adapter is not None

    def test_planner_uses_world_model(self, action_level_planner):
        """测试规划器使用世界模型"""
        assert action_level_planner.world_model is not None
        assert hasattr(action_level_planner.world_model, 'get_location')
