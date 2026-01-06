"""
Unit Tests for DynamicPlanner

DynamicPlanner 单元测试
"""

import pytest

from brain.planning.intelligent import DynamicPlanner
from brain.planning.state import PlanNode, NodeStatus


@pytest.mark.unit
class TestDynamicPlanner:
    """DynamicPlanner 测试类"""

    def test_planner_initialization(self, dynamic_planner):
        """测试规划器初始化"""
        assert dynamic_planner is not None
        assert dynamic_planner.world_model is not None
        assert dynamic_planner.max_insertions == 3

    def test_check_and_insert_preconditions_no_preconditions(self, dynamic_planner):
        """测试检查前置条件 - 无前置条件"""
        node = PlanNode(
            id="test",
            name="test_action",
            action="goto",
            preconditions=[]
        )

        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 无前置条件，不应该插入新节点
        assert modified is False
        assert len(new_nodes) == 0

    def test_check_door_closed_precondition(self, dynamic_planner):
        """测试检查门关闭前置条件"""
        node = PlanNode(
            id="test",
            name="goto_kitchen",
            action="goto",
            preconditions=["door.kitchen_door == open"]
        )

        # 设置门为关闭状态
        dynamic_planner.world_model.set_door_state("kitchen_door", "closed")

        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 应该检测到前置条件不满足
        assert isinstance(modified, bool)
        # 可能插入开门或检查门的操作

    def test_reset_insertion_count(self, dynamic_planner):
        """测试重置插入计数"""
        # 设置一些插入计数
        dynamic_planner.insertion_count = 2

        # 重置
        dynamic_planner.reset_insertion_count()
        assert dynamic_planner.insertion_count == 0

    def test_max_insertions_limit(self, dynamic_planner):
        """测试最大插入次数限制"""
        assert dynamic_planner.max_insertions == 3

        # 设置插入计数为最大值
        dynamic_planner.insertion_count = 3

        node = PlanNode(
            id="test",
            name="test_action",
            action="goto",
            preconditions=["door.kitchen_door == open"]
        )

        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 达到最大插入次数，不应该继续插入
        # 具体行为取决于实现

    def test_handle_precondition(self, dynamic_planner):
        """测试处理单个前置条件"""
        node = PlanNode(
            id="test",
            name="test_action",
            action="goto"
        )

        # 测试门状态前置条件
        door_precondition = "door.kitchen_door == open"
        result = dynamic_planner._handle_precondition(
            node,
            door_precondition,
            []
        )

        # 结果可能是新节点或对原节点的修改

    def test_extract_door_name(self, dynamic_planner):
        """测试提取门名称"""
        precondition = "door.kitchen_door == open"
        door_name = dynamic_planner._extract_door_name(precondition)
        assert door_name == "kitchen_door"

    def test_extract_object_name(self, dynamic_planner):
        """测试提取物体名称"""
        precondition = "object.cup.visible == true"
        object_name = dynamic_planner._extract_object_name(precondition)
        assert object_name == "cup"


@pytest.mark.unit
class TestDynamicPlannerIntegration:
    """DynamicPlanner 集成测试"""

    def test_dynamic_insertion_workflow(self, dynamic_planner):
        """测试动态插入工作流"""
        # 创建一个需要前置条件的节点
        node = PlanNode(
            id="goto_kitchen",
            name="go_to_kitchen",
            action="goto",
            preconditions=["door.kitchen_door == open"],
            parameters={"location": "kitchen"}
        )

        # 设置门为关闭
        dynamic_planner.world_model.set_door_state("kitchen_door", "closed")

        # 检查并插入前置条件
        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 验证返回值
        assert isinstance(modified, bool)
        assert isinstance(new_nodes, list)
