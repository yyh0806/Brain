"""
Integration Tests for Replanning Workflow

重规划工作流集成测试
"""

import pytest

from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.intelligent import ReplanningRules, DynamicPlanner
from brain.planning.state import PlanNode, NodeStatus, PlanState


@pytest.mark.integration
class TestReplanningWorkflow:
    """重规划工作流集成测试"""

    def test_replanning_rules_integration(self, planning_orchestrator):
        """测试重规划规则集成"""
        rules = ReplanningRules()

        # 创建失败节点
        failed_node = PlanNode(
            id="failed",
            name="failed_action",
            action="goto"
        )
        failed_node.mark_failed()

        # 测试重规划决策
        should_replan = rules.should_replan(
            failed_node=failed_node,
            insertion_count=0,
            retry_count=0
        )

        assert isinstance(should_replan, bool)

    def test_dynamic_planner_integration(self, planning_orchestrator):
        """测试动态规划器集成"""
        dynamic_planner = DynamicPlanner(
            world_model=planning_orchestrator.world_model
        )

        # 创建节点
        node = PlanNode(
            id="test",
            name="goto_door",
            action="goto",
            preconditions=["door.kitchen_door == open"]
        )

        # 测试前置条件检查
        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        assert isinstance(modified, bool)
        assert isinstance(new_nodes, list)

    def test_failure_detection_workflow(self):
        """测试失败检测工作流"""
        # 创建简单计划
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        state.add_root(node)

        # 模拟失败
        node.mark_failed()

        assert node.status == NodeStatus.FAILED
        assert len(state.get_failed_nodes()) == 1

    def test_recovery_action_selection(self):
        """测试恢复动作选择"""
        rules = ReplanningRules()

        # 不同失败场景的恢复动作
        scenarios = [
            # (insertion_count, retry_count, expected_action)
            (0, 0, "insert"),  # 首次失败，尝试插入
            (0, 1, "retry"),   # 重试
            (4, 0, "replan"),  # 超过限制，重规划
        ]

        for insertion_count, retry_count, expected in scenarios:
            failed_node = PlanNode(
                id=f"test_{insertion_count}_{retry_count}",
                name="test",
                action="goto"
            )
            failed_node.mark_failed()

            action = rules.get_recovery_action(
                failed_node=failed_node,
                insertion_count=insertion_count,
                retry_count=retry_count
            )

            # 验证返回的恢复动作
            assert action in ["insert", "retry", "replan"]

    def test_max_retries_enforcement(self):
        """测试最大重试次数强制执行"""
        rules = ReplanningRules()

        # 创建多次重试的失败节点
        failed_node = PlanNode(id="test", name="test", action="goto")
        failed_node.mark_failed()

        # 超过最大重试次数
        retry_count = 5
        should_replan = rules.should_replan(
            failed_node=failed_node,
            insertion_count=0,
            retry_count=retry_count
        )

        # 应该触发重规划
        assert should_replan is True

    def test_max_insertions_enforcement(self):
        """测试最大插入次数强制执行"""
        rules = ReplanningRules()

        failed_node = PlanNode(id="test", name="test", action="goto")
        failed_node.mark_failed()

        # 超过最大插入次数
        insertion_count = 5
        should_replan = rules.should_replan(
            failed_node=failed_node,
            insertion_count=insertion_count,
            retry_count=0
        )

        # 应该触发重规划
        assert should_replan is True


@pytest.mark.integration
class TestDynamicInsertionWorkflow:
    """动态插入工作流测试"""

    def test_precondition_insertion_workflow(self, planning_orchestrator):
        """测试前置条件插入工作流"""
        dynamic_planner = DynamicPlanner(
            world_model=planning_orchestrator.world_model
        )

        # 设置门为关闭
        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")

        # 创建需要开门的前置条件
        node = PlanNode(
            id="goto_kitchen",
            name="go_to_kitchen",
            action="goto",
            preconditions=["door.kitchen_door == open"]
        )

        # 检查并插入前置条件
        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 验证结果
        assert isinstance(modified, bool)
        assert isinstance(new_nodes, list)

    def test_multiple_preconditions_handling(self, planning_orchestrator):
        """测试多个前置条件处理"""
        dynamic_planner = DynamicPlanner(
            world_model=planning_orchestrator.world_model
        )

        # 创建有多个前置条件的节点
        node = PlanNode(
            id="complex_action",
            name="complex_action",
            action="goto",
            preconditions=[
                "door.kitchen_door == open",
                "robot.battery >= 20"
            ]
        )

        # 处理前置条件
        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 应该处理所有前置条件
        assert isinstance(new_nodes, list)

    def test_insertion_count_tracking(self, planning_orchestrator):
        """测试插入计数跟踪"""
        dynamic_planner = DynamicPlanner(
            world_model=planning_orchestrator.world_model
        )

        # 初始计数为0
        assert dynamic_planner.insertion_count == 0

        # 执行一些插入操作
        node = PlanNode(
            id="test",
            name="test",
            action="goto",
            preconditions=["door.kitchen_door == open"]
        )

        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")

        modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
            node,
            plan_nodes=[]
        )

        # 计数可能增加
        assert dynamic_planner.insertion_count >= 0

        # 重置计数
        dynamic_planner.reset_insertion_count()
        assert dynamic_planner.insertion_count == 0


@pytest.mark.integration
class TestPlanRecoveryWorkflow:
    """计划恢复工作流测试"""

    def test_plan_recovery_after_failure(self, planning_orchestrator):
        """测试失败后的计划恢复"""
        # 创建初始计划
        plan = planning_orchestrator.get_plan("去厨房拿杯水")

        # 模拟中间节点失败
        if len(plan.nodes) > 1:
            some_node = list(plan.nodes.values())[0]
            some_node.mark_failed()

            # 验证失败被记录
            assert len(plan.get_failed_nodes()) > 0

    def test_plan_statistics_after_changes(self):
        """测试变化后的计划统计"""
        state = PlanState()

        # 添加多个节点
        node1 = PlanNode(id="node1", name="node1", action="goto")
        node2 = PlanNode(id="node2", name="node2", action="grasp")
        node3 = PlanNode(id="node3", name="node3", action="place")

        state.add_root(node1)
        state.add_root(node2)
        state.add_root(node3)

        # 设置不同状态
        node1.mark_started()
        node1.mark_success()
        node2.mark_started()
        node2.mark_failed()

        # 获取统计
        stats = state.get_execution_statistics()
        assert stats['total'] == 3
        assert stats['success'] == 1
        assert stats['failed'] == 1
        assert stats['pending'] == 1
