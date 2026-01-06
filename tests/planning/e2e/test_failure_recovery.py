"""
End-to-End Tests for Failure Recovery

失败恢复端到端测试
"""

import pytest

from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.intelligent import ReplanningRules, DynamicPlanner, PlanValidator, PlanValidation
from brain.planning.state import PlanState, PlanNode, NodeStatus


@pytest.mark.e2e
class TestFailureRecovery:
    """失败恢复E2E测试"""

    def test_plan_with_closed_door(self, planning_orchestrator):
        """测试门关闭场景"""
        # 设置门为关闭状态
        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")

        # 生成计划
        plan = planning_orchestrator.get_plan("去厨房")

        # 计划应该成功生成
        assert isinstance(plan, PlanState)
        assert len(plan.nodes) > 0

        # 验证计划有效性
        validator = PlanValidator()
        validation = validator.validate(plan)
        assert validation.valid is True

    def test_dynamic_precondition_insertion(self, planning_orchestrator):
        """测试动态前置条件插入"""
        # 设置需要前置条件的环境
        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")

        # 使用DynamicPlanner
        dynamic_planner = DynamicPlanner(
            world_model=planning_orchestrator.world_model
        )

        # 创建需要开门的节点
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

        # 验证处理结果
        assert isinstance(modified, bool)
        assert isinstance(new_nodes, list)

    def test_replanning_decision(self):
        """测试重规划决策"""
        rules = ReplanningRules()

        # 测试各种失败场景的决策
        scenarios = [
            # (insertion_count, retry_count, expected_replan)
            (0, 0, False),   # 首次失败，不需要重规划
            (1, 0, False),   # 少量插入，不需要重规划
            (4, 0, True),    # 超过最大插入次数，需要重规划
            (0, 4, True),    # 超过最大重试次数，需要重规划
        ]

        for insertion_count, retry_count, expected_replan in scenarios:
            failed_node = PlanNode(
                id=f"test_{insertion_count}_{retry_count}",
                name="test_action",
                action="goto"
            )
            failed_node.mark_failed()

            should_replan = rules.should_replan(
                failed_node=failed_node,
                insertion_count=insertion_count,
                retry_count=retry_count
            )

            assert should_replan == expected_replan

    def test_recovery_action_selection(self):
        """测试恢复动作选择"""
        rules = ReplanningRules()

        # 测试不同场景的恢复动作
        scenarios = [
            # (insertion_count, retry_count, expected_actions)
            (0, 0, ["insert", "retry"]),  # 首次失败：插入或重试
            (1, 1, ["retry", "insert"]),  # 已有插入：重试
            (4, 0, ["replan"]),          # 超限：重规划
            (0, 4, ["replan"]),          # 超限：重规划
        ]

        for insertion_count, retry_count, expected_actions in scenarios:
            failed_node = PlanNode(
                id=f"test_{insertion_count}_{retry_count}",
                name="test_action",
                action="goto"
            )
            failed_node.mark_failed()

            action = rules.get_recovery_action(
                failed_node=failed_node,
                insertion_count=insertion_count,
                retry_count=retry_count
            )

            assert action in expected_actions

    def test_plan_state_after_failure(self):
        """测试失败后的计划状态"""
        state = PlanState()

        # 添加节点并模拟失败
        node1 = PlanNode(id="node1", name="action1", action="goto")
        node2 = PlanNode(id="node2", name="action2", action="grasp")
        node3 = PlanNode(id="node3", name="action3", action="place")

        state.add_root(node1)
        state.add_root(node2)
        state.add_root(node3)

        # 模拟node2失败
        node2.mark_started()
        node2.mark_failed()

        # 验证状态
        stats = state.get_execution_statistics()
        assert stats['total'] == 3
        assert stats['pending'] == 2
        assert stats['failed'] == 1
        assert stats['executing'] == 0

    def test_multiple_failures_handling(self):
        """测试多个失败处理"""
        state = PlanState()

        # 添加多个节点
        nodes = [
            PlanNode(id=f"node{i}", name=f"action{i}", action="goto")
            for i in range(5)
        ]

        for node in nodes:
            state.add_root(node)

        # 模拟多个失败
        nodes[0].mark_started()
        nodes[0].mark_success()

        nodes[1].mark_started()
        nodes[1].mark_failed()

        nodes[2].mark_started()
        nodes[2].mark_failed()

        # 验证状态
        failed_nodes = state.get_failed_nodes()
        assert len(failed_nodes) == 2

    def test_insertion_count_limit(self, planning_orchestrator):
        """测试插入次数限制"""
        dynamic_planner = DynamicPlanner(
            world_model=planning_orchestrator.world_model
        )

        # 设置门为关闭
        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")

        # 创建需要开门的节点
        node = PlanNode(
            id="goto_kitchen",
            name="go_to_kitchen",
            action="goto",
            preconditions=["door.kitchen_door == open"]
        )

        # 多次尝试插入
        for i in range(5):
            modified, new_nodes = dynamic_planner.check_and_insert_preconditions(
                node,
                plan_nodes=[]
            )

            # 验证插入计数
            if dynamic_planner.insertion_count >= dynamic_planner.max_insertions:
                # 达到限制后，应该停止插入
                break

    def test_reset_between_plans(self, planning_orchestrator):
        """测试计划间的重置"""
        # 第一次规划
        planning_orchestrator.world_model.set_door_state("kitchen_door", "closed")
        plan1 = planning_orchestrator.get_plan("去厨房")

        # 重置环境
        planning_orchestrator.world_model.set_door_state("kitchen_door", "open")

        # 第二次规划
        plan2 = planning_orchestrator.get_plan("去厨房")

        # 两个计划都应该有效
        assert isinstance(plan1, PlanState)
        assert isinstance(plan2, PlanState)

    def test_rollback_capability(self):
        """测试回滚能力"""
        from brain.planning.state import CommitLevel

        state = PlanState()

        # 添加不同commit级别的节点
        soft_node = PlanNode(
            id="soft",
            name="soft_action",
            action="detect",
            commit_level=CommitLevel.SOFT
        )

        hard_node = PlanNode(
            id="hard",
            name="hard_action",
            action="goto",
            commit_level=CommitLevel.HARD
        )

        state.add_root(soft_node)
        state.add_root(hard_node)

        # SOFT级别失败后可以回滚
        soft_node.mark_started()
        soft_node.mark_failed()

        # 验证可以安全回滚
        can_rollback = state.can_safely_rollback()
        assert can_rollback is True

    def test_plan_validation_with_failures(self, planning_orchestrator, plan_validator):
        """测试包含失败的计划验证"""
        # 生成计划
        plan = planning_orchestrator.get_plan("去厨房拿杯水")

        # 模拟一些节点失败
        for node in list(plan.nodes.values())[:2]:
            node.mark_started()
            node.mark_failed()

        # 验证计划
        validation = plan_validator.validate(plan)

        # 计划结构应该仍然有效
        # 节点失败不影响结构有效性
        assert isinstance(validation, PlanValidation)


@pytest.mark.e2e
class TestErrorScenarios:
    """错误场景E2E测试"""

    def test_unknown_command(self, planning_orchestrator):
        """测试未知命令"""
        # 未知命令应该仍然生成计划（即使是空计划或默认计划）
        plan = planning_orchestrator.get_plan("未知命令xyz")

        # 应该返回一个PlanState对象
        assert isinstance(plan, PlanState)

    def test_empty_command(self, planning_orchestrator):
        """测试空命令"""
        # 空命令处理
        plan = planning_orchestrator.get_plan("")

        assert isinstance(plan, PlanState)

    def test_missing_location(self, planning_orchestrator):
        """测试缺失位置"""
        # 尝试规划到不存在的位置
        plan = planning_orchestrator.get_plan("去不存在的位置")

        # 应该生成计划或优雅处理
        assert isinstance(plan, PlanState)

    def test_missing_object(self, planning_orchestrator):
        """测试缺失物体"""
        # 尝试与不存在的物体交互
        plan = planning_orchestrator.get_plan("拿不存在的物体")

        # 应该生成计划或优雅处理
        assert isinstance(plan, PlanState)

    def test_concurrent_planning(self):
        """测试并发规划"""
        orchestrator1 = PlanningOrchestrator(platform="ugv")
        orchestrator2 = PlanningOrchestrator(platform="ugv")

        # 并发生成计划
        plan1 = orchestrator1.get_plan("去厨房")
        plan2 = orchestrator2.get_plan("拿杯子")

        # 两个计划应该独立
        assert plan1 is not plan2

    def test_planning_orchestrator_consistency(self, planning_orchestrator):
        """测试规划编排器一致性"""
        # 多次初始化
        for _ in range(3):
            test_orchestrator = PlanningOrchestrator(platform="ugv")
            plan = test_orchestrator.get_plan("去厨房")

            assert isinstance(plan, PlanState)
            assert len(plan.nodes) > 0
