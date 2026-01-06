"""
End-to-End Tests for Full Task Execution

完整任务执行端到端测试
"""

import pytest

from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.state import PlanState, NodeStatus


@pytest.mark.e2e
class TestFullTaskExecution:
    """完整任务执行E2E测试"""

    def test_simple_goto_task(self, planning_orchestrator):
        """测试简单移动任务"""
        command = "去厨房"
        plan = planning_orchestrator.get_plan(command)

        # 验证计划生成
        assert isinstance(plan, PlanState)
        assert len(plan.nodes) > 0

        # 验证所有节点都在PENDING状态
        pending_nodes = plan.get_pending_nodes()
        assert len(pending_nodes) == len(plan.nodes)

    def test_complex_fetch_task(self, planning_orchestrator):
        """测试复杂取物任务"""
        command = "去厨房拿杯水"
        plan = planning_orchestrator.get_plan(command)

        # 验证计划结构
        assert len(plan.roots) > 0
        root = plan.roots[0]

        # 验证任务节点
        assert root.task is not None
        assert "拿杯水" in root.task or "拿" in root.task or "去厨房" in root.task

        # 验证子节点结构
        if root.children:
            # 应该有技能层
            skill_children = [c for c in root.children if c.skill]
            assert len(skill_children) > 0

    def test_full_workflow_planning_only(self, planning_orchestrator):
        """测试完整工作流 - 仅规划"""
        command = "去厨房拿杯子然后回来"

        # 生成计划
        plan = planning_orchestrator.get_plan(command)

        # 验证计划完整性
        assert plan is not None
        assert len(plan.nodes) > 0

        # 验证所有节点都有ID
        for node_id, node in plan.nodes.items():
            assert node.id == node_id
            assert node.name is not None

    def test_multi_skill_task(self, planning_orchestrator):
        """测试多技能任务"""
        command = "去厨房拿杯子放到餐桌"
        plan = planning_orchestrator.get_plan(command)

        # 应该生成包含多个技能的计划
        assert len(plan.nodes) > 1

        # 验证不同的技能层
        skill_nodes = [n for n in plan.nodes.values() if n.skill]
        assert len(skill_nodes) > 0

    def test_return_task(self, planning_orchestrator):
        """测试返回任务"""
        command = "回来"
        plan = planning_orchestrator.get_plan(command)

        assert isinstance(plan, PlanState)
        assert len(plan.nodes) > 0

    def test_task_with_location(self, planning_orchestrator):
        """测试带位置的任务"""
        command = "去餐桌"
        plan = planning_orchestrator.get_plan(command)

        assert len(plan.nodes) > 0

        # 验证位置信息被使用
        table_location = planning_orchestrator.world_model.get_location("table")
        assert table_location is not None

    def test_task_with_object(self, planning_orchestrator):
        """测试带物体的任务"""
        command = "拿杯子"
        plan = planning_orchestrator.get_plan(command)

        assert len(plan.nodes) > 0

        # 验证物体信息被使用
        cup_location = planning_orchestrator.world_model.get_object_location("cup")
        assert cup_location is not None

    def test_plan_structure_validity_e2e(self, planning_orchestrator, plan_validator):
        """测试计划结构有效性（E2E）"""
        commands = [
            "去厨房",
            "拿杯子",
            "去厨房拿杯水",
            "去厨房拿杯子然后回来"
        ]

        for command in commands:
            plan = planning_orchestrator.get_plan(command)
            validation = plan_validator.validate(plan)

            # 所有计划都应该有效
            assert validation.valid is True, f"Plan for '{command}' failed validation"
            assert len(validation.issues) == 0

    def test_consistent_planning_results(self, planning_orchestrator):
        """测试规划结果一致性"""
        command = "去厨房拿杯水"

        # 多次生成相同命令的计划
        plans = [planning_orchestrator.get_plan(command) for _ in range(3)]

        # 所有计划应该有相同的结构
        node_counts = [len(p.nodes) for p in plans]
        assert all(count == node_counts[0] for count in node_counts)

    def test_different_commands_different_plans(self, planning_orchestrator):
        """测试不同命令生成不同计划"""
        commands = ["去厨房", "拿杯子", "回来"]
        plans = [planning_orchestrator.get_plan(cmd) for cmd in commands]

        # 计划应该彼此独立
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                assert plans[i] is not plans[j]
                assert plans[i].nodes is not plans[j].nodes

    def test_world_model_state_after_planning(self, planning_orchestrator):
        """测试规划后的世界模型状态"""
        initial_robot_pos = planning_orchestrator.world_model.get_robot_position().copy()

        # 执行规划（不执行，只规划）
        planning_orchestrator.get_plan("去厨房")

        # 世界模型状态应该保持不变（仅规划不执行）
        final_robot_pos = planning_orchestrator.world_model.get_robot_position()
        assert initial_robot_pos == final_robot_pos

    def test_platform_specific_planning_e2e(self):
        """测试平台特定规划（E2E）"""
        platforms = ["ugv", "drone", "usv"]

        for platform in platforms:
            orchestrator = PlanningOrchestrator(platform=platform)
            plan = orchestrator.get_plan("去厨房")

            assert isinstance(plan, PlanState)
            assert len(plan.nodes) > 0


@pytest.mark.e2e
class TestHTNStructure:
    """HTN结构E2E测试"""

    def test_three_layer_hierarchy(self, planning_orchestrator):
        """测试三层层级结构"""
        command = "去厨房拿杯水"
        plan = planning_orchestrator.get_plan(command)

        # 验证三层结构存在
        task_nodes = [n for n in plan.nodes.values() if n.task and not n.skill]
        skill_nodes = [n for n in plan.nodes.values() if n.skill and not n.action]
        action_nodes = [n for n in plan.nodes.values() if n.action]

        # 至少应该有任务层和动作层
        assert len(task_nodes) > 0 or len(skill_nodes) > 0
        assert len(action_nodes) > 0

    def test_parent_child_relationships(self, planning_orchestrator):
        """测试父子关系"""
        command = "去厨房拿杯水"
        plan = planning_orchestrator.get_plan(command)

        # 验证父子关系正确
        for node in plan.nodes.values():
            for child in node.children:
                assert child.parent == node

    def test_node_hierarchy_depth(self, planning_orchestrator):
        """测试节点层级深度"""
        command = "去厨房拿杯水"
        plan = planning_orchestrator.get_plan(command)

        # 获取所有节点的层级
        levels = [node.level for node in plan.nodes.values()]

        # 应该有不同层级的节点
        if len(levels) > 1:
            assert min(levels) == 0  # 根节点层级为0
            assert max(levels) >= 1   # 应该有更深的层级


@pytest.mark.e2e
class TestPlanningOrchestratorE2E:
    """规划编排器E2E测试"""

    def test_orchestrator_initialization(self):
        """测试编排器初始化"""
        orchestrator = PlanningOrchestrator(platform="ugv")

        # 验证所有组件已初始化
        assert orchestrator.task_planner is not None
        assert orchestrator.skill_planner is not None
        assert orchestrator.action_planner is not None
        assert orchestrator.capability_registry is not None
        assert orchestrator.world_model is not None

    def test_orchestrator_multiple_plans(self):
        """测试编排器生成多个计划"""
        orchestrator = PlanningOrchestrator(platform="ugv")

        plan1 = orchestrator.get_plan("去厨房")
        plan2 = orchestrator.get_plan("拿杯子")
        plan3 = orchestrator.get_plan("回来")

        # 所有计划都应该成功生成
        assert isinstance(plan1, PlanState)
        assert isinstance(plan2, PlanState)
        assert isinstance(plan3, PlanState)

    def test_orchestrator_capability_query(self, planning_orchestrator):
        """测试编排器能力查询"""
        capabilities = planning_orchestrator.get_capabilities()

        assert isinstance(capabilities, dict)
        assert len(capabilities) > 0
