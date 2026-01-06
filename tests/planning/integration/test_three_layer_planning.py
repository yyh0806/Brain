"""
Integration Tests for Three-Layer Planning

三层规划集成测试
"""

import pytest

from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.state import PlanState, NodeStatus


@pytest.mark.integration
class TestThreeLayerPlanning:
    """三层规划集成测试"""

    def test_simple_command_planning(self, planning_orchestrator):
        """测试简单指令规划"""
        command = "去厨房"
        plan_state = planning_orchestrator.get_plan(command)

        assert isinstance(plan_state, PlanState)
        assert len(plan_state.roots) > 0
        assert len(plan_state.nodes) > 0

    def test_complex_command_planning(self, planning_orchestrator):
        """测试复杂指令规划"""
        command = "去厨房拿杯水"
        plan_state = planning_orchestrator.get_plan(command)

        assert isinstance(plan_state, PlanState)
        # 应该包含多个节点（任务、技能、动作）
        assert len(plan_state.nodes) > 1

        # 验证HTN结构
        root = plan_state.roots[0]
        assert root.task is not None

        # 应该有子节点
        if root.children:
            # 验证技能层
            for child in root.children:
                if child.skill:
                    # 验证动作层
                    for grandchild in child.children:
                        assert grandchild.action is not None

    def test_command_with_multiple_skills(self, planning_orchestrator):
        """测试包含多个技能的指令"""
        command = "去厨房拿杯子"
        plan_state = planning_orchestrator.get_plan(command)

        assert len(plan_state.nodes) > 0

        # 验证生成的计划结构合理
        nodes = list(plan_state.nodes.values())
        assert all(node.status == NodeStatus.PENDING for node in nodes)

    def test_three_layer_transformation(self, planning_orchestrator):
        """测试三层转换：任务→技能→动作"""
        command = "去厨房拿杯水"
        plan_state = planning_orchestrator.get_plan(command)

        # 验证任务层
        task_nodes = [n for n in plan_state.nodes.values() if n.task]
        assert len(task_nodes) > 0

        # 验证技能层
        skill_nodes = [n for n in plan_state.nodes.values() if n.skill]
        assert len(skill_nodes) > 0

        # 验证动作层
        action_nodes = [n for n in plan_state.nodes.values() if n.action]
        assert len(action_nodes) > 0

    def test_planner_components_integration(self, planning_orchestrator):
        """测试规划器组件集成"""
        # 验证所有组件都已初始化
        assert planning_orchestrator.task_planner is not None
        assert planning_orchestrator.skill_planner is not None
        assert planning_orchestrator.action_planner is not None
        assert planning_orchestrator.capability_registry is not None
        assert planning_orchestrator.platform_adapter is not None
        assert planning_orchestrator.world_model is not None

    def test_capability_system_integration(self, planning_orchestrator):
        """测试能力系统集成"""
        # 获取平台能力
        capabilities = planning_orchestrator.get_capabilities()
        assert isinstance(capabilities, dict)
        assert len(capabilities) > 0

    def test_plan_structure_validity(self, planning_orchestrator, plan_validator):
        """测试计划结构有效性"""
        command = "去厨房拿杯水"
        plan_state = planning_orchestrator.get_plan(command)

        # 使用PlanValidator验证
        validation_result = plan_validator.validate(plan_state)
        assert validation_result.valid is True
        assert len(validation_result.issues) == 0

    def test_world_model_integration(self, planning_orchestrator):
        """测试世界模型集成"""
        # 规划器应该能够查询世界模型
        assert planning_orchestrator.world_model is not None

        # 验证世界模型可以访问位置信息
        kitchen = planning_orchestrator.world_model.get_location("kitchen")
        assert kitchen is not None

    def test_platform_specific_planning(self):
        """测试平台特定规划"""
        # UGV平台
        ugv_orchestrator = PlanningOrchestrator(platform="ugv")
        ugv_plan = ugv_orchestrator.get_plan("去厨房")
        assert len(ugv_plan.nodes) > 0

        # Drone平台
        drone_orchestrator = PlanningOrchestrator(platform="drone")
        drone_plan = drone_orchestrator.get_plan("去厨房")
        assert len(drone_plan.nodes) > 0


@pytest.mark.integration
class TestPlanningWorkflow:
    """规划工作流测试"""

    def test_planning_workflow_consistency(self, planning_orchestrator):
        """测试规划工作流一致性"""
        # 同一个命令应该产生一致的计划
        command = "去厨房拿杯水"

        plan1 = planning_orchestrator.get_plan(command)
        plan2 = planning_orchestrator.get_plan(command)

        # 节点数量应该相同
        assert len(plan1.nodes) == len(plan2.nodes)

    def test_multiple_commands_independence(self, planning_orchestrator):
        """测试多个命令的独立性"""
        plan1 = planning_orchestrator.get_plan("去厨房")
        plan2 = planning_orchestrator.get_plan("拿杯子")

        # 两个计划应该是独立的
        assert plan1 is not plan2
        assert plan1.nodes is not plan2.nodes
