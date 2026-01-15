# -*- coding: utf-8 -*-
"""
认知层-规划层集成测试

测试认知层与规划层之间的数据流和交互
"""

import pytest
from datetime import datetime

from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.interfaces import (
    PlanningInput,
    PlanningOutput,
    PlanningStatus,
    CognitiveWorldAdapter
)
from brain.planning.state import PlanState
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief
from brain.cognitive.reasoning.reasoning_result import (
    ReasoningResult,
    ReasoningMode,
    ComplexityLevel,
    ReasoningStep
)


@pytest.mark.integration
class TestCognitivePlanningIntegration:
    """认知层-规划层集成测试"""

    def test_planning_orchestrator_with_cognitive_layer(self):
        """测试认知层集成模式的 PlanningOrchestrator"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=True
        )

        assert orchestrator.use_cognitive_layer is True
        assert orchestrator.cognitive_adapter is not None
        assert orchestrator.world_model is not None

    def test_planning_with_simple_input(self):
        """测试简单指令规划"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=False
        )

        input_data = PlanningInput(command="去厨房拿杯水")

        output = orchestrator.plan(input_data)

        assert output.planning_status == PlanningStatus.SUCCESS
        assert len(output.plan_state.nodes) > 0
        assert output.estimated_duration > 0

    def test_planning_with_reasoning_result(self):
        """测试带推理结果的规划"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=False
        )

        # 创建推理结果
        reasoning_result = ReasoningResult(
            mode=ReasoningMode.PLANNING,
            query="去厨房拿杯水",
            context_summary="用户在客厅，想去厨房拿杯子",
            complexity=ComplexityLevel.SIMPLE,
            chain=[
                ReasoningStep(
                    step_number=1,
                    question="用户的意图是什么？",
                    analysis="用户想去厨房拿杯子",
                    conclusion="这是一个导航+抓取任务",
                    confidence=0.95
                )
            ],
            decision="可以执行，需要先移动到厨房，然后抓取杯子",
            suggestion="建议检查厨房门是否打开",
            confidence=0.9,
            raw_response="推理结果"
        )

        input_data = PlanningInput(
            command="去厨房拿杯水",
            reasoning_result=reasoning_result
        )

        output = orchestrator.plan(input_data)

        assert output.planning_status == PlanningStatus.SUCCESS
        assert output.success_rate > 0.8  # 基于推理置信度

    def test_planning_with_low_confidence_rejection(self):
        """测试低置信度推理结果的拒绝"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=False
        )

        # 创建低置信度推理结果
        reasoning_result = ReasoningResult(
            mode=ReasoningMode.PLANNING,
            query="不清楚的指令",
            context_summary="指令不明确",
            complexity=ComplexityLevel.COMPLEX,
            chain=[],
            decision="指令不够明确",
            suggestion="需要澄清",
            confidence=0.2,  # 低于阈值 0.3
            raw_response="推理结果"
        )

        input_data = PlanningInput(
            command="不清楚的指令",
            reasoning_result=reasoning_result
        )

        output = orchestrator.plan(input_data)

        assert output.planning_status == PlanningStatus.REJECTED
        assert "置信度过低" in output.rejection_reason

    def test_cognitive_world_adapter(self):
        """测试 CognitiveWorldAdapter"""
        # 创建规划上下文
        planning_context = PlanningContext(
            current_position={"x": 0.0, "y": 0.0, "z": 0.0},
            current_heading=0.0,
            obstacles=[
                {"type": "static", "position": {"x": 2.0, "y": 0.0}}
            ],
            targets=[
                {"type": "object", "name": "cup", "position": {"x": 5.0, "y": 3.0}}
            ],
            points_of_interest=[
                {"name": "kitchen", "position": {"x": 5.0, "y": 3.0}}
            ],
            weather={"condition": "clear"},
            battery_level=85.0,
            signal_strength=90.0,
            available_paths=[],
            constraints=["避障"],
            recent_changes=[],
            risk_areas=[]
        )

        # 创建信念
        beliefs = [
            Belief(id="1", content="杯子在厨房", confidence=0.9),
            Belief(id="2", content="厨房门是关着的", confidence=0.8)
        ]

        # 创建适配器
        adapter = CognitiveWorldAdapter(
            planning_context=planning_context,
            beliefs=beliefs
        )

        # 测试位置查询
        kitchen = adapter.get_location("kitchen")
        assert kitchen is not None
        assert kitchen.name == "kitchen"

        # 测试机器人位置
        robot_pos = adapter.get_robot_position()
        assert robot_pos["x"] == 0.0
        assert robot_pos["y"] == 0.0

        # 测试障碍物查询
        obstacles = adapter.get_obstacles()
        assert len(obstacles) == 1

        # 测试约束查询
        constraints = adapter.get_constraints()
        assert "避障" in constraints

        # 测试电量查询
        battery = adapter.get_battery_level()
        assert battery == 85.0

    def test_update_context_in_orchestrator(self):
        """测试在编排器中更新上下文"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=True
        )

        # 创建规划上下文
        planning_context = PlanningContext(
            current_position={"x": 1.0, "y": 2.0, "z": 0.0},
            current_heading=90.0,
            obstacles=[],
            targets=[],
            points_of_interest=[],
            weather={"condition": "clear"},
            battery_level=80.0,
            signal_strength=85.0,
            available_paths=[],
            constraints=[],
            recent_changes=[],
            risk_areas=[]
        )

        beliefs = [
            Belief(id="1", content="测试信念", confidence=0.8)
        ]

        # 更新上下文
        orchestrator.update_context(planning_context, beliefs)

        # 验证更新
        assert orchestrator.cognitive_adapter is not None
        robot_pos = orchestrator.world_model.get_robot_position()
        assert robot_pos["x"] == 1.0

    def test_replanning_with_cognitive_context(self):
        """测试使用认知上下文的重规划"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=True,
            enable_replanning=True
        )

        # 创建初始计划
        input_data = PlanningInput(command="去厨房拿杯水")
        output = orchestrator.plan(input_data)

        assert output.planning_status == PlanningStatus.SUCCESS

        # 模拟执行失败
        plan_state = output.plan_state
        failed_actions = [list(plan_state.nodes.keys())[0]]

        # 创建重规划输入
        from brain.planning.interfaces import ReplanningInput
        replan_input = ReplanningInput(
            current_plan=plan_state,
            failed_actions=failed_actions,
            trigger_reason="动作失败"
        )

        # 执行重规划
        replan_output = orchestrator.replan(replan_input)

        assert replan_output is not None
        assert replan_output.new_plan is not None

    def test_planning_with_beliefs(self):
        """测试带信念的规划"""
        orchestrator = PlanningOrchestrator(
            platform="ugv",
            use_cognitive_layer=True
        )

        # 创建高置信度信念
        beliefs = [
            Belief(id="1", content="杯子在厨房", confidence=0.95),
            Belief(id="2", content="厨房门已打开", confidence=0.9),
            Belief(id="3", content="路径畅通", confidence=0.85)
        ]

        input_data = PlanningInput(
            command="去厨房拿杯水",
            beliefs=beliefs
        )

        output = orchestrator.plan(input_data)

        assert output.planning_status == PlanningStatus.SUCCESS
        # 高置信度信念应该提高成功率
        assert output.success_rate > 0.8

    def test_planning_input_summary(self):
        """测试 PlanningInput 摘要生成"""
        reasoning_result = ReasoningResult(
            mode=ReasoningMode.PLANNING,
            query="测试指令",
            context_summary="测试上下文",
            complexity=ComplexityLevel.SIMPLE,
            chain=[],
            decision="可以执行",
            suggestion="无",
            confidence=0.9,
            raw_response="response"
        )

        beliefs = [
            Belief(id="1", content="测试信念", confidence=0.8)
        ]

        input_data = PlanningInput(
            command="测试指令",
            reasoning_result=reasoning_result,
            beliefs=beliefs
        )

        summary = input_data.to_summary()

        assert "测试指令" in summary
        assert "planning" in summary
        assert "0.90" in summary
        assert "信念数量: 1" in summary

    def test_planning_output_summary(self):
        """测试 PlanningOutput 摘要生成"""
        orchestrator = PlanningOrchestrator(platform="ugv")

        input_data = PlanningInput(command="去厨房拿杯水")
        output = orchestrator.plan(input_data)

        summary = output.get_plan_summary()

        assert "规划状态: success" in summary
        assert "预计时长:" in summary
        assert "预期成功率:" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"]
