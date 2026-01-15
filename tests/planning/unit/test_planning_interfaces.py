# -*- coding: utf-8 -*-
"""
规划层接口单元测试

测试 PlanningInput、PlanningOutput、ReplanningInput、ReplanningOutput 等接口
"""

import pytest
from datetime import datetime
from dataclasses import asdict

from brain.planning.interfaces import (
    PlanningInput,
    PlanningOutput,
    ReplanningInput,
    ReplanningOutput,
    PlanningStatus
)
from brain.planning.state import PlanState, PlanNode, NodeStatus


class TestPlanningInterfaces:
    """规划层接口测试类"""

    def test_planning_input_creation(self):
        """测试 PlanningInput 创建"""
        input_data = PlanningInput(
            command="去厨房拿杯水"
        )

        assert input_data.command == "去厨房拿杯水"
        assert input_data.reasoning_result is None
        assert input_data.planning_context is None
        assert input_data.beliefs == []
        assert isinstance(input_data.timestamp, datetime)

    def test_planning_input_with_reasoning(self):
        """测试带推理结果的 PlanningInput"""
        from brain.cognitive.reasoning.reasoning_result import ReasoningResult, ReasoningMode, ComplexityLevel

        reasoning_result = ReasoningResult(
            mode=ReasoningMode.PLANNING,
            query="去厨房拿杯水",
            context_summary="当前位置：客厅，目标：厨房",
            complexity=ComplexityLevel.SIMPLE,
            chain=[],
            decision="可以执行",
            suggestion="建议先检查门状态",
            confidence=0.9,
            raw_response="response"
        )

        input_data = PlanningInput(
            command="去厨房拿杯水",
            reasoning_result=reasoning_result
        )

        assert input_data.has_reasoning() is True
        assert input_data.get_reasoning_mode() == ReasoningMode.PLANNING
        assert input_data.reasoning_result.confidence == 0.9

    def test_planning_input_with_beliefs(self):
        """测试带信念的 PlanningInput"""
        from brain.cognitive.world_model.belief.belief import Belief

        beliefs = [
            Belief(id="1", content="杯子在厨房", confidence=0.9),
            Belief(id="2", content="厨房门是关着的", confidence=0.8)
        ]

        input_data = PlanningInput(
            command="去厨房拿杯水",
            beliefs=beliefs
        )

        assert len(input_data.beliefs) == 2
        high_conf = input_data.get_high_confidence_beliefs(threshold=0.85)
        assert len(high_conf) == 1
        assert high_conf[0].content == "杯子在厨房"

    def test_planning_output_creation(self):
        """测试 PlanningOutput 创建"""
        plan_state = PlanState()
        root = PlanNode(
            id="root",
            name="test_task",
            task="test_task"
        )
        plan_state.add_root(root)

        output = PlanningOutput(
            plan_state=plan_state,
            planning_status=PlanningStatus.SUCCESS,
            estimated_duration=10.5,
            success_rate=0.95
        )

        assert output.is_successful() is True
        assert output.needs_clarification() is False
        assert output.is_rejected() is False
        assert output.estimated_duration == 10.5
        assert output.success_rate == 0.95

    def test_planning_output_failure(self):
        """测试失败的 PlanningOutput"""
        plan_state = PlanState()

        output = PlanningOutput(
            plan_state=plan_state,
            planning_status=PlanningStatus.FAILURE,
            estimated_duration=0.0,
            success_rate=0.0,
            rejection_reason="指令无法理解"
        )

        assert output.is_successful() is False
        assert output.rejection_reason == "指令无法理解"

    def test_planning_output_clarification(self):
        """测试需要澄清的 PlanningOutput"""
        plan_state = PlanState()

        output = PlanningOutput(
            plan_state=plan_state,
            planning_status=PlanningStatus.CLARIFICATION_NEEDED,
            estimated_duration=0.0,
            success_rate=0.5,
            clarification_request="请问您要拿哪个杯子？"
        )

        assert output.needs_clarification() is True
        assert output.clarification_request == "请问您要拿哪个杯子？"

    def test_planning_output_to_dict(self):
        """测试 PlanningOutput 序列化"""
        plan_state = PlanState()
        root = PlanNode(
            id="root",
            name="test_task",
            task="test_task"
        )
        plan_state.add_root(root)

        output = PlanningOutput(
            plan_state=plan_state,
            planning_status=PlanningStatus.SUCCESS,
            estimated_duration=10.5,
            success_rate=0.95
        )

        result = output.to_dict()
        assert result["status"] == "success"
        assert result["estimated_duration"] == 10.5
        assert result["success_rate"] == 0.95
        assert result["node_count"] == 1

    def test_replanning_input_creation(self):
        """测试 ReplanningInput 创建"""
        plan_state = PlanState()
        root = PlanNode(
            id="root",
            name="test_task",
            task="test_task"
        )
        plan_state.add_root(root)

        input_data = ReplanningInput(
            current_plan=plan_state,
            current_node_id="root",
            failed_actions=["action1"],
            trigger_reason="动作执行失败"
        )

        assert input_data.current_plan == plan_state
        assert input_data.current_node_id == "root"
        assert input_data.failed_actions == ["action1"]
        assert input_data.trigger_reason == "动作执行失败"
        assert input_data.urgency == "normal"

    def test_replanning_input_with_environment_changes(self):
        """测试带环境变化的 ReplanningInput"""
        plan_state = PlanState()

        input_data = ReplanningInput(
            current_plan=plan_state,
            environment_changes=[
                {"type": "obstacle", "description": "检测到新障碍物"}
            ],
            trigger_reason="环境变化",
            urgency="high"
        )

        assert len(input_data.environment_changes) == 1
        assert input_data.urgency == "high"

    def test_replanning_output_creation(self):
        """测试 ReplanningOutput 创建"""
        plan_state = PlanState()

        output = ReplanningOutput(
            new_plan=plan_state,
            replanning_type="repair",
            modified_nodes=["node1", "node2"],
            added_nodes=["node3"],
            removed_nodes=[],
            success=True,
            reason="修复了部分节点"
        )

        assert output.new_plan == plan_state
        assert output.replanning_type == "repair"
        assert len(output.modified_nodes) == 2
        assert len(output.added_nodes) == 1
        assert output.success is True
        assert output.reason == "修复了部分节点"

    def test_replanning_output_get_change_summary(self):
        """测试 ReplanningOutput 变化摘要"""
        plan_state = PlanState()

        output = ReplanningOutput(
            new_plan=plan_state,
            replanning_type="repair",
            modified_nodes=["node1", "node2"],
            added_nodes=["node3"],
            removed_nodes=["node4"],
            success=True
        )

        summary = output.get_change_summary()
        assert "repair" in summary
        assert "2" in summary  # 修改节点数
        assert "1" in summary  # 新增节点数
        assert "1" in summary  # 删除节点数


@pytest.mark.unit
class TestPlanningStatus:
    """PlanningStatus 枚举测试"""

    def test_status_values(self):
        """测试状态值"""
        assert PlanningStatus.SUCCESS.value == "success"
        assert PlanningStatus.FAILURE.value == "failure"
        assert PlanningStatus.PARTIAL.value == "partial"
        assert PlanningStatus.CLARIFICATION_NEEDED.value == "clarification"
        assert PlanningStatus.REJECTED.value == "rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
