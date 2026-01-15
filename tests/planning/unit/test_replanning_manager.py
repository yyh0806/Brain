# -*- coding: utf-8 -*-
"""
重规划管理器单元测试

测试 ReplanningManager 的核心功能
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from brain.planning.intelligent import (
    ReplanningManager,
    ReplanningTrigger,
    ReplanningStrategy,
    EnvironmentChange,
    FailureType
)
from brain.planning.state import PlanState, PlanNode, NodeStatus
from brain.planning.interfaces import ReplanningInput, ReplanningOutput


@pytest.fixture
def mock_world_model():
    """模拟世界模型"""
    model = Mock()
    model.get_location = Mock(return_value=None)
    model.get_object_location = Mock(return_value=None)
    model.get_door_state = Mock(return_value="closed")
    return model


@pytest.fixture
def sample_plan():
    """创建示例计划"""
    plan = PlanState()

    root = PlanNode(
        id="root",
        name="go_to_kitchen",
        task="go_to_kitchen",
        action="move_to",
        preconditions=["robot_at(living_room)"],
        expected_effects=["robot_at(kitchen)"]
    )

    child1 = PlanNode(
        id="open_door",
        name="open_door",
        task="go_to_kitchen",
        action="open_door",
        preconditions=["door_closed(kitchen_door)"],
        expected_effects=["door_open(kitchen_door)"]
    )

    child2 = PlanNode(
        id="pickup_cup",
        name="pickup_cup",
        task="pickup_cup",
        action="grasp",
        preconditions=["robot_at(kitchen)", "object_visible(cup)"],
        expected_effects=["holding(cup)"]
    )

    root.add_child(child1)
    root.add_child(child2)
    plan.add_root(root)

    return plan


@pytest.mark.unit
class TestReplanningManager:
    """重规划管理器测试"""

    def test_initialization(self, mock_world_model):
        """测试初始化"""
        manager = ReplanningManager(world_model=mock_world_model)

        assert manager.world_model == mock_world_model
        assert manager.insertion_count == 0
        assert manager.retry_count == 0
        assert manager.replan_count == 0
        assert manager.environment_changes == []

    def test_initialization_with_config(self, mock_world_model):
        """测试带配置的初始化"""
        config = {
            "max_insertions": 5,
            "max_retries": 5,
            "enable_plan_mending": False
        }

        manager = ReplanningManager(
            world_model=mock_world_model,
            config=config
        )

        assert manager.max_insertions == 5
        assert manager.max_retries == 5
        assert manager.enable_plan_mending is False

    def test_detect_environment_changes_with_failures(self, sample_plan):
        """测试检测失败的环境变化"""
        manager = ReplanningManager()

        input_data = ReplanningInput(
            current_plan=sample_plan,
            failed_actions=["open_door"],
            trigger_reason="动作失败"
        )

        changes = manager.detect_environment_changes(
            current_plan=input_data.current_plan,
            current_context=None,
            new_beliefs=[],
            failed_actions=input_data.failed_actions
        )

        assert len(changes) > 0
        assert changes[0].change_type == ReplanningTrigger.ACTION_FAILURE.value
        assert changes[0].severity == "high"

    def test_detect_environment_changes_with_beliefs(self, sample_plan):
        """测试检测信念矛盾"""
        from brain.cognitive.world_model.belief.belief import Belief

        manager = ReplanningManager()

        # 创建被证伪的信念
        falsified_belief = Belief(
            id="1",
            content="杯子在厨房",
            confidence=0.0,
            falsified=True
        )

        input_data = ReplanningInput(
            current_plan=sample_plan,
            new_beliefs=[falsified_belief],
            trigger_reason="信念被证伪"
        )

        changes = manager.detect_environment_changes(
            current_plan=input_data.current_plan,
            current_context=None,
            new_beliefs=input_data.new_beliefs,
            failed_actions=[]
        )

        # 应该检测到信念矛盾
        belief_changes = [c for c in changes if c.change_type == ReplanningTrigger.BELIEF_CONTRADICTION.value]
        assert len(belief_changes) > 0

    def test_make_replanning_decision_critical(self, sample_plan):
        """测试严重变化的重规划决策"""
        manager = ReplanningManager()

        # 创建严重变化
        critical_change = EnvironmentChange(
            change_id="critical_1",
            change_type=ReplanningTrigger.ENVIRONMENT_CHANGE.value,
            description="检测到严重障碍",
            severity="critical",
            affected_nodes=[]
        )

        input_data = ReplanningInput(
            current_plan=sample_plan,
            trigger_reason="严重环境变化"
        )

        decision = manager.make_replanning_decision(
            replanning_input=input_data,
            changes=[critical_change]
        )

        assert decision.should_replan is True
        assert decision.strategy == ReplanningStrategy.REPLAN
        assert decision.urgency == "critical"

    def test_make_replanning_decision_normal(self, sample_plan):
        """测试普通情况的重规划决策"""
        manager = ReplanningManager()

        input_data = ReplanningInput(
            current_plan=sample_plan,
            failed_actions=[],
            trigger_reason="正常检查"
        )

        decision = manager.make_replanning_decision(
            replanning_input=input_data,
            changes=[]
        )

        # 没有变化时，默认不重规划
        assert decision.should_replan is False
        assert decision.strategy == ReplanningStrategy.RETRY

    def test_execute_retry(self, sample_plan):
        """测试重试策略"""
        manager = ReplanningManager()

        input_data = ReplanningInput(
            current_plan=sample_plan,
            trigger_reason="重试"
        )

        from brain.planning.intelligent.replanning_manager import ReplanningDecision
        decision = ReplanningDecision(
            should_replan=False,
            strategy=ReplanningStrategy.RETRY,
            reason="重试当前动作"
        )

        output = manager.replan(input_data, decision)

        assert output.success is True
        assert output.replanning_type == "retry"
        assert output.reason == "重试当前动作 (第 1 次)"
        assert manager.retry_count == 1

    def test_execute_abort(self, sample_plan):
        """测试中止策略"""
        manager = ReplanningManager()

        input_data = ReplanningInput(
            current_plan=sample_plan,
            trigger_reason="严重错误"
        )

        from brain.planning.intelligent.replanning_manager import ReplanningDecision
        decision = ReplanningDecision(
            should_replan=True,
            strategy=ReplanningStrategy.ABORT,
            reason="严重错误，中止执行"
        )

        output = manager.replan(input_data, decision)

        assert output.success is False
        assert output.replanning_type == "abort"
        assert "中止" in output.reason

    def test_validate_plan(self, sample_plan):
        """测试计划验证"""
        manager = ReplanningManager()

        valid, issues = manager.validate_plan(sample_plan)

        # 简单计划应该是有效的
        assert valid is True
        assert len(issues) == 0

    def test_reset_counters(self, mock_world_model):
        """测试重置计数器"""
        manager = ReplanningManager(world_model=mock_world_model)

        # 增加计数
        manager.insertion_count = 5
        manager.retry_count = 3
        manager.replan_count = 2
        manager.environment_changes.append(
            EnvironmentChange(
                change_id="test",
                change_type="test",
                description="test",
                severity="low",
                affected_nodes=[]
            )
        )

        manager.reset_counters()

        assert manager.insertion_count == 0
        assert manager.retry_count == 0
        assert manager.replan_count == 0
        assert len(manager.environment_changes) == 0

    def test_get_statistics(self, mock_world_model):
        """测试获取统计信息"""
        manager = ReplanningManager(world_model=mock_world_model)

        manager.insertion_count = 3
        manager.retry_count = 2
        manager.replan_count = 1

        stats = manager.get_statistics()

        assert stats["insertion_count"] == 3
        assert stats["retry_count"] == 2
        assert stats["replan_count"] == 1


@pytest.mark.unit
class TestEnvironmentChange:
    """EnvironmentChange 测试"""

    def test_creation(self):
        """测试创建环境变化"""
        change = EnvironmentChange(
            change_id="change_1",
            change_type="obstacle",
            description="新障碍物",
            severity="high",
            affected_nodes=["node1", "node2"]
        )

        assert change.change_id == "change_1"
        assert change.change_type == "obstacle"
        assert change.severity == "high"
        assert len(change.affected_nodes) == 2
        assert isinstance(change.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
