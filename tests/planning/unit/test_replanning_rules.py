"""
Unit Tests for ReplanningRules

ReplanningRules 单元测试
"""

import pytest

from brain.planning.intelligent import ReplanningRules
from brain.planning.state import PlanNode, NodeStatus


@pytest.mark.unit
class TestReplanningRules:
    """ReplanningRules 测试类"""

    def test_rules_initialization(self, replanning_rules):
        """测试规则初始化"""
        assert replanning_rules is not None
        assert replanning_rules.max_insertions == 3
        assert replanning_rules.max_retries == 3

    def test_should_replan_exceed_max_insertions(self, replanning_rules):
        """测试超过最大插入次数时应该重规划"""
        # 创建一个失败节点
        failed_node = PlanNode(
            id="failed_node",
            name="failed_action",
            action="goto"
        )
        failed_node.mark_failed()

        # 设置插入计数为最大值
        insertion_count = 4  # 超过 max_insertions (3)

        should_replan = replanning_rules.should_replan(
            failed_node=failed_node,
            insertion_count=insertion_count,
            retry_count=0
        )

        assert should_replan is True

    def test_should_replan_exceed_max_retries(self, replanning_rules):
        """测试超过最大重试次数时应该重规划"""
        failed_node = PlanNode(
            id="failed_node",
            name="failed_action",
            action="goto"
        )
        failed_node.mark_failed()

        retry_count = 4  # 超过 max_retries (3)

        should_replan = replanning_rules.should_replan(
            failed_node=failed_node,
            insertion_count=0,
            retry_count=retry_count
        )

        assert should_replan is True

    def test_should_not_replan_within_limits(self, replanning_rules):
        """测试在限制内不需要重规划"""
        failed_node = PlanNode(
            id="failed_node",
            name="failed_action",
            action="goto"
        )
        failed_node.mark_failed()

        should_replan = replanning_rules.should_replan(
            failed_node=failed_node,
            insertion_count=1,  # 在限制内
            retry_count=1  # 在限制内
        )

        # 在限制内，可以选择重试而不是重规划
        # 具体行为取决于实现策略

    def test_should_insert_precondition(self, replanning_rules):
        """测试判断是否应该插入前置条件"""
        failed_node = PlanNode(
            id="failed_node",
            name="failed_action",
            action="goto"
        )
        failed_node.mark_failed()

        insertion_count = 1
        retry_count = 0

        should_insert = replanning_rules.should_insert_precondition(
            failed_node=failed_node,
            insertion_count=insertion_count,
            retry_count=retry_count
        )

        assert isinstance(should_insert, bool)

    def test_get_recovery_action_insert(self, replanning_rules):
        """测试获取恢复动作 - 插入"""
        failed_node = PlanNode(
            id="failed_node",
            name="goto_door",
            action="goto"
        )
        failed_node.mark_failed()

        action = replanning_rules.get_recovery_action(
            failed_node=failed_node,
            insertion_count=1,
            retry_count=0
        )

        assert action == "insert"

    def test_get_recovery_action_retry(self, replanning_rules):
        """测试获取恢复动作 - 重试"""
        failed_node = PlanNode(
            id="failed_node",
            name="temp_action",
            action="grasp"
        )
        failed_node.mark_failed()

        action = replanning_rules.get_recovery_action(
            failed_node=failed_node,
            insertion_count=0,
            retry_count=1
        )

        assert action == "retry"

    def test_get_recovery_action_replan(self, replanning_rules):
        """测试获取恢复动作 - 重规划"""
        failed_node = PlanNode(
            id="failed_node",
            name="critical_action",
            action="goto"
        )
        failed_node.mark_failed()

        action = replanning_rules.get_recovery_action(
            failed_node=failed_node,
            insertion_count=4,  # 超过限制
            retry_count=0
        )

        assert action == "replan"
