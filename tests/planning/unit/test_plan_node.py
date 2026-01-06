"""
Unit Tests for PlanNode

PlanNode 单元测试
"""

import pytest

from brain.planning.state import PlanNode, NodeStatus, CommitLevel


@pytest.mark.unit
class TestPlanNode:
    """PlanNode 测试类"""

    def test_node_creation(self, sample_plan_node):
        """测试节点创建"""
        assert sample_plan_node.id == "test_node"
        assert sample_plan_node.name == "test_action"
        assert sample_plan_node.action == "move_to"
        assert sample_plan_node.skill == "go_to_location"
        assert sample_plan_node.task == "test_task"
        assert sample_plan_node.status == NodeStatus.PENDING

    def test_node_properties(self, sample_plan_node):
        """测试节点属性"""
        # 测试 is_leaf
        assert sample_plan_node.is_leaf is True

        # 添加子节点后
        child = PlanNode(id="child", name="child_action", action="goto")
        sample_plan_node.add_child(child)
        assert sample_plan_node.is_leaf is False
        assert child.is_leaf is True

        # 测试 is_root
        assert sample_plan_node.is_root is True
        assert child.is_root is False

    def test_node_level(self, sample_plan_node):
        """测试节点层级"""
        assert sample_plan_node.level == 0

        child = PlanNode(id="child", name="child_action", action="goto")
        sample_plan_node.add_child(child)
        assert child.level == 1

        grandchild = PlanNode(id="grandchild", name="grandchild_action", action="grasp")
        child.add_child(grandchild)
        assert grandchild.level == 2

    def test_add_child(self, sample_plan_node):
        """测试添加子节点"""
        child1 = PlanNode(id="child1", name="child1", action="goto")
        child2 = PlanNode(id="child2", name="child2", action="grasp")

        sample_plan_node.add_child(child1)
        sample_plan_node.add_child(child2)

        assert len(sample_plan_node.children) == 2
        assert child1.parent == sample_plan_node
        assert child2.parent == sample_plan_node

    def test_remove_child(self, sample_plan_node):
        """测试移除子节点"""
        child = PlanNode(id="child", name="child", action="goto")
        sample_plan_node.add_child(child)

        sample_plan_node.remove_child(child)
        assert len(sample_plan_node.children) == 0
        assert child.parent is None

    def test_status_transitions(self, sample_plan_node):
        """测试状态转换"""
        # PENDING -> EXECUTING
        sample_plan_node.mark_started()
        assert sample_plan_node.status == NodeStatus.EXECUTING
        assert sample_plan_node.started_at is not None

        # EXECUTING -> SUCCESS
        sample_plan_node.mark_success()
        assert sample_plan_node.status == NodeStatus.SUCCESS
        assert sample_plan_node.completed_at is not None

    def test_mark_failed(self, sample_plan_node):
        """测试失败标记"""
        sample_plan_node.mark_started()
        sample_plan_node.mark_failed()
        assert sample_plan_node.status == NodeStatus.FAILED
        assert sample_plan_node.completed_at is not None
        assert sample_plan_node.retry_count == 1

    def test_get_all_descendants(self, sample_plan_node):
        """测试获取所有后代节点"""
        child1 = PlanNode(id="child1", name="child1", action="goto")
        child2 = PlanNode(id="child2", name="child2", action="grasp")
        grandchild = PlanNode(id="grandchild", name="grandchild", action="place")

        sample_plan_node.add_child(child1)
        sample_plan_node.add_child(child2)
        child1.add_child(grandchild)

        descendants = sample_plan_node.get_all_descendants()
        assert len(descendants) == 3
        assert child1 in descendants
        assert child2 in descendants
        assert grandchild in descendants

    def test_find_node(self, sample_plan_node):
        """测试查找节点"""
        child = PlanNode(id="child1", name="child1", action="goto")
        sample_plan_node.add_child(child)

        found = sample_plan_node.find_node("child1")
        assert found == child

        not_found = sample_plan_node.find_node("nonexistent")
        assert not_found is None

    def test_commit_level(self, sample_plan_node):
        """测试提交级别"""
        sample_plan_node.commit_level = CommitLevel.HARD
        assert sample_plan_node.commit_level == CommitLevel.HARD

        sample_plan_node.commit_level = CommitLevel.SOFT
        assert sample_plan_node.commit_level == CommitLevel.SOFT

    def test_to_dict(self, sample_plan_node):
        """测试序列化为字典"""
        data = sample_plan_node.to_dict()
        assert data['id'] == "test_node"
        assert data['name'] == "test_action"
        assert data['action'] == "move_to"
        assert data['status'] == NodeStatus.PENDING.value

    def test_can_retry(self, sample_plan_node):
        """测试重试能力"""
        # 初始状态可以重试
        assert sample_plan_node.can_retry() is True

        # 失败1次后仍可以重试（max_retries=3）
        sample_plan_node.mark_failed()
        assert sample_plan_node.can_retry() is True

        # 达到最大重试次数后不能重试
        sample_plan_node.retry_count = 3
        assert sample_plan_node.can_retry() is False


@pytest.mark.unit
class TestNodeStatus:
    """NodeStatus 测试类"""

    def test_status_values(self):
        """测试状态值"""
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.READY.value == "ready"
        assert NodeStatus.EXECUTING.value == "executing"
        assert NodeStatus.SUCCESS.value == "success"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.SKIPPED.value == "skipped"
        assert NodeStatus.CANCELLED.value == "cancelled"


@pytest.mark.unit
class TestCommitLevel:
    """CommitLevel 测试类"""

    def test_commit_level_values(self):
        """测试提交级别值"""
        assert CommitLevel.SOFT.value == "soft"
        assert CommitLevel.HARD.value == "hard"
