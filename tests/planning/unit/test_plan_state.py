"""
Unit Tests for PlanState

PlanState 单元测试
"""

import pytest

from brain.planning.state import PlanState, PlanNode, NodeStatus


@pytest.mark.unit
class TestPlanState:
    """PlanState 测试类"""

    def test_create_state(self):
        """测试创建状态"""
        state = PlanState()
        assert len(state.roots) == 0
        assert len(state.nodes) == 0

    def test_add_root(self, sample_plan_node):
        """测试添加根节点"""
        state = PlanState()
        state.add_root(sample_plan_node)

        assert len(state.roots) == 1
        assert state.roots[0] == sample_plan_node
        assert sample_plan_node.id in state.nodes

    def test_get_node(self, sample_plan_node):
        """测试获取节点"""
        state = PlanState()
        state.add_root(sample_plan_node)

        retrieved = state.get_node(sample_plan_node.id)
        assert retrieved == sample_plan_node

        not_found = state.get_node("nonexistent")
        assert not_found is None

    def test_get_pending_nodes(self, sample_plan_node):
        """测试获取待执行节点"""
        state = PlanState()
        state.add_root(sample_plan_node)

        pending = state.get_pending_nodes()
        assert len(pending) == 1
        assert sample_plan_node in pending

    def test_get_executing_nodes(self):
        """测试获取执行中节点"""
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        node.mark_started()
        state.add_root(node)

        executing = state.get_executing_nodes()
        assert len(executing) == 1
        assert node in executing

    def test_get_successful_nodes(self):
        """测试获取成功节点"""
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        node.mark_started()
        node.mark_success()
        state.add_root(node)

        successful = state.get_successful_nodes()
        assert len(successful) == 1
        assert node in successful

    def test_get_failed_nodes(self):
        """测试获取失败节点"""
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        node.mark_started()
        node.mark_failed()
        state.add_root(node)

        failed = state.get_failed_nodes()
        assert len(failed) == 1
        assert node in failed

    def test_update_node_status(self):
        """测试更新节点状态"""
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        state.add_root(node)

        state.update_node_status(node.id, NodeStatus.EXECUTING)
        assert node.status == NodeStatus.EXECUTING

        state.update_node_status(node.id, NodeStatus.SUCCESS)
        assert node.status == NodeStatus.SUCCESS

    def test_get_leaf_nodes(self):
        """测试获取叶子节点"""
        state = PlanState()
        root = PlanNode(id="root", name="root", task="root")
        child = PlanNode(id="child", name="child", action="goto")
        root.add_child(child)
        state.add_root(root)

        leaves = state.get_leaf_nodes()
        assert len(leaves) == 1
        assert child in leaves

    def test_get_nodes_by_status(self):
        """测试按状态获取节点"""
        state = PlanState()

        node1 = PlanNode(id="node1", name="node1", action="goto")
        node2 = PlanNode(id="node2", name="node2", action="grasp")
        node2.mark_started()

        state.add_root(node1)
        state.add_root(node2)

        pending = state.get_nodes_by_status(NodeStatus.PENDING)
        assert len(pending) == 1
        assert node1 in pending

        executing = state.get_nodes_by_status(NodeStatus.EXECUTING)
        assert len(executing) == 1
        assert node2 in executing

    def test_get_execution_statistics(self):
        """测试获取执行统计"""
        state = PlanState()

        # 添加多个节点
        node1 = PlanNode(id="node1", name="node1", action="goto")
        node2 = PlanNode(id="node2", name="node2", action="grasp")
        node3 = PlanNode(id="node3", name="node3", action="place")

        node1.mark_started()
        node1.mark_success()

        node2.mark_started()
        node2.mark_failed()

        state.add_root(node1)
        state.add_root(node2)
        state.add_root(node3)

        stats = state.get_execution_statistics()
        assert stats['total'] == 3
        assert stats['pending'] == 1
        assert stats['success'] == 1
        assert stats['failed'] == 1
        assert stats['success_rate'] == 0.5

    def test_clone(self):
        """测试克隆状态"""
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        state.add_root(node)

        cloned = state.clone()
        assert len(cloned.nodes) == len(state.nodes)
        assert cloned.get_node(node.id) is not None
        assert cloned.get_node(node.id) is not node  # 深拷贝

    def test_to_dict(self):
        """测试序列化为字典"""
        state = PlanState()
        node = PlanNode(id="test", name="test", action="goto")
        state.add_root(node)

        data = state.to_dict()
        assert 'roots' in data
        assert 'nodes' in data
        assert 'metadata' in data

    def test_can_safely_rollback(self):
        """测试是否可以安全回滚"""
        state = PlanState()

        # SOFT 级别的失败节点可以回滚
        node1 = PlanNode(id="node1", name="node1", action="goto")
        from brain.planning.state import CommitLevel
        node1.commit_level = CommitLevel.SOFT
        node1.mark_failed()
        state.add_root(node1)

        assert state.can_safely_rollback() is True

    def test_get_nodes_to_rollback(self):
        """测试获取需要回滚的节点"""
        state = PlanState()

        node1 = PlanNode(id="node1", name="node1", action="goto")
        node2 = PlanNode(id="node2", name="node2", action="grasp")
        from brain.planning.state import CommitLevel

        node1.commit_level = CommitLevel.SOFT
        node1.mark_started()
        node1.mark_success()

        node2.commit_level = CommitLevel.SOFT
        node2.mark_started()

        state.add_root(node1)
        state.add_root(node2)

        rollback_nodes = state.get_nodes_to_rollback()
        # node2 在执行中，应该被回滚
        assert node2 in rollback_nodes
