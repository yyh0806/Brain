"""
Unit Tests for PlanValidator

PlanValidator 单元测试
"""

import pytest

from brain.planning.intelligent import PlanValidator, PlanValidation
from brain.planning.state import PlanState, PlanNode, NodeStatus


@pytest.mark.unit
class TestPlanValidator:
    """PlanValidator 测试类"""

    def test_validator_initialization(self, plan_validator):
        """测试验证器初始化"""
        assert plan_validator is not None
        assert plan_validator.max_depth == 20

    def test_validate_simple_plan(self, plan_validator, sample_plan_state):
        """测试验证简单计划"""
        result = plan_validator.validate(sample_plan_state)

        assert isinstance(result, PlanValidation)
        assert result.valid is True
        assert len(result.issues) == 0
        assert result.node_count == 1

    def test_validate_empty_plan(self, plan_validator):
        """测试验证空计划"""
        empty_state = PlanState()
        result = plan_validator.validate(empty_state)

        assert result.valid is True
        assert result.node_count == 0

    def test_validate_plan_with_children(self, plan_validator):
        """测试验证包含子节点的计划"""
        state = PlanState()
        root = PlanNode(id="root", name="root", task="root")
        child1 = PlanNode(id="child1", name="child1", action="goto")
        child2 = PlanNode(id="child2", name="child2", action="grasp")

        root.add_child(child1)
        root.add_child(child2)
        state.add_root(root)

        result = plan_validator.validate(state)
        assert result.valid is True
        assert result.node_count == 3

    def test_detect_cycle(self, plan_validator):
        """测试检测循环依赖"""
        state = PlanState()
        node1 = PlanNode(id="node1", name="node1", action="goto")
        node2 = PlanNode(id="node2", name="node2", action="grasp")

        # 创建循环：node1 -> node2 -> node1
        node1.add_child(node2)
        node2.add_child(node1)  # 循环！

        state.add_root(node1)

        result = plan_validator.validate(state)
        # 应该检测到循环
        assert result.valid is False
        assert any("循环" in issue or "cycle" in issue.lower() for issue in result.issues)

    def test_detect_missing_name(self, plan_validator):
        """测试检测缺少名称的节点"""
        state = PlanState()
        node = PlanNode(id="test", name="", action="goto")
        state.add_root(node)

        result = plan_validator.validate(state)
        assert result.valid is False
        assert any("名称" in issue or "name" in issue.lower() for issue in result.issues)

    def test_check_depth_limit(self, plan_validator):
        """测试深度限制检查"""
        state = PlanState()
        root = PlanNode(id="root", name="root", task="root")

        # 创建一个很深的链
        current = root
        for i in range(25):  # 超过 max_depth (20)
            child = PlanNode(id=f"node_{i}", name=f"node_{i}", action="action")
            current.add_child(child)
            current = child

        state.add_root(root)

        result = plan_validator.validate(state)
        assert result.valid is False
        assert any("深度" in issue or "depth" in issue.lower() for issue in result.issues)

    def test_detect_orphan_nodes(self, plan_validator):
        """测试检测孤儿节点"""
        state = PlanState()

        root = PlanNode(id="root", name="root", task="root")
        orphan = PlanNode(id="orphan", name="orphan", action="goto")

        state.add_root(root)
        # 直接添加孤儿节点到 nodes 字典
        state.nodes[orphan.id] = orphan

        result = plan_validator.validate(state)
        # 应该检测到孤儿节点
        assert len(result.warnings) > 0
        assert any("孤儿" in warning or "orphan" in warning.lower() for warning in result.warnings)

    def test_validation_result_structure(self, plan_validator, sample_plan_state):
        """测试验证结果结构"""
        result = plan_validator.validate(sample_plan_state)

        assert hasattr(result, 'valid')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'node_count')
        assert isinstance(result.issues, list)
        assert isinstance(result.warnings, list)

    def test_add_issue(self):
        """测试添加问题"""
        result = PlanValidation(
            valid=True,
            issues=[],
            warnings=[]
        )

        result.add_issue("Test issue")

        assert result.valid is False
        assert len(result.issues) == 1
        assert result.issues[0] == "Test issue"

    def test_add_warning(self):
        """测试添加警告"""
        result = PlanValidation(
            valid=True,
            issues=[],
            warnings=[]
        )

        result.add_warning("Test warning")

        assert result.valid is True  # 警告不影响有效性
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"


@pytest.mark.unit
class TestPlanValidation:
    """PlanValidation 测试类"""

    def test_validation_result_default_values(self):
        """测试验证结果默认值"""
        result = PlanValidation(
            valid=True,
            issues=[],
            warnings=[]
        )

        assert result.valid is True
        assert result.node_count == 0
