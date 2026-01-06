"""
Unit Tests for CapabilityRegistry

CapabilityRegistry 单元测试
"""

import pytest

from brain.planning.capability import CapabilityRegistry


@pytest.mark.unit
class TestCapabilityRegistry:
    """CapabilityRegistry 测试类"""

    def test_initialization(self, capability_registry):
        """测试初始化"""
        assert capability_registry is not None
        assert len(capability_registry.capabilities) > 0

    def test_get_capability(self, capability_registry):
        """测试获取能力"""
        move_cap = capability_registry.get_capability("move_to")
        assert move_cap is not None
        assert move_cap.name == "move_to"

        not_found = capability_registry.get_capability("nonexistent_capability")
        assert not_found is None

    def test_has_capability(self, capability_registry):
        """测试检查能力存在性"""
        assert capability_registry.has_capability("move_to") is True
        assert capability_registry.has_capability("nonexistent") is False

    def test_list_capabilities_all(self, capability_registry):
        """测试列出所有能力"""
        all_caps = capability_registry.list_capabilities()
        assert len(all_caps) > 0

    def test_list_capabilities_by_platform(self, capability_registry):
        """测试按平台列出能力"""
        ugv_caps = capability_registry.list_capabilities(platform="ugv")
        assert len(ugv_caps) > 0

        # 验证所有返回的能力都支持 ugv 平台
        for cap in ugv_caps:
            assert "ugv" in cap.platforms

    def test_list_capabilities_by_type(self, capability_registry):
        """测试按类型列出能力"""
        movement_caps = capability_registry.list_capabilities(capability_type="movement")
        assert len(movement_caps) > 0

        # 验证所有返回的能力都是 movement 类型
        for cap in movement_caps:
            assert cap.type == "movement"

    def test_get_capabilities_for_platform(self, capability_registry):
        """测试获取平台特定能力"""
        ugv_caps = capability_registry.get_capabilities_for_platform("ugv")
        assert len(ugv_caps) > 0

    def test_get_capabilities_by_type_method(self, capability_registry):
        """测试获取类型特定能力"""
        manipulation_caps = capability_registry.get_capabilities_by_type("manipulation")
        assert len(manipulation_caps) > 0

    def test_validate_parameters_valid(self, capability_registry):
        """测试参数验证 - 有效参数"""
        valid, errors = capability_registry.validate_parameters(
            "move_to",
            {"position": {"x": 1.0, "y": 2.0, "z": 0.0}}
        )
        assert valid is True
        assert len(errors) == 0

    def test_validate_parameters_missing_required(self, capability_registry):
        """测试参数验证 - 缺少必需参数"""
        valid, errors = capability_registry.validate_parameters(
            "move_to",
            {}  # 缺少 position 参数
        )
        assert valid is False
        assert len(errors) > 0

    def test_validate_parameters_unknown_capability(self, capability_registry):
        """测试参数验证 - 未知能力"""
        valid, errors = capability_registry.validate_parameters(
            "unknown_capability",
            {"param": "value"}
        )
        assert valid is False
        assert len(errors) > 0

    def test_capability_structure(self, capability_registry):
        """测试能力数据结构"""
        move_cap = capability_registry.get_capability("move_to")
        assert move_cap.name == "move_to"
        assert move_cap.type in ["movement", "manipulation", "perception", "control"]
        assert move_cap.description is not None
        assert isinstance(move_cap.platforms, list)
        assert isinstance(move_cap.parameters, dict)
        assert isinstance(move_cap.preconditions, list)
        assert isinstance(move_cap.postconditions, list)

    def test_reload(self, capability_registry):
        """测试重新加载配置"""
        # 获取初始能力数量
        initial_count = len(capability_registry.capabilities)

        # 重新加载
        capability_registry.reload()

        # 验证能力数量不变
        assert len(capability_registry.capabilities) == initial_count
