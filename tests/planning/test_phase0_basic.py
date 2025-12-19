#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 0 基础功能测试

测试能力配置系统、PlanNode、PlanState和Action-level规划器
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from brain.planning.capability import CapabilityRegistry, PlatformAdapter
from brain.planning.plan_state import PlanNode, PlanState, NodeStatus
from brain.planning.action_level import ActionLevelPlanner, WorldModelMock


def test_capability_registry():
    """测试能力注册表"""
    print("\n=== 测试能力注册表 ===")
    
    registry = CapabilityRegistry()
    
    # 测试获取能力
    move_cap = registry.get_capability("move_to")
    assert move_cap is not None, "move_to能力应该存在"
    print(f"✓ move_to能力: {move_cap.description}")
    
    # 测试平台过滤
    ugv_caps = registry.get_capabilities_for_platform("ugv")
    print(f"✓ UGV平台有 {len(ugv_caps)} 个能力")
    
    # 测试参数验证
    valid, errors = registry.validate_parameters("move_to", {"position": {"x": 1, "y": 2}})
    assert valid, f"参数验证应该通过: {errors}"
    print(f"✓ 参数验证通过")
    
    print("✓ 能力注册表测试通过\n")


def test_plan_node():
    """测试PlanNode"""
    print("\n=== 测试PlanNode ===")
    
    # 创建节点
    node = PlanNode(
        id="test_1",
        name="move_to_kitchen",
        action="move_to",
        skill="去厨房",
        task="去厨房拿杯水",
        parameters={"position": {"x": 5, "y": 3}},
        preconditions=["robot.ready == True"],
        expected_effects=["robot.position.near(kitchen)"]
    )
    
    assert node.id == "test_1"
    assert node.is_leaf
    assert node.level == 0
    print(f"✓ 创建节点: {node.name}")
    
    # 测试状态
    node.mark_started()
    assert node.status == NodeStatus.EXECUTING
    print(f"✓ 节点状态: {node.status.value}")
    
    node.mark_success()
    assert node.status == NodeStatus.SUCCESS
    print(f"✓ 节点成功: {node.status.value}")
    
    # 测试HTN结构
    parent = PlanNode(id="parent", name="去厨房", skill="去厨房")
    child1 = PlanNode(id="child1", name="check_door", action="check_door_status")
    child2 = PlanNode(id="child2", name="open_door", action="open_door")
    
    parent.add_child(child1)
    parent.add_child(child2)
    
    assert len(parent.children) == 2
    assert child1.parent == parent
    print(f"✓ HTN结构: 父节点有 {len(parent.children)} 个子节点")
    
    print("✓ PlanNode测试通过\n")


def test_plan_state():
    """测试PlanState"""
    print("\n=== 测试PlanState ===")
    
    state = PlanState()
    
    # 创建根节点
    root = PlanNode(id="root", name="去厨房", skill="去厨房")
    state.add_root(root)
    
    assert len(state.roots) == 1
    assert state.get_node("root") == root
    print(f"✓ 添加根节点: {root.name}")
    
    # 测试状态查询
    pending = state.get_pending_nodes()
    assert len(pending) == 1
    print(f"✓ 待执行节点: {len(pending)}")
    
    # 测试状态更新
    state.update_node_status("root", NodeStatus.EXECUTING)
    executing = state.get_executing_nodes()
    assert len(executing) == 1
    print(f"✓ 执行中节点: {len(executing)}")
    
    stats = state.get_execution_statistics()
    print(f"✓ 执行统计: {stats}")
    
    print("✓ PlanState测试通过\n")


def test_action_level_planner():
    """测试Action-level规划器"""
    print("\n=== 测试Action-level规划器 ===")
    
    registry = CapabilityRegistry()
    adapter = PlatformAdapter(registry)
    world_model = WorldModelMock()
    
    planner = ActionLevelPlanner(
        capability_registry=registry,
        platform_adapter=adapter,
        world_model=world_model,
        platform="ugv"
    )
    
    # 测试规划"去厨房"技能
    nodes = planner.plan_skill(
        skill_name="去厨房",
        parameters={"location": "kitchen"},
        task_name="去厨房拿杯水"
    )
    
    assert len(nodes) > 0, "应该生成操作节点"
    print(f"✓ '去厨房'技能生成了 {len(nodes)} 个操作节点:")
    for node in nodes:
        print(f"  - {node.name} ({node.action})")
    
    # 测试规划"拿杯子"技能
    nodes = planner.plan_skill(
        skill_name="拿杯子",
        parameters={"object": "cup"},
        task_name="去厨房拿杯水"
    )
    
    assert len(nodes) > 0
    print(f"✓ '拿杯子'技能生成了 {len(nodes)} 个操作节点:")
    for node in nodes:
        print(f"  - {node.name} ({node.action})")
    
    print("✓ Action-level规划器测试通过\n")


def test_integration():
    """集成测试：手动创建完整计划"""
    print("\n=== 集成测试：手动创建计划 ===")
    
    # 创建组件
    registry = CapabilityRegistry()
    adapter = PlatformAdapter(registry)
    world_model = WorldModelMock()
    planner = ActionLevelPlanner(registry, adapter, world_model, "ugv")
    
    # 创建PlanState
    plan_state = PlanState()
    
    # 创建任务根节点
    task_root = PlanNode(
        id="task_1",
        name="去厨房拿杯水",
        task="去厨房拿杯水"
    )
    
    # 规划技能并添加为子节点
    skills = ["去厨房", "拿杯子", "回来", "放桌子"]
    
    for skill_name in skills:
        skill_node = PlanNode(
            id=f"skill_{skill_name}",
            name=skill_name,
            skill=skill_name,
            task="去厨房拿杯水"
        )
        
        # 生成操作节点
        if skill_name == "去厨房":
            action_nodes = planner.plan_skill(skill_name, {"location": "kitchen"}, "去厨房拿杯水")
        elif skill_name == "拿杯子":
            action_nodes = planner.plan_skill(skill_name, {"object": "cup"}, "去厨房拿杯水")
        elif skill_name == "回来":
            action_nodes = planner.plan_skill(skill_name, {}, "去厨房拿杯水")
        elif skill_name == "放桌子":
            action_nodes = planner.plan_skill("放物体", {"location": "table"}, "去厨房拿杯水")
        else:
            action_nodes = []
        
        # 添加操作节点为技能节点的子节点
        for action_node in action_nodes:
            skill_node.add_child(action_node)
        
        task_root.add_child(skill_node)
    
    # 添加到PlanState
    plan_state.add_root(task_root)
    
    # 统计
    total_nodes = len(plan_state.nodes)
    print(f"✓ 创建了完整计划:")
    print(f"  - 总节点数: {total_nodes}")
    print(f"  - 技能数: {len(task_root.children)}")
    
    # 打印计划结构
    def print_node(node: PlanNode, indent=0):
        prefix = "  " * indent
        print(f"{prefix}- {node.name} ({node.action or node.skill or node.task})")
        for child in node.children:
            print_node(child, indent + 1)
    
    print("\n计划结构:")
    print_node(task_root)
    
    print("\n✓ 集成测试通过\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0 基础功能测试")
    print("=" * 60)
    
    try:
        test_capability_registry()
        test_plan_node()
        test_plan_state()
        test_action_level_planner()
        test_integration()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
