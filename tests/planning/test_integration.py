#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整集成测试

测试"去厨房拿杯水"的完整流程
包括动态插入开门操作和搜索操作
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from brain.planning.orchestrator import PlanningOrchestrator


async def test_full_workflow():
    """测试完整工作流"""
    print("=" * 60)
    print("完整集成测试：去厨房拿杯水")
    print("=" * 60)
    
    # 创建规划编排器
    orchestrator = PlanningOrchestrator(platform="ugv")
    
    # 测试1: 只规划
    print("\n[测试1] 规划阶段")
    print("-" * 60)
    
    command = "去厨房拿杯水"
    plan_state = orchestrator.get_plan(command)
    
    print(f"指令: {command}")
    print(f"总节点数: {len(plan_state.nodes)}")
    print(f"根节点: {plan_state.roots[0].name}")
    
    # 打印计划结构
    def print_plan(node, indent=0):
        prefix = "  " * indent
        node_type = node.action or node.skill or node.task or "unknown"
        status = node.status.value
        print(f"{prefix}- {node.name} ({node_type}) [{status}]")
        for child in node.children:
            print_plan(child, indent + 1)
    
    print("\n计划结构:")
    print_plan(plan_state.roots[0])
    
    # 测试2: 规划并执行
    print("\n[测试2] 规划并执行")
    print("-" * 60)
    
    result = await orchestrator.plan_and_execute(command)
    
    print(f"执行结果:")
    print(f"  成功: {result['success']}")
    print(f"  统计: {result['statistics']}")
    print(f"  失败节点: {result['failed_nodes']}")
    
    # 测试3: 验证动态插入
    print("\n[测试3] 验证动态插入")
    print("-" * 60)
    
    # 检查是否插入了开门操作
    plan_state = orchestrator.get_plan(command)
    has_open_door = any(
        node.action == "open_door" 
        for node in plan_state.nodes.values()
    )
    
    print(f"是否插入开门操作: {has_open_door}")
    assert has_open_door, "应该插入开门操作"
    
    # 测试4: 验证HTN结构
    print("\n[测试4] 验证HTN结构")
    print("-" * 60)
    
    root = plan_state.roots[0]
    skills = [child for child in root.children if child.skill]
    print(f"技能数量: {len(skills)}")
    
    for skill in skills:
        actions = [child for child in skill.children if child.action]
        print(f"  技能 '{skill.name}' 有 {len(actions)} 个操作")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)


async def test_failure_recovery():
    """测试失败恢复"""
    print("\n" + "=" * 60)
    print("失败恢复测试")
    print("=" * 60)
    
    orchestrator = PlanningOrchestrator(platform="ugv")
    
    # 模拟门关闭的场景
    orchestrator.world_model.set_door_state("kitchen_door", "closed")
    
    command = "去厨房拿杯水"
    result = await orchestrator.plan_and_execute(command)
    
    print(f"执行结果: {result['success']}")
    print(f"统计: {result['statistics']}")
    
    # 验证是否处理了门关闭的情况
    plan_state = orchestrator.get_plan(command)
    has_check_door = any(
        node.action == "check_door_status"
        for node in plan_state.nodes.values()
    )
    
    print(f"是否插入检查门操作: {has_check_door}")
    assert has_check_door, "应该插入检查门操作"
    
    print("✓ 失败恢复测试通过")


if __name__ == "__main__":
    try:
        # 运行完整工作流测试
        asyncio.run(test_full_workflow())
        
        # 运行失败恢复测试
        asyncio.run(test_failure_recovery())
        
        print("\n" + "=" * 60)
        print("✓ 所有集成测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
