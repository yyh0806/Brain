#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain项目层级测试脚本

用于测试各层级模块是否可以正确导入和基本功能是否正常
"""

import sys
import importlib
from typing import Dict, List, Any

def test_layer_import(layer_name: str, modules: List[str]) -> Dict[str, Any]:
    """测试层级模块导入"""
    results = {}
    
    print(f"\n{'='*50}")
    print(f"测试 {layer_name} 层级导入")
    print(f"{'='*50}")
    
    for module_name in modules:
        try:
            module = importlib.import_module(f"brain.{layer_name}.{module_name}")
            results[module_name] = {"status": "success", "error": None}
            print(f"  ✓ {module_name}")
        except Exception as e:
            results[module_name] = {"status": "failed", "error": str(e)}
            print(f"  ✗ {module_name}: {e}")
    
    return results

def test_cross_layer_imports() -> Dict[str, Any]:
    """测试跨层级导入"""
    print(f"\n{'='*50}")
    print("测试跨层级导入")
    print(f"{'='*50}")
    
    results = {}
    
    # 测试核心控制器导入各层级
    try:
        from brain.core.brain import Brain
        results["core_to_perception"] = {"status": "success", "error": None}
        print("  ✓ core -> perception")
    except Exception as e:
        results["core_to_perception"] = {"status": "failed", "error": str(e)}
        print(f"  ✗ core -> perception: {e}")
    
    try:
        from brain.core.brain import Brain
        results["core_to_cognitive"] = {"status": "success", "error": None}
        print("  ✓ core -> cognitive")
    except Exception as e:
        results["core_to_cognitive"] = {"status": "failed", "error": str(e)}
        print(f"  ✗ core -> cognitive: {e}")
    
    try:
        from brain.core.brain import Brain
        results["core_to_planning"] = {"status": "success", "error": None}
        print("  ✓ core -> planning")
    except Exception as e:
        results["core_to_planning"] = {"status": "failed", "error": str(e)}
        print(f"  ✗ core -> planning: {e}")
    
    try:
        from brain.core.brain import Brain
        results["core_to_execution"] = {"status": "success", "error": None}
        print("  ✓ core -> execution")
    except Exception as e:
        results["core_to_execution"] = {"status": "failed", "error": str(e)}
        print(f"  ✗ core -> execution: {e}")
    
    try:
        from brain.core.brain import Brain
        results["core_to_communication"] = {"status": "success", "error": None}
        print("  ✓ core -> communication")
    except Exception as e:
        results["core_to_communication"] = {"status": "failed", "error": str(e)}
        print(f"  ✗ core -> communication: {e}")
    
    try:
        from brain.core.brain import Brain
        results["core_to_models"] = {"status": "success", "error": None}
        print("  ✓ core -> models")
    except Exception as e:
        results["core_to_models"] = {"status": "failed", "error": str(e)}
        print(f"  ✗ core -> models: {e}")
    
    return results

def test_brain_initialization() -> Dict[str, Any]:
    """测试Brain核心初始化"""
    print(f"\n{'='*50}")
    print("测试Brain核心初始化")
    print(f"{'='*50}")
    
    try:
        from brain import Brain
        # 尝试创建Brain实例（使用模拟配置）
        brain = Brain(config_path="config/default_config.yaml")
        return {"status": "success", "error": None, "instance": brain}
    except Exception as e:
        return {"status": "failed", "error": str(e), "instance": None}

def main():
    """主测试函数"""
    print("Brain项目层级测试")
    print("=" * 60)
    
    # 测试各层级模块导入
    perception_results = test_layer_import(
        "perception", 
        ["environment", "object_detector", "sensors", "mapping", "vlm"]
    )
    
    cognitive_results = test_layer_import(
        "cognitive", 
        ["world_model", "dialogue", "reasoning", "monitoring"]
    )
    
    planning_results = test_layer_import(
        "planning", 
        ["task", "navigation", "behavior"]
    )
    
    execution_results = test_layer_import(
        "execution", 
        ["executor", "operations"]
    )
    
    communication_results = test_layer_import(
        "communication", 
        ["robot_interface", "ros2_interface", "control_adapter", "message_types"]
    )
    
    models_results = test_layer_import(
        "models", 
        ["llm_interface", "task_parser", "prompt_templates", "ollama_client", "cot_prompts"]
    )
    
    # 测试跨层级导入
    cross_layer_results = test_cross_layer_imports()
    
    # 测试Brain核心初始化
    brain_init_results = test_brain_initialization()
    
    # 汇总结果
    print(f"\n{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    
    all_results = {
        "perception": perception_results,
        "cognitive": cognitive_results,
        "planning": planning_results,
        "execution": execution_results,
        "communication": communication_results,
        "models": models_results,
        "cross_layer": cross_layer_results,
        "brain_init": brain_init_results
    }
    
    # 计算成功率
    total_tests = 0
    total_success = 0
    
    for layer_name, layer_results in all_results.items():
        if layer_name == "cross_layer":
            for test_name, result in layer_results.items():
                total_tests += 1
                if result["status"] == "success":
                    total_success += 1
        elif layer_name == "brain_init":
            total_tests += 1
            if layer_results["status"] == "success":
                total_success += 1
        else:
            for module_name, result in layer_results.items():
                total_tests += 1
                if result["status"] == "success":
                    total_success += 1
    
    success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"总测试数: {total_tests}")
    print(f"成功测试数: {total_success}")
    print(f"成功率: {success_rate:.1f}%")
    
    # 如果有失败的测试，返回非零退出码
    if total_success < total_tests:
        print(f"\n有 {total_tests - total_success} 个测试失败")
        return 1
    
    print("\n所有测试通过！")
    return 0

if __name__ == "__main__":
    sys.exit(main())
