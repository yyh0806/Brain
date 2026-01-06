# -*- coding: utf-8 -*-
"""Brain核心模块

这个模块提供了Brain系统的核心组件，包括：
- Brain: 主要的大脑系统
- PlanningOrchestrator: 规划编排器
- Executor: 执行器
- SystemMonitor: 系统监控器

导入示例:
    from brain.core import Brain, PlanningOrchestrator, Executor, SystemMonitor
    from brain.core.brain import Brain  # 直接导入
"""

# 使用延迟导入来避免循环依赖
import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.core.brain import Brain
    from brain.planning.orchestrator import PlanningOrchestrator
    from brain.execution.executor import Executor
    from brain.core.monitor import SystemMonitor

# 定义__all__以明确导出的内容
__all__ = [
    "Brain",
    "PlanningOrchestrator",
    "Executor",
    "SystemMonitor",
    "get_core_components"
]

def _import_core_module(module_name: str, class_name: str):
    """延迟导入核心模块"""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"无法导入核心模块 {module_name}.{class_name}: {e}")

# 延迟导入属性
def __getattr__(name: str):
    """延迟导入属性，避免循环依赖"""
    if name == "Brain":
        return _import_core_module("brain.core.brain", "Brain")
    elif name == "PlanningOrchestrator":
        return _import_core_module("brain.planning.orchestrator", "PlanningOrchestrator")
    elif name == "Executor":
        return _import_core_module("brain.execution.executor", "Executor")
    elif name == "SystemMonitor":
        return _import_core_module("brain.core.monitor", "SystemMonitor")
    else:
        raise AttributeError(f"模块 {__name__} 没有属性 {name}")

def get_core_components():
    """获取所有核心组件

    Returns:
        dict: 包含所有核心组件的字典
    """
    return {
        "Brain": __getattr__("Brain"),
        "PlanningOrchestrator": __getattr__("PlanningOrchestrator"),
        "Executor": __getattr__("Executor"),
        "SystemMonitor": __getattr__("SystemMonitor")
    }

