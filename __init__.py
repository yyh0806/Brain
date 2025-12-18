"""
Brain - 无人系统任务分解大脑

一个智能任务规划与执行系统，用于无人机、无人车、无人船的自主任务管理。
通过大语言模型理解自然语言指令，分解为可执行的原子操作序列，
并提供完整的错误回退与动态重规划能力。

主要组件:
- Core: 核心大脑系统、任务规划器、执行引擎
- Perception: 感知模块，处理传感器数据
- LLM: 大语言模型接口，自然语言理解
- Operations: 原子操作定义
- Recovery: 错误处理与恢复机制
- State: 状态管理系统
- Communication: 机器人通信接口
"""

__version__ = "1.0.0"
__author__ = "Brain Team"

from brain.core.brain import Brain
from brain.planning.task.task_planner import TaskPlanner
from brain.execution.executor import Executor
from brain.core.monitor import SystemMonitor

__all__ = [
    "Brain",
    "TaskPlanner",
    "Executor",
    "SystemMonitor",
    "__version__",
]

