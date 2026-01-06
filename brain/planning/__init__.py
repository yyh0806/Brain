"""
Brain Planning Layer

规划层模块 - 负责任务规划、技能分解和动作生成

本模块提供三层HTN规划架构：
- TaskLevelPlanner: 理解自然语言指令，提取任务和技能序列
- SkillLevelPlanner: 将任务分解为技能序列
- ActionLevelPlanner: 将技能转换为具体的原子操作

主要组件：
- PlanningOrchestrator: 统一的规划与执行接口
- PlanState/PlanNode: HTN任务树数据结构
- CapabilityRegistry: 能力注册与管理
- IWorldModel: 世界模型接口
"""

# 核心编排器
from brain.planning.orchestrator import PlanningOrchestrator

# 数据结构
from brain.planning.state import PlanState, PlanNode, NodeStatus, CommitLevel

# 能力管理
from brain.planning.capability import CapabilityRegistry, PlatformAdapter, Capability

# 世界模型接口
from brain.planning.interfaces import IWorldModel, Location

# 数据模型
from brain.planning.models import Location, Door, ObjectInfo

__version__ = "0.2.0"
__author__ = "Brain Team"

__all__ = [
    # 核心编排器
    'PlanningOrchestrator',

    # 数据结构
    'PlanState',
    'PlanNode',
    'NodeStatus',
    'CommitLevel',

    # 能力管理
    'CapabilityRegistry',
    'PlatformAdapter',
    'Capability',

    # 世界模型接口
    'IWorldModel',
    'Location',

    # 数据模型
    'Door',
    'ObjectInfo',
]
