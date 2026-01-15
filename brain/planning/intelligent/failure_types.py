# -*- coding: utf-8 -*-
"""
失败类型定义 - Failure Types

避免循环导入的独立模块
"""

from enum import Enum


class FailureType(Enum):
    """失败类型"""
    PRECONDITION_FAILED = "precondition_failed"  # 前置条件不满足
    EXECUTION_FAILED = "execution_failed"        # 执行失败（机械故障）
    WORLD_STATE_CHANGED = "world_state_changed"   # 世界状态改变
    PERCEPTION_FAILED = "perception_failed"       # 感知失败
