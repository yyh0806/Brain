# -*- coding: utf-8 -*-
"""
信念管理模块 - Belief Management Module

这是认知层实现"自我否定"能力的核心模块。
"""

from brain.cognitive.world_model.belief.belief import Belief
from brain.cognitive.world_model.belief.belief_update_policy import BeliefUpdatePolicy

__all__ = [
    "Belief",
    "BeliefUpdatePolicy",
]








