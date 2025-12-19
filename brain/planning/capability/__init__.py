"""
指令能力配置系统

提供能力定义、注册、查询和平台适配功能
"""

from .capability_registry import CapabilityRegistry
from .platform_adapter import PlatformAdapter

__all__ = [
    "CapabilityRegistry",
    "PlatformAdapter",
]
