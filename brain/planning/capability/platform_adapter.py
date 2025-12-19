"""
平台适配器 - Platform Adapter

根据平台类型过滤和适配可用能力
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .capability_registry import CapabilityRegistry, Capability


class PlatformAdapter:
    """
    平台适配器
    
    根据平台类型过滤可用能力，适配操作参数
    """
    
    def __init__(self, registry: CapabilityRegistry):
        """
        初始化平台适配器
        
        Args:
            registry: 能力注册表
        """
        self.registry = registry
        logger.info("PlatformAdapter 初始化完成")
    
    def get_available_capabilities(self, platform: str) -> List[Capability]:
        """
        获取指定平台的可用能力
        
        Args:
            platform: 平台类型（drone, ugv, usv）
            
        Returns:
            可用能力列表
        """
        return self.registry.get_capabilities_for_platform(platform)
    
    def can_perform(self, platform: str, capability_name: str) -> bool:
        """
        检查平台是否能执行指定能力
        
        Args:
            platform: 平台类型
            capability_name: 能力名称
            
        Returns:
            是否能执行
        """
        capability = self.registry.get_capability(capability_name)
        if not capability:
            return False
        
        return platform in capability.platforms
    
    def adapt_parameters(
        self,
        platform: str,
        capability_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        适配参数到指定平台
        
        Args:
            platform: 平台类型
            capability_name: 能力名称
            parameters: 原始参数
            
        Returns:
            适配后的参数
        """
        capability = self.registry.get_capability(capability_name)
        if not capability:
            logger.warning(f"未知能力: {capability_name}")
            return parameters
        
        # 基础适配：根据平台调整默认值
        adapted = parameters.copy()
        
        # 平台特定的参数适配
        platform_defaults = {
            "drone": {
                "speed": 5.0,
                "altitude": 10.0
            },
            "ugv": {
                "speed": 2.0,
                "grip_force": 50.0
            },
            "usv": {
                "speed": 3.0,
                "depth": 1.0
            }
        }
        
        defaults = platform_defaults.get(platform, {})
        for key, value in defaults.items():
            if key not in adapted:
                adapted[key] = value
        
        return adapted
    
    def get_capability_info(self, platform: str) -> Dict[str, Any]:
        """
        获取平台的完整能力信息
        
        Args:
            platform: 平台类型
            
        Returns:
            能力信息字典
        """
        capabilities = self.get_available_capabilities(platform)
        
        return {
            "platform": platform,
            "total_capabilities": len(capabilities),
            "by_type": {
                "movement": len([c for c in capabilities if c.type == "movement"]),
                "manipulation": len([c for c in capabilities if c.type == "manipulation"]),
                "perception": len([c for c in capabilities if c.type == "perception"]),
                "control": len([c for c in capabilities if c.type == "control"])
            },
            "capabilities": [c.name for c in capabilities]
        }
