"""
能力注册表 - Capability Registry

负责加载、管理和查询操作能力定义
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Capability:
    """能力定义"""
    name: str
    type: str  # movement, manipulation, perception, control
    description: str
    parameters: Dict[str, List[str]]  # required, optional
    preconditions: List[str]
    postconditions: List[str]
    platforms: List[str]  # drone, ugv, usv
    default_duration: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """
    能力注册表
    
    从YAML配置文件加载能力定义，提供查询接口
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化能力注册表
        
        Args:
            config_path: YAML配置文件路径，如果为None则使用默认路径
        """
        self.capabilities: Dict[str, Capability] = {}
        
        # 默认配置文件路径
        if config_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
            config_path = base_path / "config" / "planning" / "capability_config.yaml"
        
        self.config_path = Path(config_path)
        self._load_capabilities()
        
        logger.info(f"CapabilityRegistry 初始化完成，加载了 {len(self.capabilities)} 个能力")
    
    def _load_capabilities(self):
        """从YAML文件加载能力定义"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}")
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            capabilities_data = config.get('capabilities', [])
            
            for cap_data in capabilities_data:
                capability = Capability(
                    name=cap_data['name'],
                    type=cap_data['type'],
                    description=cap_data.get('description', ''),
                    parameters=cap_data.get('parameters', {}),
                    preconditions=cap_data.get('preconditions', []),
                    postconditions=cap_data.get('postconditions', []),
                    platforms=cap_data.get('platforms', []),
                    default_duration=cap_data.get('default_duration', 5.0),
                    metadata=cap_data.get('metadata', {})
                )
                
                self.capabilities[capability.name] = capability
                
        except Exception as e:
            logger.error(f"加载能力配置失败: {e}")
            raise
    
    def get_capability(self, name: str) -> Optional[Capability]:
        """
        获取能力定义
        
        Args:
            name: 能力名称
            
        Returns:
            Capability对象，如果不存在则返回None
        """
        return self.capabilities.get(name)
    
    def has_capability(self, name: str) -> bool:
        """
        检查能力是否存在
        
        Args:
            name: 能力名称
            
        Returns:
            是否存在
        """
        return name in self.capabilities
    
    def list_capabilities(
        self, 
        platform: Optional[str] = None,
        capability_type: Optional[str] = None
    ) -> List[Capability]:
        """
        列出能力
        
        Args:
            platform: 平台过滤（drone, ugv, usv）
            capability_type: 类型过滤（movement, manipulation, perception, control）
            
        Returns:
            能力列表
        """
        result = list(self.capabilities.values())
        
        if platform:
            result = [cap for cap in result if platform in cap.platforms]
        
        if capability_type:
            result = [cap for cap in result if cap.type == capability_type]
        
        return result
    
    def get_capabilities_for_platform(self, platform: str) -> List[Capability]:
        """
        获取指定平台的所有能力
        
        Args:
            platform: 平台类型（drone, ugv, usv）
            
        Returns:
            能力列表
        """
        return self.list_capabilities(platform=platform)
    
    def get_capabilities_by_type(self, capability_type: str) -> List[Capability]:
        """
        获取指定类型的所有能力
        
        Args:
            capability_type: 能力类型（movement, manipulation, perception, control）
            
        Returns:
            能力列表
        """
        return self.list_capabilities(capability_type=capability_type)
    
    def validate_parameters(
        self, 
        capability_name: str, 
        parameters: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        验证参数是否满足能力要求
        
        Args:
            capability_name: 能力名称
            parameters: 提供的参数
            
        Returns:
            (是否有效, 错误信息列表)
        """
        capability = self.get_capability(capability_name)
        if not capability:
            return False, [f"未知能力: {capability_name}"]
        
        errors = []
        required = capability.parameters.get('required', [])
        
        # 检查必需参数
        for param in required:
            if param not in parameters:
                errors.append(f"缺少必需参数: {param}")
        
        return len(errors) == 0, errors
    
    def reload(self):
        """重新加载配置文件"""
        self.capabilities.clear()
        self._load_capabilities()
        logger.info("能力配置已重新加载")
