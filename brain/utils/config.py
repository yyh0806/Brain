"""
配置管理 - Config Manager

负责:
- 加载配置文件
- 配置验证
- 运行时配置访问
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from loguru import logger


class ConfigManager:
    """
    配置管理器
    
    管理Brain系统的配置
    """
    
    DEFAULT_CONFIG_PATH = "config/default_config.yaml"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置"""
        # 加载默认配置
        default_path = Path(__file__).parent.parent.parent / self.DEFAULT_CONFIG_PATH
        if default_path.exists():
            self.config = self._load_yaml(default_path)
            logger.info(f"加载默认配置: {default_path}")
        else:
            logger.warning(f"默认配置文件不存在: {default_path}")
            self.config = self._get_builtin_defaults()
        
        # 加载自定义配置（覆盖默认）
        if self.config_path:
            custom_path = Path(self.config_path)
            if custom_path.exists():
                custom_config = self._load_yaml(custom_path)
                self.config = self._merge_config(self.config, custom_config)
                logger.info(f"加载自定义配置: {custom_path}")
            else:
                logger.warning(f"自定义配置文件不存在: {self.config_path}")
        
        # 从环境变量覆盖
        self._override_from_env()
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载配置文件失败: {path} - {e}")
            return {}
    
    def _get_builtin_defaults(self) -> Dict[str, Any]:
        """获取内置默认配置"""
        return {
            "system": {
                "name": "Brain-Autonomous-System",
                "version": "1.0.0",
                "log_level": "INFO"
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "max_tokens": 4096,
                "temperature": 0.1,
                "timeout": 60,
                "retry_count": 3
            },
            "perception": {
                "update_rate": 10.0,
                "sensors": {}
            },
            "platforms": {
                "drone": {"enabled": True, "max_speed": 15.0},
                "ugv": {"enabled": True, "max_speed": 5.0},
                "usv": {"enabled": True, "max_speed": 8.0}
            },
            "planning": {
                "max_retries": 3,
                "timeout": 300,
                "checkpoint_interval": 5
            },
            "recovery": {
                "enabled": True,
                "max_rollback_steps": 10,
                "replan_on_failure": True
            },
            "communication": {
                "protocol": "zmq",
                "command_port": 5555,
                "telemetry_port": 5556,
                "simulation": True
            },
            "safety": {
                "geofence": {"enabled": False}
            }
        }
    
    def _merge_config(
        self, 
        base: Dict[str, Any], 
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """合并配置"""
        result = dict(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _override_from_env(self):
        """从环境变量覆盖配置"""
        env_mappings = {
            "BRAIN_LOG_LEVEL": "system.log_level",
            "BRAIN_LLM_PROVIDER": "llm.provider",
            "BRAIN_LLM_MODEL": "llm.model",
            "OPENAI_API_KEY": "llm.api_key",
            "ANTHROPIC_API_KEY": "llm.api_key",
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self.set(config_path, value)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径 (如 "llm.model")
            default: 默认值
            
        Returns:
            配置值
        """
        parts = path.split(".")
        value = self.config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """
        设置配置值
        
        Args:
            path: 配置路径
            value: 值
        """
        parts = path.split(".")
        config = self.config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置段
        
        Args:
            section: 配置段名称
            
        Returns:
            配置字典
        """
        return self.get(section, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """获取完整配置"""
        return dict(self.config)
    
    def save(self, path: Optional[str] = None):
        """
        保存配置
        
        Args:
            path: 保存路径(可选)
        """
        save_path = path or self.config_path
        if not save_path:
            logger.warning("未指定保存路径")
            return
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置已保存: {save_path}")
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
    
    def validate(self) -> bool:
        """
        验证配置
        
        Returns:
            bool: 配置是否有效
        """
        required_sections = ["system", "llm", "platforms", "planning", "recovery"]
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"缺少必需配置段: {section}")
                return False
        
        return True

