#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置加载器
支持多层级配置覆盖和模块化配置管理
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ConfigLoadOptions:
    """配置加载选项"""
    platform: Optional[str] = None
    environment: Optional[str] = None
    user: Optional[str] = None
    config_root: str = "config"
    merge_strategy: str = "override"  # override, merge, append
    validate_schema: bool = True
    env_prefix: str = "BRAIN_"


class ConfigLoader:
    """统一配置加载器"""

    def __init__(self, options: Optional[ConfigLoadOptions] = None):
        self.options = options or ConfigLoadOptions()

        # 使用绝对路径
        config_root = self.options.config_root
        if not Path(config_root).is_absolute():
            # 如果是相对路径，基于当前文件所在目录解析
            current_dir = Path(__file__).parent
            config_root = current_dir / config_root

        self.config_root = Path(config_root).resolve()
        self._schema_cache = {}

    def load_config(self) -> Dict[str, Any]:
        """
        加载完整配置
        按优先级顺序合并配置
        """
        # 定义配置加载顺序（从低优先级到高优先级）
        config_layers = [
            ("global", self._load_global_config),
            ("modules", self._load_modules_config),
            ("environments", self._load_environment_config),
            ("platforms", self._load_platform_config),
            ("users", self._load_user_config),
        ]

        merged_config = {}

        for layer_name, load_func in config_layers:
            try:
                layer_config = load_func()
                if layer_config:
                    merged_config = self._merge_configs(
                        merged_config,
                        layer_config,
                        strategy=self.options.merge_strategy
                    )
                    logger.debug(f"已加载配置层: {layer_name}")
            except Exception as e:
                logger.error(f"加载配置层 {layer_name} 失败: {e}")

        # 应用环境变量覆盖
        env_overrides = self._load_env_overrides()
        if env_overrides:
            merged_config = self._merge_configs(
                merged_config,
                env_overrides,
                strategy="override"
            )

        # 验证配置
        if self.options.validate_schema:
            self._validate_config(merged_config)

        return merged_config

    def _load_global_config(self) -> Dict[str, Any]:
        """加载全局配置"""
        config = {}

        # 加载默认配置
        defaults_file = self.config_root / "global" / "defaults.yaml"
        logger.debug(f"尝试加载默认配置: {defaults_file}")
        if defaults_file.exists():
            config.update(self._load_yaml_file(defaults_file))
            logger.debug("默认配置加载成功")
        else:
            logger.warning(f"默认配置文件不存在: {defaults_file}")

        # 加载系统配置
        system_file = self.config_root / "global" / "system.yaml"
        logger.debug(f"尝试加载系统配置: {system_file}")
        if system_file.exists():
            config.update(self._load_yaml_file(system_file))
            logger.debug("系统配置加载成功")
        else:
            logger.warning(f"系统配置文件不存在: {system_file}")

        return config

    def _load_modules_config(self) -> Dict[str, Any]:
        """加载模块配置"""
        modules_dir = self.config_root / "modules"
        if not modules_dir.exists():
            return {}

        config = {"modules": {}}

        # 遍历所有模块目录
        for module_dir in modules_dir.iterdir():
            if module_dir.is_dir():
                module_config = {}

                # 加载模块内的所有yaml文件
                for yaml_file in module_dir.glob("*.yaml"):
                    file_config = self._load_yaml_file(yaml_file)
                    module_config.update(file_config)

                config["modules"][module_dir.name] = module_config

        return config

    def _load_environment_config(self) -> Dict[str, Any]:
        """加载环境配置"""
        if not self.options.environment:
            return {}

        env_dir = self.config_root / "environments" / self.options.environment
        if not env_dir.exists():
            logger.warning(f"环境配置目录不存在: {env_dir}")
            return {}

        config = {}

        # 加载环境内的所有yaml文件
        for yaml_file in env_dir.glob("*.yaml"):
            file_config = self._load_yaml_file(yaml_file)
            config.update(file_config)

        return config

    def _load_platform_config(self) -> Dict[str, Any]:
        """加载平台配置"""
        if not self.options.platform:
            return {}

        platform_dir = self.config_root / "platforms" / self.options.platform
        if not platform_dir.exists():
            logger.warning(f"平台配置目录不存在: {platform_dir}")
            return {}

        config = {}

        # 加载平台内的所有yaml文件
        for yaml_file in platform_dir.glob("*.yaml"):
            file_config = self._load_yaml_file(yaml_file)
            config.update(file_config)

        return config

    def _load_user_config(self) -> Dict[str, Any]:
        """加载用户配置"""
        if not self.options.user:
            return {}

        user_dir = self.config_root / "users" / self.options.user
        if not user_dir.exists():
            logger.warning(f"用户配置目录不存在: {user_dir}")
            return {}

        config = {}

        # 加载用户目录下的所有yaml文件
        for yaml_file in user_dir.glob("*.yaml"):
            file_config = self._load_yaml_file(yaml_file)
            config.update(file_config)

        return config

    def _load_env_overrides(self) -> Dict[str, Any]:
        """加载环境变量覆盖"""
        overrides = {}
        prefix = self.options.env_prefix

        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 移除前缀并转换为小写
                config_key = key[len(prefix):].lower()

                # 处理嵌套键（使用双下划线分隔）
                if "__" in config_key:
                    self._set_nested_value(overrides, config_key.split("__"), value)
                else:
                    # 尝试转换值类型
                    overrides[config_key] = self._convert_env_value(value)

        return overrides

    def _set_nested_value(self, config: Dict, key_parts: List[str], value: Any):
        """设置嵌套字典值"""
        current = config
        for part in key_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[key_parts[-1]] = value

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """转换环境变量值类型"""
        # 布尔值
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # 整数
        try:
            return int(value)
        except ValueError:
            pass

        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # JSON解析
        try:
            return json.loads(value)
        except ValueError:
            pass

        # 默认返回字符串
        return value

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any],
                      strategy: str = "override") -> Dict[str, Any]:
        """
        合并配置字典
        """
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                if strategy == "merge":
                    result[key] = self._merge_configs(result[key], value, strategy)
                else:
                    result[key] = value
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                if strategy == "append":
                    result[key].extend(value)
                else:
                    result[key] = value
            else:
                result[key] = value

        return result

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"加载YAML文件失败 {file_path}: {e}")
            return {}

    def _validate_config(self, config: Dict[str, Any]):
        """验证配置"""
        # TODO: 实现配置验证逻辑
        pass

    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """获取配置模式"""
        if schema_name not in self._schema_cache:
            schema_file = self.config_root / "global" / "schemas" / f"{schema_name}_schema.json"
            if schema_file.exists():
                with open(schema_file, 'r', encoding='utf-8') as f:
                    self._schema_cache[schema_name] = json.load(f)
            else:
                self._schema_cache[schema_name] = {}

        return self._schema_cache[schema_name]


def load_config(platform: Optional[str] = None,
               environment: Optional[str] = None,
               user: Optional[str] = None,
               config_root: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：加载配置
    """
    # 如果没有指定config_root，使用当前文件所在目录
    if config_root is None:
        config_root = str(Path(__file__).parent)

    options = ConfigLoadOptions(
        platform=platform,
        environment=environment,
        user=user,
        config_root=config_root
    )
    loader = ConfigLoader(options)
    return loader.load_config()


if __name__ == "__main__":
    # 测试配置加载
    logging.basicConfig(level=logging.DEBUG)

    config = load_config(
        platform="ugv",
        environment="real_world",
        user="yangyuhui"
    )

    print("加载的配置：")
    print(json.dumps(config, indent=2, ensure_ascii=False))