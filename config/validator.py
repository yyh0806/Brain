#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证工具
验证配置文件的结构、数据类型和值的有效性
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """配置验证错误"""
    path: str  # 配置路径，如 "llm.temperature"
    message: str
    error_type: str  # required, type, range, pattern, custom
    current_value: Any = None
    expected_value: Any = None


class ConfigValidator:
    """配置验证器"""

    def __init__(self, schema_root: str = "config/global/schemas"):
        self.schema_root = Path(schema_root)
        self.schemas = {}
        self._load_schemas()

    def _load_schemas(self):
        """加载所有配置模式"""
        if not self.schema_root.exists():
            logger.warning(f"模式目录不存在: {self.schema_root}")
            return

        for schema_file in self.schema_root.glob("*_schema.json"):
            schema_name = schema_file.stem.replace("_schema", "")
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    self.schemas[schema_name] = json.load(f)
                    logger.debug(f"已加载模式: {schema_name}")
            except Exception as e:
                logger.error(f"加载模式失败 {schema_file}: {e}")

    def validate(self, config: Dict[str, Any], schema_name: Optional[str] = None) -> Tuple[bool, List[ValidationError]]:
        """
        验证配置
        返回 (是否有效, 错误列表)
        """
        errors = []

        # 如果指定了模式，使用指定模式验证
        if schema_name and schema_name in self.schemas:
            schema_errors = self._validate_against_schema(
                config,
                self.schemas[schema_name],
                ""
            )
            errors.extend(schema_errors)
        else:
            # 否则使用所有适用的模式验证
            for schema_name, schema in self.schemas.items():
                # 检查配置是否包含该模式的根键
                if schema_name in config or self._schema_matches_config(schema, config):
                    schema_errors = self._validate_against_schema(
                        config.get(schema_name, {}),
                        schema,
                        f"{schema_name}."
                    )
                    errors.extend(schema_errors)

        # 执行内置验证规则
        builtin_errors = self._validate_builtin_rules(config)
        errors.extend(builtin_errors)

        return len(errors) == 0, errors

    def _validate_against_schema(self, config: Any, schema: Dict[str, Any], path: str) -> List[ValidationError]:
        """根据模式验证配置"""
        errors = []

        # 获取类型
        expected_type = schema.get("type")
        if expected_type:
            type_error = self._validate_type(config, expected_type, path)
            if type_error:
                errors.append(type_error)
                return errors  # 类型错误时不继续验证其他属性

        # 验证必填项
        if isinstance(config, dict) and isinstance(schema, dict):
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in config:
                    errors.append(ValidationError(
                        path=f"{path}{field}",
                        message=f"缺少必填字段: {field}",
                        error_type="required"
                    ))

            # 验证每个属性
            properties = schema.get("properties", {})
            for field, value in config.items():
                if field in properties:
                    field_schema = properties[field]
                    field_errors = self._validate_against_schema(
                        value,
                        field_schema,
                        f"{path}{field}."
                    )
                    errors.extend(field_errors)

        # 验证枚举值
        if "enum" in schema and config not in schema["enum"]:
            errors.append(ValidationError(
                path=path.rstrip("."),
                message=f"值必须在枚举列表中: {schema['enum']}",
                error_type="enum",
                current_value=config,
                expected_value=schema["enum"]
            ))

        # 验证数值范围
        if isinstance(config, (int, float)):
            if "minimum" in schema and config < schema["minimum"]:
                errors.append(ValidationError(
                    path=path.rstrip("."),
                    message=f"值必须大于等于 {schema['minimum']}",
                    error_type="range",
                    current_value=config,
                    expected_value=f">= {schema['minimum']}"
                ))

            if "maximum" in schema and config > schema["maximum"]:
                errors.append(ValidationError(
                    path=path.rstrip("."),
                    message=f"值必须小于等于 {schema['maximum']}",
                    error_type="range",
                    current_value=config,
                    expected_value=f"<= {schema['maximum']}"
                ))

        # 验证字符串长度
        if isinstance(config, str):
            if "minLength" in schema and len(config) < schema["minLength"]:
                errors.append(ValidationError(
                    path=path.rstrip("."),
                    message=f"字符串长度必须大于等于 {schema['minLength']}",
                    error_type="range",
                    current_value=len(config),
                    expected_value=f">= {schema['minLength']}"
                ))

            if "maxLength" in schema and len(config) > schema["maxLength"]:
                errors.append(ValidationError(
                    path=path.rstrip("."),
                    message=f"字符串长度必须小于等于 {schema['maxLength']}",
                    error_type="range",
                    current_value=len(config),
                    expected_value=f"<= {schema['maxLength']}"
                ))

            # 验证正则表达式模式
            if "pattern" in schema:
                pattern = re.compile(schema["pattern"])
                if not pattern.match(config):
                    errors.append(ValidationError(
                        path=path.rstrip("."),
                        message=f"字符串必须匹配模式: {schema['pattern']}",
                        error_type="pattern",
                        current_value=config,
                        expected_value=schema["pattern"]
                    ))

        return errors

    def _validate_type(self, value: Any, expected_type: str, path: str) -> Optional[ValidationError]:
        """验证值类型"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            return ValidationError(
                path=path.rstrip("."),
                message=f"类型错误，期望 {expected_type}，实际 {type(value).__name__}",
                error_type="type",
                current_value=type(value).__name__,
                expected_value=expected_type
            )

        return None

    def _schema_matches_config(self, schema: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """检查模式是否适用于当前配置"""
        # 检查模式中的required字段是否在配置中
        required_fields = schema.get("required", [])
        return any(field in config for field in required_fields)

    def _validate_builtin_rules(self, config: Dict[str, Any]) -> List[ValidationError]:
        """执行内置验证规则"""
        errors = []

        # LLM配置验证
        if "llm" in config:
            llm_config = config["llm"]
            llm_errors = self._validate_llm_config(llm_config)
            errors.extend(llm_errors)

        # 感知配置验证
        if "perception" in config:
            perception_config = config["perception"]
            perception_errors = self._validate_perception_config(perception_config)
            errors.extend(perception_errors)

        # 安全配置验证
        if "safety" in config:
            safety_config = config["safety"]
            safety_errors = self._validate_safety_config(safety_config)
            errors.extend(safety_errors)

        return errors

    def _validate_llm_config(self, llm_config: Dict[str, Any]) -> List[ValidationError]:
        """验证LLM配置"""
        errors = []

        # 检查provider
        if "provider" in llm_config:
            valid_providers = ["openai", "anthropic", "ollama", "local", "azure", "custom"]
            if llm_config["provider"] not in valid_providers:
                errors.append(ValidationError(
                    path="llm.provider",
                    message=f"不支持的LLM provider: {llm_config['provider']}",
                    error_type="enum",
                    current_value=llm_config["provider"],
                    expected_value=valid_providers
                ))

        # 检查temperature范围
        if "temperature" in llm_config:
            temp = llm_config["temperature"]
            if not isinstance(temp, (int, float)) or not 0.0 <= temp <= 2.0:
                errors.append(ValidationError(
                    path="llm.temperature",
                    message="temperature必须在0.0到2.0之间",
                    error_type="range",
                    current_value=temp,
                    expected_value="[0.0, 2.0]"
                ))

        return errors

    def _validate_perception_config(self, perception_config: Dict[str, Any]) -> List[ValidationError]:
        """验证感知配置"""
        errors = []

        # 检查更新率
        if "update_rate" in perception_config:
            rate = perception_config["update_rate"]
            if not isinstance(rate, (int, float)) or rate <= 0:
                errors.append(ValidationError(
                    path="perception.update_rate",
                    message="update_rate必须大于0",
                    error_type="range",
                    current_value=rate,
                    expected_value="> 0"
                ))

        # 检查传感器配置
        if "sensors" in perception_config:
            sensors = perception_config["sensors"]
            valid_sensor_types = ["camera", "lidar", "gps", "imu"]

            for sensor_name, sensor_config in sensors.items():
                if "enabled" in sensor_config and sensor_config["enabled"]:
                    # 检查传感器类型
                    if "type" in sensor_config and sensor_config["type"] not in valid_sensor_types:
                        errors.append(ValidationError(
                            path=f"perception.sensors.{sensor_name}.type",
                            message=f"不支持的传感器类型: {sensor_config['type']}",
                            error_type="enum",
                            current_value=sensor_config["type"],
                            expected_value=valid_sensor_types
                        ))

        return errors

    def _validate_safety_config(self, safety_config: Dict[str, Any]) -> List[ValidationError]:
        """验证安全配置"""
        errors = []

        # 检查地理围栏
        if "geofence" in safety_config:
            geofence = safety_config["geofence"]
            if "default_radius" in geofence:
                radius = geofence["default_radius"]
                if not isinstance(radius, (int, float)) or radius <= 0:
                    errors.append(ValidationError(
                        path="safety.geofence.default_radius",
                        message="default_radius必须大于0",
                        error_type="range",
                        current_value=radius,
                        expected_value="> 0"
                    ))

        return errors

    def format_errors(self, errors: List[ValidationError]) -> str:
        """格式化错误信息"""
        if not errors:
            return "配置验证通过"

        formatted = ["配置验证失败:"]

        for error in errors:
            msg = f"  - {error.path}: {error.message}"
            if error.current_value is not None:
                msg += f" (当前值: {error.current_value})"
            if error.expected_value is not None:
                msg += f" (期望: {error.expected_value})"
            formatted.append(msg)

        return "\n".join(formatted)


def validate_config(config: Dict[str, Any], schema_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    便捷函数：验证配置
    返回 (是否有效, 错误信息)
    """
    validator = ConfigValidator()
    is_valid, errors = validator.validate(config, schema_name)

    if is_valid:
        return True, "配置验证通过"
    else:
        return False, validator.format_errors(errors)


if __name__ == "__main__":
    # 测试配置验证
    test_config = {
        "llm": {
            "provider": "invalid_provider",
            "temperature": 3.0,
            "model": "test-model"
        },
        "perception": {
            "update_rate": -1,
            "sensors": {
                "camera": {
                    "enabled": True,
                    "type": "camera",
                    "resolution": [1920, 1080]
                }
            }
        }
    }

    is_valid, errors = validate_config(test_config)
    print(f"验证结果: {'通过' if is_valid else '失败'}")
    print(errors)