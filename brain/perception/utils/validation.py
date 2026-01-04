# -*- coding: utf-8 -*-
"""
数据验证工具 - Data Validation Utilities

提供数据验证功能，用于验证传感器数据、位姿数据等。

Author: Brain Development Team
Date: 2025-01-04
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationStatus(Enum):
    """验证状态"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """验证结果"""
    status: ValidationStatus
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.VALID


def validate_sensor_data(data: Dict[str, Any]) -> ValidationResult:
    """
    验证传感器数据
    
    Args:
        data: 传感器数据字典
        
    Returns:
        ValidationResult: 验证结果
    """
    # 检查必要字段
    required_fields = ["timestamp", "sensor_id", "sensor_type"]
    missing_fields = [f for f in required_fields if f not in data]
    
    if missing_fields:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message=f"缺少必要字段: {missing_fields}",
            field="required_fields"
        )
    
    # 验证时间戳
    timestamp = data.get("timestamp")
    if timestamp is None or timestamp <= 0:
        return ValidationResult(
            status=ValidationStatus.WARNING,
            message="时间戳无效或为0",
            field="timestamp"
        )
    
    # 验证传感器类型
    sensor_type = data.get("sensor_type")
    if sensor_type is None:
        return ValidationResult(
            status=ValidationStatus.WARNING,
            message="传感器类型未指定",
            field="sensor_type"
        )
    
    return ValidationResult(status=ValidationStatus.VALID, message="数据验证通过")


def validate_pose(pose: Dict[str, Any]) -> ValidationResult:
    """
    验证位姿数据
    
    Args:
        pose: 位姿数据字典
        
    Returns:
        ValidationResult: 验证结果
    """
    # 检查必要字段
    if "position" not in pose:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message="位姿缺少位置信息",
            field="position"
        )
    
    position = pose["position"]
    if not isinstance(position, dict):
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message="位置信息格式错误",
            field="position"
        )
    
    # 验证坐标范围
    for axis in ["x", "y", "z"]:
        if axis in position:
            value = position[axis]
            if not isinstance(value, (int, float)):
                return ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"位置坐标 {axis} 类型错误",
                    field=f"position.{axis}"
                )
            if abs(value) > 1000:  # 合理范围检查
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"位置坐标 {axis} 值过大: {value}",
                    field=f"position.{axis}"
                )
    
    return ValidationResult(status=ValidationStatus.VALID, message="位姿验证通过")


def validate_image(image: Dict[str, Any]) -> ValidationResult:
    """
    验证图像数据
    
    Args:
        image: 图像数据字典
        
    Returns:
        ValidationResult: 验证结果
    """
    # 检查图像数组
    image_data = image.get("data")
    if image_data is None:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message="图像数据为空",
            field="data"
        )
    
    # 检查图像形状
    import numpy as np
    if not isinstance(image_data, np.ndarray):
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message="图像数据不是numpy数组",
            field="data"
        )
    
    if len(image_data.shape) != 3:
        return ValidationResult(
            status=ValidationStatus.WARNING,
            message=f"图像维度应为3D，实际为{len(image_data.shape)}D",
            field="data"
        )
    
    return ValidationResult(status=ValidationStatus.VALID, message="图像验证通过")
