"""
感知层异常定义

提供标准化的异常类型，用于感知层的错误处理
"""

from typing import Dict, Any, Optional
from datetime import datetime


class PerceptionError(Exception):
    """感知层基础异常"""
    
    def __init__(
        self,
        message: str,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.component = component
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "component": self.component,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class SensorError(PerceptionError):
    """传感器相关错误"""
    
    def __init__(
        self,
        message: str,
        sensor_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, component="sensor", context=context)
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type


class FusionError(PerceptionError):
    """传感器融合错误"""
    
    def __init__(
        self,
        message: str,
        fusion_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, component="fusion", context=context)
        self.fusion_type = fusion_type


class DetectionError(PerceptionError):
    """目标检测错误"""
    
    def __init__(
        self,
        message: str,
        detector_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, component="detection", context=context)
        self.detector_type = detector_type


class MappingError(PerceptionError):
    """地图构建错误"""
    
    def __init__(
        self,
        message: str,
        mapper_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, component="mapping", context=context)
        self.mapper_type = mapper_type


class VLMPerceptionError(PerceptionError):
    """VLM感知错误"""
    
    def __init__(
        self,
        message: str,
        vlm_model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, component="vlm", context=context)
        self.vlm_model = vlm_model









