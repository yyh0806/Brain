"""
感知层异常类定义
"""


class PerceptionError(Exception):
    """感知层基础异常"""
    pass


class SensorError(PerceptionError):
    """传感器相关异常"""
    pass


class SensorNotFoundError(SensorError):
    """传感器未找到异常"""
    pass


class SensorDisconnectedError(SensorError):
    """传感器断开连接异常"""
    pass


class SensorDataError(SensorError):
    """传感器数据异常"""
    pass


class DetectionError(PerceptionError):
    """检测相关异常"""
    pass


class MapError(PerceptionError):
    """地图相关异常"""
    pass


class FusionError(PerceptionError):
    """融合相关异常"""
    pass


class VLMError(PerceptionError):
    """VLM相关异常"""
    pass
