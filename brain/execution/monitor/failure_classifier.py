"""
失败分类器 - Failure Classifier

识别失败类型，为不同失败提供不同处理策略
"""

from enum import Enum
from typing import Dict, Any, Optional
from loguru import logger

from brain.execution.operations.base import OperationResult, OperationStatus


class FailureType(Enum):
    """失败类型"""
    PRECONDITION_FAILED = "precondition_failed"  # 前置条件不满足
    EXECUTION_FAILED = "execution_failed"        # 执行失败（机械故障）
    WORLD_STATE_CHANGED = "world_state_changed"   # 世界状态改变
    PERCEPTION_FAILED = "perception_failed"       # 感知失败


class FailureClassifier:
    """
    失败分类器
    
    Phase 1: 只实现2-3类（PRECONDITION_FAILED, EXECUTION_FAILED）
    """
    
    def __init__(self):
        """初始化失败分类器"""
        logger.info("FailureClassifier 初始化完成")
    
    def classify(
        self,
        operation_name: str,
        result: OperationResult,
        error_message: Optional[str] = None
    ) -> FailureType:
        """
        分类失败类型
        
        Args:
            operation_name: 操作名称
            result: 操作结果
            error_message: 错误信息
            
        Returns:
            失败类型
        """
        error_msg = error_message or result.error_message or ""
        error_msg_lower = error_msg.lower()
        
        # 检查前置条件失败
        if any(keyword in error_msg_lower for keyword in [
            "precondition", "前置条件", "条件不满足",
            "not ready", "not available", "door closed",
            "门关闭", "不可达", "unreachable"
        ]):
            return FailureType.PRECONDITION_FAILED
        
        # 检查感知失败
        if any(keyword in error_msg_lower for keyword in [
            "perception", "感知", "not found", "未找到",
            "not visible", "不可见", "detection failed"
        ]) and operation_name in ["search_object", "detect_objects", "query_object_location"]:
            return FailureType.PERCEPTION_FAILED
        
        # 检查世界状态改变
        if any(keyword in error_msg_lower for keyword in [
            "moved", "changed", "已移动", "已改变",
            "no longer", "不再", "disappeared", "消失"
        ]):
            return FailureType.WORLD_STATE_CHANGED
        
        # 默认：执行失败
        return FailureType.EXECUTION_FAILED
    
    def get_recovery_strategy(self, failure_type: FailureType) -> str:
        """
        获取恢复策略
        
        Args:
            failure_type: 失败类型
            
        Returns:
            恢复策略描述
        """
        strategies = {
            FailureType.PRECONDITION_FAILED: "insert_precondition_action",
            FailureType.EXECUTION_FAILED: "retry_or_alternative",
            FailureType.WORLD_STATE_CHANGED: "replan",
            FailureType.PERCEPTION_FAILED: "search_strategy"
        }
        
        return strategies.get(failure_type, "retry")
