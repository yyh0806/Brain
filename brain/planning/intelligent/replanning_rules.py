"""
重规划规则 - Replanning Rules

明确动态插入vs重规划的边界和规则
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from brain.planning.plan_state import PlanNode
from brain.execution.monitor import FailureType


class ReplanningRules:
    """
    重规划规则
    
    决定何时使用动态插入，何时需要重规划
    """
    
    def __init__(
        self,
        max_insertions: int = 3,
        max_retries: int = 3
    ):
        """
        初始化重规划规则
        
        Args:
            max_insertions: 最大插入次数
            max_retries: 最大重试次数
        """
        self.max_insertions = max_insertions
        self.max_retries = max_retries
        
        logger.info("ReplanningRules 初始化完成")
    
    def should_replan(
        self,
        node: PlanNode,
        failure_type: FailureType,
        insertion_count: int,
        retry_count: int
    ) -> bool:
        """
        判断是否应该重规划
        
        Args:
            node: 失败的节点
            failure_type: 失败类型
            insertion_count: 当前插入次数
            retry_count: 当前重试次数
            
        Returns:
            是否应该重规划
        """
        # 规则1: 超过最大插入次数
        if insertion_count >= self.max_insertions:
            logger.info("超过最大插入次数，触发重规划")
            return True
        
        # 规则2: 超过最大重试次数
        if retry_count >= self.max_retries:
            logger.info("超过最大重试次数，触发重规划")
            return True
        
        # 规则3: 世界状态改变 - 必须重规划
        if failure_type == FailureType.WORLD_STATE_CHANGED:
            logger.info("世界状态改变，触发重规划")
            return True
        
        # 规则4: 目标不可达
        if "unreachable" in str(node.expected_effects).lower() or "不可达" in str(node.expected_effects):
            logger.info("目标不可达，触发重规划")
            return True
        
        # 规则5: HTN假设错误（Phase 2实现）
        # 例如：多次搜索失败，说明假设"物体在某个位置"错误
        
        # 默认：不重规划，使用动态插入
        return False
    
    def should_insert_precondition(
        self,
        node: PlanNode,
        failure_type: FailureType
    ) -> bool:
        """
        判断是否应该插入前置条件操作
        
        Args:
            node: 失败的节点
            failure_type: 失败类型
            
        Returns:
            是否应该插入
        """
        # 前置条件失败时，插入前置操作
        if failure_type == FailureType.PRECONDITION_FAILED:
            return True
        
        return False
    
    def get_recovery_action(
        self,
        node: PlanNode,
        failure_type: FailureType,
        insertion_count: int,
        retry_count: int
    ) -> str:
        """
        获取恢复动作
        
        Args:
            node: 失败的节点
            failure_type: 失败类型
            insertion_count: 插入次数
            retry_count: 重试次数
            
        Returns:
            恢复动作：'insert', 'retry', 'replan'
        """
        if self.should_replan(node, failure_type, insertion_count, retry_count):
            return "replan"
        
        if self.should_insert_precondition(node, failure_type):
            return "insert"
        
        if failure_type == FailureType.EXECUTION_FAILED and retry_count < self.max_retries:
            return "retry"
        
        # 默认重规划
        return "replan"
