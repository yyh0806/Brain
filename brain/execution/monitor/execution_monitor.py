"""
执行监控器 - Execution Monitor

监控操作执行状态，检测失败并触发相应处理
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from brain.planning.plan_state import PlanNode, NodeStatus
from brain.execution.operations.base import OperationResult, OperationStatus
from .failure_classifier import FailureClassifier, FailureType


@dataclass
class ExecutionRecord:
    """执行记录"""
    node_id: str
    node_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[OperationResult] = None
    failure_type: Optional[FailureType] = None
    retry_count: int = 0


class ExecutionMonitor:
    """
    执行监控器
    
    监控PlanNode的执行状态，检测失败并分类
    """
    
    def __init__(self, failure_classifier: Optional[FailureClassifier] = None):
        """
        初始化执行监控器
        
        Args:
            failure_classifier: 失败分类器，如果为None则创建新的
        """
        self.failure_classifier = failure_classifier or FailureClassifier()
        self.records: Dict[str, ExecutionRecord] = {}
        self.failed_nodes: List[str] = []
        
        logger.info("ExecutionMonitor 初始化完成")
    
    def start_execution(self, node: PlanNode):
        """
        开始执行节点
        
        Args:
            node: 要执行的节点
        """
        record = ExecutionRecord(
            node_id=node.id,
            node_name=node.name,
            start_time=datetime.now()
        )
        self.records[node.id] = record
        node.mark_started()
        
        logger.debug(f"开始执行节点: {node.name} [{node.id}]")
    
    def record_success(self, node: PlanNode, result: OperationResult):
        """
        记录成功
        
        Args:
            node: 节点
            result: 操作结果
        """
        record = self.records.get(node.id)
        if record:
            record.end_time = datetime.now()
            record.result = result
        
        node.mark_success()
        logger.info(f"节点执行成功: {node.name} [{node.id}]")
    
    def record_failure(
        self,
        node: PlanNode,
        result: OperationResult,
        error_message: Optional[str] = None
    ) -> FailureType:
        """
        记录失败并分类
        
        Args:
            node: 节点
            result: 操作结果
            error_message: 错误信息
            
        Returns:
            失败类型
        """
        record = self.records.get(node.id)
        if record:
            record.end_time = datetime.now()
            record.result = result
            record.retry_count = node.retry_count
        
        node.mark_failed()
        
        # 分类失败类型
        failure_type = self.failure_classifier.classify(
            operation_name=node.action or node.name,
            result=result,
            error_message=error_message
        )
        
        if record:
            record.failure_type = failure_type
        
        if node.id not in self.failed_nodes:
            self.failed_nodes.append(node.id)
        
        logger.warning(
            f"节点执行失败: {node.name} [{node.id}], "
            f"类型: {failure_type.value}, "
            f"错误: {error_message or result.error_message}"
        )
        
        return failure_type
    
    def get_failed_nodes(self) -> List[str]:
        """获取失败的节点ID列表"""
        return self.failed_nodes.copy()
    
    def get_execution_record(self, node_id: str) -> Optional[ExecutionRecord]:
        """获取执行记录"""
        return self.records.get(node_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        total = len(self.records)
        successful = sum(1 for r in self.records.values() if r.result and r.result.status == OperationStatus.SUCCESS)
        failed = len(self.failed_nodes)
        
        failure_types = {}
        for record in self.records.values():
            if record.failure_type:
                ft = record.failure_type.value
                failure_types[ft] = failure_types.get(ft, 0) + 1
        
        return {
            "total_executed": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "failure_types": failure_types
        }
    
    def clear(self):
        """清空记录"""
        self.records.clear()
        self.failed_nodes.clear()
        logger.debug("执行记录已清空")
