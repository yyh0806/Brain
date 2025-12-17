"""
错误处理器 - Error Handler

负责:
- 错误分类与分析
- 恢复策略决策
- 错误上报
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

from brain.execution.operations.base import Operation, OperationType
from brain.state.world_state import WorldState


class ErrorType(Enum):
    """错误类型"""
    HARDWARE = "hardware"           # 硬件错误
    SOFTWARE = "software"           # 软件错误
    COMMUNICATION = "communication" # 通信错误
    NAVIGATION = "navigation"       # 导航错误
    PERCEPTION = "perception"       # 感知错误
    ENVIRONMENT = "environment"     # 环境错误
    TIMEOUT = "timeout"             # 超时错误
    RESOURCE = "resource"           # 资源错误
    SAFETY = "safety"               # 安全错误
    UNKNOWN = "unknown"             # 未知错误


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"                 # 重试
    SKIP = "skip"                   # 跳过
    ROLLBACK = "rollback"           # 回滚
    REPLAN = "replan"               # 重规划
    ABORT = "abort"                 # 中止
    MANUAL = "manual"               # 人工介入
    EMERGENCY = "emergency"         # 紧急处理


@dataclass
class ErrorAnalysis:
    """错误分析结果"""
    error_type: ErrorType
    severity: ErrorSeverity
    root_cause: str
    recoverable: bool
    needs_replan: bool
    can_rollback: bool
    retry_recommended: bool
    recommended_strategy: RecoveryStrategy
    alternative_strategies: List[RecoveryStrategy] = field(default_factory=list)
    safety_concerns: List[str] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "root_cause": self.root_cause,
            "recoverable": self.recoverable,
            "needs_replan": self.needs_replan,
            "can_rollback": self.can_rollback,
            "retry_recommended": self.retry_recommended,
            "recommended_strategy": self.recommended_strategy.value,
            "alternative_strategies": [s.value for s in self.alternative_strategies],
            "safety_concerns": self.safety_concerns,
            "additional_info": self.additional_info
        }


@dataclass
class ErrorRecord:
    """错误记录"""
    id: str
    operation_id: str
    operation_name: str
    error_type: ErrorType
    error_message: str
    analysis: Optional[ErrorAnalysis] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None


class ErrorHandler:
    """
    错误处理器
    
    分析错误原因并推荐恢复策略
    """
    
    # 错误关键词映射
    ERROR_KEYWORDS = {
        ErrorType.HARDWARE: [
            "motor", "电机", "sensor", "传感器", "battery", "电池",
            "gps", "imu", "camera", "相机"
        ],
        ErrorType.COMMUNICATION: [
            "timeout", "超时", "connection", "连接", "signal", "信号",
            "lost", "丢失", "disconnect", "断开"
        ],
        ErrorType.NAVIGATION: [
            "position", "位置", "path", "路径", "waypoint", "航点",
            "obstacle", "障碍", "collision", "碰撞"
        ],
        ErrorType.PERCEPTION: [
            "detection", "检测", "recognition", "识别", "tracking", "跟踪",
            "visibility", "能见度"
        ],
        ErrorType.ENVIRONMENT: [
            "weather", "天气", "wind", "风", "rain", "雨",
            "fog", "雾", "terrain", "地形"
        ],
        ErrorType.RESOURCE: [
            "memory", "内存", "storage", "存储", "fuel", "燃料",
            "power", "电量"
        ],
        ErrorType.SAFETY: [
            "geofence", "围栏", "altitude", "高度", "speed", "速度",
            "proximity", "接近", "emergency", "紧急"
        ]
    }
    
    # 操作类型到恢复策略的映射
    OPERATION_RECOVERY_MAP = {
        OperationType.MOVEMENT: {
            "default": RecoveryStrategy.RETRY,
            "high_severity": RecoveryStrategy.REPLAN,
            "critical": RecoveryStrategy.EMERGENCY
        },
        OperationType.PERCEPTION: {
            "default": RecoveryStrategy.RETRY,
            "high_severity": RecoveryStrategy.SKIP,
            "critical": RecoveryStrategy.REPLAN
        },
        OperationType.MANIPULATION: {
            "default": RecoveryStrategy.RETRY,
            "high_severity": RecoveryStrategy.ROLLBACK,
            "critical": RecoveryStrategy.ABORT
        },
        OperationType.COMMUNICATION: {
            "default": RecoveryStrategy.RETRY,
            "high_severity": RecoveryStrategy.RETRY,
            "critical": RecoveryStrategy.ABORT
        },
        OperationType.CONTROL: {
            "default": RecoveryStrategy.RETRY,
            "high_severity": RecoveryStrategy.ROLLBACK,
            "critical": RecoveryStrategy.EMERGENCY
        },
        OperationType.SAFETY: {
            "default": RecoveryStrategy.EMERGENCY,
            "high_severity": RecoveryStrategy.EMERGENCY,
            "critical": RecoveryStrategy.EMERGENCY
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 错误历史
        self.error_history: List[ErrorRecord] = []
        self.max_history = config.get("max_history", 100)
        
        # 错误计数
        self.error_counts: Dict[str, int] = {}
        
        # 阈值
        self.thresholds = {
            "warning": config.get("error_thresholds.warning", 3),
            "critical": config.get("error_thresholds.critical", 5)
        }
        
        logger.info("ErrorHandler 初始化完成")
    
    async def analyze(
        self,
        operation: Operation,
        error: str,
        world_state: WorldState
    ) -> ErrorAnalysis:
        """
        分析错误
        
        Args:
            operation: 失败的操作
            error: 错误信息
            world_state: 当前世界状态
            
        Returns:
            ErrorAnalysis: 错误分析结果
        """
        logger.info(f"分析错误: {error[:100]}...")
        
        # 1. 识别错误类型
        error_type = self._classify_error(error)
        
        # 2. 评估严重程度
        severity = self._assess_severity(error, operation, world_state)
        
        # 3. 分析根本原因
        root_cause = self._analyze_root_cause(error, operation, error_type)
        
        # 4. 判断是否可恢复
        recoverable = self._is_recoverable(error_type, severity, operation)
        
        # 5. 判断是否需要重规划
        needs_replan = self._needs_replan(error_type, operation, world_state)
        
        # 6. 判断是否可回滚
        can_rollback = self._can_rollback(operation, world_state)
        
        # 7. 判断是否推荐重试
        retry_recommended = self._should_retry(error, operation)
        
        # 8. 推荐恢复策略
        strategy = self._recommend_strategy(
            error_type, severity, operation,
            recoverable, needs_replan, can_rollback
        )
        
        # 9. 安全注意事项
        safety_concerns = self._identify_safety_concerns(
            error_type, operation, world_state
        )
        
        # 10. 替代策略
        alternatives = self._get_alternative_strategies(strategy, error_type)
        
        analysis = ErrorAnalysis(
            error_type=error_type,
            severity=severity,
            root_cause=root_cause,
            recoverable=recoverable,
            needs_replan=needs_replan,
            can_rollback=can_rollback,
            retry_recommended=retry_recommended,
            recommended_strategy=strategy,
            alternative_strategies=alternatives,
            safety_concerns=safety_concerns,
            additional_info={
                "operation_type": operation.type.value,
                "operation_name": operation.name,
                "error_message": error
            }
        )
        
        # 记录错误
        self._record_error(operation, error, error_type, analysis)
        
        logger.info(f"错误分析完成: 类型={error_type.value}, 严重度={severity.value}, 策略={strategy.value}")
        
        return analysis
    
    def _classify_error(self, error: str) -> ErrorType:
        """分类错误类型"""
        error_lower = error.lower()
        
        for error_type, keywords in self.ERROR_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in error_lower:
                    return error_type
        
        # 特殊检查
        if "timeout" in error_lower or "超时" in error:
            return ErrorType.TIMEOUT
        
        return ErrorType.UNKNOWN
    
    def _assess_severity(
        self,
        error: str,
        operation: Operation,
        world_state: WorldState
    ) -> ErrorSeverity:
        """评估错误严重程度"""
        # 关键词检查
        critical_keywords = ["critical", "emergency", "紧急", "严重", "失败"]
        high_keywords = ["error", "failed", "错误", "失败", "无法"]
        
        error_lower = error.lower()
        
        for kw in critical_keywords:
            if kw in error_lower:
                return ErrorSeverity.CRITICAL
        
        # 操作类型检查
        if operation.type == OperationType.SAFETY:
            return ErrorSeverity.CRITICAL
        
        if operation.type == OperationType.MOVEMENT:
            # 移动操作失败可能有安全影响
            return ErrorSeverity.HIGH
        
        for kw in high_keywords:
            if kw in error_lower:
                return ErrorSeverity.HIGH
        
        # 默认中等
        return ErrorSeverity.MEDIUM
    
    def _analyze_root_cause(
        self,
        error: str,
        operation: Operation,
        error_type: ErrorType
    ) -> str:
        """分析根本原因"""
        causes = {
            ErrorType.HARDWARE: f"硬件组件故障导致{operation.name}操作无法完成",
            ErrorType.SOFTWARE: f"软件错误导致{operation.name}操作异常",
            ErrorType.COMMUNICATION: "通信链路问题导致命令无法正确传输或反馈丢失",
            ErrorType.NAVIGATION: "导航系统异常，无法正确计算或跟踪目标位置",
            ErrorType.PERCEPTION: "感知系统无法正确获取或处理环境信息",
            ErrorType.ENVIRONMENT: "环境条件不适合当前操作执行",
            ErrorType.TIMEOUT: "操作执行时间超过预设阈值",
            ErrorType.RESOURCE: "系统资源不足以完成操作",
            ErrorType.SAFETY: "操作违反安全约束或检测到安全风险",
            ErrorType.UNKNOWN: f"未知原因导致{operation.name}操作失败"
        }
        
        base_cause = causes.get(error_type, causes[ErrorType.UNKNOWN])
        
        # 添加具体错误信息
        return f"{base_cause}。详细信息: {error[:100]}"
    
    def _is_recoverable(
        self,
        error_type: ErrorType,
        severity: ErrorSeverity,
        operation: Operation
    ) -> bool:
        """判断是否可恢复"""
        # 严重级别的错误通常不可直接恢复
        if severity == ErrorSeverity.CRITICAL:
            return error_type not in [ErrorType.HARDWARE, ErrorType.SAFETY]
        
        # 硬件错误通常不可软件恢复
        if error_type == ErrorType.HARDWARE:
            return False
        
        # 大多数其他错误可以尝试恢复
        return True
    
    def _needs_replan(
        self,
        error_type: ErrorType,
        operation: Operation,
        world_state: WorldState
    ) -> bool:
        """判断是否需要重规划"""
        # 环境变化需要重规划
        if error_type == ErrorType.ENVIRONMENT:
            return True
        
        # 导航错误可能需要重规划
        if error_type == ErrorType.NAVIGATION:
            return True
        
        # 移动操作失败后可能需要重规划
        if operation.type == OperationType.MOVEMENT:
            return True
        
        return False
    
    def _can_rollback(
        self,
        operation: Operation,
        world_state: WorldState
    ) -> bool:
        """判断是否可以回滚"""
        # 有回滚操作定义
        if operation.rollback_action:
            return True
        
        # 某些操作类型可以安全回滚
        rollbackable_types = [
            OperationType.MOVEMENT,
            OperationType.CONTROL
        ]
        
        return operation.type in rollbackable_types
    
    def _should_retry(
        self,
        error: str,
        operation: Operation
    ) -> bool:
        """判断是否应该重试"""
        # 检查错误计数
        error_key = f"{operation.name}_{operation.id}"
        count = self.error_counts.get(error_key, 0)
        
        if count >= self.thresholds["warning"]:
            return False
        
        # 临时性错误适合重试
        temporary_keywords = ["timeout", "超时", "temporary", "暂时", "retry"]
        error_lower = error.lower()
        
        for kw in temporary_keywords:
            if kw in error_lower:
                return True
        
        return True  # 默认允许重试
    
    def _recommend_strategy(
        self,
        error_type: ErrorType,
        severity: ErrorSeverity,
        operation: Operation,
        recoverable: bool,
        needs_replan: bool,
        can_rollback: bool
    ) -> RecoveryStrategy:
        """推荐恢复策略"""
        # 不可恢复
        if not recoverable:
            if severity == ErrorSeverity.CRITICAL:
                return RecoveryStrategy.EMERGENCY
            return RecoveryStrategy.ABORT
        
        # 安全相关
        if error_type == ErrorType.SAFETY:
            return RecoveryStrategy.EMERGENCY
        
        # 获取操作类型的默认策略
        op_strategies = self.OPERATION_RECOVERY_MAP.get(
            operation.type,
            {"default": RecoveryStrategy.RETRY}
        )
        
        if severity == ErrorSeverity.CRITICAL:
            return op_strategies.get("critical", RecoveryStrategy.ABORT)
        elif severity == ErrorSeverity.HIGH:
            return op_strategies.get("high_severity", RecoveryStrategy.REPLAN)
        
        # 需要重规划
        if needs_replan:
            return RecoveryStrategy.REPLAN
        
        # 可以回滚
        if can_rollback and severity >= ErrorSeverity.MEDIUM:
            return RecoveryStrategy.ROLLBACK
        
        # 默认重试
        return op_strategies.get("default", RecoveryStrategy.RETRY)
    
    def _identify_safety_concerns(
        self,
        error_type: ErrorType,
        operation: Operation,
        world_state: WorldState
    ) -> List[str]:
        """识别安全注意事项"""
        concerns = []
        
        if error_type == ErrorType.NAVIGATION:
            concerns.append("注意检查当前位置和障碍物")
        
        if error_type == ErrorType.COMMUNICATION:
            concerns.append("通信异常可能导致失联，考虑自主返航")
        
        if operation.type == OperationType.MOVEMENT:
            concerns.append("移动操作失败后确认当前位置安全")
        
        # 检查电池
        battery = world_state.get("robot.battery", 100)
        if battery < 30:
            concerns.append(f"电池电量偏低({battery}%)，考虑尽快返航")
        
        return concerns
    
    def _get_alternative_strategies(
        self,
        primary: RecoveryStrategy,
        error_type: ErrorType
    ) -> List[RecoveryStrategy]:
        """获取替代策略"""
        alternatives = []
        
        if primary != RecoveryStrategy.RETRY:
            alternatives.append(RecoveryStrategy.RETRY)
        
        if primary != RecoveryStrategy.REPLAN:
            alternatives.append(RecoveryStrategy.REPLAN)
        
        if primary not in [RecoveryStrategy.ABORT, RecoveryStrategy.EMERGENCY]:
            alternatives.append(RecoveryStrategy.ABORT)
        
        return alternatives[:3]  # 最多3个替代方案
    
    def _record_error(
        self,
        operation: Operation,
        error: str,
        error_type: ErrorType,
        analysis: ErrorAnalysis
    ):
        """记录错误"""
        record = ErrorRecord(
            id=f"err_{len(self.error_history)}_{datetime.now().timestamp()}",
            operation_id=operation.id,
            operation_name=operation.name,
            error_type=error_type,
            error_message=error,
            analysis=analysis
        )
        
        self.error_history.append(record)
        
        # 更新计数
        error_key = f"{operation.name}_{operation.id}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # 限制历史长度
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        type_counts = {}
        severity_counts = {}
        
        for record in self.error_history:
            type_counts[record.error_type.value] = (
                type_counts.get(record.error_type.value, 0) + 1
            )
            if record.analysis:
                severity_counts[record.analysis.severity.value] = (
                    severity_counts.get(record.analysis.severity.value, 0) + 1
                )
        
        return {
            "total_errors": len(self.error_history),
            "by_type": type_counts,
            "by_severity": severity_counts,
            "recent_errors": [
                {
                    "operation": r.operation_name,
                    "type": r.error_type.value,
                    "time": r.timestamp.isoformat()
                }
                for r in self.error_history[-5:]
            ]
        }
    
    def clear_error_counts(self):
        """清除错误计数"""
        self.error_counts.clear()
        logger.info("错误计数已清除")

