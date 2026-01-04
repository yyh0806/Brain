"""
断路器模式实现

用于保护外部服务调用（如VLM API），防止级联失败
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Any, Optional
from loguru import logger


class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"      # 正常状态，允许请求
    OPEN = "open"          # 打开状态，拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，允许少量请求测试


@dataclass
class CircuitBreaker:
    """
    断路器
    
    用于保护外部服务调用，当失败率达到阈值时打开断路器
    """
    
    failure_threshold: int = 5  # 失败次数阈值
    success_threshold: int = 2  # 半开状态下成功次数阈值
    timeout: float = 60.0  # 打开状态持续时间（秒）
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过断路器调用函数
        
        Args:
            func: 要调用的函数
            *args, **kwargs: 函数参数
            
        Returns:
            函数返回值
            
        Raises:
            Exception: 如果断路器打开，抛出异常
        """
        # 检查状态
        if self.state == CircuitState.OPEN:
            # 检查是否应该尝试半开
            if self.opened_at and (datetime.now() - self.opened_at).total_seconds() >= self.timeout:
                logger.info("断路器从OPEN转为HALF_OPEN，允许测试请求")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception(f"断路器打开，拒绝请求。将在 {self.timeout} 秒后重试")
        
        # 尝试调用
        try:
            result = func(*args, **kwargs)
            
            # 调用成功
            self._on_success()
            return result
            
        except Exception as e:
            # 调用失败
            self._on_failure()
            raise
    
    def _on_success(self):
        """处理成功调用"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info("断路器从HALF_OPEN转为CLOSED，服务恢复正常")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        elif self.state == CircuitState.CLOSED:
            # 重置失败计数
            self.failure_count = 0
    
    def _on_failure(self):
        """处理失败调用"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # 半开状态下失败，立即打开
            logger.warning("断路器从HALF_OPEN转为OPEN，服务仍然异常")
            self.state = CircuitState.OPEN
            self.opened_at = datetime.now()
        elif self.state == CircuitState.CLOSED:
            # 检查是否达到阈值
            if self.failure_count >= self.failure_threshold:
                logger.error(f"断路器从CLOSED转为OPEN，失败次数={self.failure_count}")
                self.state = CircuitState.OPEN
                self.opened_at = datetime.now()
    
    def reset(self):
        """重置断路器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.opened_at = None
        logger.info("断路器已重置")
    
    def get_status(self) -> dict:
        """获取断路器状态"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None
        }









