"""
信念修正策略 - Belief Revision Policy

负责：
- 处理规划失败、搜索失败时的信念更新
- 管理信念置信度的衰减和提升
- 维护信念的一致性
- 防止基于错误信念的死循环

核心原则：
- 搜索失败 → 降低概率
- 多次失败 → 移除假设
- 执行成功 → 提高置信度
- 长时间未观测 → 置信度衰减
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math
from loguru import logger


class BeliefType(Enum):
    """信念类型"""
    OBJECT_LOCATION = "object_location"      # 物体位置信念（如"cup在kitchen"）
    PATH_ACCESSIBLE = "path_accessible"      # 路径可达性信念（如"从A到B的路径畅通"）
    OBJECT_EXISTS = "object_exists"          # 物体存在性信念（如"目标物体存在"）
    ENVIRONMENT_STATE = "environment_state"  # 环境状态信念（如"天气晴朗"）


class OperationType(Enum):
    """操作类型"""
    SEARCH = "search"              # 搜索操作
    NAVIGATE = "navigate"          # 导航操作
    MANIPULATE = "manipulate"      # 操作物体
    OBSERVE = "observe"            # 观测操作
    OTHER = "other"                # 其他操作


@dataclass
class BeliefEntry:
    """信念条目"""
    belief_id: str
    belief_type: BeliefType
    description: str  # 信念描述，如"cup在kitchen"
    
    # 置信度信息
    confidence: float = 0.5  # 当前置信度 [0.0, 1.0]
    min_confidence: float = 0.05  # 最小置信度
    max_confidence: float = 0.95  # 最大置信度
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    last_observed: Optional[datetime] = None  # 最后观测时间
    last_updated: datetime = field(default_factory=datetime.now)
    
    # 失败/成功统计
    failure_count: int = 0  # 失败次数
    success_count: int = 0  # 成功次数
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    
    # 关联的操作类型
    related_operations: Set[OperationType] = field(default_factory=set)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """检查信念是否仍然有效"""
        return self.confidence > self.min_confidence
    
    def should_remove(self, removal_threshold: float) -> bool:
        """判断是否应该移除"""
        return self.confidence <= removal_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "belief_id": self.belief_id,
            "belief_type": self.belief_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_observed": self.last_observed.isoformat() if self.last_observed else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


class BeliefRevisionPolicy:
    """
    信念修正策略
    
    实现信念的置信度管理、失败处理和衰减机制
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 失败惩罚配置
        self.failure_penalty = self.config.get("failure_penalty", 0.2)
        self.max_failures_before_remove = self.config.get("max_failures_before_remove", 3)
        self.removal_threshold = self.config.get("removal_threshold", 0.1)
        
        # 成功奖励配置
        self.success_boost = self.config.get("success_boost", 0.1)
        self.max_confidence = self.config.get("max_confidence", 0.95)
        
        # 时间衰减配置
        self.decay_rate = self.config.get("decay_rate", 0.01)  # 每秒衰减
        self.decay_interval = self.config.get("decay_interval", 60.0)  # 衰减检查间隔（秒）
        self.min_confidence = self.config.get("min_confidence", 0.05)
        
        # 观测更新配置
        self.observation_boost = self.config.get("observation_boost", 0.15)
        self.observation_decay_time = self.config.get("observation_decay_time", 300.0)  # 观测失效时间（秒）
        
        # 信念存储
        self.beliefs: Dict[str, BeliefEntry] = {}
        
        # 上次衰减更新时间
        self.last_decay_update: datetime = datetime.now()
        
        logger.info("BeliefRevisionPolicy 初始化完成")
    
    def register_belief(
        self,
        belief_id: str,
        belief_type: BeliefType,
        description: str,
        initial_confidence: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BeliefEntry:
        """
        注册新信念
        
        Args:
            belief_id: 信念唯一标识
            belief_type: 信念类型
            description: 信念描述
            initial_confidence: 初始置信度
            metadata: 元数据
            
        Returns:
            BeliefEntry: 创建的信念条目
        """
        if belief_id in self.beliefs:
            logger.warning(f"信念 {belief_id} 已存在，更新描述和置信度")
            belief = self.beliefs[belief_id]
            belief.description = description
            belief.confidence = initial_confidence
            belief.last_updated = datetime.now()
            return belief
        
        belief = BeliefEntry(
            belief_id=belief_id,
            belief_type=belief_type,
            description=description,
            confidence=initial_confidence,
            min_confidence=self.min_confidence,
            max_confidence=self.max_confidence,
            metadata=metadata or {}
        )
        
        self.beliefs[belief_id] = belief
        logger.info(f"注册新信念: {belief_id} ({description}), 置信度={initial_confidence:.2f}")
        
        return belief
    
    def report_operation_failure(
        self,
        belief_id: str,
        operation_type: OperationType,
        error: Optional[str] = None
    ) -> Optional[BeliefEntry]:
        """
        报告操作失败
        
        当基于某个信念的操作失败时，降低该信念的置信度
        
        Args:
            belief_id: 相关的信念ID
            operation_type: 失败的操作类型
            error: 错误信息（可选）
            
        Returns:
            BeliefEntry: 更新后的信念条目，如果信念不存在则返回None
        """
        if belief_id not in self.beliefs:
            logger.warning(f"报告失败时未找到信念: {belief_id}")
            return None
        
        belief = self.beliefs[belief_id]
        
        # 更新失败统计
        belief.failure_count += 1
        belief.last_failure = datetime.now()
        belief.related_operations.add(operation_type)
        
        # 降低置信度
        new_confidence = max(
            self.min_confidence,
            belief.confidence - self.failure_penalty
        )
        belief.confidence = new_confidence
        belief.last_updated = datetime.now()
        
        logger.info(
            f"信念失败报告: {belief_id} ({belief.description}), "
            f"失败次数={belief.failure_count}, 新置信度={new_confidence:.2f}"
        )
        
        # 检查是否应该移除
        if belief.should_remove(self.removal_threshold):
            logger.warning(
                f"信念 {belief_id} 置信度过低 ({new_confidence:.2f}), "
                f"建议移除或标记为无效"
            )
        
        # 检查是否达到最大失败次数
        if belief.failure_count >= self.max_failures_before_remove:
            logger.warning(
                f"信念 {belief_id} 失败次数达到阈值 ({belief.failure_count}), "
                f"置信度={new_confidence:.2f}"
            )
        
        return belief
    
    def report_operation_success(
        self,
        belief_id: str,
        operation_type: OperationType
    ) -> Optional[BeliefEntry]:
        """
        报告操作成功
        
        当基于某个信念的操作成功时，提高该信念的置信度
        
        Args:
            belief_id: 相关的信念ID
            operation_type: 成功的操作类型
            
        Returns:
            BeliefEntry: 更新后的信念条目，如果信念不存在则返回None
        """
        if belief_id not in self.beliefs:
            logger.warning(f"报告成功时未找到信念: {belief_id}")
            return None
        
        belief = self.beliefs[belief_id]
        
        # 更新成功统计
        belief.success_count += 1
        belief.last_success = datetime.now()
        belief.related_operations.add(operation_type)
        
        # 提高置信度
        new_confidence = min(
            self.max_confidence,
            belief.confidence + self.success_boost
        )
        belief.confidence = new_confidence
        belief.last_updated = datetime.now()
        
        logger.info(
            f"信念成功报告: {belief_id} ({belief.description}), "
            f"成功次数={belief.success_count}, 新置信度={new_confidence:.2f}"
        )
        
        return belief
    
    def update_observation(
        self,
        belief_id: str,
        confidence: Optional[float] = None
    ) -> Optional[BeliefEntry]:
        """
        更新观测
        
        当有新的观测证据时，更新信念的置信度
        
        Args:
            belief_id: 信念ID
            confidence: 观测置信度（如果为None，则使用默认的observation_boost）
            
        Returns:
            BeliefEntry: 更新后的信念条目
        """
        if belief_id not in self.beliefs:
            logger.warning(f"更新观测时未找到信念: {belief_id}")
            return None
        
        belief = self.beliefs[belief_id]
        
        # 更新观测时间
        belief.last_observed = datetime.now()
        
        # 更新置信度
        if confidence is not None:
            # 使用观测置信度，但进行平滑更新
            belief.confidence = min(
                self.max_confidence,
                belief.confidence * 0.7 + confidence * 0.3
            )
        else:
            # 使用默认的观测提升
            belief.confidence = min(
                self.max_confidence,
                belief.confidence + self.observation_boost
            )
        
        belief.last_updated = datetime.now()
        
        logger.debug(
            f"更新观测: {belief_id}, 新置信度={belief.confidence:.2f}"
        )
        
        return belief
    
    def update_belief_decay(self) -> int:
        """
        更新信念衰减
        
        对长时间未观测的信念进行置信度衰减
        
        Returns:
            int: 更新的信念数量
        """
        now = datetime.now()
        elapsed = (now - self.last_decay_update).total_seconds()
        
        # 检查是否到了衰减时间
        if elapsed < self.decay_interval:
            return 0
        
        updated_count = 0
        
        for belief_id, belief in list(self.beliefs.items()):
            # 计算自上次观测以来的时间
            if belief.last_observed:
                time_since_observation = (now - belief.last_observed).total_seconds()
            else:
                time_since_observation = (now - belief.created_at).total_seconds()
            
            # 如果超过观测失效时间，开始衰减
            if time_since_observation > self.observation_decay_time:
                # 计算衰减量（基于时间）
                decay_amount = self.decay_rate * elapsed
                
                # 应用衰减
                new_confidence = max(
                    self.min_confidence,
                    belief.confidence - decay_amount
                )
                
                if new_confidence != belief.confidence:
                    belief.confidence = new_confidence
                    belief.last_updated = now
                    updated_count += 1
                    
                    logger.debug(
                        f"信念衰减: {belief_id}, "
                        f"时间={time_since_observation:.1f}s, "
                        f"新置信度={new_confidence:.2f}"
                    )
                    
                    # 如果置信度过低，标记为无效
                    if belief.should_remove(self.removal_threshold):
                        logger.warning(
                            f"信念 {belief_id} 因衰减置信度过低，建议移除"
                        )
        
        self.last_decay_update = now
        
        if updated_count > 0:
            logger.info(f"信念衰减更新完成: {updated_count} 个信念")
        
        return updated_count
    
    def get_belief_confidence(self, belief_id: str) -> Optional[float]:
        """
        获取信念的当前置信度
        
        Args:
            belief_id: 信念ID
            
        Returns:
            float: 置信度，如果信念不存在则返回None
        """
        if belief_id not in self.beliefs:
            return None
        
        return self.beliefs[belief_id].confidence
    
    def get_belief(self, belief_id: str) -> Optional[BeliefEntry]:
        """
        获取信念条目
        
        Args:
            belief_id: 信念ID
            
        Returns:
            BeliefEntry: 信念条目，如果不存在则返回None
        """
        return self.beliefs.get(belief_id)
    
    def get_all_beliefs(
        self,
        belief_type: Optional[BeliefType] = None,
        min_confidence: Optional[float] = None
    ) -> List[BeliefEntry]:
        """
        获取所有信念
        
        Args:
            belief_type: 过滤信念类型（可选）
            min_confidence: 最小置信度阈值（可选）
            
        Returns:
            List[BeliefEntry]: 信念列表
        """
        beliefs = list(self.beliefs.values())
        
        # 按类型过滤
        if belief_type:
            beliefs = [b for b in beliefs if b.belief_type == belief_type]
        
        # 按置信度过滤
        if min_confidence is not None:
            beliefs = [b for b in beliefs if b.confidence >= min_confidence]
        
        return beliefs
    
    def remove_belief(self, belief_id: str) -> bool:
        """
        移除信念
        
        Args:
            belief_id: 信念ID
            
        Returns:
            bool: 是否成功移除
        """
        if belief_id not in self.beliefs:
            return False
        
        del self.beliefs[belief_id]
        logger.info(f"移除信念: {belief_id}")
        return True
    
    def cleanup_invalid_beliefs(self) -> int:
        """
        清理无效信念（置信度过低的信念）
        
        Returns:
            int: 清理的信念数量
        """
        to_remove = [
            belief_id for belief_id, belief in self.beliefs.items()
            if belief.should_remove(self.removal_threshold)
        ]
        
        for belief_id in to_remove:
            self.remove_belief(belief_id)
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个无效信念")
        
        return len(to_remove)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total = len(self.beliefs)
        valid = sum(1 for b in self.beliefs.values() if b.is_valid())
        
        by_type = {}
        for belief_type in BeliefType:
            by_type[belief_type.value] = sum(
                1 for b in self.beliefs.values()
                if b.belief_type == belief_type
            )
        
        avg_confidence = (
            sum(b.confidence for b in self.beliefs.values()) / total
            if total > 0 else 0.0
        )
        
        total_failures = sum(b.failure_count for b in self.beliefs.values())
        total_successes = sum(b.success_count for b in self.beliefs.values())
        
        return {
            "total_beliefs": total,
            "valid_beliefs": valid,
            "invalid_beliefs": total - valid,
            "by_type": by_type,
            "average_confidence": avg_confidence,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "last_decay_update": self.last_decay_update.isoformat()
        }
