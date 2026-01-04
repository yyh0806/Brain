# -*- coding: utf-8 -*-
"""
信念修正策略 - Belief Update Policy

这是认知层最关键的职责：
- 接收观测结果（成功/失败）
- 更新世界模型中的置信度
- 维护"哪些假设已经被证伪"

这是认知层实现"自我否定"能力的核心。
"""

from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from brain.cognitive.interface import ObservationResult, BeliefUpdate, Belief


class BeliefUpdatePolicy:
    """信念修正策略
    
    这是认知层的"一等公民"，不是辅助功能。
    所有观测结果必须通过此策略更新信念。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 配置选项
        """
        self.config = config or {}
        
        # 信念存储
        self.beliefs: Dict[str, Belief] = {}
        
        # 已证伪的信念ID列表
        self.falsified_belief_ids: List[str] = []
        
        # 置信度更新参数
        self.success_confidence_increase = self.config.get("success_confidence_increase", 0.1)
        self.failure_confidence_decrease = self.config.get("failure_confidence_decrease", 0.2)
        self.falsification_threshold = self.config.get("falsification_threshold", 0.1)
        
        logger.info("BeliefUpdatePolicy 初始化完成")
    
    def update_belief(
        self,
        observation_result: ObservationResult
    ) -> BeliefUpdate:
        """根据观测结果更新信念
        
        Args:
            observation_result: 观测结果（执行成功/失败）
            
        Returns:
            BeliefUpdate: 信念修正结果
        """
        updated_beliefs: List[Belief] = []
        newly_falsified: List[str] = []
        
        # 根据观测结果更新相关信念
        if observation_result.status.value == "success":
            # 成功：增加相关信念的置信度
            related_beliefs = self._find_related_beliefs(observation_result)
            for belief_id in related_beliefs:
                if belief_id in self.beliefs:
                    belief = self.beliefs[belief_id]
                    if not belief.falsified:
                        old_confidence = belief.confidence
                        new_confidence = min(1.0, belief.confidence + self.success_confidence_increase)
                        belief.update_confidence(new_confidence)
                        belief.add_evidence()
                        updated_beliefs.append(belief)
                        logger.debug(f"信念 {belief_id} 置信度更新: {old_confidence:.2f} -> {new_confidence:.2f}")
        
        elif observation_result.status.value == "failure":
            # 失败：降低相关信念的置信度，可能证伪
            related_beliefs = self._find_related_beliefs(observation_result)
            for belief_id in related_beliefs:
                if belief_id in self.beliefs:
                    belief = self.beliefs[belief_id]
                    if not belief.falsified:
                        old_confidence = belief.confidence
                        new_confidence = max(0.0, belief.confidence - self.failure_confidence_decrease)
                        belief.update_confidence(new_confidence)
                        
                        # 如果置信度低于阈值，标记为证伪
                        if new_confidence < self.falsification_threshold:
                            belief.mark_falsified()
                            newly_falsified.append(belief_id)
                            if belief_id not in self.falsified_belief_ids:
                                self.falsified_belief_ids.append(belief_id)
                            logger.info(f"信念 {belief_id} 已被证伪: {belief.content}")
                        
                        updated_beliefs.append(belief)
                        logger.debug(f"信念 {belief_id} 置信度更新: {old_confidence:.2f} -> {new_confidence:.2f}")
        
        return BeliefUpdate(
            updated_beliefs=updated_beliefs,
            falsified_beliefs=newly_falsified,
            new_evidence={
                "operation_id": observation_result.operation_id,
                "status": observation_result.status.value,
                "timestamp": observation_result.timestamp.isoformat()
            }
        )
    
    def _find_related_beliefs(self, observation_result: ObservationResult) -> List[str]:
        """查找与观测结果相关的信念
        
        Args:
            observation_result: 观测结果
            
        Returns:
            List[str]: 相关信念ID列表
        """
        related = []
        
        # 根据操作类型和位置查找相关信念
        operation_type = observation_result.operation_type.lower()
        location = observation_result.location
        
        # 简单的匹配逻辑：根据操作类型和位置匹配信念
        for belief_id, belief in self.beliefs.items():
            # 如果信念内容包含操作类型或位置信息，认为是相关的
            content_lower = belief.content.lower()
            if operation_type in content_lower:
                related.append(belief_id)
            elif location and any(str(coord) in content_lower for coord in location.values()):
                related.append(belief_id)
        
        return related
    
    def add_belief(self, belief: Belief):
        """添加新信念"""
        self.beliefs[belief.id] = belief
        logger.debug(f"添加信念: {belief.id} - {belief.content}")
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """获取信念"""
        return self.beliefs.get(belief_id)
    
    def get_falsified_beliefs(self) -> List[str]:
        """获取已证伪的信念ID列表"""
        return self.falsified_belief_ids.copy()
    
    def is_falsified(self, belief_id: str) -> bool:
        """检查信念是否已被证伪"""
        return belief_id in self.falsified_belief_ids










