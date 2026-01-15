#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因果图数据结构 - Causal Graph

实现三模态融合中的因果地图模态，用于追踪状态演化和因果关系。
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math


class CausalRelationType(Enum):
    """因果关系类型"""
    SPATIAL = "spatial"       # 空间关系（A在B附近）
    TEMPORAL = "temporal"     # 时序关系（A发生在B之前）
    FUNCTIONAL = "functional" # 功能关系（A导致B）
    STATE_CHANGE = "state"    # 状态变化（A导致B状态改变）


@dataclass
class CausalEdge:
    """因果关系边

    追踪两个节点之间的因果关联，支持贝叶斯置信度更新。
    """
    cause_id: str
    effect_id: str
    relation_type: CausalRelationType
    confidence: float
    evidence_count: int = 0
    first_observed: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalNode:
    """因果图节点

    代表图中的实体（物体、事件、状态）。
    """
    id: str
    type: str  # "object", "event", "state"
    label: str
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    active: bool = True
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    def add_state(self, state: Dict[str, Any], timestamp: Optional[datetime] = None):
        """添加状态记录"""
        self.state_history.append({
            "timestamp": timestamp or datetime.now(),
            "state": state.copy()
        })
        self.last_seen = timestamp or datetime.now()

    def get_state_trajectory(self) -> List[Dict[str, Any]]:
        """获取状态演化轨迹"""
        return self.state_history


@dataclass
class CausalGraph:
    """因果图：追踪状态演化和因果关系

    实现三模态融合中的因果地图模态，支持：
    1. 追踪物体状态变化历史
    2. 检测和记录因果关系
    3. 预测可能的后果
    4. 解释状态变化的原因
    """
    nodes: Dict[str, CausalNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], CausalEdge] = field(default_factory=dict)

    def add_or_update_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        state: Optional[Dict[str, Any]] = None
    ) -> CausalNode:
        """添加或更新节点"""
        if node_id not in self.nodes:
            self.nodes[node_id] = CausalNode(
                id=node_id,
                type=node_type,
                label=label
            )

        if state:
            self.nodes[node_id].add_state(state)

        return self.nodes[node_id]

    def add_causal_relation(
        self,
        cause: str,
        effect: str,
        relation_type: CausalRelationType,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加或更新因果关系

        使用贝叶斯更新规则调整置信度。

        Args:
            cause: 原因节点ID
            effect: 效果节点ID
            relation_type: 关系类型
            confidence: 置信度 [0, 1]
            metadata: 额外元数据
        """
        edge_key = (cause, effect)

        if edge_key in self.edges:
            # 更新现有关系：贝叶斯更新
            edge = self.edges[edge_key]
            edge.evidence_count += 1
            # 增量更新置信度（加权平均）
            edge.confidence = 0.7 * edge.confidence + 0.3 * confidence
            edge.last_observed = datetime.now()
            if metadata:
                edge.metadata.update(metadata)
        else:
            # 创建新关系
            self.edges[edge_key] = CausalEdge(
                cause_id=cause,
                effect_id=effect,
                relation_type=relation_type,
                confidence=confidence,
                evidence_count=1,
                metadata=metadata or {}
            )

        # 确保节点存在
        if cause not in self.nodes:
            self.add_or_update_node(cause, "unknown", cause)
        if effect not in self.nodes:
            self.add_or_update_node(effect, "unknown", effect)

    def predict_effects(self, cause_id: str) -> List[Tuple[str, float]]:
        """预测给定原因可能导致的后果

        Args:
            cause_id: 原因节点ID

        Returns:
            (effect_id, confidence) 列表，按置信度降序排列
        """
        effects = []
        for (c, e), edge in self.edges.items():
            if c == cause_id and edge.confidence > 0.6:
                effects.append((e, edge.confidence))
        return sorted(effects, key=lambda x: x[1], reverse=True)

    def explain_state_change(self, obj_id: str) -> Optional[str]:
        """解释物体状态变化的原因

        Args:
            obj_id: 物体ID

        Returns:
            原因描述字符串，如果没有找到原因返回None
        """
        causes = []
        for (c, e), edge in self.edges.items():
            if e == obj_id and edge.confidence > 0.5:
                cause_node = self.nodes.get(c)
                if cause_node:
                    causes.append(
                        f"{cause_node.label} → {edge.relation_type.value} "
                        f"({edge.confidence:.0%})"
                    )
        return "; ".join(causes) if causes else None

    def get_state_trajectory(self, obj_id: str) -> List[Dict[str, Any]]:
        """获取物体状态演化轨迹

        Args:
            obj_id: 物体ID

        Returns:
            状态历史列表
        """
        node = self.nodes.get(obj_id)
        return node.state_history if node else []

    def get_high_confidence_relations(
        self,
        threshold: float = 0.6
    ) -> List[Tuple[str, str, float, CausalRelationType]]:
        """获取高置信度因果关系

        Args:
            threshold: 置信度阈值

        Returns:
            (cause, effect, confidence, relation_type) 列表
        """
        relations = []
        for (cause, effect), edge in self.edges.items():
            if edge.confidence >= threshold:
                relations.append((
                    cause,
                    effect,
                    edge.confidence,
                    edge.relation_type
                ))
        return relations

    def prune_old_edges(self, max_age_days: int = 7, min_evidence: int = 3):
        """修剪旧的或证据不足的边

        Args:
            max_age_days: 最大保留天数
            min_evidence: 最小证据数量
        """
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        edges_to_remove = []

        for key, edge in self.edges.items():
            if (edge.last_observed < cutoff_time and
                edge.evidence_count < min_evidence):
                edges_to_remove.append(key)

        for key in edges_to_remove:
            del self.edges[key]

    def get_statistics(self) -> Dict[str, Any]:
        """获取因果图统计信息"""
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "active_nodes": sum(1 for n in self.nodes.values() if n.active),
            "high_confidence_edges": sum(
                1 for e in self.edges.values() if e.confidence > 0.7
            ),
            "avg_confidence": (
                sum(e.confidence for e in self.edges.values()) / max(1, len(self.edges))
            )
        }
