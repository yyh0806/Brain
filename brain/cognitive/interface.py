# -*- coding: utf-8 -*-
"""
认知层统一接口 - Cognitive Layer Interface

核心架构警句（设计准则）：
感知层看到世界，认知层相信世界，规划层改变世界。

认知层只输出：状态、变化、推理、建议，绝不输出行动决策。
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# 类型检查时的导入
if TYPE_CHECKING:
    from brain.cognitive.world_model.world_model import WorldModel, PlanningContext, EnvironmentChange
    from brain.cognitive.reasoning.cot_engine import CoTEngine, ReasoningResult, ReasoningMode
    from brain.cognitive.dialogue.dialogue_manager import DialogueManager, DialogueType
    from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor, MonitorEvent
    from brain.perception.sensors.ros2_sensor_manager import PerceptionData


class ObservationStatus(Enum):
    """观测结果状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ObservationResult:
    """观测结果 - 用于信念修正
    
    这是认知层接收执行层反馈的关键接口。
    """
    operation_id: str
    operation_type: str
    status: ObservationStatus
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 位置信息（如果相关）
    location: Optional[Dict[str, float]] = None
    
    # 结果详情
    details: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息（如果失败）
    error_message: Optional[str] = None
    
    # 置信度（如果相关）
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "details": self.details,
            "error_message": self.error_message,
            "confidence": self.confidence
        }


@dataclass
class Belief:
    """信念定义"""
    id: str
    content: str  # 信念内容，如 "杯子在厨房"
    confidence: float  # 置信度 0-1
    evidence_count: int = 0  # 支持证据数量
    falsified: bool = False  # 是否已被证伪
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "falsified": self.falsified,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class BeliefUpdate:
    """信念修正结果
    
    这是认知层输出信念更新的核心数据结构。
    """
    updated_beliefs: List[Belief] = field(default_factory=list)
    falsified_beliefs: List[str] = field(default_factory=list)  # 已证伪的信念ID列表
    new_evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "updated_beliefs": [b.to_dict() for b in self.updated_beliefs],
            "falsified_beliefs": self.falsified_beliefs,
            "new_evidence": self.new_evidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CognitiveOutput:
    """认知层处理感知数据后的输出
    
    严格遵循输出边界：只包含状态、变化，不包含行动决策。
    """
    planning_context: 'PlanningContext'
    environment_changes: List['EnvironmentChange'] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "planning_context": self.planning_context.to_prompt_context() if hasattr(self.planning_context, 'to_prompt_context') else str(self.planning_context),
            "environment_changes": [c.to_dict() for c in self.environment_changes],
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DialogueResponse:
    """对话响应"""
    content: str
    dialogue_type: 'DialogueType'
    requires_user_input: bool = False
    options: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "dialogue_type": self.dialogue_type.value if hasattr(self.dialogue_type, 'value') else str(self.dialogue_type),
            "requires_user_input": self.requires_user_input,
            "options": self.options,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class CognitiveLayer:
    """认知层统一接口
    
    核心原则：感知层看到世界，认知层相信世界，规划层改变世界。
    认知层只输出：状态、变化、推理、建议，绝不输出行动决策。
    
    这个接口封装了认知层的所有功能，对外提供统一的访问方式。
    """
    
    def __init__(
        self,
        world_model: Optional['WorldModel'] = None,
        cot_engine: Optional['CoTEngine'] = None,
        dialogue_manager: Optional['DialogueManager'] = None,
        perception_monitor: Optional['PerceptionMonitor'] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            world_model: 世界模型实例
            cot_engine: CoT推理引擎实例
            dialogue_manager: 对话管理器实例
            perception_monitor: 感知监控器实例
            config: 配置选项
        """
        self.config = config or {}
        self.world_model = world_model
        self.cot_engine = cot_engine
        self.dialogue_manager = dialogue_manager
        self.perception_monitor = perception_monitor
        
        # 初始化VLM感知器
        self.vlm = None
        self._init_vlm()
        
        # 感知管理器（将由Brain传入）
        self._sensor_manager = None
        
        logger.info("CognitiveLayer 初始化完成")
    
    def _init_vlm(self):
        """初始化VLM感知器"""
        try:
            from brain.perception.vlm.vlm_perception import VLMPerception, OLLAMA_AVAILABLE
            
            if OLLAMA_AVAILABLE:
                self.vlm = VLMPerception(
                    model=self.config.get("vlm.model", "llava:7b"),
                    ollama_host=self.config.get("vlm.ollama_host", "http://localhost:11434")
                )
                logger.info("VLM感知器已初始化 (ollama:llava:7b)")
            else:
                logger.warning("Ollama不可用，VLM功能受限")
        except Exception as e:
            logger.warning(f"VLM初始化失败: {e}")
            self.vlm = None
    
    async def process_perception(
        self, 
        perception_data: 'PerceptionData'
    ) -> CognitiveOutput:
        """处理感知数据，返回认知结果（信念更新）
        
        这是认知层的核心功能：将感知数据转换为认知状态。
        
        Args:
            perception_data: 来自感知层的多传感器融合数据
            
        Returns:
            CognitiveOutput: 包含规划上下文和环境变化
            严格遵循输出边界：只包含状态、变化，不包含行动决策
            
        Note:
            - 不返回行动决策（"应该去厨房"）
            - 不返回操作序列（"先移动，再搜索"）
            - 只返回认知状态和环境变化事件
        """
        if not self.world_model:
            raise ValueError("WorldModel 未初始化")
        
        # 更新世界模型
        changes = self.world_model.update_from_perception(perception_data)
        
        # VLM场景理解（如果有RGB图像且VLM可用）
        if perception_data and perception_data.rgb_image is not None and self.vlm is not None:
            try:
                import numpy as np
                # 确保图像是numpy数组
                rgb_image = perception_data.rgb_image
                if isinstance(rgb_image, np.ndarray):
                    scene = await self.vlm.analyze_scene(rgb_image)
                    # 更新语义对象到WorldModel
                    if hasattr(scene, 'objects') and scene.objects:
                        for obj in scene.objects:
                            # 创建追踪对象ID
                            obj_id = f"vlm_{obj.label}_{len(self.world_model.tracked_objects)}"
                            
                            # 更新追踪对象
                            position = {}
                            if obj.bbox:
                                position = {
                                    "x": obj.bbox.x,
                                    "y": obj.bbox.y,
                                    "z": 0.0
                                }
                            
                            self.world_model.update_tracked_object(
                                obj_id,
                                label=obj.label,
                                position=position if position else {"x": 0.5, "y": 0.5, "z": 0.0},
                                attributes={
                                    "confidence": obj.confidence,
                                    "source": "vlm"
                                }
                            )
                        
                        logger.info(f"VLM分析完成: 检测到{len(scene.objects)}个对象")
                else:
                    logger.warning("RGB图像不是numpy数组，跳过VLM分析")
            except Exception as e:
                logger.warning(f"VLM分析失败: {e}")
        
        # 获取规划上下文
        planning_context = self.world_model.get_context_for_planning()
        
        return CognitiveOutput(
            planning_context=planning_context,
            environment_changes=changes
        )
    
    async def update_belief(
        self,
        observation_result: ObservationResult
    ) -> BeliefUpdate:
        """根据观测结果更新信念（核心功能）
        
        这是认知层最关键的职责：
        - 接收执行结果（成功/失败）
        - 更新世界模型中的置信度
        - 维护"哪些假设已经被证伪"
        
        Args:
            observation_result: 观测结果（执行成功/失败）
            
        Returns:
            BeliefUpdate: 信念修正结果，包含更新的信念和已证伪的假设
            
        Note:
            这是认知层实现"自我否定"能力的关键接口。
            所有观测结果必须通过此方法更新信念。
        """
        if not self.world_model:
            raise ValueError("WorldModel 未初始化")
        
        # 获取或创建 BeliefUpdatePolicy
        if not hasattr(self.world_model, 'belief_update_policy'):
            from brain.cognitive.world_model.belief.belief_update_policy import BeliefUpdatePolicy
            self.world_model.belief_update_policy = BeliefUpdatePolicy(
                config=self.config.get("belief_update", {})
            )
        
        # 调用信念修正策略
        return self.world_model.belief_update_policy.update_belief(observation_result)
    
    async def reason(
        self,
        query: str,
        context: Dict[str, Any],
        mode: 'ReasoningMode'
    ) -> 'ReasoningResult':
        """执行推理
        
        返回：思维链和判断（why/what changed）
        不返回：行动决策（应该怎么做）
        
        Args:
            query: 推理查询
            context: 上下文信息（包含感知数据）
            mode: 推理模式
            
        Returns:
            ReasoningResult: 包含思维链和判断
            - decision 字段：解释性判断（"为什么变化"），不是动作指令
            - suggestion 字段：建议性提示，规划层决定是否采纳
            
        Note:
            - decision 字段只包含解释性判断，不包含动作指令
            - suggestion 字段是建议，不是决策
            - 规划层根据这些信息自主决定行动
        """
        if not self.cot_engine:
            raise ValueError("CoTEngine 未初始化")
        
        return await self.cot_engine.reason(
            query=query,
            context=context,
            mode=mode
        )
    
    async def dialogue(
        self,
        message: str,
        dialogue_type: 'DialogueType',
        context: Optional[Dict[str, Any]] = None
    ) -> DialogueResponse:
        """处理对话
        
        Args:
            message: 用户消息或系统消息
            dialogue_type: 对话类型
            context: 对话上下文
            
        Returns:
            DialogueResponse: 对话响应
        """
        if not self.dialogue_manager:
            raise ValueError("DialogueManager 未初始化")
        
        # 根据对话类型调用相应方法
        if dialogue_type.value == "clarification":
            # 处理澄清请求
            result = await self.dialogue_manager.clarify_ambiguous_command(
                command=message,
                ambiguities=context.get("ambiguities", []) if context else [],
                world_context=context.get("world_context") if context else None
            )
            return DialogueResponse(
                content=result.get("question", ""),
                dialogue_type=dialogue_type,
                requires_user_input=True,
                options=context.get("options") if context else None
            )
        elif dialogue_type.value == "confirmation":
            # 处理确认请求
            confirmed = await self.dialogue_manager.request_confirmation(
                action=message,
                reason=context.get("reason", "") if context else "",
                details=context.get("details") if context else None,
                options=context.get("options") if context else None
            )
            return DialogueResponse(
                content=f"确认结果: {'已确认' if confirmed else '已取消'}",
                dialogue_type=dialogue_type,
                requires_user_input=False
            )
        else:
            # 其他对话类型
            return DialogueResponse(
                content=message,
                dialogue_type=dialogue_type,
                requires_user_input=False
            )
    
    def get_planning_context(self) -> 'PlanningContext':
        """获取规划上下文（信念状态）
        
        规划层消费此上下文，自主决定行动。
        
        Returns:
            PlanningContext: 包含完整环境信息的上下文
            
        Note:
            这个接口只提供认知状态，不提供行动建议。
            规划层根据此上下文自主生成行动计划。
        """
        if not self.world_model:
            raise ValueError("WorldModel 未初始化")
        
        return self.world_model.get_context_for_planning()
    
    def get_falsified_beliefs(self) -> List[str]:
        """获取已证伪的信念ID列表
        
        规划层可以查询此列表，避免重复失败。
        
        Returns:
            List[str]: 已证伪的信念ID列表
        """
        if not self.world_model:
            raise ValueError("WorldModel 未初始化")
        
        # 从 BeliefUpdatePolicy 获取已证伪的信念列表
        if hasattr(self.world_model, 'belief_update_policy'):
            return self.world_model.belief_update_policy.get_falsified_beliefs()
        
        return []
    
    def start_monitoring(self):
        """开始感知监控"""
        if not self.perception_monitor:
            raise ValueError("PerceptionMonitor 未初始化")
        
        # 注意：监控器只通知变化，不触发动作
        # 实际的动作触发由规划层决定
        logger.info("开始感知监控（仅通知模式）")
    
    def stop_monitoring(self):
        """停止感知监控"""
        if not self.perception_monitor:
            raise ValueError("PerceptionMonitor 未初始化")
        
        logger.info("停止感知监控")





