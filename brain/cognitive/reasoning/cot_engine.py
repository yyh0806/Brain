"""
链式思维推理引擎 - Chain of Thought Engine (优化版)

负责：
- 执行链式思维推理，生成可追溯的决策链
- 自适应决定推理深度（简单任务快速执行，复杂任务深度推理）
- 结合感知上下文进行情境感知推理
- 支持多种推理模式：规划、重规划、异常处理
- 智能缓存推理结果，提升响应性能
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re
import hashlib
from loguru import logger

# 导入缓存管理器
try:
    from brain.utils.cache_manager import get_cache_manager, cached_async
    CACHE_AVAILABLE = True
except ImportError:
    logger.warning("缓存管理器不可用，推理引擎将运行在无缓存模式")
    CACHE_AVAILABLE = False


class ReasoningMode(Enum):
    """推理模式"""
    PLANNING = "planning"               # 任务规划
    REPLANNING = "replanning"           # 重新规划
    EXCEPTION_HANDLING = "exception"    # 异常处理
    CLARIFICATION = "clarification"     # 指令澄清
    DECISION = "decision"               # 决策判断


class ComplexityLevel(Enum):
    """复杂度等级"""
    SIMPLE = "simple"       # 简单，直接执行
    MODERATE = "moderate"   # 中等，简单推理
    COMPLEX = "complex"     # 复杂，完整CoT
    CRITICAL = "critical"   # 关键，深度推理+验证


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    question: str
    analysis: str
    conclusion: str
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "question": self.question,
            "analysis": self.analysis,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence
        }
    
    def __str__(self) -> str:
        return f"步骤{self.step_number}: {self.question}\n分析: {self.analysis}\n结论: {self.conclusion}"


@dataclass
class ReasoningResult:
    """推理结果"""
    mode: ReasoningMode
    query: str
    context_summary: str
    complexity: ComplexityLevel
    chain: List[ReasoningStep]
    decision: str
    suggestion: str
    confidence: float
    raw_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "query": self.query,
            "complexity": self.complexity.value,
            "chain": [step.to_dict() for step in self.chain],
            "decision": self.decision,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_chain_summary(self) -> str:
        """获取推理链摘要"""
        if not self.chain:
            return "无推理步骤"
        
        lines = ["推理过程:"]
        for step in self.chain:
            lines.append(f"  {step.step_number}. {step.question}")
            lines.append(f"     → {step.conclusion}")
        lines.append(f"\n最终决策: {self.decision}")
        
        return "\n".join(lines)


class CoTEngine:
    """
    链式思维推理引擎
    
    通过结构化的思维链提升决策质量，自适应调整推理深度
    """
    
    # 复杂度评估因子权重
    COMPLEXITY_WEIGHTS = {
        "obstacles_count": 0.15,
        "targets_count": 0.15,
        "constraints_count": 0.20,
        "recent_changes": 0.20,
        "command_length": 0.10,
        "ambiguity_score": 0.20
    }
    
    # 复杂度阈值
    COMPLEXITY_THRESHOLDS = {
        ComplexityLevel.SIMPLE: 0.3,
        ComplexityLevel.MODERATE: 0.5,
        ComplexityLevel.COMPLEX: 0.7,
        ComplexityLevel.CRITICAL: 1.0
    }
    
    def __init__(
        self,
        llm_interface: Optional[Any] = None,
        cot_prompts: Optional[Any] = None,
        default_complexity_threshold: float = 0.5,
        enable_caching: bool = True
    ):
        """
        Args:
            llm_interface: LLM接口
            cot_prompts: CoT提示词模板
            default_complexity_threshold: 默认复杂度阈值
            enable_caching: 是否启用缓存
        """
        self.llm = llm_interface
        self.cot_prompts = cot_prompts
        self.complexity_threshold = default_complexity_threshold
        self.enable_caching = enable_caching and CACHE_AVAILABLE

        # 推理历史
        self.reasoning_history: List[ReasoningResult] = []
        self.max_history = 50

        # 缓存设置
        if self.enable_caching:
            self.cache_manager = get_cache_manager()
            # 创建专用推理缓存
            self.reasoning_cache = self.cache_manager.create_cache(
                name="reasoning_results",
                cache_type="lru",
                max_size=1000,
                ttl_seconds=1800  # 30分钟
            )
            self.complexity_cache = self.cache_manager.create_cache(
                name="complexity_assessment",
                cache_type="lru",
                max_size=2000,
                ttl_seconds=300  # 5分钟
            )
            logger.info("CoTEngine 初始化完成 (缓存已启用)")
        else:
            logger.info("CoTEngine 初始化完成 (缓存已禁用)")
    
    async def reason(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode = ReasoningMode.PLANNING,
        force_cot: bool = False
    ) -> ReasoningResult:
        """
        执行推理 (优化版 - 支持缓存)

        Args:
            query: 推理问题/任务
            context: 上下文信息（包含感知数据）
            mode: 推理模式
            force_cot: 强制使用完整CoT

        Returns:
            ReasoningResult: 包含思维链和最终决策
        """
        # 生成缓存键
        cache_key = self._generate_reasoning_cache_key(query, context, mode, force_cot)

        # 尝试从缓存获取结果
        if self.enable_caching:
            cached_result = self.reasoning_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"CoT推理缓存命中: {cache_key[:20]}...")
                # 更新缓存结果的时间戳
                cached_result.timestamp = datetime.now()
                return cached_result

        # 评估复杂度 (支持缓存)
        complexity = self.assess_complexity(query, context)

        logger.info(f"CoT推理: mode={mode.value}, complexity={complexity.value}, query={query[:50]}...")

        # 根据复杂度决定推理策略
        if complexity == ComplexityLevel.SIMPLE and not force_cot:
            result = await self._quick_reasoning(query, context, mode)
        elif complexity in [ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX] or force_cot:
            result = await self._full_cot_reasoning(query, context, mode)
        else:  # CRITICAL
            result = await self._deep_cot_reasoning(query, context, mode)

        # 缓存结果 (仅缓存置信度较高的结果)
        if self.enable_caching and result.confidence > 0.7:
            # 克隆结果避免外部修改
            cached_result = ReasoningResult(
                mode=result.mode,
                query=result.query,
                context_summary=result.context_summary,
                complexity=result.complexity,
                chain=result.chain.copy(),
                decision=result.decision,
                suggestion=result.suggestion,
                confidence=result.confidence,
                raw_response=result.raw_response,
                timestamp=result.timestamp,
                metadata=result.metadata.copy()
            )
            self.reasoning_cache.set(cache_key, cached_result)
            logger.debug(f"CoT推理结果已缓存: {cache_key[:20]}...")

        # 记录到历史
        self.reasoning_history.append(result)
        if len(self.reasoning_history) > self.max_history:
            self.reasoning_history = self.reasoning_history[-self.max_history:]

        return result

    def _generate_reasoning_cache_key(self, query: str, context: Dict[str, Any], mode: ReasoningMode, force_cot: bool) -> str:
        """生成推理缓存键"""
        # 提取关键的上下文信息用于缓存键
        key_context = {
            "robot_position": context.get("current_position", {}),
            "obstacles_count": len(context.get("obstacles", [])) if isinstance(context.get("obstacles"), list) else 0,
            "targets_count": len(context.get("targets", [])) if isinstance(context.get("targets", []), list) else 0,
            "battery_level": context.get("battery_level", 100),
            "constraints": context.get("constraints", [])[:3],  # 只取前3个约束
        }

        # 生成哈希
        key_data = {
            "query": query[:100],  # 限制查询长度
            "mode": mode.value,
            "force_cot": force_cot,
            "context": key_context
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def assess_complexity(self, query: str, context: Dict[str, Any]) -> ComplexityLevel:
        """
        评估问题复杂度 (优化版 - 支持缓存)

        Args:
            query: 问题/任务
            context: 上下文

        Returns:
            ComplexityLevel: 复杂度等级
        """
        # 尝试从缓存获取复杂度评估结果
        if self.enable_caching:
            complexity_key = self._generate_complexity_cache_key(query, context)
            cached_complexity = self.complexity_cache.get(complexity_key)
            if cached_complexity is not None:
                logger.debug(f"复杂度评估缓存命中: {complexity_key[:20]}...")
                return cached_complexity

        # 计算复杂度分数
        score = self._calculate_complexity_score(query, context)

        # 确定复杂度等级
        for level, threshold in sorted(self.COMPLEXITY_THRESHOLDS.items(), key=lambda x: x[1]):
            if score <= threshold:
                complexity = level
                break
        else:
            complexity = ComplexityLevel.CRITICAL

        # 缓存复杂度评估结果
        if self.enable_caching:
            self.complexity_cache.set(complexity_key, complexity)

        return complexity

    def _calculate_complexity_score(self, query: str, context: Dict[str, Any]) -> float:
        """计算复杂度分数"""
        score = 0.0

        # 障碍物数量
        obstacles = context.get("obstacles", [])
        if isinstance(obstacles, int):
            obstacle_count = obstacles
        else:
            obstacle_count = len(obstacles) if obstacles else 0
        score += min(obstacle_count / 10, 1.0) * self.COMPLEXITY_WEIGHTS["obstacles_count"]

        # 目标数量
        targets = context.get("targets", [])
        if isinstance(targets, int):
            target_count = targets
        else:
            target_count = len(targets) if targets else 0
        score += min(target_count / 5, 1.0) * self.COMPLEXITY_WEIGHTS["targets_count"]

        # 约束数量
        constraints = context.get("constraints", [])
        constraint_count = len(constraints) if constraints else 0
        score += min(constraint_count / 5, 1.0) * self.COMPLEXITY_WEIGHTS["constraints_count"]

        # 最近变化
        changes = context.get("recent_changes", [])
        change_count = len(changes) if changes else 0
        score += min(change_count / 3, 1.0) * self.COMPLEXITY_WEIGHTS["recent_changes"]

        # 指令长度（越长可能越复杂）
        score += min(len(query) / 200, 1.0) * self.COMPLEXITY_WEIGHTS["command_length"]

        # 模糊度评估（简化：检查不确定词汇）
        ambiguous_words = ["那边", "附近", "差不多", "大概", "可能", "或者", "应该"]
        ambiguity = sum(1 for word in ambiguous_words if word in query) / len(ambiguous_words)
        score += ambiguity * self.COMPLEXITY_WEIGHTS["ambiguity_score"]

        return score

    def _generate_complexity_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """生成复杂度评估缓存键"""
        # 提取影响复杂度的关键因素
        key_data = {
            "query_length": len(query),
            "has_ambiguous_words": any(word in query for word in ["那边", "附近", "差不多", "大概", "可能", "或者", "应该"]),
            "obstacles_count": len(context.get("obstacles", [])) if isinstance(context.get("obstacles"), list) else 0,
            "targets_count": len(context.get("targets", [])) if isinstance(context.get("targets", []), list) else 0,
            "constraints_count": len(context.get("constraints", [])) if context.get("constraints") else 0,
            "recent_changes_count": len(context.get("recent_changes", [])) if context.get("recent_changes") else 0,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _quick_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode
    ) -> ReasoningResult:
        """快速推理（简单任务）"""
        # 构建简单提示
        prompt = self._build_quick_prompt(query, context, mode)
        
        # 调用LLM
        response = await self._call_llm(prompt, max_tokens=500)
        
        # 解析响应
        decision, suggestion = self._parse_quick_response(response)
        
        # 创建简单的推理步骤
        chain = [ReasoningStep(
            step_number=1,
            question="任务分析",
            analysis=f"简单任务，直接执行: {query}",
            conclusion=decision,
            confidence=0.9
        )]
        
        return ReasoningResult(
            mode=mode,
            query=query,
            context_summary=self._summarize_context(context),
            complexity=ComplexityLevel.SIMPLE,
            chain=chain,
            decision=decision,
            suggestion=suggestion,
            confidence=0.9,
            raw_response=response
        )
    
    async def _full_cot_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode
    ) -> ReasoningResult:
        """完整CoT推理"""
        # 构建CoT提示
        prompt = self._build_cot_prompt(query, context, mode)
        
        # 调用LLM
        response = await self._call_llm(prompt, max_tokens=2000)
        
        # 解析CoT响应
        chain, decision, suggestion, confidence = self._parse_cot_response(response)
        
        return ReasoningResult(
            mode=mode,
            query=query,
            context_summary=self._summarize_context(context),
            complexity=ComplexityLevel.COMPLEX,
            chain=chain,
            decision=decision,
            suggestion=suggestion,
            confidence=confidence,
            raw_response=response
        )
    
    async def _deep_cot_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode
    ) -> ReasoningResult:
        """深度CoT推理（关键任务，带验证）"""
        # 第一轮：完整推理
        first_result = await self._full_cot_reasoning(query, context, mode)
        
        # 第二轮：验证推理
        verification_prompt = self._build_verification_prompt(
            query, context, first_result
        )
        verification_response = await self._call_llm(verification_prompt, max_tokens=1000)
        
        # 合并验证结果
        verified = self._parse_verification(verification_response)
        
        if verified["is_valid"]:
            first_result.confidence = min(first_result.confidence + 0.1, 1.0)
            first_result.metadata["verification"] = "passed"
        else:
            # 如果验证失败，调整决策
            first_result.suggestion = verified.get("alternative", first_result.suggestion)
            first_result.confidence *= 0.8
            first_result.metadata["verification"] = "adjusted"
            first_result.metadata["verification_notes"] = verified.get("notes", "")
        
        first_result.complexity = ComplexityLevel.CRITICAL
        return first_result
    
    def _build_quick_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode
    ) -> str:
        """构建快速推理提示"""
        context_str = self._summarize_context(context)
        
        return f"""你是一个无人系统的决策引擎。请快速分析并给出决策。

## 当前环境
{context_str}

## 任务
{query}

## 模式
{mode.value}

请直接给出决策和建议，格式：
决策: [你的决策]
建议: [具体操作建议]"""
    
    def _build_cot_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        mode: ReasoningMode
    ) -> str:
        """构建完整CoT提示"""
        context_str = self._summarize_context(context)
        
        # 根据模式选择问题框架
        if mode == ReasoningMode.PLANNING:
            questions = [
                "1. 当前环境中有哪些关键因素？（障碍物、目标、约束）",
                "2. 这些因素如何影响任务执行？",
                "3. 可行的操作序列有哪些？",
                "4. 最优序列是什么？为什么？",
                "5. 有哪些潜在风险需要考虑？"
            ]
        elif mode == ReasoningMode.REPLANNING:
            questions = [
                "1. 环境发生了什么变化？",
                "2. 原计划的哪些部分受到影响？",
                "3. 是否需要重新规划？",
                "4. 新的计划应该如何调整？",
                "5. 调整后的风险评估是什么？"
            ]
        elif mode == ReasoningMode.EXCEPTION_HANDLING:
            questions = [
                "1. 发生了什么异常？",
                "2. 异常的原因是什么？",
                "3. 有哪些处理选项？",
                "4. 每个选项的利弊是什么？",
                "5. 推荐的处理方式是什么？"
            ]
        else:
            questions = [
                "1. 问题的核心是什么？",
                "2. 有哪些相关信息？",
                "3. 可能的答案有哪些？",
                "4. 哪个答案最合理？",
                "5. 还需要什么信息？"
            ]
        
        questions_str = "\n".join(questions)
        
        return f"""你是一个无人系统的智能决策引擎。请使用链式思维(Chain of Thought)进行分析。

## 当前感知状态
{context_str}

## 任务目标
{query}

## 推理模式
{mode.value}

## 推理过程
请按照以下步骤逐步分析：

{questions_str}

对于每个步骤，请提供：
- 分析过程
- 结论
- 置信度（0-1）

## 最终决策
在完成所有分析后，给出：
- 决策: [明确的决策]
- 建议: [具体的操作建议]
- 置信度: [0-1的数值]

请开始推理："""
    
    def _build_verification_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        result: ReasoningResult
    ) -> str:
        """构建验证提示"""
        return f"""请验证以下推理和决策是否合理。

## 原始任务
{query}

## 环境上下文
{self._summarize_context(context)}

## 推理过程
{result.get_chain_summary()}

## 验证要点
1. 推理逻辑是否有漏洞？
2. 是否遗漏了重要因素？
3. 决策是否安全可行？
4. 是否有更好的替代方案？

请给出验证结果：
有效性: [yes/no]
问题: [如果有问题，说明是什么]
替代方案: [如果有更好的方案]"""
    
    async def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """调用LLM"""
        if not self.llm:
            # 模拟响应
            return self._generate_mock_response(prompt)
        
        try:
            from brain.models.llm_interface import LLMMessage
            
            messages = [
                LLMMessage(
                    role="system",
                    content="你是一个专业的无人系统决策引擎，擅长进行结构化的链式思维推理。"
                ),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await self.llm.chat(messages, max_tokens=max_tokens)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """生成模拟响应（用于测试）"""
        return """## 推理过程

### 步骤1: 环境分析
分析: 当前环境中需要考虑障碍物位置、目标状态和路径可行性
结论: 环境相对简单，可以执行任务
置信度: 0.85

### 步骤2: 任务分解
分析: 将任务分解为导航和执行两个阶段
结论: 先移动到目标位置，再执行具体操作
置信度: 0.9

### 步骤3: 路径规划
分析: 考虑障碍物分布，选择最短安全路径
结论: 沿当前方向直行是最优选择
置信度: 0.8

### 步骤4: 风险评估
分析: 评估执行过程中可能遇到的问题
结论: 风险可控，继续执行
置信度: 0.85

## 最终决策
决策: 执行任务
建议: 按照规划的路径移动到目标位置，执行指定操作
置信度: 0.85"""
    
    def _parse_quick_response(self, response: str) -> Tuple[str, str]:
        """解析快速推理响应"""
        decision = "执行"
        suggestion = "继续当前任务"
        
        lines = response.split("\n")
        for line in lines:
            if line.startswith("决策:") or line.startswith("决策："):
                decision = line.split(":", 1)[-1].strip()
            elif line.startswith("建议:") or line.startswith("建议："):
                suggestion = line.split(":", 1)[-1].strip()
        
        return decision, suggestion
    
    def _parse_cot_response(
        self,
        response: str
    ) -> Tuple[List[ReasoningStep], str, str, float]:
        """解析CoT响应"""
        chain = []
        decision = "执行"
        suggestion = "继续当前任务"
        confidence = 0.8
        
        # 解析步骤
        step_pattern = r"###?\s*步骤(\d+)[：:]\s*(.+?)(?=###?\s*步骤|\#\#\s*最终|$)"
        step_matches = re.findall(step_pattern, response, re.DOTALL)
        
        for match in step_matches:
            step_num = int(match[0])
            content = match[1].strip()
            
            # 提取分析和结论
            analysis = ""
            conclusion = ""
            step_confidence = 0.8
            
            if "分析:" in content or "分析：" in content:
                analysis = re.search(r"分析[：:]\s*(.+?)(?=结论|置信度|$)", content, re.DOTALL)
                analysis = analysis.group(1).strip() if analysis else content[:100]
            
            if "结论:" in content or "结论：" in content:
                conclusion_match = re.search(r"结论[：:]\s*(.+?)(?=置信度|$)", content, re.DOTALL)
                conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
            
            if "置信度:" in content or "置信度：" in content:
                conf_match = re.search(r"置信度[：:]\s*([\d.]+)", content)
                if conf_match:
                    step_confidence = float(conf_match.group(1))
            
            chain.append(ReasoningStep(
                step_number=step_num,
                question=f"步骤{step_num}分析",
                analysis=analysis or content[:200],
                conclusion=conclusion or "分析完成",
                confidence=step_confidence
            ))
        
        # 解析最终决策
        decision_pattern = r"决策[：:]\s*(.+?)(?=建议|置信度|\n|$)"
        decision_match = re.search(decision_pattern, response)
        if decision_match:
            decision = decision_match.group(1).strip()
        
        suggestion_pattern = r"建议[：:]\s*(.+?)(?=置信度|\n\n|$)"
        suggestion_match = re.search(suggestion_pattern, response, re.DOTALL)
        if suggestion_match:
            suggestion = suggestion_match.group(1).strip()
        
        conf_pattern = r"最终.*?置信度[：:]\s*([\d.]+)"
        conf_match = re.search(conf_pattern, response, re.DOTALL)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        # 如果没有解析到步骤，创建默认步骤
        if not chain:
            chain.append(ReasoningStep(
                step_number=1,
                question="综合分析",
                analysis=response[:500],
                conclusion=decision,
                confidence=confidence
            ))
        
        return chain, decision, suggestion, confidence
    
    def _parse_verification(self, response: str) -> Dict[str, Any]:
        """解析验证响应"""
        result = {
            "is_valid": True,
            "issues": [],
            "alternative": None,
            "notes": ""
        }
        
        response_lower = response.lower()
        
        if "有效性: no" in response_lower or "有效性：no" in response_lower:
            result["is_valid"] = False
        elif "有效性: yes" in response_lower or "有效性：yes" in response_lower:
            result["is_valid"] = True
        
        # 提取问题
        issue_match = re.search(r"问题[：:]\s*(.+?)(?=替代方案|$)", response, re.DOTALL)
        if issue_match:
            result["notes"] = issue_match.group(1).strip()
        
        # 提取替代方案
        alt_match = re.search(r"替代方案[：:]\s*(.+)", response, re.DOTALL)
        if alt_match:
            result["alternative"] = alt_match.group(1).strip()
        
        return result
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """将上下文转换为文本摘要"""
        if not context:
            return "无上下文信息"
        
        # 如果上下文已经有to_prompt_context方法
        if hasattr(context, "to_prompt_context"):
            return context.to_prompt_context()
        
        lines = []
        
        if "current_position" in context:
            pos = context["current_position"]
            lines.append(f"当前位置: ({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}, {pos.get('z', 0):.1f})")
        
        if "obstacles" in context:
            obstacles = context["obstacles"]
            count = len(obstacles) if isinstance(obstacles, list) else obstacles
            lines.append(f"障碍物: {count}个")
        
        if "targets" in context:
            targets = context["targets"]
            count = len(targets) if isinstance(targets, list) else targets
            lines.append(f"目标: {count}个")
        
        if "battery_level" in context:
            lines.append(f"电池: {context['battery_level']:.0f}%")
        
        if "constraints" in context:
            for c in context["constraints"][:3]:
                lines.append(f"约束: {c}")
        
        if "recent_changes" in context:
            for change in context["recent_changes"][:3]:
                if isinstance(change, dict):
                    lines.append(f"变化: {change.get('description', '未知')}")
                else:
                    lines.append(f"变化: {change}")
        
        if "changes" in context:
            lines.append(f"环境变化: {context['changes']}")
        
        if "current_plan" in context:
            lines.append(f"当前计划: {context['current_plan'][:100] if isinstance(context['current_plan'], str) else '已有计划'}")
        
        return "\n".join(lines) if lines else "基本环境状态正常"
    
    def get_recent_reasoning(self, count: int = 5) -> List[ReasoningResult]:
        """获取最近的推理历史"""
        return self.reasoning_history[-count:]
    
    def clear_history(self):
        """清除推理历史"""
        self.reasoning_history.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.enable_caching:
            return {"caching_enabled": False}

        stats = {
            "caching_enabled": True,
            "reasoning_cache": self.reasoning_cache.stats(),
            "complexity_cache": self.complexity_cache.stats()
        }
        return stats

    def clear_cache(self, cache_type: str = "all"):
        """清空缓存"""
        if not self.enable_caching:
            logger.warning("缓存未启用，无法清空")
            return

        if cache_type == "all" or cache_type == "reasoning":
            self.reasoning_cache.clear()
            logger.info("推理结果缓存已清空")

        if cache_type == "all" or cache_type == "complexity":
            self.complexity_cache.clear()
            logger.info("复杂度评估缓存已清空")

    def invalidate_cache_by_pattern(self, pattern: str):
        """按模式失效缓存"""
        if not self.enable_caching:
            logger.warning("缓存未启用，无法失效")
            return

        # 失效推理缓存
        self.cache_manager.invalidate_pattern(pattern, "reasoning_results")
        # 失效复杂度缓存
        self.cache_manager.invalidate_pattern(pattern, "complexity_assessment")
        logger.info(f"缓存已按模式失效: {pattern}")

    def optimize_cache(self):
        """优化缓存性能"""
        if not self.enable_caching:
            logger.warning("缓存未启用，无法优化")
            return

        # 清理过期条目
        self.cache_manager.cleanup_expired()

        # 打印统计信息
        stats = self.get_cache_stats()
        logger.info(f"缓存优化完成: 推理缓存命中率 {stats['reasoning_cache'].get('hit_rate', 0)*100:.1f}%, "
                   f"复杂度缓存命中率 {stats['complexity_cache'].get('hit_rate', 0)*100:.1f}%")

