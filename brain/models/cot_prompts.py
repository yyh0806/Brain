"""
CoT专用提示词模板 - Chain of Thought Prompts

提供结构化的思维链提示词，支持：
- 任务规划
- 重新规划（感知驱动）
- 异常处理
- 指令澄清
- 决策判断
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """提示词类型"""
    TASK_PLANNING = "task_planning"
    PERCEPTION_REPLANNING = "perception_replanning"
    EXCEPTION_HANDLING = "exception_handling"
    COMMAND_CLARIFICATION = "command_clarification"
    DECISION_MAKING = "decision_making"
    OPERATION_SEQUENCE = "operation_sequence"


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    prompt_type: PromptType
    system_prompt: str
    user_prompt_template: str
    output_format: str
    examples: List[Dict[str, str]]


class CoTPrompts:
    """
    CoT提示词管理器
    
    提供针对不同场景优化的思维链提示词
    """
    
    # 系统角色定义
    SYSTEM_ROLES = {
        "planner": """你是一个专业的无人系统任务规划器。
你的职责是：
1. 分析任务目标和当前环境
2. 生成安全、高效的操作序列
3. 考虑潜在风险和约束条件
4. 提供清晰的推理过程

你需要结合实时感知数据进行规划，确保计划的可行性。""",
        
        "replanner": """你是一个感知驱动的重规划专家。
你的职责是：
1. 分析环境变化对当前计划的影响
2. 决定是否需要调整计划
3. 生成新的计划以适应变化
4. 保持任务目标的连续性

你需要快速响应环境变化，在保证安全的前提下完成任务。""",
        
        "exception_handler": """你是一个异常处理专家。
你的职责是：
1. 分析异常原因和影响
2. 评估可用的恢复选项
3. 推荐最佳处理策略
4. 确保系统安全

你需要冷静分析问题，给出可靠的解决方案。""",
        
        "clarifier": """你是一个智能指令理解助手。
你的职责是：
1. 识别指令中的模糊点
2. 结合环境信息推断可能意图
3. 生成恰当的澄清问题
4. 帮助用户明确指令

你需要友好、简洁地与用户沟通。""",
        
        "decision_maker": """你是一个决策支持系统。
你的职责是：
1. 分析决策所需的各种因素
2. 评估不同选项的利弊
3. 基于证据给出建议
4. 量化决策的置信度

你需要提供透明、可追溯的决策过程。"""
    }
    
    # 任务规划提示词模板
    TASK_PLANNING_TEMPLATE = PromptTemplate(
        name="任务规划",
        prompt_type=PromptType.TASK_PLANNING,
        system_prompt=SYSTEM_ROLES["planner"],
        user_prompt_template="""## 当前感知状态
{perception_context}

## 任务目标
{task_description}

## 可用操作
{available_operations}

## 推理要求
请按照以下步骤进行链式思维推理：

### 步骤1: 环境分析
分析当前环境中的关键因素：
- 障碍物分布和危险区域
- 目标位置和状态
- 可用路径和通行条件
- 环境约束（天气、电量、信号等）

### 步骤2: 任务分解
将任务分解为子任务：
- 识别任务的核心目标
- 确定必要的中间步骤
- 考虑依赖关系

### 步骤3: 方案设计
设计可行的执行方案：
- 生成操作序列
- 估计执行时间
- 预留应急措施

### 步骤4: 风险评估
评估方案的风险：
- 识别潜在问题点
- 评估失败概率
- 准备备选方案

### 步骤5: 最终决策
给出最终的执行计划。

请开始推理：""",
        output_format="""{output_format}

## 输出格式
请按以下JSON格式输出最终计划：
```json
{{
    "decision": "执行/拒绝/需要澄清",
    "operations": [
        {{"name": "操作名", "params": {{}}, "estimated_time": "时间"}},
        ...
    ],
    "reasoning_summary": "推理摘要",
    "confidence": 0.0-1.0,
    "risks": ["风险1", "风险2"],
    "fallback_plan": "备选方案描述"
}}
```""",
        examples=[
            {
                "task": "去东边50米处拍照",
                "context": "当前位置(0,0,10)，东边30米有一棵树",
                "output": """### 步骤1: 环境分析
当前位置在原点上方10米，需要向东移动50米。
东边30米处有障碍物（树），需要绕行或提高高度。

### 步骤2: 任务分解
1. 调整高度确保安全
2. 向东飞行50米
3. 悬停并拍照
4. 返回或等待下一指令

### 步骤3: 方案设计
操作序列：
1. SET_ALTITUDE(15) - 提高到15米避开障碍
2. MOVE_TO(50, 0, 15) - 向东飞行
3. HOVER() - 悬停稳定
4. TAKE_PHOTO() - 拍照

### 步骤4: 风险评估
- 树木高度估计10米，15米高度应该安全
- 无其他明显障碍
- 风险等级：低

### 步骤5: 最终决策
执行上述计划，预计耗时45秒"""
            }
        ]
    )
    
    # 感知驱动重规划提示词
    PERCEPTION_REPLANNING_TEMPLATE = PromptTemplate(
        name="感知驱动重规划",
        prompt_type=PromptType.PERCEPTION_REPLANNING,
        system_prompt=SYSTEM_ROLES["replanner"],
        user_prompt_template="""## 原始任务
{original_task}

## 当前执行状态
{execution_status}

## 环境变化
{environment_changes}

## 当前感知状态
{perception_context}

## 原计划剩余操作
{remaining_operations}

## 推理要求
请分析环境变化并决定是否需要重规划：

### 步骤1: 变化影响分析
分析环境变化对任务执行的影响：
- 变化的性质和严重程度
- 对当前操作的直接影响
- 对后续计划的潜在影响

### 步骤2: 继续/调整判断
判断是否需要调整计划：
- 能否继续原计划？
- 需要什么程度的调整？
- 是否需要完全重新规划？

### 步骤3: 新方案设计（如需要）
如果需要调整，设计新方案：
- 保持原任务目标
- 适应新的环境条件
- 最小化偏离原计划

### 步骤4: 安全性验证
验证新方案的安全性：
- 是否规避了新的障碍/风险
- 资源是否充足
- 是否满足所有约束

### 步骤5: 决策
给出最终决定和新计划。

请开始推理：""",
        output_format="""## 输出格式
```json
{{
    "decision": "continue/adjust/replan/abort",
    "reason": "决策理由",
    "affected_operations": ["受影响的操作"],
    "new_operations": [
        {{"name": "操作名", "params": {{}}}},
        ...
    ],
    "confidence": 0.0-1.0,
    "user_confirmation_required": true/false,
    "confirmation_question": "需要用户确认的问题（如果需要）"
}}
```""",
        examples=[]
    )
    
    # 异常处理提示词
    EXCEPTION_HANDLING_TEMPLATE = PromptTemplate(
        name="异常处理",
        prompt_type=PromptType.EXCEPTION_HANDLING,
        system_prompt=SYSTEM_ROLES["exception_handler"],
        user_prompt_template="""## 异常信息
{exception_info}

## 发生异常的操作
{failed_operation}

## 当前系统状态
{system_state}

## 任务上下文
{task_context}

## 推理要求
请分析异常并推荐处理方案：

### 步骤1: 异常诊断
分析异常的原因：
- 直接原因是什么？
- 是否有潜在的深层问题？
- 异常是否可能复发？

### 步骤2: 影响评估
评估异常的影响范围：
- 对当前任务的影响
- 对系统状态的影响
- 对安全性的影响

### 步骤3: 选项分析
分析可用的处理选项：
- 重试当前操作
- 跳过并继续
- 回滚到检查点
- 中止任务
- 其他替代方案

### 步骤4: 方案评估
评估每个选项：
- 成功概率
- 风险程度
- 资源消耗
- 对任务目标的影响

### 步骤5: 推荐方案
给出推荐的处理方案。

请开始推理：""",
        output_format="""## 输出格式
```json
{{
    "diagnosis": "异常诊断结果",
    "severity": "low/medium/high/critical",
    "recommended_action": "retry/skip/rollback/abort/alternative",
    "action_details": "具体操作说明",
    "recovery_steps": ["恢复步骤1", "恢复步骤2"],
    "confidence": 0.0-1.0,
    "warnings": ["警告信息"]
}}
```""",
        examples=[]
    )
    
    # 指令澄清提示词
    COMMAND_CLARIFICATION_TEMPLATE = PromptTemplate(
        name="指令澄清",
        prompt_type=PromptType.COMMAND_CLARIFICATION,
        system_prompt=SYSTEM_ROLES["clarifier"],
        user_prompt_template="""## 用户指令
{user_command}

## 当前环境
{environment_context}

## 检测到的模糊点
{detected_ambiguities}

## 对话历史
{conversation_history}

## 推理要求
请分析指令并生成澄清问题：

### 步骤1: 指令分析
分析用户指令的意图：
- 核心目标是什么？
- 哪些信息是明确的？
- 哪些信息是缺失或模糊的？

### 步骤2: 环境关联
将指令与当前环境关联：
- 指令中的指代词对应什么？
- 环境中有哪些可能的目标？
- 是否有多种合理解释？

### 步骤3: 问题设计
设计澄清问题：
- 问题要简洁明了
- 如果可能，提供选项
- 避免一次问太多问题

### 步骤4: 确认策略
确定最佳的澄清策略：
- 最需要澄清的是什么？
- 如何引导用户提供信息？

请生成澄清对话：""",
        output_format="""## 输出格式
```json
{{
    "understood_parts": ["已理解的部分"],
    "ambiguous_parts": ["模糊的部分"],
    "clarification_question": "澄清问题",
    "options": ["选项1", "选项2"],
    "default_interpretation": "默认解释（如果用户不回应）",
    "confidence": 0.0-1.0
}}
```""",
        examples=[
            {
                "command": "去那边看看",
                "context": "东边有一栋建筑，西边有一辆车",
                "output": """### 步骤1: 指令分析
用户想要前往某个位置进行观察。
明确信息：需要移动并观察
模糊信息："那边"指向不明确

### 步骤2: 环境关联
当前环境中有两个显著目标：
- 东边的建筑
- 西边的车辆
"那边"可能指其中任何一个

### 步骤3: 问题设计
需要确认具体方向和目标

```json
{
    "understood_parts": ["需要移动", "需要观察"],
    "ambiguous_parts": ["目标方向", "观察对象"],
    "clarification_question": "您说的'那边'是指哪个方向？",
    "options": ["东边的建筑", "西边的车辆", "其他方向"],
    "default_interpretation": "最近的兴趣点",
    "confidence": 0.6
}
```"""
            }
        ]
    )
    
    # 决策判断提示词
    DECISION_MAKING_TEMPLATE = PromptTemplate(
        name="决策判断",
        prompt_type=PromptType.DECISION_MAKING,
        system_prompt=SYSTEM_ROLES["decision_maker"],
        user_prompt_template="""## 决策问题
{decision_question}

## 可用选项
{options}

## 相关信息
{relevant_info}

## 约束条件
{constraints}

## 推理要求
请进行决策分析：

### 步骤1: 问题理解
明确决策的核心问题：
- 需要做出什么决定？
- 决策的影响范围是什么？
- 时间紧迫程度如何？

### 步骤2: 选项分析
分析每个选项：
- 选项的优势
- 选项的劣势
- 潜在风险

### 步骤3: 约束检查
检查约束条件：
- 哪些选项满足所有约束？
- 哪些约束是硬性的？
- 是否有可以放宽的约束？

### 步骤4: 证据权衡
基于证据进行权衡：
- 支持各选项的证据
- 证据的可靠性
- 不确定性因素

### 步骤5: 最终决策
给出推荐决策和置信度。

请开始推理：""",
        output_format="""## 输出格式
```json
{{
    "decision": "推荐的选项",
    "confidence": 0.0-1.0,
    "reasoning": "决策理由",
    "pros": ["优点1", "优点2"],
    "cons": ["缺点1", "缺点2"],
    "alternatives": ["次优选项"],
    "conditions": ["决策成立的条件"]
}}
```""",
        examples=[]
    )
    
    # 操作序列生成提示词
    OPERATION_SEQUENCE_TEMPLATE = PromptTemplate(
        name="操作序列生成",
        prompt_type=PromptType.OPERATION_SEQUENCE,
        system_prompt=SYSTEM_ROLES["planner"],
        user_prompt_template="""## 平台类型
{platform_type}

## 任务描述
{task_description}

## 当前状态
{current_state}

## 可用原子操作
{atomic_operations}

## 约束条件
{constraints}

## 要求
请生成一个可执行的原子操作序列：

### 分析任务
理解任务的本质需求

### 规划路径
确定完成任务的步骤

### 生成序列
输出具体的操作序列

### 验证可行性
检查序列的可执行性

请生成操作序列：""",
        output_format="""## 输出格式
```json
{{
    "task_analysis": "任务分析",
    "operations": [
        {{
            "step": 1,
            "operation": "操作名称",
            "parameters": {{}},
            "preconditions": ["前置条件"],
            "expected_result": "预期结果",
            "timeout": 10
        }}
    ],
    "total_estimated_time": "总预计时间",
    "checkpoints": [1, 3, 5],
    "abort_conditions": ["中止条件"]
}}
```""",
        examples=[]
    )
    
    def __init__(self):
        """初始化提示词管理器"""
        self.templates: Dict[PromptType, PromptTemplate] = {
            PromptType.TASK_PLANNING: self.TASK_PLANNING_TEMPLATE,
            PromptType.PERCEPTION_REPLANNING: self.PERCEPTION_REPLANNING_TEMPLATE,
            PromptType.EXCEPTION_HANDLING: self.EXCEPTION_HANDLING_TEMPLATE,
            PromptType.COMMAND_CLARIFICATION: self.COMMAND_CLARIFICATION_TEMPLATE,
            PromptType.DECISION_MAKING: self.DECISION_MAKING_TEMPLATE,
            PromptType.OPERATION_SEQUENCE: self.OPERATION_SEQUENCE_TEMPLATE,
        }
        
        # 自定义模板存储
        self.custom_templates: Dict[str, PromptTemplate] = {}
    
    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """获取提示词模板"""
        return self.templates.get(prompt_type)
    
    def build_prompt(
        self,
        prompt_type: PromptType,
        variables: Dict[str, str]
    ) -> Dict[str, str]:
        """
        构建完整的提示词
        
        Args:
            prompt_type: 提示词类型
            variables: 变量替换字典
            
        Returns:
            Dict: 包含system_prompt和user_prompt
        """
        template = self.templates.get(prompt_type)
        if not template:
            raise ValueError(f"未知的提示词类型: {prompt_type}")
        
        # 替换用户提示词中的变量
        user_prompt = template.user_prompt_template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            user_prompt = user_prompt.replace(placeholder, str(value))
        
        # 添加输出格式说明
        if template.output_format:
            output_format = template.output_format
            for key, value in variables.items():
                placeholder = "{" + key + "}"
                output_format = output_format.replace(placeholder, str(value))
            user_prompt += "\n\n" + output_format
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt
        }
    
    def build_planning_prompt(
        self,
        task_description: str,
        perception_context: str,
        available_operations: str
    ) -> Dict[str, str]:
        """构建任务规划提示词"""
        return self.build_prompt(
            PromptType.TASK_PLANNING,
            {
                "task_description": task_description,
                "perception_context": perception_context,
                "available_operations": available_operations,
                "output_format": ""
            }
        )
    
    def build_replanning_prompt(
        self,
        original_task: str,
        execution_status: str,
        environment_changes: str,
        perception_context: str,
        remaining_operations: str
    ) -> Dict[str, str]:
        """构建重规划提示词"""
        return self.build_prompt(
            PromptType.PERCEPTION_REPLANNING,
            {
                "original_task": original_task,
                "execution_status": execution_status,
                "environment_changes": environment_changes,
                "perception_context": perception_context,
                "remaining_operations": remaining_operations
            }
        )
    
    def build_exception_prompt(
        self,
        exception_info: str,
        failed_operation: str,
        system_state: str,
        task_context: str
    ) -> Dict[str, str]:
        """构建异常处理提示词"""
        return self.build_prompt(
            PromptType.EXCEPTION_HANDLING,
            {
                "exception_info": exception_info,
                "failed_operation": failed_operation,
                "system_state": system_state,
                "task_context": task_context
            }
        )
    
    def build_clarification_prompt(
        self,
        user_command: str,
        environment_context: str,
        detected_ambiguities: str,
        conversation_history: str = "无"
    ) -> Dict[str, str]:
        """构建指令澄清提示词"""
        return self.build_prompt(
            PromptType.COMMAND_CLARIFICATION,
            {
                "user_command": user_command,
                "environment_context": environment_context,
                "detected_ambiguities": detected_ambiguities,
                "conversation_history": conversation_history
            }
        )
    
    def build_decision_prompt(
        self,
        decision_question: str,
        options: str,
        relevant_info: str,
        constraints: str
    ) -> Dict[str, str]:
        """构建决策判断提示词"""
        return self.build_prompt(
            PromptType.DECISION_MAKING,
            {
                "decision_question": decision_question,
                "options": options,
                "relevant_info": relevant_info,
                "constraints": constraints
            }
        )
    
    def build_operation_sequence_prompt(
        self,
        platform_type: str,
        task_description: str,
        current_state: str,
        atomic_operations: str,
        constraints: str
    ) -> Dict[str, str]:
        """构建操作序列生成提示词"""
        return self.build_prompt(
            PromptType.OPERATION_SEQUENCE,
            {
                "platform_type": platform_type,
                "task_description": task_description,
                "current_state": current_state,
                "atomic_operations": atomic_operations,
                "constraints": constraints
            }
        )
    
    def add_custom_template(
        self,
        name: str,
        system_prompt: str,
        user_prompt_template: str,
        output_format: str = ""
    ):
        """添加自定义模板"""
        self.custom_templates[name] = PromptTemplate(
            name=name,
            prompt_type=PromptType.DECISION_MAKING,  # 默认类型
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            output_format=output_format,
            examples=[]
        )
    
    def get_custom_template(self, name: str) -> Optional[PromptTemplate]:
        """获取自定义模板"""
        return self.custom_templates.get(name)
    
    def build_custom_prompt(
        self,
        template_name: str,
        variables: Dict[str, str]
    ) -> Dict[str, str]:
        """使用自定义模板构建提示词"""
        template = self.custom_templates.get(template_name)
        if not template:
            raise ValueError(f"未找到自定义模板: {template_name}")
        
        user_prompt = template.user_prompt_template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            user_prompt = user_prompt.replace(placeholder, str(value))
        
        if template.output_format:
            user_prompt += "\n\n" + template.output_format
        
        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt
        }

