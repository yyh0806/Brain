"""
提示词模板 - Prompt Templates

为各种任务提供标准化的提示词模板
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from string import Template


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    system_prompt: str
    user_prompt_template: str
    output_format: str = "json"
    examples: List[Dict[str, str]] = None
    
    def format(self, **kwargs) -> str:
        """格式化用户提示词"""
        return Template(self.user_prompt_template).safe_substitute(**kwargs)


class PromptTemplates:
    """
    提示词模板集合
    
    提供任务规划相关的各种提示词模板
    """
    
    # 任务解析模板
    TASK_PARSING = PromptTemplate(
        name="task_parsing",
        system_prompt="""你是一个无人系统任务规划专家。你的职责是理解自然语言指令，并将其分解为结构化的任务描述。

你需要分析用户指令，识别:
1. 任务类型 (巡逻、监控、运输、搜索、测绘等)
2. 目标位置或区域
3. 执行参数 (高度、速度、时间等)
4. 约束条件 (禁区、时限等)
5. 成功条件

请严格按照指定的JSON格式输出。""",
        user_prompt_template="""
平台类型: $platform_type
用户指令: $command

当前环境状态:
$environment_state

当前系统状态:
$world_state

请将上述指令解析为结构化任务描述。

输出格式:
{
    "task_type": "任务类型",
    "name": "任务名称",
    "priority": 1-5的优先级,
    "parameters": {
        "target": "目标描述或位置",
        "area": "操作区域",
        "altitude": 高度(米),
        "speed": 速度(米/秒),
        "duration": 持续时间(秒)
    },
    "subtasks": [
        {
            "task_type": "子任务类型",
            "name": "子任务名称",
            "parameters": {...},
            "dependencies": ["依赖的子任务名称"]
        }
    ],
    "constraints": {
        "time_limit": 时间限制(秒),
        "no_fly_zones": ["禁飞区列表"],
        "weather_conditions": "天气要求"
    },
    "success_criteria": "成功条件描述"
}""",
        output_format="json"
    )
    
    # 错误分析模板
    ERROR_ANALYSIS = PromptTemplate(
        name="error_analysis",
        system_prompt="""你是一个无人系统故障分析专家。你的职责是分析操作执行失败的原因，并提供恢复建议。

请分析错误的:
1. 根本原因
2. 是否可恢复
3. 推荐的恢复策略
4. 是否需要重规划

请严格按照指定的JSON格式输出。""",
        user_prompt_template="""
失败的操作:
- 名称: $operation_name
- 类型: $operation_type
- 参数: $operation_params

错误信息: $error_message

当前世界状态:
$world_state

请分析此错误并提供恢复建议。

输出格式:
{
    "error_type": "错误类型",
    "root_cause": "根本原因分析",
    "severity": "严重程度(low/medium/high/critical)",
    "recoverable": true/false,
    "recovery_strategy": "恢复策略",
    "needs_replan": true/false,
    "can_rollback": true/false,
    "retry_recommended": true/false,
    "alternative_actions": ["可选的替代操作"],
    "safety_concerns": ["安全注意事项"]
}""",
        output_format="json"
    )
    
    # 重规划模板
    REPLANNING = PromptTemplate(
        name="replanning",
        system_prompt="""你是一个无人系统任务规划专家。你的职责是在任务执行失败后，根据当前状态重新规划剩余任务。

规划时需要考虑:
1. 已完成的操作
2. 失败的操作及原因
3. 当前环境状态变化
4. 剩余的任务目标
5. 可用资源和约束

请生成一个新的操作序列来完成任务目标。""",
        user_prompt_template="""
原始任务指令: $original_command

已完成的操作:
$completed_operations

失败的操作:
$failed_operation

失败原因: $error

当前环境状态:
$environment_state

当前系统状态:
$world_state

请重新规划完成任务所需的操作序列。

输出格式:
{
    "replan_reason": "重规划原因",
    "strategy": "重规划策略",
    "operations": [
        {
            "name": "操作名称",
            "type": "操作类型",
            "parameters": {...},
            "estimated_duration": 预计时间(秒)
        }
    ],
    "risk_assessment": "风险评估",
    "alternative_plan": "备选方案（如果有）"
}""",
        output_format="json"
    )
    
    # 操作验证模板
    OPERATION_VALIDATION = PromptTemplate(
        name="operation_validation",
        system_prompt="""你是一个无人系统安全专家。你的职责是验证操作序列的安全性和可行性。

验证内容:
1. 操作序列的逻辑一致性
2. 参数的合理性
3. 安全约束的满足
4. 资源可用性
5. 潜在风险

请严格按照指定的JSON格式输出。""",
        user_prompt_template="""
平台类型: $platform_type

操作序列:
$operations

安全约束:
$constraints

当前状态:
$world_state

请验证此操作序列。

输出格式:
{
    "valid": true/false,
    "issues": [
        {
            "operation_index": 操作索引,
            "issue_type": "问题类型",
            "description": "问题描述",
            "severity": "严重程度",
            "suggestion": "修改建议"
        }
    ],
    "warnings": ["警告信息"],
    "risk_level": "overall风险等级(low/medium/high)",
    "recommendations": ["优化建议"]
}""",
        output_format="json"
    )
    
    # 场景理解模板
    SCENE_UNDERSTANDING = PromptTemplate(
        name="scene_understanding",
        system_prompt="""你是一个环境感知专家。你的职责是理解和描述无人系统所处的场景环境。

请分析:
1. 场景类型和特征
2. 检测到的物体
3. 潜在的障碍和风险
4. 适合执行的任务类型
5. 环境条件""",
        user_prompt_template="""
传感器数据摘要:
$sensor_data

检测到的物体:
$detected_objects

当前位置: $current_position

请分析当前场景。

输出格式:
{
    "scene_type": "场景类型",
    "description": "场景描述",
    "key_features": ["关键特征"],
    "obstacles": [
        {
            "type": "障碍类型",
            "position": "位置",
            "risk_level": "风险等级"
        }
    ],
    "safe_zones": ["安全区域"],
    "recommended_actions": ["推荐操作"],
    "environmental_conditions": {
        "visibility": "能见度",
        "terrain": "地形",
        "weather_impact": "天气影响"
    }
}""",
        output_format="json"
    )
    
    # 任务优化模板
    TASK_OPTIMIZATION = PromptTemplate(
        name="task_optimization",
        system_prompt="""你是一个无人系统任务优化专家。你的职责是优化操作序列以提高效率和安全性。

优化目标:
1. 减少总执行时间
2. 降低能耗
3. 提高任务成功率
4. 增强安全性
5. 优化路径""",
        user_prompt_template="""
当前操作序列:
$operations

平台参数:
$platform_params

环境约束:
$constraints

请优化此操作序列。

输出格式:
{
    "optimization_applied": ["应用的优化"],
    "optimized_operations": [
        {
            "name": "操作名称",
            "parameters": {...},
            "changes": ["修改说明"]
        }
    ],
    "estimated_improvement": {
        "time_saved": "节省时间",
        "energy_saved": "节省能耗",
        "safety_improvement": "安全性提升"
    },
    "trade_offs": ["权衡说明"]
}""",
        output_format="json"
    )
    
    @classmethod
    def get_template(cls, name: str) -> Optional[PromptTemplate]:
        """获取指定名称的模板"""
        templates = {
            "task_parsing": cls.TASK_PARSING,
            "error_analysis": cls.ERROR_ANALYSIS,
            "replanning": cls.REPLANNING,
            "operation_validation": cls.OPERATION_VALIDATION,
            "scene_understanding": cls.SCENE_UNDERSTANDING,
            "task_optimization": cls.TASK_OPTIMIZATION
        }
        return templates.get(name)
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """列出所有可用模板"""
        return [
            "task_parsing",
            "error_analysis", 
            "replanning",
            "operation_validation",
            "scene_understanding",
            "task_optimization"
        ]

