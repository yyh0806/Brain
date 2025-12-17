"""
任务解析器 - Task Parser

负责:
- 使用LLM解析自然语言指令
- 提取任务结构
- 验证解析结果
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from brain.models.llm_interface import LLMInterface, LLMMessage
from brain.models.prompt_templates import PromptTemplates


@dataclass
class ParsedTask:
    """解析后的任务"""
    task_type: str
    name: str
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    subtasks: List['ParsedTask'] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: str = ""
    raw_response: Optional[Dict[str, Any]] = None
    parse_time: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_type": self.task_type,
            "name": self.name,
            "priority": self.priority,
            "parameters": self.parameters,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "constraints": self.constraints,
            "success_criteria": self.success_criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParsedTask':
        """从字典创建"""
        subtasks = [
            cls.from_dict(st) 
            for st in data.get("subtasks", [])
        ]
        
        return cls(
            task_type=data.get("task_type", "unknown"),
            name=data.get("name", "unnamed"),
            priority=data.get("priority", 1),
            parameters=data.get("parameters", {}),
            subtasks=subtasks,
            constraints=data.get("constraints", {}),
            success_criteria=data.get("success_criteria", "")
        )


class TaskParser:
    """
    任务解析器
    
    使用LLM将自然语言指令解析为结构化任务
    """
    
    # 支持的任务类型
    SUPPORTED_TASK_TYPES = [
        "patrol",       # 巡逻
        "survey",       # 勘察
        "delivery",     # 运输
        "inspection",   # 检查
        "monitoring",   # 监控
        "search",       # 搜索
        "rescue",       # 救援
        "mapping",      # 测绘
        "tracking",     # 跟踪
        "custom"        # 自定义
    ]
    
    # 平台特定参数
    PLATFORM_PARAMS = {
        "drone": {
            "required": ["altitude"],
            "optional": ["speed", "heading", "camera_angle"]
        },
        "ugv": {
            "required": [],
            "optional": ["speed", "path_type"]
        },
        "usv": {
            "required": [],
            "optional": ["speed", "depth"]
        }
    }
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.template = PromptTemplates.get_template("task_parsing")
        
        logger.info("TaskParser 初始化完成")
    
    async def parse(
        self,
        command: str,
        platform_type: str,
        environment_state: Optional[Dict[str, Any]] = None,
        world_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        解析自然语言指令
        
        Args:
            command: 自然语言指令
            platform_type: 平台类型
            environment_state: 环境状态
            world_state: 世界状态
            
        Returns:
            Dict: 解析后的任务字典
        """
        logger.info(f"解析指令: {command}")
        
        # 预处理指令
        processed_command = self._preprocess_command(command)
        
        # 格式化提示词
        user_prompt = self.template.format(
            platform_type=platform_type,
            command=processed_command,
            environment_state=json.dumps(
                environment_state or {}, 
                ensure_ascii=False, 
                indent=2
            ),
            world_state=json.dumps(
                world_state or {}, 
                ensure_ascii=False, 
                indent=2
            )
        )
        
        # 调用LLM
        messages = [
            LLMMessage(role="system", content=self.template.system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]
        
        response = await self.llm.chat(messages)
        
        # 解析响应
        try:
            parsed = self._parse_response(response.content)
            
            # 验证解析结果
            validated = self._validate_parsed_task(parsed, platform_type)
            
            # 补充缺失的默认值
            enriched = self._enrich_task(validated, platform_type)
            
            logger.info(f"任务解析成功: {enriched.get('task_type', 'unknown')}")
            
            return enriched
            
        except Exception as e:
            logger.error(f"任务解析失败: {e}")
            # 返回基本任务结构
            return self._create_fallback_task(command, platform_type)
    
    def _preprocess_command(self, command: str) -> str:
        """预处理指令"""
        # 去除多余空白
        command = " ".join(command.split())
        
        # 规范化常见表达
        replacements = {
            "飞到": "前往",
            "开到": "前往",
            "航行到": "前往",
            "拍一下": "拍照",
            "录一下": "录像",
            "看一下": "检查",
            "找一下": "搜索"
        }
        
        for old, new in replacements.items():
            command = command.replace(old, new)
        
        return command
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """解析LLM响应"""
        # 尝试直接解析JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"无法解析响应为JSON: {content[:200]}...")
    
    def _validate_parsed_task(
        self, 
        parsed: Dict[str, Any],
        platform_type: str
    ) -> Dict[str, Any]:
        """验证解析后的任务"""
        # 验证任务类型
        task_type = parsed.get("task_type", "").lower()
        if task_type not in self.SUPPORTED_TASK_TYPES:
            logger.warning(f"未知任务类型: {task_type}, 使用 custom")
            parsed["task_type"] = "custom"
        
        # 验证优先级
        priority = parsed.get("priority", 1)
        if not isinstance(priority, int) or priority < 1 or priority > 5:
            parsed["priority"] = 3
        
        # 验证平台特定参数
        params = parsed.get("parameters", {})
        platform_config = self.PLATFORM_PARAMS.get(platform_type, {})
        
        for required_param in platform_config.get("required", []):
            if required_param not in params:
                logger.warning(f"缺少必需参数: {required_param}")
        
        return parsed
    
    def _enrich_task(
        self, 
        parsed: Dict[str, Any],
        platform_type: str
    ) -> Dict[str, Any]:
        """补充任务默认值"""
        # 默认参数
        default_params = {
            "drone": {
                "altitude": 30.0,
                "speed": 5.0
            },
            "ugv": {
                "speed": 2.0
            },
            "usv": {
                "speed": 3.0
            }
        }
        
        params = parsed.get("parameters", {})
        defaults = default_params.get(platform_type, {})
        
        for key, value in defaults.items():
            if key not in params:
                params[key] = value
        
        parsed["parameters"] = params
        
        # 确保有任务名称
        if not parsed.get("name"):
            parsed["name"] = f"{parsed.get('task_type', 'task')}_{datetime.now().strftime('%H%M%S')}"
        
        return parsed
    
    def _create_fallback_task(
        self, 
        command: str,
        platform_type: str
    ) -> Dict[str, Any]:
        """创建后备任务结构"""
        # 简单的关键词匹配
        task_type = "custom"
        
        keywords = {
            "巡逻": "patrol",
            "巡视": "patrol",
            "勘察": "survey",
            "勘测": "survey",
            "运送": "delivery",
            "配送": "delivery",
            "检查": "inspection",
            "检测": "inspection",
            "监控": "monitoring",
            "监视": "monitoring",
            "搜索": "search",
            "寻找": "search",
            "测绘": "mapping",
            "建图": "mapping"
        }
        
        for keyword, ttype in keywords.items():
            if keyword in command:
                task_type = ttype
                break
        
        return {
            "task_type": task_type,
            "name": f"task_{datetime.now().strftime('%H%M%S')}",
            "priority": 3,
            "parameters": {
                "raw_command": command
            },
            "subtasks": [],
            "constraints": {},
            "success_criteria": "任务完成"
        }
    
    async def parse_batch(
        self,
        commands: List[str],
        platform_type: str,
        environment_state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量解析指令
        
        Args:
            commands: 指令列表
            platform_type: 平台类型
            environment_state: 环境状态
            
        Returns:
            List[Dict]: 解析结果列表
        """
        results = []
        
        for command in commands:
            try:
                parsed = await self.parse(
                    command=command,
                    platform_type=platform_type,
                    environment_state=environment_state
                )
                results.append(parsed)
            except Exception as e:
                logger.error(f"解析失败: {command[:50]}... - {e}")
                results.append(self._create_fallback_task(command, platform_type))
        
        return results
    
    async def extract_positions(
        self, 
        command: str
    ) -> List[Dict[str, Any]]:
        """
        从指令中提取位置信息
        
        Args:
            command: 自然语言指令
            
        Returns:
            List[Dict]: 位置列表
        """
        prompt = f"""从以下指令中提取所有提到的位置信息:

指令: {command}

请以JSON数组格式返回，每个位置包含:
- name: 位置名称
- type: 位置类型 (point/area/path)
- coordinates: 坐标（如果明确给出）
- description: 描述

如果没有明确的坐标，coordinates可以为null。"""

        response = await self.llm.complete(prompt)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return []
    
    async def extract_constraints(
        self, 
        command: str
    ) -> Dict[str, Any]:
        """
        从指令中提取约束条件
        
        Args:
            command: 自然语言指令
            
        Returns:
            Dict: 约束条件
        """
        prompt = f"""从以下指令中提取所有约束条件:

指令: {command}

请以JSON格式返回，包含:
- time_constraints: 时间约束
- spatial_constraints: 空间约束
- resource_constraints: 资源约束
- safety_constraints: 安全约束
- other_constraints: 其他约束

每个约束包含 type, value, description 字段。"""

        response = await self.llm.complete(prompt)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {}

