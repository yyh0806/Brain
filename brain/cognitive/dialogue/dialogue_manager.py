"""
多轮对话管理器 - Dialogue Manager

负责：
- 管理与用户/LLM的多轮对话
- 指令澄清：当指令模糊时主动询问
- 执行确认：关键操作前请求确认
- 进度汇报：执行过程中汇报并接收调整
- 维护对话历史和上下文
"""

from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
import asyncio
from loguru import logger

# 导入类型定义
from brain.cognitive.dialogue.dialogue_types import (
    DialogueType,
    DialogueState,
    DialogueMessage,
    DialogueContext
)


class DialogueManager:
    """
    多轮对话管理器
    
    管理与用户的交互，支持澄清、确认、汇报等对话模式
    """
    
    def __init__(
        self,
        llm_interface: Optional[Any] = None,
        user_callback: Optional[Callable[[str, List[str]], Awaitable[str]]] = None
    ):
        """
        Args:
            llm_interface: LLM接口，用于生成对话
            user_callback: 用户交互回调函数
        """
        self.llm = llm_interface
        self.user_callback = user_callback
        
        # 当前对话上下文
        self.current_context: Optional[DialogueContext] = None
        
        # 自动确认模式（用于测试）
        self.auto_confirm = False
        self.auto_confirm_delay = 0.5
        
        # 对话历史存档
        self.archived_contexts: List[DialogueContext] = []
        
        logger.info("DialogueManager 初始化完成")
    
    def start_session(self, session_id: str, mission_id: Optional[str] = None) -> DialogueContext:
        """开始新的对话会话"""
        if self.current_context:
            self.archived_contexts.append(self.current_context)
        
        self.current_context = DialogueContext(
            session_id=session_id,
            mission_id=mission_id
        )
        
        logger.info(f"开始对话会话: {session_id}")
        return self.current_context
    
    def end_session(self):
        """结束当前会话"""
        if self.current_context:
            self.current_context.state = DialogueState.COMPLETED
            self.archived_contexts.append(self.current_context)
            logger.info(f"结束对话会话: {self.current_context.session_id}")
            self.current_context = None
    
    async def clarify_ambiguous_command(
        self,
        command: str,
        ambiguities: List[Dict[str, Any]],
        world_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        当指令模糊时，生成澄清问题
        
        Args:
            command: 原始指令
            ambiguities: 模糊点列表
            world_context: 世界模型上下文
            
        Returns:
            Dict: 包含澄清问题和用户回答
        """
        if not self.current_context:
            self.start_session(f"clarify_{datetime.now().timestamp()}")
        
        # 构建澄清提示
        prompt = self._build_clarification_prompt(command, ambiguities, world_context)
        
        # 使用LLM生成澄清问题
        if self.llm:
            question = await self._generate_clarification_question(prompt)
        else:
            # 默认问题
            question = f"关于指令 '{command}'，请问: {ambiguities[0].get('question', '请提供更多细节')}"
        
        # 记录到对话历史
        self.current_context.add_message(DialogueMessage(
            role="assistant",
            content=question,
            dialogue_type=DialogueType.CLARIFICATION,
            metadata={"original_command": command, "ambiguities": ambiguities}
        ))
        
        self.current_context.state = DialogueState.WAITING_USER_RESPONSE
        self.current_context.pending_question = question
        
        # 获取用户回答
        user_response = await self._get_user_response(
            question,
            options=ambiguities[0].get("options") if ambiguities else None
        )
        
        # 记录用户回答
        self.current_context.add_message(DialogueMessage(
            role="user",
            content=user_response,
            dialogue_type=DialogueType.USER_INPUT
        ))
        
        self.current_context.state = DialogueState.PROCESSING
        
        return {
            "question": question,
            "response": user_response,
            "clarified_command": f"{command} ({user_response})"
        }
    
    async def request_confirmation(
        self,
        action: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None
    ) -> bool:
        """
        关键操作前请求用户确认
        
        Args:
            action: 要执行的操作描述
            reason: 需要确认的原因
            details: 额外详情
            options: 可选选项（默认 ["确认", "取消"]）
            
        Returns:
            bool: 用户是否确认
        """
        if not self.current_context:
            self.start_session(f"confirm_{datetime.now().timestamp()}")
        
        options = options or ["确认执行", "取消操作", "修改计划"]
        
        # 构建确认消息
        message = f"需要确认:\n\n操作: {action}\n原因: {reason}"
        if details:
            detail_str = "\n".join([f"- {k}: {v}" for k, v in details.items()])
            message += f"\n\n详情:\n{detail_str}"
        message += f"\n\n请选择: {' / '.join(options)}"
        
        # 记录到对话历史
        self.current_context.add_message(DialogueMessage(
            role="assistant",
            content=message,
            dialogue_type=DialogueType.CONFIRMATION,
            metadata={"action": action, "reason": reason, "details": details}
        ))
        
        self.current_context.state = DialogueState.WAITING_USER_RESPONSE
        self.current_context.pending_question = message
        self.current_context.pending_options = options
        
        # 获取用户响应
        response = await self._get_user_response(message, options)
        
        # 记录用户响应
        self.current_context.add_message(DialogueMessage(
            role="user",
            content=response,
            dialogue_type=DialogueType.USER_INPUT
        ))
        
        self.current_context.state = DialogueState.PROCESSING
        
        # 判断是否确认
        confirmed = self._parse_confirmation(response, options)
        
        logger.info(f"用户确认: {confirmed} (响应: {response})")
        return confirmed
    
    async def report_progress(
        self,
        status: str,
        progress_percent: float,
        current_operation: Optional[str] = None,
        world_state_summary: Optional[str] = None,
        allow_adjustment: bool = True
    ) -> Optional[str]:
        """
        汇报执行进度，接收用户调整
        
        Args:
            status: 当前状态描述
            progress_percent: 进度百分比
            current_operation: 当前操作
            world_state_summary: 世界状态摘要
            allow_adjustment: 是否允许用户调整
            
        Returns:
            Optional[str]: 用户的调整指令（如果有）
        """
        if not self.current_context:
            self.start_session(f"progress_{datetime.now().timestamp()}")
        
        # 构建进度报告
        report_lines = [
            f"📊 执行进度: {progress_percent:.0f}%",
            f"状态: {status}"
        ]
        
        if current_operation:
            report_lines.append(f"当前操作: {current_operation}")
        
        if world_state_summary:
            report_lines.append(f"\n环境状态:\n{world_state_summary}")
        
        if allow_adjustment:
            report_lines.append("\n如需调整，请输入指令。输入 '继续' 或直接回车继续执行。")
        
        report = "\n".join(report_lines)
        
        # 记录到对话历史
        self.current_context.add_message(DialogueMessage(
            role="assistant",
            content=report,
            dialogue_type=DialogueType.PROGRESS_REPORT,
            metadata={
                "progress": progress_percent,
                "status": status,
                "operation": current_operation
            }
        ))
        
        if allow_adjustment:
            self.current_context.state = DialogueState.WAITING_USER_RESPONSE
            
            # 获取用户响应（可选）
            response = await self._get_user_response(
                report,
                options=["继续", "暂停", "取消", "调整"],
                timeout=5.0  # 5秒超时
            )
            
            # 记录用户响应
            if response and response not in ["继续", "", "continue"]:
                self.current_context.add_message(DialogueMessage(
                    role="user",
                    content=response,
                    dialogue_type=DialogueType.USER_INPUT
                ))
                return response
        
        return None
    
    async def report_and_confirm(
        self,
        message: str,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        汇报情况并请求确认
        
        用于感知变化时的汇报确认
        """
        full_message = message
        if suggestion:
            full_message += f"\n建议: {suggestion}"
        
        return await self.request_confirmation(
            action=suggestion or "继续执行",
            reason=message,
            details=details
        )
    
    async def report_error(
        self,
        error: str,
        operation: str,
        suggestions: List[str],
        allow_choice: bool = True
    ) -> str:
        """
        报告错误并提供处理选项
        
        Args:
            error: 错误信息
            operation: 出错的操作
            suggestions: 处理建议列表
            allow_choice: 是否允许用户选择
            
        Returns:
            str: 用户选择的处理方式
        """
        if not self.current_context:
            self.start_session(f"error_{datetime.now().timestamp()}")
        
        # 构建错误报告
        report_lines = [
            f"⚠️ 执行错误",
            f"操作: {operation}",
            f"错误: {error}",
            "\n建议处理方式:"
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            report_lines.append(f"  {i}. {suggestion}")
        
        report = "\n".join(report_lines)
        
        # 记录到对话历史
        self.current_context.add_message(DialogueMessage(
            role="assistant",
            content=report,
            dialogue_type=DialogueType.ERROR_REPORT,
            metadata={"error": error, "operation": operation, "suggestions": suggestions}
        ))
        
        if allow_choice:
            response = await self._get_user_response(report, options=suggestions)
            
            self.current_context.add_message(DialogueMessage(
                role="user",
                content=response,
                dialogue_type=DialogueType.USER_INPUT
            ))
            
            return response
        
        return suggestions[0] if suggestions else "跳过"
    
    async def send_information(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """发送信息通知（不需要响应）"""
        if not self.current_context:
            self.start_session(f"info_{datetime.now().timestamp()}")
        
        self.current_context.add_message(DialogueMessage(
            role="assistant",
            content=message,
            dialogue_type=DialogueType.INFORMATION,
            metadata=metadata or {}
        ))
        
        # 通知用户（如果有回调）
        if self.user_callback:
            try:
                await asyncio.wait_for(
                    self.user_callback(message, []),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                pass
        
        logger.info(f"[信息] {message}")
    
    def _build_clarification_prompt(
        self,
        command: str,
        ambiguities: List[Dict[str, Any]],
        world_context: Optional[str]
    ) -> str:
        """构建澄清提示"""
        prompt = f"""用户给出了一个模糊的指令，需要澄清。

原始指令: "{command}"

模糊点:
"""
        for amb in ambiguities:
            prompt += f"- {amb.get('aspect', '未知')}: {amb.get('question', '需要更多信息')}\n"
            if amb.get('options'):
                prompt += f"  可能的选项: {', '.join(amb['options'])}\n"
        
        if world_context:
            prompt += f"\n当前环境:\n{world_context}\n"
        
        prompt += "\n请生成一个友好、清晰的问题来澄清用户意图。问题应该简洁明了，如果有选项可以列出供用户选择。"
        
        return prompt
    
    async def _generate_clarification_question(self, prompt: str) -> str:
        """使用LLM生成澄清问题"""
        try:
            from brain.models.llm_interface import LLMMessage
            
            messages = [
                LLMMessage(role="system", content="你是一个友好的无人系统助手，帮助用户明确指令。"),
                LLMMessage(role="user", content=prompt)
            ]
            
            response = await self.llm.chat(messages)
            return response.content
            
        except Exception as e:
            logger.warning(f"LLM生成澄清问题失败: {e}")
            return "请提供更多细节以便我更好地理解您的指令。"
    
    async def _get_user_response(
        self,
        prompt: str,
        options: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> str:
        """获取用户响应"""
        # 自动确认模式（用于测试）
        if self.auto_confirm:
            await asyncio.sleep(self.auto_confirm_delay)
            if options:
                return options[0]
            return "确认"
        
        # 使用回调获取用户输入
        if self.user_callback:
            try:
                if timeout:
                    response = await asyncio.wait_for(
                        self.user_callback(prompt, options or []),
                        timeout=timeout
                    )
                else:
                    response = await self.user_callback(prompt, options or [])
                return response
            except asyncio.TimeoutError:
                logger.debug("用户响应超时，使用默认选项")
                return options[0] if options else "继续"
            except Exception as e:
                logger.warning(f"获取用户响应失败: {e}")
                return options[0] if options else "继续"
        
        # 无回调，使用默认值
        logger.warning("无用户交互回调，使用默认确认")
        return options[0] if options else "确认"
    
    def _parse_confirmation(self, response: str, options: List[str]) -> bool:
        """解析确认响应"""
        response_lower = response.lower().strip()
        
        # 肯定词
        positive_words = ["确认", "是", "好", "yes", "ok", "继续", "确定", "同意", "执行"]
        # 否定词
        negative_words = ["取消", "否", "不", "no", "cancel", "停止", "拒绝"]
        
        for word in positive_words:
            if word in response_lower:
                return True
        
        for word in negative_words:
            if word in response_lower:
                return False
        
        # 检查是否选择了第一个选项（通常是确认）
        if options and response in options:
            return options.index(response) == 0
        
        # 默认为确认
        return True
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取当前对话历史"""
        if self.current_context:
            return [m.to_dict() for m in self.current_context.history]
        return []
    
    def set_auto_confirm(self, enabled: bool, delay: float = 0.5):
        """设置自动确认模式（用于测试）"""
        self.auto_confirm = enabled
        self.auto_confirm_delay = delay
        logger.info(f"自动确认模式: {'启用' if enabled else '禁用'}")
    
    def set_user_callback(self, callback: Callable[[str, List[str]], Awaitable[str]]):
        """设置用户交互回调"""
        self.user_callback = callback

