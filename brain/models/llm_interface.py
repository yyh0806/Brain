"""
LLM接口 - LLM Interface

负责:
- 与大语言模型通信
- 处理请求和响应
- 错误重试
- 多模型支持
"""

import asyncio
import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger


class LLMProvider(Enum):
    """LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE = "azure"
    CUSTOM = "custom"
    OLLAMA = "ollama"  # 本地Ollama


@dataclass
class LLMMessage:
    """LLM消息"""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None
    

@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    finish_reason: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Any = None


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: float = 60.0
    retry_count: int = 3
    retry_delay: float = 1.0


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """发送聊天请求"""
        pass
    
    @abstractmethod
    async def complete(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """发送补全请求"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化OpenAI客户端"""
        try:
            import openai
            
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key 未设置")
                return
            
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout
            )
            logger.info("OpenAI 客户端初始化完成")
            
        except ImportError:
            logger.error("openai 库未安装")
        except Exception as e:
            logger.error(f"OpenAI 客户端初始化失败: {e}")
    
    async def chat(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """发送聊天请求"""
        if not self.client:
            raise RuntimeError("OpenAI 客户端未初始化")
        
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=formatted_messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            **{k: v for k, v in kwargs.items() 
               if k not in ["model", "max_tokens", "temperature"]}
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_response=response
        )
    
    async def complete(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """发送补全请求"""
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)


class AnthropicClient(BaseLLMClient):
    """Anthropic客户端"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化Anthropic客户端"""
        try:
            import anthropic
            
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("Anthropic API key 未设置")
                return
            
            self.client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=self.config.timeout
            )
            logger.info("Anthropic 客户端初始化完成")
            
        except ImportError:
            logger.error("anthropic 库未安装")
        except Exception as e:
            logger.error(f"Anthropic 客户端初始化失败: {e}")
    
    async def chat(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """发送聊天请求"""
        if not self.client:
            raise RuntimeError("Anthropic 客户端未初始化")
        
        # 分离系统消息
        system_message = ""
        chat_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        response = await self.client.messages.create(
            model=kwargs.get("model", self.config.model),
            system=system_message,
            messages=chat_messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature)
        )
        
        return LLMResponse(
            content=response.content[0].text,
            finish_reason=response.stop_reason,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            },
            raw_response=response
        )
    
    async def complete(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """发送补全请求"""
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)


class LocalLLMClient(BaseLLMClient):
    """本地LLM客户端 (兼容OpenAI API的本地服务)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化本地LLM客户端"""
        try:
            import httpx
            
            base_url = self.config.api_base or "http://localhost:8000/v1"
            
            self.client = httpx.AsyncClient(
                base_url=base_url,
                timeout=self.config.timeout
            )
            logger.info(f"本地LLM客户端初始化完成: {base_url}")
            
        except ImportError:
            logger.error("httpx 库未安装")
        except Exception as e:
            logger.error(f"本地LLM客户端初始化失败: {e}")
    
    async def chat(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """发送聊天请求"""
        if not self.client:
            raise RuntimeError("本地LLM客户端未初始化")
        
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": kwargs.get("model", self.config.model),
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature)
            }
        )
        
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            model=data.get("model", self.config.model),
            usage=data.get("usage", {}),
            raw_response=data
        )
    
    async def complete(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """发送补全请求"""
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)


class LLMInterface:
    """
    LLM接口
    
    统一的LLM访问接口，支持多种提供商
    """
    
    CLIENT_CLASSES = {
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.LOCAL: LocalLLMClient,
        LLMProvider.AZURE: OpenAIClient,  # Azure使用相同的接口
        LLMProvider.CUSTOM: LocalLLMClient,
        # Ollama 在下面单独处理
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # 解析配置
        provider_str = config.get("provider", "openai")
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            logger.warning(f"未知的LLM提供商: {provider_str}, 使用openai")
            provider = LLMProvider.OPENAI
        
        # 获取API key
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        api_key = config.get("api_key") or os.getenv(api_key_env)
        
        self.llm_config = LLMConfig(
            provider=provider,
            model=config.get("model", "gpt-4"),
            api_key=api_key,
            api_base=config.get("api_base"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.1),
            timeout=config.get("timeout", 60.0),
            retry_count=config.get("retry_count", 3),
            retry_delay=config.get("retry_delay", 1.0)
        )
        
        # 初始化客户端
        if provider == LLMProvider.OLLAMA:
            # Ollama 单独处理
            from brain.models.ollama_client import OllamaClient
            self.client = OllamaClient(self.llm_config)
        else:
            client_class = self.CLIENT_CLASSES.get(provider, OpenAIClient)
            self.client = client_class(self.llm_config)
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0
        }
        
        logger.info(f"LLMInterface 初始化完成, 提供商: {provider.value}, 模型: {self.llm_config.model}")
    
    async def chat(
        self,
        messages: Union[List[LLMMessage], List[Dict[str, str]]],
        **kwargs
    ) -> LLMResponse:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 响应
        """
        # 转换消息格式
        if messages and isinstance(messages[0], dict):
            messages = [
                LLMMessage(role=m["role"], content=m["content"])
                for m in messages
            ]
        
        return await self._request_with_retry(
            self.client.chat,
            messages,
            **kwargs
        )
    
    async def complete(
        self,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        发送补全请求
        
        Args:
            prompt: 提示词
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 响应
        """
        return await self._request_with_retry(
            self.client.complete,
            prompt,
            **kwargs
        )
    
    async def _request_with_retry(
        self,
        func,
        *args,
        **kwargs
    ) -> LLMResponse:
        """带重试的请求"""
        self.stats["total_requests"] += 1
        
        last_error = None
        
        for attempt in range(self.llm_config.retry_count):
            try:
                response = await func(*args, **kwargs)
                
                self.stats["successful_requests"] += 1
                self.stats["total_tokens"] += response.usage.get("total_tokens", 0)
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"LLM请求失败 (尝试 {attempt + 1}/{self.llm_config.retry_count}): {e}")
                
                if attempt < self.llm_config.retry_count - 1:
                    await asyncio.sleep(self.llm_config.retry_delay * (attempt + 1))
        
        self.stats["failed_requests"] += 1
        raise RuntimeError(f"LLM请求失败，已重试{self.llm_config.retry_count}次: {last_error}")
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成JSON格式的响应
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            schema: JSON Schema (可选)
            
        Returns:
            Dict: 解析后的JSON
        """
        messages = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        # 添加JSON格式要求
        json_instruction = "\n请以JSON格式回复，不要包含其他文字。"
        if schema:
            json_instruction += f"\nJSON Schema: {json.dumps(schema, ensure_ascii=False)}"
        
        messages.append(LLMMessage(
            role="user",
            content=prompt + json_instruction
        ))
        
        response = await self.chat(messages)
        
        # 解析JSON
        try:
            # 尝试直接解析
            return json.loads(response.content)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            
            raise ValueError(f"无法解析JSON响应: {content}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            )
        }

