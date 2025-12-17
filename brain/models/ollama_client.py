"""
Ollama 客户端 - 本地大模型支持

支持通过Ollama运行本地大模型，如：
- deepseek-r1
- llama3
- qwen
- mistral
等
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import httpx
from loguru import logger

from brain.models.llm_interface import BaseLLMClient, LLMConfig, LLMMessage, LLMResponse


class OllamaClient(BaseLLMClient):
    """
    Ollama 客户端
    
    通过 Ollama API 调用本地大模型
    Ollama API 文档: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    
    DEFAULT_BASE_URL = "http://localhost:11434"
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_base or self.DEFAULT_BASE_URL
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化HTTP客户端"""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.config.timeout,
                    write=30.0,
                    pool=10.0
                )
            )
            logger.info(f"Ollama 客户端初始化完成: {self.base_url}, 模型: {self.config.model}")
            
        except Exception as e:
            logger.error(f"Ollama 客户端初始化失败: {e}")
    
    async def chat(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        发送聊天请求
        
        使用 Ollama 的 /api/chat 接口
        """
        if not self.client:
            raise RuntimeError("Ollama 客户端未初始化")
        
        # 格式化消息
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # 构建请求
        request_data = {
            "model": kwargs.get("model", self.config.model),
            "messages": formatted_messages,
            "stream": False,  # 不使用流式响应
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        logger.debug(f"Ollama 请求: model={request_data['model']}")
        
        try:
            response = await self.client.post(
                "/api/chat",
                json=request_data
            )
            response.raise_for_status()
            
            data = response.json()
            
            # 解析响应
            message = data.get("message", {})
            content = message.get("content", "")
            
            # 计算token使用（Ollama返回的信息）
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
            
            return LLMResponse(
                content=content,
                finish_reason=data.get("done_reason", "stop"),
                model=data.get("model", self.config.model),
                usage=usage,
                raw_response=data
            )
            
        except httpx.TimeoutException:
            raise RuntimeError(f"Ollama 请求超时 (timeout={self.config.timeout}s)")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama HTTP错误: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Ollama 请求失败: {e}")
    
    async def complete(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """
        发送补全请求
        
        使用 Ollama 的 /api/generate 接口
        """
        if not self.client:
            raise RuntimeError("Ollama 客户端未初始化")
        
        request_data = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }
        
        try:
            response = await self.client.post(
                "/api/generate",
                json=request_data
            )
            response.raise_for_status()
            
            data = response.json()
            
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            }
            
            return LLMResponse(
                content=data.get("response", ""),
                finish_reason="stop" if data.get("done", False) else "length",
                model=data.get("model", self.config.model),
                usage=usage,
                raw_response=data
            )
            
        except Exception as e:
            raise RuntimeError(f"Ollama 补全请求失败: {e}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """列出可用的模型"""
        if not self.client:
            return []
        
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    async def check_model_available(self, model_name: str) -> bool:
        """检查模型是否可用"""
        models = await self.list_models()
        available_names = [m.get("name", "") for m in models]
        
        # 支持模糊匹配
        for name in available_names:
            if model_name in name or name in model_name:
                return True
        return False
    
    async def pull_model(self, model_name: str) -> bool:
        """拉取模型（如果不存在）"""
        if not self.client:
            return False
        
        try:
            logger.info(f"正在拉取模型: {model_name}")
            response = await self.client.post(
                "/api/pull",
                json={"name": model_name, "stream": False},
                timeout=httpx.Timeout(None)  # 无超时
            )
            response.raise_for_status()
            logger.info(f"模型拉取完成: {model_name}")
            return True
        except Exception as e:
            logger.error(f"模型拉取失败: {e}")
            return False
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()
            self.client = None


# 便捷函数：测试Ollama连接
async def test_ollama_connection(
    base_url: str = "http://localhost:11434",
    model: str = "deepseek-r1:latest"
) -> bool:
    """
    测试Ollama连接
    
    Returns:
        bool: 连接是否成功
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            # 测试连接
            response = await client.get("/api/tags")
            response.raise_for_status()
            
            models = response.json().get("models", [])
            logger.info(f"Ollama 连接成功, 可用模型: {[m.get('name') for m in models]}")
            
            # 检查目标模型
            available = any(model in m.get("name", "") for m in models)
            if available:
                logger.info(f"模型 {model} 可用")
            else:
                logger.warning(f"模型 {model} 不可用，请先运行: ollama pull {model}")
            
            return True
            
    except httpx.ConnectError:
        logger.error(f"无法连接到 Ollama ({base_url})，请确保 Ollama 正在运行")
        return False
    except Exception as e:
        logger.error(f"Ollama 连接测试失败: {e}")
        return False


