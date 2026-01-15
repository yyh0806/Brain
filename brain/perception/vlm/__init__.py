#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM (Vision Language Model) 感知模块
"""

from typing import Dict, Any, Optional

def get_vlm_client(use_mock: bool = False, config: Dict = None) -> Any:
    """
    获取VLM客户端

    Args:
        use_mock: 是否使用Mock客户端（默认False）
        config: 配置字典

    Returns:
        VLM客户端实例
    """
    if use_mock:
        from .mock_client import MockVLMClient
        return MockVLMClient()
    else:
        from .ollama_client import OllamaVLMClient
        return OllamaVLMClient(
            model=config.get("model", "llava:7b") if config else "llava:7b",
            base_url=config.get("ollama_host", "http://localhost:11434") if config else "http://localhost:11434"
        )

# Re-export for backwards compatibility
from .ollama_client import OllamaVLMClient
from .mock_client import MockVLMClient

__all__ = [
    'OllamaVLMClient',
    'MockVLMClient',
    'get_vlm_client'
]
