#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock VLM 客户端 - 用于演示语义地图融合
不依赖真实的VLM模型，而是返回模拟的语义物体检测结果
"""

import random
import time
from datetime import datetime
from typing import Dict, Any, List, Optional


class MockVLMClient:
    """模拟VLM客户端 - 用于演示"""

    def __init__(self, model: str = "mock-llava"):
        self.model = model
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # 模拟的物体库
        self.mock_objects = [
            {"name": "door", "position": "left", "size": "large"},
            {"name": "person", "position": "center", "size": "medium"},
            {"name": "building", "position": "right", "size": "large"},
            {"name": "car", "position": "left", "size": "medium"},
            {"name": "obstacle", "position": "center", "size": "small"},
            {"name": "table", "position": "right", "size": "medium"},
            {"name": "chair", "position": "center", "size": "small"},
            {"name": "wall", "position": "left", "size": "large"},
        ]

    def _encode_image_from_ros(self, ros_image) -> str:
        """Mock编码 - 不做任何事"""
        return "mock_image_data"

    def analyze_image(
        self,
        image_data: str,
        prompt: Optional[str] = None,
        robot_pose: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        模拟图像分析 - 返回随机语义物体

        Args:
            image_data: 图像数据（忽略）
            prompt: 提示词（忽略）
            robot_pose: 机器人位姿
            timestamp: 时间戳

        Returns:
            模拟的语义分析结果
        """
        self.total_requests += 1
        start_time = time.time()

        # 模拟处理延迟（1-2秒）
        time.sleep(0.5)

        # 随机选择2-5个物体
        num_objects = random.randint(2, 5)
        detected_objects = random.sample(self.mock_objects, num_objects)

        # 添加一些随机变化
        for obj in detected_objects:
            obj["confidence"] = round(random.uniform(0.6, 0.95), 2)

        processing_time = time.time() - start_time

        result = {
            "objects": detected_objects,
            "description": f"Mock VLM detected {num_objects} objects: " +
                          ", ".join([f"{obj['name']}({obj['position']})" for obj in detected_objects]),
            "confidence": 0.8,
            "timestamp": timestamp or datetime.now(),
            "robot_pose": robot_pose or {},
            "processing_time": processing_time,
            "model": self.model
        }

        self.successful_requests += 1
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests)
        }


def get_vlm_client(use_mock: bool = True) -> Any:
    """
    获取VLM客户端

    Args:
        use_mock: 是否使用Mock客户端（默认True）

    Returns:
        VLM客户端实例
    """
    if use_mock:
        return MockVLMClient()
    else:
        # 导入真实的Ollama客户端
        from .ollama_client import OllamaVLMClient
        return OllamaVLMClient()
