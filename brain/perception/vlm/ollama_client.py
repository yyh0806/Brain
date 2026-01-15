#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama VLM 客户端
支持视觉语言模型 (LLaVA) 进行图像语义理解
"""

import json
import base64
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from io import BytesIO


class OllamaVLMClient:
    """Ollama VLM 客户端"""

    def __init__(self, model: str = "llava:7b", base_url: str = "http://localhost:11434"):
        """
        初始化 Ollama VLM 客户端

        Args:
            model: 模型名称 (默认: llava:7b)
            base_url: Ollama API 地址 (默认: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timeout = 30  # 30秒超时

        # 用于统计的变量
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    def _encode_image(self, image_array: np.ndarray) -> str:
        """
        将 numpy 图像数组编码为 base64

        Args:
            image_array: RGB 图像数组 (H, W, 3)

        Returns:
            base64 编码的字符串
        """
        # 如果图像是 uint8，直接编码
        if image_array.dtype == np.uint8:
            # 将 numpy 数组转换为 bytes
            success, buffer = cv2.imencode('.jpg', image_array)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            # 转换为 uint8
            image_uint8 = (image_array * 255).astype(np.uint8)
            success, buffer = cv2.imencode('.jpg', image_uint8)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(buffer.tobytes()).decode('utf-8')

    def _encode_image_from_ros(self, ros_image) -> str:
        """
        从 ROS Image 消息编码图像为JPEG格式

        Args:
            ros_image: ROS2 Sensor Image 消息

        Returns:
            base64 编码的JPEG图像字符串
        """
        try:
            from PIL import Image

            # 将ROS图像数据转换为numpy数组
            if ros_image.encoding in ['rgb8', 'bgr8']:
                # RGB/BGR图像
                arr = np.frombuffer(ros_image.data, dtype=np.uint8)
                height, width = ros_image.height, ros_image.width

                if len(arr) == height * width * 3:
                    arr = arr.reshape((height, width, 3))
                else:
                    raise ValueError(f"Image size mismatch: expected {height*width*3}, got {len(arr)}")

                # 转换BGR到RGB
                if ros_image.encoding == 'bgr8':
                    arr = arr[:, :, ::-1].copy()

                # 创建PIL Image
                pil_image = Image.fromarray(arr, 'RGB')

            elif ros_image.encoding == 'mono8':
                # 灰度图像
                arr = np.frombuffer(ros_image.data, dtype=np.uint8)
                height, width = ros_image.height, ros_image.width
                arr = arr.reshape((height, width))
                pil_image = Image.fromarray(arr, 'L')

                # 转换为RGB
                pil_image = pil_image.convert('RGB')

            else:
                raise ValueError(f"Unsupported encoding: {ros_image.encoding}")

            # 编码为JPEG
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            jpeg_bytes = buffer.getvalue()

            # Base64编码
            return base64.b64encode(jpeg_bytes).decode('utf-8')

        except ImportError:
            # 如果没有PIL，尝试使用cv2
            try:
                import cv2

                # 将ROS图像转换为numpy数组
                arr = np.frombuffer(ros_image.data, dtype=np.uint8)

                if ros_image.encoding == 'bgr8':
                    arr = arr.reshape((ros_image.height, ros_image.width, 3))
                elif ros_image.encoding == 'rgb8':
                    arr = arr.reshape((ros_image.height, ros_image.width, 3))
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif ros_image.encoding == 'mono8':
                    arr = arr.reshape((ros_image.height, ros_image.width))
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    raise ValueError(f"Unsupported encoding: {ros_image.encoding}")

                # 编码为JPEG
                success, jpeg_bytes = cv2.imencode('.jpg', arr)
                if not success:
                    raise ValueError("Failed to encode image with cv2")

                return base64.b64encode(jpeg_bytes.tobytes()).decode('utf-8')

            except ImportError:
                # 如果PIL和cv2都不可用，使用原始方法（会失败但至少尝试）
                return base64.b64encode(ros_image.data).decode('utf-8')

    def _create_semantic_prompt(self, task: str = "describe") -> str:
        """
        创建语义理解提示词

        Args:
            task: 任务类型 ("describe", "objects", "navigate")

        Returns:
            提示词字符串
        """
        prompts = {
            "describe": (
                "Describe this image in detail. Focus on identifying objects, "
                "their spatial relationships, and any relevant semantic information."
            ),
            "objects": (
                "List all visible objects in this image. For each object, provide: "
                "1) Object name (e.g., door, person, building, car), "
                "2) Approximate position (left, center, right), "
                "3) Size estimate. "
                "Format as JSON: {'objects': [{'name': 'door', 'position': 'left', 'size': 'large'}]}"
            ),
            "navigate": (
                "Analyze this image for navigation purposes. Identify: "
                "1) Obstacles, 2) Free space, 3) Doors/entrances, 4) Paths. "
                "Format as JSON."
            )
        }
        return prompts.get(task, prompts["describe"])

    def analyze_image(
        self,
        image_data: str,
        prompt: Optional[str] = None,
        robot_pose: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        分析图像并返回语义理解结果

        Args:
            image_data: base64 编码的图像数据
            prompt: 自定义提示词 (可选)
            robot_pose: 机器人位姿 {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
            timestamp: 图像时间戳

        Returns:
            语义分析结果字典
        """
        self.total_requests += 1
        start_time = time.time()

        # 使用默认提示词或自定义提示词
        if prompt is None:
            prompt = self._create_semantic_prompt("objects")

        try:
            # 构建 Ollama API 请求
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # 较低温度以获得更确定的结果
                    "num_predict": 500
                }
            }

            # 发送请求
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            response_text = result.get("response", "")

            # 尝试解析 JSON 响应
            semantic_data = self._parse_vlm_response(response_text)

            # 添加元数据
            processing_time = time.time() - start_time
            semantic_data.update({
                "timestamp": timestamp or datetime.now(),
                "robot_pose": robot_pose or {},
                "processing_time": processing_time,
                "model": self.model,
                "raw_response": response_text
            })

            self.successful_requests += 1
            return semantic_data

        except requests.exceptions.Timeout:
            self.failed_requests += 1
            return {
                "error": "VLM request timeout",
                "timestamp": timestamp or datetime.now(),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            self.failed_requests += 1
            return {
                "error": str(e),
                "timestamp": timestamp or datetime.now(),
                "processing_time": time.time() - start_time
            }

    def _parse_vlm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析 VLM 响应文本，提取结构化语义信息

        Args:
            response_text: VLM 返回的原始文本

        Returns:
            结构化的语义数据
        """
        # 尝试提取 JSON
        objects = []
        try:
            # 查找 JSON 块
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                objects = parsed.get("objects", [])
        except:
            pass

        # 如果没有找到 JSON，尝试解析文本
        if not objects:
            objects = self._extract_objects_from_text(response_text)

        return {
            "objects": objects,
            "description": response_text,
            "confidence": 0.7  # 默认置信度
        }

    def _extract_objects_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取物体信息 (启发式方法)

        Args:
            text: VLM 描述文本

        Returns:
            物体列表
        """
        objects = []
        # 常见物体类别
        common_objects = [
            "door", "person", "building", "car", "chair", "table",
            "wall", "window", "floor", "ceiling", "obstacle",
            "门", "人", "建筑", "车", "椅子", "桌子", "墙", "窗户", "地板", "天花板", "障碍物"
        ]

        text_lower = text.lower()
        for obj_name in common_objects:
            if obj_name.lower() in text_lower:
                objects.append({
                    "name": obj_name,
                    "position": "unknown",  # VLM 可能提供位置信息
                    "size": "unknown",
                    "confidence": 0.6
                })

        return objects

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests)
        }


# 单例实例
_vlm_client: Optional[OllamaVLMClient] = None


def get_vlm_client() -> OllamaVLMClient:
    """获取 VLM 客户端单例"""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = OllamaVLMClient()
    return _vlm_client


# 导入 opencv (用于图像编码)
try:
    import cv2
except ImportError:
    cv2 = None
