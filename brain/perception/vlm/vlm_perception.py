"""
视觉语言模型感知 - VLM Perception

负责:
- 使用VLM（如LLaVA）进行场景理解
- 根据自然语言描述查找目标
- 回答空间问题
- 可选的YOLO快速检测集成

支持的VLM后端:
1. Ollama本地模型（LLaVA, MiniCPM-V等）
2. API调用（可扩展）
"""

import asyncio
import base64
import io
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL不可用，图像处理功能受限")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama不可用，将使用模拟模式")


class DetectionSource(Enum):
    """检测来源"""
    VLM = "vlm"
    YOLO = "yolo"
    FUSED = "fused"


@dataclass
class BoundingBox:
    """边界框"""
    x: float  # 中心x (0-1 归一化)
    y: float  # 中心y (0-1 归一化)
    width: float  # 宽度 (0-1 归一化)
    height: float  # 高度 (0-1 归一化)
    
    def to_pixel(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """转换为像素坐标 (x1, y1, x2, y2)"""
        x1 = int((self.x - self.width / 2) * img_width)
        y1 = int((self.y - self.height / 2) * img_height)
        x2 = int((self.x + self.width / 2) * img_width)
        y2 = int((self.y + self.height / 2) * img_height)
        return (x1, y1, x2, y2)
    
    def center_pixel(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """获取中心像素坐标"""
        return (int(self.x * img_width), int(self.y * img_height))


@dataclass
class DetectedObject:
    """检测到的物体"""
    id: str
    label: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: DetectionSource = DetectionSource.VLM
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 空间信息（相对于图像）
    position_description: str = ""  # 如 "左上角", "中央", "右下"
    estimated_distance: Optional[float] = None  # 估计距离（米）
    estimated_direction: Optional[str] = None  # 方向描述
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height
            } if self.bbox else None,
            "description": self.description,
            "position": self.position_description,
            "distance": self.estimated_distance,
            "direction": self.estimated_direction
        }


@dataclass
class SceneDescription:
    """场景描述"""
    summary: str
    objects: List[DetectedObject]
    spatial_relations: List[str]  # 物体间的空间关系描述
    navigation_hints: List[str]   # 导航提示
    potential_targets: List[str]  # 可能的目标
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "objects": [obj.to_dict() for obj in self.objects],
            "spatial_relations": self.spatial_relations,
            "navigation_hints": self.navigation_hints,
            "potential_targets": self.potential_targets
        }


@dataclass
class TargetSearchResult:
    """目标搜索结果"""
    found: bool
    target_description: str
    matched_objects: List[DetectedObject]
    best_match: Optional[DetectedObject] = None
    suggested_action: str = ""  # 如 "向左转", "前进"
    confidence: float = 0.0
    explanation: str = ""


class VLMPerception:
    """
    视觉语言模型感知
    
    使用VLM进行开放词汇的场景理解和目标检测
    """
    
    # 场景分析提示词
    SCENE_ANALYSIS_PROMPT = """分析这张图像中的场景。请识别并描述：

1. **场景概述**：这是什么样的环境？（室内/室外、城市/乡村等）

2. **可见物体**：列出所有可见的重要物体，包括：
   - 建筑物（类型、特征）
   - 门、窗户、入口
   - 道路、路径
   - 障碍物
   - 人、车辆
   - 其他显著物体

3. **空间信息**：描述物体的大致位置（左/中/右，近/远）

4. **导航信息**：如果要移动，有哪些可通行的方向？

请用JSON格式回复：
```json
{
    "summary": "场景简述",
    "objects": [
        {"label": "物体名", "position": "位置描述", "description": "详细描述"}
    ],
    "spatial_relations": ["物体A在物体B的左边"],
    "navigation_hints": ["前方可通行", "左侧有障碍"]
}
```"""

    TARGET_SEARCH_PROMPT = """在这张图像中寻找：{target}

请仔细分析图像，判断：
1. 目标是否可见？如果可见，在图像的什么位置？
2. 如果目标不完全可见，是否有相关的线索？
3. 建议的行动方向是什么？

用JSON格式回复：
```json
{{
    "found": true/false,
    "target_position": "位置描述（如：图像中央、右侧远处）",
    "confidence": 0.0-1.0,
    "suggested_action": "建议动作（如：向右转、前进）",
    "explanation": "解释"
}}
```"""

    SPATIAL_QUERY_PROMPT = """关于这张图像，请回答：{query}

请基于图像内容给出准确的回答。如果涉及方向或位置，使用机器人视角的描述（左/右/前/后）。"""

    def __init__(
        self,
        model: str = "llava:latest",
        ollama_host: str = "http://localhost:11434",
        use_yolo: bool = False,
        yolo_model: str = "yolov8n.pt"
    ):
        """
        Args:
            model: Ollama模型名称
            ollama_host: Ollama服务地址
            use_yolo: 是否使用YOLO进行快速检测
            yolo_model: YOLO模型路径
        """
        self.model = model
        self.ollama_host = ollama_host
        self.use_yolo = use_yolo
        
        # 初始化Ollama客户端
        if OLLAMA_AVAILABLE:
            try:
                self.ollama_client = ollama.Client(host=ollama_host)
                logger.info(f"VLM初始化: 使用Ollama模型 {model}")
            except Exception as e:
                logger.warning(f"Ollama客户端初始化失败: {e}")
                self.ollama_client = None
        else:
            self.ollama_client = None
        
        # 初始化YOLO（可选）
        self.yolo_detector = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_detector = YOLO(yolo_model)
                logger.info(f"YOLO初始化: {yolo_model}")
            except Exception as e:
                logger.warning(f"YOLO初始化失败: {e}")
        
        # 缓存
        self._last_scene: Optional[SceneDescription] = None
        self._detection_history: List[DetectedObject] = []
        
        logger.info("VLMPerception 初始化完成")
    
    async def describe_scene(self, image: np.ndarray) -> SceneDescription:
        """
        描述当前场景
        
        Args:
            image: RGB图像 (H, W, 3)
            
        Returns:
            SceneDescription: 场景描述
        """
        logger.debug("开始场景分析...")
        
        # 编码图像
        image_base64 = self._encode_image(image)
        
        # 调用VLM
        response = await self._call_vlm(
            prompt=self.SCENE_ANALYSIS_PROMPT,
            image=image_base64
        )
        
        # 解析响应
        scene = self._parse_scene_response(response)
        
        # 如果启用YOLO，补充检测结果
        if self.use_yolo and self.yolo_detector:
            yolo_objects = await self._yolo_detect(image)
            scene = self._fuse_detections(scene, yolo_objects)
        
        self._last_scene = scene
        logger.debug(f"场景分析完成: {len(scene.objects)} 个物体")
        
        return scene
    
    async def find_target(
        self,
        image: np.ndarray,
        target_description: str
    ) -> TargetSearchResult:
        """
        根据描述查找目标
        
        Args:
            image: RGB图像
            target_description: 目标描述（如"建筑的门"）
            
        Returns:
            TargetSearchResult: 搜索结果
        """
        logger.debug(f"搜索目标: {target_description}")
        
        # 编码图像
        image_base64 = self._encode_image(image)
        
        # 构建提示词
        prompt = self.TARGET_SEARCH_PROMPT.format(target=target_description)
        
        # 调用VLM
        response = await self._call_vlm(prompt=prompt, image=image_base64)
        
        # 解析响应
        result = self._parse_target_response(response, target_description)
        
        logger.debug(f"目标搜索结果: found={result.found}, confidence={result.confidence}")
        
        return result
    
    async def answer_spatial_query(
        self,
        image: np.ndarray,
        query: str
    ) -> str:
        """
        回答空间问题
        
        Args:
            image: RGB图像
            query: 空间问题（如"门在哪个方向"）
            
        Returns:
            str: 回答
        """
        logger.debug(f"空间查询: {query}")
        
        image_base64 = self._encode_image(image)
        prompt = self.SPATIAL_QUERY_PROMPT.format(query=query)
        
        response = await self._call_vlm(prompt=prompt, image=image_base64)
        
        return response
    
    async def _call_vlm(self, prompt: str, image: str) -> str:
        """调用VLM"""
        if self.ollama_client:
            try:
                response = self.ollama_client.generate(
                    model=self.model,
                    prompt=prompt,
                    images=[image],
                    stream=False
                )
                return response.get("response", "")
            except Exception as e:
                logger.error(f"VLM调用失败: {e}")
                return self._generate_mock_response(prompt)
        else:
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """生成模拟响应（用于测试）"""
        prompt_lower = prompt.lower()
        
        # 检查是否为目标搜索请求（优先级最高）
        # 特征：prompt以"在这张图像中寻找"开头
        if prompt.startswith("在这张图像中寻找"):
            return """```json
{
    "found": true,
    "target_position": "前方中央偏右，距离约20米",
    "confidence": 0.85,
    "suggested_action": "直行前进约15米，然后稍微右转",
    "explanation": "目标门位于建筑物正面，当前可以看到，建议直接前进"
}
```"""
        # 检查是否为场景分析请求
        # 特征：prompt以"分析这张图像"开头
        elif prompt.startswith("分析这张图像") or "场景概述" in prompt:
            return """```json
{
    "summary": "这是一个室外城市环境，前方可见一栋建筑物",
    "objects": [
        {"label": "建筑", "position": "前方中央", "description": "一栋多层建筑，有玻璃门"},
        {"label": "门", "position": "建筑正面", "description": "玻璃自动门，位于建筑入口"},
        {"label": "道路", "position": "下方", "description": "平坦的人行道"},
        {"label": "树木", "position": "右侧", "description": "几棵绿树"}
    ],
    "spatial_relations": ["门位于建筑正面", "树木在建筑右侧"],
    "navigation_hints": ["前方道路畅通", "可直接前往建筑入口"]
}
```"""
        else:
            return "根据图像分析，前方约10米处有一个入口。"
    
    def _encode_image(self, image: np.ndarray) -> str:
        """将图像编码为base64"""
        if not PIL_AVAILABLE:
            return ""
        
        # 确保图像是uint8类型
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image)
        
        # 编码为base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def _parse_scene_response(self, response: str) -> SceneDescription:
        """解析场景分析响应"""
        objects = []
        spatial_relations = []
        navigation_hints = []
        summary = "场景分析完成"
        
        try:
            # 提取JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = {}
            
            summary = data.get("summary", summary)
            spatial_relations = data.get("spatial_relations", [])
            navigation_hints = data.get("navigation_hints", [])
            
            # 解析物体
            for i, obj_data in enumerate(data.get("objects", [])):
                position_desc = obj_data.get("position", "")
                
                # 根据位置描述推断边界框
                bbox = self._estimate_bbox_from_position(position_desc)
                
                detected_obj = DetectedObject(
                    id=f"vlm_{i}",
                    label=obj_data.get("label", "unknown"),
                    confidence=0.7,
                    bbox=bbox,
                    description=obj_data.get("description", ""),
                    position_description=position_desc,
                    source=DetectionSource.VLM
                )
                objects.append(detected_obj)
                
        except Exception as e:
            logger.warning(f"解析场景响应失败: {e}")
        
        # 提取潜在目标
        potential_targets = [
            obj.label for obj in objects
            if obj.label in ["门", "入口", "建筑", "目标", "door", "entrance", "building"]
        ]
        
        return SceneDescription(
            summary=summary,
            objects=objects,
            spatial_relations=spatial_relations,
            navigation_hints=navigation_hints,
            potential_targets=potential_targets,
            raw_response=response
        )
    
    def _estimate_bbox_from_position(self, position: str) -> Optional[BoundingBox]:
        """根据位置描述估算边界框"""
        # 默认中心
        x, y = 0.5, 0.5
        width, height = 0.2, 0.2
        
        position_lower = position.lower()
        
        # 水平位置
        if "左" in position or "left" in position_lower:
            x = 0.25
        elif "右" in position or "right" in position_lower:
            x = 0.75
        
        # 垂直位置
        if "上" in position or "top" in position_lower:
            y = 0.25
        elif "下" in position or "bottom" in position_lower:
            y = 0.75
        
        # 距离
        if "远" in position or "far" in position_lower:
            width, height = 0.1, 0.1
            y = 0.3  # 远处物体通常在上方
        elif "近" in position or "close" in position_lower:
            width, height = 0.3, 0.3
            y = 0.7  # 近处物体通常在下方
        
        return BoundingBox(x=x, y=y, width=width, height=height)
    
    def _parse_target_response(
        self,
        response: str,
        target_description: str
    ) -> TargetSearchResult:
        """解析目标搜索响应"""
        found = False
        confidence = 0.0
        suggested_action = ""
        explanation = ""
        matched_objects = []
        
        try:
            data = {}
            
            # 方法1: 尝试提取```json代码块
            json_match = re.search(r'```json\s*\n?(.*?)\n?```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # 方法2: 尝试直接找JSON对象（花括号匹配）
            if not data:
                # 找到第一个{到最后一个}
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = response[start:end+1]
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            found = data.get("found", False)
            confidence = float(data.get("confidence", 0.0))
            suggested_action = data.get("suggested_action", "")
            explanation = data.get("explanation", "")
            
            if found:
                position_desc = data.get("target_position", "")
                bbox = self._estimate_bbox_from_position(position_desc)
                
                detected = DetectedObject(
                    id="target_0",
                    label=target_description,
                    confidence=confidence,
                    bbox=bbox,
                    description=explanation,
                    position_description=position_desc,
                    source=DetectionSource.VLM
                )
                matched_objects.append(detected)
                
        except Exception as e:
            logger.warning(f"解析目标响应失败: {e}")
        
        return TargetSearchResult(
            found=found,
            target_description=target_description,
            matched_objects=matched_objects,
            best_match=matched_objects[0] if matched_objects else None,
            suggested_action=suggested_action,
            confidence=confidence,
            explanation=explanation
        )
    
    async def _yolo_detect(self, image: np.ndarray) -> List[DetectedObject]:
        """使用YOLO检测"""
        if not self.yolo_detector:
            return []
        
        objects = []
        
        try:
            results = self.yolo_detector(image, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x, y, w, h = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    
                    detected = DetectedObject(
                        id=f"yolo_{i}",
                        label=label,
                        confidence=conf,
                        bbox=BoundingBox(x=x, y=y, width=w, height=h),
                        source=DetectionSource.YOLO
                    )
                    objects.append(detected)
                    
        except Exception as e:
            logger.warning(f"YOLO检测失败: {e}")
        
        return objects
    
    def _fuse_detections(
        self,
        scene: SceneDescription,
        yolo_objects: List[DetectedObject]
    ) -> SceneDescription:
        """融合VLM和YOLO检测结果"""
        fused_objects = list(scene.objects)
        
        for yolo_obj in yolo_objects:
            # 检查是否与VLM检测重复
            is_duplicate = False
            for vlm_obj in scene.objects:
                if self._is_same_object(vlm_obj, yolo_obj):
                    # 更新置信度
                    vlm_obj.confidence = max(vlm_obj.confidence, yolo_obj.confidence)
                    vlm_obj.bbox = yolo_obj.bbox  # YOLO边界框更准确
                    vlm_obj.source = DetectionSource.FUSED
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                fused_objects.append(yolo_obj)
        
        scene.objects = fused_objects
        return scene
    
    def _is_same_object(
        self,
        obj1: DetectedObject,
        obj2: DetectedObject,
        iou_threshold: float = 0.3
    ) -> bool:
        """判断两个检测是否为同一物体"""
        # 标签匹配
        label_match = obj1.label.lower() in obj2.label.lower() or \
                      obj2.label.lower() in obj1.label.lower()
        
        if not label_match:
            return False
        
        # 边界框IoU匹配
        if obj1.bbox and obj2.bbox:
            iou = self._compute_iou(obj1.bbox, obj2.bbox)
            return iou > iou_threshold
        
        return label_match
    
    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """计算IoU"""
        x1_min = box1.x - box1.width / 2
        x1_max = box1.x + box1.width / 2
        y1_min = box1.y - box1.height / 2
        y1_max = box1.y + box1.height / 2
        
        x2_min = box2.x - box2.width / 2
        x2_max = box2.x + box2.width / 2
        y2_min = box2.y - box2.height / 2
        y2_max = box2.y + box2.height / 2
        
        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        area1 = box1.width * box1.height
        area2 = box2.width * box2.height
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def get_last_scene(self) -> Optional[SceneDescription]:
        """获取最近的场景描述"""
        return self._last_scene
    
    def get_detection_history(self, count: int = 10) -> List[DetectedObject]:
        """获取检测历史"""
        return self._detection_history[-count:]

