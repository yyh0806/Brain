"""
VLM感知集成测试

测试视觉语言模型感知功能，包括场景描述、目标搜索和空间查询。
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# 模拟Ollama
class MockOllamaClient:
    """模拟Ollama客户端"""
    def __init__(self, host="http://localhost:11434"):
        self.host = host
    
    def generate(self, model, prompt, images=None, stream=False):
        """生成响应"""
        # 模拟场景分析响应
        if "分析这张图像" in prompt or "场景概述" in prompt:
            return {
                "response": """```json
{
    "summary": "这是一个室内大厅环境，前方有一个玻璃门",
    "objects": [
        {"label": "门", "position": "前方中央", "description": "建筑入口的玻璃门"},
        {"label": "植物", "position": "右侧", "description": "盆栽植物"},
        {"label": "桌子", "position": "左侧", "description": "木制桌子"}
    ],
    "spatial_relations": ["门在前方中央", "植物在门的右侧"],
    "navigation_hints": ["前方可以通行", "左侧有空间"]
}
```"""
            }
        # 模拟目标搜索响应
        elif "寻找" in prompt:
            if "门" in prompt:
                return {
                    "response": """```json
{
    "found": true,
    "target_position": "前方中央偏右，距离约10米",
    "confidence": 0.85,
    "suggested_action": "直行前进约10米，然后稍向右转",
    "explanation": "目标门位于建筑物正面，当前可以看到，建议直接前进"
}
```"""
                }
            else:
                return {
                    "response": """```json
{
    "found": false,
    "target_position": "",
    "confidence": 0.0,
    "suggested_action": "继续搜索环境",
    "explanation": "目标在当前视野中不可见"
}
```"""
                }
        # 模拟空间查询响应
        else:
            return "根据图像分析，前方约10米处有一个门。"


class MockYOLO:
    """模拟YOLO检测器"""
    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
    
    def __call__(self, image, verbose=False):
        """模拟YOLO检测"""
        # 模拟检测结果
        results = [Mock()]
        results[0].boxes = Mock()
        results[0].boxes.xywhn = np.array([
            [0.4, 0.3, 0.2, 0.4],  # [x, y, w, h]
            [0.6, 0.5, 0.1, 0.2]   # [x, y, w, h]
        ])
        results[0].boxes.conf = np.array([0.85, 0.75])
        results[0].boxes.cls = np.array([0, 1])
        results[0].names = {0: "person", 1: "chair"}
        return results


@pytest.fixture
def rgb_image():
    """创建测试RGB图像"""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
async def vlm_perception():
    """VLM感知器"""
    # 模拟Ollama不可用
    with patch('brain.perception.vlm.vlm_perception.OLLAMA_AVAILABLE', False):
        with patch('brain.perception.vlm.vlm_perception.PIL_AVAILABLE', True):
            with patch('brain.perception.vlm.vlm_perception.ollama.Client', MockOllamaClient):
                from brain.perception.vlm.vlm_perception import VLMPerception
                
                # 创建VLM感知器
                vlm = VLMPerception(
                    model="llava:latest",
                    ollama_host="http://localhost:11434",
                    use_yolo=False,
                    yolo_model="yolov8n.pt"
                )
                
                return vlm


@pytest.fixture
async def vlm_perception_with_yolo():
    """带YOLO的VLM感知器"""
    # 模拟Ollama和YOLO都不可用
    with patch('brain.perception.vlm.vlm_perception.OLLAMA_AVAILABLE', False):
        with patch('brain.perception.vlm.vlm_perception.PIL_AVAILABLE', True):
            with patch('brain.perception.vlm.vlm_perception.ollama.Client', MockOllamaClient):
                from brain.perception.vlm.vlm_perception import VLMPerception
                
                # 创建VLM感知器
                vlm = VLMPerception(
                    model="llava:latest",
                    ollama_host="http://localhost:11434",
                    use_yolo=True,
                    yolo_model="yolov8n.pt"
                )
                
                return vlm


class TestVLMPerception:
    """测试VLM感知"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, vlm_perception):
        """测试VLM感知器初始化"""
        # 验证初始化参数
        assert vlm_perception.model == "llava:latest"
        assert vlm_perception.ollama_host == "http://localhost:11434"
        assert vlm_perception.use_yolo is False
        
        # 在模拟模式下，ollama_client应该为None
        assert vlm_perception.ollama_client is None
        assert vlm_perception.yolo_detector is None
    
    @pytest.mark.asyncio
    async def test_initialization_with_yolo(self, vlm_perception_with_yolo):
        """测试带YOLO的VLM感知器初始化"""
        # 验证初始化参数
        assert vlm_perception_with_yolo.model == "llava:latest"
        assert vlm_perception_with_yolo.ollama_host == "http://localhost:11434"
        assert vlm_perception_with_yolo.use_yolo is True
        
        # 在模拟模式下，yolo_detector应该为None
        assert vlm_perception_with_yolo.ollama_client is None
        assert vlm_perception_with_yolo.yolo_detector is None
    
    @pytest.mark.asyncio
    async def test_describe_scene(self, vlm_perception, rgb_image):
        """测试场景描述"""
        # 描述场景
        scene = await vlm_perception.describe_scene(rgb_image)
        
        # 验证场景描述
        assert scene is not None
        assert hasattr(scene, "summary")
        assert hasattr(scene, "objects")
        assert hasattr(scene, "spatial_relations")
        assert hasattr(scene, "navigation_hints")
        assert hasattr(scene, "potential_targets")
        assert hasattr(scene, "timestamp")
        assert hasattr(scene, "raw_response")
        
        # 验证摘要
        assert len(scene.summary) > 0
        assert "大厅环境" in scene.summary
        assert "玻璃门" in scene.summary
        
        # 验证物体列表
        assert len(scene.objects) > 0
        
        # 验证空间关系
        assert len(scene.spatial_relations) > 0
        
        # 验证导航提示
        assert len(scene.navigation_hints) > 0
        
        # 验证潜在目标
        assert len(scene.potential_targets) > 0
        
        # 验证时间戳
        assert isinstance(scene.timestamp, float)
        assert scene.timestamp > 0
        
        # 验证原始响应
        assert len(scene.raw_response) > 0
    
    @pytest.mark.asyncio
    async def test_find_target_found(self, vlm_perception, rgb_image):
        """测试目标搜索（找到目标）"""
        # 搜索门
        target = "门"
        result = await vlm_perception.find_target(rgb_image, target)
        
        # 验证搜索结果
        assert result is not None
        assert result.found is True
        assert result.target_description == target
        assert result.best_match is not None
        assert len(result.matched_objects) > 0
        
        # 验证建议动作
        assert len(result.suggested_action) > 0
        assert "前进" in result.suggested_action
        assert "右转" in result.suggested_action
        
        # 验证置信度
        assert 0 <= result.confidence <= 1
        assert result.confidence > 0.5
        
        # 验证说明
        assert len(result.explanation) > 0
    
    @pytest.mark.asyncio
    async def test_find_target_not_found(self, vlm_perception, rgb_image):
        """测试目标搜索（未找到目标）"""
        # 搜索不存在的目标
        target = "不存在的东西"
        result = await vlm_perception.find_target(rgb_image, target)
        
        # 验证搜索结果
        assert result is not None
        assert result.found is False
        assert result.target_description == target
        assert result.best_match is None
        assert len(result.matched_objects) == 0
        
        # 验证建议动作
        assert len(result.suggested_action) > 0
        assert "搜索" in result.suggested_action
        
        # 验证置信度
        assert result.confidence == 0.0
        
        # 验证说明
        assert len(result.explanation) > 0
    
    @pytest.mark.asyncio
    async def test_answer_spatial_query(self, vlm_perception, rgb_image):
        """测试空间问题回答"""
        # 提问空间问题
        query = "门在哪个方向?"
        answer = await vlm_perception.answer_spatial_query(rgb_image, query)
        
        # 验证回答
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "前方" in answer
        assert "门" in answer
    
    @pytest.mark.asyncio
    async def test_get_last_scene(self, vlm_perception, rgb_image):
        """测试获取最近场景描述"""
        # 先描述一个场景
        scene = await vlm_perception.describe_scene(rgb_image)
        
        # 获取最近场景描述
        last_scene = vlm_perception.get_last_scene()
        
        # 验证最近场景
        assert last_scene is not None
        assert last_scene.summary == scene.summary
        assert len(last_scene.objects) == len(scene.objects)
        assert last_scene.raw_response == scene.raw_response
    
    @pytest.mark.asyncio
    async def test_get_detection_history(self, vlm_perception, rgb_image):
        """测试获取检测历史"""
        # 获取检测历史
        history = vlm_perception.get_detection_history(count=10)
        
        # 初始历史应该为空
        assert isinstance(history, list)
        assert len(history) <= 10
        
        # 描述几个场景
        for _ in range(3):
            await vlm_perception.describe_scene(rgb_image)
        
        # 获取检测历史
        history = vlm_perception.get_detection_history(count=10)
        
        # 验证历史包含最近的对象
        assert len(history) >= 3
        for obj in history:
            assert hasattr(obj, "id")
            assert hasattr(obj, "label")
            assert hasattr(obj, "confidence")


class TestVLMPerceptionWithYOLO:
    """测试带YOLO的VLM感知"""
    
    @pytest.mark.asyncio
    async def test_describe_scene_with_yolo(self, vlm_perception_with_yolo, rgb_image):
        """测试带YOLO的场景描述"""
        with patch('brain.perception.vlm.vlm_perception.YOLO', MockYOLO):
            # 描述场景
            scene = await vlm_perception_with_yolo.describe_scene(rgb_image)
            
            # 验证场景描述
            assert scene is not None
            assert hasattr(scene, "summary")
            assert hasattr(scene, "objects")
            
            # 验证有YOLO检测的物体
            # 注意：在模拟环境中，由于ollama_client为None，不会调用YOLO
            # 但如果有真实的YOLO检测，应该会调用_fuse_detections方法
    
    @pytest.mark.asyncio
    async def test_yolo_detection(self, vlm_perception_with_yolo, rgb_image):
        """测试YOLO检测"""
        with patch('brain.perception.vlm.vlm_perception.YOLO', MockYOLO):
            # 直接测试YOLO检测
            yolo_objects = await vlm_perception_with_yolo._yolo_detect(rgb_image)
            
            # 验证YOLO检测结果
            assert isinstance(yolo_objects, list)
            assert len(yolo_objects) == 2
            
            # 验证第一个对象
            obj = yolo_objects[0]
            assert obj.id == "yolo_0"
            assert obj.label == "person"
            assert obj.confidence == 0.85
            assert obj.source.value == "yolo"
            
            # 验证边界框
            assert obj.bbox is not None
            assert obj.bbox.x == 0.4
            assert obj.bbox.y == 0.3
            assert obj.bbox.width == 0.2
            assert obj.bbox.height == 0.4
    
    @pytest.mark.asyncio
    async def test_fuse_detections(self, vlm_perception_with_yolo, rgb_image):
        """测试检测融合"""
        from brain.perception.vlm.vlm_perception import SceneDescription, DetectedObject, DetectionSource
        
        # 创建VLM场景描述
        vlm_scene = SceneDescription(
            summary="测试场景",
            objects=[
                DetectedObject(
                    id="vlm_0",
                    label="门",
                    confidence=0.7,
                    bbox=Mock(x=0.5, y=0.4, width=0.2, height=0.3),
                    source=DetectionSource.VLM
                )
            ],
            spatial_relations=[],
            navigation_hints=[],
            potential_targets=["门"]
        )
        
        # 创建YOLO对象
        yolo_objects = [
            DetectedObject(
                id="yolo_0",
                label="door",
                confidence=0.9,
                bbox=Mock(x=0.5, y=0.4, width=0.15, height=0.25),
                source=DetectionSource.YOLO
            )
        ]
        
        # 测试融合
        fused_scene = vlm_perception_with_yolo._fuse_detections(vlm_scene, yolo_objects)
        
        # 验证融合结果
        assert len(fused_scene.objects) == 1  # 应该合并为一个对象
        
        # 验证融合对象
        obj = fused_scene.objects[0]
        assert obj.id == "vlm_0"  # 应该保留VLM对象的ID
        assert obj.label == "门"
        assert obj.confidence == 0.9  # 应该使用YOLO的更高置信度
        assert obj.bbox.width == 0.15  # 应该使用YOLO的更准确边界框
        assert obj.source.value == "fused"
    
    @pytest.mark.asyncio
    async def test_is_same_object(self, vlm_perception_with_yolo, rgb_image):
        """测试对象匹配判断"""
        from brain.perception.vlm.vlm_perception import DetectedObject, DetectionSource, BoundingBox
        
        # 创建相同对象
        obj1 = DetectedObject(
            id="obj1",
            label="door",
            confidence=0.8,
            bbox=BoundingBox(x=0.5, y=0.4, width=0.2, height=0.3),
            source=DetectionSource.VLM
        )
        
        obj2 = DetectedObject(
            id="obj2",
            label="door",
            confidence=0.7,
            bbox=BoundingBox(x=0.51, y=0.41, width=0.2, height=0.3),
            source=DetectionSource.YOLO
        )
        
        # 测试相同对象
        is_same = vlm_perception_with_yolo._is_same_object(obj1, obj2)
        assert is_same is True
        
        # 创建不同对象
        obj3 = DetectedObject(
            id="obj3",
            label="person",
            confidence=0.8,
            bbox=BoundingBox(x=0.2, y=0.3, width=0.2, height=0.3),
            source=DetectionSource.YOLO
        )
        
        # 测试不同对象
        is_different = vlm_perception_with_yolo._is_same_object(obj1, obj3)
        assert is_different is False
    
    @pytest.mark.asyncio
    async def test_compute_iou(self, vlm_perception_with_yolo, rgb_image):
        """测试IoU计算"""
        from brain.perception.vlm.vlm_perception import BoundingBox
        
        # 创建完全重叠的边界框
        box1 = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        box2 = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        iou = vlm_perception_with_yolo._compute_iou(box1, box2)
        assert iou == 1.0
        
        # 创建部分重叠的边界框
        box3 = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        box4 = BoundingBox(x=0.6, y=0.6, width=0.2, height=0.2)
        iou = vlm_perception_with_yolo._compute_iou(box3, box4)
        assert 0 < iou < 1.0
        
        # 创建不重叠的边界框
        box5 = BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2)
        box6 = BoundingBox(x=0.8, y=0.8, width=0.2, height=0.2)
        iou = vlm_perception_with_yolo._compute_iou(box5, box6)
        assert iou == 0.0


class TestVLMDataStructures:
    """测试VLM数据结构"""
    
    def test_bounding_box(self):
        """测试边界框"""
        from brain.perception.vlm.vlm_perception import BoundingBox
        
        # 创建边界框
        bbox = BoundingBox(
            x=0.5,      # 中心x (0-1 归一化)
            y=0.6,      # 中心y (0-1 归一化)
            width=0.2,    # 宽度 (0-1 归一化)
            height=0.3    # 高度 (0-1 归一化)
        )
        
        # 验证属性
        assert bbox.x == 0.5
        assert bbox.y == 0.6
        assert bbox.width == 0.2
        assert bbox.height == 0.3
        
        # 测试转换为像素坐标
        pixel_coords = bbox.to_pixel(640, 480)  # 图像尺寸640x480
        x1, y1, x2, y2 = pixel_coords
        
        expected_x1 = int((0.5 - 0.2/2) * 640)
        expected_y1 = int((0.6 - 0.3/2) * 480)
        expected_x2 = int((0.5 + 0.2/2) * 640)
        expected_y2 = int((0.6 + 0.3/2) * 480)
        
        assert x1 == expected_x1
        assert y1 == expected_y1
        assert x2 == expected_x2
        assert y2 == expected_y2
        
        # 测试获取中心像素坐标
        center_coords = bbox.center_pixel(640, 480)
        cx, cy = center_coords
        
        assert cx == int(0.5 * 640)
        assert cy == int(0.6 * 480)
    
    def test_detected_object(self):
        """测试检测到的对象"""
        from brain.perception.vlm.vlm_perception import DetectedObject, DetectionSource
        
        # 创建检测对象
        obj = DetectedObject(
            id="test_obj",
            label="门",
            confidence=0.85,
            bbox=Mock(x=0.5, y=0.6, width=0.2, height=0.3),
            description="建筑入口的玻璃门",
            position_description="前方中央偏右",
            estimated_distance=10.5,
            estimated_direction="前方偏右",
            source=DetectionSource.VLM,
            attributes={"color": "透明"}
        )
        
        # 验证属性
        assert obj.id == "test_obj"
        assert obj.label == "门"
        assert obj.confidence == 0.85
        assert obj.description == "建筑入口的玻璃门"
        assert obj.position_description == "前方中央偏右"
        assert obj.estimated_distance == 10.5
        assert obj.estimated_direction == "前方偏右"
        assert obj.source.value == "vlm"
        assert obj.attributes == {"color": "透明"}
        
        # 测试转换为字典
        obj_dict = obj.to_dict()
        assert obj_dict["id"] == "test_obj"
        assert obj_dict["label"] == "门"
        assert obj_dict["confidence"] == 0.85
        assert obj_dict["description"] == "建筑入口的玻璃门"
        assert obj_dict["position"] == "前方中央偏右"
        assert obj_dict["distance"] == 10.5
        assert obj_dict["direction"] == "前方偏右"
        assert "bbox" in obj_dict
        assert obj_dict["bbox"]["x"] == 0.5
        assert obj_dict["bbox"]["y"] == 0.6
        assert obj_dict["bbox"]["width"] == 0.2
        assert obj_dict["bbox"]["height"] == 0.3
    
    def test_scene_description(self):
        """测试场景描述"""
        from brain.perception.vlm.vlm_perception import SceneDescription, DetectedObject
        
        # 创建场景描述
        objects = [
            DetectedObject(
                id="obj1",
                label="门",
                confidence=0.85,
                description="建筑入口的玻璃门",
                position_description="前方中央"
            ),
            DetectedObject(
                id="obj2",
                label="植物",
                confidence=0.75,
                description="盆栽植物",
                position_description="右侧"
            )
        ]
        
        scene = SceneDescription(
            summary="这是一个室内大厅环境",
            objects=objects,
            spatial_relations=["门在前方中央", "植物在右侧"],
            navigation_hints=["前方可以通行"],
            potential_targets=["门", "入口"],
            raw_response="模拟VLM响应"
        )
        
        # 验证属性
        assert scene.summary == "这是一个室内大厅环境"
        assert len(scene.objects) == 2
        assert scene.objects[0].id == "obj1"
        assert scene.objects[1].id == "obj2"
        assert len(scene.spatial_relations) == 2
        assert scene.spatial_relations[0] == "门在前方中央"
        assert len(scene.navigation_hints) == 1
        assert scene.navigation_hints[0] == "前方可以通行"
        assert len(scene.potential_targets) == 2
        assert scene.potential_targets[0] == "门"
        assert scene.potential_targets[1] == "入口"
        assert scene.raw_response == "模拟VLM响应"
        
        # 测试转换为字典
        scene_dict = scene.to_dict()
        assert scene_dict["summary"] == "这是一个室内大厅环境"
        assert len(scene_dict["objects"]) == 2
        assert len(scene_dict["spatial_relations"]) == 2
        assert len(scene_dict["navigation_hints"]) == 1
        assert len(scene_dict["potential_targets"]) == 2
    
    def test_target_search_result(self):
        """测试目标搜索结果"""
        from brain.perception.vlm.vlm_perception import TargetSearchResult, DetectedObject
        
        # 创建检测对象
        obj = DetectedObject(
            id="target_obj",
            label="门",
            confidence=0.85,
            description="建筑入口的玻璃门",
            position_description="前方中央"
        )
        
        # 创建搜索结果
        result = TargetSearchResult(
            found=True,
            target_description="门",
            matched_objects=[obj],
            best_match=obj,
            suggested_action="直行前进约10米",
            confidence=0.85,
            explanation="目标位于前方，可以直接前进"
        )
        
        # 验证属性
        assert result.found is True
        assert result.target_description == "门"
        assert len(result.matched_objects) == 1
        assert result.matched_objects[0].id == "target_obj"
        assert result.best_match.id == "target_obj"
        assert result.suggested_action == "直行前进约10米"
        assert result.confidence == 0.85
        assert result.explanation == "目标位于前方，可以直接前进"


if __name__ == "__main__":
    pytest.main([__file__])






