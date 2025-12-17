"""
目标检测器 - Object Detector

负责:
- 目标检测与分类
- 目标跟踪
- 特征提取
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
from loguru import logger

from brain.perception.environment import (
    ObjectType,
    DetectedObject,
    Position3D,
    BoundingBox
)


class DetectionMode(Enum):
    """检测模式"""
    FAST = "fast"          # 快速检测
    ACCURATE = "accurate"  # 精确检测
    TRACKING = "tracking"  # 跟踪模式


@dataclass
class Detection:
    """检测结果"""
    object_type: ObjectType
    confidence: float
    bounding_box_2d: Tuple[int, int, int, int]  # x, y, w, h
    position_3d: Optional[Position3D] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackedObject:
    """跟踪的物体"""
    track_id: str
    object_type: ObjectType
    position: Position3D
    velocity: Dict[str, float]
    history: List[Position3D] = field(default_factory=list)
    lost_frames: int = 0
    age: int = 0
    
    def predict_position(self, dt: float = 1.0) -> Position3D:
        """预测未来位置"""
        return Position3D(
            x=self.position.x + self.velocity.get("vx", 0) * dt,
            y=self.position.y + self.velocity.get("vy", 0) * dt,
            z=self.position.z + self.velocity.get("vz", 0) * dt
        )


class ObjectDetector:
    """
    目标检测器
    
    集成目标检测模型，提供检测和跟踪功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 检测模式
        self.mode = DetectionMode(self.config.get("mode", "fast"))
        
        # 跟踪器
        self.tracks: Dict[str, TrackedObject] = {}
        self.next_track_id = 0
        
        # 检测阈值
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # 类别映射
        self.class_mapping = {
            0: ObjectType.PERSON,
            1: ObjectType.VEHICLE,
            2: ObjectType.BUILDING,
            3: ObjectType.TREE,
            4: ObjectType.OBSTACLE,
        }
        
        logger.info(f"ObjectDetector 初始化完成, 模式: {self.mode.value}")
    
    async def detect(
        self, 
        image: Any,
        depth_map: Optional[Any] = None
    ) -> List[Detection]:
        """
        执行目标检测
        
        Args:
            image: 输入图像
            depth_map: 深度图 (可选)
            
        Returns:
            List[Detection]: 检测结果列表
        """
        detections = []
        
        try:
            # 这里应该调用实际的检测模型 (YOLO, SSD, etc.)
            # 示例中返回模拟数据
            
            # 模拟检测结果
            mock_detections = [
                {
                    "class_id": 0,
                    "confidence": 0.85,
                    "bbox": (100, 100, 50, 120),
                    "depth": 5.0
                },
                {
                    "class_id": 1,
                    "confidence": 0.75,
                    "bbox": (300, 200, 100, 80),
                    "depth": 15.0
                }
            ]
            
            for det in mock_detections:
                if det["confidence"] >= self.confidence_threshold:
                    object_type = self.class_mapping.get(
                        det["class_id"], 
                        ObjectType.UNKNOWN
                    )
                    
                    # 计算3D位置 (如果有深度信息)
                    position_3d = None
                    if depth_map is not None or "depth" in det:
                        # 简化的3D位置计算
                        x, y, w, h = det["bbox"]
                        depth = det.get("depth", 10.0)
                        
                        # 假设相机参数
                        fx, fy = 500, 500  # 焦距
                        cx, cy = 320, 240  # 主点
                        
                        # 计算3D坐标
                        x_3d = (x + w/2 - cx) * depth / fx
                        y_3d = (y + h/2 - cy) * depth / fy
                        z_3d = depth
                        
                        position_3d = Position3D(x=x_3d, y=y_3d, z=z_3d)
                    
                    detection = Detection(
                        object_type=object_type,
                        confidence=det["confidence"],
                        bounding_box_2d=det["bbox"],
                        position_3d=position_3d
                    )
                    detections.append(detection)
            
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
        
        return detections
    
    async def detect_and_track(
        self, 
        image: Any,
        depth_map: Optional[Any] = None
    ) -> List[TrackedObject]:
        """
        检测并跟踪目标
        
        Args:
            image: 输入图像
            depth_map: 深度图 (可选)
            
        Returns:
            List[TrackedObject]: 跟踪的物体列表
        """
        # 执行检测
        detections = await self.detect(image, depth_map)
        
        # 更新跟踪
        self._update_tracks(detections)
        
        # 返回活跃的跟踪
        active_tracks = [
            track for track in self.tracks.values()
            if track.lost_frames < 5
        ]
        
        return active_tracks
    
    def _update_tracks(self, detections: List[Detection]):
        """更新跟踪状态"""
        # 简单的最近邻关联
        matched_tracks = set()
        matched_detections = set()
        
        for i, det in enumerate(detections):
            if det.position_3d is None:
                continue
            
            min_distance = float('inf')
            best_track = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                if track.object_type != det.object_type:
                    continue
                
                # 使用预测位置进行匹配
                predicted = track.predict_position(0.1)
                distance = det.position_3d.distance_to(predicted)
                
                if distance < min_distance and distance < 3.0:  # 3米阈值
                    min_distance = distance
                    best_track = track_id
            
            if best_track:
                # 更新跟踪
                track = self.tracks[best_track]
                
                # 计算速度
                dt = 0.1  # 假设帧间隔
                track.velocity = {
                    "vx": (det.position_3d.x - track.position.x) / dt,
                    "vy": (det.position_3d.y - track.position.y) / dt,
                    "vz": (det.position_3d.z - track.position.z) / dt
                }
                
                track.position = det.position_3d
                track.history.append(det.position_3d)
                track.lost_frames = 0
                track.age += 1
                
                # 限制历史长度
                if len(track.history) > 50:
                    track.history = track.history[-50:]
                
                matched_tracks.add(best_track)
                matched_detections.add(i)
        
        # 处理未匹配的检测 - 创建新跟踪
        for i, det in enumerate(detections):
            if i in matched_detections:
                continue
            if det.position_3d is None:
                continue
            
            track_id = f"track_{self.next_track_id}"
            self.next_track_id += 1
            
            self.tracks[track_id] = TrackedObject(
                track_id=track_id,
                object_type=det.object_type,
                position=det.position_3d,
                velocity={"vx": 0, "vy": 0, "vz": 0},
                history=[det.position_3d]
            )
        
        # 更新未匹配的跟踪
        for track_id in self.tracks:
            if track_id not in matched_tracks:
                self.tracks[track_id].lost_frames += 1
        
        # 清理丢失的跟踪
        to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.lost_frames > 10
        ]
        for track_id in to_remove:
            del self.tracks[track_id]
    
    async def detect_specific(
        self, 
        image: Any,
        target_types: List[ObjectType]
    ) -> List[Detection]:
        """
        检测特定类型的目标
        
        Args:
            image: 输入图像
            target_types: 目标类型列表
            
        Returns:
            List[Detection]: 匹配的检测结果
        """
        all_detections = await self.detect(image)
        
        return [
            det for det in all_detections
            if det.object_type in target_types
        ]
    
    async def detect_in_area(
        self,
        image: Any,
        roi: Tuple[int, int, int, int]  # x, y, w, h
    ) -> List[Detection]:
        """
        在指定区域内检测
        
        Args:
            image: 输入图像
            roi: 感兴趣区域
            
        Returns:
            List[Detection]: 区域内的检测结果
        """
        all_detections = await self.detect(image)
        
        rx, ry, rw, rh = roi
        
        def in_roi(bbox):
            x, y, w, h = bbox
            cx, cy = x + w/2, y + h/2
            return rx <= cx <= rx + rw and ry <= cy <= ry + rh
        
        return [
            det for det in all_detections
            if in_roi(det.bounding_box_2d)
        ]
    
    def get_track(self, track_id: str) -> Optional[TrackedObject]:
        """获取指定跟踪"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """获取所有活跃跟踪"""
        return [
            track for track in self.tracks.values()
            if track.lost_frames < 5
        ]
    
    def clear_tracks(self):
        """清除所有跟踪"""
        self.tracks.clear()
        logger.info("所有跟踪已清除")
    
    def to_detected_objects(
        self, 
        tracks: List[TrackedObject]
    ) -> List[DetectedObject]:
        """将跟踪转换为检测物体"""
        return [
            DetectedObject(
                id=track.track_id,
                object_type=track.object_type,
                position=track.position,
                velocity=track.velocity,
                confidence=1.0 - track.lost_frames * 0.1,
                track_id=track.track_id
            )
            for track in tracks
        ]

