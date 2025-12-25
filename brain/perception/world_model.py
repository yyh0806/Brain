#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局世界模型（Global World Model）

维护持久的环境表示，包括占据地图、语义信息、空间关系等。

核心特性：
- 持久性：地图在多次感知更新之间保持
- 融合能力：整合来自激光雷达、点云、语义信息的多源数据
- 贝叶斯更新：地图单元状态使用概率更新而非覆盖
- 分层表示：不同层级的抽象（底层占据、中层语义、高层场景）
- 时间衰减：旧数据随时间降低置信度
- 查询接口：支持对地图的查询和检索
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from brain.perception.mapping.occupancy_mapper import OccupancyMapper, OccupancyGrid, CellState
from brain.perception.vlm.vlm_perception import DetectedObject, SceneDescription


@dataclass
class MapMetadata:
    """地图元数据"""
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    confidence: float = 1.0  # 整体置信度


@dataclass
class SemanticObject:
    """语义物体（持久化版本）"""
    id: str
    label: str
    confidence: float
    bounding_box: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    spatial_position: Optional[Tuple[float, float, float]] = None  # (x, y, z)
    last_seen: datetime = field(default_factory=datetime.now)
    update_count: int = 0


@dataclass
class SpatialRelation:
    """空间关系（持久化版本）"""
    object1: str  # 物体1的ID或标签
    object2: str  # 物体2的ID或标签
    relation: str  # 关系类型：near, above, below, left_of, right_of, etc.
    confidence: float
    last_seen: datetime = field(default_factory=datetime.now)


class WorldModel:
    """
    全局世界模型
    
    持久维护环境状态，包括：
    - 占据地图（底层感知）
    - 语义物体列表（中层理解）
    - 空间关系（场景上下文）
    - 地图元数据（管理和调试信息）
    """
    
    def __init__(
        self,
        resolution: float = 0.1,
        map_size: float = 50.0,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        
        # 占据地图
        self.occupancy_mapper = OccupancyMapper(
            resolution=resolution,
            map_size=map_size,
            config=self.config.get("occupancy", {})
        )
        
        # 语义信息
        self.semantic_objects: Dict[str, SemanticObject] = {}  # 按ID或标签索引
        self.spatial_relations: List[SpatialRelation] = []
        
        # 地图元数据
        self.metadata = MapMetadata(
            created_at=datetime.now(),
            last_updated=datetime.now(),
            update_count=0,
            confidence=1.0
        )
        
        # 更新参数
        self.occupied_prob = self.config.get("occupied_prob", 0.7)
        self.free_prob = self.config.get("free_prob", 0.3)
        self.decay_rate = self.config.get("decay_rate", 0.1)  # 时间衰减率
        self.semantic_decay = self.config.get("semantic_decay", 0.05)  # 语义信息衰减率
        
        logger.info(f"WorldModel 初始化: 分辨率={resolution}m, 地图大小={map_size}m")
    
    def update_with_perception(self, perception_data: Any) -> None:
        """
        从感知数据更新世界模型
        
        Args:
            perception_data: PerceptionData对象，包含传感器数据
        """
        # 更新地图元数据
        self.metadata.last_updated = datetime.now()
        self.metadata.update_count += 1
        
        # 1. 更新占据地图（从激光雷达、点云）
        self._update_occupancy(perception_data)
        
        # 2. 更新语义信息（如果VLM结果存在）
        self._update_semantic(perception_data)
        
        # 3. 时间衰减（降低旧数据的置信度）
        self._apply_decay()
        
        # 4. 更新整体置信度
        self._update_confidence()
        
        logger.debug(f"WorldModel更新完成: 更新次数={self.metadata.update_count}")
    
    def _update_occupancy(self, perception_data: Any) -> None:
        """更新占据地图"""
        # 检查是否有激光雷达数据
        has_laser = hasattr(perception_data, 'laser_ranges') and perception_data.laser_ranges is not None
        has_pointcloud = hasattr(perception_data, 'pointcloud') and perception_data.pointcloud is not None
        has_pose = hasattr(perception_data, 'pose') and perception_data.pose is not None
        
        # 转换位姿为元组格式
        pose = None
        if has_pose and perception_data.pose:
            pose = (perception_data.pose.x, perception_data.pose.y, perception_data.pose.yaw)
        
        # 从激光雷达更新（优先）
        if has_laser and perception_data.laser_ranges and perception_data.laser_angles:
            self.occupancy_mapper.update_from_laser(
                perception_data.laser_ranges,
                perception_data.laser_angles,
                pose=pose
            )
            logger.debug(f"从激光雷达更新占据地图: {len(perception_data.laser_ranges)}个点")
        
        # 从点云更新
        elif has_pointcloud and perception_data.pointcloud is not None:
            self.occupancy_mapper.update_from_pointcloud(
                perception_data.pointcloud,
                pose=pose
            )
            logger.debug(f"从点云更新占据地图: {perception_data.pointcloud.shape[0]}个点")
    
    def _update_semantic(self, perception_data: Any) -> None:
        """更新语义信息"""
        # 检查是否有VLM结果
        has_scene_desc = hasattr(perception_data, 'scene_description') and perception_data.scene_description is not None
        has_semantic_objs = hasattr(perception_data, 'semantic_objects') and perception_data.semantic_objects
        
        if not has_scene_desc and not has_semantic_objs:
            return
        
        # 1. 更新场景描述
        if has_scene_desc and perception_data.scene_description:
            # 场景描述可以存储在metadata中
            self.metadata.scene_description = perception_data.scene_description.summary
        
        # 2. 更新语义物体
        if has_semantic_objs and perception_data.semantic_objects:
            for obj in perception_data.semantic_objects:
                obj_key = obj.label
                
                # 如果物体已存在，更新其信息
                if obj_key in self.semantic_objects:
                    existing = self.semantic_objects[obj_key]
                    # 使用加权平均更新置信度
                    existing.confidence = (
                        existing.confidence * (1 - self.semantic_decay) +
                        obj.confidence * self.semantic_decay
                    )
                    existing.last_seen = datetime.now()
                    existing.update_count += 1
                    
                    # 更新位置信息（如果有）
                    if hasattr(obj, 'bounding_box') and obj.bounding_box:
                        existing.bounding_box = obj.bounding_box
                    if hasattr(obj, 'description') and obj.description:
                        existing.description = obj.description
                else:
                    # 创建新的语义物体
                    self.semantic_objects[obj_key] = SemanticObject(
                        id=obj_key,
                        label=obj.label,
                        confidence=obj.confidence,
                        bounding_box=getattr(obj, 'bounding_box', None),
                        description=getattr(obj, 'description', None),
                        spatial_position=None,  # 可以从bounding_box计算
                        last_seen=datetime.now(),
                        update_count=1
                    )
            
            logger.debug(f"更新语义物体: {len(self.semantic_objects)}个物体")
        
        # 3. 更新空间关系
        if hasattr(perception_data, 'spatial_relations') and perception_data.spatial_relations:
            # 可以根据物体ID匹配关系
            self.spatial_relations = perception_data.spatial_relations.copy()
            logger.debug(f"更新空间关系: {len(self.spatial_relations)}条")
    
    def _apply_decay(self) -> None:
        """应用时间衰减"""
        # 对占据地图应用衰减（通过OccupancyMapper的贝叶斯更新）
        # 这里不需要额外操作，因为OccupancyMapper已经实现了概率更新
        
        # 对语义物体应用衰减
        for obj_key in list(self.semantic_objects.keys()):
            obj = self.semantic_objects[obj_key]
            
            # 计算年龄（秒）
            age = (datetime.now() - obj.last_seen).total_seconds()
            
            # 衰减因子（指数衰减）
            decay_factor = np.exp(-self.semantic_decay * age / 60.0)  # 每分钟衰减
            
            # 应用衰减
            obj.confidence *= decay_factor
            
            # 如果置信度太低，移除物体
            if obj.confidence < 0.1:
                del self.semantic_objects[obj_key]
                logger.debug(f"移除过期语义物体: {obj_key}")
    
    def _update_confidence(self) -> None:
        """更新整体地图置信度"""
        # 基于地图覆盖率和语义物体数量计算
        grid = self.occupancy_mapper.get_grid()
        total_cells = grid.width * grid.height
        
        # 计算已知区域的比例
        occupied_cells = np.sum(grid.data != CellState.UNKNOWN)
        coverage_ratio = occupied_cells / total_cells if total_cells > 0 else 0
        
        # 基于覆盖率和语义物体数量计算置信度
        semantic_score = min(1.0, len(self.semantic_objects) / 10.0)  # 假设最多10个物体
        self.metadata.confidence = 0.5 * coverage_ratio + 0.5 * semantic_score
    
    def query_occupancy(self, x: float, y: float, radius: float = 0.5) -> bool:
        """
        查询位置的占用状态
        
        Args:
            x: 世界坐标X
            y: 世界坐标Y
            radius: 查询半径（米）
        
        Returns:
            是否被占用
        """
        # 转换为栅格坐标
        gx, gy = self.occupancy_mapper.grid.world_to_grid(x, y)
        
        if not self.occupancy_mapper.grid.is_valid(gx, gy):
            return False
        
        # 检查占用状态
        return self.occupancy_mapper.grid.is_occupied(gx, gy)
    
    def get_occupancy_status(self, x: float, y: float) -> float:
        """
        获取位置的占用概率
        
        Args:
            x: 世界坐标X
            y: 世界坐标Y
        
        Returns:
            占用概率（0-1）
        """
        gx, gy = self.occupancy_mapper.grid.world_to_grid(x, y)
        
        if not self.occupancy_mapper.grid.is_valid(gx, gy):
            return 0.0  # 未知区域
        
        # 返回栅格值（需要归一化到0-1）
        cell_value = int(self.occupancy_mapper.grid.get_cell(gx, gy))
        if cell_value == CellState.OCCUPIED:
            return 1.0
        elif cell_value == CellState.FREE:
            return 0.0
        else:
            return 0.5  # 未知
    
    def get_global_map(self) -> OccupancyGrid:
        """
        获取全局占据地图快照
        
        Returns:
            OccupancyGrid对象（包含地图数据的深拷贝）
        """
        grid = self.occupancy_mapper.get_grid()
        
        # 返回深拷贝以避免外部修改
        return OccupancyGrid(
            width=grid.width,
            height=grid.height,
            resolution=grid.resolution,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
            data=grid.data.copy()
        )
    
    def get_semantic_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取语义地图
        
        Returns:
            按区域分组的语义信息
        """
        # 按空间位置分组物体
        by_position: Dict[str, List[Dict[str, Any]]] = {
            "objects": []
        }
        
        for obj in self.semantic_objects.values():
            obj_info = {
                "label": obj.label,
                "confidence": obj.confidence,
                "last_seen": obj.last_seen.isoformat(),
                "update_count": obj.update_count
            }
            
            if obj.bounding_box:
                obj_info["bounding_box"] = obj.bounding_box
            if obj.description:
                obj_info["description"] = obj.description
            if obj.spatial_position:
                obj_info["position"] = obj.spatial_position
            
            by_position["objects"].append(obj_info)
        
        return by_position
    
    def get_map_statistics(self) -> Dict[str, Any]:
        """
        获取地图统计信息
        
        Returns:
            地图统计字典
        """
        grid = self.occupancy_mapper.get_grid()
        total_cells = grid.width * grid.height
        
        occupied_cells = np.sum(grid.data == CellState.OCCUPIED)
        free_cells = np.sum(grid.data == CellState.FREE)
        unknown_cells = np.sum(grid.data == CellState.UNKNOWN)
        
        return {
            "total_cells": total_cells,
            "occupied_cells": int(occupied_cells),
            "free_cells": int(free_cells),
            "unknown_cells": int(unknown_cells),
            "occupied_ratio": float(occupied_cells) / total_cells if total_cells > 0 else 0,
            "free_ratio": float(free_cells) / total_cells if total_cells > 0 else 0,
            "semantic_objects_count": len(self.semantic_objects),
            "spatial_relations_count": len(self.spatial_relations),
            "map_age_seconds": (datetime.now() - self.metadata.created_at).total_seconds(),
            "last_update": self.metadata.last_updated.isoformat(),
            "update_count": self.metadata.update_count,
            "confidence": self.metadata.confidence
        }
    
    def reset(self) -> None:
        """重置世界模型"""
        # 重置占据地图
        self.occupancy_mapper = OccupancyMapper(
            resolution=self.occupancy_mapper.resolution,
            map_size=self.occupancy_mapper.map_size,
            config=self.config.get("occupancy", {})
        )
        
        # 重置语义信息
        self.semantic_objects.clear()
        self.spatial_relations.clear()
        
        # 重置元数据
        self.metadata = MapMetadata(
            created_at=datetime.now(),
            last_updated=datetime.now(),
            update_count=0,
            confidence=1.0
        )
        
        logger.info("WorldModel已重置")

