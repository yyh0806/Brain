# -*- coding: utf-8 -*-
"""
增强的世界模型 - 集成SLAM

这是WorldModel的SLAM集成版本，实现了：
1. 几何层：引用SLAM地图（零拷贝）
2. 语义层：独立管理，叠加到几何层
3. 因果层：推理和关系管理

相比原版world_model.py的改动：
- ✅ 移除独立的current_map（numpy数组）
- ✅ 引用SLAM Manager的地图（零拷贝）
- ✅ 新增semantic_overlays（语义叠加）
- ✅ 保持向后兼容性（geometric_map属性）
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from copy import deepcopy
import math
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入SLAM集成
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

try:
    from slam_integration.src import SLAMManager, SLAMConfig, CoordinateTransformer
    SLAM_AVAILABLE = True
    logger.info("SLAM集成模块可用")
except ImportError:
    SLAM_AVAILABLE = False
    logger.warning("SLAM集成模块不可用，将使用传统模式")

# 导入原始WorldModel的类型定义
from brain.cognitive.world_model.environment_change import (
    ChangeType,
    ChangePriority,
    EnvironmentChange
)
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.object_tracking.tracked_object import TrackedObject
from brain.cognitive.world_model.semantic.semantic_object import (
    ObjectState,
    SemanticObject,
    ExplorationFrontier
)
from brain.cognitive.world_model.causal_graph import (
    CausalGraph,
    CausalNode,
    CausalRelationType
)

# 导入 PerceptionData
try:
    from brain.perception.ros2_sensor_manager import PerceptionData
    PERCEPTION_DATA_AVAILABLE = True
except ImportError:
    PERCEPTION_DATA_AVAILABLE = False
    PerceptionData = None


@dataclass
class SemanticLabel:
    """语义标签 - 用于叠加到几何地图"""
    label: str
    confidence: float
    object_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnhancedMap:
    """增强地图 - 几何+语义叠加"""
    geometric_layer: Any  # SLAM OccupancyGrid（零拷贝引用）
    semantic_overlays: Dict[Tuple[int, int], SemanticLabel]  # 语义标注
    risk_areas: Optional[np.ndarray] = None  # 风险区域标注
    exploration_frontier: List[ExplorationFrontier] = field(default_factory=list)


class EnhancedWorldModel:
    """
    增强的世界模型 - 集成SLAM

    三模态架构：
    1. 几何层（Geometric Layer）：引用SLAM地图（零拷贝）
    2. 语义层（Semantic Layer）：VLM检测的语义物体
    3. 因果层（Causal Layer）：状态演化追踪和推理

    相比原版WorldModel的改进：
    - ✅ 零拷贝引用SLAM地图，避免数据复制
    - ✅ 清晰的三层架构
    - ✅ 支持语义叠加到几何地图
    - ✅ 更好的内存管理
    """

    # 感兴趣的物体类型
    NAVIGATION_OBJECTS = {
        "门", "door", "entrance", "入口", "出口", "exit",
        "建筑", "building", "房子", "house",
        "路", "road", "path", "道路", "走廊", "corridor",
        "楼梯", "stairs", "电梯", "elevator"
    }

    # 障碍物类型
    OBSTACLE_TYPES = {
        "墙", "wall", "障碍", "obstacle", "栏杆", "fence",
        "车", "car", "vehicle", "人", "person", "pedestrian"
    }

    # 变化类型配置
    CHANGE_CONFIG = {
        ChangeType.NEW_OBSTACLE: {
            "priority": ChangePriority.HIGH,
            "threshold": 0.7,
            "requires_replan": True,
            "requires_confirmation": False
        },
        ChangeType.PATH_BLOCKED: {
            "priority": ChangePriority.CRITICAL,
            "threshold": 0.9,
            "requires_replan": True,
            "requires_confirmation": True
        },
        ChangeType.TARGET_APPEARED: {
            "priority": ChangePriority.HIGH,
            "threshold": 0.6,
            "requires_replan": False,
            "requires_confirmation": True
        },
        ChangeType.TARGET_MOVED: {
            "priority": ChangePriority.MEDIUM,
            "threshold": 0.5,
            "requires_replan": True,
            "requires_confirmation": False
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # ========== SLAM集成 ==========
        # 初始化SLAM Manager
        if SLAM_AVAILABLE:
            slam_config = SLAMConfig(
                backend=self.config.get("slam_backend", "fast_livo"),
                resolution=self.config.get("map_resolution", 0.1),
                zero_copy=True
            )
            self.slam_manager = SLAMManager(slam_config)
            self.coordinate_transformer = CoordinateTransformer(self.slam_manager)
            logger.info("SLAM Manager已初始化")
        else:
            self.slam_manager = None
            self.coordinate_transformer = None
            logger.warning("SLAM不可用，使用传统模式")

        # ========== 几何层（SLAM地图引用）==========
        # 注意：不再维护current_map numpy数组
        # 改为通过slam_manager.slam_map引用SLAM地图（零拷贝）
        self._slam_map_ref = None  # SLAM地图引用
        self._slam_pose_ref = None  # SLAM位姿引用

        # ========== 语义层（独立管理）==========
        self.semantic_objects: Dict[str, SemanticObject] = {}
        self._object_counter = 0
        self.max_semantic_objects = self.config.get("max_semantic_objects", 500)

        # 语义叠加：将语义信息叠加到几何地图的栅格上
        # key: (grid_x, grid_y), value: SemanticLabel
        self.semantic_overlays: Dict[Tuple[int, int], SemanticLabel] = {}

        # ========== 因果层（独立管理）==========
        self.causal_graph = CausalGraph()
        self.state_history: List[Dict[str, Any]] = []
        self.last_state: Dict[str, Any] = {}

        # ========== 机器人状态 ==========
        self.robot_position: Dict[str, float] = {"x": 0, "y": 0, "z": 0}
        self.robot_velocity: Dict[str, float] = {"vx": 0, "vy": 0, "vz": 0}
        self.robot_heading: float = 0.0

        # ========== 环境状态 ==========
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.weather: Dict[str, Any] = {
            "condition": "clear",
            "temperature": 25.0
        }

        # ========== 探索管理 ==========
        self.exploration_frontiers: List[ExplorationFrontier] = []
        self.explored_positions: Set[Tuple[int, int]] = set()
        self.grid_resolution = self.config.get("grid_resolution", 0.5)

        # ========== 目标管理 ==========
        self.current_target: Optional[SemanticObject] = None

        # ========== 变化检测 ==========
        self.previous_state: Optional[Dict[str, Any]] = None
        self.pending_changes: List[EnvironmentChange] = []
        self.change_history: List[EnvironmentChange] = []
        self.max_history = self.config.get("max_change_history", 100)
        self.last_update: datetime = datetime.now()

        # ========== 认知特有标注 ==========
        self.risk_areas: Optional[np.ndarray] = None  # 风险区域标注
        self.task_hotspots: List[Dict[str, Any]] = []

        logger.info("EnhancedWorldModel初始化完成 (SLAM集成模式)")

    # ========== SLAM地图访问接口 ==========

    @property
    def slam_map(self):
        """
        获取SLAM地图（零拷贝引用）

        直接引用ROS2 OccupancyGrid消息，不进行数据复制
        """
        if self.slam_manager:
            return self.slam_manager.slam_map
        return self._slam_map_ref

    @property
    def slam_pose(self):
        """获取SLAM位姿（零拷贝引用）"""
        if self.slam_manager:
            return self.slam_manager.slam_pose
        return self._slam_pose_ref

    @property
    def geometric_map(self) -> Optional[np.ndarray]:
        """
        获取几何地图（numpy数组格式）

        注意：这会进行数据转换，不是零拷贝
        建议优先使用slam_map属性获取零拷贝访问

        这是向后兼容的接口，用于替换原版的current_map
        """
        if self.slam_manager:
            return self.slam_manager.get_geometric_map()
        return None

    def get_map_metadata(self):
        """获取地图元数据"""
        if self.slam_manager:
            return self.slam_manager.get_map_metadata()
        return None

    def world_to_grid(self, world_position: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标 → 栅格坐标"""
        if self.slam_manager:
            return self.slam_manager.world_to_grid(world_position)
        raise ValueError("SLAM Manager不可用")

    def grid_to_world(self, grid_position: Tuple[int, int]) -> Tuple[float, float]:
        """栅格坐标 → 世界坐标"""
        if self.slam_manager:
            return self.slam_manager.grid_to_world(grid_position)
        raise ValueError("SLAM Manager不可用")

    # ========== 感知数据更新 ==========

    async def update_from_slam(self):
        """
        从SLAM更新几何地图（零拷贝引用）

        这是新增的接口，用于从SLAM更新几何层
        """
        if self.slam_manager:
            # 等待SLAM地图就绪
            await self.slam_manager.wait_for_map(timeout=5.0)

            # 零拷贝引用SLAM地图（通过slam_manager.slam_map属性）
            # 不进行数据复制
            self._slam_map_ref = self.slam_manager.slam_map
            self._slam_pose_ref = self.slam_manager.slam_pose

            logger.info("SLAM地图已更新（零拷贝引用）")

    async def update_from_perception(
        self,
        perception_data: Union['PerceptionData', Dict[str, Any]]
    ) -> List[EnvironmentChange]:
        """
        从感知数据更新世界模型

        重点：不再更新几何地图（由SLAM负责）
        只更新语义层（语义物体）和因果层

        Args:
            perception_data: PerceptionData对象或字典

        Returns:
            检测到的环境变化列表
        """
        self._save_previous_state()
        changes = []

        # 处理Dict格式（向后兼容）
        if isinstance(perception_data, dict):
            return self._update_from_dict(perception_data)

        # 更新机器人位姿（如果SLAM不可用）
        if perception_data.pose and not self.slam_manager:
            self.robot_position.update({
                "x": perception_data.pose.x,
                "y": perception_data.pose.y,
                "z": perception_data.pose.z
            })
            self.robot_heading = perception_data.pose.yaw

        # 更新语义物体（VLM检测结果）
        if hasattr(perception_data, 'semantic_objects') and perception_data.semantic_objects:
            semantic_changes = self._update_semantic_objects(perception_data.semantic_objects)
            changes.extend(semantic_changes)

        # 语义叠加到几何地图
        if self.slam_manager and self.semantic_objects:
            self._update_semantic_overlays()

        # 更新因果图
        current_state = {
            "timestamp": datetime.now(),
            "robot_position": self.robot_position.copy(),
            "semantic_objects": len(self.semantic_objects)
        }
        self._detect_and_update_causal_relations(self.last_state, current_state)
        self.last_state = current_state

        self.last_update = datetime.now()
        return changes

    def _update_semantic_objects(self, semantic_objects: List[Any]) -> List[EnvironmentChange]:
        """
        更新语义物体（VLM检测结果）

        这是新增的方法，专门处理语义层更新
        """
        changes = []

        for obj_data in semantic_objects:
            # 提取物体信息
            if hasattr(obj_data, 'label'):
                label = obj_data.label
                confidence = getattr(obj_data, 'confidence', 0.8)
                world_pos = getattr(obj_data, 'world_position', (0, 0, 0))
            else:
                # 字典格式
                label = obj_data.get('label', 'unknown')
                confidence = obj_data.get('confidence', 0.8)
                world_pos = obj_data.get('position', (0, 0, 0))

            # 转换为2D坐标（SemanticObject使用Tuple[float, float]）
            world_position_2d = (world_pos[0], world_pos[1]) if len(world_pos) >= 2 else (0.0, 0.0)

            # 创建或更新语义物体
            obj_id = f"semantic_{self._object_counter}"
            self._object_counter += 1

            semantic_obj = SemanticObject(
                id=obj_id,
                label=label,
                world_position=world_position_2d,
                confidence=confidence,
                state=ObjectState.DETECTED
            )

            # 检查是否与已有物体匹配
            matched = self._match_semantic_object(semantic_obj)
            if matched:
                # 更新已有物体
                self.semantic_objects[matched].world_position = world_position_2d
                self.semantic_objects[matched].confidence = confidence
                self.semantic_objects[matched].state = ObjectState.TRACKED
            else:
                # 新物体
                self.semantic_objects[obj_id] = semantic_obj
                changes.append(EnvironmentChange(
                    change_type=ChangeType.NEW_OBSTACLE if label in self.OBSTACLE_TYPES else ChangeType.TARGET_APPEARED,
                    priority=ChangePriority.HIGH,
                    description=f"检测到新物体: {label}",
                    data={"object_id": obj_id, "label": label, "position": world_position_2d},
                    confidence=confidence
                ))

        # 内存管理：限制语义物体数量
        if len(self.semantic_objects) > self.max_semantic_objects:
            self._cleanup_semantic_objects()

        return changes

    def _update_semantic_overlays(self):
        """
        将语义物体叠加到几何地图的栅格上

        这是新增的方法，实现语义层到几何层的叠加
        """
        self.semantic_overlays.clear()

        for obj_id, obj in self.semantic_objects.items():
            try:
                # 将物体世界坐标转换为栅格坐标
                world_pos = obj.world_position if len(obj.world_position) >= 2 else (0, 0)
                grid_pos = self.world_to_grid(world_pos)

                # 添加语义标签到栅格
                self.semantic_overlays[grid_pos] = SemanticLabel(
                    label=obj.label,
                    confidence=obj.confidence,
                    object_id=obj.id
                )
            except Exception as e:
                logger.debug(f"无法将物体{obj_id}叠加到地图: {e}")

    def _match_semantic_object(self, new_obj: SemanticObject) -> Optional[str]:
        """匹配新物体与已有物体"""
        for obj_id, existing_obj in self.semantic_objects.items():
            # 检查标签相似度
            if new_obj.label.lower() in existing_obj.label.lower() or \
               existing_obj.label.lower() in new_obj.label.lower():
                # 检查位置距离
                new_pos = new_obj.world_position
                existing_pos = existing_obj.world_position
                if len(new_pos) >= 2 and len(existing_pos) >= 2:
                    dist = math.sqrt(
                        (new_pos[0] - existing_pos[0])**2 +
                        (new_pos[1] - existing_pos[1])**2
                    )
                    if dist < 2.0:  # 2米阈值
                        return obj_id
        return None

    def _cleanup_semantic_objects(self):
        """清理旧的语义物体（LRU策略）"""
        # 按last_seen时间排序，删除最旧的
        sorted_objects = sorted(
            self.semantic_objects.items(),
            key=lambda x: x[1].last_seen if hasattr(x[1], 'last_seen') else datetime.min
        )

        # 删除最旧的20%
        num_to_remove = len(self.semantic_objects) - int(self.max_semantic_objects * 0.8)
        for obj_id, _ in sorted_objects[:num_to_remove]:
            del self.semantic_objects[obj_id]
            logger.debug(f"清理旧语义物体: {obj_id}")

    def _detect_and_update_causal_relations(self, old_state: Dict, new_state: Dict):
        """检测因果关系并更新因果图"""
        if not old_state or not new_state:
            return

        # 检测物体位置变化
        if "semantic_objects" in new_state:
            old_count = old_state.get("semantic_objects", 0)
            new_count = new_state["semantic_objects"]
            if new_count > old_count:
                # 新物体出现
                self.causal_graph.add_node(
                    CausalNode(
                        id=f"object_appearance_{datetime.now().timestamp()}",
                        type="object_appearance",
                        properties={"count": new_count - old_count}
                    )
                )

    def _save_previous_state(self):
        """保存前一个状态"""
        self.previous_state = {
            "robot_position": self.robot_position.copy(),
            "semantic_objects_count": len(self.semantic_objects),
            "timestamp": datetime.now()
        }

    def _update_from_dict(self, data: Dict[str, Any]) -> List[EnvironmentChange]:
        """从字典更新（向后兼容）"""
        # 简化实现，处理基本字段
        if "robot_position" in data:
            self.robot_position.update(data["robot_position"])

        return []

    # ========== 增强地图访问 ==========

    def get_enhanced_map(self) -> Optional[EnhancedMap]:
        """
        获取增强地图（几何+语义叠加）

        Returns:
            EnhancedMap: 包含几何层（SLAM地图）和语义层
        """
        if not self.slam_map:
            logger.warning("SLAM地图不可用")
            return None

        return EnhancedMap(
            geometric_layer=self.slam_map,  # SLAM地图引用
            semantic_overlays=self.semantic_overlays,  # 语义标注
            risk_areas=self.risk_areas,  # 风险区域
            exploration_frontier=self.exploration_frontiers  # 探索边界
        )

    def get_location(self, object_name: str) -> Optional[Dict[str, Any]]:
        """
        获取物体位置（从语义层）

        Args:
            object_name: 物体名称

        Returns:
            位置信息字典 {position, confidence, ...}
        """
        for obj_id, obj in self.semantic_objects.items():
            if object_name.lower() in obj.label.lower():
                return {
                    "position": obj.world_position,
                    "confidence": obj.confidence,
                    "state": obj.state,
                    "id": obj.id
                }
        return None

    # ========== 规划上下文生成 ==========

    def get_planning_context(self) -> PlanningContext:
        """
        生成规划上下文

        Returns:
            PlanningContext: 包含地图、障碍物、目标等信息
        """
        return PlanningContext(
            map_data=self.geometric_map,  # numpy数组格式（兼容）
            map_metadata=self.get_map_metadata(),
            obstacles=list(self.tracked_objects.values()),
            targets=[obj for obj in self.semantic_objects.values() if obj.is_target],
            robot_position=self.robot_position,
            exploration_frontiers=self.exploration_frontiers
        )

    # ========== 清理和关闭 ==========

    def shutdown(self):
        """关闭世界模型"""
        if self.slam_manager:
            self.slam_manager.shutdown()
        logger.info("EnhancedWorldModel已关闭")


# ========== 向后兼容适配器 ==========

class WorldModelAdapter:
    """
    WorldModel适配器 - 将原版WorldModel适配到SLAM集成版本

    提供向后兼容的接口，逐步迁移
    """

    def __init__(self, enhanced_model: EnhancedWorldModel):
        self.enhanced_model = enhanced_model

    @property
    def current_map(self) -> Optional[np.ndarray]:
        """向后兼容：current_map属性"""
        return self.enhanced_model.geometric_map

    def update_from_perception(self, perception_data) -> List[EnvironmentChange]:
        """向后兼容：update_from_perception方法"""
        return asyncio.run(self.enhanced_model.update_from_perception(perception_data))


if __name__ == "__main__":
    # 测试代码
    async def test_enhanced_world_model():
        print("=" * 60)
        print("EnhancedWorldModel测试")
        print("=" * 60)

        # 1. 创建增强世界模型
        model = EnhancedWorldModel()
        print("\n[测试1] EnhancedWorldModel创建成功")

        # 2. 等待SLAM地图
        if model.slam_manager:
            print("\n[测试2] 等待SLAM地图...")
            success = await model.slam_manager.wait_for_map(timeout=3.0)
            if success:
                print("SLAM地图已就绪")
            else:
                print("SLAM地图超时（可能没有SLAM节点运行）")

        # 3. 测试坐标转换
        try:
            world_pos = (5.0, 3.0)
            grid_pos = model.world_to_grid(world_pos)
            recovered = model.grid_to_world(grid_pos)
            print(f"\n[测试3] 坐标转换: {world_pos} -> {grid_pos} -> {recovered}")
        except Exception as e:
            print(f"\n[测试3] 坐标转换测试跳过: {e}")

        # 4. 获取增强地图
        enhanced_map = model.get_enhanced_map()
        if enhanced_map:
            print(f"\n[测试4] 增强地图获取成功")
            print(f"   几何层: {'有' if enhanced_map.geometric_layer else '无'}")
            print(f"   语义标注: {len(enhanced_map.semantic_overlays)}个")

        # 5. 清理
        model.shutdown()
        print("\n[测试5] EnhancedWorldModel已关闭")

        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    import asyncio
    asyncio.run(test_enhanced_world_model())
