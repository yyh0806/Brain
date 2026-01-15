# -*- coding: utf-8 -*-
"""
增量变化检测器测试
"""

import pytest
import numpy as np
import time
from brain.cognitive.world_model.change_detector import (
    IncrementalChangeDetector,
    SemanticObjectChangeDetector,
    MapChangeDetector
)


class MockSemanticObject:
    """模拟语义物体"""
    def __init__(self, label, position, confidence=0.8):
        self.label = label
        self.world_position = position
        self.confidence = confidence


class TestIncrementalChangeDetector:
    """测试增量变化检测器"""

    def test_initialization(self):
        """测试初始化"""
        detector = IncrementalChangeDetector()
        assert detector._total_updates == 0
        assert detector._total_changes_detected == 0
        assert len(detector._dirty_flags) == 0

    def test_hash_computation(self):
        """测试哈希计算"""
        detector = IncrementalChangeDetector()

        obj1 = {"label": "door", "position": (5.0, 3.0)}
        obj2 = {"label": "door", "position": (5.0, 3.0)}
        obj3 = {"label": "door", "position": (6.0, 3.0)}

        hash1 = detector.compute_hash(obj1)
        hash2 = detector.compute_hash(obj2)
        hash3 = detector.compute_hash(obj3)

        # 相同对象应该产生相同哈希
        assert hash1 == hash2
        # 不同对象应该产生不同哈希
        assert hash1 != hash3
        # 哈希应该是32位MD5字符串
        assert len(hash1) == 32

    def test_update_new_object(self):
        """测试更新新对象"""
        detector = IncrementalChangeDetector()

        obj = {"label": "door", "position": (5.0, 3.0)}
        changed = detector.update_object("obj1", obj)

        # 新对象应该被标记为变化
        assert changed is True
        assert "obj1" in detector.get_changed_objects()
        assert detector._total_updates == 1
        assert detector._total_changes_detected == 1

    def test_update_unchanged_object(self):
        """测试更新未变化对象"""
        detector = IncrementalChangeDetector()

        obj = {"label": "door", "position": (5.0, 3.0)}
        detector.update_object("obj1", obj)

        # 清空脏标记
        detector.clear_dirty_flags()

        # 更新相同对象
        changed = detector.update_object("obj1", obj)

        # 应该没有变化
        assert changed is False
        assert len(detector.get_changed_objects()) == 0
        assert detector._total_updates == 2

    def test_update_changed_object(self):
        """测试更新变化对象"""
        detector = IncrementalChangeDetector()

        obj1 = {"label": "door", "position": (5.0, 3.0)}
        obj2 = {"label": "door", "position": (6.0, 3.0)}

        detector.update_object("obj1", obj1)
        detector.clear_dirty_flags()

        # 更新变化的对象
        changed = detector.update_object("obj1", obj2)

        # 应该检测到变化
        assert changed is True
        assert "obj1" in detector.get_changed_objects()

    def test_remove_object(self):
        """测试移除对象"""
        detector = IncrementalChangeDetector()

        obj = {"label": "door", "position": (5.0, 3.0)}
        detector.update_object("obj1", obj)

        # 移除对象
        detector.remove_object("obj1")

        assert "obj1" not in detector._object_hashes
        assert "obj1" not in detector.get_changed_objects()

    def test_clear_dirty_flags(self):
        """测试清空脏标记"""
        detector = IncrementalChangeDetector()

        obj = {"label": "door", "position": (5.0, 3.0)}
        detector.update_object("obj1", obj)

        assert len(detector.get_changed_objects()) == 1

        # 清空脏标记
        detector.clear_dirty_flags()

        assert len(detector.get_changed_objects()) == 0

    def test_get_statistics(self):
        """测试获取统计信息"""
        detector = IncrementalChangeDetector()

        obj1 = {"label": "door", "position": (5.0, 3.0)}
        obj2 = {"label": "door", "position": (6.0, 3.0)}

        detector.update_object("obj1", obj1)
        detector.update_object("obj2", obj2)
        detector.update_object("obj1", obj2)  # 变化

        stats = detector.get_statistics()

        assert stats["total_updates"] == 3
        assert stats["total_changes_detected"] == 3  # obj1(新) + obj2(新) + obj1(变化)
        assert stats["change_rate"] == 1.0
        assert stats["active_objects"] == 2


class TestSemanticObjectChangeDetector:
    """测试语义物体变化检测器"""

    def test_semantic_hash_computation(self):
        """测试语义物体哈希计算"""
        detector = SemanticObjectChangeDetector()

        obj1 = MockSemanticObject("door", (5.0, 3.0), 0.8)
        obj2 = MockSemanticObject("door", (5.0, 3.0), 0.8)
        obj3 = MockSemanticObject("door", (6.0, 3.0), 0.8)

        hash1 = detector.compute_hash(obj1)
        hash2 = detector.compute_hash(obj2)
        hash3 = detector.compute_hash(obj3)

        # 相同对象应该产生相同哈希
        assert hash1 == hash2
        # 不同位置应该产生不同哈希
        assert hash1 != hash3

    def test_batch_update_new_objects(self):
        """测试批量更新新对象"""
        detector = SemanticObjectChangeDetector()

        semantic_objects = {
            "obj1": MockSemanticObject("door", (5.0, 3.0)),
            "obj2": MockSemanticObject("person", (7.0, 2.0)),
        }

        new, changed, removed = detector.update_semantic_objects(semantic_objects)

        assert len(new) == 2
        assert "obj1" in new
        assert "obj2" in new
        assert len(changed) == 0
        assert len(removed) == 0

    def test_batch_update_unchanged(self):
        """测试批量更新无变化"""
        detector = SemanticObjectChangeDetector()

        semantic_objects = {
            "obj1": MockSemanticObject("door", (5.0, 3.0)),
        }

        # 第一次更新
        detector.update_semantic_objects(semantic_objects)

        # 第二次更新（无变化）
        new, changed, removed = detector.update_semantic_objects(semantic_objects)

        assert len(new) == 0
        assert len(changed) == 0
        assert len(removed) == 0

    def test_batch_update_changed_objects(self):
        """测试批量更新变化对象"""
        detector = SemanticObjectChangeDetector()

        semantic_objects_v1 = {
            "obj1": MockSemanticObject("door", (5.0, 3.0)),
        }

        semantic_objects_v2 = {
            "obj1": MockSemanticObject("door", (6.0, 3.0)),  # 位置变化
        }

        # 第一次更新
        detector.update_semantic_objects(semantic_objects_v1)

        # 第二次更新（有变化）
        new, changed, removed = detector.update_semantic_objects(semantic_objects_v2)

        assert len(new) == 0
        assert len(changed) == 1
        assert "obj1" in changed
        assert len(removed) == 0

    def test_batch_update_removed_objects(self):
        """测试批量更新移除对象"""
        detector = SemanticObjectChangeDetector()

        semantic_objects_v1 = {
            "obj1": MockSemanticObject("door", (5.0, 3.0)),
            "obj2": MockSemanticObject("person", (7.0, 2.0)),
        }

        semantic_objects_v2 = {
            "obj1": MockSemanticObject("door", (5.0, 3.0)),
            # obj2 被移除
        }

        # 第一次更新
        detector.update_semantic_objects(semantic_objects_v1)

        # 第二次更新
        new, changed, removed = detector.update_semantic_objects(semantic_objects_v2)

        assert len(new) == 0
        assert len(changed) == 0
        assert len(removed) == 1
        assert "obj2" in removed


class TestMapChangeDetector:
    """测试地图变化检测器"""

    def test_initial_map_detection(self):
        """测试初始地图检测"""
        detector = MapChangeDetector()

        map1 = np.zeros((10, 10), dtype=np.int8)
        has_changed, ratio = detector.detect_map_changes(map1)

        # 初始地图应该被标记为变化
        assert has_changed is True
        assert ratio == 1.0

    def test_unchanged_map_detection(self):
        """测试未变化地图检测"""
        detector = MapChangeDetector()

        map1 = np.zeros((10, 10), dtype=np.int8)
        map2 = np.zeros((10, 10), dtype=np.int8)

        detector.detect_map_changes(map1)
        has_changed, ratio = detector.detect_map_changes(map2)

        # 相同地图应该无变化
        assert has_changed is False
        assert ratio == 0.0

    def test_changed_map_detection(self):
        """测试变化地图检测"""
        detector = MapChangeDetector()

        map1 = np.zeros((10, 10), dtype=np.int8)
        map2 = np.zeros((10, 10), dtype=np.int8)
        map2[5, 5] = 100  # 修改一个栅格

        detector.detect_map_changes(map1)
        has_changed, ratio = detector.detect_map_changes(map2)

        # 应该检测到变化
        assert has_changed is True
        assert ratio == 0.01  # 1/100

    def test_below_threshold_change(self):
        """测试低于阈值的变化"""
        detector = MapChangeDetector()

        map1 = np.zeros((10, 10), dtype=np.int8)
        map2 = np.zeros((10, 10), dtype=np.int8)
        map2[5, 5] = 100  # 1% 变化

        detector.detect_map_changes(map1)
        has_changed, ratio = detector.detect_map_changes(map2, threshold=0.05)

        # 低于阈值，应该不触发变化
        assert has_changed is False
        assert ratio == 0.01

    def test_shape_change_detection(self):
        """测试尺寸变化检测"""
        detector = MapChangeDetector()

        map1 = np.zeros((10, 10), dtype=np.int8)
        map2 = np.zeros((20, 20), dtype=np.int8)

        detector.detect_map_changes(map1)
        has_changed, ratio = detector.detect_map_changes(map2)

        # 尺寸不同应该触发完全变化
        assert has_changed is True
        assert ratio == 1.0

    def test_get_map_diff(self):
        """测试获取地图差异"""
        detector = MapChangeDetector()

        map1 = np.zeros((10, 10), dtype=np.int8)
        map2 = np.zeros((10, 10), dtype=np.int8)
        map2[5, 5] = 100  # 新增占据
        map2[3, 3] = -1   # 改为未知

        detector.detect_map_changes(map1)
        diff = detector.get_map_diff(map2)

        assert diff is not None
        added, removed, changed = diff

        # (5, 5) 应该是新增的占据
        assert added[5, 5] is True
        # 没有被移除的占据
        assert not np.any(removed)
        # (3, 3) 应该是变化的
        assert changed[3, 3] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
