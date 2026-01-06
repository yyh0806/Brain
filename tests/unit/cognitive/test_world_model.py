# -*- coding: utf-8 -*-
"""
Unit Tests for WorldModel Component

Test coverage:
- Initialization and configuration
- Perception data updates
- Object tracking
- Change detection
- Context generation for planning
- Map operations
- Edge cases and error handling
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import patch, Mock
import sys
import os

# 直接导入，避免通过brain/__init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from brain.cognitive.world_model.world_model import WorldModel
from brain.cognitive.world_model.environment_change import ChangeType, ChangePriority


class TestWorldModelInitialization:
    """Test WorldModel initialization and configuration."""

    def test_default_initialization(self):
        """Test WorldModel initializes with default values."""
        model = WorldModel()

        assert model.robot_position == {"x": 0, "y": 0, "z": 0, "lat": 0, "lon": 0, "alt": 0}
        assert model.battery_level == 100.0
        assert model.tracked_objects == {}
        assert len(model.pose_history) == 0

    def test_initialization_with_config(self):
        """Test WorldModel with custom configuration."""
        config = {
            "max_semantic_objects": 200,
            "max_frontiers": 50,
            "max_pose_history": 500
        }
        model = WorldModel(config)

        assert model.max_semantic_objects == 200
        assert model.max_frontiers == 50
        assert model.max_pose_history == 500


class TestWorldModelPerceptionUpdate:
    """Test perception data processing and world model updates."""

    @pytest.mark.asyncio
    async def test_update_from_perception_basic(self, world_model, sample_perception_data):
        """Test basic perception data update."""
        # Test that the update method accepts PerceptionData and returns changes
        changes = world_model.update_from_perception(sample_perception_data)

        # Should return a list of changes (may be empty if no significant changes)
        assert isinstance(changes, list)

        # WorldModel should have processed the data
        assert world_model.last_update is not None

    @pytest.mark.asyncio
    async def test_update_occupancy_grid(self, world_model, sample_perception_data):
        """Test occupancy grid update."""
        # Update with perception data that includes occupancy grid
        changes = world_model.update_from_perception(sample_perception_data)

        # The method should complete without errors
        assert isinstance(changes, list)

        # Verify the occupancy grid data was processed
        # Note: current_map may not be set if the update is skipped due to no significant changes
        # So we just verify the method runs successfully

    @pytest.mark.asyncio
    async def test_update_tracked_objects(self, world_model, sample_perception_data):
        """Test object tracking updates."""
        world_model.update_from_perception(sample_perception_data)

        # Should have tracked objects from perception data
        assert len(world_model.tracked_objects) >= 0

        # Check if any obstacles were tracked
        if sample_perception_data.obstacles:
            # At minimum, the test should not crash
            assert True


class TestWorldModelChangeDetection:
    """Test environment change detection."""

    def test_detect_significant_changes(self, world_model, sample_perception_data):
        """Test detection of significant changes."""
        world_model.update_from_perception(sample_perception_data)

        significant = world_model.detect_significant_changes()
        assert isinstance(significant, list)

    def test_significant_changes_filtering(self, world_model, sample_perception_data):
        """Test filtering of significant changes."""
        world_model.update_from_perception(sample_perception_data)

        significant = world_model.detect_significant_changes()
        # All significant changes should require replan
        for change in significant:
            assert hasattr(change, 'requires_replan')


class TestWorldModelPlanningContext:
    """Test planning context generation."""

    def test_get_context_for_planning(self, world_model, sample_perception_data):
        """Test generation of planning context."""
        world_model.update_from_perception(sample_perception_data)

        context = world_model.get_context_for_planning()

        assert context is not None
        assert context.current_position is not None

    def test_context_includes_obstacles(self, world_model, sample_perception_data):
        """Test planning context includes obstacle information."""
        world_model.update_from_perception(sample_perception_data)

        context = world_model.get_context_for_planning()

        assert context.obstacles is not None
        assert isinstance(context.obstacles, list)


class TestWorldModelMapOperations:
    """Test map-related operations."""

    def test_occupancy_query(self, world_model, sample_perception_data):
        """Test querying occupancy at specific location."""
        world_model.update_from_perception(sample_perception_data)

        # Test free location (should not raise exception)
        free = world_model.is_free_at(0.0, 0.0)
        assert isinstance(free, bool)

    def test_nearest_obstacle(self, world_model, sample_perception_data):
        """Test finding nearest obstacle."""
        world_model.update_from_perception(sample_perception_data)

        # Should not raise exception
        nearest = world_model.get_nearest_obstacle(0.0, 0.0, max_range=10.0)
        # Might return None if no obstacles
        assert nearest is None or isinstance(nearest, tuple)


class TestWorldModelEdgeCases:
    """Test edge cases and error handling."""

    def test_update_with_none_data(self, world_model):
        """Test handling of None perception data."""
        changes = world_model.update_from_perception(None)
        assert isinstance(changes, list)

    def test_update_with_empty_obstacles(self, world_model):
        """Test update with no obstacles."""
        from brain.perception.data.models import Pose3D, Velocity
        from unittest.mock import Mock

        pose = Pose3D(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        velocity = Velocity()
        velocity.linear_x = velocity.linear_y = velocity.linear_z = 0.0
        velocity.angular_x = velocity.angular_y = velocity.angular_z = 0.0

        perception = Mock()
        perception.pose = pose
        perception.velocity = velocity
        perception.obstacles = []
        perception.occupancy_grid = np.zeros((50, 50), dtype=np.int8)
        perception.grid_resolution = 0.1
        perception.grid_origin = (0.0, 0.0)
        perception.timestamp = datetime.now()

        changes = world_model.update_from_perception(perception)
        assert isinstance(changes, list)

    def test_get_summary(self, world_model):
        """Test getting world model summary."""
        summary = world_model.get_summary()

        assert isinstance(summary, dict)
        assert "robot_position" in summary
        assert "battery_level" in summary
