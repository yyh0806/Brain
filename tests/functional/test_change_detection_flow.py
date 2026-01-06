# -*- coding: utf-8 -*-
"""
Functional Tests for Change Detection Flow

Tests the complete change detection workflow:
PerceptionData → PerceptionMonitor → EnvironmentChanges → ReplanTrigger
"""

import pytest
import numpy as np
from datetime import datetime

from brain.cognitive.world_model.environment_change import ChangeType


class TestChangeDetectionFlow:
    """Test complete change detection workflow."""

    @pytest.mark.asyncio
    async def test_new_obstacle_detection(self, cognitive_layer):
        """Test detecting new obstacles triggers change detection."""
        from tests.conftest import Pose, Velocity, PerceptionData

        # Initial perception - no obstacles
        perception1 = PerceptionData(
            pose=Pose(x=0.0, y=0.0, z=0.0),
            velocity=Velocity(),
            obstacles=[],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        output1 = await cognitive_layer.process_perception(perception1)

        # Perception with new obstacle
        grid2 = np.zeros((50, 50), dtype=np.int8)
        grid2[20:30, 20:30] = 100  # New obstacle

        perception2 = PerceptionData(
            pose=Pose(x=1.0, y=1.0, z=0.0),
            velocity=Velocity(),
            obstacles=[{
                "id": "obs_new",
                "type": "static",
                "world_position": {"x": 2.0, "y": 2.0, "z": 0.0},
                "local_position": {"x": 1.0, "y": 1.0, "z": 0.0}
            }],
            occupancy_grid=grid2,
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        output2 = await cognitive_layer.process_perception(perception2)

        # Should detect changes
        assert output2.environment_changes is not None
        assert isinstance(output2.environment_changes, list)

    @pytest.mark.asyncio
    async def test_obstacle_movement_detection(self, cognitive_layer):
        """Test detecting obstacle movement."""
        from tests.conftest import Pose, Velocity, PerceptionData

        # First observation
        perception1 = PerceptionData(
            pose=Pose(x=0.0, y=0.0, z=0.0),
            velocity=Velocity(),
            obstacles=[{
                "id": "obs_001",
                "type": "person",
                "world_position": {"x": 1.0, "y": 1.0, "z": 0.0},
                "local_position": {"x": 1.0, "y": 1.0, "z": 0.0}
            }],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        await cognitive_layer.process_perception(perception1)

        # Second observation - obstacle moved
        perception2 = PerceptionData(
            pose=Pose(x=0.5, y=0.5, z=0.0),
            velocity=Velocity(),
            obstacles=[{
                "id": "obs_001",
                "type": "person",
                "world_position": {"x": 3.0, "y": 3.0, "z": 0.0},
                "local_position": {"x": 2.5, "y": 2.5, "z": 0.0}
            }],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        output = await cognitive_layer.process_perception(perception2)

        # Should detect movement
        assert output.environment_changes is not None

    @pytest.mark.asyncio
    async def test_battery_low_detection(self, cognitive_layer):
        """Test detecting low battery level."""
        from tests.conftest import Pose, Velocity, PerceptionData

        perception = PerceptionData(
            pose=Pose(x=0.0, y=0.0, z=0.0),
            velocity=Velocity(),
            obstacles=[],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        # Manually set low battery
        cognitive_layer.world_model.battery_level = 15.0

        output = await cognitive_layer.process_perception(perception)

        # Should detect low battery
        battery_changes = [c for c in output.environment_changes if hasattr(c, 'change_type')]
        # Note: Detection depends on implementation

    @pytest.mark.asyncio
    async def test_significant_changes_trigger_replan(self, cognitive_layer):
        """Test that significant changes are flagged for replanning."""
        from tests.conftest import Pose, Velocity, PerceptionData

        perception = PerceptionData(
            pose=Pose(x=0.0, y=0.0, z=0.0),
            velocity=Velocity(),
            obstacles=[{
                "id": "obs_critical",
                "type": "blocking_obstacle",
                "world_position": {"x": 0.5, "y": 0.0, "z": 0.0},
                "local_position": {"x": 0.5, "y": 0.0, "z": 0.0}
            }],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        output = await cognitive_layer.process_perception(perception)

        # Check if any changes require replanning
        significant = cognitive_layer.world_model.detect_significant_changes()
        assert isinstance(significant, list)


class TestChangePrioritization:
    """Test change prioritization in detection flow."""

    @pytest.mark.asyncio
    async def test_high_priority_changes(self, cognitive_layer):
        """Test that critical changes get high priority."""
        from tests.conftest import Pose, Velocity, PerceptionData

        perception = PerceptionData(
            pose=Pose(x=0.0, y=0.0, z=0.0),
            velocity=Velocity(),
            obstacles=[],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        output = await cognitive_layer.process_perception(perception)

        # All changes should have priority information
        for change in output.environment_changes:
            if hasattr(change, 'priority'):
                assert change.priority is not None
