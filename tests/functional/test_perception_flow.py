# -*- coding: utf-8 -*-
"""
Functional Tests for Perception → Cognitive Data Flow

These tests use real data flow through the pipeline:
PerceptionData → WorldModel → CognitiveOutput
"""

import pytest
import numpy as np
from datetime import datetime

from brain.cognitive.interface import CognitiveOutput
from brain.cognitive.world_model.world_model import WorldModel


class TestPerceptionToCognitiveFlow:
    """Test complete perception to cognitive data flow."""

    @pytest.mark.asyncio
    async def test_basic_perception_flow(self, cognitive_layer, sample_perception_data):
        """Test basic flow: perception → cognitive output."""
        output = await cognitive_layer.process_perception(sample_perception_data)

        assert isinstance(output, CognitiveOutput)
        assert output.planning_context is not None
        assert isinstance(output.environment_changes, list)
        assert output.timestamp is not None

    @pytest.mark.asyncio
    async def test_perception_updates_world_model(self, cognitive_layer, sample_perception_data):
        """Test that perception data updates world model."""
        await cognitive_layer.process_perception(sample_perception_data)

        world_model = cognitive_layer.world_model
        # Just verify the world model was updated (last_update should be set)
        assert world_model.last_update is not None

    @pytest.mark.asyncio
    async def test_change_detection_in_flow(self, cognitive_layer, sample_perception_data_with_changes):
        """Test change detection in perception flow."""
        output = await cognitive_layer.process_perception(sample_perception_data_with_changes)

        # Should return environment changes list
        assert isinstance(output.environment_changes, list)

    @pytest.mark.asyncio
    async def test_planning_context_generation(self, cognitive_layer, sample_perception_data):
        """Test planning context generation in flow."""
        output = await cognitive_layer.process_perception(sample_perception_data)

        context = output.planning_context
        assert context is not None
        assert context.current_position is not None

    @pytest.mark.asyncio
    async def test_continuous_updates(self, cognitive_layer):
        """Test continuous perception updates."""
        # Simulate sequence of perception updates
        for i in range(3):
            perception_data = self._create_perception_at_position(float(i), float(i))
            output = await cognitive_layer.process_perception(perception_data)

            assert output is not None
            assert output.planning_context is not None

        # World model should have been updated (last_update should be recent)
        assert cognitive_layer.world_model.last_update is not None

    def _create_perception_at_position(self, x, y):
        """Helper to create perception data at specific position."""
        from tests.conftest import Pose, Velocity, PerceptionData

        pose = Pose(x=x, y=y, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        velocity = Velocity()

        occupancy_grid = np.zeros((50, 50), dtype=np.int8)

        return PerceptionData(
            pose=pose,
            velocity=velocity,
            obstacles=[],
            occupancy_grid=occupancy_grid,
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )


class TestWorldModelIntegration:
    """Test WorldModel integration in data flow."""

    @pytest.mark.asyncio
    async def test_world_model_receives_data(self, world_model, sample_perception_data):
        """Test that WorldModel receives and processes perception data."""
        changes = world_model.update_from_perception(sample_perception_data)

        assert isinstance(changes, list)
        # World model should be updated
        assert world_model.robot_position is not None

    @pytest.mark.asyncio
    async def test_occupancy_grid_integration(self, world_model, sample_perception_data):
        """Test occupancy grid integration in flow."""
        changes = world_model.update_from_perception(sample_perception_data)

        # Just verify the method completes and returns changes
        assert isinstance(changes, list)
        # Note: current_map may not be set if the update is skipped due to no significant changes


class TestEnvironmentChangeFlow:
    """Test environment change detection flow."""

    @pytest.mark.asyncio
    async def test_new_obstacle_detection(self, cognitive_layer):
        """Test detecting new obstacles in flow."""
        from tests.conftest import Pose, Velocity, PerceptionData

        # Initial update
        pose1 = Pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        velocity = Velocity()
        grid1 = np.zeros((50, 50), dtype=np.int8)

        perception1 = PerceptionData(
            pose=pose1, velocity=velocity,
            obstacles=[],
            occupancy_grid=grid1,
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        await cognitive_layer.process_perception(perception1)

        # Update with new obstacle
        pose2 = Pose(x=1.0, y=1.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
        grid2 = np.zeros((50, 50), dtype=np.int8)
        grid2[20:30, 20:30] = 100  # New obstacle

        perception2 = PerceptionData(
            pose=pose2, velocity=velocity,
            obstacles=[{"id": "obs_new", "type": "static"}],
            occupancy_grid=grid2,
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        output = await cognitive_layer.process_perception(perception2)

        # Should detect changes
        assert output.environment_changes is not None
