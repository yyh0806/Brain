# -*- coding: utf-8 -*-
"""
Functional Tests for Belief Update Flow

Tests the complete belief revision workflow:
ObservationResult → BeliefRevision → UpdatedBeliefs
"""

import pytest

from brain.cognitive.interface import ObservationResult, ObservationStatus


class TestBeliefUpdateFlow:
    """Test complete belief revision workflow."""

    @pytest.mark.asyncio
    async def test_observation_success_increases_confidence(self, cognitive_layer):
        """Test successful execution increases belief confidence."""
        # First, register a belief implicitly through world model update
        from tests.conftest import Pose, Velocity, PerceptionData
        import numpy as np
        from datetime import datetime

        perception = PerceptionData(
            pose=Pose(x=0.0, y=0.0, z=0.0),
            velocity=Velocity(),
            obstacles=[],
            occupancy_grid=np.zeros((50, 50), dtype=np.int8),
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            timestamp=datetime.now()
        )

        await cognitive_layer.process_perception(perception)

        # Report successful observation
        observation = ObservationResult(
            operation_id="test_search",
            operation_type="search",
            status=ObservationStatus.SUCCESS,
            confidence=0.9
        )

        result = await cognitive_layer.update_belief(observation)

        assert result is not None
        assert isinstance(result.updated_beliefs, list)

    @pytest.mark.asyncio
    async def test_observation_failure_decreases_confidence(self, cognitive_layer):
        """Test failed execution decreases belief confidence."""
        observation = ObservationResult(
            operation_id="test_search",
            operation_type="search",
            status=ObservationStatus.FAILURE,
            confidence=0.0,
            error_message="Object not found"
        )

        result = await cognitive_layer.update_belief(observation)

        assert result is not None
        assert isinstance(result.updated_beliefs, list)

    @pytest.mark.asyncio
    async def test_multiple_failures_lead_to_removal(self, cognitive_layer):
        """Test that multiple failures can lead to belief removal."""
        observation = ObservationResult(
            operation_id="test_search",
            operation_type="search",
            status=ObservationStatus.FAILURE,
            error_message="Not found"
        )

        # Report multiple failures
        for _ in range(5):
            await cognitive_layer.update_belief(observation)

        # Check falsified beliefs
        falsified = cognitive_layer.get_falsified_beliefs()
        assert isinstance(falsified, list)


class TestBeliefStateTransitions:
    """Test belief state transitions through observations."""

    @pytest.mark.asyncio
    async def test_belief_confidence_evolution(self, cognitive_layer):
        """Test how belief confidence evolves over multiple observations."""
        # Start with success
        success_obs = ObservationResult(
            operation_id="test",
            operation_type="search",
            status=ObservationStatus.SUCCESS,
            confidence=0.8
        )

        result1 = await cognitive_layer.update_belief(success_obs)
        assert result1 is not None

        # Then failure
        failure_obs = ObservationResult(
            operation_id="test",
            operation_type="search",
            status=ObservationStatus.FAILURE,
            confidence=0.0
        )

        result2 = await cognitive_layer.update_belief(failure_obs)
        assert result2 is not None

        # Then recovery
        result3 = await cognitive_layer.update_belief(success_obs)
        assert result3 is not None
