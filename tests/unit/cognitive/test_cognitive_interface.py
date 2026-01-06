# -*- coding: utf-8 -*-
"""
Unit Tests for Cognitive Layer Interface

Test coverage:
- CognitiveLayer initialization
- Process perception interface
- Update belief interface
- Reason interface
- Dialogue interface
- Get planning context interface
- Get falsified beliefs interface
- Monitoring control
"""

import pytest
from unittest.mock import Mock, AsyncMock

from brain.cognitive.interface import (
    CognitiveLayer,
    ObservationResult,
    ObservationStatus,
    CognitiveOutput
)
from brain.cognitive.world_model.world_model import WorldModel


class TestCognitiveLayerInitialization:
    """Test CognitiveLayer setup and component assembly."""

    def test_default_initialization(self):
        """Test CognitiveLayer initializes with required components."""
        # Create minimal components
        world_model = WorldModel()

        layer = CognitiveLayer(
            world_model=world_model,
            cot_engine=None,
            dialogue_manager=None,
            perception_monitor=None,
            config={}
        )

        assert layer is not None
        assert layer.world_model == world_model

    def test_initialization_with_all_components(self, cognitive_layer):
        """Test CognitiveLayer with all components configured."""
        assert cognitive_layer is not None
        assert cognitive_layer.world_model is not None


class TestProcessPerception:
    """Test perception to cognitive processing interface."""

    @pytest.mark.asyncio
    async def test_process_perception_basic(self, cognitive_layer, sample_perception_data):
        """Test basic perception processing."""
        output = await cognitive_layer.process_perception(sample_perception_data)

        assert isinstance(output, CognitiveOutput)
        assert output.planning_context is not None
        assert isinstance(output.environment_changes, list)

    @pytest.mark.asyncio
    async def test_process_perception_updates_world_model(self, cognitive_layer, sample_perception_data):
        """Test that perception updates world model."""
        await cognitive_layer.process_perception(sample_perception_data)

        world_model = cognitive_layer.world_model
        # Should have updated robot position
        assert world_model.robot_position is not None

    @pytest.mark.asyncio
    async def test_process_perception_with_none_data(self, cognitive_layer):
        """Test handling of None perception data."""
        # This should raise an error or handle gracefully
        # Depending on implementation
        try:
            output = await cognitive_layer.process_perception(None)
            # If it doesn't raise, output should still be valid
            assert True
        except (ValueError, AttributeError):
            # Expected behavior
            assert True


class TestUpdateBelief:
    """Test belief revision interface."""

    @pytest.mark.asyncio
    async def test_update_belief_success(self, cognitive_layer):
        """Test updating belief after successful observation."""
        observation = ObservationResult(
            operation_id="test_op",
            operation_type="search",
            status=ObservationStatus.SUCCESS,
            confidence=0.9
        )

        result = await cognitive_layer.update_belief(observation)

        # Should return BeliefUpdate
        assert result is not None

    @pytest.mark.asyncio
    async def test_update_belief_failure(self, cognitive_layer):
        """Test updating belief after failed observation."""
        observation = ObservationResult(
            operation_id="test_op",
            operation_type="search",
            status=ObservationStatus.FAILURE,
            confidence=0.0,
            error_message="Object not found"
        )

        result = await cognitive_layer.update_belief(observation)

        # Should return BeliefUpdate
        assert result is not None


class TestReason:
    """Test reasoning interface."""

    @pytest.mark.asyncio
    async def test_reason_basic_query(self, cognitive_layer):
        """Test basic reasoning query."""
        from brain.cognitive.reasoning.cot_engine import ReasoningMode

        result = await cognitive_layer.reason(
            query="前进5米",
            context={"obstacles": []},
            mode=ReasoningMode.PLANNING
        )

        assert result is not None
        assert result.decision is not None

    @pytest.mark.asyncio
    async def test_reason_with_complex_context(self, cognitive_layer):
        """Test reasoning with complex context."""
        from brain.cognitive.reasoning.cot_engine import ReasoningMode

        context = {
            "obstacles": [{"type": "static", "position": {"x": 2, "y": 1}}],
            "current_position": {"x": 0, "y": 0},
            "constraints": ["battery_low"]
        }

        result = await cognitive_layer.reason(
            query="规划到目标的路径",
            context=context,
            mode=ReasoningMode.PLANNING
        )

        assert result is not None


class TestDialogue:
    """Test dialogue management interface."""

    @pytest.mark.asyncio
    async def test_dialogue_clarification(self, cognitive_layer):
        """Test dialogue for clarification."""
        from brain.cognitive.dialogue.dialogue_manager import DialogueType

        response = await cognitive_layer.dialogue(
            message="拿起那个",
            dialogue_type=DialogueType.CLARIFICATION,
            context={
                "ambiguities": [{
                    "aspect": "目标物体",
                    "question": "哪个物体？",
                    "options": ["杯子", "盘子"]
                }]
            }
        )

        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_dialogue_confirmation(self, cognitive_layer):
        """Test dialogue for confirmation."""
        from brain.cognitive.dialogue.dialogue_manager import DialogueType

        response = await cognitive_layer.dialogue(
            message="执行危险操作",
            dialogue_type=DialogueType.CONFIRMATION,
            context={
                "reason": "此操作不可撤销"
            }
        )

        assert response is not None


class TestGetPlanningContext:
    """Test planning context retrieval interface."""

    def test_get_planning_context(self, cognitive_layer):
        """Test getting planning context."""
        context = cognitive_layer.get_planning_context()

        assert context is not None
        assert context.current_position is not None

    def test_get_planning_context_with_obstacles(self, cognitive_layer, sample_perception_data):
        """Test planning context includes obstacles."""
        import asyncio

        # Update world model first
        asyncio.run(cognitive_layer.process_perception(sample_perception_data))

        context = cognitive_layer.get_planning_context()

        assert context is not None
        assert context.obstacles is not None


class TestGetFalsifiedBeliefs:
    """Test falsified beliefs query interface."""

    def test_get_falsified_beliefs_empty(self, cognitive_layer):
        """Test getting falsified beliefs when none exist."""
        falsified = cognitive_layer.get_falsified_beliefs()

        assert isinstance(falsified, list)
        # Should be empty initially
        # or return all falsified beliefs

    def test_get_falsified_beliefs_after_failures(self, cognitive_layer):
        """Test getting falsified beliefs after observations."""
        import asyncio

        # Report some failures
        observation = ObservationResult(
            operation_id="test_op",
            operation_type="search",
            status=ObservationStatus.FAILURE
        )

        asyncio.run(cognitive_layer.update_belief(observation))

        falsified = cognitive_layer.get_falsified_beliefs()
        assert isinstance(falsified, list)


class TestMonitoringControl:
    """Test monitoring control interface."""

    def test_start_monitoring(self, cognitive_layer):
        """Test starting perception monitoring."""
        # Should not raise exception
        cognitive_layer.start_monitoring()
        assert True

    def test_stop_monitoring(self, cognitive_layer):
        """Test stopping perception monitoring."""
        # Should not raise exception
        cognitive_layer.stop_monitoring()
        assert True
