# -*- coding: utf-8 -*-
"""
Functional Tests for Reasoning Flow

Tests the complete reasoning workflow:
Query + Context → CoTEngine → ReasoningResult
"""

import pytest

from brain.cognitive.reasoning.cot_engine import ReasoningMode


class TestReasoningFlow:
    """Test complete reasoning workflow."""

    @pytest.mark.asyncio
    async def test_simple_reasoning_flow(self, cognitive_layer):
        """Test reasoning flow for simple queries."""
        result = await cognitive_layer.reason(
            query="前进",
            context={"obstacles": []},
            mode=ReasoningMode.PLANNING
        )

        assert result is not None
        assert result.decision is not None
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_complex_reasoning_flow(self, cognitive_layer):
        """Test reasoning flow for complex tasks."""
        context = {
            "obstacles": [
                {"type": "static", "position": {"x": 2, "y": 1}},
                {"type": "person", "position": {"x": 3, "y": 2}}
            ],
            "current_position": {"x": 0, "y": 0}
        }

        result = await cognitive_layer.reason(
            query="规划一条安全的路径到目标",
            context=context,
            mode=ReasoningMode.PLANNING
        )

        assert result is not None
        assert result.decision is not None
        assert len(result.chain) >= 1

    @pytest.mark.asyncio
    async def test_reasoning_with_world_context(self, cognitive_layer, sample_perception_data):
        """Test reasoning using world model context."""
        # Update world model first
        await cognitive_layer.process_perception(sample_perception_data)

        # Get planning context
        planning_context = cognitive_layer.get_planning_context()

        # Use planning context in reasoning
        result = await cognitive_layer.reason(
            query="根据当前情况决策",
            context={
                "obstacles": planning_context.obstacles,
                "current_position": planning_context.current_position
            },
            mode=ReasoningMode.PLANNING
        )

        assert result is not None


class TestReasoningModes:
    """Test different reasoning modes."""

    @pytest.mark.asyncio
    async def test_planning_mode(self, cognitive_layer):
        """Test reasoning in planning mode."""
        result = await cognitive_layer.reason(
            query="规划路径",
            context={},
            mode=ReasoningMode.PLANNING
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_replanning_mode(self, cognitive_layer):
        """Test reasoning in replanning mode."""
        result = await cognitive_layer.reason(
            query="重新规划",
            context={"obstacles": [{"type": "dynamic"}]},
            mode=ReasoningMode.REPLANNING
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_exception_handling_mode(self, cognitive_layer):
        """Test reasoning in exception handling mode."""
        result = await cognitive_layer.reason(
            query="处理异常",
            context={"error": "path_blocked"},
            mode=ReasoningMode.EXCEPTION_HANDLING
        )

        assert result is not None
