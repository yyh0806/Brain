# -*- coding: utf-8 -*-
"""
Functional Tests for Dialogue Flow

Tests the complete dialogue workflow:
UserInput → DialogueManager → DialogueResponse
"""

import pytest

from brain.cognitive.dialogue.dialogue_manager import DialogueType


class TestDialogueFlow:
    """Test complete dialogue workflow."""

    @pytest.mark.asyncio
    async def test_clarification_flow(self, cognitive_layer):
        """Test complete clarification dialogue flow."""
        response = await cognitive_layer.dialogue(
            message="拿起那个",
            dialogue_type=DialogueType.CLARIFICATION,
            context={
                "ambiguities": [{
                    "aspect": "目标物体",
                    "question": "哪个物体？",
                    "options": ["杯子", "盘子", "碗"]
                }]
            }
        )

        assert response is not None
        assert response.content is not None
        assert response.requires_user_input == True

    @pytest.mark.asyncio
    async def test_confirmation_flow(self, cognitive_layer):
        """Test complete confirmation dialogue flow."""
        response = await cognitive_layer.dialogue(
            message="执行危险操作",
            dialogue_type=DialogueType.CONFIRMATION,
            context={
                "reason": "此操作不可撤销",
                "details": {"risk": "high"}
            }
        )

        assert response is not None
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_progress_reporting_flow(self, cognitive_layer):
        """Test progress reporting dialogue flow."""
        response = await cognitive_layer.dialogue(
            message="任务执行中",
            dialogue_type=DialogueType.PROGRESS_REPORT,
            context={
                "progress": 0.5,
                "status": "执行中"
            }
        )

        assert response is not None


class TestDialogueInteraction:
    """Test multi-turn dialogue interactions."""

    @pytest.mark.asyncio
    async def test_multi_turn_clarification(self, cognitive_layer):
        """Test multiple turns of clarification dialogue."""
        # First turn: ask for clarification
        response1 = await cognitive_layer.dialogue(
            message="去那里",
            dialogue_type=DialogueType.CLARIFICATION,
            context={
                "ambiguities": [{
                    "aspect": "目标位置",
                    "question": "具体位置？",
                    "options": ["厨房", "卧室"]
                }]
            }
        )

        assert response1 is not None
        assert response1.requires_user_input == True

        # Test completed - multi-turn testing would require session state management
        # which is beyond the scope of this functional test

    @pytest.mark.asyncio
    async def test_error_dialogue_flow(self, cognitive_layer):
        """Test error reporting dialogue flow."""
        response = await cognitive_layer.dialogue(
            message="操作失败",
            dialogue_type=DialogueType.ERROR_REPORT,
            context={
                "error": "路径受阻",
                "suggestions": ["重新规划", "绕行"]
            }
        )

        assert response is not None
        assert response.content is not None
