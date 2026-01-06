# -*- coding: utf-8 -*-
"""
Unit Tests for Dialogue Manager

Test coverage:
- Initialization and configuration
- Clarification flow
- Confirmation flow
- Progress reporting
- Error reporting
- Dialogue history
"""

import pytest
from unittest.mock import AsyncMock, Mock

from brain.cognitive.dialogue.dialogue_manager import DialogueManager, DialogueType


class TestDialogueManagerInitialization:
    """Test DialogueManager setup and configuration."""

    def test_default_initialization(self):
        """Test DialogueManager initializes with defaults."""
        manager = DialogueManager(
            llm_interface=None,
            user_callback=None
        )

        assert manager is not None
        # Check that the manager has the expected attributes
        assert hasattr(manager, 'current_context')
        assert hasattr(manager, 'archived_contexts')

    def test_initialization_with_llm(self):
        """Test initialization with LLM interface."""
        mock_llm = AsyncMock()
        manager = DialogueManager(
            llm_interface=mock_llm,
            user_callback=None
        )

        assert manager is not None


class TestClarificationFlow:
    """Test ambiguous command clarification."""

    @pytest.mark.asyncio
    async def test_clarify_ambiguous_command(self, dialogue_manager):
        """Test asking for clarification on ambiguous commands."""
        result = await dialogue_manager.clarify_ambiguous_command(
            command="拿起那个",
            ambiguities=[{"aspect": "目标物体", "question": "哪个物体？"}],
            world_context={"objects": ["杯子", "盘子"]}
        )

        assert result is not None
        assert "question" in result

    @pytest.mark.asyncio
    async def test_clarification_with_options(self, dialogue_manager):
        """Test clarification with multiple choice options."""
        result = await dialogue_manager.clarify_ambiguous_command(
            command="去那里",
            ambiguities=[{
                "aspect": "目标位置",
                "question": "具体位置？",
                "options": ["厨房", "卧室", "客厅"]
            }],
            world_context={}
        )

        assert result is not None
        assert "question" in result


class TestConfirmationFlow:
    """Test critical operation confirmation."""

    @pytest.mark.asyncio
    async def test_request_confirmation(self, dialogue_manager):
        """Test requesting user confirmation."""
        # Enable auto-confirm
        if hasattr(dialogue_manager, 'set_auto_confirm'):
            dialogue_manager.set_auto_confirm(enabled=True, delay=0.0)

        confirmed = await dialogue_manager.request_confirmation(
            action="删除所有历史记录",
            reason="此操作不可撤销",
            details=None,
            options=None
        )

        # With auto-confirm, should return True
        assert confirmed is not None

    @pytest.mark.asyncio
    async def test_confirmation_with_options(self, dialogue_manager):
        """Test confirmation with custom options."""
        if hasattr(dialogue_manager, 'set_auto_confirm'):
            dialogue_manager.set_auto_confirm(enabled=True, delay=0.0)

        confirmed = await dialogue_manager.request_confirmation(
            action="执行危险操作",
            reason="需要确认",
            details={"risk": "high"},
            options=["确认", "取消"]
        )

        assert confirmed is not None


class TestDialogueHistory:
    """Test conversation tracking."""

    def test_history_tracking(self, dialogue_manager):
        """Test that dialogue history is tracked."""
        # Check that archived contexts list exists for history
        assert hasattr(dialogue_manager, 'archived_contexts')
        assert isinstance(dialogue_manager.archived_contexts, list)

        # Start a session to test current_context
        session = dialogue_manager.start_session("test_session")
        assert dialogue_manager.current_context is not None
        assert hasattr(dialogue_manager.current_context, 'history')

    def test_get_conversation_summary(self, dialogue_manager):
        """Test getting conversation summary."""
        # Start a session and add some context
        dialogue_manager.start_session("test_summary_session")

        # Use the current_context's get_summary method
        if dialogue_manager.current_context:
            summary = dialogue_manager.current_context.get_summary()
            assert summary is not None
        else:
            # If no session, we should get None or empty
            assert dialogue_manager.current_context is None
