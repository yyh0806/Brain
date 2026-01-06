# -*- coding: utf-8 -*-
"""
Unit Tests for Belief Revision System

Test coverage:
- Belief registration and initialization
- Operation failure handling
- Operation success handling
- Confidence decay over time
- Observation updates
- Belief removal
- Statistics and reporting
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from brain.cognitive.world_model.belief_revision import (
    BeliefRevisionPolicy, BeliefType, OperationType, BeliefEntry
)


class TestBeliefEntry:
    """Test BeliefEntry dataclass."""

    def test_create_belief_entry(self):
        """Test creating a belief entry."""
        belief = BeliefEntry(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test belief",
            confidence=0.8
        )

        assert belief.belief_id == "test_belief"
        assert belief.belief_type == BeliefType.OBJECT_LOCATION
        assert belief.confidence == 0.8
        assert belief.failure_count == 0
        assert belief.success_count == 0

    def test_belief_is_valid(self):
        """Test belief validity check."""
        belief = BeliefEntry(
            belief_id="test",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test",
            confidence=0.6,
            min_confidence=0.5  # Set higher threshold for this test
        )

        assert belief.is_valid() == True

        belief.confidence = 0.3  # Below min_confidence threshold
        assert belief.is_valid() == False

    def test_belief_should_remove(self):
        """Test belief removal threshold."""
        belief = BeliefEntry(
            belief_id="test",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test",
            confidence=0.5
        )

        # Should not remove at default threshold
        assert belief.should_remove(0.2) == False

        # Should remove if below threshold
        assert belief.should_remove(0.6) == True


class TestBeliefRegistration:
    """Test belief registration and initialization."""

    def test_register_new_belief(self, belief_revision_policy):
        """Test registering a new belief."""
        belief = belief_revision_policy.register_belief(
            belief_id="cup_in_kitchen",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="杯子在厨房",
            initial_confidence=0.7
        )

        assert belief is not None
        assert belief.belief_id == "cup_in_kitchen"
        assert belief.confidence == 0.7
        assert belief.failure_count == 0

    def test_register_duplicate_belief(self, belief_revision_policy):
        """Test registering duplicate belief updates existing."""
        belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Initial",
            initial_confidence=0.5
        )

        # Register same belief again
        updated_belief = belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Updated",
            initial_confidence=0.8
        )

        assert updated_belief.description == "Updated"
        assert updated_belief.confidence == 0.8

    def test_belief_types(self, belief_revision_policy):
        """Test different belief types."""
        types = [
            BeliefType.OBJECT_LOCATION,
            BeliefType.PATH_ACCESSIBLE,
            BeliefType.OBJECT_EXISTS,
            BeliefType.ENVIRONMENT_STATE
        ]

        for i, btype in enumerate(types):
            belief = belief_revision_policy.register_belief(
                belief_id=f"belief_{i}",
                belief_type=btype,
                description=f"Test {btype.value}",
                initial_confidence=0.7
            )
            assert belief.belief_type == btype


class TestOperationFailureHandling:
    """Test belief updates on operation failures."""

    def test_single_failure_reduces_confidence(self, belief_revision_policy):
        """Test that operation failure reduces confidence."""
        belief = belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test belief",
            initial_confidence=0.8
        )

        updated = belief_revision_policy.report_operation_failure(
            belief_id="test_belief",
            operation_type=OperationType.SEARCH
        )

        assert updated.confidence < 0.8
        assert updated.failure_count == 1

    def test_multiple_failures(self, belief_revision_policy):
        """Test multiple failures continue reducing confidence."""
        belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test belief",
            initial_confidence=0.9
        )

        for _ in range(3):
            belief_revision_policy.report_operation_failure(
                belief_id="test_belief",
                operation_type=OperationType.SEARCH
            )

        belief = belief_revision_policy.get_belief("test_belief")
        assert belief.failure_count == 3
        assert belief.confidence < 0.5  # Should be significantly lower

    def test_failure_with_error_message(self, belief_revision_policy):
        """Test failure reporting with error message."""
        belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test belief"
        )

        updated = belief_revision_policy.report_operation_failure(
            belief_id="test_belief",
            operation_type=OperationType.SEARCH,
            error="Object not found at location"
        )

        assert updated is not None
        assert updated.failure_count == 1


class TestOperationSuccessHandling:
    """Test belief updates on operation successes."""

    def test_success_increases_confidence(self, belief_revision_policy):
        """Test that operation success increases confidence."""
        belief = belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test belief",
            initial_confidence=0.6
        )

        updated = belief_revision_policy.report_operation_success(
            belief_id="test_belief",
            operation_type=OperationType.SEARCH
        )

        assert updated.confidence > 0.6
        assert updated.success_count == 1

    def test_multiple_successes(self, belief_revision_policy):
        """Test multiple successes increase confidence up to max."""
        belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test belief",
            initial_confidence=0.5
        )

        # Many successes should approach but not exceed max
        for _ in range(10):
            belief_revision_policy.report_operation_success(
                belief_id="test_belief",
                operation_type=OperationType.SEARCH
            )

        belief = belief_revision_policy.get_belief("test_belief")
        assert belief.success_count == 10
        assert belief.confidence <= 0.95  # Max confidence


class TestBeliefDecay:
    """Test confidence decay over time."""

    def test_decay_old_beliefs(self, belief_revision_policy):
        """Test that old beliefs decay in confidence."""
        belief = belief_revision_policy.register_belief(
            belief_id="old_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Old belief",
            initial_confidence=0.8
        )

        # Simulate time passing
        with patch('brain.cognitive.world_model.belief_revision.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=400)

            updated_count = belief_revision_policy.update_belief_decay()

        assert updated_count >= 1

        updated_belief = belief_revision_policy.get_belief("old_belief")
        assert updated_belief.confidence < 0.8

    def test_observation_prevents_decay(self, belief_revision_policy):
        """Test that recent observation prevents decay."""
        belief = belief_revision_policy.register_belief(
            belief_id="active_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Active belief",
            initial_confidence=0.8
        )

        # Update observation recently
        belief_revision_policy.update_observation("active_belief", confidence=0.9)

        # Try decay - should not decay much
        with patch('brain.cognitive.world_model.belief_revision.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(seconds=100)
            belief_revision_policy.update_belief_decay()

        belief = belief_revision_policy.get_belief("active_belief")
        assert belief.confidence > 0.7  # Should still be high


class TestBeliefRetrieval:
    """Test belief retrieval operations."""

    def test_get_belief_confidence(self, belief_revision_policy):
        """Test getting belief confidence."""
        belief_revision_policy.register_belief(
            belief_id="test_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Test",
            initial_confidence=0.75
        )

        confidence = belief_revision_policy.get_belief_confidence("test_belief")
        assert confidence == 0.75

    def test_get_nonexistent_belief(self, belief_revision_policy):
        """Test getting non-existent belief returns None."""
        confidence = belief_revision_policy.get_belief_confidence("nonexistent")
        assert confidence is None

    def test_get_belief_by_type(self, belief_revision_policy):
        """Test filtering beliefs by type."""
        belief_revision_policy.register_belief(
            belief_id="loc_1", belief_type=BeliefType.OBJECT_LOCATION,
            description="Location 1", initial_confidence=0.8
        )
        belief_revision_policy.register_belief(
            belief_id="path_1", belief_type=BeliefType.PATH_ACCESSIBLE,
            description="Path 1", initial_confidence=0.7
        )
        belief_revision_policy.register_belief(
            belief_id="loc_2", belief_type=BeliefType.OBJECT_LOCATION,
            description="Location 2", initial_confidence=0.6
        )

        location_beliefs = belief_revision_policy.get_all_beliefs(
            belief_type=BeliefType.OBJECT_LOCATION
        )

        assert len(location_beliefs) == 2
        assert all(b.belief_type == BeliefType.OBJECT_LOCATION for b in location_beliefs)


class TestBeliefStatistics:
    """Test belief statistics and reporting."""

    def test_get_statistics(self, belief_revision_policy):
        """Test getting belief statistics."""
        # Register multiple beliefs
        belief_revision_policy.register_belief(
            belief_id="b1", belief_type=BeliefType.OBJECT_LOCATION,
            description="B1", initial_confidence=0.8
        )
        belief_revision_policy.register_belief(
            belief_id="b2", belief_type=BeliefType.PATH_ACCESSIBLE,
            description="B2", initial_confidence=0.5
        )

        stats = belief_revision_policy.get_statistics()

        assert stats["total_beliefs"] == 2
        assert "average_confidence" in stats
        assert "by_type" in stats

    def test_cleanup_invalid_beliefs(self, belief_revision_policy):
        """Test cleanup of low-confidence beliefs."""
        belief_revision_policy.register_belief(
            belief_id="invalid_belief",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Invalid",
            initial_confidence=0.15
        )

        # Report failures to reduce confidence further
        for _ in range(5):
            belief_revision_policy.report_operation_failure(
                belief_id="invalid_belief",
                operation_type=OperationType.SEARCH
            )

        removed_count = belief_revision_policy.cleanup_invalid_beliefs()

        assert removed_count >= 1
        assert belief_revision_policy.get_belief("invalid_belief") is None

    def test_remove_belief(self, belief_revision_policy):
        """Test removing a belief."""
        belief_revision_policy.register_belief(
            belief_id="to_remove",
            belief_type=BeliefType.OBJECT_LOCATION,
            description="Will be removed"
        )

        # Verify belief exists
        assert belief_revision_policy.get_belief("to_remove") is not None

        # Remove belief
        result = belief_revision_policy.remove_belief("to_remove")
        assert result == True

        # Verify belief is gone
        assert belief_revision_policy.get_belief("to_remove") is None
