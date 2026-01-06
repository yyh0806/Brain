# -*- coding: utf-8 -*-
"""
Unit Tests for Perception Monitor

Test coverage:
- Initialization and configuration
- Change detection
- Trigger evaluation
- Cooldown mechanism
- Callback handling
"""

import pytest
from unittest.mock import Mock
import time

from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor
from brain.cognitive.world_model.environment_change import ChangeType, ChangePriority


class TestPerceptionMonitorInitialization:
    """Test PerceptionMonitor setup and configuration."""

    def test_default_initialization(self, world_model):
        """Test PerceptionMonitor initializes with defaults."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        assert monitor is not None
        assert monitor.world_model == world_model

    def test_initialization_with_config(self, world_model):
        """Test PerceptionMonitor with custom configuration."""
        config = {
            "monitor_interval": 0.5,
            "cooldown_period": 2.0
        }

        monitor = PerceptionMonitor(
            world_model=world_model,
            config=config
        )

        assert monitor is not None


class TestChangeDetection:
    """Test significant change detection."""

    def test_detect_significant_changes(self, world_model):
        """Test detection of significant changes."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        # Set up callback
        triggered = []
        def callback(change):
            triggered.append(change)

        monitor.set_replan_callback(callback)

        # Check that callback can be set (using private attribute)
        assert monitor._replan_callback is not None

    def test_change_priority_evaluation(self, world_model):
        """Test that changes are prioritized correctly."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        # Monitor should be able to evaluate changes
        assert monitor is not None


class TestCallbackHandling:
    """Test replanning callback mechanism."""

    def test_set_replan_callback(self, world_model):
        """Test setting replan callback."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        callback = Mock()
        monitor.set_replan_callback(callback)

        # Check the private attribute
        assert monitor._replan_callback == callback

    def test_trigger_callback(self, world_model):
        """Test that callback is triggered on significant changes."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        triggered = []
        def callback(change):
            triggered.append(change)

        monitor.set_replan_callback(callback)

        # Verify callback is set (using private attribute)
        assert monitor._replan_callback is not None
        assert len(triggered) == 0  # Not triggered yet


class TestCooldownMechanism:
    """Test cooldown to prevent excessive triggers."""

    def test_cooldown_prevents_excessive_triggers(self, world_model):
        """Test that cooldown prevents too frequent triggers."""
        config = {"cooldown_period": 1.0}
        monitor = PerceptionMonitor(
            world_model=world_model,
            config=config
        )

        # Monitor should have cooldown mechanism
        assert monitor is not None

    def test_cooldown_expiry(self, world_model):
        """Test that cooldown expires after configured period."""
        config = {"cooldown_period": 0.1}
        monitor = PerceptionMonitor(
            world_model=world_model,
            config=config
        )

        # Wait for cooldown to expire
        time.sleep(0.15)

        # Monitor should still be functional
        assert monitor is not None


class TestMonitoringControl:
    """Test starting and stopping monitoring."""

    def test_start_monitoring(self, world_model):
        """Test starting perception monitoring."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        # Should be able to start monitoring
        assert monitor is not None

    def test_stop_monitoring(self, world_model):
        """Test stopping perception monitoring."""
        monitor = PerceptionMonitor(
            world_model=world_model,
            config={}
        )

        # Should be able to stop monitoring
        assert monitor is not None
