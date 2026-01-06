# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for all test types

This file provides common test fixtures for unit, functional, and integration tests.
Fixtures follow real data flow testing philosophy: use real data structures,
only mock external dependencies (LLM, Isaac Sim).
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock PerceptionData classes for testing
class Pose:
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

class Velocity:
    def __init__(self, linear_x=0.0, linear_y=0.0, linear_z=0.0,
                 angular_x=0.0, angular_y=0.0, angular_z=0.0):
        self.linear_x = linear_x
        self.linear_y = linear_y
        self.linear_z = linear_z
        self.angular_x = angular_x
        self.angular_y = angular_y
        self.angular_z = angular_z

class PerceptionData:
    def __init__(self, pose=None, velocity=None, occupancy_grid=None,
                 grid_resolution=0.1, grid_origin=(0.0, 0.0),
                 obstacles=None, timestamp=None, pointcloud=None, rgb_image=None,
                 semantic_objects=None):
        self.pose = pose or Pose()
        self.velocity = velocity or Velocity()
        self.occupancy_grid = occupancy_grid if occupancy_grid is not None else np.zeros((50, 50), dtype=np.int8)
        self.grid_resolution = grid_resolution
        self.grid_origin = grid_origin
        self.obstacles = obstacles or []
        self.timestamp = timestamp or datetime.now()
        self.pointcloud = pointcloud
        self.rgb_image = rgb_image
        # Initialize semantic_objects to match real PerceptionData
        self.semantic_objects = semantic_objects or []

# Lazy import helpers to avoid circular dependencies
def import_cognitive_components():
    """Import cognitive components on demand."""
    from brain.cognitive.interface import CognitiveLayer, ObservationResult, ObservationStatus
    from brain.cognitive.world_model.world_model import WorldModel
    from brain.cognitive.reasoning.cot_engine import CoTEngine, ReasoningMode, ComplexityLevel
    from brain.cognitive.dialogue.dialogue_manager import DialogueManager, DialogueType
    from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor
    return {
        'CognitiveLayer': CognitiveLayer,
        'ObservationResult': ObservationResult,
        'ObservationStatus': ObservationStatus,
        'WorldModel': WorldModel,
        'CoTEngine': CoTEngine,
        'ReasoningMode': ReasoningMode,
        'ComplexityLevel': ComplexityLevel,
        'DialogueManager': DialogueManager,
        'DialogueType': DialogueType,
        'PerceptionMonitor': PerceptionMonitor,
    }


# ============= Real Perception Data Fixtures =============

@pytest.fixture
def sample_pose():
    """Create sample robot pose data."""
    return Pose(
        x=1.0, y=2.0, z=0.0,
        roll=0.0, pitch=0.0, yaw=1.57
    )


@pytest.fixture
def sample_velocity():
    """Create sample velocity data."""
    return Velocity(
        linear_x=0.5, linear_y=0.0, linear_z=0.0,
        angular_x=0.0, angular_y=0.0, angular_z=0.1
    )


@pytest.fixture
def sample_perception_data(sample_pose, sample_velocity):
    """Create complete sample perception data with real flow."""
    # Create occupancy grid
    occupancy_grid = np.zeros((100, 100), dtype=np.int8)
    occupancy_grid[40:60, 40:60] = 100  # Add some obstacles

    # Create obstacles list
    obstacles = [
        {
            "id": "obs_001",
            "type": "static_obstacle",
            "world_position": {"x": 2.0, "y": 1.0, "z": 0.0},
            "local_position": {"x": 1.0, "y": -1.0, "z": 0.0},
            "confidence": 0.9
        },
        {
            "id": "obs_002",
            "type": "person",
            "world_position": {"x": 3.0, "y": 2.0, "z": 0.0},
            "local_position": {"x": 2.0, "y": 0.0, "z": 0.0},
            "confidence": 0.85
        }
    ]

    return PerceptionData(
        pose=sample_pose,
        velocity=sample_velocity,
        occupancy_grid=occupancy_grid,
        grid_resolution=0.1,
        grid_origin=(0.0, 0.0),
        obstacles=obstacles,
        semantic_objects=[],  # Initialize empty list
        timestamp=datetime.now(),
        pointcloud=None,  # Optional
        rgb_image=None    # Optional
    )


@pytest.fixture
def sample_perception_data_with_changes():
    """Create perception data representing environment changes."""
    pose = Pose(x=2.0, y=3.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
    velocity = Velocity(linear_x=0.3, linear_y=0.0, linear_z=0.0,
                       angular_x=0.0, angular_y=0.0, angular_z=0.0)

    # Create grid with new obstacle
    occupancy_grid = np.zeros((100, 100), dtype=np.int8)
    occupancy_grid[20:30, 50:60] = 100  # New obstacle appeared

    obstacles = [
        {
            "id": "obs_new_001",
            "type": "dynamic_obstacle",
            "world_position": {"x": 1.5, "y": 0.5, "z": 0.0},
            "local_position": {"x": -0.5, "y": -2.5, "z": 0.0},
            "confidence": 0.95
        }
    ]

    return PerceptionData(
        pose=pose,
        velocity=velocity,
        occupancy_grid=occupancy_grid,
        grid_resolution=0.1,
        grid_origin=(0.0, 0.0),
        obstacles=obstacles,
        timestamp=datetime.now()
    )


# ============= Cognitive Component Fixtures =============

@pytest.fixture
def world_model():
    """Create WorldModel instance for testing."""
    from brain.cognitive.world_model.world_model import WorldModel
    return WorldModel(config={
        "max_map_size": 10000,
        "max_semantic_objects": 100,
        "enable_memory_management": False  # Disable for tests
    })


@pytest.fixture
def belief_revision_policy():
    """Create BeliefRevisionPolicy for testing."""
    from brain.cognitive.world_model.belief_revision import BeliefRevisionPolicy
    return BeliefRevisionPolicy(config={
        "failure_penalty": 0.2,
        "success_boost": 0.1,
        "max_failures_before_remove": 3,
        "decay_rate": 0.01
    })


@pytest.fixture
def mock_llm_interface():
    """Create mock LLM interface for reasoning tests."""
    mock_llm = AsyncMock()

    # Mock response for reasoning
    mock_llm.chat = AsyncMock(return_value=Mock(
        content="""## 推理过程

### 步骤1: 环境分析
分析: 当前环境中有2个障碍物，机器人位于(1.0, 2.0)
结论: 环境相对清晰，可以继续执行任务
置信度: 0.85

### 步骤2: 路径评估
分析: 检查当前路径是否受阻
结论: 当前路径可行，无需重规划
置信度: 0.90

## 最终决策
决策: 继续执行当前任务
建议: 保持当前速度和方向
置信度: 0.87"""
    ))

    return mock_llm


@pytest.fixture
def cot_engine(mock_llm_interface):
    """Create CoTEngine with mocked LLM for testing."""
    from brain.cognitive.reasoning.cot_engine import CoTEngine
    return CoTEngine(
        llm_interface=mock_llm_interface,
        enable_caching=False  # Disable caching for tests
    )


@pytest.fixture
def dialogue_manager():
    """Create DialogueManager for testing."""
    from brain.cognitive.dialogue.dialogue_manager import DialogueManager
    manager = DialogueManager(
        llm_interface=None,  # Use default responses
        user_callback=None
    )
    # Enable auto-confirm for faster tests
    if hasattr(manager, 'set_auto_confirm'):
        manager.set_auto_confirm(enabled=True, delay=0.0)
    return manager


@pytest.fixture
def perception_monitor(world_model):
    """Create PerceptionMonitor for testing."""
    from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor
    return PerceptionMonitor(
        world_model=world_model,
        config={"monitor_interval": 0.1}
    )


@pytest.fixture
def cognitive_layer(world_model, cot_engine, dialogue_manager, perception_monitor):
    """Create fully configured CognitiveLayer for testing."""
    from brain.cognitive.interface import CognitiveLayer
    return CognitiveLayer(
        world_model=world_model,
        cot_engine=cot_engine,
        dialogue_manager=dialogue_manager,
        perception_monitor=perception_monitor,
        config={}
    )


@pytest.fixture
def belief_revision_policy():
    """Create BeliefRevisionPolicy for testing."""
    from brain.cognitive.world_model.belief_revision import BeliefRevisionPolicy
    return BeliefRevisionPolicy(config={
        "failure_penalty": 0.2,
        "success_boost": 0.1,
        "max_failures_before_remove": 3,
        "decay_rate": 0.01
    })


@pytest.fixture
def mock_llm_interface():
    """Create mock LLM interface for reasoning tests."""
    mock_llm = AsyncMock()

    # Mock response for reasoning
    mock_llm.chat = AsyncMock(return_value=Mock(
        content="""## 推理过程

### 步骤1: 环境分析
分析: 当前环境中有2个障碍物，机器人位于(1.0, 2.0)
结论: 环境相对清晰，可以继续执行任务
置信度: 0.85

### 步骤2: 路径评估
分析: 检查当前路径是否受阻
结论: 当前路径可行，无需重规划
置信度: 0.90

## 最终决策
决策: 继续执行当前任务
建议: 保持当前速度和方向
置信度: 0.87"""
    ))

    return mock_llm


@pytest.fixture
def cot_engine(mock_llm_interface):
    """Create CoTEngine with mocked LLM for testing."""
    from brain.cognitive.reasoning.cot_engine import CoTEngine
    return CoTEngine(
        llm_interface=mock_llm_interface,
        enable_caching=False  # Disable caching for tests
    )


@pytest.fixture
def dialogue_manager():
    """Create DialogueManager for testing."""
    from brain.cognitive.dialogue.dialogue_manager import DialogueManager
    manager = DialogueManager(
        llm_interface=None,  # Use default responses
        user_callback=None
    )
    # Enable auto-confirm for faster tests
    if hasattr(manager, 'set_auto_confirm'):
        manager.set_auto_confirm(enabled=True, delay=0.0)
    return manager


@pytest.fixture
def perception_monitor(world_model):
    """Create PerceptionMonitor for testing."""
    from brain.cognitive.monitoring.perception_monitor import PerceptionMonitor
    return PerceptionMonitor(
        world_model=world_model,
        config={"monitor_interval": 0.1}
    )


@pytest.fixture
def cognitive_layer(world_model, cot_engine, dialogue_manager, perception_monitor):
    """Create fully configured CognitiveLayer for testing."""
    from brain.cognitive.interface import CognitiveLayer
    return CognitiveLayer(
        world_model=world_model,
        cot_engine=cot_engine,
        dialogue_manager=dialogue_manager,
        perception_monitor=perception_monitor,
        config={}
    )


# ============= Test Helper Functions =============

@pytest.fixture
def assert_belief_state():
    """Helper to assert belief state."""
    def _assert(belief, expected_confidence, expected_failures=None):
        assert belief.confidence == pytest.approx(expected_confidence, 0.01)
        if expected_failures is not None:
            assert belief.failure_count == expected_failures
    return _assert


@pytest.fixture
def create_observation_result():
    """Helper to create ObservationResult instances."""
    def _create(
        operation_id: str,
        status: str = "success",
        confidence: float = 1.0
    ):
        return ObservationResult(
            operation_id=operation_id,
            operation_type="search",
            status=ObservationStatus[status.upper()],
            confidence=confidence
        )
    return _create


# ============= Async Test Support =============

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============= Mock Isaac Sim Fixtures =============

@pytest.fixture
def mock_isaac_sim_environment():
    """Mock Isaac Sim environment for non-integration tests."""
    return {
        "robot_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "obstacles": [],
        "targets": [],
        "battery_level": 100.0
    }


# ============= Performance Testing Fixtures =============

@pytest.fixture
def performance_metrics():
    """Helper to collect performance metrics."""
    metrics = {
        "start_time": None,
        "end_time": None,
        "memory_usage": []
    }

    def record(metric_name: str, value: Any):
        metrics[metric_name] = value

    def get_elapsed():
        if metrics["start_time"] and metrics["end_time"]:
            return metrics["end_time"] - metrics["start_time"]
        return None

    metrics.record = record
    metrics.get_elapsed = get_elapsed

    return metrics


# ============= Pytest Configuration =============

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "functional: Functional tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "isaac_sim: Tests requiring Isaac Sim environment"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
