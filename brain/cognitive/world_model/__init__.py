# -*- coding: utf-8 -*-
"""
Brain Cognitive World Model Module

This module provides the cognitive world model implementation for the Brain system,
including multi-sensor fusion, and situational awareness.

Note: Sensor-related files (sensor_manager, sensor_interface, data_converter, sensor_input_types)
have been moved to brain/perception/ as they belong to the perception layer.
"""

from brain.cognitive.world_model.world_model import WorldModel
from brain.cognitive.world_model.environment_change import (
    EnvironmentChange,
    ChangeType,
    ChangePriority
)
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.object_tracking import TrackedObject
from brain.cognitive.world_model.semantic import (
    ObjectState,
    SemanticObject,
    ExplorationFrontier
)
from brain.cognitive.world_model.belief import (
    Belief,
    BeliefUpdatePolicy
)

__all__ = [
    # Core world model
    "WorldModel",
    
    # Environment changes
    "EnvironmentChange",
    "ChangeType",
    "ChangePriority",
    
    # Planning context
    "PlanningContext",
    
    # Object tracking
    "TrackedObject",
    
    # Semantic understanding
    "ObjectState",
    "SemanticObject",
    "ExplorationFrontier",
    
    # Belief management
    "Belief",
    "BeliefUpdatePolicy",
]
