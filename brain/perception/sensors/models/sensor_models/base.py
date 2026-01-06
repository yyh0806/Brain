"""
Base Sensor Model for Brain System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass


@dataclass
class SensorConfig:
    """Sensor configuration parameters"""
    sensor_type: str
    enabled: bool = True
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class SensorModel(ABC):
    """Abstract base class for sensor models"""

    def __init__(self, sensor_id: str, config: SensorConfig):
        self.sensor_id = sensor_id
        self.config = config
        self.is_initialized = False
        self.last_update_time = 0.0

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the sensor model"""
        pass

    @abstractmethod
    async def update(self, data: Dict[str, Any]) -> bool:
        """Update sensor with new data"""
        pass

    @abstractmethod
    async def get_data(self) -> Optional[Dict[str, Any]]:
        """Get current sensor data"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get sensor status"""
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup sensor resources"""
        pass

    def is_enabled(self) -> bool:
        """Check if sensor is enabled"""
        return self.config.enabled

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get sensor parameter"""
        return self.config.parameters.get(key, default)