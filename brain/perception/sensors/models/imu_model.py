"""
IMU Sensor Model
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import SensorModel, SensorConfig


class IMUModel(SensorModel):
    """IMU sensor model implementation"""

    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, config)
        self.orientation = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        self.angular_velocity = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.linear_acceleration = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.update_rate = config.parameters.get('update_rate', 100.0)

    async def initialize(self) -> bool:
        """Initialize IMU sensor"""
        try:
            self.is_initialized = True
            return True
        except Exception:
            return False

    async def update(self, data: Dict[str, Any]) -> bool:
        """Update IMU with new sensor data"""
        if not self.is_initialized:
            return False

        try:
            if 'orientation' in data:
                self.orientation.update(data['orientation'])
            if 'angular_velocity' in data:
                self.angular_velocity.update(data['angular_velocity'])
            if 'linear_acceleration' in data:
                self.linear_acceleration.update(data['linear_acceleration'])

            self.last_update_time = data.get('timestamp', 0.0)
            return True

        except Exception:
            return False

    async def get_data(self) -> Optional[Dict[str, Any]]:
        """Get current IMU data"""
        if not self.is_initialized:
            return None

        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.last_update_time,
            'orientation': self.orientation,
            'angular_velocity': self.angular_velocity,
            'linear_acceleration': self.linear_acceleration
        }

    def get_status(self) -> Dict[str, Any]:
        """Get IMU status"""
        return {
            'sensor_id': self.sensor_id,
            'type': 'imu',
            'initialized': self.is_initialized,
            'enabled': self.is_enabled(),
            'last_update': self.last_update_time,
            'update_rate': self.update_rate
        }

    async def cleanup(self):
        """Cleanup IMU resources"""
        self.is_initialized = False