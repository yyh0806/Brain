"""
GPS Sensor Model
"""

from typing import Dict, Any, Optional
from .base import SensorModel, SensorConfig


class GPSModel(SensorModel):
    """GPS sensor model implementation"""

    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, config)
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.accuracy = 0.0
        self.update_rate = config.parameters.get('update_rate', 10.0)

    async def initialize(self) -> bool:
        """Initialize GPS sensor"""
        try:
            self.is_initialized = True
            return True
        except Exception:
            return False

    async def update(self, data: Dict[str, Any]) -> bool:
        """Update GPS with new position data"""
        if not self.is_initialized:
            return False

        try:
            if 'latitude' in data:
                self.latitude = data['latitude']
            if 'longitude' in data:
                self.longitude = data['longitude']
            if 'altitude' in data:
                self.altitude = data['altitude']
            if 'accuracy' in data:
                self.accuracy = data['accuracy']

            self.last_update_time = data.get('timestamp', 0.0)
            return True

        except Exception:
            return False

    async def get_data(self) -> Optional[Dict[str, Any]]:
        """Get current GPS data"""
        if not self.is_initialized:
            return None

        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.last_update_time,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'accuracy': self.accuracy
        }

    def get_status(self) -> Dict[str, Any]:
        """Get GPS status"""
        return {
            'sensor_id': self.sensor_id,
            'type': 'gps',
            'initialized': self.is_initialized,
            'enabled': self.is_enabled(),
            'last_update': self.last_update_time,
            'update_rate': self.update_rate,
            'accuracy': self.accuracy
        }

    async def cleanup(self):
        """Cleanup GPS resources"""
        self.is_initialized = False