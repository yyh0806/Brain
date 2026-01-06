"""
Camera Sensor Model
"""

from typing import Dict, Any, Optional
import numpy as np
from .base import SensorModel, SensorConfig


class CameraModel(SensorModel):
    """Camera sensor model implementation"""

    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, config)
        self.image = None
        self.width = config.parameters.get('width', 640)
        self.height = config.parameters.get('height', 480)
        self.fps = config.parameters.get('fps', 30.0)
        self.encoding = config.parameters.get('encoding', 'rgb8')

    async def initialize(self) -> bool:
        """Initialize camera sensor"""
        try:
            self.is_initialized = True
            return True
        except Exception:
            return False

    async def update(self, data: Dict[str, Any]) -> bool:
        """Update camera with new image data"""
        if not self.is_initialized:
            return False

        try:
            if 'image' in data:
                self.image = data['image']
            elif 'data' in data:
                # Raw image data
                self.image = np.frombuffer(data['data'], dtype=np.uint8)
                self.image = self.image.reshape((self.height, self.width, -1))
            else:
                return False

            self.last_update_time = data.get('timestamp', 0.0)
            return True

        except Exception:
            return False

    async def get_data(self) -> Optional[Dict[str, Any]]:
        """Get current camera data"""
        if not self.is_initialized:
            return None

        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.last_update_time,
            'image': self.image,
            'width': self.width,
            'height': self.height,
            'encoding': self.encoding
        }

    def get_status(self) -> Dict[str, Any]:
        """Get camera status"""
        return {
            'sensor_id': self.sensor_id,
            'type': 'camera',
            'initialized': self.is_initialized,
            'enabled': self.is_enabled(),
            'last_update': self.last_update_time,
            'resolution': f"{self.width}x{self.height}"
        }

    async def cleanup(self):
        """Cleanup camera resources"""
        self.image = None
        self.is_initialized = False