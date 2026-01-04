"""
LiDAR Sensor Model
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .base import SensorModel, SensorConfig


class LidarModel(SensorModel):
    """LiDAR sensor model implementation"""

    def __init__(self, sensor_id: str, config: SensorConfig):
        super().__init__(sensor_id, config)
        self.point_cloud = []
        self.max_range = config.parameters.get('max_range', 100.0)
        self.num_points = config.parameters.get('num_points', 1000)
        self.scan_frequency = config.parameters.get('scan_frequency', 10.0)

    async def initialize(self) -> bool:
        """Initialize LiDAR sensor"""
        try:
            self.is_initialized = True
            return True
        except Exception:
            return False

    async def update(self, data: Dict[str, Any]) -> bool:
        """Update LiDAR with new scan data"""
        if not self.is_initialized:
            return False

        try:
            # Process point cloud data
            if 'points' in data:
                self.point_cloud = data['points']
            elif 'ranges' in data and 'angles' in data:
                # Convert polar to Cartesian
                ranges = np.array(data['ranges'])
                angles = np.array(data['angles'])
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)
                z = np.zeros_like(x)
                self.point_cloud = np.column_stack([x, y, z])
            else:
                return False

            self.last_update_time = data.get('timestamp', 0.0)
            return True

        except Exception:
            return False

    async def get_data(self) -> Optional[Dict[str, Any]]:
        """Get current LiDAR data"""
        if not self.is_initialized:
            return None

        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.last_update_time,
            'point_cloud': self.point_cloud,
            'num_points': len(self.point_cloud),
            'max_range': self.max_range
        }

    def get_status(self) -> Dict[str, Any]:
        """Get LiDAR status"""
        return {
            'sensor_id': self.sensor_id,
            'type': 'lidar',
            'initialized': self.is_initialized,
            'enabled': self.is_enabled(),
            'last_update': self.last_update_time,
            'point_count': len(self.point_cloud)
        }

    async def cleanup(self):
        """Cleanup LiDAR resources"""
        self.point_cloud.clear()
        self.is_initialized = False