"""
Sensor Fusion Module for Brain System
"""

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from brain.data_structures.models import SensorData, FusionResult, WorldObject


class SensorFusion:
    """Multi-sensor data fusion implementation"""

    def __init__(self):
        self.sensor_inputs = {}
        self.fusion_history = []
        self.fusion_weights = {
            'lidar': 0.4,
            'camera': 0.3,
            'imu': 0.2,
            'gps': 0.1
        }

    def add_sensor_data(self, sensor_data: SensorData):
        """Add sensor data to fusion queue"""
        self.sensor_inputs[sensor_data.sensor_id] = sensor_data

    def fuse_objects(self, sensor_objects: List[List[WorldObject]]) -> List[WorldObject]:
        """Fuse object detections from multiple sensors"""
        if not sensor_objects:
            return []

        # Simple object fusion based on spatial proximity
        fused_objects = []

        # Flatten all objects with sensor info
        all_objects = []
        for i, objects in enumerate(sensor_objects):
            for obj in objects:
                all_objects.append({
                    'object': obj,
                    'sensor_idx': i
                })

        # Cluster objects by position
        clusters = self._cluster_objects_by_position(all_objects)

        # Create fused objects from clusters
        for cluster in clusters:
            if len(cluster) > 0:
                fused_obj = self._create_fused_object(cluster)
                fused_objects.append(fused_obj)

        return fused_objects

    def _cluster_objects_by_position(self, objects: List[Dict], threshold: float = 2.0) -> List[List[Dict]]:
        """Cluster objects by spatial proximity"""
        clusters = []
        used_indices = set()

        for i, obj1 in enumerate(objects):
            if i in used_indices:
                continue

            cluster = [obj1]
            used_indices.add(i)

            for j, obj2 in enumerate(objects[i+1:], i+1):
                if j in used_indices:
                    continue

                # Calculate distance
                pos1 = obj1['object'].position
                pos2 = obj2['object'].position
                distance = np.sqrt((pos1[0] - pos2[0])**2 +
                                 (pos1[1] - pos2[1])**2 +
                                 (pos1[2] - pos2[2])**2)

                if distance < threshold:
                    cluster.append(obj2)
                    used_indices.add(j)

            clusters.append(cluster)

        return clusters

    def _create_fused_object(self, cluster: List[Dict]) -> WorldObject:
        """Create a fused object from a cluster of detections"""
        if len(cluster) == 1:
            return cluster[0]['object']

        # Average position
        positions = [obj['object'].position for obj in cluster]
        avg_position = tuple(np.mean(positions, axis=0))

        # Average orientation
        orientations = [obj['object'].orientation for obj in cluster]
        avg_orientation = tuple(np.mean(orientations, axis=0))

        # Use the type with highest confidence
        best_obj = max(cluster, key=lambda x: x['object'].confidence)

        # Average confidence weighted by sensor weights
        avg_confidence = np.mean([
            obj['object'].confidence * self.fusion_weights.get(obj['object'].type, 1.0)
            for obj in cluster
        ])

        return WorldObject(
            id=f"fused_{best_obj['object'].id}",
            type=best_obj['object'].type,
            position=avg_position,
            orientation=avg_orientation,
            size=best_obj['object'].size,
            velocity=best_obj['object'].velocity,
            confidence=min(1.0, avg_confidence * len(cluster)),  # Boost confidence for multi-sensor
            timestamp=max(obj['object'].timestamp for obj in cluster),
            attributes=best_obj['object'].attributes
        )

    def get_fusion_result(self, timestamp: Optional[float] = None) -> Optional[FusionResult]:
        """Get current fusion result"""
        if not self.sensor_inputs:
            return None

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Extract objects from sensor data
        sensor_objects = []
        source_sensors = []

        for sensor_data in self.sensor_inputs.values():
            if hasattr(sensor_data, 'objects') and sensor_data.objects:
                sensor_objects.append(sensor_data.objects)
                source_sensors.append(sensor_data.sensor_id)

        # Perform fusion
        fused_objects = self.fuse_objects(sensor_objects)
        confidence_scores = [obj.confidence for obj in fused_objects]

        return FusionResult(
            timestamp=timestamp,
            fused_objects=fused_objects,
            confidence_scores=confidence_scores,
            source_sensors=source_sensors,
            fusion_method="spatial_proximity",
            metadata={
                'num_sensors': len(source_sensors),
                'num_objects': len(fused_objects),
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0
            }
        )

    def clear(self):
        """Clear all sensor inputs and history"""
        self.sensor_inputs.clear()
        self.fusion_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        return {
            'active_sensors': len(self.sensor_inputs),
            'fusion_history_size': len(self.fusion_history),
            'fusion_weights': self.fusion_weights.copy()
        }