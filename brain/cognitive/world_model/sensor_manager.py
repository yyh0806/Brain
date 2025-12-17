# -*- coding: utf-8 -*-
"""
Multi-Sensor Manager Module

This module provides comprehensive sensor management capabilities including
multi-sensor synchronization, data quality assessment, and temporal alignment
for the Brain cognitive world model system.

Author: Brain Development Team
Date: 2025-12-17
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
from collections import defaultdict, deque
import threading
import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
import queue

from .sensor_interface import BaseSensor, SensorConfig, create_sensor
from .sensor_input_types import (
    SensorDataPacket,
    SensorType,
    SensorDataBuffer,
    QualityAssessment,
)

# Configure logging
logger = logging.getLogger(__name__)


class SensorSyncStatus(Enum):
    """Synchronization status for sensors."""
    SYNCHRONIZED = "synchronized"
    UNSYNCHRONIZED = "unsynchronized"
    CALIBRATING = "calibrating"
    ERROR = "error"
    OFFLINE = "offline"


class SyncMethod(Enum):
    """Available synchronization methods."""
    TIMESTAMP_ALIGNMENT = "timestamp_alignment"
    HARDWARE_TRIGGER = "hardware_trigger"
    SOFTWARE_TRIGGER = "software_trigger"
    INTERPOLATION = "interpolation"


@dataclass
class SensorGroup:
    """Group of sensors that should be synchronized together."""
    group_id: str
    sensor_ids: Set[str]
    sync_method: SyncMethod
    sync_tolerance: float = 0.01  # seconds
    priority_sensors: List[str] = field(default_factory=list)
    max_sync_attempts: int = 3


@dataclass
class SynchronizedDataPacket:
    """Synchronized multi-sensor data packet."""
    timestamp: float
    sensor_packets: Dict[str, SensorDataPacket]
    sync_quality: float
    sync_method: SyncMethod
    group_id: str

    # Temporal alignment information
    max_timestamp_delta: float
    avg_timestamp_delta: float

    def get_sensor_data(self, sensor_id: str) -> Optional[SensorDataPacket]:
        """Get data packet for specific sensor."""
        return self.sensor_packets.get(sensor_id)

    def has_sensor(self, sensor_id: str) -> bool:
        """Check if packet contains data from specific sensor."""
        return sensor_id in self.sensor_packets

    def get_sensor_types(self) -> Set[SensorType]:
        """Get all sensor types in this synchronized packet."""
        return {packet.sensor_type for packet in self.sensor_packets.values()}


@dataclass
class DataQualityAssessment:
    """Assessment of sensor data quality and reliability."""
    sensor_id: str
    timestamp: float
    overall_quality: float  # 0.0 to 1.0
    completeness: float     # 0.0 to 1.0
    consistency: float     # 0.0 to 1.0
    timeliness: float      # 0.0 to 1.0
    accuracy: float        # 0.0 to 1.0

    # Quality metrics
    data_points: int = 0
    noise_level: float = 0.0
    outlier_ratio: float = 0.0
    missing_data_ratio: float = 0.0

    # Quality flags
    has_anomalies: bool = False
    is_degraded: bool = False
    is_invalid: bool = False

    # Issues and recommendations
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SensorPerformanceMetrics:
    """Performance metrics for a sensor."""
    sensor_id: str
    update_period_mean: float
    update_period_std: float
    processing_time_mean: float
    processing_time_std: float
    quality_mean: float
    quality_std: float
    packet_loss_rate: float
    uptime_percentage: float
    last_update_time: float

    # Recent history
    recent_periods: List[float] = field(default_factory=list)
    recent_qualities: List[float] = field(default_factory=list)


class MultiSensorManager:
    """
    Comprehensive multi-sensor management system.

    This class provides unified management of multiple sensors including
    synchronization, quality assessment, temporal alignment, and data
    routing for downstream processing.
    """

    def __init__(self, max_sync_window: float = 0.1, max_buffer_size: int = 1000):
        """
        Initialize multi-sensor manager.

        Args:
            max_sync_window: Maximum time window for synchronization (seconds)
            max_buffer_size: Maximum buffer size for each sensor
        """
        self.max_sync_window = max_sync_window
        self.max_buffer_size = max_buffer_size

        # Sensor management
        self.sensors: Dict[str, BaseSensor] = {}
        self.sensor_configs: Dict[str, SensorConfig] = {}
        self.sensor_groups: Dict[str, SensorGroup] = {}
        self.active_sensors: Set[str] = set()

        # Data buffering
        self.global_buffer: Dict[str, SensorDataBuffer] = defaultdict(
            lambda: SensorDataBuffer(max_buffer_size, max_age_seconds=5.0)
        )

        # Synchronization state
        self.sync_status: Dict[str, SensorSyncStatus] = {}
        self.sync_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Thread safety
        self._manager_lock = threading.RLock()
        self._sync_lock = threading.RLock()

        # Callbacks for synchronized data
        self._sync_callbacks: List[Callable[[SynchronizedDataPacket], None]] = []

        # Threading for synchronization
        self._sync_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="SensorSync")
        self._sync_stop_event = threading.Event()

        # Performance monitoring
        self.performance_metrics: Dict[str, SensorPerformanceMetrics] = {}
        self._monitoring_thread: Optional[threading.Thread] = None

        # Statistics
        self.stats = {
            "total_sync_operations": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "average_sync_quality": 0.0,
            "sync_rate": 0.0,
            "start_time": time.time(),
        }

        logger.info("Multi-sensor manager initialized")

    def add_sensor(self, sensor: BaseSensor, config: SensorConfig) -> bool:
        """
        Add a sensor to the manager.

        Args:
            sensor: Sensor instance to add
            config: Sensor configuration

        Returns:
            True if sensor added successfully, False otherwise
        """
        with self._manager_lock:
            if sensor.config.sensor_id in self.sensors:
                logger.warning(f"Sensor {sensor.config.sensor_id} already exists")
                return False

            self.sensors[sensor.config.sensor_id] = sensor
            self.sensor_configs[sensor.config.sensor_id] = config
            self.sync_status[sensor.config.sensor_id] = SensorSyncStatus.UNSYNCHRONIZED

            # Add callback for data collection
            sensor.add_callback(self._on_sensor_data)

            logger.info(f"Added sensor {sensor.config.sensor_id} of type {config.sensor_type}")
            return True

    def remove_sensor(self, sensor_id: str) -> bool:
        """
        Remove a sensor from the manager.

        Args:
            sensor_id: ID of sensor to remove

        Returns:
            True if sensor removed successfully, False otherwise
        """
        with self._manager_lock:
            if sensor_id not in self.sensors:
                logger.warning(f"Sensor {sensor_id} not found")
                return False

            sensor = self.sensors[sensor_id]
            sensor.stop()
            sensor.remove_callback(self._on_sensor_data)

            del self.sensors[sensor_id]
            del self.sensor_configs[sensor_id]
            del self.sync_status[sensor_id]
            del self.global_buffer[sensor_id]

            # Remove from all groups
            for group in self.sensor_groups.values():
                group.sensor_ids.discard(sensor_id)

            self.active_sensors.discard(sensor_id)

            logger.info(f"Removed sensor {sensor_id}")
            return True

    def create_sensor_group(self, group_id: str, sensor_ids: List[str],
                          sync_method: SyncMethod = SyncMethod.TIMESTAMP_ALIGNMENT,
                          sync_tolerance: float = 0.01) -> bool:
        """
        Create a sensor synchronization group.

        Args:
            group_id: Unique identifier for the group
            sensor_ids: List of sensor IDs to include in group
            sync_method: Synchronization method to use
            sync_tolerance: Time tolerance for synchronization

        Returns:
            True if group created successfully, False otherwise
        """
        with self._manager_lock:
            # Validate all sensors exist
            for sensor_id in sensor_ids:
                if sensor_id not in self.sensors:
                    logger.error(f"Sensor {sensor_id} not found for group {group_id}")
                    return False

            if group_id in self.sensor_groups:
                logger.warning(f"Sensor group {group_id} already exists")
                return False

            group = SensorGroup(
                group_id=group_id,
                sensor_ids=set(sensor_ids),
                sync_method=sync_method,
                sync_tolerance=sync_tolerance
            )

            self.sensor_groups[group_id] = group

            # Initialize sync status for all sensors in group
            for sensor_id in sensor_ids:
                self.sync_status[sensor_id] = SensorSyncStatus.UNSYNCHRONIZED

            logger.info(f"Created sensor group {group_id} with sensors {sensor_ids}")
            return True

    def start_sensors(self, sensor_ids: Optional[List[str]] = None) -> bool:
        """
        Start specified sensors or all sensors.

        Args:
            sensor_ids: List of sensor IDs to start, None for all

        Returns:
            True if all sensors started successfully
        """
        sensors_to_start = sensor_ids if sensor_ids else list(self.sensors.keys())
        success = True

        for sensor_id in sensors_to_start:
            if sensor_id not in self.sensors:
                logger.error(f"Sensor {sensor_id} not found")
                success = False
                continue

            sensor = self.sensors[sensor_id]
            if sensor.start():
                self.active_sensors.add(sensor_id)
                self.sync_status[sensor_id] = SensorSyncStatus.CALIBRATING
                logger.info(f"Started sensor {sensor_id}")
            else:
                logger.error(f"Failed to start sensor {sensor_id}")
                success = False

        # Start synchronization if not already running
        if self.active_sensors and not self._monitoring_thread:
            self._start_synchronization()

        return success

    def stop_sensors(self, sensor_ids: Optional[List[str]] = None) -> None:
        """
        Stop specified sensors or all sensors.

        Args:
            sensor_ids: List of sensor IDs to stop, None for all
        """
        sensors_to_stop = sensor_ids if sensor_ids else list(self.sensors.keys())

        for sensor_id in sensors_to_stop:
            if sensor_id in self.sensors:
                sensor = self.sensors[sensor_id]
                sensor.stop()
                self.active_sensors.discard(sensor_id)
                self.sync_status[sensor_id] = SensorSyncStatus.OFFLINE
                logger.info(f"Stopped sensor {sensor_id}")

        # Stop synchronization if no active sensors
        if not self.active_sensors and self._monitoring_thread:
            self._stop_synchronization()

    def start_synchronization(self) -> None:
        """Start multi-sensor synchronization processes."""
        self._start_synchronization()

    def stop_synchronization(self) -> None:
        """Stop multi-sensor synchronization processes."""
        self._stop_synchronization()

    def add_sync_callback(self, callback: Callable[[SynchronizedDataPacket], None]) -> None:
        """
        Add callback for synchronized data packets.

        Args:
            callback: Function to call with synchronized data
        """
        with self._sync_lock:
            self._sync_callbacks.append(callback)

    def remove_sync_callback(self, callback: Callable[[SynchronizedDataPacket], None]) -> None:
        """
        Remove callback for synchronized data packets.

        Args:
            callback: Function to remove
        """
        with self._sync_lock:
            if callback in self._sync_callbacks:
                self._sync_callbacks.remove(callback)

    def get_synchronized_data(self, group_id: str, count: int = 1) -> List[SynchronizedDataPacket]:
        """
        Get recent synchronized data for a specific group.

        Args:
            group_id: Sensor group ID
            count: Number of packets to retrieve

        Returns:
            List of synchronized data packets
        """
        with self._sync_lock:
            if group_id not in self.sync_buffers:
                return []

            buffer = self.sync_buffers[group_id]
            return list(buffer)[-count:] if count > 1 else [buffer[-1]] if buffer else []

    def assess_sensor_quality(self, sensor_id: str, window_seconds: float = 5.0) -> DataQualityAssessment:
        """
        Assess the quality of data from a specific sensor.

        Args:
            sensor_id: Sensor ID to assess
            window_seconds: Time window for assessment

        Returns:
            Quality assessment for the sensor
        """
        current_time = time.time()
        start_time = current_time - window_seconds

        # Get recent data from buffer
        buffer = self.global_buffer[sensor_id]
        recent_packets = buffer.get_by_timerange(start_time, current_time)

        if not recent_packets:
            return DataQualityAssessment(
                sensor_id=sensor_id,
                timestamp=current_time,
                overall_quality=0.0,
                completeness=0.0,
                consistency=0.0,
                timeliness=0.0,
                accuracy=0.0,
                issues=["No data available"],
                is_invalid=True
            )

        # Calculate quality metrics
        qualities = [packet.quality_score for packet in recent_packets]
        timestamps = [packet.timestamp for packet in recent_packets]

        completeness = len(recent_packets) / max(1, window_seconds * self.sensor_configs[sensor_id].update_rate)
        completeness = min(1.0, completeness)

        overall_quality = np.mean(qualities)
        consistency = 1.0 - np.std(qualities) if len(qualities) > 1 else 1.0

        # Assess timeliness (how recent is the data)
        most_recent = max(timestamps)
        age = current_time - most_recent
        timeliness = max(0.0, 1.0 - age / window_seconds)

        # Assess accuracy (placeholder - would need sensor-specific assessment)
        accuracy = overall_quality  # Simplified

        # Determine issues
        issues = []
        has_anomalies = False
        is_degraded = False
        is_invalid = False

        if overall_quality < 0.3:
            issues.append("Very poor data quality")
            is_invalid = True
        elif overall_quality < 0.6:
            issues.append("Degraded data quality")
            is_degraded = True

        if completeness < 0.5:
            issues.append("Low data completeness")
            is_degraded = True

        if timeliness < 0.3:
            issues.append("Data is stale")
            is_degraded = True

        if age > window_seconds:
            issues.append("No recent data")
            is_invalid = True

        has_anomalies = len(issues) > 0

        return DataQualityAssessment(
            sensor_id=sensor_id,
            timestamp=current_time,
            overall_quality=overall_quality,
            completeness=completeness,
            consistency=consistency,
            timeliness=timeliness,
            accuracy=accuracy,
            data_points=len(recent_packets),
            issues=issues,
            has_anomalies=has_anomalies,
            is_degraded=is_degraded,
            is_invalid=is_invalid,
            recommendations=self._generate_quality_recommendations(overall_quality, completeness, timeliness)
        )

    def get_sensor_performance_metrics(self, sensor_id: str) -> Optional[SensorPerformanceMetrics]:
        """
        Get performance metrics for a specific sensor.

        Args:
            sensor_id: Sensor ID

        Returns:
            Performance metrics or None if sensor not found
        """
        if sensor_id not in self.sensors:
            return None

        sensor = self.sensors[sensor_id]
        stats = sensor.get_statistics()

        return SensorPerformanceMetrics(
            sensor_id=sensor_id,
            update_period_mean=1.0 / max(0.001, stats["average_update_rate"]),
            update_period_std=0.1,  # Placeholder
            processing_time_mean=np.mean(self._get_processing_times(sensor_id)),
            processing_time_std=np.std(self._get_processing_times(sensor_id)),
            quality_mean=stats.get("average_quality", 0.0),
            quality_std=0.1,  # Placeholder
            packet_loss_rate=stats.get("loss_rate_percent", 0.0),
            uptime_percentage=100.0 if stats["is_running"] else 0.0,
            last_update_time=stats["last_update_time"]
        )

    def get_manager_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive manager statistics.

        Returns:
            Dictionary containing manager statistics
        """
        with self._manager_lock:
            current_time = time.time()
            runtime = current_time - self.stats["start_time"]

            # Calculate sync rate
            if runtime > 0:
                self.stats["sync_rate"] = self.stats["successful_syncs"] / runtime

            return {
                "runtime_seconds": runtime,
                "total_sensors": len(self.sensors),
                "active_sensors": len(self.active_sensors),
                "sensor_groups": len(self.sensor_groups),
                "sync_operations": {
                    "total": self.stats["total_sync_operations"],
                    "successful": self.stats["successful_syncs"],
                    "failed": self.stats["failed_syncs"],
                    "success_rate": self.stats["successful_syncs"] / max(1, self.stats["total_sync_operations"]) * 100,
                    "rate_per_second": self.stats["sync_rate"],
                },
                "sensor_status": {
                    sensor_id: {
                        "status": self.sync_status.get(sensor_id, "unknown"),
                        "is_active": sensor_id in self.active_sensors,
                        "buffer_size": self.global_buffer[sensor_id].size(),
                    }
                    for sensor_id in self.sensors.keys()
                },
                "group_status": {
                    group_id: {
                        "sensor_count": len(group.sensor_ids),
                        "active_sensors": len([sid for sid in group.sensor_ids if sid in self.active_sensors]),
                        "sync_method": group.sync_method.value,
                        "sync_tolerance": group.sync_tolerance,
                    }
                    for group_id, group in self.sensor_groups.items()
                }
            }

    def _start_synchronization(self) -> None:
        """Start synchronization threads."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._sync_stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._synchronization_loop,
            name="SensorSyncManager",
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started sensor synchronization")

    def _stop_synchronization(self) -> None:
        """Stop synchronization threads."""
        self._sync_stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        self._sync_executor.shutdown(wait=True)
        logger.info("Stopped sensor synchronization")

    def _synchronization_loop(self) -> None:
        """Main synchronization loop."""
        logger.info("Sensor synchronization loop started")

        while not self._sync_stop_event.is_set():
            try:
                current_time = time.time()

                # Synchronize each group
                for group_id, group in self.sensor_groups.items():
                    if self._can_synchronize_group(group, current_time):
                        self._sync_executor.submit(self._synchronize_group, group, current_time)

                # Sleep for next cycle
                self._sync_stop_event.wait(0.01)  # 100 Hz sync attempt rate

            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")

        logger.info("Sensor synchronization loop ended")

    def _can_synchronize_group(self, group: SensorGroup, current_time: float) -> bool:
        """
        Check if a group can be synchronized.

        Args:
            group: Sensor group to check
            current_time: Current timestamp

        Returns:
            True if group can be synchronized
        """
        # Check if enough sensors are active
        active_sensors_in_group = group.sensor_ids.intersection(self.active_sensors)
        if len(active_sensors_in_group) < 2:
            return False

        # Check if we have recent data from all sensors
        for sensor_id in active_sensors_in_group:
            buffer = self.global_buffer[sensor_id]
            latest_data = buffer.get_latest(1)
            if not latest_data:
                return False

            # Check data recency
            packet = latest_data[0]
            if current_time - packet.timestamp > self.max_sync_window:
                return False

        return True

    def _synchronize_group(self, group: SensorGroup, target_time: float) -> Optional[SynchronizedDataPacket]:
        """
        Synchronize a sensor group around target time.

        Args:
            group: Sensor group to synchronize
            target_time: Target synchronization time

        Returns:
            Synchronized data packet or None if synchronization failed
        """
        try:
            self.stats["total_sync_operations"] += 1

            # Collect data for each sensor
            sensor_packets = {}
            timestamp_deltas = []

            for sensor_id in group.sensor_ids:
                if sensor_id not in self.active_sensors:
                    continue

                # Find best data packet around target time
                buffer = self.global_buffer[sensor_id]
                time_window = group.sync_tolerance

                best_packet = self._find_closest_packet(buffer, target_time, time_window)
                if best_packet:
                    sensor_packets[sensor_id] = best_packet
                    timestamp_deltas.append(abs(best_packet.timestamp - target_time))

            # Check if we have enough sensors for synchronization
            if len(sensor_packets) < 2:
                self.stats["failed_syncs"] += 1
                return None

            # Calculate synchronization quality
            max_delta = max(timestamp_deltas) if timestamp_deltas else float('inf')
            avg_delta = np.mean(timestamp_deltas) if timestamp_deltas else 0

            # Quality based on timestamp alignment
            sync_quality = max(0.0, 1.0 - max_delta / group.sync_tolerance)

            # Create synchronized packet
            synced_packet = SynchronizedDataPacket(
                timestamp=target_time,
                sensor_packets=sensor_packets,
                sync_quality=sync_quality,
                sync_method=group.sync_method,
                group_id=group.group_id,
                max_timestamp_delta=max_delta,
                avg_timestamp_delta=avg_delta
            )

            # Store synchronized packet
            with self._sync_lock:
                self.sync_buffers[group.group_id].append(synced_packet)

                # Notify callbacks
                for callback in self._sync_callbacks:
                    try:
                        callback(synced_packet)
                    except Exception as e:
                        logger.error(f"Error in sync callback: {e}")

            self.stats["successful_syncs"] += 1

            # Update sync status
            for sensor_id in sensor_packets.keys():
                self.sync_status[sensor_id] = SensorSyncStatus.SYNCHRONIZED

            return synced_packet

        except Exception as e:
            logger.error(f"Error synchronizing group {group.group_id}: {e}")
            self.stats["failed_syncs"] += 1
            return None

    def _find_closest_packet(self, buffer: SensorDataBuffer, target_time: float,
                           time_window: float) -> Optional[SensorDataPacket]:
        """
        Find the packet closest to target time within window.

        Args:
            buffer: Sensor data buffer
            target_time: Target timestamp
            time_window: Time window to search

        Returns:
            Closest data packet or None if none found
        """
        # Get packets in time window
        start_time = target_time - time_window
        end_time = target_time + time_window

        packets = buffer.get_by_timerange(start_time, end_time)
        if not packets:
            return None

        # Find closest packet
        closest_packet = min(packets, key=lambda p: abs(p.timestamp - target_time))
        return closest_packet

    def _on_sensor_data(self, packet: SensorDataPacket) -> None:
        """
        Callback for receiving data from sensors.

        Args:
            packet: Sensor data packet
        """
        # Add to global buffer
        self.global_buffer[packet.sensor_id].add(packet)

        # Update sensor sync status
        if packet.sensor_id in self.sync_status:
            if self.sync_status[packet.sensor_id] == SensorSyncStatus.OFFLINE:
                self.sync_status[packet.sensor_id] = SensorSyncStatus.CALIBRATING

    def _get_processing_times(self, sensor_id: str) -> List[float]:
        """Get recent processing times for a sensor."""
        # This would track processing times in a real implementation
        return [0.05]  # Placeholder

    def _generate_quality_recommendations(self, quality: float, completeness: float,
                                        timeliness: float) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []

        if quality < 0.5:
            recommendations.append("Check sensor calibration and alignment")

        if completeness < 0.7:
            recommendations.append("Verify sensor power and connections")

        if timeliness < 0.7:
            recommendations.append("Check processing pipeline performance")

        return recommendations

    def __del__(self):
        """Cleanup when manager is destroyed."""
        self.stop_sensors()
        self.stop_synchronization()