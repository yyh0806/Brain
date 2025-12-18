#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Input Module Demo

This script demonstrates the functionality of the sensor input module
by creating multiple sensors, synchronizing their data, and performing
format conversions.

Author: Brain Development Team
Date: 2025-12-17
"""

import sys
import os
import time
import numpy as np
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from brain.cognitive.world_model.sensor_input_types import (
    SensorType, PointCloudData, ImageData, IMUData
)
from brain.cognitive.world_model.sensor_interface import (
    SensorConfig, create_sensor
)
from brain.cognitive.world_model.sensor_manager import (
    MultiSensorManager, SyncMethod
)
from brain.cognitive.world_model.data_converter import (
    DataFormat, ConversionOptions, create_converter
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_demo_sensors():
    """Create demo sensors for testing."""
    print("Creating demo sensors...")

    # Create sensor configurations
    configs = []
    sensors = []

    # Point cloud sensor (LiDAR)
    pc_config = SensorConfig(
        sensor_id="lidar_front",
        sensor_type=SensorType.POINT_CLOUD,
        update_rate=10.0,
        frame_id="lidar_frame"
    )
    pc_sensor = create_sensor(pc_config)
    configs.append(pc_config)
    sensors.append(pc_sensor)

    # Camera sensor
    img_config = SensorConfig(
        sensor_id="camera_front",
        sensor_type=SensorType.IMAGE,
        update_rate=30.0,
        frame_id="camera_frame"
    )
    img_sensor = create_sensor(img_config)
    configs.append(img_config)
    sensors.append(img_sensor)

    # IMU sensor
    imu_config = SensorConfig(
        sensor_id="imu_main",
        sensor_type=SensorType.IMU,
        update_rate=100.0,
        frame_id="imu_frame"
    )
    imu_sensor = create_sensor(imu_config)
    configs.append(imu_config)
    sensors.append(imu_sensor)

    # GPS sensor
    gps_config = SensorConfig(
        sensor_id="gps_main",
        sensor_type=SensorType.GPS,
        update_rate=5.0,
        frame_id="gps_frame"
    )
    gps_sensor = create_sensor(gps_config)
    configs.append(gps_config)
    sensors.append(gps_sensor)

    print("Created {} sensors:".format(len(sensors)))
    for config in configs:
        print("  - {} ({}) @ {} Hz".format(config.sensor_id, config.sensor_type.value, config.update_rate))

    return sensors, configs


def demo_synchronization():
    """Demonstrate multi-sensor synchronization."""
    print("\n" + "="*60)
    print("MULTI-SENSOR SYNCHRONIZATION DEMO")
    print("="*60)

    # Create sensor manager
    manager = MultiSensorManager(max_sync_window=0.1)

    # Create sensors
    sensors, configs = create_demo_sensors()

    # Add sensors to manager
    for sensor, config in zip(sensors, configs):
        manager.add_sensor(sensor, config)

    # Create synchronization groups
    manager.create_sensor_group(
        group_id="perception_group",
        sensor_ids=["lidar_front", "camera_front"],
        sync_method=SyncMethod.TIMESTAMP_ALIGNMENT,
        sync_tolerance=0.02
    )

    manager.create_sensor_group(
        group_id="localization_group",
        sensor_ids=["imu_main", "gps_main"],
        sync_method=SyncMethod.TIMESTAMP_ALIGNMENT,
        sync_tolerance=0.05
    )

    # Add sync callback
    def on_sync_data(sync_packet):
        print("Received synchronized data from group '{}':".format(sync_packet.group_id))
        print("  - Timestamp: {:.3f}".format(sync_packet.timestamp))
        print("  - Sync quality: {:.3f}".format(sync_packet.sync_quality))
        print("  - Sensors: {}".format(list(sync_packet.sensor_packets.keys())))
        print("  - Max time delta: {:.1f} ms".format(sync_packet.max_timestamp_delta*1000))
        print()

    manager.add_sync_callback(on_sync_data)

    # Start sensors
    print("Starting sensors...")
    manager.start_sensors()
    manager.start_synchronization()

    print("Sensors started. Collecting data for 5 seconds...")

    # Run for demonstration period
    time.sleep(5.0)

    # Show statistics
    print("\nSensor Statistics:")
    stats = manager.get_manager_statistics()
    print(f"  - Runtime: {stats['runtime_seconds']:.1f} seconds")
    print(f"  - Active sensors: {stats['active_sensors']}")
    print(f"  - Sync operations: {stats['sync_operations']['total']}")
    print(f"  - Successful syncs: {stats['sync_operations']['successful']}")
    print(f"  - Sync success rate: {stats['sync_operations']['success_rate']:.1f}%")

    # Show individual sensor statistics
    print("\nIndividual Sensor Statistics:")
    for sensor_id, status in stats['sensor_status'].items():
        print(f"  - {sensor_id}: {status['status']} (buffer: {status['buffer_size']})")

    # Show quality assessment
    print("\nData Quality Assessment:")
    for sensor_id in manager.active_sensors:
        assessment = manager.assess_sensor_quality(sensor_id)
        print(f"  - {sensor_id}:")
        print(f"    * Overall quality: {assessment.overall_quality:.3f}")
        print(f"    * Completeness: {assessment.completeness:.3f}")
        print(f"    * Timeliness: {assessment.timeliness:.3f}")
        print(f"    * Issues: {len(assessment.issues)}")

    # Clean up
    print("\nStopping sensors...")
    manager.stop_sensors()
    manager.stop_synchronization()
    print("Demo completed.")


def demo_data_conversion():
    """Demonstrate data format conversion."""
    print("\n" + "="*60)
    print("DATA FORMAT CONVERSION DEMO")
    print("="*60)

    # Create sample data
    print("Creating sample sensor data...")

    # Point cloud data
    points = np.random.rand(1000, 3) * 10  # 1000 points in 10m cube
    intensity = np.random.rand(1000)
    pc_data = PointCloudData(
        points=points,
        intensity=intensity,
        timestamp=time.time(),
        frame_id="lidar_frame"
    )

    print(f"Created point cloud with {pc_data.point_count} points")
    print(f"Point cloud bounds: {pc_data.get_bounds()}")

    # Image data
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    img_data = ImageData(
        image=image,
        timestamp=time.time(),
        frame_id="camera_frame"
    )

    print(f"Created image with size {img_data.width}x{img_data.height}")

    # Convert to different formats
    print("\nConverting data to different formats...")

    formats_to_test = [
        (DataFormat.JSON, "JSON"),
        (DataFormat.CSV, "CSV (for point cloud)"),
        (DataFormat.BINARY, "Binary"),
        (DataFormat.PCD, "PCD (Point Cloud Data)")
    ]

    for format_type, format_name in formats_to_test:
        print(f"\nConverting to {format_name}:")

        try:
            converter = create_converter(format_type)
            options = ConversionOptions(
                target_format=format_type,
                include_metadata=True,
                max_size_mb=1  # Limit size for demo
            )

            # Convert point cloud
            result = converter.convert(pc_data, options)

            if result.success:
                print(f"  ✓ Conversion successful")
                print(f"  - Format: {result.format.value}")
                print(f"  - Conversion time: {result.conversion_time*1000:.1f} ms")

                if isinstance(result.data, str):
                    print(f"  - Data size: {len(result.data)} characters")
                    print(f"  - Preview: {result.data[:100]}...")
                elif isinstance(result.data, bytes):
                    print(f"  - Data size: {len(result.data)} bytes")
                else:
                    print(f"  - Data type: {type(result.data)}")
            else:
                print(f"  ✗ Conversion failed: {result.error_message}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\nData conversion demo completed.")


def demo_sensor_factory():
    """Demonstrate sensor factory functionality."""
    print("\n" + "="*60)
    print("SENSOR FACTORY DEMO")
    print("="*60)

    print("Creating different types of sensors using factory...")

    sensor_types = [
        (SensorType.POINT_CLOUD, "lidar_rear"),
        (SensorType.IMAGE, "camera_rear"),
        (SensorType.IMU, "imu_backup"),
        (SensorType.GPS, "gps_backup")
    ]

    sensors = []

    for sensor_type, sensor_id in sensor_types:
        print(f"\nCreating {sensor_type.value} sensor: {sensor_id}")

        config = SensorConfig(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            update_rate=10.0
        )

        try:
            sensor = create_sensor(config)
            sensors.append(sensor)
            print(f"  ✓ Successfully created {type(sensor).__name__}")

            # Show sensor configuration
            print(f"  - Sensor ID: {sensor.config.sensor_id}")
            print(f"  - Sensor Type: {sensor.config.sensor_type.value}")
            print(f"  - Update Rate: {sensor.config.update_rate} Hz")
            print(f"  - Frame ID: {sensor.config.frame_id}")

        except Exception as e:
            print(f"  ✗ Failed to create sensor: {e}")

    print(f"\nSuccessfully created {len(sensors)} sensors using factory pattern")

    # Demonstrate sensor capabilities
    print("\nDemonstrating sensor data acquisition...")

    for sensor in sensors[:2]:  # Test first two sensors
        print(f"\nTesting {sensor.config.sensor_id}:")

        try:
            # Start sensor
            if sensor.start():
                print("  ✓ Sensor started successfully")

                # Wait for some data
                time.sleep(0.2)

                # Get latest data
                latest_data = sensor.get_latest_data(1)
                if latest_data:
                    packet = latest_data[0]
                    print(f"  ✓ Received data packet:")
                    print(f"    - Sensor ID: {packet.sensor_id}")
                    print(f"    - Timestamp: {packet.timestamp:.3f}")
                    print(f"    - Quality score: {packet.quality_score:.3f}")
                    print(f"    - Processing time: {packet.processing_time*1000:.1f} ms")

                    # Show data-specific info
                    if isinstance(packet.data, PointCloudData):
                        print(f"    - Point count: {packet.data.point_count}")
                        print(f"    - Has intensity: {packet.data.has_intensity}")
                    elif isinstance(packet.data, ImageData):
                        print(f"    - Image size: {packet.data.width}x{packet.data.height}")
                        print(f"    - Channels: {packet.data.channels}")
                else:
                    print("  - No data received")

                # Stop sensor
                sensor.stop()
                print("  ✓ Sensor stopped")
            else:
                print("  ✗ Failed to start sensor")

        except Exception as e:
            print(f"  ✗ Error during sensor operation: {e}")

    print("\nSensor factory demo completed.")


def main():
    """Main demonstration function."""
    print("SENSOR INPUT MODULE DEMONSTRATION")
    print("==================================")
    print("This demo showcases the capabilities of the Brain sensor input module")
    print("including multi-sensor management, synchronization, and data conversion.\n")

    # Set up logging
    setup_logging()

    try:
        # Run demonstrations
        demo_sensor_factory()
        demo_synchronization()
        demo_data_conversion()

        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe sensor input module is ready for integration with the")
        print("Brain cognitive world model system.")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()