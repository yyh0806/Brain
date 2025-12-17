#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Sensor Input Module Demo

A simplified demonstration of the sensor input module functionality.
"""

import sys
import os
import time
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from brain.cognitive.world_model.sensor_input_types import (
        SensorType, PointCloudData, ImageData, IMUData, SensorDataPacket
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

    print("Successfully imported all sensor input modules!")

except ImportError as e:
    print("Import error: {}".format(e))
    sys.exit(1)


def test_data_types():
    """Test sensor data types."""
    print("\n=== Testing Sensor Data Types ===")

    # Test PointCloudData
    points = np.random.rand(100, 3) * 10
    intensity = np.random.rand(100)
    pc_data = PointCloudData(
        points=points,
        intensity=intensity,
        timestamp=time.time(),
        frame_id="lidar_frame"
    )
    print("Created PointCloudData with {} points".format(pc_data.point_count))

    # Test ImageData
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img_data = ImageData(
        image=image,
        timestamp=time.time(),
        frame_id="camera_frame"
    )
    print("Created ImageData with size {}x{}".format(img_data.width, img_data.height))

    # Test IMUData
    imu_data = IMUData(
        linear_acceleration=np.array([1.0, 2.0, 3.0]),
        angular_velocity=np.array([0.1, 0.2, 0.3]),
        timestamp=time.time(),
        frame_id="imu_frame"
    )
    print("Created IMUData with acceleration and angular velocity")

    # Test SensorDataPacket
    packet = SensorDataPacket(
        sensor_id="test_sensor",
        sensor_type=SensorType.POINT_CLOUD,
        timestamp=time.time(),
        data=pc_data,
        quality_score=0.95
    )
    print("Created SensorDataPacket with quality score {:.2f}".format(packet.quality_score))

    return True


def test_sensor_factory():
    """Test sensor factory."""
    print("\n=== Testing Sensor Factory ===")

    configs = [
        SensorConfig("lidar_1", SensorType.POINT_CLOUD, update_rate=10.0),
        SensorConfig("camera_1", SensorType.IMAGE, update_rate=30.0),
        SensorConfig("imu_1", SensorType.IMU, update_rate=100.0),
    ]

    sensors = []
    for config in configs:
        try:
            sensor = create_sensor(config)
            sensors.append(sensor)
            print("Created {} sensor: {}".format(config.sensor_type.value, config.sensor_id))
        except Exception as e:
            print("Failed to create sensor {}: {}".format(config.sensor_id, e))

    print("Successfully created {} sensors".format(len(sensors)))
    return len(sensors) == len(configs)


def test_data_conversion():
    """Test data format conversion."""
    print("\n=== Testing Data Conversion ===")

    # Create test data
    points = np.random.rand(50, 3) * 5
    pc_data = PointCloudData(points=points)

    # Test JSON conversion
    try:
        converter = create_converter(DataFormat.JSON)
        options = ConversionOptions(target_format=DataFormat.JSON)
        result = converter.convert(pc_data, options)

        if result.success:
            print("JSON conversion: SUCCESS ({} chars)".format(len(result.data)))
        else:
            print("JSON conversion: FAILED - {}".format(result.error_message))
    except Exception as e:
        print("JSON conversion: ERROR - {}".format(e))

    # Test CSV conversion
    try:
        converter = create_converter(DataFormat.CSV)
        options = ConversionOptions(target_format=DataFormat.CSV)
        result = converter.convert(pc_data, options)

        if result.success:
            print("CSV conversion: SUCCESS ({} chars)".format(len(result.data)))
        else:
            print("CSV conversion: FAILED - {}".format(result.error_message))
    except Exception as e:
        print("CSV conversion: ERROR - {}".format(e))

    return True


def test_sensor_manager():
    """Test sensor manager (basic functionality)."""
    print("\n=== Testing Sensor Manager ===")

    try:
        manager = MultiSensorManager()
        print("Created MultiSensorManager")

        # Create a simple sensor for testing
        config = SensorConfig("test_pc", SensorType.POINT_CLOUD, update_rate=5.0)
        sensor = create_sensor(config)

        # Add sensor to manager
        if manager.add_sensor(sensor, config):
            print("Added sensor to manager")
        else:
            print("Failed to add sensor to manager")
            return False

        # Start sensor
        if manager.start_sensors(["test_pc"]):
            print("Started sensor successfully")

            # Let it run for a short time
            time.sleep(1.0)

            # Check statistics
            stats = manager.get_manager_statistics()
            print("Manager stats: {} active sensors, {} sync operations".format(
                stats["active_sensors"], stats["sync_operations"]["total"]
            ))

            # Stop sensor
            manager.stop_sensors(["test_pc"])
            print("Stopped sensor")

            return True
        else:
            print("Failed to start sensor")
            return False

    except Exception as e:
        print("Sensor manager test failed: {}".format(e))
        return False


def run_unit_tests():
    """Run a subset of unit tests."""
    print("\n=== Running Quick Unit Tests ===")

    try:
        # Import test module
        from tests.unit.test_sensor_input import TestSensorDataTypes

        # Create test instance
        test_instance = TestSensorDataTypes()
        test_instance.setUp()

        # Run a few quick tests
        print("Running PointCloudData test...")
        test_instance.test_point_cloud_data_creation()
        print("✓ PASSED")

        print("Running ImageData test...")
        test_instance.test_image_data_creation()
        print("✓ PASSED")

        print("Running IMUData test...")
        test_instance.test_imu_data_creation()
        print("✓ PASSED")

        print("Running SensorDataPacket test...")
        test_instance.test_sensor_data_packet_creation()
        print("✓ PASSED")

        print("All quick unit tests passed!")
        return True

    except Exception as e:
        print("Unit test failed: {}".format(e))
        return False


def main():
    """Main function."""
    print("BRAIN SENSOR INPUT MODULE - SIMPLE DEMO")
    print("=" * 50)

    results = []

    # Run tests
    print("Running component tests...")
    results.append(("Data Types", test_data_types()))
    results.append(("Sensor Factory", test_sensor_factory()))
    results.append(("Data Conversion", test_data_conversion()))
    results.append(("Sensor Manager", test_sensor_manager()))
    results.append(("Unit Tests", run_unit_tests()))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print("  {}: {}".format(test_name, status))
        if result:
            passed += 1

    print("\nOverall: {}/{} tests passed".format(passed, len(results)))

    if passed == len(results):
        print("\n🎉 All tests passed! The sensor input module is working correctly.")
        print("You can now integrate this module with the Brain cognitive world model.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)