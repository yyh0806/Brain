# Communication Layer Testing Framework

This document provides a comprehensive testing framework for the communication layer, including unit tests, integration tests, performance benchmarks, and load testing strategies.

## 1. Testing Architecture Overview

```python
import asyncio
import unittest
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Optional
import tempfile
import json
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    "simulation_mode": True,
    "mock_robot": True,
    "test_timeout": 30.0,
    "performance_iterations": 1000,
    "load_test_duration": 60.0,
    "concurrent_connections": 10
}
```

## 2. Unit Testing Suite

### 2.1 Robot Interface Tests

```python
import unittest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from brain.communication.robot_interface import (
    RobotInterface,
    ConnectionStatus,
    CommandResponse,
    PreflightCheckResult
)

class TestRobotInterface(unittest.IsolatedAsyncioTestCase):
    """Test suite for RobotInterface"""

    async def asyncSetUp(self):
        """Set up test fixtures before each test"""
        self.config = {
            "simulation": True,
            "heartbeat_timeout": 5.0,
            "command_port": 5555,
            "telemetry_port": 5556
        }
        self.interface = RobotInterface(self.config)
        await self.interface.connect()

    async def asyncTearDown(self):
        """Clean up after each test"""
        await self.interface.disconnect()

    async def test_connection_status(self):
        """Test connection status management"""
        self.assertEqual(self.interface.status, ConnectionStatus.CONNECTED)

        await self.interface.disconnect()
        self.assertEqual(self.interface.status, ConnectionStatus.DISCONNECTED)

    async def test_basic_commands(self):
        """Test basic robot commands"""
        # Test takeoff
        response = await self.interface.takeoff(altitude=10.0)
        self.assertIsInstance(response, CommandResponse)
        self.assertTrue(response.success)
        self.assertIn("altitude", response.data)
        self.assertEqual(response.data["altitude"], 10.0)

        # Test land
        response = await self.interface.land()
        self.assertTrue(response.success)
        self.assertTrue(response.data.get("landed", False))

    async def test_command_timeout(self):
        """Test command timeout handling"""
        # Patch _simulate_command to delay
        async def delayed_simulate(command, params):
            await asyncio.sleep(2.0)
            return CommandResponse(success=True)

        self.interface._simulate_command = delayed_simulate

        # Send command with short timeout
        response = await self.interface.send_command(
            "test_command",
            timeout=1.0
        )

        self.assertFalse(response.success)
        self.assertIn("超时", response.error)

    async def test_telemetry_callbacks(self):
        """Test telemetry callback registration and execution"""
        callback_called = False
        telemetry_data = None

        def telemetry_callback(telemetry):
            nonlocal callback_called, telemetry_data
            callback_called = True
            telemetry_data = telemetry

        # Register callback
        self.interface.on_telemetry(telemetry_callback)

        # Simulate telemetry update
        from brain.communication.message_types import TelemetryMessage
        test_telemetry = TelemetryMessage(
            message_id="test_telem",
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0
        )

        self.interface._handle_telemetry(test_telemetry)

        # Verify callback was called
        self.assertTrue(callback_called)
        self.assertIsNotNone(telemetry_data)
        self.assertEqual(telemetry_data.latitude, 37.7749)

    async def test_preflight_check(self):
        """Test preflight check functionality"""
        # In simulation mode, should pass
        result = await self.interface.preflight_check()
        self.assertIsInstance(result, PreflightCheckResult)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.issues), 0)

    async def test_wait_operations(self):
        """Test wait operations"""
        # Wait for altitude (simulation mode)
        result = await self.interface.wait_for_altitude(10.0, timeout=5.0)
        self.assertTrue(result)

        # Wait for position (simulation mode)
        result = await self.interface.wait_for_position(
            {"lat": 37.7749, "lon": -122.4194},
            timeout=5.0
        )
        self.assertTrue(result)

    async def test_concurrent_commands(self):
        """Test concurrent command execution"""
        # Send multiple commands concurrently
        commands = [
            self.interface.send_command("test1"),
            self.interface.send_command("test2"),
            self.interface.send_command("test3")
        ]

        responses = await asyncio.gather(*commands)

        # All should succeed
        for response in responses:
            self.assertTrue(response.success)

    async def test_error_handling(self):
        """Test error handling"""
        # Test with disconnected interface
        await self.interface.disconnect()

        response = await self.interface.send_command("takeoff")
        self.assertFalse(response.success)
        self.assertIn("未连接", response.error)

    async def test_status_reporting(self):
        """Test status reporting"""
        status = await self.interface.get_status()
        self.assertIsInstance(status, dict)
        self.assertIn("battery", status)
        self.assertIn("gps_status", status)
        self.assertIn("armed", status)


class TestRobotInterfacePerformance(unittest.IsolatedAsyncioTestCase):
    """Performance tests for RobotInterface"""

    async def asyncSetUp(self):
        self.interface = RobotInterface({"simulation": True})
        await self.interface.connect()

    async def test_command_throughput(self):
        """Test command throughput"""
        num_commands = 1000
        start_time = time.time()

        # Send commands sequentially
        for _ in range(num_commands):
            await self.interface.send_command("test_command")

        elapsed_time = time.time() - start_time
        throughput = num_commands / elapsed_time

        # Should handle at least 100 commands per second
        self.assertGreater(throughput, 100)

        print(f"Command throughput: {throughput:.2f} commands/sec")

    async def test_concurrent_command_performance(self):
        """Test concurrent command performance"""
        num_commands = 1000
        batch_size = 50

        start_time = time.time()

        # Send commands in batches
        for _ in range(num_commands // batch_size):
            batch = [
                self.interface.send_command("test_command")
                for _ in range(batch_size)
            ]
            await asyncio.gather(*batch)

        elapsed_time = time.time() - start_time
        throughput = num_commands / elapsed_time

        self.assertGreater(throughput, 200)

        print(f"Concurrent command throughput: {throughput:.2f} commands/sec")
```

### 2.2 ROS2 Interface Tests

```python
import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
from brain.communication.ros2_interface import (
    ROS2Interface,
    ROS2Config,
    ROS2Mode,
    TwistCommand,
    SensorData
)

class TestROS2Interface(unittest.IsolatedAsyncioTestCase):
    """Test suite for ROS2Interface"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.config = ROS2Config(
            mode=ROS2Mode.SIMULATION,
            node_name="test_brain_node"
        )
        self.ros2 = ROS2Interface(self.config)
        await self.ros2.initialize()

    async def asyncTearDown(self):
        """Clean up after tests"""
        await self.ros2.shutdown()

    async def test_initialization(self):
        """Test ROS2 interface initialization"""
        self.assertTrue(self.ros2.is_running())
        self.assertEqual(self.ros2.get_mode(), ROS2Mode.SIMULATION)

    async def test_twist_command_publishing(self):
        """Test Twist command publishing"""
        cmd = TwistCommand(linear_x=1.0, angular_z=0.5)
        await self.ros2.publish_twist(cmd)

        # In simulation mode, should update odometry
        odom = self.ros2.get_odometry()
        self.assertIsNotNone(odom)
        self.assertEqual(odom["linear_velocity"]["x"], 1.0)
        self.assertEqual(odom["angular_velocity"]["z"], 0.5)

    async def test_sensor_data_access(self):
        """Test sensor data access"""
        sensor_data = self.ros2.get_sensor_data()
        self.assertIsInstance(sensor_data, SensorData)

        # Check RGB image
        rgb_image = self.ros2.get_rgb_image()
        self.assertIsNotNone(rgb_image)
        self.assertIsInstance(rgb_image, np.ndarray)

        # Check odometry
        odometry = self.ros2.get_odometry()
        self.assertIsNotNone(odometry)
        self.assertIn("position", odometry)
        self.assertIn("orientation", odometry)

        # Check current pose
        pose = self.ros2.get_current_pose()
        self.assertIsInstance(pose, tuple)
        self.assertEqual(len(pose), 3)  # x, y, yaw

    async def test_twist_command_factory(self):
        """Test Twist command factory methods"""
        # Test forward
        cmd = TwistCommand.forward(2.0)
        self.assertEqual(cmd.linear_x, 2.0)
        self.assertEqual(cmd.angular_z, 0.0)

        # Test turn left
        cmd = TwistCommand.turn_left(1.0, 0.5)
        self.assertEqual(cmd.linear_x, 1.0)
        self.assertEqual(cmd.angular_z, 0.5)

        # Test stop
        cmd = TwistCommand.stop()
        self.assertEqual(cmd.linear_x, 0.0)
        self.assertEqual(cmd.angular_z, 0.0)

    async def test_sensor_callbacks(self):
        """Test sensor data callbacks"""
        callback_called = False
        received_data = None

        def sensor_callback(data):
            nonlocal callback_called, received_data
            callback_called = True
            received_data = data

        # Register callback
        self.ros2.register_sensor_callback("rgb_image", sensor_callback)

        # Update simulated image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.ros2.update_simulated_image(test_image)

        # Wait for callback
        await asyncio.sleep(0.1)

        self.assertTrue(callback_called)
        self.assertIsNotNone(received_data)
        np.testing.assert_array_equal(received_data.rgb_image, test_image)

    async def test_duration_commands(self):
        """Test duration-based commands"""
        cmd = TwistCommand(linear_x=1.0)

        start_time = time.time()
        await self.ros2.publish_twist_for_duration(cmd, duration=0.5, rate=10)

        elapsed = time.time() - start_time
        # Should take approximately 0.5 seconds
        self.assertAlmostEqual(elapsed, 0.5, delta=0.1)


class TestROS2RealMode(unittest.IsolatedAsyncioTestCase):
    """Test ROS2 interface in real mode (requires actual ROS2)"""

    @unittest.skipUnless(
        "ros2_available" in globals() and globals()["ros2_available"],
        "ROS2 not available"
    )
    async def test_real_mode_initialization(self):
        """Test real mode initialization"""
        config = ROS2Config(
            mode=ROS2Mode.REAL,
            node_name="test_real_node"
        )

        # Mock ROS2 components for testing
        with patch('rclpy.init'), \
             patch('rclpy.create_node') as mock_create_node, \
             patch('rclpy.ok', return_value=True):

            mock_node = Mock()
            mock_create_node.return_value = mock_node

            ros2 = ROS2Interface(config)
            await ros2.initialize()

            self.assertEqual(ros2.get_mode(), ROS2Mode.REAL)

            await ros2.shutdown()
```

### 2.3 Control Adapter Tests

```python
import unittest
import math
from unittest.mock import AsyncMock
import asyncio
from brain.communication.control_adapter import (
    ControlAdapter,
    PlatformType,
    PlatformCapabilities
)

class MockROS2Interface:
    """Mock ROS2 interface for testing"""
    def __init__(self):
        self.published_commands = []

    async def publish_twist(self, cmd):
        self.published_commands.append(cmd)

class TestControlAdapter(unittest.IsolatedAsyncioTestCase):
    """Test suite for ControlAdapter"""

    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.mock_ros2 = MockROS2Interface()
        self.ackermann_capabilities = PlatformCapabilities(
            platform_type=PlatformType.ACKERMANN,
            max_linear_speed=5.0,
            max_angular_speed=1.0,
            wheelbase=2.5
        )
        self.differential_capabilities = PlatformCapabilities(
            platform_type=PlatformType.DIFFERENTIAL,
            max_linear_speed=3.0,
            max_angular_speed=2.0,
            track_width=1.0
        )

    async def test_ackermann_control(self):
        """Test Ackermann platform control"""
        adapter = ControlAdapter(
            self.mock_ros2,
            PlatformType.ACKERMANN,
            self.ackermann_capabilities
        )

        # Test forward motion
        await adapter.set_velocity(2.0, 0.0)
        self.assertEqual(len(self.mock_ros2.published_commands), 1)
        cmd = self.mock_ros2.published_commands[0]
        self.assertEqual(cmd.linear_x, 2.0)
        self.assertEqual(cmd.angular_z, 0.0)

        # Test turn
        await adapter.set_velocity(1.0, 0.5)
        cmd = self.mock_ros2.published_commands[1]
        self.assertEqual(cmd.linear_x, 1.0)
        self.assertEqual(cmd.angular_z, 0.5)

    async def test_differential_control(self):
        """Test differential platform control"""
        adapter = ControlAdapter(
            self.mock_ros2,
            PlatformType.DIFFERENTIAL,
            self.differential_capabilities
        )

        # Test rotation
        await adapter.rotate_left(1.0)
        cmd = self.mock_ros2.published_commands[0]
        self.assertEqual(cmd.linear_x, 0.0)
        self.assertEqual(cmd.angular_z, 1.0)

        # Test forward motion
        await adapter.move_forward(2.0)
        cmd = self.mock_ros2.published_commands[1]
        self.assertEqual(cmd.linear_x, 2.0)
        self.assertEqual(cmd.angular_z, 0.0)

    async def test_speed_limiting(self):
        """Test speed limiting"""
        adapter = ControlAdapter(
            self.mock_ros2,
            PlatformType.ACKERMANN,
            self.ackermann_capabilities
        )

        # Try to exceed limits
        await adapter.set_velocity(10.0, 5.0)
        cmd = self.mock_ros2.published_commands[0]

        # Should be limited
        self.assertLessEqual(cmd.linear_x, self.ackermann_capabilities.max_linear_speed)
        self.assertLessEqual(abs(cmd.angular_z), self.ackermann_capabilities.max_angular_speed)

    def test_turn_radius_calculation(self):
        """Test turn radius calculation"""
        adapter = ControlAdapter(
            self.mock_ros2,
            PlatformType.ACKERMANN,
            self.ackermann_capabilities
        )

        # Straight line
        radius = adapter.compute_turn_radius(1.0, 0.0)
        self.assertEqual(radius, float('inf'))

        # Turning
        radius = adapter.compute_turn_radius(1.0, 0.5)
        self.assertEqual(radius, 2.0)  # 1.0 / 0.5 = 2.0

    def test_turn_planning(self):
        """Test intersection turn planning"""
        adapter = ControlAdapter(
            self.mock_ros2,
            PlatformType.ACKERMANN,
            self.ackermann_capabilities
        )

        # Test left turn
        plan = adapter.plan_turn_at_intersection("left")
        self.assertEqual(len(plan), 3)

        # Check turn segment
        _, angular = plan[1]
        self.assertGreater(angular, 0)  # Positive angular velocity for left turn

        # Test right turn
        plan = adapter.plan_turn_at_intersection("right")
        _, angular = plan[1]
        self.assertLess(angular, 0)  # Negative angular velocity for right turn
```

### 2.4 Message Types Tests

```python
import unittest
import json
from datetime import datetime
from brain.communication.message_types import (
    MessageType,
    CommandMessage,
    ResponseMessage,
    TelemetryMessage,
    parse_message
)

class TestMessageTypes(unittest.TestCase):
    """Test suite for message types"""

    def test_base_message_serialization(self):
        """Test base message serialization"""
        from brain.communication.message_types import BaseMessage

        msg = BaseMessage(
            message_id="test_001",
            message_type=MessageType.COMMAND,
            source="brain",
            target="robot"
        )

        # Test to_dict
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["message_id"], "test_001")
        self.assertEqual(msg_dict["message_type"], "command")

        # Test to_json
        msg_json = msg.to_json()
        parsed = json.loads(msg_json)
        self.assertEqual(parsed["message_id"], "test_001")

    def test_command_message(self):
        """Test CommandMessage functionality"""
        msg = CommandMessage(
            message_id="cmd_001",
            source="brain",
            target="robot",
            command="takeoff",
            parameters={"altitude": 10.0},
            timeout=30.0
        )

        # Test serialization
        msg_dict = msg.to_dict()
        self.assertEqual(msg_dict["command"], "takeoff")
        self.assertEqual(msg_dict["parameters"]["altitude"], 10.0)

        # Test deserialization
        restored = CommandMessage.from_dict(msg_dict)
        self.assertEqual(restored.command, "takeoff")
        self.assertEqual(restored.parameters["altitude"], 10.0)

    def test_telemetry_message(self):
        """Test TelemetryMessage functionality"""
        msg = TelemetryMessage(
            message_id="tel_001",
            source="robot",
            target="brain",
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0,
            battery=85.5
        )

        # Test serialization
        msg_dict = msg.to_dict()

        self.assertEqual(msg_dict["position"]["latitude"], 37.7749)
        self.assertEqual(msg_dict["position"]["longitude"], -122.4194)
        self.assertEqual(msg_dict["position"]["altitude"], 100.0)
        self.assertEqual(msg_dict["system"]["battery"], 85.5)

        # Test deserialization
        restored = TelemetryMessage.from_dict(msg_dict)
        self.assertEqual(restored.latitude, 37.7749)
        self.assertEqual(restored.longitude, -122.4194)
        self.assertEqual(restored.altitude, 100.0)
        self.assertEqual(restored.battery, 85.5)

    def test_message_parsing(self):
        """Test message parsing utility"""
        # Command message
        cmd_data = {
            "message_id": "cmd_001",
            "message_type": "command",
            "command": "land",
            "parameters": {}
        }

        msg = parse_message(cmd_data)
        self.assertIsInstance(msg, CommandMessage)
        self.assertEqual(msg.command, "land")

        # Response message
        resp_data = {
            "message_id": "resp_001",
            "message_type": "response",
            "success": True,
            "data": {"status": "completed"}
        }

        msg = parse_message(resp_data)
        self.assertIsInstance(msg, ResponseMessage)
        self.assertTrue(msg.success)

    def test_message_validation(self):
        """Test message validation"""
        # Command without command field should raise error
        with self.assertRaises(ValueError):
            CommandMessage(
                message_id="invalid",
                command="",  # Empty command
                source="brain",
                target="robot"
            )
```

## 3. Integration Testing

```python
import unittest
import asyncio
import tempfile
import yaml
from pathlib import Path
from brain.communication.robot_interface import RobotInterface
from brain.communication.ros2_interface import ROS2Interface, ROS2Config
from brain.communication.control_adapter import ControlAdapter, PlatformType

class TestCommunicationIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for communication components"""

    async def asyncSetUp(self):
        """Set up integration test environment"""
        # Create test configuration
        self.test_config = {
            "simulation": True,
            "robot": {
                "platform_type": "ackermann",
                "kinematics": {
                    "max_linear_speed": 2.0,
                    "max_angular_speed": 1.0
                }
            }
        }

    async def test_full_command_flow(self):
        """Test complete command flow from RobotInterface through ROS2"""
        # Initialize components
        robot_interface = RobotInterface(self.test_config)
        ros2_config = ROS2Config(mode=ROS2Mode.SIMULATION)
        ros2_interface = ROS2Interface(ros2_config)

        await robot_interface.connect()
        await ros2_interface.initialize()

        # Send command through robot interface
        response = await robot_interface.send_command("test_move")
        self.assertTrue(response.success)

        # Check that ROS2 interface received commands
        # (This would require connecting the components in real implementation)

        await ros2_interface.shutdown()
        await robot_interface.disconnect()

    async def test_telemetry_flow(self):
        """Test telemetry data flow"""
        robot_interface = RobotInterface(self.test_config)
        await robot_interface.connect()

        # Simulate telemetry reception
        telemetry_received = False

        def telemetry_handler(telemetry):
            nonlocal telemetry_received
            telemetry_received = True

        robot_interface.on_telemetry(telemetry_handler)

        # Simulate telemetry update
        from brain.communication.message_types import TelemetryMessage
        test_telemetry = TelemetryMessage(
            message_id="test_tel",
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0
        )

        robot_interface._handle_telemetry(test_telemetry)

        # Verify telemetry was processed
        self.assertTrue(telemetry_received)
        self.assertEqual(robot_interface.latest_telemetry, test_telemetry)

        await robot_interface.disconnect()
```

## 4. Performance Testing Framework

```python
import asyncio
import time
import statistics
import psutil
import threading
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

class PerformanceTester:
    """Performance testing framework"""

    def __init__(self):
        self.results = {}
        self.monitoring = False
        self.monitor_thread = None
        self.cpu_samples = []
        self.memory_samples = []

    async def measure_throughput(
        self,
        func: Callable,
        iterations: int = 1000,
        concurrency: int = 1
    ) -> Dict[str, Any]:
        """Measure function throughput"""
        start_time = time.time()

        if concurrency == 1:
            # Sequential execution
            for _ in range(iterations):
                await func()
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(concurrency)

            async def limited_func():
                async with semaphore:
                    return await func()

            tasks = [limited_func() for _ in range(iterations)]
            await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time
        throughput = iterations / elapsed_time

        return {
            "iterations": iterations,
            "elapsed_time": elapsed_time,
            "throughput": throughput,
            "concurrency": concurrency
        }

    async def measure_latency(
        self,
        func: Callable,
        iterations: int = 100
    ) -> Dict[str, Any]:
        """Measure function latency statistics"""
        latencies = []

        for _ in range(iterations):
            start = time.time()
            await func()
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)

        return {
            "iterations": iterations,
            "latency_avg": statistics.mean(latencies),
            "latency_min": min(latencies),
            "latency_max": max(latencies),
            "latency_p50": statistics.median(latencies),
            "latency_p95": self._percentile(latencies, 95),
            "latency_p99": self._percentile(latencies, 99)
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def start_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()

        def monitor():
            while self.monitoring:
                self.cpu_samples.append(psutil.cpu_percent())
                self.memory_samples.append(psutil.virtual_memory().percent)
                time.sleep(0.1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        self.monitoring = False

        return {
            "cpu_avg": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "cpu_max": max(self.cpu_samples) if self.cpu_samples else 0,
            "memory_avg": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "memory_max": max(self.memory_samples) if self.memory_samples else 0
        }

class PerformanceTestSuite(unittest.IsolatedAsyncioTestCase):
    """Performance test suite"""

    async def asyncSetUp(self):
        """Set up performance tests"""
        self.tester = PerformanceTester()

        # Initialize components for testing
        self.robot_interface = RobotInterface({"simulation": True})
        await self.robot_interface.connect()

        self.ros2_interface = ROS2Interface(ROS2Config(mode=ROS2Mode.SIMULATION))
        await self.ros2_interface.initialize()

    async def test_command_throughput(self):
        """Test command throughput performance"""
        print("\n=== Command Throughput Test ===")

        # Test sequential throughput
        throughput_result = await self.tester.measure_throughput(
            lambda: self.robot_interface.send_command("test_command"),
            iterations=1000,
            concurrency=1
        )
        print(f"Sequential throughput: {throughput_result['throughput']:.2f} cmd/sec")

        # Test concurrent throughput
        throughput_result = await self.tester.measure_throughput(
            lambda: self.robot_interface.send_command("test_command"),
            iterations=1000,
            concurrency=10
        )
        print(f"Concurrent throughput (10): {throughput_result['throughput']:.2f} cmd/sec")

        # Assert minimum throughput requirement
        self.assertGreater(throughput_result['throughput'], 100)

    async def test_ros2_publish_throughput(self):
        """Test ROS2 publish throughput"""
        print("\n=== ROS2 Publish Throughput Test ===")

        cmd = TwistCommand(linear_x=1.0, angular_z=0.5)

        throughput_result = await self.tester.measure_throughput(
            lambda: self.ros2_interface.publish_twist(cmd),
            iterations=5000,
            concurrency=1
        )

        print(f"ROS2 publish throughput: {throughput_result['throughput']:.2f} pub/sec")

        # Should handle at least 1000 publishes per second
        self.assertGreater(throughput_result['throughput'], 1000)

    async def test_message_serialization_latency(self):
        """Test message serialization latency"""
        print("\n=== Message Serialization Latency Test ===")

        from brain.communication.message_types import TelemetryMessage

        # Create test message
        telemetry = TelemetryMessage(
            message_id="test",
            latitude=37.7749,
            longitude=-122.4194,
            altitude=100.0
        )

        # Test JSON serialization
        latency_result = await self.tester.measure_latency(
            lambda: telemetry.to_json(),
            iterations=1000
        )

        print(f"JSON serialization latency: {latency_result['latency_avg']:.2f}ms avg")
        print(f"  95th percentile: {latency_result['latency_p95']:.2f}ms")

        # JSON serialization should be fast
        self.assertLess(latency_result['latency_avg'], 1.0)  # < 1ms average

    async def test_system_resource_usage(self):
        """Test system resource usage under load"""
        print("\n=== System Resource Usage Test ===")

        self.tester.start_monitoring()

        # Run high load
        tasks = [
            self.robot_interface.send_command(f"test_{i}")
            for i in range(100)
        ]
        await asyncio.gather(*tasks)

        resource_stats = self.tester.stop_monitoring()

        print(f"CPU usage - Avg: {resource_stats['cpu_avg']:.1f}%, Max: {resource_stats['cpu_max']:.1f}%")
        print(f"Memory usage - Avg: {resource_stats['memory_avg']:.1f}%, Max: {resource_stats['memory_max']:.1f}%")

        # CPU usage should be reasonable
        self.assertLess(resource_stats['cpu_avg'], 50.0)
        self.assertLess(resource_stats['memory_avg'], 100.0)
```

## 5. Load Testing Framework

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import statistics

class LoadTester:
    """Load testing framework for communication endpoints"""

    def __init__(self, base_url: str = "ws://localhost:8765"):
        self.base_url = base_url
        self.results = []

    async def run_load_test(
        self,
        duration: float,
        concurrent_connections: int,
        messages_per_second: float
    ) -> Dict[str, Any]:
        """Run load test with specified parameters"""
        print(f"\nRunning load test: {duration}s, {concurrent_connections} connections, {messages_per_second} msg/s")

        start_time = time.time()
        end_time = start_time + duration

        # Create connection tasks
        tasks = [
            self._run_connection(duration, messages_per_second / concurrent_connections)
            for _ in range(concurrent_connections)
        ]

        # Run all connections
        connection_results = await asyncio.gather(*tasks)

        # Aggregate results
        all_latencies = []
        all_messages = 0
        all_errors = 0

        for result in connection_results:
            all_latencies.extend(result["latencies"])
            all_messages += result["messages_sent"]
            all_errors += result["errors"]

        actual_duration = time.time() - start_time

        return {
            "duration": actual_duration,
            "concurrent_connections": concurrent_connections,
            "target_rps": messages_per_second,
            "actual_rps": all_messages / actual_duration,
            "total_messages": all_messages,
            "total_errors": all_errors,
            "error_rate": all_errors / all_messages if all_messages > 0 else 0,
            "latency_avg": statistics.mean(all_latencies) if all_latencies else 0,
            "latency_p95": self._percentile(all_latencies, 95) if all_latencies else 0,
            "latency_p99": self._percentile(all_latencies, 99) if all_latencies else 0
        }

    async def _run_connection(self, duration: float, rps: float) -> Dict[str, Any]:
        """Run a single connection for load testing"""
        latencies = []
        messages_sent = 0
        errors = 0

        try:
            # Connect to WebSocket
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.base_url) as ws:
                    start_time = time.time()
                    message_interval = 1.0 / rps if rps > 0 else 1.0
                    next_message = start_time

                    while time.time() - start_time < duration:
                        current_time = time.time()

                        if current_time >= next_message:
                            # Send message
                            msg_start = time.time()

                            try:
                                await ws.send_json({
                                    "type": "test_message",
                                    "timestamp": current_time,
                                    "message_id": f"msg_{messages_sent}"
                                })

                                # Wait for response
                                response = await ws.receive_json(timeout=1.0)

                                latency = (time.time() - msg_start) * 1000
                                latencies.append(latency)
                                messages_sent += 1

                                # Schedule next message
                                next_message = current_time + message_interval

                            except Exception as e:
                                errors += 1
                                print(f"Error: {e}")

                        # Small sleep to prevent busy waiting
                        await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Connection error: {e}")
            errors += 1

        return {
            "latencies": latencies,
            "messages_sent": messages_sent,
            "errors": errors
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestLoadScenarios(unittest.TestCase):
    """Load test scenarios"""

    def test_websocket_load(self):
        """Test WebSocket server under load"""
        async def run_test():
            load_tester = LoadTester()

            # Test scenarios
            scenarios = [
                (30.0, 10, 50),   # 30s, 10 connections, 50 RPS
                (30.0, 50, 100),  # 30s, 50 connections, 100 RPS
                (60.0, 100, 200), # 60s, 100 connections, 200 RPS
            ]

            for duration, connections, rps in scenarios:
                result = await load_tester.run_load_test(duration, connections, rps)

                print(f"\nScenario: {duration}s, {connections} connections, {rps} RPS")
                print(f"  Actual RPS: {result['actual_rps']:.2f}")
                print(f"  Error Rate: {result['error_rate']:.2%}")
                print(f"  Avg Latency: {result['latency_avg']:.2f}ms")
                print(f"  95th Percentile: {result['latency_p95']:.2f}ms")
                print(f"  99th Percentile: {result['latency_p99']:.2f}ms")

                # Assert performance requirements
                self.assertLess(result['error_rate'], 0.01)  # < 1% error rate
                self.assertLess(result['latency_avg'], 100)  # < 100ms average

        asyncio.run(run_test())
```

## 6. Test Execution and Reporting

```python
import unittest
import sys
import time
from io import StringIO
from datetime import datetime
import json

class TestRunner:
    """Custom test runner with detailed reporting"""

    def __init__(self):
        self.results = {}

    def run_tests(self, test_modules: List[str]) -> Dict[str, Any]:
        """Run tests and generate report"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Load test modules
        for module_name in test_modules:
            try:
                module = __import__(module_name)
                suite.addTests(loader.loadTestsFromModule(module))
            except ImportError as e:
                print(f"Warning: Could not load module {module_name}: {e}")

        # Run tests
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2,
            buffer=True
        )

        start_time = time.time()
        result = runner.run(suite)
        elapsed_time = time.time() - start_time

        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration": elapsed_time,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
            "output": stream.getvalue(),
            "details": {
                "failures": [
                    {"test": str(test), "error": error}
                    for test, error in result.failures
                ],
                "errors": [
                    {"test": str(test), "error": error}
                    for test, error in result.errors
                ]
            }
        }

        return report

    def save_report(self, report: Dict[str, Any], filename: str):
        """Save test report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("TEST REPORT SUMMARY")
        print(f"{'='*60}")
        print(f"Tests Run: {report['tests_run']}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Failures: {report['failures']}")
        print(f"Errors: {report['errors']}")
        print(f"Duration: {report['duration']:.2f}s")
        print(f"\nReport saved to: {filename}")


# Usage example:
if __name__ == "__main__":
    # Create test runner
    runner = TestRunner()

    # Define test modules
    test_modules = [
        "test_robot_interface",
        "test_ros2_interface",
        "test_control_adapter",
        "test_message_types",
        "test_performance"
    ]

    # Run tests
    report = runner.run_tests(test_modules)

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"test_report_{timestamp}.json"
    runner.save_report(report, report_filename)
```

## 7. Continuous Integration Configuration

```yaml
# .github/workflows/test_communication.yml
name: Communication Layer Tests

on:
  push:
    branches: [ main, develop ]
    paths: [ 'brain/communication/**' ]
  pull_request:
    branches: [ main ]
    paths: [ 'brain/communication/**' ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
        pip install pytest-benchmark

    - name: Run unit tests
      run: |
        pytest tests/communication/test_unit/ -v --cov=brain.communication --cov-report=xml

    - name: Run integration tests
      run: |
        pytest tests/communication/test_integration/ -v

    - name: Run performance tests
      run: |
        pytest tests/communication/test_performance/ -v --benchmark-only

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: communication
        name: codecov-communication

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          test_reports/
          .benchmarks/
```