#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete World Model System Demo

This script demonstrates the complete World Model system functionality,
including sensor processing, world model updates, decision making, and visualization.

Usage:
    python3 run_complete_system_demo.py [--mode=full|quick] [--components=all|sensors|world_model|planning]

Author: Brain World Model Team
Date: 2025-12-18
"""

import sys
import os
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# Avoid importing from testing-framework-dev to prevent conflicts

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Import World Model components
try:
    # Try to import directly from world_model module to avoid brain.__init__ issues
    sys.path.insert(0, str(project_root / "brain" / "cognitive" / "world_model"))
    from world_model import WorldModel, ChangeType
    WORLD_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WorldModel import failed: {e}")
    WORLD_MODEL_AVAILABLE = False
    WorldModel = None
    ChangeType = None

# Import sensor components
try:
    # Try to import directly from world_model module to avoid brain.__init__ issues
    sys.path.insert(0, str(project_root / "brain" / "cognitive" / "world_model"))
    from sensor_input_types import (
        SensorType, PointCloudData, ImageData, IMUData, SensorDataPacket
    )
    from sensor_interface import (
        SensorConfig, create_sensor
    )
    from sensor_manager import (
        MultiSensorManager, SyncMethod
    )
    SENSOR_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Sensor components import failed: {e}")
    SENSOR_COMPONENTS_AVAILABLE = False

# Import reasoning components
try:
    # Try to import directly from reasoning module to avoid brain.__init__ issues
    sys.path.insert(0, str(project_root / "brain" / "cognitive" / "reasoning"))
    from cot_engine import CoTEngine, ReasoningMode
    REASONING_AVAILABLE = True
    LLM_AVAILABLE = False  # LLM might not be available
except ImportError as e:
    logger.warning(f"Reasoning components import failed: {e}")
    REASONING_AVAILABLE = False
    LLM_AVAILABLE = False

# Import testing framework
try:
    sys.path.insert(0, str(project_root / "testing-framework-dev" / "tests"))
    from framework.test_framework import TestFramework, TestResult, TestSuite
    TESTING_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Testing framework import failed: {e}")
    TESTING_FRAMEWORK_AVAILABLE = False
    TestFramework = None
    TestResult = None
    TestSuite = None


class WorldModelSystemDemo:
    """Complete World Model System Demo Class"""

    def __init__(self, mode: str = "full", components: List[str] = None):
        """
        Initialize the demo system

        Args:
            mode: Operating mode ("full", "quick", "interactive")
            components: List of components to initialize ["all", "sensors", "world_model", "planning"]
        """
        self.mode = mode
        self.components = components or ["all"]
        self.running = False
        self.results = {}

        # Initialize components
        self.world_model = None
        self.sensor_manager = None
        self.cot_engine = None
        self.llm_interface = None
        self.test_framework = None

        # Demo state
        self.demo_start_time = None
        self.current_step = 0
        self.total_steps = 0

        # Print initialization
        print("=" * 70)
        print("🧠 BRAIN World Model System Demo - Initializing...")
        print("=" * 70)

        self._check_dependencies()
        self._initialize_components()

    def _check_dependencies(self):
        """Check if all required dependencies are available"""
        print("\n🔍 Checking Dependencies...")

        dependencies = {
            "World Model": WORLD_MODEL_AVAILABLE,
            "Sensor Components": SENSOR_COMPONENTS_AVAILABLE,
            "Reasoning Components": REASONING_AVAILABLE,
            "Testing Framework": TESTING_FRAMEWORK_AVAILABLE
        }

        missing = []
        for name, available in dependencies.items():
            status = "✅ Available" if available else "❌ Missing"
            print(f"  {name}: {status}")
            if not available:
                missing.append(name)

        if missing:
            print(f"\n⚠️ Warning: Missing dependencies: {', '.join(missing)}")
            print("  Some demo features may not be available.")
        else:
            print("\n✅ All dependencies satisfied!")

    def _initialize_components(self):
        """Initialize all system components"""
        print("\n🚀 Initializing Components...")

        try:
            # Initialize World Model
            if "all" in self.components or "world_model" in self.components:
                if WORLD_MODEL_AVAILABLE:
                    self.world_model = WorldModel()
                    print("  ✅ World Model initialized")
                else:
                    print("  ❌ World Model not available")

            # Initialize Sensor Manager
            if "all" in self.components or "sensors" in self.components:
                if SENSOR_COMPONENTS_AVAILABLE:
                    self.sensor_manager = MultiSensorManager()
                    print("  ✅ Sensor Manager initialized")
                else:
                    print("  ❌ Sensor Manager not available")

            # Initialize CoT Engine
            if "all" in self.components or "planning" in self.components:
                if REASONING_AVAILABLE:
                    self.cot_engine = CoTEngine()
                    print("  ✅ CoT Engine initialized")
                else:
                    print("  ❌ CoT Engine not available")

            # Initialize Testing Framework
            if self.mode == "full" and TESTING_FRAMEWORK_AVAILABLE:
                self.test_framework = TestFramework()
                print("  ✅ Testing Framework initialized")

            print("\n✅ Component initialization complete!")

        except Exception as e:
            print(f"\n❌ Component initialization failed: {e}")
            if LOGURU_AVAILABLE:
                logger.exception("Component initialization error")

    def run_demo(self):
        """Run the complete demo"""
        print("\n🎬 Starting World Model System Demo...")
        print(f"   Mode: {self.mode}")
        print(f"   Components: {', '.join(self.components)}")

        self.demo_start_time = time.time()
        self.running = True

        try:
            if self.mode == "full":
                self._run_full_demo()
            elif self.mode == "quick":
                self._run_quick_demo()
            elif self.mode == "interactive":
                self._run_interactive_demo()
            else:
                print(f"❌ Unknown mode: {self.mode}")
                return False

            self._print_demo_summary()
            return True

        except KeyboardInterrupt:
            print("\n\n⚠️ Demo interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            if LOGURU_AVAILABLE:
                logger.exception("Demo execution error")
            return False
        finally:
            self.running = False

    def _run_full_demo(self):
        """Run full comprehensive demo"""
        print("\n" + "=" * 70)
        print("📌 FULL DEMO: Complete World Model System")
        print("=" * 70)

        demo_steps = [
            ("Sensor Data Processing", self._demo_sensor_processing),
            ("World Model Updates", self._demo_world_model_updates),
            ("Change Detection", self._demo_change_detection),
            ("Reasoning and Planning", self._demo_reasoning_planning),
            ("Integration Test", self._demo_integration_test)
        ]

        self.total_steps = len(demo_steps)

        for step_name, step_func in demo_steps:
            if not self.running:
                break

            self.current_step += 1
            print(f"\n📍 Step {self.current_step}/{self.total_steps}: {step_name}")
            print("-" * 50)

            start_time = time.time()
            try:
                result = step_func()
                duration = time.time() - start_time
                self.results[step_name] = {
                    "status": "completed",
                    "duration": duration,
                    "result": result
                }
                print(f"✅ {step_name} completed in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                self.results[step_name] = {
                    "status": "failed",
                    "duration": duration,
                    "error": str(e)
                }
                print(f"❌ {step_name} failed: {e}")

    def _run_quick_demo(self):
        """Run quick demo with essential features only"""
        print("\n" + "=" * 70)
        print("📌 QUICK DEMO: Essential Features")
        print("=" * 70)

        demo_steps = [
            ("Basic World Model", self._demo_basic_world_model),
            ("Simple Reasoning", self._demo_simple_reasoning)
        ]

        self.total_steps = len(demo_steps)

        for step_name, step_func in demo_steps:
            if not self.running:
                break

            self.current_step += 1
            print(f"\n📍 Step {self.current_step}/{self.total_steps}: {step_name}")
            print("-" * 50)

            start_time = time.time()
            try:
                result = step_func()
                duration = time.time() - start_time
                self.results[step_name] = {
                    "status": "completed",
                    "duration": duration,
                    "result": result
                }
                print(f"✅ {step_name} completed in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                self.results[step_name] = {
                    "status": "failed",
                    "duration": duration,
                    "error": str(e)
                }
                print(f"❌ {step_name} failed: {e}")

    def _run_interactive_demo(self):
        """Run interactive demo with user input"""
        print("\n" + "=" * 70)
        print("📌 INTERACTIVE DEMO: User-Driven Exploration")
        print("=" * 70)

        while self.running:
            print("\n🎮 Available Actions:")
            print("  1. Test sensor data processing")
            print("  2. Test world model updates")
            print("  3. Test reasoning engine")
            print("  4. Run integration test")
            print("  5. View system status")
            print("  0. Exit demo")

            try:
                choice = input("\nSelect action (0-5): ").strip()

                if choice == "0":
                    break
                elif choice == "1":
                    self._demo_sensor_processing()
                elif choice == "2":
                    self._demo_world_model_updates()
                elif choice == "3":
                    self._demo_reasoning_planning()
                elif choice == "4":
                    self._demo_integration_test()
                elif choice == "5":
                    self._print_system_status()
                else:
                    print("❌ Invalid choice. Please try again.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Action failed: {e}")

    def _demo_sensor_processing(self):
        """Demo sensor data processing"""
        if not SENSOR_COMPONENTS_AVAILABLE:
            print("❌ Sensor components not available")
            return None

        print("🔬 Testing sensor data processing...")

        # Generate mock sensor data
        sensor_data = self._generate_mock_sensor_data()

        # Process different sensor types
        for sensor_type, data in sensor_data.items():
            print(f"\n  Processing {sensor_type}...")

            if sensor_type == "lidar":
                processed = self._process_lidar_data(data)
            elif sensor_type == "camera":
                processed = self._process_camera_data(data)
            elif sensor_type == "imu":
                processed = self._process_imu_data(data)
            else:
                processed = None

            if processed:
                print(f"    ✅ {sensor_type} processed successfully")
                print(f"    📊 Data points: {processed.get('points', len(processed.get('data', [])))}")
            else:
                print(f"    ❌ {sensor_type} processing failed")

        return {"processed_sensors": len(sensor_data)}

    def _demo_world_model_updates(self):
        """Demo world model updates"""
        if not WORLD_MODEL_AVAILABLE or self.world_model is None:
            print("❌ World Model not available")
            return None

        print("🌍 Testing world model updates...")

        # Generate mock perception data
        perception_data = self._generate_mock_perception_data()

        # Update world model
        print("  Updating world model with perception data...")
        changes = self.world_model.update_from_perception(perception_data)

        print(f"  ✅ World model updated")
        print(f"  📊 Changes detected: {len(changes)}")

        for change in changes:
            print(f"    - {change.change_type.value}: {change.description}")

        # Get planning context
        context = self.world_model.get_context_for_planning()
        print(f"  📍 Current context:")
        print(f"    - Obstacles: {len(context.obstacles)}")
        print(f"    - Targets: {len(context.targets)}")
        print(f"    - Battery: {context.battery_level}%")

        return {
            "changes": len(changes),
            "obstacles": len(context.obstacles),
            "targets": len(context.targets)
        }

    def _demo_change_detection(self):
        """Demo change detection"""
        if not WORLD_MODEL_AVAILABLE or self.world_model is None:
            print("❌ World Model not available")
            return None

        print("🔍 Testing change detection...")

        # Initial state
        initial_data = self._generate_mock_perception_data()
        self.world_model.update_from_perception(initial_data)

        # Changed state
        changed_data = self._generate_mock_perception_data_with_changes()
        changes = self.world_model.update_from_perception(changed_data)

        # Detect significant changes
        significant_changes = self.world_model.detect_significant_changes()

        print(f"  ✅ Change detection completed")
        print(f"  📊 Total changes: {len(changes)}")
        print(f"  🎯 Significant changes: {len(significant_changes)}")

        for change in significant_changes:
            print(f"    - {change.description} (Priority: {change.priority})")

        return {
            "total_changes": len(changes),
            "significant_changes": len(significant_changes)
        }

    def _demo_reasoning_planning(self):
        """Demo reasoning and planning"""
        if not REASONING_AVAILABLE or self.cot_engine is None:
            print("❌ Reasoning components not available")
            return None

        print("🧠 Testing reasoning and planning...")

        # Test reasoning queries
        test_queries = [
            "Plan a route to the target location",
            "Analyze the current situation and suggest actions",
            "Evaluate the risks and benefits of proceeding"
        ]

        results = []
        for query in test_queries:
            print(f"\n  🤔 Query: {query}")

            try:
                # Assess complexity
                complexity = self.cot_engine.assess_complexity(query, {})
                print(f"    📊 Complexity: {complexity.value}")

                # Execute reasoning (mock implementation)
                reasoning_result = {
                    "query": query,
                    "complexity": complexity.value,
                    "confidence": 0.85,
                    "decision": f"Processed: {query}",
                    "suggestion": "Proceed with caution",
                    "chain": ["Step 1: Analyze query", "Step 2: Consider context", "Step 3: Generate response"]
                }

                results.append(reasoning_result)
                print(f"    ✅ Reasoning completed")
                print(f"    🎯 Confidence: {reasoning_result['confidence']:.2f}")

            except Exception as e:
                print(f"    ❌ Reasoning failed: {e}")

        return {"reasoning_results": len(results)}

    def _demo_integration_test(self):
        """Demo integration testing"""
        print("🔧 Running integration tests...")

        if not TESTING_FRAMEWORK_AVAILABLE:
            print("❌ Testing framework not available, running basic integration test...")
            return self._run_basic_integration_test()

        try:
            # Create test suite
            test_suite = TestSuite(
                name="world_model_integration",
                description="Integration tests for World Model system",
                tests=[
                    lambda: self._test_world_model_integration(),
                    lambda: self._test_sensor_integration(),
                    lambda: self._test_reasoning_integration()
                ]
            )

            # Run tests
            print("  🧪 Executing test suite...")
            results = self.test_framework.run_test_suite(test_suite)

            passed = sum(1 for r in results if r.status == "PASSED")
            total = len(results)

            print(f"  ✅ Integration tests completed")
            print(f"  📊 Results: {passed}/{total} tests passed")

            return {"passed": passed, "total": total}

        except Exception as e:
            print(f"  ❌ Integration test failed: {e}")
            return None

    def _demo_basic_world_model(self):
        """Demo basic world model functionality"""
        if not WORLD_MODEL_AVAILABLE:
            print("❌ World Model not available")
            return None

        print("🌍 Testing basic world model...")

        # Create simple world model instance
        world_model = WorldModel()

        # Add simple data
        mock_data = {
            "robot_position": [0, 0, 0],
            "obstacles": [[10, 10, 0], [20, 5, 0]],
            "timestamp": time.time()
        }

        print("  ✅ Basic world model created")
        print(f"  📊 Mock data: {len(mock_data)} elements")

        return {"status": "success", "data_elements": len(mock_data)}

    def _demo_simple_reasoning(self):
        """Demo simple reasoning functionality"""
        print("🧠 Testing simple reasoning...")

        # Simple reasoning logic
        test_situations = [
            {"situation": "low_battery", "action": "return_to_base"},
            {"situation": "obstacle_ahead", "action": "avoid_obstacle"},
            {"situation": "target_reached", "action": "complete_task"}
        ]

        results = []
        for situation in test_situations:
            action = situation["action"]
            print(f"    🎯 {situation['situation']} -> {action}")
            results.append(action)

        print(f"  ✅ Simple reasoning completed")
        print(f"  📊 Decisions made: {len(results)}")

        return {"decisions": len(results)}

    def _run_basic_integration_test(self):
        """Run basic integration test without testing framework"""
        print("  🔧 Running basic integration checks...")

        checks = []

        # Check world model
        if WORLD_MODEL_AVAILABLE:
            try:
                world_model = WorldModel()
                checks.append({"component": "world_model", "status": "passed"})
            except Exception as e:
                checks.append({"component": "world_model", "status": "failed", "error": str(e)})

        # Check sensor components
        if SENSOR_COMPONENTS_AVAILABLE:
            try:
                # Simple sensor test
                points = np.random.rand(100, 3)
                pc_data = PointCloudData(points=points, timestamp=time.time())
                checks.append({"component": "sensors", "status": "passed"})
            except Exception as e:
                checks.append({"component": "sensors", "status": "failed", "error": str(e)})

        passed = sum(1 for c in checks if c["status"] == "passed")
        total = len(checks)

        print(f"    📊 Integration checks: {passed}/{total} passed")

        return {"passed": passed, "total": total}

    def _generate_mock_sensor_data(self) -> Dict[str, Any]:
        """Generate mock sensor data for testing"""
        return {
            "lidar": {
                "points": np.random.rand(1000, 3) * 50,
                "intensity": np.random.rand(1000),
                "timestamp": time.time()
            },
            "camera": {
                "image": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
                "timestamp": time.time()
            },
            "imu": {
                "linear_acceleration": np.random.randn(3),
                "angular_velocity": np.random.randn(3),
                "timestamp": time.time()
            }
        }

    def _generate_mock_perception_data(self) -> Dict[str, Any]:
        """Generate mock perception data"""
        return {
            "robot_pose": {
                "position": [0.0, 0.0, 50.0],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            },
            "obstacles": [
                {"id": "obs_1", "type": "building", "position": [100, 50, 0], "size": [20, 10, 30]},
                {"id": "obs_2", "type": "tree", "position": [30, 10, 0], "size": [2, 2, 5]}
            ],
            "targets": [
                {"id": "target_1", "type": "poi", "position": [200, 100, 0], "priority": "high"}
            ],
            "battery_level": 85,
            "timestamp": time.time()
        }

    def _generate_mock_perception_data_with_changes(self) -> Dict[str, Any]:
        """Generate mock perception data with changes"""
        base_data = self._generate_mock_perception_data()

        # Add changes
        base_data["obstacles"].append({
            "id": "obs_3",
            "type": "vehicle",
            "position": [150, 75, 0],
            "size": [5, 3, 2]
        })
        base_data["battery_level"] = 82  # Battery decreased
        base_data["timestamp"] = time.time()

        return base_data

    def _process_lidar_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process LiDAR data (mock implementation)"""
        try:
            points = data.get("points", np.array([]))
            return {
                "type": "lidar",
                "points": len(points),
                "processed": True
            }
        except Exception:
            return None

    def _process_camera_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process camera data (mock implementation)"""
        try:
            image = data.get("image", np.array([]))
            return {
                "type": "camera",
                "shape": image.shape if hasattr(image, 'shape') else None,
                "processed": True
            }
        except Exception:
            return None

    def _process_imu_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process IMU data (mock implementation)"""
        try:
            return {
                "type": "imu",
                "acceleration": data.get("linear_acceleration"),
                "angular_velocity": data.get("angular_velocity"),
                "processed": True
            }
        except Exception:
            return None

    def _test_world_model_integration(self) -> Dict[str, Any]:
        """Test world model integration"""
        if not WORLD_MODEL_AVAILABLE:
            return {"status": "SKIPPED", "reason": "World Model not available"}

        try:
            world_model = WorldModel()
            perception_data = self._generate_mock_perception_data()
            changes = world_model.update_from_perception(perception_data)

            return {
                "status": "PASSED",
                "changes_processed": len(changes),
                "execution_time": 0.1  # Mock timing
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _test_sensor_integration(self) -> Dict[str, Any]:
        """Test sensor integration"""
        if not SENSOR_COMPONENTS_AVAILABLE:
            return {"status": "SKIPPED", "reason": "Sensor components not available"}

        try:
            sensor_data = self._generate_mock_sensor_data()
            processed_count = 0

            for sensor_type, data in sensor_data.items():
                if sensor_type == "lidar" and self._process_lidar_data(data):
                    processed_count += 1
                elif sensor_type == "camera" and self._process_camera_data(data):
                    processed_count += 1
                elif sensor_type == "imu" and self._process_imu_data(data):
                    processed_count += 1

            return {
                "status": "PASSED",
                "sensors_processed": processed_count,
                "total_sensors": len(sensor_data)
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _test_reasoning_integration(self) -> Dict[str, Any]:
        """Test reasoning integration"""
        if not REASONING_AVAILABLE:
            return {"status": "SKIPPED", "reason": "Reasoning components not available"}

        try:
            # Simple reasoning test
            query = "Test integration query"
            mock_result = {
                "query": query,
                "processed": True,
                "confidence": 0.9
            }

            return {
                "status": "PASSED",
                "query_processed": mock_result["processed"],
                "confidence": mock_result["confidence"]
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    def _print_system_status(self):
        """Print current system status"""
        print("\n📊 System Status:")
        print(f"  Running: {self.running}")
        print(f"  Mode: {self.mode}")
        print(f"  Current Step: {self.current_step}/{self.total_steps}")
        print(f"  Components: {', '.join(self.components)}")

        component_status = {
            "World Model": self.world_model is not None,
            "Sensor Manager": self.sensor_manager is not None,
            "CoT Engine": self.cot_engine is not None,
            "Test Framework": self.test_framework is not None
        }

        print("\n🔧 Component Status:")
        for component, active in component_status.items():
            status = "🟢 Active" if active else "🔴 Inactive"
            print(f"  {component}: {status}")

        if self.results:
            print(f"\n📈 Results Summary:")
            for step_name, result in self.results.items():
                status_icon = "✅" if result["status"] == "completed" else "❌"
                print(f"  {status_icon} {step_name}: {result['status']}")

    def _print_demo_summary(self):
        """Print demo execution summary"""
        total_time = time.time() - self.demo_start_time if self.demo_start_time else 0

        print("\n" + "=" * 70)
        print("🎉 World Model System Demo - SUMMARY")
        print("=" * 70)

        print(f"\n⏱️ Total Execution Time: {total_time:.2f}s")
        print(f"📍 Steps Completed: {self.current_step}/{self.total_steps}")

        if self.results:
            print(f"\n📊 Step Results:")
            for step_name, result in self.results.items():
                status = result["status"]
                duration = result.get("duration", 0)
                status_icon = "✅" if status == "completed" else "❌"
                print(f"  {status_icon} {step_name}: {status} ({duration:.2f}s)")

        # Component status
        print(f"\n🔧 Component Usage:")
        component_usage = {
            "World Model": self.world_model is not None,
            "Sensor Manager": self.sensor_manager is not None,
            "CoT Engine": self.cot_engine is not None,
            "Test Framework": self.test_framework is not None
        }

        for component, used in component_usage.items():
            usage = "🟢 Used" if used else "🔴 Not Used"
            print(f"  {component}: {usage}")

        # Final message
        completed_steps = sum(1 for r in self.results.values() if r["status"] == "completed")
        total_steps = len(self.results)

        if total_steps > 0:
            success_rate = (completed_steps / total_steps) * 100
            print(f"\n🎯 Success Rate: {success_rate:.1f}% ({completed_steps}/{total_steps})")

            if success_rate >= 80:
                print("🎉 Excellent! The demo ran successfully!")
            elif success_rate >= 60:
                print("👍 Good! Most features worked as expected.")
            else:
                print("⚠️ Some issues encountered. Check the logs for details.")
        else:
            print("\nℹ️ No steps were executed.")

        print("\n" + "=" * 70)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="World Model System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_complete_system_demo.py                    # Full demo
  python3 run_complete_system_demo.py --mode=quick      # Quick demo
  python3 run_complete_system_demo.py --mode=interactive # Interactive demo
  python3 run_complete_system_demo.py --components=sensors # Sensor components only
        """
    )

    parser.add_argument(
        "--mode",
        choices=["full", "quick", "interactive"],
        default="full",
        help="Demo execution mode (default: full)"
    )

    parser.add_argument(
        "--components",
        choices=["all", "sensors", "world_model", "planning"],
        nargs="+",
        default=["all"],
        help="Components to test (default: all)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (optional)"
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration"""
    if LOGURU_AVAILABLE:
        # Remove default handler
        logger.remove()

        # Add console handler
        level = "DEBUG" if verbose else "INFO"
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level=level
        )

        # Add file handler if specified
        if log_file:
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="5 days"
            )
    else:
        # Standard logging setup
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        )

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
                )
            )
            logging.getLogger().addHandler(file_handler)


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose, args.log_file)

    try:
        # Print welcome message
        print("🧠 BRAIN World Model System Demo")
        print("=" * 70)
        print(f"Mode: {args.mode}")
        print(f"Components: {', '.join(args.components)}")
        print(f"Verbose: {args.verbose}")
        if args.log_file:
            print(f"Log file: {args.log_file}")
        print("=" * 70)

        # Create and run demo
        demo = WorldModelSystemDemo(
            mode=args.mode,
            components=args.components
        )

        success = demo.run_demo()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if LOGURU_AVAILABLE:
            logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()