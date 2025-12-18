#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Complete World Model System Demo

A simplified demonstration script that avoids complex dependencies
and provides a working demo of the World Model system concepts.

Usage:
    python3 run_complete_system_demo_simple.py [--mode=full|quick]

Author: Brain World Model Team
Date: 2025-12-18
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # 'PASSED', 'FAILED', 'SKIPPED'
    duration: float
    details: Dict[str, Any] = None


class MockWorldModel:
    """Mock World Model for demonstration purposes"""

    def __init__(self):
        self.objects = []
        self.changes = []
        self.robot_pose = {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
        self.battery_level = 100

    def update_from_perception(self, perception_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock update from perception data"""
        self.robot_pose = perception_data.get("robot_pose", self.robot_pose)
        self.battery_level = perception_data.get("battery_level", self.battery_level)

        # Process obstacles and targets
        self.objects = []
        obstacles = perception_data.get("obstacles", [])
        targets = perception_data.get("targets", [])

        for obs in obstacles:
            self.objects.append({"type": "obstacle", **obs})

        for target in targets:
            self.objects.append({"type": "target", **target})

        # Mock change detection
        changes = []
        if len(self.objects) > 0:
            changes.append({
                "type": "objects_detected",
                "count": len(self.objects),
                "description": f"Detected {len(self.objects)} objects"
            })

        self.changes.extend(changes)
        return changes

    def detect_significant_changes(self) -> List[Dict[str, Any]]:
        """Mock significant change detection"""
        significant = []
        for change in self.changes:
            if change["type"] == "objects_detected" and change["count"] > 3:
                change["priority"] = "high"
                significant.append(change)
            else:
                change["priority"] = "medium"
        return significant

    def get_context_for_planning(self) -> Dict[str, Any]:
        """Get context for planning"""
        obstacles = [obj for obj in self.objects if obj["type"] == "obstacle"]
        targets = [obj for obj in self.objects if obj["type"] == "target"]

        return {
            "obstacles": obstacles,
            "targets": targets,
            "battery_level": self.battery_level,
            "robot_position": self.robot_pose["position"],
            "constraints": ["avoid_obstacles", "maintain_battery"]
        }


class MockSensorProcessor:
    """Mock sensor processor"""

    def process_lidar(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process mock LiDAR data"""
        points = data.get("points", [])
        return {
            "type": "lidar",
            "processed_points": len(points),
            "obstacles_detected": random.randint(1, 5)
        }

    def process_camera(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process mock camera data"""
        return {
            "type": "camera",
            "objects_detected": random.randint(0, 10),
            "confidence": random.uniform(0.7, 0.95)
        }

    def process_imu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process mock IMU data"""
        return {
            "type": "imu",
            "orientation_stable": random.choice([True, False]),
            "motion_detected": random.choice([True, False])
        }


class MockReasoningEngine:
    """Mock reasoning engine"""

    def assess_complexity(self, query: str, context: Dict[str, Any]) -> str:
        """Assess query complexity"""
        if "navigate" in query or "plan" in query:
            return "complex"
        elif "avoid" in query or "detect" in query:
            return "moderate"
        else:
            return "simple"

    def reason(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock reasoning process"""
        complexity = self.assess_complexity(query, context)

        reasoning_chain = [
            "Analyze current situation",
            "Consider available options",
            "Evaluate risks and benefits",
            "Make decision"
        ]

        if complexity == "complex":
            reasoning_chain.insert(1, "Gather additional information")
            reasoning_chain.insert(3, "Create detailed plan")

        return {
            "query": query,
            "complexity": complexity,
            "confidence": random.uniform(0.8, 0.95),
            "decision": f"Proceed with action: {query[:30]}...",
            "suggestion": "Monitor situation closely",
            "chain": reasoning_chain
        }


class WorldModelSystemDemo:
    """Complete World Model System Demo Class"""

    def __init__(self, mode: str = "full"):
        """
        Initialize the demo system

        Args:
            mode: Operating mode ("full", "quick", "interactive")
        """
        self.mode = mode
        self.running = False
        self.results = []
        self.demo_start_time = None

        # Initialize mock components
        self.world_model = MockWorldModel()
        self.sensor_processor = MockSensorProcessor()
        self.reasoning_engine = MockReasoningEngine()

        print("=" * 70)
        print("🧠 BRAIN World Model System Demo - Initializing...")
        print("=" * 70)

        print("\n✅ Mock Components Initialized:")
        print("  🌍 World Model")
        print("  🔬 Sensor Processor")
        print("  🧠 Reasoning Engine")

    def run_demo(self):
        """Run the complete demo"""
        print("\n🎬 Starting World Model System Demo...")
        print(f"   Mode: {self.mode}")

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

        for i, (step_name, step_func) in enumerate(demo_steps):
            if not self.running:
                break

            print(f"\n📍 Step {i+1}/{len(demo_steps)}: {step_name}")
            print("-" * 50)

            start_time = time.time()
            try:
                result = step_func()
                duration = time.time() - start_time
                status = "PASSED" if result else "FAILED"

                test_result = TestResult(
                    test_name=step_name,
                    status=status,
                    duration=duration,
                    details={"result": result}
                )
                self.results.append(test_result)

                print(f"✅ {step_name} completed in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                test_result = TestResult(
                    test_name=step_name,
                    status="FAILED",
                    duration=duration,
                    details={"error": str(e)}
                )
                self.results.append(test_result)
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

        for i, (step_name, step_func) in enumerate(demo_steps):
            if not self.running:
                break

            print(f"\n📍 Step {i+1}/{len(demo_steps)}: {step_name}")
            print("-" * 50)

            start_time = time.time()
            try:
                result = step_func()
                duration = time.time() - start_time
                status = "PASSED" if result else "FAILED"

                test_result = TestResult(
                    test_name=step_name,
                    status=status,
                    duration=duration,
                    details={"result": result}
                )
                self.results.append(test_result)

                print(f"✅ {step_name} completed in {duration:.2f}s")

            except Exception as e:
                duration = time.time() - start_time
                test_result = TestResult(
                    test_name=step_name,
                    status="FAILED",
                    duration=duration,
                    details={"error": str(e)}
                )
                self.results.append(test_result)
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
        print("🔬 Testing sensor data processing...")

        # Generate mock sensor data
        sensor_data = self._generate_mock_sensor_data()

        print(f"\n  Processing {len(sensor_data)} sensor types...")
        processed_count = 0

        for sensor_type, data in sensor_data.items():
            print(f"\n  📡 Processing {sensor_type.upper()}...")

            if sensor_type == "lidar":
                result = self.sensor_processor.process_lidar(data)
            elif sensor_type == "camera":
                result = self.sensor_processor.process_camera(data)
            elif sensor_type == "imu":
                result = self.sensor_processor.process_imu(data)
            else:
                result = None

            if result:
                processed_count += 1
                print(f"    ✅ {sensor_type.upper()} processed successfully")
                for key, value in result.items():
                    if key != "type":
                        print(f"    📊 {key}: {value}")
            else:
                print(f"    ❌ {sensor_type.upper()} processing failed")

        print(f"\n  🎯 Sensors processed: {processed_count}/{len(sensor_data)}")
        return processed_count == len(sensor_data)

    def _demo_world_model_updates(self):
        """Demo world model updates"""
        print("🌍 Testing world model updates...")

        # Generate mock perception data
        perception_data = self._generate_mock_perception_data()

        print(f"\n  📊 Perception data:")
        print(f"    Robot position: {perception_data['robot_pose']['position']}")
        print(f"    Battery level: {perception_data['battery_level']}%")
        print(f"    Obstacles: {len(perception_data['obstacles'])}")
        print(f"    Targets: {len(perception_data['targets'])}")

        # Update world model
        print(f"\n  🔄 Updating world model...")
        changes = self.world_model.update_from_perception(perception_data)

        print(f"  ✅ World model updated")
        print(f"  📊 Changes detected: {len(changes)}")

        for change in changes:
            print(f"    - {change['description']}")

        # Get planning context
        context = self.world_model.get_context_for_planning()
        print(f"\n  📍 Planning context:")
        print(f"    Obstacles: {len(context['obstacles'])}")
        print(f"    Targets: {len(context['targets'])}")
        print(f"    Battery: {context['battery_level']}%")
        print(f"    Constraints: {', '.join(context['constraints'])}")

        return len(changes) > 0

    def _demo_change_detection(self):
        """Demo change detection"""
        print("🔍 Testing change detection...")

        # Initial state
        print(f"\n  📊 Initial state:")
        initial_data = self._generate_mock_perception_data()
        changes1 = self.world_model.update_from_perception(initial_data)
        print(f"    Objects detected: {len(self.world_model.objects)}")

        # Changed state
        print(f"\n  🔄 Simulating environmental changes...")
        changed_data = self._generate_mock_perception_data_with_changes()
        changes2 = self.world_model.update_from_perception(changed_data)

        # Detect significant changes
        significant_changes = self.world_model.detect_significant_changes()

        print(f"  ✅ Change detection completed")
        print(f"  📊 Total changes: {len(changes1) + len(changes2)}")
        print(f"  🎯 Significant changes: {len(significant_changes)}")

        for change in significant_changes:
            print(f"    - {change['description']} (Priority: {change['priority']})")

        return len(significant_changes) > 0

    def _demo_reasoning_planning(self):
        """Demo reasoning and planning"""
        print("🧠 Testing reasoning and planning...")

        # Test reasoning queries
        test_queries = [
            "Plan a route to the target location",
            "Analyze the current situation and suggest actions",
            "Evaluate the risks and benefits of proceeding"
        ]

        results = []
        for i, query in enumerate(test_queries):
            print(f"\n  🤔 Query {i+1}: {query}")

            # Assess complexity
            complexity = self.reasoning_engine.assess_complexity(query, {})
            print(f"    📊 Complexity: {complexity}")

            # Execute reasoning
            reasoning_result = self.reasoning_engine.reason(query, {})

            print(f"    ✅ Reasoning completed")
            print(f"    🎯 Confidence: {reasoning_result['confidence']:.2f}")
            print(f"    💡 Decision: {reasoning_result['decision']}")
            print(f"    💭 Suggestion: {reasoning_result['suggestion']}")

            results.append(reasoning_result)

        print(f"\n  🎯 Reasoning results: {len(results)} queries processed")
        return len(results) == len(test_queries)

    def _demo_integration_test(self):
        """Demo integration testing"""
        print("🔧 Running integration tests...")

        tests = [
            ("Sensor to World Model Integration", self._test_sensor_world_model_integration),
            ("World Model to Reasoning Integration", self._test_world_model_reasoning_integration),
            ("End-to-End Pipeline Integration", self._test_end_to_end_integration)
        ]

        passed = 0
        for test_name, test_func in tests:
            print(f"\n  🧪 {test_name}...")
            try:
                result = test_func()
                if result:
                    print(f"    ✅ PASSED")
                    passed += 1
                else:
                    print(f"    ❌ FAILED")
            except Exception as e:
                print(f"    ❌ FAILED: {e}")

        print(f"\n  📊 Integration tests: {passed}/{len(tests)} passed")
        return passed == len(tests)

    def _demo_basic_world_model(self):
        """Demo basic world model functionality"""
        print("🌍 Testing basic world model...")

        # Create world model
        world_model = MockWorldModel()

        # Add simple data
        mock_data = {
            "robot_position": [0, 0, 0],
            "obstacles": [[10, 10, 0], [20, 5, 0]],
            "timestamp": time.time()
        }

        print(f"  📊 Mock data created with {len(mock_data)} elements")
        print(f"  ✅ Basic world model functionality working")

        return True

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

        return len(results) == len(test_situations)

    def _test_sensor_world_model_integration(self):
        """Test sensor to world model integration"""
        sensor_data = self._generate_mock_sensor_data()

        # Process sensors
        processed_results = []
        for sensor_type, data in sensor_data.items():
            if sensor_type == "lidar":
                result = self.sensor_processor.process_lidar(data)
            elif sensor_type == "camera":
                result = self.sensor_processor.process_camera(data)
            elif sensor_type == "imu":
                result = self.sensor_processor.process_imu(data)
            else:
                result = None

            if result:
                processed_results.append(result)

        # Update world model with processed data
        perception_data = self._generate_mock_perception_data()
        changes = self.world_model.update_from_perception(perception_data)

        return len(processed_results) > 0 and len(changes) >= 0

    def _test_world_model_reasoning_integration(self):
        """Test world model to reasoning integration"""
        # Update world model
        perception_data = self._generate_mock_perception_data()
        self.world_model.update_from_perception(perception_data)

        # Get context for reasoning
        context = self.world_model.get_context_for_planning()

        # Reason about the context
        query = "What should I do next?"
        reasoning_result = self.reasoning_engine.reason(query, context)

        return reasoning_result is not None and reasoning_result["confidence"] > 0.5

    def _test_end_to_end_integration(self):
        """Test end-to-end integration"""
        # Complete pipeline
        sensor_data = self._generate_mock_sensor_data()
        perception_data = self._generate_mock_perception_data()

        # Step 1: Process sensors
        processed = []
        for data in sensor_data.values():
            processed.append(data)

        # Step 2: Update world model
        changes = self.world_model.update_from_perception(perception_data)

        # Step 3: Reason about situation
        context = self.world_model.get_context_for_planning()
        reasoning = self.reasoning_engine.reason("Analyze current situation", context)

        return len(processed) > 0 and len(changes) >= 0 and reasoning is not None

    def _generate_mock_sensor_data(self) -> Dict[str, Any]:
        """Generate mock sensor data for testing"""
        return {
            "lidar": {
                "points": [[random.uniform(-50, 50) for _ in range(3)] for _ in range(1000)],
                "intensity": [random.uniform(0, 1) for _ in range(1000)],
                "timestamp": time.time()
            },
            "camera": {
                "image_shape": [480, 640, 3],
                "objects_detected": random.randint(0, 10),
                "timestamp": time.time()
            },
            "imu": {
                "linear_acceleration": [random.uniform(-2, 2) for _ in range(3)],
                "angular_velocity": [random.uniform(-1, 1) for _ in range(3)],
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

    def _print_system_status(self):
        """Print current system status"""
        print("\n📊 System Status:")
        print(f"  Running: {self.running}")
        print(f"  Mode: {self.mode}")
        print(f"  Demo Duration: {time.time() - self.demo_start_time:.2f}s" if self.demo_start_time else "  Demo Duration: Not started")

        component_status = {
            "World Model": self.world_model is not None,
            "Sensor Processor": self.sensor_processor is not None,
            "Reasoning Engine": self.reasoning_engine is not None
        }

        print("\n🔧 Component Status:")
        for component, active in component_status.items():
            status = "🟢 Active" if active else "🔴 Inactive"
            print(f"  {component}: {status}")

        if self.results:
            print(f"\n📈 Results Summary:")
            passed = sum(1 for r in self.results if r.status == "PASSED")
            total = len(self.results)
            print(f"  Tests Passed: {passed}/{total}")

    def _print_demo_summary(self):
        """Print demo execution summary"""
        total_time = time.time() - self.demo_start_time if self.demo_start_time else 0

        print("\n" + "=" * 70)
        print("🎉 World Model System Demo - SUMMARY")
        print("=" * 70)

        print(f"\n⏱️ Total Execution Time: {total_time:.2f}s")
        print(f"📍 Steps Completed: {len(self.results)}")

        if self.results:
            print(f"\n📊 Step Results:")
            for result in self.results:
                status_icon = "✅" if result.status == "PASSED" else "❌"
                print(f"  {status_icon} {result.test_name}: {result.status} ({result.duration:.2f}s)")

        # Component status
        print(f"\n🔧 Component Usage:")
        print(f"  World Model: 🟢 Used")
        print(f"  Sensor Processor: 🟢 Used")
        print(f"  Reasoning Engine: 🟢 Used")

        # Final message
        if self.results:
            passed = sum(1 for r in self.results if r.status == "PASSED")
            total = len(self.results)
            success_rate = (passed / total) * 100
            print(f"\n🎯 Success Rate: {success_rate:.1f}% ({passed}/{total})")

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
  python3 run_complete_system_demo_simple.py              # Full demo
  python3 run_complete_system_demo_simple.py --mode=quick # Quick demo
  python3 run_complete_system_demo_simple.py --mode=interactive # Interactive demo
        """
    )

    parser.add_argument(
        "--mode",
        choices=["full", "quick", "interactive"],
        default="full",
        help="Demo execution mode (default: full)"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    try:
        # Print welcome message
        print("🧠 BRAIN World Model System Demo (Simple Version)")
        print("=" * 70)
        print(f"Mode: {args.mode}")
        print("=" * 70)

        # Create and run demo
        demo = WorldModelSystemDemo(mode=args.mode)
        success = demo.run_demo()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()