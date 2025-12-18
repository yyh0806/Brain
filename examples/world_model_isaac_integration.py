#!/usr/bin/env python3
"""
World Model与Isaac Sim集成示例
展示如何将World Model系统与Isaac Sim仿真环境集成
实现感知、规划、执行闭环
"""

import asyncio
import sys
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from brain.platforms.isaac_sim_interface import (
    IsaacSimInterface,
    RobotConfig,
    SensorConfig,
    SimulationMode,
    create_isaac_sim_interface
)

# 导入World Model相关模块（假设已存在）
try:
    from brain.perception.sensor_processor import SensorProcessor
    from brain.planning.task_planner import TaskPlanner
    from brain.planning.motion_planner import MotionPlanner
    from brain.execution.robot_controller import RobotController
    from brain.cognitive.world_model import WorldModel
except ImportError as e:
    print(f"⚠️  World Model模块导入失败: {e}")
    print("使用模拟模块...")
    # 创建模拟模块用于演示
    class SensorProcessor:
        def __init__(self):
            pass
        async def process_sensor_data(self, data):
            return {"objects": [], "obstacles": []}

    class TaskPlanner:
        def __init__(self):
            pass
        async def plan_tasks(self, goal, world_state):
            return [{"type": "move", "target": [0, 0, 0]}]

    class MotionPlanner:
        def __init__(self):
            pass
        async def plan_motion(self, task, world_state):
            return {"trajectory": []}

    class RobotController:
        def __init__(self):
            pass
        async def execute_trajectory(self, trajectory):
            return {"success": True}

    class WorldModel:
        def __init__(self):
            self.state = {}
        async def update(self, perception_data):
            self.state.update(perception_data)
        async def get_state(self):
            return self.state

@dataclass
class Task:
    """任务定义"""
    task_id: str
    task_type: str
    goal: Dict[str, Any]
    priority: int = 1
    deadline: Optional[float] = None

@dataclass
class PerceptionResult:
    """感知结果"""
    timestamp: float
    objects: List[Dict[str, Any]]
    obstacles: List[Dict[str, Any]]
    robot_states: List[Dict[str, Any]]

@dataclass
class PlanningResult:
    """规划结果"""
    task_plan: List[Dict[str, Any]]
    motion_plan: Dict[str, Any]
    execution_time: float

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    execution_time: float
    feedback: Dict[str, Any]

class WorldModelIsaacIntegration:
    """
    World Model与Isaac Sim集成类

    实现完整的感知-规划-执行闭环：
    1. 感知：从仿真环境获取传感器数据
    2. 世界建模：构建和维护世界状态
    3. 任务规划：根据目标规划任务序列
    4. 运动规划：规划具体运动轨迹
    5. 执行控制：控制机器人执行动作
    6. 反馈监控：监控执行结果并调整
    """

    def __init__(self, config_path: str = None):
        """
        初始化集成系统

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or str(project_root / "config" / "isaac_sim_config.yaml")
        self.config = self._load_config()

        # 核心组件
        self.sim_interface: IsaacSimInterface = None
        self.world_model: WorldModel = None
        self.sensor_processor: SensorProcessor = None
        self.task_planner: TaskPlanner = None
        self.motion_planner: MotionPlanner = None
        self.robot_controller: RobotController = None

        # 系统状态
        self.is_initialized = False
        self.is_running = False
        self.current_tasks: List[Task] = []
        self.execution_history: List[Dict[str, Any]] = []

        # 性能监控
        self.performance_stats = {
            "perception_times": [],
            "planning_times": [],
            "execution_times": [],
            "total_cycles": 0,
            "success_rate": 0.0
        }

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "simulation": {
                "mode": "headless",
                "physics_engine": "physx",
                "physics_dt": 0.016666
            },
            "integration": {
                "control_frequency": 30.0,  # Hz
                "perception_frequency": 10.0,  # Hz
                "planning_frequency": 5.0,  # Hz
                "max_planning_time": 1.0  # seconds
            },
            "robots": {
                "franka_emika": {
                    "type": "franka",
                    "position": [0.0, 0.0, 0.0]
                }
            },
            "sensors": {
                "rgb_camera": {
                    "type": "camera",
                    "resolution": [640, 480],
                    "attach_to": "franka_emika"
                }
            }
        }

    async def initialize(self):
        """初始化集成系统"""
        print("🚀 初始化World Model与Isaac Sim集成系统...")

        try:
            # 1. 初始化仿真环境
            await self._initialize_simulation()

            # 2. 初始化World Model组件
            await self._initialize_world_model()

            # 3. 初始化感知-规划-执行模块
            await self._initialize_modules()

            # 4. 设置仿真场景
            await self._setup_simulation_scene()

            self.is_initialized = True
            print("✅ 集成系统初始化完成")

        except Exception as e:
            print(f"❌ 集成系统初始化失败: {e}")
            raise

    async def _initialize_simulation(self):
        """初始化仿真环境"""
        print("  🖥️  初始化仿真环境...")

        sim_config = self.config["simulation"]
        self.sim_interface = IsaacSimInterface(
            simulation_mode=SimulationMode(sim_config["mode"]),
            headless=(sim_config["mode"] == "headless")
        )

        await self.sim_interface.initialize()
        await self.sim_interface.start_simulation()

    async def _initialize_world_model(self):
        """初始化World Model"""
        print("  🧠 初始化World Model...")
        self.world_model = WorldModel()

    async def _initialize_modules(self):
        """初始化感知-规划-执行模块"""
        print("  🔧 初始化功能模块...")

        # 感知模块
        self.sensor_processor = SensorProcessor()

        # 规划模块
        self.task_planner = TaskPlanner()
        self.motion_planner = MotionPlanner()

        # 执行模块
        self.robot_controller = RobotController()

    async def _setup_simulation_scene(self):
        """设置仿真场景"""
        print("  🎬 设置仿真场景...")

        # 创建机器人
        for robot_name, robot_config in self.config["robots"].items():
            config = RobotConfig(
                robot_type=robot_config["type"],
                robot_id=robot_name,
                position=tuple(robot_config["position"]),
                orientation=tuple(robot_config.get("orientation", [0, 0, 0, 1]))
            )
            await self.sim_interface.create_robot(config)

        # 创建传感器
        for sensor_name, sensor_config in self.config["sensors"].items():
            config = SensorConfig(
                sensor_type=sensor_config["type"],
                sensor_name=sensor_name.split("_")[0],
                attach_to_robot=sensor_config.get("attach_to"),
                sensor_params=sensor_config
            )
            await self.sim_interface.create_sensor(config)

        print("  ✅ 场景设置完成")

    async def run_control_loop(self, duration: float = 30.0):
        """
        运行控制循环

        Args:
            duration: 运行时长（秒）
        """
        print(f"\n🔄 启动控制循环 ({duration}秒)...")

        self.is_running = True
        start_time = time.time()
        cycle_count = 0

        # 控制频率配置
        integration_config = self.config.get("integration", {})
        control_freq = integration_config.get("control_frequency", 30.0)
        perception_freq = integration_config.get("perception_frequency", 10.0)
        planning_freq = integration_config.get("planning_frequency", 5.0)

        control_dt = 1.0 / control_freq
        last_perception_time = 0
        last_planning_time = 0

        try:
            while (time.time() - start_time) < duration and self.is_running:
                cycle_start = time.time()

                # 仿真步进
                await self.sim_interface.step_simulation(control_dt)

                # 感知更新（较低频率）
                current_time = time.time() - start_time
                if current_time - last_perception_time >= 1.0 / perception_freq:
                    await self._perception_update()
                    last_perception_time = current_time

                # 规划更新（更低频率）
                if current_time - last_planning_time >= 1.0 / planning_freq:
                    await self._planning_update()
                    last_planning_time = current_time

                # 执行控制（每周期）
                await self._execution_update()

                cycle_count += 1
                cycle_time = time.time() - cycle_start

                # 维持控制频率
                if cycle_time < control_dt:
                    await asyncio.sleep(control_dt - cycle_time)

                # 打印状态（每100个周期）
                if cycle_count % 100 == 0:
                    await self._print_system_status(cycle_count, current_time)

        except Exception as e:
            print(f"❌ 控制循环异常: {e}")
            raise

        finally:
            self.is_running = False

        print(f"\n✅ 控制循环完成，总周期数: {cycle_count}")
        await self._print_performance_stats()

    async def _perception_update(self) -> PerceptionResult:
        """感知更新"""
        start_time = time.time()

        try:
            # 1. 从仿真环境获取传感器数据
            sensor_data = {}
            for sensor_id in self.sim_interface.sensors:
                try:
                    data = await self.sim_interface.get_sensor_data(sensor_id)
                    sensor_data[sensor_id] = data
                except Exception as e:
                    print(f"  ⚠️  传感器数据获取失败 {sensor_id}: {e}")

            # 2. 获取机器人状态
            robot_states = {}
            for robot_id in self.sim_interface.robots:
                try:
                    state = await self.sim_interface.get_robot_state(robot_id)
                    robot_states[robot_id] = state
                except Exception as e:
                    print(f"  ⚠️  机器人状态获取失败 {robot_id}: {e}")

            # 3. 处理传感器数据
            perception_result = await self.sensor_processor.process_sensor_data({
                "sensors": sensor_data,
                "robots": robot_states
            })

            # 4. 更新World Model
            await self.world_model.update(perception_result)

            # 5. 记录性能
            processing_time = time.time() - start_time
            self.performance_stats["perception_times"].append(processing_time)

            return PerceptionResult(
                timestamp=self.sim_interface.state.time,
                objects=perception_result.get("objects", []),
                obstacles=perception_result.get("obstacles", []),
                robot_states=list(robot_states.values())
            )

        except Exception as e:
            print(f"❌ 感知更新失败: {e}")
            return PerceptionResult(
                timestamp=self.sim_interface.state.time,
                objects=[],
                obstacles=[],
                robot_states=[]
            )

    async def _planning_update(self) -> Optional[PlanningResult]:
        """规划更新"""
        start_time = time.time()

        try:
            # 1. 获取当前世界状态
            world_state = await self.world_model.get_state()

            # 2. 检查是否有待执行任务
            if not self.current_tasks:
                # 如果没有任务，生成示例任务
                await self._generate_sample_tasks()

            # 3. 任务规划
            current_task = self.current_tasks[0] if self.current_tasks else None
            if current_task:
                task_plan = await self.task_planner.plan_tasks(
                    current_task.goal,
                    world_state
                )

                # 4. 运动规划
                if task_plan:
                    motion_plan = await self.motion_planner.plan_motion(
                        task_plan[0],  # 执行第一个任务
                        world_state
                    )

                    planning_time = time.time() - start_time
                    self.performance_stats["planning_times"].append(planning_time)

                    return PlanningResult(
                        task_plan=task_plan,
                        motion_plan=motion_plan,
                        execution_time=planning_time
                    )

            return None

        except Exception as e:
            print(f"❌ 规划更新失败: {e}")
            return None

    async def _execution_update(self) -> Optional[ExecutionResult]:
        """执行更新"""
        start_time = time.time()

        try:
            # 1. 获取最新的规划结果
            # 这里简化处理，实际应该有规划结果队列
            # planning_result = self.get_latest_planning_result()

            # 2. 执行控制命令
            # if planning_result and planning_result.motion_plan:
            #     execution_result = await self.robot_controller.execute_trajectory(
            #         planning_result.motion_plan["trajectory"]
            #     )
            # else:
            #     # 执行默认控制
            #     execution_result = await self._default_control()

            # 简化执行：直接控制仿真中的机器人
            execution_result = await self._default_control()

            # 3. 记录执行结果
            execution_time = time.time() - start_time
            self.performance_stats["execution_times"].append(execution_time)

            # 4. 更新执行历史
            self.execution_history.append({
                "timestamp": self.sim_interface.state.time,
                "success": execution_result["success"],
                "execution_time": execution_time,
                "feedback": execution_result.get("feedback", {})
            })

            return ExecutionResult(
                success=execution_result["success"],
                execution_time=execution_time,
                feedback=execution_result.get("feedback", {})
            )

        except Exception as e:
            print(f"❌ 执行更新失败: {e}")
            return ExecutionResult(
                success=False,
                execution_time=time.time() - start_time,
                feedback={"error": str(e)}
            )

    async def _default_control(self) -> Dict[str, Any]:
        """默认控制策略"""
        control_commands = {}

        # 为每个机器人生成默认控制命令
        for robot_id in self.sim_interface.robots:
            if "franka" in robot_id.lower():
                # Franka机械臂：简单的正弦运动
                t = self.sim_interface.state.time
                control_commands[robot_id] = {
                    "joint_positions": {
                        "panda_joint1": 0.2 * np.sin(0.5 * t),
                        "panda_joint2": 0.2 * np.cos(0.5 * t),
                        "panda_joint7": 0.1 * np.sin(1.0 * t)
                    }
                }

            elif "husky" in robot_id.lower():
                # Husky移动机器人：圆周运动
                t = self.sim_interface.state.time
                linear_vel = 0.3
                angular_vel = 0.2
                control_commands[robot_id] = {
                    "linear_velocity": [linear_vel * np.cos(angular_vel * t),
                                      linear_vel * np.sin(angular_vel * t), 0],
                    "angular_velocity": [0, 0, angular_vel]
                }

        # 发送控制命令
        for robot_id, command in control_commands.items():
            try:
                await self.sim_interface.set_robot_command(robot_id, command)
            except Exception as e:
                print(f"  ⚠️  控制命令发送失败 {robot_id}: {e}")

        return {"success": True, "commands_sent": len(control_commands)}

    async def _generate_sample_tasks(self):
        """生成示例任务"""
        current_time = self.sim_interface.state.time

        # 示例任务：机械臂抓取
        sample_tasks = [
            Task(
                task_id="grasp_cube",
                task_type="grasp",
                goal={
                    "target_object": "cube",
                    "target_position": [0.3, 0.0, 0.8],
                    "grasp_pose": [0.3, 0.0, 0.9]
                },
                priority=1
            ),
            Task(
                task_id="move_to_location",
                task_type="navigation",
                goal={
                    "target_location": [2.0, 2.0, 0.0],
                    "target_orientation": [0, 0, 0, 1]
                },
                priority=2
            )
        ]

        self.current_tasks = sample_tasks
        print(f"  📋 生成了 {len(sample_tasks)} 个示例任务")

    async def _print_system_status(self, cycle_count: int, current_time: float):
        """打印系统状态"""
        print(f"  🔄 周期: {cycle_count:5d}, 时间: {current_time:6.2f}s, "
              f"任务: {len(self.current_tasks)}, 成功率: {self.performance_stats['success_rate']:.1%}")

    async def _print_performance_stats(self):
        """打印性能统计"""
        print("\n📊 系统性能统计:")
        print(f"  总周期数: {self.performance_stats['total_cycles']}")

        if self.performance_stats["perception_times"]:
            avg_perception = np.mean(self.performance_stats["perception_times"])
            print(f"  感知平均耗时: {avg_perception*1000:.2f}ms")

        if self.performance_stats["planning_times"]:
            avg_planning = np.mean(self.performance_stats["planning_times"])
            print(f"  规划平均耗时: {avg_planning*1000:.2f}ms")

        if self.performance_stats["execution_times"]:
            avg_execution = np.mean(self.performance_stats["execution_times"])
            print(f"  执行平均耗时: {avg_execution*1000:.2f}ms")

        if self.execution_history:
            success_count = sum(1 for h in self.execution_history if h["success"])
            self.performance_stats["success_rate"] = success_count / len(self.execution_history)
            print(f"  任务成功率: {self.performance_stats['success_rate']:.1%}")

    async def add_task(self, task: Task):
        """添加新任务"""
        self.current_tasks.append(task)
        print(f"➕ 添加新任务: {task.task_id} ({task.task_type})")

    async def remove_task(self, task_id: str):
        """移除任务"""
        self.current_tasks = [t for t in self.current_tasks if t.task_id != task_id]
        print(f"➖ 移除任务: {task_id}")

    async def shutdown(self):
        """关闭系统"""
        print("\n🛑 关闭集成系统...")

        self.is_running = False

        if self.sim_interface:
            await self.sim_interface.shutdown()

        # 保存执行历史
        await self._save_execution_history()

        print("✅ 系统关闭完成")

    async def _save_execution_history(self):
        """保存执行历史"""
        try:
            history_file = project_root / "data" / "execution_history.json"
            history_file.parent.mkdir(exist_ok=True)

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "performance_stats": self.performance_stats,
                    "execution_history": self.execution_history[-100:]  # 保存最近100条记录
                }, f, indent=2, ensure_ascii=False)

            print(f"💾 执行历史已保存到: {history_file}")

        except Exception as e:
            print(f"⚠️  执行历史保存失败: {e}")

async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="World Model与Isaac Sim集成示例")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="运行时长（秒）"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="headless",
        choices=["headless", "gui"],
        help="仿真模式"
    )

    args = parser.parse_args()

    # 创建集成系统
    integration = WorldModelIsaacIntegration(args.config)

    # 如果指定了仿真模式，覆盖配置
    if args.mode:
        integration.config["simulation"]["mode"] = args.mode

    try:
        print("🎬 启动World Model与Isaac Sim集成示例")
        print(f"📄 配置文件: {integration.config_path}")
        print(f"🖥️  仿真模式: {args.mode}")
        print(f"⏱️  运行时长: {args.duration}秒")
        print("-" * 60)

        # 初始化系统
        await integration.initialize()

        # 运行控制循环
        await integration.run_control_loop(args.duration)

    except KeyboardInterrupt:
        print("\n\n⏹️  系统被用户中断")
    except Exception as e:
        print(f"\n❌ 系统异常退出: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())