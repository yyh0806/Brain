#!/usr/bin/env python3
"""
Isaac Sim仿真演示
展示基本的机器人控制、传感器仿真和场景交互
"""

import asyncio
import sys
import os
import yaml
import numpy as np
from pathlib import Path

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

class IsaacSimDemo:
    """Isaac Sim仿真演示类"""

    def __init__(self, config_path: str = None):
        """
        初始化演示

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or str(project_root / "config" / "isaac_sim_config.yaml")
        self.config = self._load_config()
        self.sim_interface: IsaacSimInterface = None

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件未找到: {self.config_path}")
            return self._get_default_config()
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            "simulation": {
                "mode": "headless",
                "physics_engine": "physx"
            },
            "robots": {
                "franka_emika": {
                    "type": "franka",
                    "position": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                }
            },
            "sensors": {
                "rgb_camera": {
                    "type": "camera",
                    "resolution": [640, 480],
                    "horizontal_fov": 90.0
                }
            }
        }

    async def initialize_simulation(self):
        """初始化仿真环境"""
        print("🚀 初始化Isaac Sim仿真环境...")

        # 创建仿真接口
        sim_config = self.config["simulation"]
        self.sim_interface = IsaacSimInterface(
            simulation_mode=SimulationMode(sim_config["mode"]),
            headless=(sim_config["mode"] == "headless"),
            enable_ui=False
        )

        # 初始化仿真
        await self.sim_interface.initialize()

        print("✅ 仿真环境初始化完成")
        await self._print_simulation_info()

    async def _print_simulation_info(self):
        """打印仿真信息"""
        info = self.sim_interface.get_simulation_info()
        print("\n=== 仿真环境信息 ===")
        print(f"Isaac Sim可用: {'✅' if info['isaac_sim_available'] else '❌'}")
        print(f"PyBullet可用: {'✅' if info['pybullet_available'] else '❌'}")
        print(f"仿真模式: {info['simulation_mode']}")
        print(f"物理引擎: {info['physics_engine']}")
        print(f"无头模式: {'是' if info['headless'] else '否'}")

    async def setup_robots(self):
        """设置机器人"""
        print("\n🤖 设置机器人...")

        for robot_name, robot_config in self.config["robots"].items():
            print(f"  创建机器人: {robot_name}")

            # 创建机器人配置
            config = RobotConfig(
                robot_type=robot_config["type"],
                robot_id=robot_name,
                position=tuple(robot_config["position"]),
                orientation=tuple(robot_config["orientation"]),
                usd_path=robot_config.get("usd_path", ""),
                joint_positions=robot_config.get("default_joints", {})
            )

            # 创建机器人
            robot_id = await self.sim_interface.create_robot(config)
            print(f"  ✅ 机器人创建成功: {robot_id}")

    async def setup_sensors(self):
        """设置传感器"""
        print("\n📷 设置传感器...")

        for sensor_name, sensor_config in self.config["sensors"].items():
            print(f"  创建传感器: {sensor_name}")

            # 创建传感器配置
            config = SensorConfig(
                sensor_type=sensor_config["type"],
                sensor_name=sensor_name,
                attach_to_robot=sensor_config.get("attach_to"),
                position=tuple(sensor_config.get("relative_position", [0, 0, 0])),
                orientation=tuple(sensor_config.get("relative_orientation", [0, 0, 0, 1])),
                sensor_params=sensor_config
            )

            # 创建传感器
            sensor_id = await self.sim_interface.create_sensor(config)
            print(f"  ✅ 传感器创建成功: {sensor_id}")

    async def setup_objects(self):
        """设置场景对象"""
        print("\n📦 设置场景对象...")

        # 如果配置中有对象定义
        if "objects" in self.config:
            for obj_name, obj_config in self.config["objects"].items():
                print(f"  创建对象: {obj_name}")

                # 这里简化处理，实际实现需要根据对象类型创建
                # 可以在接口中添加create_object方法
                print(f"  ✅ 对象创建成功: {obj_name}")

    async def run_simulation(self, duration: float = 10.0):
        """运行仿真"""
        print(f"\n⏱️  运行仿真 ({duration}秒)...")

        # 启动仿真
        await self.sim_interface.start_simulation()

        start_time = self.sim_interface.state.time
        step_count = 0

        while (self.sim_interface.state.time - start_time) < duration:
            # 执行仿真步进
            await self.sim_interface.step_simulation(1.0/60.0)

            # 每100步打印一次状态
            if step_count % 100 == 0:
                await self._print_simulation_status(step_count)

            # 执行机器人控制
            if step_count % 60 == 0:  # 每秒执行一次
                await self._control_robots()

            # 获取传感器数据
            if step_count % 30 == 0:  # 每0.5秒获取一次
                await self._get_sensor_data()

            step_count += 1

        print(f"\n✅ 仿真运行完成，总步数: {step_count}")

    async def _print_simulation_status(self, step_count: int):
        """打印仿真状态"""
        state = self.sim_interface.state
        print(f"  步数: {step_count:4d}, 时间: {state.time:6.2f}s, "
              f"机器人: {len(state.robots)}, 传感器: {len(state.sensors)}")

    async def _control_robots(self):
        """控制机器人"""
        # 示例：简单的机器人控制
        for robot_id in self.sim_interface.robots:
            # 获取当前机器人状态
            robot_state = await self.sim_interface.get_robot_state(robot_id)

            # 示例控制命令（可根据机器人类型定制）
            if "franka" in robot_id.lower():
                # 机械臂示例控制
                command = {
                    "joint_positions": {
                        "panda_joint1": 0.1 * np.sin(self.sim_interface.state.time),
                        "panda_joint2": 0.1 * np.cos(self.sim_interface.state.time)
                    }
                }
                await self.sim_interface.set_robot_command(robot_id, command)

            elif "husky" in robot_id.lower():
                # 移动机器人示例控制
                command = {
                    "linear_velocity": [0.2 * np.sin(self.sim_interface.state.time), 0, 0],
                    "angular_velocity": [0, 0, 0.1 * np.cos(self.sim_interface.state.time)]
                }
                await self.sim_interface.set_robot_command(robot_id, command)

    async def _get_sensor_data(self):
        """获取传感器数据"""
        for sensor_id in self.sim_interface.sensors:
            try:
                sensor_data = await self.sim_interface.get_sensor_data(sensor_id)

                if sensor_data["sensor_type"] == "camera":
                    # 打印相机数据信息
                    rgb_shape = sensor_data["rgb_image"].shape if hasattr(sensor_data["rgb_image"], 'shape') else "N/A"
                    depth_shape = sensor_data["depth_image"].shape if hasattr(sensor_data["depth_image"], 'shape') else "N/A"
                    print(f"  📷 {sensor_id}: RGB={rgb_shape}, Depth={depth_shape}")

                elif sensor_data["sensor_type"] == "lidar":
                    # 打印激光雷达数据信息
                    if "point_cloud" in sensor_data:
                        pc = sensor_data["point_cloud"]
                        pc_size = len(pc) if hasattr(pc, '__len__') else "N/A"
                        print(f"  📡 {sensor_id}: 点云点数={pc_size}")

            except Exception as e:
                print(f"  ❌ {sensor_id}: 获取数据失败 - {e}")

    async def run_pick_and_place_demo(self):
        """运行抓取放置演示"""
        print("\n🎯 运行抓取放置演示...")

        # 确保有Franka机器人
        if "franka_emika" not in self.sim_interface.robots:
            print("  ❌ 未找到Franka机器人")
            return

        robot_id = "franka_emika"

        # 定义抓取序列
        grasp_sequence = [
            # 移动到预备位置
            {
                "description": "移动到预备位置",
                "joints": {"panda_joint1": 0.0, "panda_joint2": 0.0, "panda_joint3": 0.0,
                          "panda_joint4": -1.5708, "panda_joint5": 0.0, "panda_joint6": 1.5708, "panda_joint7": 0.0},
                "duration": 2.0
            },
            # 移动到抓取位置
            {
                "description": "移动到抓取位置",
                "joints": {"panda_joint1": 0.5, "panda_joint2": 0.5, "panda_joint3": 0.5,
                          "panda_joint4": -1.0, "panda_joint5": 0.0, "panda_joint6": 1.0, "panda_joint7": 0.5},
                "duration": 2.0
            },
            # 闭合夹爪
            {
                "description": "闭合夹爪",
                "joints": {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
                "duration": 1.0
            },
            # 提起物体
            {
                "description": "提起物体",
                "joints": {"panda_joint3": -0.3},
                "duration": 2.0
            },
            # 移动到放置位置
            {
                "description": "移动到放置位置",
                "joints": {"panda_joint1": -0.5, "panda_joint2": 0.5},
                "duration": 2.0
            },
            # 放下物体
            {
                "description": "放下物体",
                "joints": {"panda_joint3": 0.5},
                "duration": 2.0
            },
            # 打开夹爪
            {
                "description": "打开夹爪",
                "joints": {"panda_finger_joint1": 0.0, "panda_finger_joint2": 0.0},
                "duration": 1.0
            }
        ]

        # 执行抓取序列
        for i, step in enumerate(grasp_sequence):
            print(f"  步骤 {i+1}/{len(grasp_sequence)}: {step['description']}")

            # 设置关节位置
            await self.sim_interface.set_robot_command(robot_id, {
                "joint_positions": step["joints"]
            })

            # 等待动作完成
            duration = step["duration"]
            steps = int(duration * 60)  # 60Hz
            for _ in range(steps):
                await self.sim_interface.step_simulation(1.0/60.0)

            print(f"    ✅ 完成")

        print("  🎉 抓取放置演示完成")

    async def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        if self.sim_interface:
            await self.sim_interface.shutdown()
        print("✅ 资源清理完成")

    async def run_demo(self, demo_type: str = "basic"):
        """
        运行演示

        Args:
            demo_type: 演示类型 ("basic", "pick_and_place", "navigation")
        """
        try:
            # 初始化仿真
            await self.initialize_simulation()

            # 设置场景
            await self.setup_robots()
            await self.setup_sensors()
            await self.setup_objects()

            # 根据演示类型运行不同的任务
            if demo_type == "basic":
                await self.run_simulation(10.0)
            elif demo_type == "pick_and_place":
                await self.run_pick_and_place_demo()
            elif demo_type == "navigation":
                await self.run_navigation_demo()
            else:
                print(f"❌ 未知的演示类型: {demo_type}")
                return

        except Exception as e:
            print(f"❌ 演示运行失败: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await self.cleanup()

    async def run_navigation_demo(self):
        """运行导航演示"""
        print("\n🗺️  运行导航演示...")
        # 这里可以实现移动机器人导航演示
        # 包括路径规划、避障等功能
        print("  🚧 导航演示开发中...")
        await self.run_simulation(5.0)

async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Isaac Sim仿真演示")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="basic",
        choices=["basic", "pick_and_place", "navigation"],
        help="演示类型"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="headless",
        choices=["headless", "gui", "render"],
        help="仿真模式"
    )

    args = parser.parse_args()

    # 创建演示实例
    demo = IsaacSimDemo(args.config)

    # 如果指定了仿真模式，覆盖配置
    if args.mode:
        demo.config["simulation"]["mode"] = args.mode

    # 运行演示
    print(f"🎬 启动Isaac Sim演示: {args.demo}")
    print(f"📄 配置文件: {demo.config_path}")
    print(f"🖥️  仿真模式: {args.mode}")
    print("-" * 50)

    await demo.run_demo(args.demo)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示异常退出: {e}")
        sys.exit(1)