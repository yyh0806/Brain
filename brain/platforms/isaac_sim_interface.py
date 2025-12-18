#!/usr/bin/env python3
"""
Isaac Sim仿真环境接口
提供与NVIDIA Isaac Sim的高保真物理仿真集成
支持机器人仿真、传感器模拟和场景管理
"""

import asyncio
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationMode(Enum):
    """仿真模式枚举"""
    HEADLESS = "headless"  # 无头模式，仅计算
    GUI = "gui"  # GUI模式，可视化
    RENDER = "render"  # 渲染模式，高质量图像

class PhysicsEngine(Enum):
    """物理引擎选择"""
    PHYSX = "physx"  # NVIDIA PhysX（默认）
    BULLET = "bullet"  # Bullet Physics（备选）

@dataclass
class RobotConfig:
    """机器人配置"""
    robot_type: str  # 机器人类型
    usd_path: str = ""  # USD文件路径
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    joint_positions: Dict[str, float] = field(default_factory=dict)
    robot_id: Optional[str] = None

@dataclass
class SensorConfig:
    """传感器配置"""
    sensor_type: str  # 传感器类型：camera, lidar, imu, etc.
    sensor_name: str
    attach_to_robot: Optional[str] = None  # 附着的机器人ID
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    sensor_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationState:
    """仿真状态"""
    time: float = 0.0
    step_count: int = 0
    is_running: bool = False
    is_paused: bool = False
    robots: Dict[str, Any] = field(default_factory=dict)
    sensors: Dict[str, Any] = field(default_factory=dict)
    objects: Dict[str, Any] = field(default_factory=dict)

class IsaacSimInterface:
    """
    Isaac Sim仿真接口类

    提供与Isaac Sim的集成接口，包括：
    - 场景创建和管理
    - 机器人导入和控制
    - 传感器仿真和数据获取
    - 物理仿真控制
    - 渲染和图像输出
    """

    def __init__(
        self,
        simulation_mode: SimulationMode = SimulationMode.HEADLESS,
        physics_engine: PhysicsEngine = PhysicsEngine.PHYSX,
        headless: bool = True,
        enable_ui: bool = False,
        enable_ros: bool = False
    ):
        """
        初始化Isaac Sim接口

        Args:
            simulation_mode: 仿真模式
            physics_engine: 物理引擎选择
            headless: 是否无头模式
            enable_ui: 是否启用UI
            enable_ros: 是否启用ROS集成
        """
        self.simulation_mode = simulation_mode
        self.physics_engine = physics_engine
        self.headless = headless
        self.enable_ui = enable_ui
        self.enable_ros = enable_ros

        # 核心组件
        self.simulation_app = None
        self.world = None
        self.scene = None

        # 仿真状态
        self.state = SimulationState()
        self.robots: Dict[str, RobotConfig] = {}
        self.sensors: Dict[str, SensorConfig] = {}

        # 初始化标志
        self._initialized = False
        self._isaac_sim_available = False
        self._pybullet_available = False

        # 尝试导入Isaac Sim
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖可用性"""
        try:
            import isaacsim
            from omni.isaac.core import SimulationContext
            from omni.isaac.core.world import World
            from omni.isaac.core.objects import GroundPlane

            self._isaac_sim_available = True
            self._simulation_context_class = SimulationContext
            self._world_class = World
            self._ground_plane_class = GroundPlane

            logger.info("✓ Isaac Sim模块导入成功")

        except ImportError as e:
            logger.warning(f"Isaac Sim不可用: {e}")
            logger.info("尝试使用PyBullet作为备选仿真环境...")

            try:
                import pybullet as p
                self._pybullet_available = True
                self._pybullet = p
                logger.info("✓ PyBullet模块导入成功")

            except ImportError as e2:
                logger.error(f"PyBullet也不可用: {e2}")
                raise ImportError("无可用的仿真环境（Isaac Sim或PyBullet）")

    async def initialize(self):
        """初始化仿真环境"""
        if self._initialized:
            logger.warning("仿真环境已初始化")
            return

        logger.info(f"初始化仿真环境 (模式: {self.simulation_mode.value})")

        try:
            if self._isaac_sim_available:
                await self._initialize_isaac_sim()
            elif self._pybullet_available:
                await self._initialize_pybullet()
            else:
                raise RuntimeError("无可用的仿真环境")

            self._initialized = True
            self.state.is_running = False
            logger.info("✓ 仿真环境初始化完成")

        except Exception as e:
            logger.error(f"仿真环境初始化失败: {e}")
            raise

    async def _initialize_isaac_sim(self):
        """初始化Isaac Sim"""
        from omni.isaac.core import SimulationContext

        # 创建仿真应用
        simulation_context_kwargs = {
            "physics_dt": 1.0 / 60.0,  # 60Hz
            "rendering_dt": 1.0 / 60.0,
            "stage_units_in_meters": 1.0,
        }

        if self.headless:
            simulation_context_kwargs["headless"] = True

        self.simulation_app = self._simulation_context_class(
            **simulation_context_kwargs
        )

        # 创建世界
        self.world = self._world_class(
            physics_dt=1.0/60.0,
            rendering_dt=1.0/60.0,
            stage_units_in_meters=1.0
        )

        # 创建场景
        await self.world.initialize_simulation_context()

        # 添加地面
        self.scene = self._ground_plane_class(
            prim_path="/World/GroundPlane",
            size=100.0,
            color=np.array([0.5, 0.5, 0.5])
        )

        logger.info("✓ Isaac Sim环境初始化完成")

    async def _initialize_pybullet(self):
        """初始化PyBullet作为备选"""
        # 连接PyBullet
        if self.headless:
            self._pybullet.connect(self._pybullet.DIRECT)
        else:
            self._pybullet.connect(self._pybullet.GUI)

        # 设置重力
        self._pybullet.setGravity(0, 0, -9.81)

        # 设置仿真参数
        self._pybullet.setPhysicsEngineParameter(
            numSolverIterations=50,
            contactBreakingThreshold=0.00001
        )

        # 创建地面
        ground_collision = self._pybullet.createCollisionShape(
            self._pybullet.GEOM_BOX,
            halfExtents=[50, 50, 0.1]
        )
        ground_id = self._pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_collision,
            basePosition=[0, 0, -0.1]
        )

        self._ground_id = ground_id
        logger.info("✓ PyBullet环境初始化完成")

    async def create_robot(self, config: RobotConfig) -> str:
        """
        创建机器人

        Args:
            config: 机器人配置

        Returns:
            机器人ID
        """
        if not self._initialized:
            raise RuntimeError("仿真环境未初始化")

        robot_id = config.robot_id or f"robot_{len(self.robots)}"

        try:
            if self._isaac_sim_available:
                robot_id = await self._create_robot_isaac(config, robot_id)
            elif self._pybullet_available:
                robot_id = await self._create_robot_pybullet(config, robot_id)
            else:
                raise RuntimeError("无可用的仿真环境")

            self.robots[robot_id] = config
            config.robot_id = robot_id

            logger.info(f"✓ 机器人创建成功: {robot_id}")
            return robot_id

        except Exception as e:
            logger.error(f"机器人创建失败: {e}")
            raise

    async def _create_robot_isaac(self, config: RobotConfig, robot_id: str) -> str:
        """使用Isaac Sim创建机器人"""
        from omni.isaac.core.robots import Robot

        # 根据机器人类型选择USD路径
        if not config.usd_path:
            # 使用默认的机器人模型
            if config.robot_type == "franka":
                config.usd_path = "/Isaac/Robots/Franka/franka.usd"
            elif config.robot_type == "ur10":
                config.usd_path = "/Isaac/Robots/UniversalRobots/UR10/ur10.usd"
            elif config.robot_type == "quadrotor":
                config.usd_path = "/Isaac/Robots/Quadrotor/quadrotor.usd"
            else:
                # 使用通用机器人模型
                config.usd_path = "/Isaac/Robots/MobileRobots/ClearpathHusky/husky.usd"

        # 创建机器人
        robot = Robot(
            prim_path=f"/World/{robot_id}",
            name=robot_id,
            usd_path=config.usd_path,
            position=config.position,
            orientation=config.orientation
        )

        # 添加到世界
        self.world.add_robot(robot)

        # 设置关节位置
        if config.joint_positions:
            await robot.set_joint_positions(config.joint_positions)

        # 存储机器人对象
        self.state.robots[robot_id] = robot

        return robot_id

    async def _create_robot_pybullet(self, config: RobotConfig, robot_id: str) -> str:
        """使用PyBullet创建机器人"""
        # 简化实现：创建一个基本的盒子机器人
        collision_shape = self._pybullet.createCollisionShape(
            self._pybullet.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3]
        )

        visual_shape = self._pybullet.createVisualShape(
            self._pybullet.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3],
            rgbaColor=[0.5, 0.5, 0.8, 1]
        )

        robot_body = self._pybullet.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=config.position
        )

        # 存储机器人对象
        self.state.robots[robot_id] = robot_body

        return robot_id

    async def create_sensor(self, config: SensorConfig) -> str:
        """
        创建传感器

        Args:
            config: 传感器配置

        Returns:
            传感器ID
        """
        if not self._initialized:
            raise RuntimeError("仿真环境未初始化")

        sensor_id = f"{config.sensor_type}_{config.sensor_name}"

        try:
            if self._isaac_sim_available:
                await self._create_sensor_isaac(config, sensor_id)
            elif self._pybullet_available:
                await self._create_sensor_pybullet(config, sensor_id)
            else:
                raise RuntimeError("无可用的仿真环境")

            self.sensors[sensor_id] = config

            logger.info(f"✓ 传感器创建成功: {sensor_id}")
            return sensor_id

        except Exception as e:
            logger.error(f"传感器创建失败: {e}")
            raise

    async def _create_sensor_isaac(self, config: SensorConfig, sensor_id: str):
        """使用Isaac Sim创建传感器"""
        from omni.isaac.sensor import Camera, Lidar

        prim_path = f"/World/{sensor_id}"

        if config.sensor_type == "camera":
            # 创建相机
            camera = Camera(
                prim_path=prim_path,
                position=config.position,
                orientation=config.orientation,
                resolution=tuple(config.sensor_params.get("resolution", [640, 480])),
                horizontal_fov=config.sensor_params.get("fov", 90.0)
            )

            if config.attach_to_robot and config.attach_to_robot in self.state.robots:
                # 将相机附着到机器人
                robot = self.state.robots[config.attach_to_robot]
                robot.attach(camera)

            self.state.sensors[sensor_id] = camera

        elif config.sensor_type == "lidar":
            # 创建激光雷达
            lidar = Lidar(
                prim_path=prim_path,
                position=config.position,
                orientation=config.orientation,
                horizontal_fov=config.sensor_params.get("horizontal_fov", 360.0),
                vertical_fov=config.sensor_params.get("vertical_fov", 30.0),
                horizontal_resolution=config.sensor_params.get("horizontal_resolution", 0.25),
                max_range=config.sensor_params.get("max_range", 10.0),
                min_range=config.sensor_params.get("min_range", 0.1)
            )

            if config.attach_to_robot and config.attach_to_robot in self.state.robots:
                robot = self.state.robots[config.attach_to_robot]
                robot.attach(lidarnar)

            self.state.sensors[sensor_id] = lidar

    async def _create_sensor_pybullet(self, config: SensorConfig, sensor_id: str):
        """使用PyBullet创建传感器"""
        if config.sensor_type == "camera":
            # PyBullet相机实现
            camera_config = {
                "width": config.sensor_params.get("resolution", [640, 480])[0],
                "height": config.sensor_params.get("resolution", [640, 480])[1],
                "fov": config.sensor_params.get("fov", 60),
                "near": 0.1,
                "far": 10.0
            }

            self.state.sensors[sensor_id] = camera_config

    async def step_simulation(self, dt: Optional[float] = None) -> SimulationState:
        """
        执行仿真步进

        Args:
            dt: 时间步长（秒）

        Returns:
            当前仿真状态
        """
        if not self._initialized:
            raise RuntimeError("仿真环境未初始化")

        if not self.state.is_running:
            logger.warning("仿真未运行")
            return self.state

        try:
            if self._isaac_sim_available:
                await self._step_isaac_sim(dt)
            elif self._pybullet_available:
                await self._step_pybullet(dt)
            else:
                raise RuntimeError("无可用的仿真环境")

            self.state.time += dt or 1.0/60.0
            self.state.step_count += 1

            return self.state

        except Exception as e:
            logger.error(f"仿真步进失败: {e}")
            raise

    async def _step_isaac_sim(self, dt: Optional[float] = None):
        """Isaac Sim步进"""
        if dt:
            self.world.physics_step(dt)
        else:
            self.world.step(render=self.simulation_mode == SimulationMode.RENDER)

    async def _step_pybullet(self, dt: Optional[float] = None):
        """PyBullet步进"""
        step_dt = dt or 1.0/240.0  # PyBullet默认240Hz
        self._pybullet.stepSimulation()

    async def start_simulation(self):
        """启动仿真"""
        if not self._initialized:
            raise RuntimeError("仿真环境未初始化")

        logger.info("启动仿真")
        self.state.is_running = True
        self.state.is_paused = False

        if self._isaac_sim_available:
            self.world.reset()
        elif self._pybullet_available:
            self._pybullet.resetSimulation()

    async def stop_simulation(self):
        """停止仿真"""
        logger.info("停止仿真")
        self.state.is_running = False
        self.state.is_paused = False

    async def pause_simulation(self):
        """暂停仿真"""
        logger.info("暂停仿真")
        self.state.is_paused = True

    async def resume_simulation(self):
        """恢复仿真"""
        logger.info("恢复仿真")
        self.state.is_paused = False

    async def get_sensor_data(self, sensor_id: str) -> Dict[str, Any]:
        """
        获取传感器数据

        Args:
            sensor_id: 传感器ID

        Returns:
            传感器数据字典
        """
        if sensor_id not in self.sensors:
            raise ValueError(f"传感器不存在: {sensor_id}")

        try:
            if self._isaac_sim_available:
                return await self._get_sensor_data_isaac(sensor_id)
            elif self._pybullet_available:
                return await self._get_sensor_data_pybullet(sensor_id)
            else:
                raise RuntimeError("无可用的仿真环境")

        except Exception as e:
            logger.error(f"获取传感器数据失败: {e}")
            raise

    async def _get_sensor_data_isaac(self, sensor_id: str) -> Dict[str, Any]:
        """获取Isaac Sim传感器数据"""
        sensor = self.state.sensors[sensor_id]
        config = self.sensors[sensor_id]

        if config.sensor_type == "camera":
            # 获取相机数据
            rgb_data = sensor.get_rgba()
            depth_data = sensor.get_depth()

            return {
                "sensor_id": sensor_id,
                "sensor_type": "camera",
                "timestamp": self.state.time,
                "rgb_image": rgb_data,
                "depth_image": depth_data,
                "camera_info": {
                    "resolution": config.sensor_params.get("resolution", [640, 480]),
                    "fov": config.sensor_params.get("fov", 90.0)
                }
            }

        elif config.sensor_type == "lidar":
            # 获取激光雷达数据
            point_cloud = sensor.get_point_cloud_data()

            return {
                "sensor_id": sensor_id,
                "sensor_type": "lidar",
                "timestamp": self.state.time,
                "point_cloud": point_cloud,
                "lidar_info": {
                    "max_range": config.sensor_params.get("max_range", 10.0),
                    "min_range": config.sensor_params.get("min_range", 0.1)
                }
            }

        return {"sensor_id": sensor_id, "timestamp": self.state.time}

    async def _get_sensor_data_pybullet(self, sensor_id: str) -> Dict[str, Any]:
        """获取PyBullet传感器数据"""
        config = self.sensors[sensor_id]

        if config.sensor_type == "camera":
            # PyBullet相机渲染
            width = config.sensor_params.get("resolution", [640, 480])[0]
            height = config.sensor_params.get("resolution", [640, 480])[1]

            # 简化实现：返回模拟数据
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            depth_image = np.zeros((height, width), dtype=np.float32)

            return {
                "sensor_id": sensor_id,
                "sensor_type": "camera",
                "timestamp": self.state.time,
                "rgb_image": rgb_image,
                "depth_image": depth_image,
                "camera_info": {
                    "resolution": [width, height],
                    "fov": config.sensor_params.get("fov", 60.0)
                }
            }

        return {"sensor_id": sensor_id, "timestamp": self.state.time}

    async def get_robot_state(self, robot_id: str) -> Dict[str, Any]:
        """
        获取机器人状态

        Args:
            robot_id: 机器人ID

        Returns:
            机器人状态字典
        """
        if robot_id not in self.robots:
            raise ValueError(f"机器人不存在: {robot_id}")

        try:
            if self._isaac_sim_available:
                return await self._get_robot_state_isaac(robot_id)
            elif self._pybullet_available:
                return await self._get_robot_state_pybullet(robot_id)
            else:
                raise RuntimeError("无可用的仿真环境")

        except Exception as e:
            logger.error(f"获取机器人状态失败: {e}")
            raise

    async def _get_robot_state_isaac(self, robot_id: str) -> Dict[str, Any]:
        """获取Isaac Sim机器人状态"""
        robot = self.state.robots[robot_id]

        # 获取位置和姿态
        position, orientation = robot.get_world_pose()

        # 获取关节状态
        joint_positions = robot.get_joint_positions()
        joint_velocities = robot.get_joint_velocities()

        return {
            "robot_id": robot_id,
            "timestamp": self.state.time,
            "position": position.tolist(),
            "orientation": orientation.tolist(),
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities
        }

    async def _get_robot_state_pybullet(self, robot_id: str) -> Dict[str, Any]:
        """获取PyBullet机器人状态"""
        robot_body = self.state.robots[robot_id]

        # 获取位置和姿态
        position, orientation = self._pybullet.getBasePositionAndOrientation(robot_body)

        # 获取线性和角速度
        linear_vel, angular_vel = self._pybullet.getBaseVelocity(robot_body)

        return {
            "robot_id": robot_id,
            "timestamp": self.state.time,
            "position": list(position),
            "orientation": list(orientation),
            "linear_velocity": list(linear_vel),
            "angular_velocity": list(angular_vel)
        }

    async def set_robot_command(self, robot_id: str, command: Dict[str, Any]):
        """
        设置机器人控制命令

        Args:
            robot_id: 机器人ID
            command: 控制命令
        """
        if robot_id not in self.robots:
            raise ValueError(f"机器人不存在: {robot_id}")

        try:
            if self._isaac_sim_available:
                await self._set_robot_command_isaac(robot_id, command)
            elif self._pybullet_available:
                await self._set_robot_command_pybullet(robot_id, command)
            else:
                raise RuntimeError("无可用的仿真环境")

        except Exception as e:
            logger.error(f"设置机器人命令失败: {e}")
            raise

    async def _set_robot_command_isaac(self, robot_id: str, command: Dict[str, Any]):
        """设置Isaac Sim机器人命令"""
        robot = self.state.robots[robot_id]

        if "joint_positions" in command:
            await robot.set_joint_positions(command["joint_positions"])

        if "joint_velocities" in command:
            await robot.set_joint_velocities(command["joint_velocities"])

    async def _set_robot_command_pybullet(self, robot_id: str, command: Dict[str, Any]):
        """设置PyBullet机器人命令"""
        robot_body = self.state.robots[robot_id]

        if "linear_velocity" in command:
            self._pybullet.resetBaseVelocity(
                robot_body,
                linearVelocity=command["linear_velocity"]
            )

    async def reset_simulation(self):
        """重置仿真"""
        logger.info("重置仿真")

        if self._isaac_sim_available:
            self.world.reset()
        elif self._pybullet_available:
            self._pybullet.resetSimulation()

        # 重置状态
        self.state = SimulationState()
        self.robots.clear()
        self.sensors.clear()

    async def shutdown(self):
        """关闭仿真环境"""
        logger.info("关闭仿真环境")

        if self.state.is_running:
            await self.stop_simulation()

        if self.simulation_app:
            self.simulation_app.close()
        elif self._pybullet_available:
            self._pybullet.disconnect()

        self._initialized = False

    def get_simulation_info(self) -> Dict[str, Any]:
        """
        获取仿真环境信息

        Returns:
            仿真环境信息字典
        """
        return {
            "initialized": self._initialized,
            "isaac_sim_available": self._isaac_sim_available,
            "pybullet_available": self._pybullet_available,
            "simulation_mode": self.simulation_mode.value,
            "physics_engine": self.physics_engine.value,
            "headless": self.headless,
            "state": {
                "time": self.state.time,
                "step_count": self.state.step_count,
                "is_running": self.state.is_running,
                "is_paused": self.state.is_paused,
                "num_robots": len(self.robots),
                "num_sensors": len(self.sensors)
            }
        }

# 便捷函数
async def create_isaac_sim_interface(**kwargs) -> IsaacSimInterface:
    """
    创建Isaac Sim接口的便捷函数

    Args:
        **kwargs: 传递给IsaacSimInterface的参数

    Returns:
        初始化的IsaacSimInterface实例
    """
    interface = IsaacSimInterface(**kwargs)
    await interface.initialize()
    return interface