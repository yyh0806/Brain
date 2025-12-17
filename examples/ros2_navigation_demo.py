#!/usr/bin/env python3
"""
ROS2导航Demo - 让小车前往建筑门口

这个示例演示了完整的感知驱动导航流程：
1. 接收自然语言指令
2. 使用VLM进行场景理解
3. 探索环境寻找目标
4. 规划并执行导航

使用方法:
    # 模拟模式（无需ROS2）
    python examples/ros2_navigation_demo.py
    
    # 真实ROS2环境
    python examples/ros2_navigation_demo.py --mode real

注意：需要先安装依赖
    pip install -e .
    
如果使用VLM，需要确保Ollama运行中：
    ollama run llava:latest
"""

import asyncio
import argparse
import sys
import os
import math
import re
import subprocess
import signal
import atexit
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# 导入模块
from brain.ros2.ros2_interface import ROS2Interface, ROS2Config, ROS2Mode
from brain.perception.ros2_sensor_manager import ROS2SensorManager
from brain.perception.vlm_perception import VLMPerception
from brain.cognitive.world_model import WorldModel
from brain.platforms.robot_capabilities import create_ugv_capabilities
from brain.operations.ros2_ugv import ROS2UGVOperations
from brain.navigation.exploration_planner import ExplorationPlanner, ExplorationConfig
from brain.ros2.control_adapter import ControlAdapter, PlatformType, PlatformCapabilities
from brain.navigation.smooth_executor import SmoothExecutor, SmoothExecutionConfig
from brain.navigation.intersection_navigator import IntersectionNavigator
try:
    from brain.visualization import RViz2Visualizer
    RVIZ2_AVAILABLE = True
except ImportError:
    RVIZ2_AVAILABLE = False


class NavigationDemo:
    """导航Demo"""
    
    def __init__(self, mode: str = "simulation", config_path: str = None):
        """
        Args:
            mode: 运行模式 "simulation" 或 "real"
            config_path: 配置文件路径
        """
        self.mode = ROS2Mode.REAL if mode == "real" else ROS2Mode.SIMULATION
        self.config = self._load_config(config_path)
        
        # 组件
        self.ros2: ROS2Interface = None
        self.sensor_manager: ROS2SensorManager = None
        self.vlm: VLMPerception = None
        self.world_model: WorldModel = None
        self.ugv_ops: ROS2UGVOperations = None
        self.planner: ExplorationPlanner = None
        self.control_adapter: ControlAdapter = None
        self.smooth_executor: SmoothExecutor = None
        self.intersection_navigator: IntersectionNavigator = None
        self.rviz2_visualizer = None
        self._rviz2_process = None  # RViz2进程
        
        self._initialized = False
    
    def _load_config(self, config_path: str = None) -> dict:
        """加载配置"""
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # 默认配置
        return {
            "ros2": {
                "topics": {
                    "cmd_vel": "/car3/twist",
                    "rgb_image": "/camera/rgb/image_raw",
                    "depth_image": "/camera/depth/image_raw",
                    "laser_scan": "/scan",
                    "odom": "/odom"
                }
            },
            "perception": {
                "vlm": {
                    "model": "llava:latest",
                    "ollama_host": "http://localhost:11434"
                }
            },
            "navigation": {
                "motion": {
                    "exploration_speed": 0.3,
                    "approach_speed": 0.2
                },
                "safety": {
                    "obstacle_distance": 0.5
                }
            }
        }
    
    async def initialize(self):
        """初始化所有组件"""
        logger.info(f"初始化导航系统 (模式: {self.mode.value})")
        
        # 1. ROS2接口
        ros2_config = ROS2Config(
            mode=self.mode,
            topics=self.config.get("ros2", {}).get("topics", {})
        )
        self.ros2 = ROS2Interface(ros2_config)
        await self.ros2.initialize()
        
        # 2. 传感器管理器
        self.sensor_manager = ROS2SensorManager(
            self.ros2,
            self.config.get("perception", {})
        )
        
        # 3. VLM感知
        vlm_config = self.config.get("perception", {}).get("vlm", {})
        self.vlm = VLMPerception(
            model=vlm_config.get("model", "llava:latest"),
            ollama_host=vlm_config.get("ollama_host", "http://localhost:11434")
        )
        
        # 4. 世界模型（统一版本，包含语义功能）
        self.world_model = WorldModel(
            self.config.get("world_model", {})
        )
        
        # 5. 机器人能力
        robot_capabilities = create_ugv_capabilities(
            name="UGV_Demo",
            cmd_vel_topic=self.config.get("ros2", {}).get("topics", {}).get("cmd_vel", "/car3/twist")
        )
        
        # 6. UGV操作
        self.ugv_ops = ROS2UGVOperations(
            self.ros2,
            self.config.get("navigation", {}).get("motion", {})
        )
        
        # 7. 控制适配器
        robot_config = self.config.get("robot", {})
        platform_type_str = robot_config.get("platform_type", "ackermann")
        platform_type = PlatformType.ACKERMANN if platform_type_str == "ackermann" else PlatformType.DIFFERENTIAL
        
        kinematics = robot_config.get("kinematics", {})
        capabilities = PlatformCapabilities(
            platform_type=platform_type,
            max_linear_speed=kinematics.get("max_linear_speed", 1.0),
            max_angular_speed=kinematics.get("max_angular_speed", 1.0),
            max_acceleration=kinematics.get("max_acceleration", 0.5),
            min_turn_radius=kinematics.get("min_turn_radius", 0.0),
            wheelbase=kinematics.get("wheelbase", 2.0),
            track_width=kinematics.get("track_width", 1.0)
        )
        
        self.control_adapter = ControlAdapter(
            self.ros2,
            platform_type,
            capabilities,
            kinematics
        )
        
        # 8. 平滑执行器
        smooth_config = self.config.get("navigation", {}).get("smooth_execution", {})
        smooth_exec_config = SmoothExecutionConfig(
            control_rate=smooth_config.get("control_rate", 10.0),
            perception_update_rate=smooth_config.get("perception_update_rate", 2.0),
            vlm_analysis_interval=smooth_config.get("vlm_analysis_interval", 3.5),
            obstacle_check_distance=smooth_config.get("obstacle_check_distance", 1.0),
            emergency_stop_distance=smooth_config.get("emergency_stop_distance", 0.3),
            speed_adjustment_factor=smooth_config.get("speed_adjustment_factor", 0.8),
            min_speed=smooth_config.get("min_speed", 0.1),
            max_speed=smooth_config.get("max_speed", 0.5)
        )
        
        self.smooth_executor = SmoothExecutor(
            self.control_adapter,
            self.sensor_manager,
            self.world_model,
            self.vlm,
            smooth_exec_config
        )
        
        # 9. 路口导航器
        intersection_config = self.config.get("navigation", {}).get("intersection", {})
        self.intersection_navigator = IntersectionNavigator(
            self.control_adapter,
            self.smooth_executor,
            self.sensor_manager,
            self.world_model,
            self.vlm,
            intersection_config
        )
        
        # 10. 探索规划器
        nav_config = self.config.get("navigation", {})
        exploration_config = ExplorationConfig(
            max_exploration_time=nav_config.get("exploration", {}).get("max_exploration_time", 300),
            exploration_speed=nav_config.get("motion", {}).get("exploration_speed", 0.3),
            approach_speed=nav_config.get("motion", {}).get("approach_speed", 0.2),
            obstacle_distance=nav_config.get("safety", {}).get("obstacle_distance", 0.5)
        )
        
        self.planner = ExplorationPlanner(
            world_model=self.world_model,
            vlm=self.vlm,
            sensor_manager=self.sensor_manager,
            ugv_ops=self.ugv_ops,
            robot_capabilities=robot_capabilities,
            config=exploration_config
        )
        
        # 11. RViz2地图可视化器（可选）
        viz_config = self.config.get("visualization", {})
        
        if RVIZ2_AVAILABLE and self.mode == ROS2Mode.REAL:
            try:
                rviz_config = viz_config.get("rviz2", {})
                if rviz_config.get("enabled", True):
                    self.rviz2_visualizer = RViz2Visualizer(
                        ros2_interface=self.ros2,
                        world_model=self.world_model,
                        config=rviz_config
                    )
                    # 在ROS2节点初始化后调用
                    self.rviz2_visualizer.initialize()
                    logger.info("RViz2可视化器初始化完成")
                    
                    # 自动启动RViz2窗口
                    self._start_rviz2()
            except Exception as e:
                logger.warning(f"RViz2可视化器初始化失败: {e}，继续运行（无可视化）")
                self.rviz2_visualizer = None
        else:
            if viz_config.get("enabled", True):
                logger.info("RViz2可视化不可用（ROS2未启用或RViz2未安装）")
        
        self._initialized = True
        logger.info("导航系统初始化完成")
    
    def _start_rviz2(self):
        """自动启动RViz2"""
        if self._rviz2_process is not None:
            # 已经在运行
            return
        
        try:
            # 查找RViz2配置文件
            project_root = Path(__file__).parent.parent
            rviz_config_file = project_root / "brain_visualization.rviz"
            
            # 构建启动命令
            cmd = ["rviz2"]
            if rviz_config_file.exists():
                cmd.extend(["-d", str(rviz_config_file)])
                logger.info(f"使用RViz2配置文件: {rviz_config_file}")
            else:
                logger.warning(f"RViz2配置文件不存在: {rviz_config_file}，使用默认配置")
            
            # 启动RViz2（在后台）
            # 使用nohup和重定向输出，避免阻塞
            env = os.environ.copy()
            # 确保ROS2环境变量已设置
            if "ROS_DISTRO" not in env:
                # 尝试source ROS2环境
                ros_setup = "/opt/ros/galactic/setup.bash"
                if os.path.exists(ros_setup):
                    # 通过bash source环境变量
                    import subprocess as sp
                    result = sp.run(
                        f'source {ros_setup} && env',
                        shell=True,
                        capture_output=True,
                        text=True,
                        executable='/bin/bash'
                    )
                    for line in result.stdout.splitlines():
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env[key] = value
            
            # 启动RViz2进程
            self._rviz2_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                preexec_fn=os.setsid  # 创建新的进程组
            )
            
            logger.info(f"RViz2已自动启动 (PID: {self._rviz2_process.pid})")
            logger.info("  RViz2窗口应该已经打开，请查看可视化数据")
            
            # 注册退出时关闭RViz2
            atexit.register(self._stop_rviz2)
            
        except FileNotFoundError:
            logger.warning("rviz2命令未找到，请确保已安装ROS2和RViz2")
            logger.warning("  安装命令: sudo apt install ros-galactic-rviz2")
        except Exception as e:
            logger.warning(f"启动RViz2失败: {e}")
            logger.warning("  可以手动启动: rviz2 -d brain_visualization.rviz")
    
    def _stop_rviz2(self):
        """停止RViz2进程"""
        if self._rviz2_process is not None:
            try:
                # 发送SIGTERM信号给整个进程组
                os.killpg(os.getpgid(self._rviz2_process.pid), signal.SIGTERM)
                # 等待进程结束
                self._rviz2_process.wait(timeout=2)
                logger.info("RViz2已关闭")
            except ProcessLookupError:
                # 进程已经结束
                pass
            except subprocess.TimeoutExpired:
                # 强制终止
                try:
                    os.killpg(os.getpgid(self._rviz2_process.pid), signal.SIGKILL)
                    logger.warning("RViz2被强制终止")
                except:
                    pass
            except Exception as e:
                logger.warning(f"关闭RViz2时出错: {e}")
            finally:
                self._rviz2_process = None
    
    async def navigate_to_target(
        self,
        target_description: str,
        interactive: bool = True
    ):
        """
        导航到指定目标
        
        Args:
            target_description: 目标描述（如"建筑的门口"或"前面的路口右转"）
            interactive: 是否交互式（显示进度）
        """
        if not self._initialized:
            await self.initialize()
        
        # 启动可视化（如果可用）
        if self.rviz2_visualizer:
            logger.info("RViz2可视化已就绪")
            logger.info(f"  地图话题: {self.rviz2_visualizer.config.map_topic}")
            logger.info(f"  轨迹话题: {self.rviz2_visualizer.config.path_topic}")
            logger.info(f"  标记话题: {self.rviz2_visualizer.config.markers_topic}")
            logger.info(f"  位姿话题: {self.rviz2_visualizer.config.pose_topic}")
            
            # 自动启动RViz2
            self._start_rviz2()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"任务: {target_description}")
        logger.info(f"{'='*50}\n")
        
        # 检查是否是路口导航任务
        is_intersection_task = any(keyword in target_description for keyword in [
            "路口", "intersection", "交叉", "右转", "左转", "直行", 
            "turn right", "turn left", "go straight"
        ])
        
        if is_intersection_task:
            return await self._navigate_intersection(target_description, interactive)
        else:
            return await self._navigate_exploration(target_description, interactive)
    
    async def _navigate_intersection(
        self,
        target_description: str,
        interactive: bool = True
    ):
        """路口导航"""
        logger.info("使用路口导航器")
        
        # 提取转弯方向
        turn_direction = "right"  # 默认右转
        if "左转" in target_description or "left" in target_description.lower():
            turn_direction = "left"
        elif "直行" in target_description or "straight" in target_description.lower():
            turn_direction = "straight"
        elif "右转" in target_description or "right" in target_description.lower():
            turn_direction = "right"
        
        logger.info(f"转弯方向: {turn_direction}")
        
        # 提取前进距离（从"前进X米"或"前进 X 米"中提取）
        forward_distance = 0.0
        distance_match = re.search(r'前进\s*(\d+(?:\.\d+)?)\s*米', target_description)
        if distance_match:
            forward_distance = float(distance_match.group(1))
            logger.info(f"检测到前进距离: {forward_distance} 米")
        elif "前进" in target_description:
            # 如果没有明确距离，尝试提取数字
            num_match = re.search(r'(\d+(?:\.\d+)?)\s*米', target_description)
            if num_match:
                forward_distance = float(num_match.group(1))
                logger.info(f"检测到前进距离: {forward_distance} 米")
        
        # 进度回调
        def progress_callback(message):
            logger.info(f"[路口导航] {message}")
            if interactive:
                print(f"  → {message}")
        
        # 重规划回调
        def replan_callback(message: str = ""):
            logger.warning(f"触发重规划: {message}")
            if interactive:
                print(f"  ⚠️  检测到环境变化，触发重规划: {message}")
        
        # 执行路口导航
        viz_task = None
        try:
            # 启动可视化更新循环（如果使用RViz2）
            if self.rviz2_visualizer:
                async def update_viz():
                    update_count = 0
                    while True:
                        try:
                            self.rviz2_visualizer.update()
                            update_count += 1
                            if update_count % 10 == 0:  # 每5秒打印一次
                                logger.debug(f"RViz2可视化更新中... (已更新 {update_count} 次)")
                            await asyncio.sleep(0.2)  # 5 Hz更新，更频繁
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.warning(f"可视化更新错误: {e}")
                            await asyncio.sleep(0.2)
                viz_task = asyncio.create_task(update_viz())
                logger.info("RViz2可视化更新循环已启动")
            
            # 1. 执行转弯
            turn_success = await self.intersection_navigator.execute_turn(
                turn_direction,
                replan_callback=replan_callback
            )
            
            if not turn_success:
                if interactive:
                    print(f"\n{'='*50}")
                    print(f"路口导航结果: ❌ 失败（转弯失败）")
                    print(f"{'='*50}")
                return {"success": False, "type": "intersection", "direction": turn_direction, "error": "转弯失败"}
            
            # 2. 如果指定了前进距离，执行前进
            if forward_distance > 0:
                logger.info(f"转弯完成，开始前进 {forward_distance} 米")
                if interactive:
                    print(f"\n  → 转弯完成，开始前进 {forward_distance} 米...")
                
                # 获取当前位姿作为起点
                perception = await self.sensor_manager.get_fused_perception()
                if not perception or not perception.pose:
                    logger.warning("无法获取位姿，使用默认前进时间")
                    # 使用时间估算：距离/速度
                    estimated_time = forward_distance / 0.5  # 假设速度0.5 m/s
                    await self.smooth_executor.execute_continuous(
                        target_speed=0.5,
                        target_angular=0.0,
                        duration=estimated_time,
                        progress_callback=progress_callback
                    )
                else:
                    start_pose = perception.pose
                    start_x = start_pose.x
                    start_y = start_pose.y
                    
                    # 持续前进直到达到目标距离
                    target_reached = False
                    max_time = forward_distance / 0.3 + 5.0  # 最大时间（考虑误差）
                    start_time = asyncio.get_event_loop().time()
                    
                    while not target_reached:
                        # 检查是否超时
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > max_time:
                            logger.warning(f"前进超时（{elapsed:.1f}秒），停止")
                            break
                        
                        # 获取当前位姿
                        perception = await self.sensor_manager.get_fused_perception()
                        if perception and perception.pose:
                            current_x = perception.pose.x
                            current_y = perception.pose.y
                            
                            # 计算已前进距离
                            dx = current_x - start_x
                            dy = current_y - start_y
                            traveled = math.sqrt(dx**2 + dy**2)
                            
                            remaining = forward_distance - traveled
                            
                            if remaining <= 0.2:  # 到达目标（0.2米容差）
                                target_reached = True
                                logger.info(f"已前进 {traveled:.2f} 米，到达目标")
                                if interactive:
                                    print(f"  ✅ 已前进 {traveled:.2f} 米，到达目标")
                                break
                            
                            # 继续前进
                            await self.smooth_executor.execute_continuous(
                                target_speed=0.5,
                                target_angular=0.0,
                                duration=0.5,  # 每0.5秒检查一次
                                progress_callback=lambda msg: None
                            )
                        else:
                            # 无法获取位姿，使用时间估算
                            await asyncio.sleep(0.1)
                    
                    if not target_reached:
                        logger.warning("前进未完全完成，但已停止")
                
                logger.info(f"前进 {forward_distance} 米完成")
                if interactive:
                    print(f"  ✅ 前进完成")
            
            if interactive:
                print(f"\n{'='*50}")
                print(f"路口导航结果: ✅ 成功")
                if forward_distance > 0:
                    print(f"已完成: {turn_direction}转 + 前进 {forward_distance} 米")
                print(f"{'='*50}")
            
            return {
                "success": True, 
                "type": "intersection", 
                "direction": turn_direction,
                "forward_distance": forward_distance
            }
            
        except Exception as e:
            logger.error(f"路口导航异常: {e}")
            import traceback
            traceback.print_exc()
            if interactive:
                print(f"❌ 路口导航失败: {e}")
            return {"success": False, "type": "intersection", "error": str(e)}
        finally:
            # 停止可视化更新循环
            if viz_task:
                viz_task.cancel()
                try:
                    await viz_task
                except asyncio.CancelledError:
                    pass
    
    async def _navigate_exploration(
        self,
        target_description: str,
        interactive: bool = True
    ):
        """探索式导航"""
        logger.info("使用探索规划器")
        
        # 进度回调
        def progress_callback(message, state):
            logger.info(f"[{state.value}] {message}")
            if interactive:
                print(f"  → {message}")
        
        # 找到目标回调
        async def target_found_callback(target):
            logger.info(f"🎯 找到目标: {target.label}")
            logger.info(f"   位置: ({target.world_position[0]:.1f}, {target.world_position[1]:.1f})")
            if interactive:
                print(f"\n🎯 找到目标: {target.label}")
                print(f"   位置: ({target.world_position[0]:.1f}, {target.world_position[1]:.1f})")
        
        # 执行导航
        result = await self.planner.execute_exploration(
            target_description,
            progress_callback=progress_callback,
            target_found_callback=target_found_callback
        )
        
        # 显示结果
        logger.info(f"\n{'='*50}")
        logger.info(f"导航结果: {'成功' if result.success else '失败'}")
        logger.info(f"状态: {result.state.value}")
        logger.info(f"耗时: {result.elapsed_time:.1f} 秒")
        logger.info(f"执行操作: {result.operations_executed} 次")
        if result.target_found:
            logger.info(f"目标位置: {result.target_position}")
            logger.info(f"最终距离: {result.final_distance:.2f} 米")
        logger.info(f"{'='*50}\n")
        
        if interactive:
            print(f"\n{'='*50}")
            print(f"导航结果: {'✅ 成功' if result.success else '❌ 失败'}")
            print(f"消息: {result.message}")
            print(f"{'='*50}")
        
        return result
    
    async def demo_basic_control(self):
        """演示基本控制"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("\n演示基本控制命令")
        print("\n=== 基本控制演示 ===")
        
        # 前进
        print("1. 前进 1 米...")
        op = self.ugv_ops.move_forward(1.0, speed=0.5)
        await self.ugv_ops.execute(op)
        await asyncio.sleep(0.5)
        
        # 左转
        print("2. 左转 45 度...")
        import math
        op = self.ugv_ops.turn_left(math.pi / 4)
        await self.ugv_ops.execute(op)
        await asyncio.sleep(0.5)
        
        # 前进
        print("3. 前进 0.5 米...")
        op = self.ugv_ops.move_forward(0.5, speed=0.3)
        await self.ugv_ops.execute(op)
        await asyncio.sleep(0.5)
        
        # 停止
        print("4. 停止")
        op = self.ugv_ops.stop()
        await self.ugv_ops.execute(op)
        
        print("=== 基本控制演示完成 ===\n")
    
    async def demo_scene_analysis(self):
        """演示场景分析"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("\n演示VLM场景分析")
        print("\n=== VLM场景分析演示 ===")
        
        # 获取感知数据
        perception = await self.sensor_manager.get_fused_perception()
        
        if perception.rgb_image is not None:
            print("正在分析场景...")
            
            # 场景描述
            scene = await self.vlm.describe_scene(perception.rgb_image)
            
            print(f"\n场景摘要: {scene.summary}")
            print(f"\n检测到的物体:")
            for obj in scene.objects:
                print(f"  - {obj.label}: {obj.position_description}")
            
            print(f"\n空间关系:")
            for rel in scene.spatial_relations:
                print(f"  - {rel}")
            
            print(f"\n导航提示:")
            for hint in scene.navigation_hints:
                print(f"  - {hint}")
            
            # 搜索特定目标
            print("\n搜索 '门'...")
            search_result = await self.vlm.find_target(perception.rgb_image, "门")
            
            if search_result.found:
                print(f"  找到目标! 置信度: {search_result.confidence:.2f}")
                print(f"  建议动作: {search_result.suggested_action}")
            else:
                print("  未找到目标")
        else:
            print("无法获取图像数据")
        
        print("\n=== VLM场景分析演示完成 ===\n")
    
    async def demo_exploration(self):
        """演示探索功能"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("\n演示探索功能")
        print("\n=== 探索功能演示 ===")
        
        # 简单探索
        print("开始探索环境...")
        
        for i in range(3):
            print(f"\n探索步骤 {i+1}/3")
            
            # 获取感知
            perception = await self.sensor_manager.get_fused_perception()
            
            # 显示状态
            pose = perception.pose
            if pose:
                print(f"  位置: ({pose.x:.1f}, {pose.y:.1f})")
                print(f"  朝向: {math.degrees(pose.yaw):.0f}°")
            
            # 前方距离
            front_dist = perception.get_front_distance()
            print(f"  前方距离: {front_dist:.1f} 米")
            
            # 根据情况移动
            if front_dist > 1.0:
                print("  → 前进")
                op = self.ugv_ops.move_forward(1.0, speed=0.3)
            else:
                print("  → 转向")
                import random
                if random.random() > 0.5:
                    op = self.ugv_ops.rotate_left(math.pi / 4)
                else:
                    op = self.ugv_ops.rotate_right(math.pi / 4)
            
            await self.ugv_ops.execute(op)
            await asyncio.sleep(1.0)
        
        print("\n=== 探索功能演示完成 ===\n")
    
    async def shutdown(self):
        """关闭系统"""
        # 停止可视化
        if self.rviz2_visualizer:
            try:
                self.rviz2_visualizer.stop()
                logger.info("RViz2可视化已停止")
            except Exception as e:
                logger.warning(f"停止RViz2可视化失败: {e}")
        
        # 关闭RViz2进程
        self._stop_rviz2()
        
        if self.ros2:
            await self.ros2.shutdown()
        logger.info("系统已关闭")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ROS2导航Demo")
    parser.add_argument(
        "--mode", 
        choices=["simulation", "real"],
        default="simulation",
        help="运行模式"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--demo",
        choices=["navigation", "control", "scene", "exploration", "all"],
        default="navigation",
        help="演示类型"
    )
    parser.add_argument(
        "--target",
        default="前面建筑的门口",
        help="导航目标描述"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║           ROS2 感知驱动导航系统 Demo                         ║
║           Perception-Driven Navigation Demo                  ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # 创建Demo实例
    demo = NavigationDemo(
        mode=args.mode,
        config_path=args.config
    )
    
    try:
        if args.demo == "navigation" or args.demo == "all":
            print(f"\n🚗 开始导航任务: '{args.target}'")
            print("-" * 50)
            result = await demo.navigate_to_target(args.target)
        
        if args.demo == "control" or args.demo == "all":
            await demo.demo_basic_control()
        
        if args.demo == "scene" or args.demo == "all":
            await demo.demo_scene_analysis()
        
        if args.demo == "exploration" or args.demo == "all":
            await demo.demo_exploration()
            
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await demo.shutdown()
        print("\n程序结束")


if __name__ == "__main__":
    asyncio.run(main())

