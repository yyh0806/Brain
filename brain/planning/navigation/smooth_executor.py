"""
平滑执行器 - Smooth Executor

负责:
- 持续前进+周期感知微调（避免"停-感知-走"的呆滞行为）
- 实时障碍物检测和路径调整
- 动态速度控制
"""

import asyncio
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from brain.communication.control_adapter import ControlAdapter, PlatformType, PlatformCapabilities
from brain.perception.ros2_sensor_manager import ROS2SensorManager, PerceptionData
from brain.perception.vlm_perception import VLMPerception, SceneDescription
from brain.cognitive.world_model import WorldModel


@dataclass
class SmoothExecutionConfig:
    """平滑执行配置"""
    control_rate: float = 10.0  # Hz，控制频率
    perception_update_rate: float = 2.0  # Hz，感知更新频率
    vlm_analysis_interval: float = 3.0  # 秒，VLM分析间隔
    obstacle_check_distance: float = 1.0  # 米，障碍物检测距离
    emergency_stop_distance: float = 0.3  # 米，紧急停止距离
    speed_adjustment_factor: float = 0.8  # 速度调整因子
    min_speed: float = 0.1  # 最小速度
    max_speed: float = 0.5  # 最大速度


class SmoothExecutor:
    """
    平滑执行器
    
    实现持续前进+周期感知微调的执行模式
    """
    
    def __init__(
        self,
        control_adapter: ControlAdapter,
        sensor_manager: ROS2SensorManager,
        world_model: WorldModel,
        vlm: Optional[VLMPerception] = None,
        config: Optional[SmoothExecutionConfig] = None
    ):
        self.control = control_adapter
        self.sensors = sensor_manager
        self.world_model = world_model
        self.vlm = vlm
        self.config = config or SmoothExecutionConfig()
        
        # 执行状态
        self._is_running = False
        self._abort_requested = False
        self._current_speed = 0.0
        self._current_angular = 0.0
        self._target_speed = 0.0
        self._target_angular = 0.0
        
        # 感知状态
        self._last_perception: Optional[PerceptionData] = None
        self._last_vlm_analysis: Optional[datetime] = None
        self._last_scene: Optional[SceneDescription] = None
        
        # 回调
        self._obstacle_callback: Optional[Callable] = None
        self._progress_callback: Optional[Callable] = None
        
        logger.info("SmoothExecutor 初始化完成")
    
    async def execute_continuous(
        self,
        target_speed: float = 0.5,
        target_angular: float = 0.0,
        duration: float = 0.0,
        obstacle_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[Callable] = None
    ):
        """
        执行持续运动
        
        Args:
            target_speed: 目标线速度
            target_angular: 目标角速度
            duration: 持续时间（秒），0表示持续直到被中止
            obstacle_check: 障碍物检测回调
            progress_callback: 进度回调
        """
        self._is_running = True
        self._abort_requested = False
        self._target_speed = target_speed
        self._target_angular = target_angular
        self._obstacle_callback = obstacle_check
        self._progress_callback = progress_callback
        
        control_interval = 1.0 / self.config.control_rate
        perception_interval = 1.0 / self.config.perception_update_rate
        
        start_time = datetime.now()
        last_control_time = datetime.now()
        last_perception_time = datetime.now()
        last_vlm_time = datetime.now()
        
        logger.info(f"开始平滑执行: speed={target_speed}, angular={target_angular}")
        
        try:
            while not self._abort_requested:
                now = datetime.now()
                
                # 检查持续时间
                if duration > 0:
                    elapsed = (now - start_time).total_seconds()
                    if elapsed >= duration:
                        logger.info("达到执行时间，停止")
                        break
                
                # 控制循环（高频）
                if (now - last_control_time).total_seconds() >= control_interval:
                    await self._control_loop()
                    last_control_time = now
                
                # 感知循环（中频）
                if (now - last_perception_time).total_seconds() >= perception_interval:
                    await self._perception_loop()
                    last_perception_time = now
                
                # VLM分析循环（低频）
                if self.vlm and (now - last_vlm_time).total_seconds() >= self.config.vlm_analysis_interval:
                    await self._vlm_analysis_loop()
                    last_vlm_time = now
                
                # 短暂休眠以避免CPU占用过高
                await asyncio.sleep(0.01)
            
            # 停止
            await self.control.stop()
            logger.info("平滑执行完成")
            
        except Exception as e:
            logger.error(f"平滑执行异常: {e}")
            await self.control.stop()
            raise
        finally:
            self._is_running = False
    
    async def _control_loop(self):
        """控制循环：根据当前状态调整速度"""
        # 获取最新感知数据
        perception = self._last_perception
        
        # 障碍物检测
        if perception:
            front_dist = perception.get_front_distance()
            
            # 紧急停止
            if front_dist < self.config.emergency_stop_distance:
                self._current_speed = 0.0
                self._current_angular = 0.0
                await self.control.stop()
                if self._obstacle_callback:
                    self._obstacle_callback()
                return
            
            # 根据前方距离调整速度
            if front_dist < self.config.obstacle_check_distance:
                # 减速
                safe_speed = max(
                    self.config.min_speed,
                    (front_dist - self.config.emergency_stop_distance) * self.config.speed_adjustment_factor
                )
                self._current_speed = min(safe_speed, self._target_speed)
            else:
                # 正常速度
                self._current_speed = self._target_speed
        
        # 平滑调整速度（避免突变）
        speed_diff = self._target_speed - self._current_speed
        if abs(speed_diff) > 0.05:
            self._current_speed += speed_diff * 0.3  # 平滑过渡
        
        angular_diff = self._target_angular - self._current_angular
        if abs(angular_diff) > 0.05:
            self._current_angular += angular_diff * 0.3
        
        # 应用控制
        await self.control.set_velocity_continuous(
            self._current_speed,
            self._current_angular
        )
    
    async def _perception_loop(self):
        """感知循环：更新感知数据并检测障碍物"""
        try:
            perception = await self.sensors.get_fused_perception()
            self._last_perception = perception
            
            # 更新世界模型（直接传递 PerceptionData）
            changes = self.world_model.update_from_perception(perception)
            
            # 检查是否需要重规划
            significant_changes = self.world_model.detect_significant_changes()
            if significant_changes:
                for change in significant_changes:
                    if change.requires_replan:
                        logger.warning(f"检测到需要重规划的变化: {change.description}")
                        if self._progress_callback:
                            self._progress_callback(f"环境变化: {change.description}")
        
        except Exception as e:
            logger.warning(f"感知循环异常: {e}")
    
    async def _vlm_analysis_loop(self):
        """VLM分析循环：场景理解和目标检测"""
        if not self.vlm or not self._last_perception:
            return
        
        try:
            if self._last_perception.rgb_image is None:
                return
            
            # 场景分析
            scene = await self.vlm.describe_scene(self._last_perception.rgb_image)
            self._last_scene = scene
            self._last_vlm_analysis = datetime.now()
            
            # 根据场景调整行为
            # 例如：检测到路口、障碍物等
            
        except Exception as e:
            logger.warning(f"VLM分析循环异常: {e}")
    
    def adjust_speed(self, speed: float, angular: float = None):
        """动态调整目标速度"""
        self._target_speed = max(self.config.min_speed, min(speed, self.config.max_speed))
        if angular is not None:
            self._target_angular = angular
        logger.debug(f"调整速度: {self._target_speed}, {self._target_angular}")
    
    def abort(self):
        """中止执行"""
        self._abort_requested = True
        logger.info("请求中止平滑执行")
    
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._is_running
    
    def get_current_speed(self) -> Tuple[float, float]:
        """获取当前速度"""
        return (self._current_speed, self._current_angular)
    
    def get_last_scene(self) -> Optional[SceneDescription]:
        """获取最后一次VLM场景分析结果"""
        return self._last_scene

