"""
执行引擎 - Executor

负责:
- 执行原子操作
- 监控执行状态
- 处理超时和重试
- 收集执行结果
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from loguru import logger

from brain.execution.operations.base import (
    Operation, 
    OperationResult, 
    OperationStatus,
    OperationType
)
from brain.state.world_state import WorldState
from brain.communication.robot_interface import RobotInterface


class ExecutionMode(Enum):
    """执行模式"""
    NORMAL = "normal"          # 正常执行
    STEP_BY_STEP = "step"      # 单步执行
    SIMULATION = "simulation"  # 仿真模式
    DRY_RUN = "dry_run"        # 干跑模式


@dataclass
class ExecutionContext:
    """执行上下文"""
    operation: Operation
    start_time: datetime
    timeout: float
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_operations: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    total_time: float = 0.0
    average_time: float = 0.0


class Executor:
    """
    执行引擎
    
    将操作发送到机器人平台执行，监控执行状态
    """
    
    def __init__(
        self, 
        world_state: WorldState,
        config: Optional[Dict[str, Any]] = None
    ):
        self.world_state = world_state
        self.config = config or {}
        
        self.mode = ExecutionMode.NORMAL
        self.metrics = ExecutionMetrics()
        
        # 操作处理器映射
        self.operation_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        
        # 当前执行上下文
        self.current_context: Optional[ExecutionContext] = None
        
        # 取消标志
        self._cancelled = False
        
        logger.info("Executor 初始化完成")
    
    def _register_default_handlers(self):
        """注册默认操作处理器"""
        # 移动操作
        self.register_handler("takeoff", self._handle_takeoff)
        self.register_handler("land", self._handle_land)
        self.register_handler("goto", self._handle_goto)
        self.register_handler("hover", self._handle_hover)
        self.register_handler("orbit", self._handle_orbit)
        self.register_handler("follow_path", self._handle_follow_path)
        self.register_handler("return_to_home", self._handle_return_to_home)
        
        # 感知操作
        self.register_handler("scan_area", self._handle_scan_area)
        self.register_handler("capture_image", self._handle_capture_image)
        self.register_handler("record_video", self._handle_record_video)
        self.register_handler("detect_objects", self._handle_detect_objects)
        
        # 任务操作
        self.register_handler("pickup", self._handle_pickup)
        self.register_handler("dropoff", self._handle_dropoff)
        
        # 控制操作
        self.register_handler("wait", self._handle_wait)
        self.register_handler("check_status", self._handle_check_status)
        
        # 通信操作
        self.register_handler("send_data", self._handle_send_data)
        self.register_handler("broadcast_status", self._handle_broadcast_status)
    
    def register_handler(
        self, 
        operation_name: str, 
        handler: Callable
    ):
        """注册操作处理器"""
        self.operation_handlers[operation_name] = handler
        logger.debug(f"注册操作处理器: {operation_name}")
    
    async def execute(
        self,
        operation: Operation,
        robot_interface: RobotInterface,
        timeout: Optional[float] = None
    ) -> OperationResult:
        """
        执行单个操作
        
        Args:
            operation: 要执行的操作
            robot_interface: 机器人通信接口
            timeout: 超时时间(秒)
            
        Returns:
            OperationResult: 执行结果
        """
        self._cancelled = False
        start_time = datetime.now()
        
        # 设置超时
        timeout = timeout or operation.estimated_duration * 2 + 30
        
        # 创建执行上下文
        self.current_context = ExecutionContext(
            operation=operation,
            start_time=start_time,
            timeout=timeout,
            max_retries=self.config.get("max_retries", 3)
        )
        
        logger.info(f"开始执行操作: {operation.name} [{operation.id}]")
        
        try:
            # 干跑模式
            if self.mode == ExecutionMode.DRY_RUN:
                return await self._dry_run_operation(operation)
            
            # 仿真模式
            if self.mode == ExecutionMode.SIMULATION:
                return await self._simulate_operation(operation)
            
            # 正常执行
            result = await self._execute_with_retry(
                operation=operation,
                robot_interface=robot_interface,
                context=self.current_context
            )
            
            # 更新指标
            self._update_metrics(operation, result, start_time)
            
            # 记录历史
            self._record_execution(operation, result)
            
            return result
            
        except asyncio.CancelledError:
            logger.warning(f"操作 {operation.name} 被取消")
            return OperationResult(
                status=OperationStatus.CANCELLED,
                error_message="操作被取消"
            )
        except Exception as e:
            logger.error(f"操作执行异常: {e}")
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message=str(e)
            )
        finally:
            self.current_context = None
    
    async def _execute_with_retry(
        self,
        operation: Operation,
        robot_interface: RobotInterface,
        context: ExecutionContext
    ) -> OperationResult:
        """带重试的执行"""
        last_error = None
        
        while context.retry_count <= context.max_retries:
            try:
                # 执行操作
                result = await asyncio.wait_for(
                    self._do_execute(operation, robot_interface),
                    timeout=context.timeout
                )
                
                if result.status == OperationStatus.SUCCESS:
                    return result
                
                # 检查是否可重试
                if not result.retryable:
                    return result
                
                last_error = result.error_message
                
            except asyncio.TimeoutError:
                last_error = f"操作超时 ({context.timeout}s)"
                logger.warning(f"操作 {operation.name} 超时")
            
            # 重试
            context.retry_count += 1
            if context.retry_count <= context.max_retries:
                wait_time = min(2 ** context.retry_count, 30)  # 指数退避
                logger.info(f"等待 {wait_time}s 后重试 ({context.retry_count}/{context.max_retries})")
                await asyncio.sleep(wait_time)
                self.metrics.retried += 1
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message=f"达到最大重试次数. 最后错误: {last_error}",
            retryable=False
        )
    
    async def _do_execute(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """实际执行操作"""
        handler = self.operation_handlers.get(operation.name)
        
        if handler:
            return await handler(operation, robot_interface)
        else:
            # 通用执行
            return await self._generic_execute(operation, robot_interface)
    
    async def _generic_execute(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """通用操作执行"""
        try:
            # 发送命令到机器人
            response = await robot_interface.send_command(
                command=operation.name,
                parameters=operation.parameters
            )
            
            if response.success:
                # 等待操作完成
                completion = await robot_interface.wait_for_completion(
                    operation_id=operation.id,
                    timeout=operation.estimated_duration * 2
                )
                
                if completion.completed:
                    return OperationResult(
                        status=OperationStatus.SUCCESS,
                        data=completion.data
                    )
                else:
                    return OperationResult(
                        status=OperationStatus.FAILED,
                        error_message=completion.error or "操作未能完成",
                        retryable=True
                    )
            else:
                return OperationResult(
                    status=OperationStatus.FAILED,
                    error_message=response.error or "命令发送失败",
                    retryable=True
                )
                
        except Exception as e:
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message=str(e),
                retryable=True
            )
    
    # ==================== 操作处理器 ====================
    
    async def _handle_takeoff(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理起飞操作"""
        altitude = operation.parameters.get("altitude", 10.0)
        
        logger.info(f"执行起飞: 目标高度 {altitude}m")
        
        # 预起飞检查
        preflight = await robot_interface.preflight_check()
        if not preflight.passed:
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message=f"预飞检查失败: {preflight.issues}",
                retryable=False
            )
        
        # 解锁
        await robot_interface.arm()
        
        # 起飞
        response = await robot_interface.takeoff(altitude=altitude)
        
        if response.success:
            # 等待到达目标高度
            reached = await robot_interface.wait_for_altitude(
                target=altitude,
                tolerance=0.5,
                timeout=30
            )
            
            if reached:
                return OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"altitude": altitude}
                )
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message="起飞失败",
            retryable=True
        )
    
    async def _handle_land(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理降落操作"""
        position = operation.parameters.get("position")
        
        logger.info(f"执行降落: {'指定位置' if position else '当前位置'}")
        
        if position:
            # 先飞到指定位置
            await robot_interface.goto(position=position)
        
        # 执行降落
        response = await robot_interface.land()
        
        if response.success:
            # 等待落地
            landed = await robot_interface.wait_for_landed(timeout=60)
            
            if landed:
                # 上锁
                await robot_interface.disarm()
                
                return OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"landed": True}
                )
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message="降落失败",
            retryable=True
        )
    
    async def _handle_goto(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理移动操作"""
        position = operation.parameters.get("position")
        speed = operation.parameters.get("speed")
        heading = operation.parameters.get("heading")
        
        if not position:
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message="缺少目标位置",
                retryable=False
            )
        
        logger.info(f"执行移动: 目标 {position}")
        
        response = await robot_interface.goto(
            position=position,
            speed=speed,
            heading=heading
        )
        
        if response.success:
            # 等待到达
            reached = await robot_interface.wait_for_position(
                target=position,
                tolerance=1.0,
                timeout=operation.estimated_duration * 2
            )
            
            if reached:
                return OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"position": position}
                )
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message="移动失败",
            retryable=True
        )
    
    async def _handle_hover(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理悬停操作"""
        duration = operation.parameters.get("duration", 5.0)
        position = operation.parameters.get("position")
        
        logger.info(f"执行悬停: {duration}秒")
        
        if position:
            await robot_interface.goto(position=position)
        
        response = await robot_interface.hover(duration=duration)
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None,
            data={"duration": duration}
        )
    
    async def _handle_orbit(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理环绕操作"""
        center = operation.parameters.get("center")
        radius = operation.parameters.get("radius", 10.0)
        speed = operation.parameters.get("speed", 2.0)
        direction = operation.parameters.get("direction", "cw")  # cw/ccw
        
        logger.info(f"执行环绕: 中心 {center}, 半径 {radius}m")
        
        response = await robot_interface.orbit(
            center=center,
            radius=radius,
            speed=speed,
            clockwise=(direction == "cw")
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None
        )
    
    async def _handle_follow_path(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理路径跟踪操作"""
        waypoints = operation.parameters.get("waypoints", [])
        speed = operation.parameters.get("speed")
        
        if not waypoints:
            return OperationResult(
                status=OperationStatus.FAILED,
                error_message="缺少航点",
                retryable=False
            )
        
        logger.info(f"执行路径跟踪: {len(waypoints)} 个航点")
        
        response = await robot_interface.follow_path(
            waypoints=waypoints,
            speed=speed
        )
        
        if response.success:
            # 等待路径完成
            completed = await robot_interface.wait_for_path_completion(
                timeout=operation.estimated_duration * 2
            )
            
            if completed:
                return OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"waypoints_completed": len(waypoints)}
                )
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message="路径跟踪失败",
            retryable=True
        )
    
    async def _handle_return_to_home(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理返航操作"""
        logger.info("执行返航")
        
        response = await robot_interface.return_to_home()
        
        if response.success:
            reached = await robot_interface.wait_for_home(
                timeout=operation.estimated_duration * 2
            )
            
            if reached:
                return OperationResult(
                    status=OperationStatus.SUCCESS,
                    data={"returned": True}
                )
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message="返航失败",
            retryable=True
        )
    
    async def _handle_scan_area(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理区域扫描操作"""
        area = operation.parameters.get("area")
        resolution = operation.parameters.get("resolution", "medium")
        
        logger.info(f"执行区域扫描: {area}")
        
        response = await robot_interface.scan_area(
            area=area,
            resolution=resolution
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None,
            data=response.data if response.success else None
        )
    
    async def _handle_capture_image(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理拍照操作"""
        target = operation.parameters.get("target")
        zoom = operation.parameters.get("zoom", 1.0)
        
        logger.info(f"执行拍照: 目标 {target}")
        
        response = await robot_interface.capture_image(
            target=target,
            zoom=zoom
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None,
            data={"image_path": response.data.get("path")} if response.success else None
        )
    
    async def _handle_record_video(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理录像操作"""
        duration = operation.parameters.get("duration", 30)
        quality = operation.parameters.get("quality", "1080p")
        
        logger.info(f"执行录像: {duration}秒, {quality}")
        
        response = await robot_interface.record_video(
            duration=duration,
            quality=quality
        )
        
        if response.success:
            # 等待录制完成
            await asyncio.sleep(duration)
            
            return OperationResult(
                status=OperationStatus.SUCCESS,
                data={"video_path": response.data.get("path")}
            )
        
        return OperationResult(
            status=OperationStatus.FAILED,
            error_message=response.error or "录像失败"
        )
    
    async def _handle_detect_objects(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理目标检测操作"""
        object_types = operation.parameters.get("object_types", ["all"])
        area = operation.parameters.get("area")
        
        logger.info(f"执行目标检测: 类型 {object_types}")
        
        response = await robot_interface.detect_objects(
            object_types=object_types,
            area=area
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None,
            data=response.data if response.success else None
        )
    
    async def _handle_pickup(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理拾取操作"""
        object_id = operation.parameters.get("object_id")
        grip_force = operation.parameters.get("grip_force", 50)
        
        logger.info(f"执行拾取: 对象 {object_id}")
        
        response = await robot_interface.pickup(
            object_id=object_id,
            grip_force=grip_force
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None,
            data={"picked_up": object_id} if response.success else None
        )
    
    async def _handle_dropoff(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理放下操作"""
        position = operation.parameters.get("position")
        release_height = operation.parameters.get("release_height", 1.0)
        
        logger.info(f"执行放下: 位置 {position}")
        
        response = await robot_interface.dropoff(
            position=position,
            release_height=release_height
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None
        )
    
    async def _handle_wait(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理等待操作"""
        duration = operation.parameters.get("duration", 1.0)
        
        logger.info(f"执行等待: {duration}秒")
        
        await asyncio.sleep(duration)
        
        return OperationResult(
            status=OperationStatus.SUCCESS,
            data={"waited": duration}
        )
    
    async def _handle_check_status(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理状态检查操作"""
        logger.info("执行状态检查")
        
        status = await robot_interface.get_status()
        
        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=status
        )
    
    async def _handle_send_data(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理数据发送操作"""
        data = operation.parameters.get("data")
        destination = operation.parameters.get("destination")
        
        logger.info(f"执行数据发送: 目标 {destination}")
        
        response = await robot_interface.send_data(
            data=data,
            destination=destination
        )
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None
        )
    
    async def _handle_broadcast_status(
        self,
        operation: Operation,
        robot_interface: RobotInterface
    ) -> OperationResult:
        """处理状态广播操作"""
        logger.info("执行状态广播")
        
        response = await robot_interface.broadcast_status()
        
        return OperationResult(
            status=OperationStatus.SUCCESS if response.success else OperationStatus.FAILED,
            error_message=response.error if not response.success else None
        )
    
    # ==================== 辅助方法 ====================
    
    async def _dry_run_operation(self, operation: Operation) -> OperationResult:
        """干跑模式 - 不实际执行"""
        logger.info(f"[DRY RUN] 操作: {operation.name}, 参数: {operation.parameters}")
        return OperationResult(
            status=OperationStatus.SUCCESS,
            data={"dry_run": True}
        )
    
    async def _simulate_operation(self, operation: Operation) -> OperationResult:
        """仿真模式 - 模拟执行"""
        logger.info(f"[SIMULATION] 操作: {operation.name}")
        
        # 模拟执行时间
        await asyncio.sleep(operation.estimated_duration * 0.1)
        
        return OperationResult(
            status=OperationStatus.SUCCESS,
            data={"simulated": True}
        )
    
    def _update_metrics(
        self,
        operation: Operation,
        result: OperationResult,
        start_time: datetime
    ):
        """更新执行指标"""
        self.metrics.total_operations += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        self.metrics.total_time += duration
        
        if result.status == OperationStatus.SUCCESS:
            self.metrics.successful += 1
        else:
            self.metrics.failed += 1
        
        self.metrics.average_time = (
            self.metrics.total_time / self.metrics.total_operations
        )
    
    def _record_execution(self, operation: Operation, result: OperationResult):
        """记录执行历史"""
        record = {
            "operation_id": operation.id,
            "operation_name": operation.name,
            "status": result.status.value,
            "timestamp": datetime.now().isoformat(),
            "data": result.data,
            "error": result.error_message
        }
        
        self.execution_history.append(record)
        
        # 限制历史长度
        max_history = self.config.get("max_history", 1000)
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]
    
    def cancel(self):
        """取消当前执行"""
        self._cancelled = True
        logger.warning("收到取消请求")
    
    def set_mode(self, mode: ExecutionMode):
        """设置执行模式"""
        self.mode = mode
        logger.info(f"执行模式设置为: {mode.value}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        return {
            "total_operations": self.metrics.total_operations,
            "successful": self.metrics.successful,
            "failed": self.metrics.failed,
            "retried": self.metrics.retried,
            "success_rate": (
                self.metrics.successful / self.metrics.total_operations
                if self.metrics.total_operations > 0 else 0
            ),
            "total_time": self.metrics.total_time,
            "average_time": self.metrics.average_time
        }

