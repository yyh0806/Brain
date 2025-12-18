"""
Command Queuing System for Brain Communication Layer

Provides intelligent command queuing with:
- Priority-based command processing
- Command batching and optimization
- Rate limiting and throttling
- Command conflict resolution
- Performance monitoring
"""

import asyncio
import heapq
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from loguru import logger

from .ros2_interface import TwistCommand


class CommandPriority(Enum):
    """命令优先级"""
    EMERGENCY = 0      # 紧急停止等安全命令
    CRITICAL = 1      # 避障等关键命令
    HIGH = 2          # 任务执行命令
    NORMAL = 3        # 常规导航命令
    LOW = 4           # 优化调整命令
    BACKGROUND = 5    # 后台任务


class CommandType(Enum):
    """命令类型"""
    STOP = "stop"
    MOVE = "move"
    TURN = "turn"
    AVOID = "avoid"
    NAVIGATE = "navigate"
    ADJUST = "adjust"
    CUSTOM = "custom"


@dataclass
class QueuedCommand:
    """队列中的命令"""
    id: str
    command_type: CommandType
    priority: CommandPriority
    command: Any
    timestamp: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 5.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """优先级队列比较（数值越小优先级越高）"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        # 相同优先级按时间排序
        return self.timestamp < other.timestamp

    def is_expired(self) -> bool:
        """检查命令是否过期"""
        if self.deadline:
            return datetime.now() > self.deadline
        return False

    def is_timed_out(self) -> bool:
        """检查命令是否超时"""
        return (datetime.now() - self.timestamp).total_seconds() > self.timeout

    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries


@dataclass
class CommandResult:
    """命令执行结果"""
    command_id: str
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class CommandConflictResolver:
    """命令冲突解决器"""

    def __init__(self):
        self.conflict_rules: Dict[Tuple[CommandType, CommandType], Callable] = {
            # 相反方向的运动命令冲突
            (CommandType.MOVE, CommandType.MOVE): self._resolve_move_conflict,
            # 转向与直线运动冲突
            (CommandType.TURN, CommandType.MOVE): self._resolve_turn_move_conflict,
            # 停止命令优先级最高
            (CommandType.STOP, CommandType.MOVE): self._resolve_stop_conflict,
            (CommandType.STOP, CommandType.TURN): self._resolve_stop_conflict,
        }

    def resolve_conflict(self, cmd1: QueuedCommand, cmd2: QueuedCommand) -> Optional[QueuedCommand]:
        """解决命令冲突，返回应该保留的命令，None表示都丢弃"""
        key = (cmd1.command_type, cmd2.command_type)
        reverse_key = (cmd2.command_type, cmd1.command_type)

        if key in self.conflict_rules:
            return self.conflict_rules[key](cmd1, cmd2)
        elif reverse_key in self.conflict_rules:
            return self.conflict_rules[reverse_key](cmd2, cmd1)

        # 默认保留优先级更高的命令
        return cmd1 if cmd1.priority.value < cmd2.priority.value else cmd2

    def _resolve_move_conflict(self, cmd1: QueuedCommand, cmd2: QueuedCommand) -> QueuedCommand:
        """解决移动命令冲突"""
        # 如果方向相反，保留更紧急的命令
        twist1, twist2 = cmd1.command, cmd2.command

        if isinstance(twist1, TwistCommand) and isinstance(twist2, TwistCommand):
            # 检查是否相反方向
            if (twist1.linear_x * twist2.linear_x < 0) or (twist1.angular_z * twist2.angular_z < 0):
                return cmd1 if cmd1.priority.value < cmd2.priority.value else cmd2

        # 同方向，保留最新的命令
        return cmd2 if cmd2.timestamp > cmd1.timestamp else cmd1

    def _resolve_turn_move_conflict(self, turn_cmd: QueuedCommand, move_cmd: QueuedCommand) -> QueuedCommand:
        """解决转向与移动冲突"""
        # 停止命令解决冲突，然后执行转向
        if move_cmd.priority.value <= CommandPriority.HIGH.value:
            return turn_cmd  # 转向优先
        return move_cmd  # 移动优先

    def _resolve_stop_conflict(self, stop_cmd: QueuedCommand, other_cmd: QueuedCommand) -> QueuedCommand:
        """解决停止命令冲突"""
        # 停止命令总是优先
        return stop_cmd


class CommandBatcher:
    """命令批处理器"""

    def __init__(self, batch_timeout: float = 0.1, max_batch_size: int = 10):
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size
        self.batch_window: List[QueuedCommand] = []
        self.last_batch_time = time.time()

    def add_command(self, command: QueuedCommand) -> Optional[List[QueuedCommand]]:
        """添加命令到批处理窗口，返回就绪的批次"""
        self.batch_window.append(command)

        # 检查是否应该触发批处理
        current_time = time.time()
        time_elapsed = current_time - self.last_batch_time

        if (len(self.batch_window) >= self.max_batch_size or
            time_elapsed >= self.batch_timeout or
            command.priority.value <= CommandPriority.HIGH.value):

            batch = self.batch_window.copy()
            self.batch_window.clear()
            self.last_batch_time = current_time

            return self._optimize_batch(batch)

        return None

    def _optimize_batch(self, batch: List[QueuedCommand]) -> List[QueuedCommand]:
        """优化批次中的命令"""
        if len(batch) <= 1:
            return batch

        # 按优先级排序
        batch.sort(key=lambda x: (x.priority.value, x.timestamp))

        # 移除冗余命令
        optimized = []
        for cmd in batch:
            should_keep = True

            # 检查与已优化命令的冲突
            for existing_cmd in optimized:
                if self._commands_conflict(cmd, existing_cmd):
                    # 保留优先级更高的
                    if cmd.priority.value < existing_cmd.priority.value:
                        optimized.remove(existing_cmd)
                    else:
                        should_keep = False
                        break

            if should_keep:
                optimized.append(cmd)

        return optimized

    def _commands_conflict(self, cmd1: QueuedCommand, cmd2: QueuedCommand) -> bool:
        """检查两个命令是否冲突"""
        # 相同类型的连续命令可能冲突
        if cmd1.command_type == cmd2.command_type:
            if cmd1.command_type in [CommandType.MOVE, CommandType.TURN]:
                return True

        # 停止命令与任何运动命令冲突
        if cmd1.command_type == CommandType.STOP and cmd2.command_type in [CommandType.MOVE, CommandType.TURN]:
            return True

        return False


class RateLimiter:
    """命令频率限制器"""

    def __init__(self, max_commands_per_second: float = 20.0):
        self.max_rate = max_commands_per_second
        self.min_interval = 1.0 / max_commands_per_second
        self.last_command_time = 0.0
        self.command_count = 0
        self.window_start = time.time()
        self.window_duration = 1.0

    def can_send_command(self) -> bool:
        """检查是否可以发送命令"""
        current_time = time.time()

        # 检查最小间隔
        if current_time - self.last_command_time < self.min_interval:
            return False

        # 检查窗口内命令数量
        if current_time - self.window_start > self.window_duration:
            # 重置窗口
            self.window_start = current_time
            self.command_count = 0
        elif self.command_count >= self.max_rate:
            return False

        return True

    def record_command(self):
        """记录命令发送"""
        current_time = time.time()
        self.last_command_time = current_time
        self.command_count += 1


class CommandQueue:
    """智能命令队列"""

    def __init__(
        self,
        max_size: int = 1000,
        batch_timeout: float = 0.1,
        max_rate: float = 20.0,
        enable_batching: bool = True
    ):
        self.max_size = max_size
        self.enable_batching = enable_batching

        # 优先级队列
        self._priority_queue: List[QueuedCommand] = []
        self._queue_lock = asyncio.Lock()

        # 执行状态
        self._executing = False
        self._execution_task: Optional[asyncio.Task] = None

        # 冲突解决器
        self.conflict_resolver = CommandConflictResolver()

        # 批处理器
        if enable_batching:
            self.batcher = CommandBatcher(batch_timeout=batch_timeout)

        # 频率限制器
        self.rate_limiter = RateLimiter(max_commands_per_second=max_rate)

        # 统计信息
        self.stats = {
            "commands_queued": 0,
            "commands_executed": 0,
            "commands_failed": 0,
            "commands_dropped": 0,
            "conflicts_resolved": 0,
            "batches_processed": 0,
            "average_queue_size": 0.0,
            "average_execution_time": 0.0
        }

        # 回调函数
        self.execution_callbacks: List[Callable[[CommandResult], None]] = []

        # 队列大小历史
        self._size_history: deque = deque(maxlen=100)

        logger.info("命令队列初始化完成")

    async def start(self):
        """启动命令队列执行"""
        if self._executing:
            return

        self._executing = True
        self._execution_task = asyncio.create_task(self._execution_loop())
        logger.info("命令队列执行已启动")

    async def stop(self):
        """停止命令队列执行"""
        self._executing = False

        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass

        logger.info("命令队列执行已停止")

    async def enqueue_command(
        self,
        command: Any,
        command_type: CommandType = CommandType.CUSTOM,
        priority: CommandPriority = CommandPriority.NORMAL,
        timeout: float = 5.0,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """入队命令"""
        command_id = f"cmd_{int(time.time() * 1000000)}"

        queued_cmd = QueuedCommand(
            id=command_id,
            command_type=command_type,
            priority=priority,
            command=command,
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {}
        )

        async with self._queue_lock:
            # 检查队列大小
            if len(self._priority_queue) >= self.max_size:
                # 移除最低优先级的命令
                self._priority_queue.sort(reverse=True)
                removed = self._priority_queue.pop()
                logger.warning(f"队列已满，丢弃低优先级命令: {removed.id}")
                self.stats["commands_dropped"] += 1

            # 检查冲突并解决
            conflicts = self._find_conflicts(queued_cmd)
            if conflicts:
                resolved = self._resolve_conflicts(queued_cmd, conflicts)
                if resolved is None:
                    logger.debug(f"命令冲突被丢弃: {command_id}")
                    self.stats["commands_dropped"] += 1
                    return command_id
                queued_cmd = resolved

            heapq.heappush(self._priority_queue, queued_cmd)
            self.stats["commands_queued"] += 1

            # 更新统计
            self._update_size_stats()

        logger.debug(f"命令入队: {command_id} ({command_type.value}, {priority.name})")
        return command_id

    def _find_conflicts(self, new_cmd: QueuedCommand) -> List[QueuedCommand]:
        """查找与新技术冲突的命令"""
        conflicts = []
        for cmd in self._priority_queue:
            if self.conflict_resolver.resolve_conflict(new_cmd, cmd) != cmd:
                conflicts.append(cmd)
        return conflicts

    def _resolve_conflicts(self, new_cmd: QueuedCommand, conflicts: List[QueuedCommand]) -> Optional[QueuedCommand]:
        """解决命令冲突"""
        if not conflicts:
            return new_cmd

        # 移除冲突的命令
        for conflict_cmd in conflicts:
            if conflict_cmd in self._priority_queue:
                self._priority_queue.remove(conflict_cmd)
                heapq.heapify(self._priority_queue)
                self.stats["conflicts_resolved"] += 1

        # 解决冲突
        resolved_cmd = new_cmd
        for conflict_cmd in conflicts:
            resolved_cmd = self.conflict_resolver.resolve_conflict(resolved_cmd, conflict_cmd)

        return resolved_cmd

    async def _execution_loop(self):
        """命令执行循环"""
        logger.info("命令执行循环开始")

        while self._executing:
            try:
                # 获取要执行的命令
                commands_to_execute = await self._get_commands_to_execute()

                if commands_to_execute:
                    await self._execute_commands(commands_to_execute)
                else:
                    # 没有命令时短暂休眠
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"命令执行循环错误: {e}")
                await asyncio.sleep(0.1)

        logger.info("命令执行循环结束")

    async def _get_commands_to_execute(self) -> List[QueuedCommand]:
        """获取要执行的命令"""
        commands = []

        async with self._queue_lock:
            # 检查优先级队列
            while self._priority_queue:
                cmd = heapq.heappop(self._priority_queue)

                # 检查命令是否有效
                if cmd.is_expired() or cmd.is_timed_out():
                    logger.debug(f"命令过期/超时，丢弃: {cmd.id}")
                    self.stats["commands_dropped"] += 1
                    continue

                commands.append(cmd)

                # 频率限制
                if not self.rate_limiter.can_send_command():
                    # 将命令放回队列
                    heapq.heappush(self._priority_queue, cmd)
                    break

                # 批处理优化
                if self.enable_batching:
                    batch = self.batcher.add_command(cmd)
                    if batch:
                        commands.extend(batch[1:])  # 第一个命令已经在commands中
                        self.stats["batches_processed"] += 1
                        break

                self.rate_limiter.record_command()

        return commands

    async def _execute_commands(self, commands: List[QueuedCommand]):
        """执行命令列表"""
        for cmd in commands:
            try:
                start_time = time.time()
                success = await self._execute_single_command(cmd)
                execution_time = time.time() - start_time

                result = CommandResult(
                    command_id=cmd.id,
                    success=success,
                    execution_time=execution_time
                )

                if success:
                    self.stats["commands_executed"] += 1
                    logger.debug(f"命令执行成功: {cmd.id}")
                else:
                    self.stats["commands_failed"] += 1

                # 更新平均执行时间
                self.stats["average_execution_time"] = (
                    (self.stats["average_execution_time"] * (self.stats["commands_executed"] + self.stats["commands_failed"] - 1) +
                     execution_time) / (self.stats["commands_executed"] + self.stats["commands_failed"])
                )

                # 触发回调
                for callback in self.execution_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"命令执行回调错误: {e}")

            except Exception as e:
                logger.error(f"命令执行异常 {cmd.id}: {e}")
                self.stats["commands_failed"] += 1

            # 更新队列大小统计
            self._update_size_stats()

    async def _execute_single_command(self, cmd: QueuedCommand) -> bool:
        """执行单个命令"""
        try:
            # 根据命令类型执行相应的操作
            if isinstance(cmd.command, TwistCommand):
                # 这里应该注入实际的ROS2接口
                # await self.ros2_interface.publish_twist(cmd.command)
                logger.debug(f"执行Twist命令: {cmd.command.to_dict()}")
                return True
            else:
                # 自定义命令处理
                logger.debug(f"执行自定义命令: {cmd.command}")
                return True

        except Exception as e:
            logger.error(f"命令执行失败 {cmd.id}: {e}")

            # 重试机制
            if cmd.can_retry():
                cmd.retry_count += 1
                cmd.timestamp = datetime.now()  # 重置时间戳
                heapq.heappush(self._priority_queue, cmd)
                logger.info(f"命令重试 {cmd.id}: 第{cmd.retry_count}次")
                return False

            return False

    def _update_size_stats(self):
        """更新队列大小统计"""
        current_size = len(self._priority_queue)
        self._size_history.append(current_size)

        # 计算平均队列大小
        if self._size_history:
            self.stats["average_queue_size"] = sum(self._size_history) / len(self._size_history)

    def add_execution_callback(self, callback: Callable[[CommandResult], None]):
        """添加命令执行回调"""
        self.execution_callbacks.append(callback)

    def remove_execution_callback(self, callback: Callable[[CommandResult], None]):
        """移除命令执行回调"""
        if callback in self.execution_callbacks:
            self.execution_callbacks.remove(callback)

    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        return {
            **self.stats,
            "current_queue_size": len(self._priority_queue),
            "executing": self._executing,
            "rate_limiter_stats": {
                "current_rate": self.rate_limiter.command_count / max(1, time.time() - self.rate_limiter.window_start),
                "max_rate": self.rate_limiter.max_rate
            }
        }

    async def clear_queue(self) -> int:
        """清空队列"""
        async with self._queue_lock:
            count = len(self._priority_queue)
            self._priority_queue.clear()
            logger.info(f"清空命令队列: 移除 {count} 个命令")
            return count

    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        async with self._queue_lock:
            # 按优先级统计
            priority_counts = defaultdict(int)
            type_counts = defaultdict(int)

            for cmd in self._priority_queue:
                priority_counts[cmd.priority.name] += 1
                type_counts[cmd.command_type.value] += 1

            return {
                "total_commands": len(self._priority_queue),
                "by_priority": dict(priority_counts),
                "by_type": dict(type_counts),
                "oldest_command": min((cmd.timestamp for cmd in self._priority_queue), default=None),
                "newest_command": max((cmd.timestamp for cmd in self._priority_queue), default=None)
            }

    async def enqueue_emergency_stop(self) -> str:
        """入队紧急停止命令"""
        return await self.enqueue_command(
            command=TwistCommand.stop(),
            command_type=CommandType.STOP,
            priority=CommandPriority.EMERGENCY,
            timeout=1.0,
            max_retries=5
        )

    async def enqueue_movement(
        self,
        linear_x: float = 0.0,
        angular_z: float = 0.0,
        priority: CommandPriority = CommandPriority.NORMAL
    ) -> str:
        """入队运动命令"""
        return await self.enqueue_command(
            command=TwistCommand(linear_x=linear_x, angular_z=angular_z),
            command_type=CommandType.MOVE,
            priority=priority
        )

    async def enqueue_turn(
        self,
        angular_z: float,
        linear_x: float = 0.0,
        priority: CommandPriority = CommandPriority.HIGH
    ) -> str:
        """入队转向命令"""
        return await self.enqueue_command(
            command=TwistCommand(linear_x=linear_x, angular_z=angular_z),
            command_type=CommandType.TURN,
            priority=priority
        )