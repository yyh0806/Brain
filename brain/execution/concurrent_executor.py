"""
Concurrent Execution Framework for Brain System

Provides high-performance concurrent operation execution with:
- Parallel operation execution
- Resource-aware scheduling
- Dependency management
- Load balancing
- Circuit breaker patterns
- Performance monitoring
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from loguru import logger

from .executor import Executor, ExecutionContext, ExecutionMode, ExecutionMetrics
from brain.execution.operations.base import Operation, OperationResult, OperationStatus


class OperationDependency(Enum):
    """操作依赖类型"""
    SEQUENTIAL = "sequential"      # 顺序依赖
    PARALLEL = "parallel"          # 并行执行
    MUTUAL_EXCLUSIVE = "mutex"     # 互斥执行
    RESOURCE_SHARING = "share"     # 资源共享


class ResourceLock:
    """资源锁"""

    def __init__(self, resource_id: str, max_concurrent: int = 1):
        self.resource_id = resource_id
        self.max_concurrent = max_concurrent
        self.current_users: Set[str] = set()
        self.waiting_queue: asyncio.Queue = asyncio.Queue()
        self._lock = asyncio.Lock()

    async def acquire(self, operation_id: str, timeout: Optional[float] = None) -> bool:
        """获取资源锁"""
        async with self._lock:
            if len(self.current_users) < self.max_concurrent:
                self.current_users.add(operation_id)
                return True

        # 等待资源释放
        try:
            future = asyncio.Future()
            await self.waiting_queue.put((operation_id, future))
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            # 从等待队列中移除
            self._remove_from_queue(operation_id)
            return False

    async def release(self, operation_id: str):
        """释放资源锁"""
        async with self._lock:
            if operation_id in self.current_users:
                self.current_users.remove(operation_id)

                # 唤醒等待的操作
                if not self.waiting_queue.empty():
                    next_op_id, future = await self.waiting_queue.get()
                    self.current_users.add(next_op_id)
                    future.set_result(True)

    def _remove_from_queue(self, operation_id: str):
        """从等待队列中移除操作"""
        # 这是一个简化的实现，实际中可能需要更复杂的逻辑
        pass

    def get_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return {
            "resource_id": self.resource_id,
            "max_concurrent": self.max_concurrent,
            "current_users": len(self.current_users),
            "waiting_queue_size": self.waiting_queue.qsize()
        }


@dataclass
class ConcurrentExecutionTask:
    """并发执行任务"""
    operation: Operation
    operation_id: str
    dependencies: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[OperationResult] = None
    retry_count: int = 0
    max_retries: int = 3

    @property
    def is_pending(self) -> bool:
        return self.started_at is None

    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.completed_at is None

    @property
    def is_completed(self) -> bool:
        return self.completed_at is not None

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries


class DependencyResolver:
    """依赖关系解析器"""

    def __init__(self):
        self.dependencies: Dict[str, List[str]] = {}
        self.reverse_dependencies: Dict[str, List[str]] = {}

    def add_dependency(self, operation_id: str, depends_on: str):
        """添加依赖关系"""
        if operation_id not in self.dependencies:
            self.dependencies[operation_id] = []
        self.dependencies[operation_id].append(depends_on)

        if depends_on not in self.reverse_dependencies:
            self.reverse_dependencies[depends_on] = []
        self.reverse_dependencies[depends_on].append(operation_id)

    def get_ready_operations(self, pending_tasks: Set[str]) -> List[str]:
        """获取可以执行的操作（没有未完成的依赖）"""
        ready_ops = []

        for op_id in pending_tasks:
            deps = self.dependencies.get(op_id, [])
            if not deps or all(dep not in pending_tasks for dep in deps):
                ready_ops.append(op_id)

        return ready_ops

    def has_circular_dependency(self) -> bool:
        """检测循环依赖"""
        visited = set()
        rec_stack = set()

        def has_cycle(op_id: str) -> bool:
            visited.add(op_id)
            rec_stack.add(op_id)

            for dep in self.dependencies.get(op_id, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(op_id)
            return False

        return any(has_cycle(op_id) for op_id in self.dependencies.keys())


class LoadBalancer:
    """负载均衡器"""

    def __init__(self, max_concurrent_operations: int = 10):
        self.max_concurrent = max_concurrent_operations
        self.worker_loads: Dict[str, int] = {}
        self.worker_tasks: Dict[str, Set[str]] = {}

    def assign_task(self, operation_id: str, task_type: str = "default") -> str:
        """分配任务到工作器"""
        # 选择负载最低的工作器
        min_load = float('inf')
        selected_worker = f"worker_0"

        for i in range(self.max_concurrent_operations):
            worker_id = f"worker_{i}"
            load = self.worker_loads.get(worker_id, 0)
            if load < min_load:
                min_load = load
                selected_worker = worker_id

        # 分配任务
        self.worker_loads[selected_worker] = self.worker_loads.get(selected_worker, 0) + 1
        if selected_worker not in self.worker_tasks:
            self.worker_tasks[selected_worker] = set()
        self.worker_tasks[selected_worker].add(operation_id)

        return selected_worker

    def release_task(self, worker_id: str, operation_id: str):
        """释放任务"""
        if worker_id in self.worker_loads:
            self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - 1)

        if worker_id in self.worker_tasks:
            self.worker_tasks[worker_id].discard(operation_id)

    def get_load_stats(self) -> Dict[str, Any]:
        """获取负载统计"""
        total_load = sum(self.worker_loads.values())
        return {
            "max_concurrent": self.max_concurrent_operations,
            "current_load": total_load,
            "utilization": total_load / self.max_concurrent_operations,
            "worker_loads": self.worker_loads.copy()
        }


class CircuitBreaker:
    """熔断器"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """检查是否可以执行"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("熔断器状态: HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("熔断器状态: CLOSED")

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"熔断器触发: {self.failure_count} 次失败")

    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time
        }


class ConcurrentExecutor:
    """并发执行器"""

    def __init__(
        self,
        base_executor: Executor,
        max_concurrent_operations: int = 10,
        enable_circuit_breaker: bool = True,
        enable_load_balancing: bool = True
    ):
        self.base_executor = base_executor
        self.max_concurrent_operations = max_concurrent_operations

        # 任务管理
        self.pending_tasks: Dict[str, ConcurrentExecutionTask] = {}
        self.running_tasks: Dict[str, ConcurrentExecutionTask] = {}
        self.completed_tasks: Dict[str, ConcurrentExecutionTask] = {}

        # 资源管理
        self.resource_locks: Dict[str, ResourceLock] = {}
        self._resource_lock = asyncio.Lock()

        # 依赖管理
        self.dependency_resolver = DependencyResolver()

        # 负载均衡
        self.load_balancer = LoadBalancer(max_concurrent_operations) if enable_load_balancing else None

        # 熔断器
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None

        # 执行控制
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_operations)
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

        # 统计信息
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "parallel_efficiency": 0.0,
            "average_execution_time": 0.0,
            "resource_contentions": 0
        }

        logger.info("并发执行器初始化完成")

    async def start(self):
        """启动并发执行器"""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("并发执行器已启动")

    async def stop(self):
        """停止并发执行器"""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # 等待所有任务完成
        while self.running_tasks:
            await asyncio.sleep(0.1)

        logger.info("并发执行器已停止")

    async def submit_operation(
        self,
        operation: Operation,
        dependencies: Optional[List[str]] = None,
        required_resources: Optional[List[str]] = None,
        priority: int = 0
    ) -> str:
        """提交操作到并发执行器"""
        operation_id = f"op_{int(time.time() * 1000000)}_{operation.id}"

        task = ConcurrentExecutionTask(
            operation=operation,
            operation_id=operation_id,
            dependencies=dependencies or [],
            required_resources=required_resources or [],
            priority=priority
        )

        # 添加依赖关系
        for dep_id in task.dependencies:
            self.dependency_resolver.add_dependency(operation_id, dep_id)

        # 检查循环依赖
        if self.dependency_resolver.has_circular_dependency():
            raise ValueError(f"检测到循环依赖: {operation_id}")

        self.pending_tasks[operation_id] = task
        self.metrics["total_tasks"] += 1

        logger.debug(f"操作已提交: {operation_id}")
        return operation_id

    async def _scheduler_loop(self):
        """调度循环"""
        logger.info("调度循环开始")

        while self._running:
            try:
                # 获取可执行的操作
                ready_operations = self._get_ready_operations()

                # 按优先级排序
                ready_operations.sort(
                    key=lambda op_id: (
                        self.pending_tasks[op_id].priority,
                        self.pending_tasks[op_id].created_at
                    ),
                    reverse=True
                )

                # 尝试执行就绪的操作
                for op_id in ready_operations:
                    if await self._try_execute_operation(op_id):
                        # 从待执行队列移除
                        task = self.pending_tasks.pop(op_id)
                        self.running_tasks[op_id] = task

                # 清理已完成的任务
                self._cleanup_completed_tasks()

                # 短暂休眠
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度循环错误: {e}")
                await asyncio.sleep(0.1)

        logger.info("调度循环结束")

    def _get_ready_operations(self) -> List[str]:
        """获取可执行的操作"""
        pending_op_ids = set(self.pending_tasks.keys())

        # 获取没有未完成依赖的操作
        ready_ops = self.dependency_resolver.get_ready_operations(pending_op_ids)

        # 检查资源可用性
        ready_with_resources = []
        for op_id in ready_ops:
            task = self.pending_tasks[op_id]
            if self._are_resources_available(task.required_resources):
                ready_with_resources.append(op_id)

        return ready_with_resources

    def _are_resources_available(self, required_resources: List[str]) -> bool:
        """检查资源是否可用"""
        for resource_id in required_resources:
            if resource_id not in self.resource_locks:
                # 创建资源锁
                max_concurrent = 1  # 默认互斥
                self.resource_locks[resource_id] = ResourceLock(resource_id, max_concurrent)

            lock = self.resource_locks[resource_id]
            if len(lock.current_users) >= lock.max_concurrent:
                return False

        return True

    async def _try_execute_operation(self, operation_id: str) -> bool:
        """尝试执行操作"""
        task = self.pending_tasks[operation_id]

        # 检查熔断器
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning(f"熔断器阻止执行: {operation_id}")
            return False

        # 获取信号量
        if not self._execution_semaphore.locked():
            try:
                await asyncio.wait_for(
                    self._execution_semaphore.acquire(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                return False

        # 分配工作器
        worker_id = "default"
        if self.load_balancer:
            worker_id = self.load_balancer.assign_task(operation_id, task.operation.name)

        try:
            # 获取资源锁
            acquired_resources = []
            for resource_id in task.required_resources:
                lock = self.resource_locks[resource_id]
                if await lock.acquire(operation_id, timeout=0.1):
                    acquired_resources.append(resource_id)
                else:
                    # 资源获取失败，释放已获取的资源
                    for acquired_id in acquired_resources:
                        await self.resource_locks[acquired_id].release(operation_id)
                    self._execution_semaphore.release()
                    if self.load_balancer:
                        self.load_balancer.release_task(worker_id, operation_id)
                    self.metrics["resource_contentions"] += 1
                    return False

            # 启动执行任务
            asyncio.create_task(
                self._execute_task_with_resources(
                    task, worker_id, acquired_resources
                )
            )

            return True

        except Exception as e:
            logger.error(f"执行操作准备失败 {operation_id}: {e}")
            self._execution_semaphore.release()
            if self.load_balancer:
                self.load_balancer.release_task(worker_id, operation_id)
            return False

    async def _execute_task_with_resources(
        self,
        task: ConcurrentExecutionTask,
        worker_id: str,
        acquired_resources: List[str]
    ):
        """执行任务（带资源管理）"""
        operation_id = task.operation_id
        task.started_at = datetime.now()

        try:
            logger.debug(f"开始执行操作: {operation_id} (工作器: {worker_id})")

            # 实际执行操作
            result = await self.base_executor.execute(
                operation=task.operation,
                robot_interface=None,  # 这里需要传入实际的robot_interface
                timeout=task.operation.estimated_duration * 2
            )

            task.result = result
            task.completed_at = datetime.now()

            # 更新指标
            if result.status == OperationStatus.SUCCESS:
                self.metrics["completed_tasks"] += 1
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
            else:
                self.metrics["failed_tasks"] += 1
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()

            # 计算执行时间
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self._update_average_execution_time(execution_time)

            logger.debug(f"操作执行完成: {operation_id}, 状态: {result.status.value}")

        except Exception as e:
            logger.error(f"操作执行异常 {operation_id}: {e}")
            task.result = OperationResult(
                status=OperationStatus.FAILED,
                error_message=str(e)
            )
            task.completed_at = datetime.now()
            self.metrics["failed_tasks"] += 1

        finally:
            # 释放资源
            for resource_id in acquired_resources:
                await self.resource_locks[resource_id].release(operation_id)

            # 释放信号量和工作器
            self._execution_semaphore.release()
            if self.load_balancer:
                self.load_balancer.release_task(worker_id, operation_id)

            # 移动到已完成队列
            if operation_id in self.running_tasks:
                completed_task = self.running_tasks.pop(operation_id)
                self.completed_tasks[operation_id] = completed_task

    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        # 保留最近1000个完成的任务
        max_completed = 1000
        if len(self.completed_tasks) > max_completed:
            # 按完成时间排序，删除最旧的
            sorted_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].completed_at or datetime.min
            )
            to_remove = sorted_tasks[:-max_completed]
            for task_id, _ in to_remove:
                del self.completed_tasks[task_id]

    def _update_average_execution_time(self, execution_time: float):
        """更新平均执行时间"""
        total_completed = self.metrics["completed_tasks"] + self.metrics["failed_tasks"]
        if total_completed > 0:
            current_avg = self.metrics["average_execution_time"]
            self.metrics["average_execution_time"] = (
                (current_avg * (total_completed - 1) + execution_time) / total_completed
            )

        # 计算并行效率
        self.metrics["parallel_efficiency"] = min(1.0, len(self.running_tasks) / self.max_concurrent_operations)

    async def wait_for_completion(self, operation_ids: List[str], timeout: Optional[float] = None) -> Dict[str, OperationResult]:
        """等待操作完成"""
        if not operation_ids:
            return {}

        start_time = time.time()
        results = {}

        while operation_ids:
            # 检查已完成的操作
            for op_id in operation_ids.copy():
                if op_id in self.completed_tasks:
                    task = self.completed_tasks[op_id]
                    results[op_id] = task.result
                    operation_ids.remove(op_id)

            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                break

            await asyncio.sleep(0.1)

        return results

    def get_task_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if operation_id in self.pending_tasks:
            task = self.pending_tasks[operation_id]
            return {
                "status": "pending",
                "operation_id": operation_id,
                "operation_name": task.operation.name,
                "created_at": task.created_at,
                "dependencies": task.dependencies,
                "required_resources": task.required_resources,
                "priority": task.priority
            }
        elif operation_id in self.running_tasks:
            task = self.running_tasks[operation_id]
            return {
                "status": "running",
                "operation_id": operation_id,
                "operation_name": task.operation.name,
                "started_at": task.started_at,
                "execution_time": (datetime.now() - task.started_at).total_seconds()
            }
        elif operation_id in self.completed_tasks:
            task = self.completed_tasks[operation_id]
            return {
                "status": "completed",
                "operation_id": operation_id,
                "operation_name": task.operation.name,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "execution_time": (task.completed_at - task.started_at).total_seconds(),
                "result_status": task.result.status.value if task.result else None,
                "retry_count": task.retry_count
            }
        else:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        metrics = self.metrics.copy()

        # 添加当前状态
        metrics.update({
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "total_resources": len(self.resource_locks),
            "circuit_breaker": self.circuit_breaker.get_state() if self.circuit_breaker else None,
            "load_balancer": self.load_balancer.get_load_stats() if self.load_balancer else None
        })

        return metrics

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        return {
            resource_id: lock.get_usage()
            for resource_id, lock in self.resource_locks.items()
        }

    async def cancel_operation(self, operation_id: str) -> bool:
        """取消操作"""
        if operation_id in self.pending_tasks:
            # 从待执行队列移除
            task = self.pending_tasks.pop(operation_id)
            logger.info(f"取消待执行操作: {operation_id}")
            return True
        elif operation_id in self.running_tasks:
            # 正在运行的操作无法直接取消，但可以标记
            logger.warning(f"操作正在运行中，无法直接取消: {operation_id}")
            return False

        return False

    def clear_completed_tasks(self) -> int:
        """清空已完成的任务"""
        count = len(self.completed_tasks)
        self.completed_tasks.clear()
        logger.info(f"清空已完成的任务: {count} 个")
        return count