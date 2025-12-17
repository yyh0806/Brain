"""
Brain 系统高级使用示例

演示高级功能：
- 自定义操作处理器
- 检查点和恢复
- 多任务管理
- 实时监控
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from datetime import datetime
from loguru import logger

from brain.core.brain import Brain, BrainStatus
from brain.core.executor import Executor, ExecutionMode
from brain.operations.base import Operation, OperationResult, OperationStatus
from brain.state.mission_state import MissionStatus


async def example_custom_operation_handler():
    """自定义操作处理器示例"""
    from brain.core.brain import Brain
    
    brain = Brain()
    
    # 定义自定义操作处理器
    async def custom_scan_handler(operation: Operation, robot_interface):
        """自定义扫描处理器"""
        logger.info(f"执行自定义扫描: {operation.parameters}")
        
        # 自定义扫描逻辑
        scan_area = operation.parameters.get("area")
        resolution = operation.parameters.get("resolution", "high")
        
        # 执行扫描
        await asyncio.sleep(2)  # 模拟扫描时间
        
        # 返回结果
        return OperationResult(
            status=OperationStatus.SUCCESS,
            data={
                "scanned_area": scan_area,
                "detected_objects": 5,
                "scan_time": 2.0
            }
        )
    
    # 注册自定义处理器
    brain.executor.register_handler("custom_scan", custom_scan_handler)
    
    logger.info("自定义操作处理器已注册")
    
    await brain.shutdown()


async def example_checkpoint_recovery():
    """检查点恢复示例"""
    from brain.core.brain import Brain
    from brain.state.checkpoint import CheckpointManager
    
    brain = Brain()
    await brain.robot_interface.connect()
    
    try:
        # 创建任务
        command = "执行一个多步骤的复杂任务"
        mission = await brain.process_command(
            command=command,
            platform_type="drone"
        )
        
        # 手动创建检查点
        await brain.checkpoint_manager.create_checkpoint(
            mission_id=mission.id,
            stage="before_execution",
            data={
                "world_state": brain.world_state.to_dict(),
                "mission_operations": [op.to_dict() for op in mission.operations]
            }
        )
        
        logger.info("检查点已创建")
        
        # 模拟任务中断
        # 假设在某个点任务失败...
        
        # 获取最新检查点恢复
        checkpoint = await brain.checkpoint_manager.get_latest_checkpoint(mission.id)
        if checkpoint:
            logger.info(f"从检查点恢复: {checkpoint.stage}")
            brain.world_state.restore(checkpoint.data.get("world_state", {}))
            
            # 可以从这里继续执行
            
    finally:
        await brain.shutdown()


async def example_multi_mission():
    """多任务管理示例"""
    from brain.core.brain import Brain
    
    brain = Brain()
    await brain.robot_interface.connect()
    
    try:
        # 创建多个任务
        missions = []
        
        commands = [
            ("巡逻区域A", "drone", 1),
            ("监控目标点B", "drone", 2),
            ("采集水样", "usv", 3)
        ]
        
        for cmd, platform, priority in commands:
            mission = await brain.process_command(
                command=cmd,
                platform_type=platform,
                context={"priority": priority}
            )
            missions.append(mission)
            logger.info(f"任务创建: {mission.id} - {cmd}")
        
        # 按优先级排序执行
        missions.sort(key=lambda m: m.priority)
        
        for mission in missions:
            logger.info(f"执行任务: {mission.id}")
            # await brain.execute_mission(mission.id)
        
        # 查看所有任务状态
        for mid, mission in brain.missions.items():
            status = brain.mission_state.get_mission_status(mid)
            logger.info(f"任务 {mid}: {status}")
            
    finally:
        await brain.shutdown()


async def example_realtime_monitoring():
    """实时监控示例"""
    from brain.core.brain import Brain
    
    brain = Brain()
    await brain.robot_interface.connect()
    await brain.monitor.start()
    
    try:
        # 注册遥测回调
        def on_telemetry(telemetry):
            logger.info(f"遥测: 电池={telemetry.battery}%, 位置=({telemetry.latitude}, {telemetry.longitude})")
        
        brain.robot_interface.on_telemetry(on_telemetry)
        
        # 注册事件回调
        brain.on("mission_completed", lambda m: logger.info(f"任务完成: {m.id}"))
        brain.on("mission_planned", lambda m: logger.info(f"任务规划完成: {m.id}"))
        
        # 创建并执行任务
        mission = await brain.process_command(
            command="执行巡逻任务",
            platform_type="drone"
        )
        
        # 在执行过程中获取状态
        async def monitor_status():
            while brain.status == BrainStatus.EXECUTING:
                status = brain.get_status()
                logger.info(f"系统状态: {status['status']}")
                await asyncio.sleep(1)
        
        # 并行执行任务和监控
        await asyncio.gather(
            brain.execute_mission(mission.id),
            monitor_status()
        )
        
    finally:
        await brain.monitor.stop()
        await brain.shutdown()


async def example_error_handling():
    """错误处理示例"""
    from brain.core.brain import Brain
    from brain.recovery.error_handler import ErrorHandler, RecoveryStrategy
    
    brain = Brain()
    await brain.robot_interface.connect()
    
    try:
        # 创建任务
        mission = await brain.process_command(
            command="执行一个测试任务",
            platform_type="drone"
        )
        
        # 模拟错误发生
        test_operation = mission.operations[0] if mission.operations else None
        
        if test_operation:
            # 分析错误
            analysis = await brain.error_handler.analyze(
                operation=test_operation,
                error="GPS信号丢失，无法定位",
                world_state=brain.world_state
            )
            
            logger.info(f"错误类型: {analysis.error_type.value}")
            logger.info(f"严重程度: {analysis.severity.value}")
            logger.info(f"可恢复: {analysis.recoverable}")
            logger.info(f"推荐策略: {analysis.recommended_strategy.value}")
            logger.info(f"安全注意事项: {analysis.safety_concerns}")
            
            # 根据策略处理
            if analysis.recommended_strategy == RecoveryStrategy.REPLAN:
                logger.info("触发重规划...")
                # 重规划逻辑
            elif analysis.recommended_strategy == RecoveryStrategy.ROLLBACK:
                logger.info("触发回滚...")
                # 回滚逻辑
            elif analysis.recommended_strategy == RecoveryStrategy.EMERGENCY:
                logger.info("触发紧急处理...")
                await brain.emergency_stop()
                
    finally:
        await brain.shutdown()


async def example_dry_run():
    """干跑模式示例 - 不实际执行操作"""
    from brain.core.brain import Brain
    from brain.core.executor import ExecutionMode
    
    brain = Brain()
    
    # 设置为干跑模式
    brain.executor.set_mode(ExecutionMode.DRY_RUN)
    
    await brain.robot_interface.connect()
    
    try:
        # 创建任务
        mission = await brain.process_command(
            command="执行一个复杂的测试任务包括起飞、巡逻和降落",
            platform_type="drone"
        )
        
        logger.info("干跑模式执行任务（不会实际发送指令）...")
        
        # 执行（实际不会发送任何指令）
        result = await brain.execute_mission(mission.id)
        
        logger.info(f"干跑完成: {result.value}")
        
        # 查看执行指标
        metrics = brain.executor.get_metrics()
        logger.info(f"执行指标: {metrics}")
        
    finally:
        await brain.shutdown()


if __name__ == "__main__":
    # 选择要运行的示例
    asyncio.run(example_custom_operation_handler())
    # asyncio.run(example_checkpoint_recovery())
    # asyncio.run(example_multi_mission())
    # asyncio.run(example_realtime_monitoring())
    # asyncio.run(example_error_handling())
    # asyncio.run(example_dry_run())

