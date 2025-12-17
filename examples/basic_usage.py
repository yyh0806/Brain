"""
Brain 系统基础使用示例

演示如何使用Brain系统执行无人机任务
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from loguru import logger

# 创建日志目录
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

# 配置日志
logger.add(str(log_dir / "brain_{time}.log"), rotation="1 day", level="INFO")


async def main():
    """主函数"""
    # 导入Brain系统
    from brain.core.brain import Brain
    
    # 创建Brain实例
    logger.info("初始化 Brain 系统...")
    brain = Brain()
    
    # 连接机器人
    logger.info("连接机器人...")
    await brain.robot_interface.connect()
    
    # 启动系统监控
    await brain.monitor.start()
    
    try:
        # 示例1: 简单巡逻任务
        logger.info("=" * 50)
        logger.info("示例1: 处理自然语言指令")
        logger.info("=" * 50)
        
        # 发送自然语言指令
        command = "起飞到30米高度，然后飞到东边200米处的位置进行区域扫描，完成后返回起飞点降落"
        
        mission = await brain.process_command(
            command=command,
            platform_type="drone",
            context={"mission_name": "巡逻任务"}
        )
        
        logger.info(f"任务已创建: ID={mission.id}")
        logger.info(f"解析出的操作数量: {len(mission.operations)}")
        
        # 打印操作序列
        for i, op in enumerate(mission.operations):
            logger.info(f"  操作 {i+1}: {op.name} - {op.parameters}")
        
        # 执行任务
        logger.info("开始执行任务...")
        final_status = await brain.execute_mission(
            mission_id=mission.id,
            auto_recovery=True
        )
        
        logger.info(f"任务执行完成, 状态: {final_status.value}")
        
        # 示例2: 监控任务
        logger.info("=" * 50)
        logger.info("示例2: 监控任务")
        logger.info("=" * 50)
        
        command2 = "在指定区域悬停5分钟进行监控录像"
        
        mission2 = await brain.process_command(
            command=command2,
            platform_type="drone"
        )
        
        logger.info(f"监控任务已创建: {mission2.id}")
        
        # 获取系统状态
        status = brain.get_status()
        logger.info(f"系统状态: {status}")
        
    except Exception as e:
        logger.error(f"执行异常: {e}")
        
    finally:
        # 关闭系统
        await brain.monitor.stop()
        await brain.shutdown()
        logger.info("Brain 系统已关闭")


async def example_with_error_recovery():
    """演示错误恢复功能"""
    from brain.core.brain import Brain
    
    brain = Brain()
    await brain.robot_interface.connect()
    
    try:
        # 创建一个可能失败的任务
        command = "飞到远距离目标点进行检查"
        
        mission = await brain.process_command(
            command=command,
            platform_type="drone"
        )
        
        # 注册事件回调
        @brain.on("mission_completed")
        def on_complete(mission):
            logger.info(f"任务 {mission.id} 完成!")
        
        # 执行（启用自动恢复）
        result = await brain.execute_mission(
            mission_id=mission.id,
            auto_recovery=True  # 启用自动错误恢复
        )
        
        logger.info(f"执行结果: {result}")
        
    finally:
        await brain.shutdown()


async def example_ugv_task():
    """无人车任务示例"""
    from brain.core.brain import Brain
    
    brain = Brain()
    await brain.robot_interface.connect()
    
    try:
        # 无人车配送任务
        command = "从仓库出发，前往A点接货，然后送到B点卸货，最后返回仓库"
        
        mission = await brain.process_command(
            command=command,
            platform_type="ugv"  # 指定为无人车
        )
        
        logger.info(f"UGV任务: {mission.id}")
        for op in mission.operations:
            logger.info(f"  {op.name}: {op.parameters}")
        
    finally:
        await brain.shutdown()


async def example_usv_task():
    """无人船任务示例"""
    from brain.core.brain import Brain
    
    brain = Brain()
    await brain.robot_interface.connect()
    
    try:
        # 无人船巡逻任务
        command = "在指定海域进行巡逻，检测水面目标，完成后返回港口"
        
        mission = await brain.process_command(
            command=command,
            platform_type="usv"  # 指定为无人船
        )
        
        logger.info(f"USV任务: {mission.id}")
        for op in mission.operations:
            logger.info(f"  {op.name}: {op.parameters}")
        
    finally:
        await brain.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

