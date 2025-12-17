"""
使用 Ollama 本地大模型示例

本示例展示如何配置 Brain 系统使用 Ollama 运行的本地大模型

前提条件:
1. 安装 Ollama: https://ollama.ai/
2. 拉取模型: ollama pull deepseek-r1:latest
3. 确保 Ollama 正在运行: ollama serve
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from loguru import logger

# 配置日志
logger.remove()  # 移除默认handler
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


async def test_ollama_connection():
    """测试 Ollama 连接"""
    from brain.llm.ollama_client import test_ollama_connection as test_conn
    
    logger.info("=" * 50)
    logger.info("测试 Ollama 连接")
    logger.info("=" * 50)
    
    # 测试连接
    success = await test_conn(
        base_url="http://localhost:11434",
        model="deepseek-r1:latest"
    )
    
    if success:
        logger.info("✓ Ollama 连接正常!")
    else:
        logger.error("✗ Ollama 连接失败!")
        logger.info("请确保:")
        logger.info("  1. Ollama 已安装: curl -fsSL https://ollama.ai/install.sh | sh")
        logger.info("  2. Ollama 正在运行: ollama serve")
        logger.info("  3. 模型已拉取: ollama pull deepseek-r1:latest")
    
    return success


async def test_ollama_chat():
    """测试 Ollama 聊天"""
    from brain.llm.ollama_client import OllamaClient
    from brain.llm.llm_interface import LLMConfig, LLMProvider, LLMMessage
    
    logger.info("=" * 50)
    logger.info("测试 Ollama 聊天")
    logger.info("=" * 50)
    
    # 创建配置
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="deepseek-r1:latest",
        api_base="http://localhost:11434",
        max_tokens=1024,
        temperature=0.1,
        timeout=120.0
    )
    
    # 创建客户端
    client = OllamaClient(config)
    
    # 测试简单对话
    messages = [
        LLMMessage(role="system", content="你是一个无人机任务规划助手。"),
        LLMMessage(role="user", content="请用一句话描述无人机起飞操作。")
    ]
    
    logger.info("发送测试消息...")
    try:
        response = await client.chat(messages)
        logger.info(f"模型: {response.model}")
        logger.info(f"响应: {response.content}")
        logger.info(f"Token使用: {response.usage}")
        return True
    except Exception as e:
        logger.error(f"聊天失败: {e}")
        return False
    finally:
        await client.close()


async def run_brain_with_ollama():
    """使用 Ollama 运行 Brain 系统"""
    from brain.core.brain import Brain
    from brain.utils.config import ConfigManager
    
    logger.info("=" * 50)
    logger.info("使用 Ollama 运行 Brain 系统")
    logger.info("=" * 50)
    
    # 方法1: 使用默认配置文件（已配置为Ollama）
    brain = Brain()
    
    # 方法2: 或者手动指定配置
    # config = ConfigManager()
    # config.set("llm.provider", "ollama")
    # config.set("llm.model", "deepseek-r1:latest")
    # config.set("llm.api_base", "http://localhost:11434")
    # brain = Brain(config_path=None)  # 然后手动设置
    
    await brain.robot_interface.connect()
    
    try:
        # 发送自然语言指令
        command = "起飞到20米高度，然后向前飞行50米"
        
        logger.info(f"处理指令: {command}")
        
        mission = await brain.process_command(
            command=command,
            platform_type="drone"
        )
        
        logger.info(f"任务ID: {mission.id}")
        logger.info(f"生成的操作序列:")
        for i, op in enumerate(mission.operations):
            logger.info(f"  {i+1}. {op.name}: {op.parameters}")
        
        return True
        
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await brain.shutdown()


async def list_available_models():
    """列出可用的 Ollama 模型"""
    from brain.llm.ollama_client import OllamaClient
    from brain.llm.llm_interface import LLMConfig, LLMProvider
    
    logger.info("=" * 50)
    logger.info("列出可用的 Ollama 模型")
    logger.info("=" * 50)
    
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="",
        api_base="http://localhost:11434"
    )
    
    client = OllamaClient(config)
    
    try:
        models = await client.list_models()
        
        if models:
            logger.info("可用模型:")
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024 ** 3)  # 转换为GB
                logger.info(f"  - {name} ({size:.1f} GB)")
        else:
            logger.warning("没有找到可用模型")
            logger.info("请运行: ollama pull deepseek-r1:latest")
            
    finally:
        await client.close()


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("  Brain + Ollama 本地大模型示例  ")
    logger.info("=" * 60)
    
    # 1. 测试连接
    if not await test_ollama_connection():
        return
    
    # 2. 列出模型
    await list_available_models()
    
    # 3. 测试聊天
    await test_ollama_chat()
    
    # 4. 运行Brain系统
    await run_brain_with_ollama()


if __name__ == "__main__":
    asyncio.run(main())


