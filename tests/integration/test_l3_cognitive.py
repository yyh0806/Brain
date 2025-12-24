#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L3: 认知层验证测试

验证认知层的世界模型更新、变化检测、信念修正、CoT推理和对话管理功能
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.isaac_sim_test_framework import IsaacSimTestFramework, TestResult


class L3CognitiveTests:
    """L3 认知层验证测试"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/environments/isaac_sim/nova_carter.yaml"
        self.framework = IsaacSimTestFramework(self.config_path)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有L3测试"""
        print("=" * 70)
        print("L3: 认知层验证测试")
        print("=" * 70)
        
        try:
            await self.framework.setup()
            await asyncio.sleep(3.0)
            
            # 运行各项测试
            await self.framework.run_test("L3-001: 世界模型更新", self.framework.verify_world_model_update)
            await self.framework.run_test("L3-002: 环境变化检测", self.framework.verify_environment_change_detection)
            await self.framework.run_test("L3-003: 信念修正", self.framework.verify_belief_revision)
            await self.framework.run_test("L3-004: CoT推理", self.framework.verify_cot_reasoning)
            await self.framework.run_test("L3-005: 对话管理", self.framework.verify_dialogue_management)
            
            summary = self.framework.get_test_summary()
            
            print("\n" + "=" * 70)
            print("测试摘要")
            print("=" * 70)
            print(f"总测试数: {summary['total_tests']}")
            print(f"通过: {summary['passed']}")
            print(f"失败: {summary['failed']}")
            print(f"成功率: {summary['success_rate']:.1f}%")
            print("=" * 70)
            
            return summary
        except Exception as e:
            logger.error(f"L3测试套件异常: {e}")
            return {"error": str(e)}
        finally:
            await self.framework.cleanup()


async def main():
    tests = L3CognitiveTests()
    summary = await tests.run_all_tests()
    sys.exit(0 if 'error' not in summary else 1)


if __name__ == "__main__":
    asyncio.run(main())




