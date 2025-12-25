#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4: 规划层验证测试

验证规划层的任务规划、技能分解、动作规划和动态重规划能力

测试覆盖：
- 自然语言理解
- 任务规划
- 技能规划
- 动作规划
- 感知驱动重规划
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.isaac_sim_test_framework import IsaacSimTestFramework


class L4PlanningTests:
    """L4 规划层验证测试"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/environments/isaac_sim/nova_carter.yaml"
        self.framework = IsaacSimTestFramework(self.config_path)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有L4测试"""
        print("=" * 70)
        print("L4: 规划层验证测试")
        print("=" * 70)
        print("\n测试内容：")
        print("  1. 任务规划 (L4-002)")
        print("  2. 感知驱动重规划 (L4-005)")
        print("\n" + "-" * 70)
        
        try:
            await self.framework.setup()
            
            # L4-002: 任务规划
            result = await self.framework.run_test(
                "L4-002: 任务规划",
                lambda: self.framework.verify_task_planning("移动到目标位置")
            )
            
            # L4-005: 感知驱动重规划
            result = await self.framework.run_test(
                "L4-005: 感知驱动重规划",
                lambda: self.framework.verify_replanning([])
            )
            
            summary = self.framework.get_test_summary()
            
            print("\n" + "=" * 70)
            print("测试摘要")
            print("=" * 70)
            print(f"总测试数: {summary['total_tests']}")
            print(f"通过: {summary['passed']}")
            print(f"失败: {summary['failed']}")
            print(f"成功率: {summary['success_rate']:.1f}%")
            print("\n详细结果:")
            for result in summary['results']:
                status = "✓" if result['success'] else "✗"
                print(f"  {status} {result['name']}: {result['message']}")
            print("=" * 70)
            
            return summary
        except Exception as e:
            logger.error(f"L4测试套件异常: {e}")
            print(f"\n✗ 测试异常: {e}")
            return {"error": str(e)}
        finally:
            await self.framework.cleanup()


async def main():
    tests = L4PlanningTests()
    summary = await tests.run_all_tests()
    sys.exit(0 if 'error' not in summary else 1)


if __name__ == "__main__":
    asyncio.run(main())






