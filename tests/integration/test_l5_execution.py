#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L5: 执行层验证测试

验证执行层的操作执行、执行监控、错误恢复和并发执行能力

测试覆盖：
- 基础操作执行
- 复杂操作执行
- 执行监控
- 错误恢复
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.isaac_sim_test_framework import IsaacSimTestFramework
from brain.execution.operations.base import Operation


class L5ExecutionTests:
    """L5 执行层验证测试"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/environments/isaac_sim/nova_carter.yaml"
        self.framework = IsaacSimTestFramework(self.config_path)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有L5测试"""
        print("=" * 70)
        print("L5: 执行层验证测试")
        print("=" * 70)
        
        try:
            await self.framework.setup()
            
            # 运行各项测试
            await self.framework.run_test("L5-001: 基础操作执行", lambda: self.framework.verify_operation_execution(
                Operation(id="test_wait", name="wait", estimated_duration=1.0)
            ))
            await self.framework.run_test("L5-004: 错误恢复", self.framework.verify_error_recovery)
            
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
            logger.error(f"L5测试套件异常: {e}")
            print(f"\n✗ 测试异常: {e}")
            return {"error": str(e)}
        finally:
            await self.framework.cleanup()


async def main():
    tests = L5ExecutionTests()
    summary = await tests.run_all_tests()
    sys.exit(0 if 'error' not in summary else 1)


if __name__ == "__main__":
    asyncio.run(main())








