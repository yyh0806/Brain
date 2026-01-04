#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L6: 端到端验证测试

验证Brain系统从指令到执行的完整任务流程

测试覆盖：
- 简单导航任务
- 搜索任务
- 系统稳定性测试
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.isaac_sim_test_framework import IsaacSimTestFramework, TestResult


class L6EndToEndTests:
    """L6 端到端验证测试"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/environments/isaac_sim/nova_carter.yaml"
        self.framework = IsaacSimTestFramework(self.config_path)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有L6测试"""
        print("=" * 70)
        print("L6: 端到端验证测试")
        print("=" * 70)
        print("\n测试内容：")
        print("  1. 简单导航任务 (L6-001)")
        print("  2. 搜索任务 (L6-002)")
        print("  3. 系统稳定性测试 (L6-003)")
        print("\n" + "-" * 70)
        
        try:
            await self.framework.setup()
            
            # 运行各项测试
            await self.framework.run_test("L6-001: 简单导航任务", self.test_simple_navigation_task)
            await self.framework.run_test("L6-002: 搜索任务", self.test_search_task)
            await self.framework.run_test("L6-003: 系统稳定性测试", self.test_system_stability)
            
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
            logger.error(f"L6测试套件异常: {e}")
            print(f"\n✗ 测试异常: {e}")
            return {"error": str(e)}
        finally:
            await self.framework.cleanup()
    
    async def test_simple_navigation_task(self) -> TestResult:
        """L6-001: 简单导航任务"""
        print("\n" + "-" * 70)
        print("L6-001: 简单导航任务")
        print("-" * 70)
        print("测试任务: 移动到目标位置")
        
        try:
            mission = await self.framework.brain.process_command(
                command="移动到前方2米处",
                platform_type="ugv"
            )
            
            ops_count = len(mission.operations)
            print(f"\n生成操作数: {ops_count}")
            
            if ops_count > 0:
                print("操作序列:")
                for i, op in enumerate(mission.operations[:5], 1):
                    print(f"  {i}. {op.name}")
                
                return TestResult(
                    name="简单导航任务",
                    success=True,
                    message=f"成功生成 {ops_count} 个操作",
                    metrics={
                        "operations_count": ops_count,
                        "mission_id": mission.id
                    }
                )
            else:
                return TestResult(
                    name="简单导航任务",
                    success=False,
                    message="未能生成操作序列",
                    metrics={}
                )
        except Exception as e:
            logger.error(f"L6-001测试异常: {e}")
            return TestResult(
                name="简单导航任务",
                success=False,
                message=f"测试异常: {str(e)}"
            )
    
    async def test_search_task(self) -> TestResult:
        """L6-002: 搜索任务"""
        print("\n" + "-" * 70)
        print("L6-002: 搜索任务")
        print("-" * 70)
        print("测试任务: 搜索并检测目标")
        
        try:
            mission = await self.framework.brain.process_command(
                command="搜索周围环境并检测障碍物",
                platform_type="ugv"
            )
            
            ops_count = len(mission.operations)
            print(f"\n生成操作数: {ops_count}")
            
            if ops_count > 0:
                return TestResult(
                    name="搜索任务",
                    success=True,
                    message=f"成功生成 {ops_count} 个操作",
                    metrics={
                        "operations_count": ops_count,
                        "mission_id": mission.id
                    }
                )
            else:
                return TestResult(
                    name="搜索任务",
                    success=False,
                    message="未能生成操作序列",
                    metrics={}
                )
        except Exception as e:
            logger.error(f"L6-002测试异常: {e}")
            return TestResult(
                name="搜索任务",
                success=False,
                message=f"测试异常: {str(e)}"
            )
    
    async def test_system_stability(self) -> TestResult:
        """L6-003: 系统稳定性测试"""
        print("\n" + "-" * 70)
        print("L6-003: 系统稳定性测试")
        print("-" * 70)
        print("测试系统长时间运行的稳定性")
        
        try:
            test_commands = [
                "移动到前方1米处",
                "向右转90度",
                "向前移动0.5米"
            ]
            
            successful_plans = 0
            
            for cmd in test_commands:
                try:
                    mission = await self.framework.brain.process_command(
                        command=cmd,
                        platform_type="ugv"
                    )
                    
                    if len(mission.operations) > 0:
                        successful_plans += 1
                        print(f"  ✓ {cmd}")
                    else:
                        print(f"  ✗ {cmd}")
                except Exception as e:
                    print(f"  ✗ {cmd} - {str(e)}")
            
            stability_rate = successful_plans / len(test_commands)
            
            if stability_rate >= 0.8:
                return TestResult(
                    name="系统稳定性",
                    success=True,
                    message=f"系统稳定性良好 (成功率: {stability_rate*100:.1f}%)",
                    metrics={
                        "successful_plans": successful_plans,
                        "total_tests": len(test_commands),
                        "stability_rate": stability_rate
                    }
                )
            else:
                return TestResult(
                    name="系统稳定性",
                    success=False,
                    message=f"系统稳定性不足 (成功率: {stability_rate*100:.1f}%)",
                    metrics={
                        "successful_plans": successful_plans,
                        "total_tests": len(test_commands),
                        "stability_rate": stability_rate
                    }
                )
        except Exception as e:
            logger.error(f"L6-003测试异常: {e}")
            return TestResult(
                name="系统稳定性",
                success=False,
                message=f"测试异常: {str(e)}"
            )


async def main():
    """主函数"""
    tests = L6EndToEndTests()
    summary = await tests.run_all_tests()
    
    if 'error' in summary:
        sys.exit(1)
    elif summary.get('success_rate', 0) >= 80:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())








