#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6层验证测试执行脚本

统一运行L1-L6所有验证测试，生成综合报告
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.test_l1_sensor_data_flow import L1SensorDataFlowTests
from tests.integration.test_l2_perception import L2PerceptionTests
from tests.integration.test_l3_cognitive import L3CognitiveTests
from tests.integration.test_l4_planning import L4PlanningTests
from tests.integration.test_l5_execution import L5ExecutionTests
from tests.integration.test_l6_end_to_end import L6EndToEndTests


class TestRunner:
    """测试运行器"""
    
    def __init__(self, config_path=None):
        """初始化测试运行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.test_results: Dict[str, Any] = {}
        self.start_time = time.time()
        
        # 测试套件
        self.test_suites = {
            "L1": L1SensorDataFlowTests(config_path),
            "L2": L2PerceptionTests(config_path),
            "L3": L3CognitiveTests(config_path),
            "L4": L4PlanningTests(config_path),
            "L5": L5ExecutionTests(config_path),
            "L6": L6EndToEndTests(config_path)
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有6层测试
        
        Returns:
            综合测试结果
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "Brain系统6层验证测试" + " " * 20)
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"配置文件: {self.config_path or '默认配置'}")
        print("=" * 80)
        
        # 依次运行各层测试
        for layer_name, test_suite in self.test_suites.items():
            print(f"\n\n{'#' * 80}")
            print(f"# {layer_name} 测试")
            print(f"{'#' * 80}")
            
            try:
                result = await test_suite.run_all_tests()
                self.test_results[layer_name] = result
                
                # 输出层测试摘要
                self._print_layer_summary(layer_name, result)
                
            except Exception as e:
                logger.error(f"{layer_name} 测试失败: {e}")
                self.test_results[layer_name] = {
                    "error": str(e),
                    "success_rate": 0
                }
        
        # 生成综合报告
        end_time = time.time()
        total_time = end_time - self.start_time
        
        report = self._generate_comprehensive_report(total_time)
        self._print_comprehensive_summary(report)
        
        # 保存报告
        self._save_report(report)
        
        return report
    
    def _print_layer_summary(self, layer_name: str, result: Dict[str, Any]):
        """打印单层测试摘要"""
        if "error" in result:
            print(f"\n✗ {layer_name} 测试异常: {result['error']}")
            return
        
        total = result.get('total_tests', 0)
        passed = result.get('passed', 0)
        failed = result.get('failed', 0)
        rate = result.get('success_rate', 0)
        
        print(f"\n{layer_name} 测试结果:")
        print(f"  总测试数: {total}")
        print(f"  通过: {passed}")
        print(f"  失败: {failed}")
        print(f"  成功率: {rate:.1f}%")
        
        if rate >= 80:
            print(f"  ✓ {layer_name} 测试通过")
        elif rate >= 60:
            print(f"  ⚠ {layer_name} 测试部分通过")
        else:
            print(f"  ✗ {layer_name} 测试未通过")
    
    def _generate_comprehensive_report(self, total_time: float) -> Dict[str, Any]:
        """生成综合测试报告"""
        # 统计所有测试
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        layer_stats = {}
        
        for layer_name, result in self.test_results.items():
            if "error" in result:
                layer_stats[layer_name] = {
                    "status": "ERROR",
                    "error": result['error']
                }
                continue
            
            layer_tests = result.get('total_tests', 0)
            layer_passed = result.get('passed', 0)
            layer_failed = result.get('failed', 0)
            layer_rate = result.get('success_rate', 0)
            
            total_tests += layer_tests
            total_passed += layer_passed
            total_failed += layer_failed
            
            layer_stats[layer_name] = {
                "total_tests": layer_tests,
                "passed": layer_passed,
                "failed": layer_failed,
                "success_rate": layer_rate,
                "status": "PASS" if layer_rate >= 80 else "FAIL"
            }
        
        # 计算整体成功率
        overall_rate = total_passed / total_tests * 100 if total_tests > 0 else 0
        
        # 评估整体结果
        if overall_rate >= 90:
            overall_status = "EXCELLENT"
        elif overall_rate >= 80:
            overall_status = "GOOD"
        elif overall_rate >= 60:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "POOR"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": total_time,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_success_rate": overall_rate,
            "overall_status": overall_status,
            "layer_results": layer_stats,
            "test_details": self.test_results
        }
    
    def _print_comprehensive_summary(self, report: Dict[str, Any]):
        """打印综合摘要"""
        print("\n\n" + "=" * 80)
        print(" " * 20 + "综合测试报告" + " " * 20)
        print("=" * 80)
        
        # 整体统计
        print(f"\n整体统计:")
        print(f"  总测试数: {report['total_tests']}")
        print(f"  通过: {report['total_passed']}")
        print(f"  失败: {report['total_failed']}")
        print(f"  成功率: {report['overall_success_rate']:.1f}%")
        print(f"   状态: {report['overall_status']}")
        
        # 状态图
        status_colors = {
            "EXCELLENT": "🟢",
            "GOOD": "🟢",
            "ACCEPTABLE": "🟡",
            "POOR": "🔴",
            "ERROR": "🟣"
        }
        
        # 各层结果
        print(f"\n各层测试结果:")
        for layer, stats in report['layer_results'].items():
            if "error" in stats:
                print(f"  {status_colors['ERROR']} {layer}: ERROR - {stats['error']}")
            else:
                status = stats['status']
                rate = stats['success_rate']
                print(f"  {status_colors.get(status, '⚪')} {layer}: {rate:.1f}% - {status}")
        
        # 时间统计
        duration = report['total_duration_seconds']
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"\n总耗时: {minutes}分{seconds}秒")
        
        # 结论
        print("\n" + "=" * 80)
        if report['overall_status'] in ["EXCELLENT", "GOOD"]:
            print("✅ 所有测试通过，系统验证成功")
        elif report['overall_status'] == "ACCEPTABLE":
            print("⚠️ 部分测试未通过，请关注失败的测试项")
        else:
            print("❌ 测试未通过，系统存在问题，需要修复")
        print("=" * 80)
    
    def _save_report(self, report: Dict[str, Any]):
        """保存测试报告到文件"""
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"brain_test_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✓ 测试报告已保存: {report_file}")
            
            # 保存人类可读报告
            readable_file = reports_dir / f"brain_test_report_{timestamp}.txt"
            with open(readable_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Brain系统6层验证测试报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"生成时间: {report['timestamp']}\n")
                f.write(f"总耗时: {report['total_duration_seconds']:.1f}秒\n\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("整体统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"总测试数: {report['total_tests']}\n")
                f.write(f"通过: {report['total_passed']}\n")
                f.write(f"失败: {report['total_failed']}\n")
                f.write(f"成功率: {report['overall_success_rate']:.1f}%\n")
                f.write(f"状态: {report['overall_status']}\n\n")
                
                f.write("各层结果\n")
                f.write("-" * 40 + "\n")
                for layer, stats in report['layer_results'].items():
                    if "error" in stats:
                        f.write(f"{layer}: ERROR - {stats['error']}\n")
                    else:
                        f.write(f"{layer}: {stats['success_rate']:.1f}% - {stats['status']}\n")
                
                f.write("\n详细结果\n")
                f.write("-" * 40 + "\n")
                for layer, result in report['test_details'].items():
                    f.write(f"\n{layer} 测试:\n")
                    if "error" in result:
                        f.write(f"  错误: {result['error']}\n")
                    else:
                        f.write(f"  总数: {result['total_tests']}\n")
                        f.write(f"  通过: {result['passed']}\n")
                        f.write(f"  失败: {result['failed']}\n")
                        
                        if 'results' in result:
                            for test_result in result['results']:
                                status = "✓" if test_result['success'] else "✗"
                                f.write(f"    {status} {test_result['name']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
            print(f"✓ 可读报告已保存: {readable_file}")
            
        except Exception as e:
            logger.error(f"保存测试报告失败: {e}")


async def main():
    """主函数"""
    # 解析命令行参数
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    runner = TestRunner(config_path)
    report = await runner.run_all_tests()
    
    # 根据测试结果返回退出码
    if report['overall_status'] in ["EXCELLENT", "GOOD"]:
        sys.exit(0)
    elif report['overall_status'] == "ACCEPTABLE":
        sys.exit(2)  # 部分通过，返回警告退出码
    else:
        sys.exit(1)  # 失败


if __name__ == "__main__":
    asyncio.run(main())

