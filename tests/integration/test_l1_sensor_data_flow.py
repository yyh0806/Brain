#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L1: 传感器数据流验证测试

验证传感器数据从ROS2话题到Brain系统的完整数据流

测试覆盖：
- ROS2话题连接验证
- 传感器数据接收频率验证
- 数据质量验证
- 多传感器时间同步
- 传感器数据融合
"""

import asyncio
import sys
from pathlib import Path
import time
from typing import Dict, Any, Optional
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.isaac_sim_test_framework import IsaacSimTestFramework, TestResult
from brain.communication.ros2_interface import TwistCommand


class L1SensorDataFlowTests:
    """L1 传感器数据流验证测试"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化测试套件
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "config/environments/isaac_sim/nova_carter.yaml"
        self.framework = IsaacSimTestFramework(self.config_path)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有L1测试
        
        Returns:
            测试摘要
        """
        print("=" * 70)
        print("L1: 传感器数据流验证测试")
        print("=" * 70)
        print("\n测试内容：")
        print("  1. ROS2话题连接验证 (L1-001)")
        print("  2. 数据接收频率验证 (L1-002)")
        print("  3. 数据质量验证 (L1-003)")
        print("  4. 多传感器时间同步 (L1-004)")
        print("  5. 传感器数据融合 (L1-005)")
        print("\n" + "-" * 70)
        
        try:
            # 设置测试环境
            await self.framework.setup()
            
            # L1-001: ROS2话题连接验证
            result = await self.framework.run_test(
                "L1-001: ROS2话题连接验证",
                self.framework.verify_ros2_topics_connection
            )
            
            # L1-002: 数据接收频率验证
            result = await self.framework.run_test(
                "L1-002: 数据接收频率验证",
                self.framework.verify_sensor_data_rate
            )
            
            # L1-003: 数据质量验证
            result = await self.framework.run_test(
                "L1-003: 数据质量验证",
                self.framework.verify_data_quality
            )
            
            # L1-004: 多传感器时间同步
            result = await self.framework.run_test(
                "L1-004: 多传感器时间同步",
                self.framework.verify_sensor_synchronization
            )
            
            # L1-005: 传感器数据融合
            result = await self.framework.run_test(
                "L1-005: 传感器数据融合",
                self.framework.verify_sensor_fusion
            )
            
            # 获取测试摘要
            summary = self.framework.get_test_summary()
            
            # 输出测试摘要
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
            logger.error(f"L1测试套件异常: {e}")
            print(f"\n✗ 测试异常: {e}")
            return {"error": str(e)}
        
        finally:
            # 清理测试环境
            await self.framework.cleanup()
    
    async def run_test_l1_001(self) -> Dict[str, Any]:
        """运行L1-001: ROS2话题连接验证"""
        print("\n" + "-" * 70)
        print("L1-001: ROS2话题连接验证")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            result = await self.framework.verify_ros2_topics_connection()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            print(f"\n连接话题: {len(result.metrics.get('connected_topics', []))}")
            print(f"未连接话题: {len(result.metrics.get('disconnected_topics', []))}")
            print(f"连接率: {result.metrics.get('connection_rate', 0):.1f}%")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L1-001测试异常: {e}")
            return {
                "name": "L1-001: ROS2话题连接验证",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l1_002(self) -> Dict[str, Any]:
        """运行L1-002: 数据接收频率验证"""
        print("\n" + "-" * 70)
        print("L1-002: 数据接收频率验证")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=10.0)
            await asyncio.sleep(2.0)  # 额外等待以收集更多数据
            
            result = await self.framework.verify_sensor_data_rate()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出各传感器的数据频率
            if 'metrics' in result:
                rate_metrics = result['metrics']
                for topic, metrics in rate_metrics.items():
                    expected = metrics.get('expected_rate', 0)
                    actual = metrics.get('actual_rate', 0)
                    error = metrics.get('error_percent', 0)
                    passed = "✓" if metrics.get('passed', False) else "✗"
                    print(f"\n  {passed} {topic}:")
                    print(f"     预期频率: {expected:.1f} Hz")
                    print(f"     实际频率: {actual:.1f} Hz")
                    print(f"     误差: {error:.1f}%")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L1-002测试异常: {e}")
            return {
                "name": "L1-002: 数据接收频率验证",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l1_003(self) -> Dict[str, Any]:
        """运行L1-003: 数据质量验证"""
        print("\n" + "-" * 70)
        print("L1-003: 数据质量验证")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=10.0)
            await asyncio.sleep(3.0)  # 等待以收集足够的数据
            
            result = await self.framework.verify_data_quality()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出质量指标
            if 'metrics' in result:
                quality_metrics = result['metrics']
                total = quality_metrics.get('total_checks', 0)
                passed = quality_metrics.get('passed_checks', 0)
                rate = quality_metrics.get('success_rate', 0)
                
                print(f"\n  总检查项: {total}")
                print(f"  通过检查项: {passed}")
                print(f"  质量评分: {rate*100:.1f}%")
                
                if 'quality_metrics' in quality_metrics:
                    for topic, metrics in quality_metrics['quality_metrics'].items():
                        completeness = metrics.get('completeness', 0) * 100
                        timeliness = metrics.get('timeliness', 0) * 100
                        quality_score = metrics.get('quality_score', 0) * 100
                        passed = "✓" if metrics.get('passed', False) else "✗"
                        print(f"\n  {passed} {topic}:")
                        print(f"     完整性: {completeness:.1f}%")
                        print(f"     时效性: {timeliness:.1f}%")
                        print(f"     质量评分: {quality_score:.1f}%")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L1-003测试异常: {e}")
            return {
                "name": "L1-003: 数据质量验证",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l1_004(self) -> Dict[str, Any]:
        """运行L1-004: 多传感器时间同步"""
        print("\n" + "-" * 70)
        print("L1-004: 多传感器时间同步")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=10.0)
            await asyncio.sleep(2.0)  # 等待收集数据
            
            result = await self.framework.verify_sensor_synchronization()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出同步指标
            if 'metrics' in result:
                sync_metrics = result['metrics']
                max_diff = sync_metrics.get('max_timestamp_diff_ms', 0)
                avg_diff = sync_metrics.get('avg_timestamp_diff_ms', 0)
                threshold = sync_metrics.get('sync_threshold_ms', 0)
                
                print(f"\n  最大时间差: {max_diff:.1f} ms")
                print(f"  平均时间差: {avg_diff:.1f} ms")
                print(f"  同步阈值: {threshold:.1f} ms")
                
                if max_diff < threshold:
                    print(f"\n  ✓ 时间同步质量良好")
                else:
                    print(f"\n  ✗ 时间同步超出阈值")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L1-004测试异常: {e}")
            return {
                "name": "L1-004: 多传感器时间同步",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l1_005(self) -> Dict[str, Any]:
        """运行L1-005: 传感器数据融合"""
        print("\n" + "-" * 70)
        print("L1-005: 传感器数据融合")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=10.0)
            await asyncio.sleep(2.0)
            
            result = await self.framework.verify_sensor_fusion()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出融合指标
            if 'metrics' in result:
                fusion_metrics = result['metrics']
                data_available = fusion_metrics.get('data_available', False)
                used_sources = fusion_metrics.get('used_sources', [])
                fusion_count = fusion_metrics.get('fusion_count', 0)
                
                print(f"\n  数据可用: {data_available}")
                print(f"  融合数据源数量: {fusion_count}")
                print(f"  使用的数据源: {used_sources}")
                
                if data_available and fusion_count >= 2:
                    print(f"\n  ✓ 融合质量良好")
                else:
                    print(f"\n  ✗ 融合质量不足")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L1-005测试异常: {e}")
            return {
                "name": "L1-005: 传感器数据融合",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()


async def main():
    """主函数"""
    tests = L1SensorDataFlowTests()
    summary = await tests.run_all_tests()
    
    # 返回退出码
    if 'error' in summary:
        sys.exit(1)
    elif summary.get('success_rate', 0) >= 80:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

