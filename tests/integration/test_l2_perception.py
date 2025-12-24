#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2: 感知层验证测试

验证感知层的数据处理、目标检测、地图构建和VLM感知能力

测试覆盖：
- 感知数据管道
- 目标检测
- 占据栅格地图
- VLM场景理解
- 感知异常处理
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


class L2PerceptionTests:
    """L2 感知层验证测试"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化测试套件
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "config/environments/isaac_sim/nova_carter.yaml"
        self.framework = IsaacSimTestFramework(self.config_path)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有L2测试
        
        Returns:
            测试摘要
        """
        print("=" * 70)
        print("L2: 感知层验证测试")
        print("=" * 70)
        print("\n测试内容：")
        print("  1. 感知数据管道 (L2-001)")
        print("  2. 目标检测 (L2-002)")
        print("  3. 占据栅格地图 (L2-003)")
        print("  4. VLM场景理解 (L2-004)")
        print("  5. 感知异常处理 (L2-005)")
        print("\n" + "-" * 70)
        
        try:
            # 设置测试环境
            await self.framework.setup()
            
            # 等待传感器数据
            print("等待传感器数据...")
            data_ready = await self.framework.wait_for_topics(timeout=15.0)
            if not data_ready:
                logger.warning("传感器数据未就绪，继续测试...")
            await asyncio.sleep(3.0)  # 额外等待以收集数据
            
            # L2-001: 感知数据管道
            result = await self.framework.run_test(
                "L2-001: 感知数据管道",
                self.framework.verify_perception_pipeline
            )
            
            # L2-002: 目标检测
            result = await self.framework.run_test(
                "L2-002: 目标检测",
                self.framework.verify_object_detection
            )
            
            # L2-003: 占据栅格地图
            result = await self.framework.run_test(
                "L2-003: 占据栅格地图",
                self.framework.verify_occupancy_mapping
            )
            
            # L2-004: VLM场景理解
            result = await self.framework.run_test(
                "L2-004: VLM场景理解",
                self.framework.verify_vlm_scene_understanding
            )
            
            # L2-005: 感知异常处理
            result = await self.framework.run_test(
                "L2-005: 感知异常处理",
                self.framework.verify_perception_anomaly_handling
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
            logger.error(f"L2测试套件异常: {e}")
            print(f"\n✗ 测试异常: {e}")
            return {"error": str(e)}
        
        finally:
            # 清理测试环境
            await self.framework.cleanup()
    
    async def run_test_l2_001(self) -> Dict[str, Any]:
        """运行L2-001: 感知数据管道"""
        print("\n" + "-" * 70)
        print("L2-001: 感知数据管道")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待传感器数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=15.0)
            await asyncio.sleep(2.0)
            
            result = await self.framework.verify_perception_pipeline()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出管道详情
            if 'metrics' in result:
                pipeline_stages = result['metrics'].get('pipeline_stages', [])
                completeness = result['metrics'].get('completeness', 0)
                processing_time = result['metrics'].get('processing_time_ms', 0)
                
                print(f"\n管道完成度: {completeness*100:.1f}%")
                print(f"处理时间: {processing_time:.1f} ms")
                print("\n处理阶段:")
                for stage in pipeline_stages:
                    stage_name = stage.get('stage', 'unknown')
                    stage_status = "✓" if stage.get('status') == 'success' else "✗"
                    print(f"  {stage_status} {stage_name}")
                    
                    if stage.get('objects_detected'):
                        print(f"       检测到 {stage['objects_detected']} 个对象")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L2-001测试异常: {e}")
            return {
                "name": "L2-001: 感知数据管道",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l2_002(self) -> Dict[str, Any]:
        """运行L2-002: 目标检测"""
        print("\n" + "-" * 70)
        print("L2-002: 目标检测")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待传感器数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=15.0)
            await asyncio.sleep(2.0)
            
            result = await self.framework.verify_object_detection()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出检测详情
            if 'metrics' in result:
                total_detections = result['metrics'].get('total_detections', 0)
                confidence_stats = result['metrics'].get('confidence_stats', {})
                by_type = result['metrics'].get('by_type', {})
                
                print(f"\n总检测数: {total_detections}")
                print(f"平均置信度: {confidence_stats.get('mean', 0):.2f}")
                print(f"最小置信度: {confidence_stats.get('min', 0):.2f}")
                print(f"最大置信度: {confidence_stats.get('max', 0):.2f}")
                
                if by_type:
                    print("\n按类型统计:")
                    for obj_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {obj_type}: {count} 个")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L2-002测试异常: {e}")
            return {
                "name": "L2-002: 目标检测",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l2_003(self) -> Dict[str, Any]:
        """运行L2-003: 占据栅格地图"""
        print("\n" + "-" * 70)
        print("L2-003: 占据栅格地图")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待传感器数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=15.0)
            await asyncio.sleep(2.0)
            
            result = await self.framework.verify_occupancy_mapping()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出地图详情
            if 'metrics' in result:
                grid_shape = result['metrics'].get('grid_shape', 'unknown')
                total_cells = result['metrics'].get('total_cells', 0)
                occupied_cells = result['metrics'].get('occupied_cells', 0)
                free_cells = result['metrics'].get('free_cells', 0)
                unknown_cells = result['metrics'].get('unknown_cells', 0)
                occupancy_rate = result['metrics'].get('occupancy_rate', 0)
                
                print(f"\n地图尺寸: {grid_shape}")
                print(f"总单元格: {total_cells}")
                print(f"占据单元格: {occupied_cells} ({occupied_cells/total_cells*100:.1f}%)")
                print(f"空闲单元格: {free_cells} ({free_cells/total_cells*100:.1f}%)")
                print(f"未知单元格: {unknown_cells} ({unknown_cells/total_cells*100:.1f}%)")
                print(f"占据率: {occupancy_rate*100:.1f}%")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L2-003测试异常: {e}")
            return {
                "name": "L2-003: 占据栅格地图",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l2_004(self) -> Dict[str, Any]:
        """运行L2-004: VLM场景理解"""
        print("\n" + "-" * 70)
        print("L2-004: VLM场景理解")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待传感器数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=15.0)
            await asyncio.sleep(2.0)
            
            result = await self.framework.verify_vlm_scene_understanding()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出VLM详情
            if 'metrics' in result:
                objects_count = result['metrics'].get('objects_count', 0)
                object_types = result['metrics'].get('object_types', {})
                
                print(f"\n识别对象数: {objects_count}")
                
                if object_types:
                    print("\n按类型统计:")
                    for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {obj_type}: {count} 个")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L2-004测试异常: {e}")
            return {
                "name": "L2-004: VLM场景理解",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()
    
    async def run_test_l2_005(self) -> Dict[str, Any]:
        """运行L2-005: 感知异常处理"""
        print("\n" + "-" * 70)
        print("L2-005: 感知异常处理")
        print("-" * 70)
        
        try:
            await self.framework.setup()
            
            # 等待传感器数据
            print("等待传感器数据...")
            await self.framework.wait_for_topics(timeout=15.0)
            await asyncio.sleep(3.0)  # 等待以检测可能的数据丢失
            
            result = await self.framework.verify_perception_anomaly_handling()
            
            status = "通过" if result.success else "失败"
            print(f"状态: {status}")
            print(f"消息: {result.message}")
            
            # 输出异常处理详情
            if 'metrics' in result:
                anomaly_metrics = result['metrics']
                
                # 检查数据丢失
                data_loss_handling = anomaly_metrics.get('data_loss_handling', [])
                if data_loss_handling:
                    print(f"\n检测到 {len(data_loss_handling)} 个数据丢失:")
                    for loss in data_loss_handling:
                        topic = loss.get('topic', 'unknown')
                        handled = loss.get('handled', False)
                        handled_str = "已处理" if handled else "未处理"
                        print(f"  - {topic}: {handled_str}")
                
                # 传感器故障恢复
                sensor_failure_recovery = anomaly_metrics.get('sensor_failure_recovery', [])
                if sensor_failure_recovery:
                    print(f"\n传感器故障恢复:")
                    for recovery in sensor_failure_recovery:
                        print(f"  - {recovery}")
                
                # 超时处理
                timeout_handling = anomaly_metrics.get('timeout_handling', [])
                if timeout_handling:
                    print(f"\n超时处理:")
                    for timeout in timeout_handling:
                        print(f"  - {timeout}")
            
            return result.__dict__
            
        except Exception as e:
            logger.error(f"L2-005测试异常: {e}")
            return {
                "name": "L2-005: 感知异常处理",
                "success": False,
                "message": f"测试异常: {str(e)}"
            }
        
        finally:
            await self.framework.cleanup()


async def main():
    """主函数"""
    tests = L2PerceptionTests()
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


