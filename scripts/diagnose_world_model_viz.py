#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldModel可视化诊断工具

诊断WorldModel的状态，验证所有数据字段是否正确初始化，
并测试每个可视化生成方法。

Usage:
    export ROS_DOMAIN_ID=42
    python3 scripts/diagnose_world_model_viz.py
"""

import sys
import os
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from loguru import logger


class WorldModelDiagnostic:
    """WorldModel诊断器"""

    def __init__(self):
        self.results = {
            'world_model_status': 'NOT_INITIALIZED',
            'semantic_grid': 'NOT_TESTED',
            'semantic_markers': 'NOT_TESTED',
            'trajectory': 'NOT_TESTED',
            'frontiers': 'NOT_TESTED',
            'belief_markers': 'NOT_TESTED',
            'change_markers': 'NOT_TESTED',
            'vlm_markers': 'NOT_TESTED'
        }

        self.errors = []
        self.warnings = []

    def run_diagnostic(self):
        """运行完整诊断"""
        print("\n" + "=" * 80)
        print("🔍 WorldModel可视化诊断")
        print("=" * 80)
        print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        # 1. 检查WorldModel导入
        self._test_world_model_import()

        # 2. 检查WorldModel实例化
        self._test_world_model_instantiation()

        # 3. 检查数据字段初始化
        self._test_data_fields()

        # 4. 测试可视化生成方法
        if self.world_model:
            self._test_visualization_generation()

        # 5. 生成诊断报告
        self._generate_report()

    def _test_world_model_import(self):
        """测试WorldModel导入"""
        print("📦 测试1: WorldModel导入")
        try:
            from brain.cognitive.world_model.world_model import WorldModel
            self.world_model_class = WorldModel
            print("  ✅ WorldModel导入成功")
            self.results['world_model_status'] = 'IMPORT_OK'
        except Exception as e:
            self.errors.append(f"WorldModel导入失败: {e}")
            print(f"  ❌ WorldModel导入失败: {e}")
            self.results['world_model_status'] = 'IMPORT_FAILED'

    def _test_world_model_instantiation(self):
        """测试WorldModel实例化"""
        if self.results['world_model_status'] != 'IMPORT_OK':
            return

        print("\n📦 测试2: WorldModel实例化")
        try:
            self.world_model = self.world_model_class(config={
                'map_resolution': 0.1,
                'map_size': 100.0
            })
            print("  ✅ WorldModel实例化成功")
            self.results['world_model_status'] = 'INSTANTIATED'

            # 检查ROS2初始化
            try:
                import rclpy
                rclpy.init(args=None)
                self.ros2_available = True
                print("  ✅ ROS2环境可用")
            except Exception as e:
                self.ros2_available = False
                self.warnings.append(f"ROS2环境不可用: {e}")
                print(f"  ⚠️  ROS2环境不可用: {e}")

        except Exception as e:
            self.errors.append(f"WorldModel实例化失败: {e}")
            print(f"  ❌ WorldModel实例化失败: {e}")
            self.results['world_model_status'] = 'INSTANTIATION_FAILED'

    def _test_data_fields(self):
        """测试数据字段初始化"""
        if not hasattr(self, 'world_model') or self.world_model is None:
            return

        print("\n📊 测试3: 数据字段初始化")
        wm = self.world_model

        # 检查current_map
        if wm.current_map is None:
            self.warnings.append("current_map为None，创建默认值")
            wm.current_map = np.full((100, 100), -1, dtype=np.int8)
            wm.map_resolution = 0.1
            wm.map_origin = (0.0, 0.0)
            print("  ⚠️  current_map为None，已创建默认值 (100x100)")
        else:
            print(f"  ✅ current_map已初始化: {wm.current_map.shape}")

        # 检查semantic_objects
        if not hasattr(wm, 'semantic_objects'):
            wm.semantic_objects = {}
            print("  ⚠️  semantic_objects不存在，已创建")
        else:
            print(f"  ✅ semantic_objects: {len(wm.semantic_objects)} 个物体")

        # 检查pose_history
        if not hasattr(wm, 'pose_history'):
            wm.pose_history = []
            print("  ⚠️  pose_history不存在，已创建")
        else:
            print(f"  ✅ pose_history: {len(wm.pose_history)} 个记录")

        # 检查exploration_frontiers
        if not hasattr(wm, 'exploration_frontiers'):
            wm.exploration_frontiers = []
            print("  ⚠️  exploration_frontiers不存在，已创建")
        else:
            print(f"  ✅ exploration_frontiers: {len(wm.exploration_frontiers)} 个前沿")

        # 检查belief_revision_policy
        if hasattr(wm, 'belief_revision_policy') and wm.belief_revision_policy is not None:
            print(f"  ✅ belief_revision_policy: 已启用")
        else:
            self.warnings.append("belief_revision_policy未启用")
            print(f"  ⚠️  belief_revision_policy: 未启用")

        # 检查pending_changes
        if not hasattr(wm, 'pending_changes'):
            wm.pending_changes = []
            print("  ⚠️  pending_changes不存在，已创建")
        else:
            print(f"  ✅ pending_changes: {len(wm.pending_changes)} 个变化")

    def _test_visualization_generation(self):
        """测试可视化生成方法"""
        if not self.ros2_available:
            print("\n⚠️  跳过可视化测试 (ROS2不可用)")
            return

        print("\n🎨 测试4: 可视化生成方法")

        try:
            from brain.cognitive.world_model.world_model_visualizer import WorldModelVisualizer

            # 创建可视化器
            visualizer = WorldModelVisualizer(
                world_model=self.world_model,
                publish_rate=2.0
            )
            print("  ✅ WorldModelVisualizer实例化成功")

            # 测试每个生成方法
            self._test_semantic_grid(visualizer)
            self._test_semantic_markers(visualizer)
            self._test_trajectory(visualizer)
            self._test_frontiers(visualizer)
            self._test_belief_markers(visualizer)
            self._test_change_markers(visualizer)
            self._test_vlm_markers(visualizer)

            # 清理
            visualizer.destroy_node()
            print("\n  ✅ 可视化器测试完成")

        except Exception as e:
            self.errors.append(f"可视化测试失败: {e}")
            print(f"  ❌ 可视化测试失败: {e}")

    def _test_semantic_grid(self, visualizer):
        """测试语义占据栅格生成"""
        try:
            grid = visualizer._generate_semantic_grid()
            if grid is not None:
                print(f"  ✅ semantic_grid: {grid.info.width}x{grid.info.height}, {len(grid.data)} cells")
                self.results['semantic_grid'] = 'OK'
            else:
                self.errors.append("semantic_grid生成返回None")
                print(f"  ❌ semantic_grid: 返回None")
                self.results['semantic_grid'] = 'FAILED'
        except Exception as e:
            self.errors.append(f"semantic_grid生成错误: {e}")
            print(f"  ❌ semantic_grid错误: {e}")
            self.results['semantic_grid'] = 'ERROR'

    def _test_semantic_markers(self, visualizer):
        """测试语义物体标注生成"""
        try:
            markers = visualizer._generate_semantic_markers()
            if markers is not None and len(markers.markers) > 0:
                print(f"  ✅ semantic_markers: {len(markers.markers)} 个标记")
                self.results['semantic_markers'] = 'OK'
            else:
                self.warnings.append("semantic_markers为空")
                print(f"  ⚠️  semantic_markers: 为空")
                self.results['semantic_markers'] = 'EMPTY'
        except Exception as e:
            self.errors.append(f"semantic_markers生成错误: {e}")
            print(f"  ❌ semantic_markers错误: {e}")
            self.results['semantic_markers'] = 'ERROR'

    def _test_trajectory(self, visualizer):
        """测试机器人轨迹生成"""
        # 先添加一些轨迹数据
        if len(self.world_model.pose_history) == 0:
            self.world_model._record_pose({
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'yaw': 0.0, 'velocity': {}
            })
            self.world_model._record_pose({
                'x': 1.0, 'y': 0.0, 'z': 0.0,
                'yaw': 0.0, 'velocity': {}
            })

        try:
            trajectory = visualizer._generate_trajectory()
            if trajectory is not None:
                print(f"  ✅ trajectory: {len(trajectory.poses)} 个位姿")
                self.results['trajectory'] = 'OK'
            else:
                self.errors.append("trajectory生成返回None")
                print(f"  ❌ trajectory: 返回None")
                self.results['trajectory'] = 'FAILED'
        except Exception as e:
            self.errors.append(f"trajectory生成错误: {e}")
            print(f"  ❌ trajectory错误: {e}")
            self.results['trajectory'] = 'ERROR'

    def _test_frontiers(self, visualizer):
        """测试探索前沿生成"""
        # 添加一些前沿数据
        if len(self.world_model.exploration_frontiers) == 0:
            from brain.cognitive.world_model.semantic.semantic_object import ExplorationFrontier
            self.world_model.exploration_frontiers.append(
                ExplorationFrontier(
                    id='frontier_1',
                    position=(5.0, 5.0),
                    direction=0.0,
                    priority=0.8
                )
            )

        try:
            frontiers = visualizer._generate_frontier_markers()
            if frontiers is not None and len(frontiers.markers) > 0:
                print(f"  ✅ frontiers: {len(frontiers.markers)} 个标记")
                self.results['frontiers'] = 'OK'
            else:
                self.warnings.append("frontiers为空")
                print(f"  ⚠️  frontiers: 为空")
                self.results['frontiers'] = 'EMPTY'
        except Exception as e:
            self.errors.append(f"frontiers生成错误: {e}")
            print(f"  ❌ frontiers错误: {e}")
            self.results['frontiers'] = 'ERROR'

    def _test_belief_markers(self, visualizer):
        """测试信念标记生成"""
        try:
            markers = visualizer._generate_belief_markers()
            if markers is not None and len(markers.markers) > 0:
                print(f"  ✅ belief_markers: {len(markers.markers)} 个标记")
                self.results['belief_markers'] = 'OK'
            else:
                self.warnings.append("belief_markers为空或belief_policy未启用")
                print(f"  ⚠️  belief_markers: 为空或未启用")
                self.results['belief_markers'] = 'EMPTY'
        except Exception as e:
            self.errors.append(f"belief_markers生成错误: {e}")
            print(f"  ❌ belief_markers错误: {e}")
            self.results['belief_markers'] = 'ERROR'

    def _test_change_markers(self, visualizer):
        """测试变化事件标记生成"""
        # 添加一些变化事件
        if len(self.world_model.pending_changes) == 0:
            from brain.cognitive.world_model.environment_change import (
                EnvironmentChange, ChangeType, ChangePriority
            )
            self.world_model.pending_changes.append(
                EnvironmentChange(
                    change_type=ChangeType.NEW_OBSTACLE,
                    priority=ChangePriority.HIGH,
                    description="测试变化事件",
                    data={}
                )
            )

        try:
            markers = visualizer._generate_change_markers()
            if markers is not None and len(markers.markers) > 0:
                print(f"  ✅ change_markers: {len(markers.markers)} 个标记")
                self.results['change_markers'] = 'OK'
            else:
                self.warnings.append("change_markers为空")
                print(f"  ⚠️  change_markers: 为空")
                self.results['change_markers'] = 'EMPTY'
        except Exception as e:
            self.errors.append(f"change_markers生成错误: {e}")
            print(f"  ❌ change_markers错误: {e}")
            self.results['change_markers'] = 'ERROR'

    def _test_vlm_markers(self, visualizer):
        """测试VLM检测标记生成"""
        try:
            markers = visualizer._generate_vlm_markers()
            if markers is not None and len(markers.markers) > 0:
                print(f"  ✅ vlm_markers: {len(markers.markers)} 个标记")
                self.results['vlm_markers'] = 'OK'
            else:
                self.warnings.append("vlm_markers为空（需要VLM检测数据）")
                print(f"  ⚠️  vlm_markers: 为空（需要VLM检测数据）")
                self.results['vlm_markers'] = 'EMPTY'
        except Exception as e:
            self.errors.append(f"vlm_markers生成错误: {e}")
            print(f"  ❌ vlm_markers错误: {e}")
            self.results['vlm_markers'] = 'ERROR'

    def _generate_report(self):
        """生成诊断报告"""
        print("\n" + "=" * 80)
        print("📋 诊断报告")
        print("=" * 80)

        # 测试结果
        print("\n📊 测试结果:")
        for test, result in self.results.items():
            status_icon = '✅' if result == 'OK' else '⚠️' if 'EMPTY' in result else '❌'
            print(f"  {status_icon} {test}: {result}")

        # 错误和警告
        if self.errors:
            print(f"\n❌ 错误 ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\n⚠️  警告 ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        # 总体评估
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r == 'OK')
        failed_tests = sum(1 for r in self.results.values() if 'FAILED' in r or 'ERROR' in r)

        print(f"\n📈 总体评估:")
        print(f"  通过: {passed_tests}/{total_tests}")
        print(f"  失败: {failed_tests}/{total_tests}")

        if passed_tests == total_tests:
            print(f"  状态: ✅ 所有关键功能正常")
        elif passed_tests >= total_tests * 0.7:
            print(f"  状态: ⚠️  大部分功能正常")
        else:
            print(f"  状态: ❌ 存在严重问题")

        # 建议
        print(f"\n💡 建议:")
        if self.results['semantic_grid'] != 'OK':
            print(f"  • 检查current_map初始化和地图生成逻辑")
        if self.results['semantic_markers'] != 'OK':
            print(f"  • 检查semantic_objects数据来源")
        if self.results['trajectory'] != 'OK':
            print(f"  • 确保pose_history有数据")
        if not self.ros2_available:
            print(f"  • 安装并配置ROS2环境")

        # 导出报告
        self._export_report()

        print("\n" + "=" * 80)

    def _export_report(self):
        """导出诊断报告到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f".cursor/diagnostic_report_{timestamp}.txt"

        try:
            with open(filename, 'w') as f:
                f.write("WorldModel可视化诊断报告\n")
                f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                f.write("测试结果:\n")
                for test, result in self.results.items():
                    status_icon = '✅' if result == 'OK' else '⚠️' if 'EMPTY' in result else '❌'
                    f.write(f"  {status_icon} {test}: {result}\n")

                if self.errors:
                    f.write(f"\n错误 ({len(self.errors)}):\n")
                    for i, error in enumerate(self.errors, 1):
                        f.write(f"  {i}. {error}\n")

                if self.warnings:
                    f.write(f"\n警告 ({len(self.warnings)}):\n")
                    for i, warning in enumerate(self.warnings, 1):
                        f.write(f"  {i}. {warning}\n")

            logger.info(f"诊断报告已导出: {filename}")

        except Exception as e:
            logger.warning(f"导出诊断报告失败: {e}")


def main():
    """主函数"""
    diagnostic = WorldModelDiagnostic()
    diagnostic.run_diagnostic()


if __name__ == '__main__':
    main()
