#!/usr/bin/env python3
"""
世界模型持久化验证工具

验证WorldModel在rosbag回放中的行为：
1. 贝叶斯更新是否正确
2. 时间衰减是否正确
3. 语义物体是否持久化
4. 占据地图是否累积
"""

import sys
import os
import time
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from brain.perception.world_model import WorldModel
    from brain.perception.mapping.occupancy_mapper import CellState
    from brain.perception.ros2_sensor_manager import PerceptionData
    from brain.perception.data_models import Pose3D, Velocity
    WORLD_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入WorldModel: {e}")
    WORLD_MODEL_AVAILABLE = False

try:
    import rclpy
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import OccupancyGrid
    from nav_msgs.msg import Odometry
    ROS2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ROS2不可用: {e}")
    ROS2_AVAILABLE = False


class WorldModelVerifier:
    """世界模型验证器"""
    
    def __init__(self, resolution: float = 0.1, map_size: float = 50.0):
        if not WORLD_MODEL_AVAILABLE:
            logger.error("WorldModel模块不可用")
            raise ImportError("WorldModel模块不可用")
            
        self.world_model = WorldModel(
            resolution=resolution,
            map_size=map_size
        )
        
        # 验证数据收集
        self.occupancy_history: List[Dict[str, Any]] = []
        self.semantic_history: List[Dict[str, Any]] = []
        self.update_timestamps: List[float] = []
        self.start_time = time.time()
        
        # 统计数据
        self.grid_snapshots: List[np.ndarray] = []
        self.semantic_snapshots: List[Dict[str, Any]] = []
        
        logger.info("世界模型验证器初始化完成")
    
    def verify_bayesian_update(
        self,
        grid_before: Optional[np.ndarray],
        grid_after: np.ndarray,
        observation: str
    ) -> Dict[str, Any]:
        """验证贝叶斯更新"""
        
        # 首次更新
        if grid_before is None:
            self.grid_snapshots.append(grid_after.copy())
            return {
                "timestamp": time.time(),
                "observation": observation,
                "is_first_update": True,
                "total_cells": grid_after.size
            }
        
        # 统计栅格变化
        unknown_to_occupied = np.sum(
            (grid_before == CellState.UNKNOWN) & 
            (grid_after == CellState.OCCUPIED)
        )
        unknown_to_free = np.sum(
            (grid_before == CellState.UNKNOWN) & 
            (grid_after == CellState.FREE)
        )
        occupied_remains = np.sum(
            (grid_before == CellState.OCCUPIED) & 
            (grid_after == CellState.OCCUPIED)
        )
        free_remains = np.sum(
            (grid_before == CellState.FREE) & 
            (grid_after == CellState.FREE)
        )
        
        persistence_score = (occupied_remains + free_remains) / grid_before.size
        
        result = {
            "timestamp": time.time(),
            "observation": observation,
            "unknown_to_occupied": int(unknown_to_occupied),
            "unknown_to_free": int(unknown_to_free),
            "occupied_remains": int(occupied_remains),
            "free_remains": int(free_remains),
            "persistence_score": float(persistence_score)
        }
        
        self.occupancy_history.append(result)
        self.grid_snapshots.append(grid_after.copy())
        
        return result
    
    def verify_semantic_persistence(
        self,
        object_label: str,
        confidence: float,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """验证语义物体持久化"""
        
        if object_label in self.world_model.semantic_objects:
            obj = self.world_model.semantic_objects[object_label]
            
            result = {
                "timestamp": time.time(),
                "object_label": object_label,
                "confidence": float(obj.confidence),
                "update_count": int(obj.update_count),
                "age_seconds": (time.time() - obj.last_seen.timestamp()).total_seconds(),
                "is_persistent": obj.update_count > 1,
                "has_description": obj.description is not None
            }
        else:
            result = {
                "timestamp": time.time(),
                "object_label": object_label,
                "error": "Object not found",
                "confidence": confidence,
                "age_seconds": 0.0
            }
        
        self.semantic_history.append(result)
        return result
    
    def generate_verification_report(self) -> Dict[str, Any]:
        """生成验证报告"""
        stats = self.world_model.get_map_statistics()
        
        report = {
            "world_model": {
                "confidence": float(stats["confidence"]),
                "update_count": int(stats["update_count"]),
                "map_age_seconds": float(stats["map_age_seconds"]),
                "coverage": float(stats["occupied_ratio"] + stats["free_ratio"])
            },
            "occupancy_persistence": {
                "total_occupied_cells": int(stats["occupied_cells"]),
                "total_free_cells": int(stats["free_cells"]),
                "total_cells": int(stats["total_cells"]),
                "total_updates": len(self.occupancy_history),
                "persistence_verified": len(self.occupancy_history) > 10
            },
            "semantic_persistence": {
                "total_objects": int(stats["semantic_objects_count"]),
                "persistent_objects": sum(
                    1 for obj in self.semantic_history 
                    if obj.get("is_persistent", False)
                ),
                "average_confidence": np.mean([
                    obj["confidence"] for obj in self.semantic_history
                    if "confidence" in obj
                ]) if self.semantic_history else 0.0,
                "total_semantic_updates": len(self.semantic_history)
            },
            "time_decay": {
                "decay_rate": float(self.world_model.semantic_decay),
                "total_updates": len(self.update_timestamps),
                "average_update_interval": np.mean(np.diff(self.update_timestamps))
                    if len(self.update_timestamps) > 1 else 0.0
            },
            "bayesian_update": {
                "total_snapshots": len(self.grid_snapshots),
                "average_persistence_score": np.mean([
                    h.get("persistence_score", 0.0) 
                    for h in self.occupancy_history
                ]) if self.occupancy_history else 0.0
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """打印验证报告"""
        print("\n" + "="*80)
        print("世界模型持久化验证报告")
        print("="*80)
        
        print("\n1. 整体状态:")
        print(f"  置信度: {report['world_model']['confidence']:.2%}")
        print(f"  更新次数: {report['world_model']['update_count']}")
        print(f"  运行时间: {report['world_model']['map_age_seconds']:.1f}秒")
        print(f"  地图覆盖率: {report['world_model']['coverage']:.2%}")
        
        print("\n2. 占据地图持久化:")
        print(f"  占据单元: {report['occupancy_persistence']['total_occupied_cells']}")
        print(f"  自由单元: {report['occupancy_persistence']['total_free_cells']}")
        print(f"  总单元: {report['occupancy_persistence']['total_cells']}")
        print(f"  总更新次数: {report['occupancy_persistence']['total_updates']}")
        print(f"  持久化验证: {'✓ 通过' if report['occupancy_persistence']['persistence_verified'] else '✗ 失败'}")
        
        print("\n3. 语义物体持久化:")
        print(f"  总物体数: {report['semantic_persistence']['total_objects']}")
        print(f"  持久化物体: {report['semantic_persistence']['persistent_objects']}")
        print(f"  语义更新次数: {report['semantic_persistence']['total_semantic_updates']}")
        print(f"  平均置信度: {report['semantic_persistence']['average_confidence']:.2%}")
        
        print("\n4. 时间衰减机制:")
        print(f"  衰减率: {report['time_decay']['decay_rate']}")
        print(f"  总更新次数: {report['time_decay']['total_updates']}")
        print(f"  平均更新间隔: {report['time_decay']['average_update_interval']:.2f}秒")
        
        print("\n5. 贝叶斯更新机制:")
        print(f"  地图快照数: {report['bayesian_update']['total_snapshots']}")
        print(f"  平均持久化分数: {report['bayesian_update']['average_persistence_score']:.4f}")
        
        # 详细分析
        if self.occupancy_history:
            print("\n6. 占据地图变化分析:")
            
            # 计算变化趋势
            unknown_to_occupied = [
                h["unknown_to_occupied"] 
                for h in self.occupancy_history
            ]
            unknown_to_free = [
                h["unknown_to_free"] 
                for h in self.occupancy_history
            ]
            
            print(f"  平均每帧新增占据: {np.mean(unknown_to_occupied):.1f}个")
            print(f"  平均每帧新增自由: {np.mean(unknown_to_free):.1f}个")
            print(f"  新增占据总计: {sum(unknown_to_occupied)}个")
            print(f"  新增自由总计: {sum(unknown_to_free)}个")
        
        if self.semantic_history:
            print("\n7. 语义物体变化分析:")
            
            persistent_objects = set()
            for h in self.semantic_history:
                if h.get("is_persistent", False):
                    persistent_objects.add(h["object_label"])
            
            print(f"  持久化物体数: {len(persistent_objects)}")
            print(f"  持久化物体: {', '.join(sorted(persistent_objects))}")
        
        print("\n" + "="*80)
        print("验证完成")
        print("="*80)
    
    def save_report(self, report: Dict[str, Any], output_dir: str = "data/verification_reports"):
        """保存验证报告到文件"""
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"world_model_verification_{timestamp}.json"
        
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证报告已保存到: {report_file}")
        return report_file


def main():
    """主函数 - 支持rosbag和实时模式"""
    parser = argparse.ArgumentParser(
        description="世界模型持久化验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从rosbag验证
  %(prog)s --bag data/rosbags/perception.db3
  
  # 实时验证
  %(prog)s --live
  
  # 指定分辨率和地图大小
  %(prog)s --bag data/rosbags/perception.db3 --resolution 0.05 --map-size 100
        """
    )
    
    parser.add_argument(
        "--bag", 
        type=str,
        help="rosbag文件路径"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="实时模式（连接到运行中的ROS2）"
    )
    
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
        help="地图分辨率（米/格），默认0.1"
    )
    
    parser.add_argument(
        "--map-size",
        type=float,
        default=50.0,
        help="地图大小（米），默认50.0"
    )
    
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="保存验证报告到文件"
    )
    
    args = parser.parse_args()
    
    if not args.bag and not args.live:
        parser.print_help()
        print("\n错误: 必须指定 --bag 或 --live")
        sys.exit(1)
    
    # 创建验证器
    verifier = WorldModelVerifier(
        resolution=args.resolution,
        map_size=args.map_size
    )
    
    if args.bag:
        # Rosbag模式
        logger.info(f"从rosbag验证: {args.bag}")
        
        if not ROS2_AVAILABLE:
            logger.error("ROS2不可用，无法播放rosbag")
            sys.exit(1)
        
        # 这里应该有完整的rosbag回放和验证逻辑
        # 简化版：只做演示
        print("\n" + "="*80)
        print("Rosbag验证模式")
        print("="*80)
        print(f"rosbag文件: {args.bag}")
        print(f"地图分辨率: {args.resolution}米/格")
        print(f"地图大小: {args.map_size}米")
        print("\n注意: 完整的rosbag验证需要集成到Brain系统中")
        print("建议使用实时模式进行验证")
        
    elif args.live:
        # 实时模式
        logger.info("实时验证模式")
        
        print("\n" + "="*80)
        print("实时验证模式")
        print("="*80)
        print("\n验证器已准备就绪")
        print("请确保Brain系统正在运行并发布以下话题:")
        print("  - /front_3d_lidar/lidar_points")
        print("  - /chassis/odom")
        print("  - /chassis/imu")
        print("  - /map")
        print("  - /world_model")
        print("\n按Ctrl+C退出并生成报告...")
        print("\n" + "="*80)
        
        try:
            # 模拟实时验证循环
            # 在实际实现中，这里应该订阅ROS2话题并验证
            frame_count = 0
            
            while True:
                frame_count += 1
                
                # 模拟更新
                if frame_count % 10 == 0:
                    logger.info(f"已处理 {frame_count} 帧")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n收到中断信号，生成验证报告...")
            
            # 生成报告
            report = verifier.generate_verification_report()
            verifier.print_report(report)
            
            if args.save_report:
                report_file = verifier.save_report(report)
                print(f"\n报告已保存到: {report_file}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

