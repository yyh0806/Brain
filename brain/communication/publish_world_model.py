#!/usr/bin/env python3
"""
发布世界模型状态到ROS2话题

这个脚本为Brain系统添加世界模型话题发布功能，
让WebViz等可视化工具能够查看世界模型的详细状态。

功能：
- 发布占据栅格地图到 /map 话题
- 发布世界模型状态到 /world_model 话题
- 定期更新（可配置频率）
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String, Header
import numpy as np
from datetime import datetime
import json
from loguru import logger
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from brain.perception.world_model import WorldModel
    from brain.perception.data_models import Pose3D, Velocity
    WORLD_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入WorldModel: {e}")
    WORLD_MODEL_AVAILABLE = False


class WorldModelPublisher:
    """世界模型ROS2发布器"""
    
    def __init__(self, node_name: str = "world_model_publisher",
                 publish_rate: float = 10.0):
        """
        初始化世界模型发布器
        
        Args:
            node_name: ROS2节点名称
            publish_rate: 发布频率（Hz）
        """
        self.publish_rate = publish_rate
        
        # 创建ROS2节点
        self.node = Node(node_name)
        logger.info(f"ROS2节点 '{node_name}' 创建成功")
        
        # QoS配置 - 使用最佳努力策略
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST(10)
        )
        
        # 创建发布者
        self.map_publisher = self.node.create_publisher(
            OccupancyGrid,
            '/map',
            qos
        )
        
        self.world_model_publisher = self.node.create_publisher(
            String,
            '/world_model',
            qos
        )
        
        self.pointcloud_publisher = self.node.create_publisher(
            PointCloud2,
            '/world_model_pointcloud',
            qos
        )
        
        self.pose_publisher = self.node.create_publisher(
            PoseStamped,
            '/world_model_pose',
            qos
        )
        
        logger.info("所有话题发布者已创建")
        
        # 世界模型引用（将在运行时设置）
        self.world_model = None
        
        # 统计信息
        self.message_count = 0
        self.last_publish_time = datetime.now()
        
        logger.info(f"发布器初始化完成，发布频率: {publish_rate}Hz")
    
    def set_world_model(self, world_model: WorldModel):
        """设置世界模型引用"""
        self.world_model = world_model
        logger.info("世界模型引用已设置")
    
    def publish_occupancy_grid(self):
        """发布占据栅格地图"""
        if self.world_model is None:
            logger.warning("世界模型未设置，跳过地图发布")
            return
        
        try:
            # 获取占据地图
            grid = self.world_model.occupancy_mapper.get_grid()
            
            if grid is None or grid.data is None:
                logger.warning("占据地图为空")
                return
            
            # 创建OccupancyGrid消息
            occupancy_msg = OccupancyGrid()
            occupancy_msg.header = Header()
            occupancy_msg.header.stamp = self.node.get_clock().now().to_msg()
            occupancy_msg.header.frame_id = "map"
            
            # 设置地图信息
            occupancy_msg.info.map_load_time = 0.0
            occupancy_msg.info.resolution = self.world_model.occupancy_mapper.resolution
            occupancy_msg.info.width = grid.width
            occupancy_msg.info.height = grid.height
            occupancy_msg.info.origin.position.x = grid.origin_x
            occupancy_msg.info.origin.position.y = grid.origin_y
            
            # 转换栅格数据
            # CellState: UNKNOWN=-1, FREE=0, OCCUPIED=100
            # 映射到OccupancyGrid标准: UNKNOWN=255, FREE=0, OCCUPIED=100
            occupancy_data = grid.data.copy().astype(np.int8)
            
            # 转换CellState到OccupancyGrid标准
            from brain.perception.mapping.occupancy_mapper import CellState
            
            # 创建映射: -1(UNKNOWN) -> 255, 0(FREE) -> 0, 100(OCCUPIED) -> 100
            mask_unknown = occupancy_data == CellState.UNKNOWN
            mask_free = occupancy_data == CellState.FREE
            mask_occupied = occupancy_data == CellState.OCCUPIED
            
            occupancy_data[mask_unknown] = 255
            occupancy_data[mask_occupied] = 100
            # FREE保持为0
            
            occupancy_msg.data = occupancy_data.flatten().tolist()
            
            # 发布地图
            self.map_publisher.publish(occupancy_msg)
            
            logger.debug(f"占据地图已发布: {grid.width}x{grid.height}, "
                         f"occupied={np.sum(occupancy_data == 100)}, "
                         f"free={np.sum(occupancy_data == 0)}, "
                         f"unknown={np.sum(occupancy_data == 255)}")
            
        except Exception as e:
            logger.error(f"发布占据地图失败: {e}")
    
    def publish_world_model_state(self):
        """发布世界模型详细状态"""
        if self.world_model is None:
            logger.warning("世界模型未设置，跳过状态发布")
            return
        
        try:
            # 获取世界模型统计
            stats = self.world_model.get_map_statistics()
            
            # 获取语义物体
            semantic_objects_list = []
            for label, obj in self.world_model.semantic_objects.items():
                semantic_objects_list.append({
                    "label": label,
                    "confidence": float(obj.confidence),
                    "update_count": obj.update_count,
                    "last_seen": obj.last_seen.isoformat(),
                    "description": obj.description if obj.description else "",
                    "has_position": obj.spatial_position is not None
                })
            
            # 构建完整的状态字典
            world_model_state = {
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "created_at": self.world_model.metadata.created_at.isoformat(),
                    "last_updated": self.world_model.metadata.last_updated.isoformat(),
                    "update_count": int(self.world_model.metadata.update_count),
                    "confidence": float(self.world_model.metadata.confidence)
                },
                "occupancy_map": {
                    "resolution": self.world_model.occupancy_mapper.resolution,
                    "map_size": self.world_model.occupancy_mapper.map_size,
                    "grid_size": f"{stats['total_cells']} cells",
                    "occupied_cells": stats['occupied_cells'],
                    "free_cells": stats['free_cells'],
                    "unknown_cells": stats['unknown_cells'],
                    "coverage_ratio": stats['occupied_ratio'] + stats['free_ratio']
                },
                "semantic_objects": {
                    "total_count": len(semantic_objects_list),
                    "persistent_count": sum(1 for obj in semantic_objects_list if obj['update_count'] > 1),
                    "objects": semantic_objects_list
                },
                "spatial_relations": {
                    "count": len(self.world_model.spatial_relations),
                    "relations": [
                        {
                            "object1": rel.object1,
                            "object2": rel.object2,
                            "relation": rel.relation,
                            "confidence": float(rel.confidence),
                            "last_seen": rel.last_seen.isoformat()
                        }
                        for rel in self.world_model.spatial_relations
                    ]
                },
                "persistence": {
                    "decay_rate": self.world_model.semantic_decay,
                    "map_age_seconds": stats['map_age_seconds'],
                    "is_persistent": stats['update_count'] > 10
                }
            }
            
            # 转换为JSON字符串
            state_json = json.dumps(world_model_state, indent=2, ensure_ascii=False)
            
            # 创建消息
            msg = String()
            msg.data = state_json
            
            # 发布
            self.world_model_publisher.publish(msg)
            
            logger.debug(f"世界模型状态已发布: {len(semantic_objects_list)}个物体, "
                         f"confidence={world_model_state['metadata']['confidence']:.2%}")
            
        except Exception as e:
            logger.error(f"发布世界模型状态失败: {e}")
    
    def publish_pointcloud(self):
        """发布世界模型点云（可选）"""
        # 这个功能可以从OccupancyMapper中提取占据点
        # 暂时留空，因为点云数据量较大
        pass
    
    def publish(self):
        """发布所有世界模型数据"""
        self.publish_occupancy_grid()
        self.publish_world_model_state()
        self.message_count += 1
        self.last_publish_time = datetime.now()
    
    def spin(self):
        """发布循环"""
        logger.info(f"开始发布循环，频率: {self.publish_rate}Hz")
        
        rate = self.node.create_rate(self.publish_rate)
        
        try:
            while rclpy.ok():
                self.publish()
                rate.sleep()
                
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止发布")
        except Exception as e:
            logger.error(f"发布循环错误: {e}")
        finally:
            logger.info(f"总共发布了 {self.message_count} 条消息")
            logger.info("关闭ROS2节点")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="世界模型ROS2话题发布器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--node-name",
        type=str,
        default="world_model_publisher",
        help="ROS2节点名称"
    )
    
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="发布频率（Hz），默认10.0"
    )
    
    args = parser.parse_args()
    
    logger.info("========================================")
    logger.info("世界模型ROS2发布器")
    logger.info("========================================")
    logger.info(f"节点名称: {args.node_name}")
    logger.info(f"发布频率: {args.rate}Hz")
    
    if not WORLD_MODEL_AVAILABLE:
        logger.error("WorldModel模块不可用，无法运行")
        logger.error("请确保Brain系统正确安装")
        sys.exit(1)
    
    # 创建发布器
    publisher = WorldModelPublisher(
        node_name=args.node_name,
        publish_rate=args.rate
    )
    
    # 注意：这里world_model需要从Brain主系统传入
    # 实际使用中，这个发布器应该集成到brain/brain.py中
    # 这里单独运行只能发布空数据
    
    logger.warning("世界模型发布器已创建，但未连接到实际世界模型")
    logger.warning("请将此发布器集成到Brain主系统中")
    logger.info("开始发布循环（将发布空数据）...")
    
    # 开始发布
    publisher.spin()


if __name__ == "__main__":
    main()



