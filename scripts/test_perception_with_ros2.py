#!/usr/bin/env python3
"""
使用真实 ROS2 topic 数据测试感知模块

功能:
1. 连接到 ROS2 (ROS_DOMAIN_ID=42)
2. 订阅传感器话题 (RGB相机、点云、IMU、里程计)
3. 测试 VLM 场景理解 (llava:7b)
4. 测试数据融合
5. 测试占据栅格地图生成
6. 输出测试结果和可视化

Author: Brain Development Team
Date: 2025-01-08
"""

import os
import sys
import asyncio
import time
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

# 设置 ROS_DOMAIN_ID
os.environ['ROS_DOMAIN_ID'] = '42'

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2

from loguru import logger

# 感知模块导入
from brain.perception.sensors.ros2_sensor_manager import (
    ROS2SensorManager,
    PerceptionData,
    SensorType
)
from brain.perception.understanding.vlm_perception import VLMPerception, DetectedObject
from brain.perception.mapping.occupancy_mapper import OccupancyMapper
from brain.perception.sensors.fusion import EKFPoseFusion, ObstacleDetector
from brain.utils.config import ConfigManager


class PerceptionTestNode(Node):
    """感知测试节点 - 订阅 ROS2 话题并测试感知功能"""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__('perception_test_node')
        
        # 加载配置
        self.config = ConfigManager(config_path)
        
        # CvBridge 用于图像转换
        self.cv_bridge = CvBridge()
        
        # 数据存储
        self.sensor_data = {
            'rgb_image': None,
            'depth_image': None,
            'pointcloud': None,
            'imu': None,
            'odom': None,
            'laser_scan': None
        }
        
        self.last_timestamps = {}
        self.test_results = []
        
        # 初始化感知组件
        self._init_perception_components()
        
        # 订阅话题
        self._setup_subscribers()
        
        logger.info("感知测试节点初始化完成")

    def _init_perception_components(self):
        """初始化感知组件"""
        try:
            # VLM 感知
            logger.info("初始化 VLM 感知...")
            self.vlm = VLMPerception(model="llava:7b")
            logger.success("VLM 感知初始化成功")
            
            # 占据栅格地图
            logger.info("初始化占据栅格地图...")
            self.mapper = OccupancyMapper(
                resolution=0.1,
                map_size=50.0
            )
            logger.success("占据栅格地图初始化成功")
            
            # 位姿融合
            logger.info("初始化位姿融合...")
            self.pose_fusion = EKFPoseFusion()
            logger.success("位姿融合初始化成功")
            
            # 障碍物检测
            logger.info("初始化障碍物检测...")
            self.obstacle_detector = ObstacleDetector()
            logger.success("障碍物检测初始化成功")
            
        except Exception as e:
            logger.error(f"感知组件初始化失败: {e}")
            raise

    def _setup_subscribers(self):
        """设置 ROS2 订阅者"""
        # RGB 相机
        self.create_subscription(
            Image,
            '/front_stereo_camera/left/image_raw',
            self.rgb_callback,
            10
        )
        logger.info("订阅 RGB 相机: /front_stereo_camera/left/image_raw")
        
        # 深度相机 (如果可用)
        self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        logger.info("订阅深度相机: /camera/depth/image_rect_raw")
        
        # 点云
        self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            self.pointcloud_callback,
            10
        )
        logger.info("订阅点云: /front_3d_lidar/lidar_points")
        
        # IMU
        self.create_subscription(
            Imu,
            '/chassis/imu',
            self.imu_callback,
            10
        )
        logger.info("订阅 IMU: /chassis/imu")
        
        # 里程计
        self.create_subscription(
            Odometry,
            '/chassis/odom',
            self.odom_callback,
            10
        )
        logger.info("订阅里程计: /chassis/odom")
        
        # 激光雷达扫描 (如果可用)
        self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )
        logger.info("订阅激光雷达: /scan")

    def rgb_callback(self, msg: Image):
        """RGB 相机回调"""
        try:
            # 转换为 OpenCV 格式
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.sensor_data['rgb_image'] = cv_image
            self.last_timestamps['rgb'] = time.time()
            
            logger.debug(f"收到 RGB 图像: {msg.width}x{msg.height}")
            
        except Exception as e:
            logger.error(f"RGB 图像转换失败: {e}")

    def depth_callback(self, msg: Image):
        """深度相机回调"""
        try:
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.sensor_data['depth_image'] = cv_depth
            self.last_timestamps['depth'] = time.time()
            
            logger.debug(f"收到深度图像: {msg.width}x{msg.height}")
            
        except Exception as e:
            logger.error(f"深度图像转换失败: {e}")

    def pointcloud_callback(self, msg: PointCloud2):
        """点云回调"""
        try:
            # 转换点云数据
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
            self.sensor_data['pointcloud'] = points
            self.last_timestamps['pointcloud'] = time.time()
            
            logger.debug(f"收到点云: {len(points)} 个点")
            
        except Exception as e:
            logger.error(f"点云处理失败: {e}")

    def imu_callback(self, msg: Imu):
        """IMU 回调"""
        try:
            self.sensor_data['imu'] = {
                'linear_acceleration': [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ],
                'angular_velocity': [
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ],
                'orientation': [
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w
                ],
                'timestamp': time.time()
            }
            self.last_timestamps['imu'] = time.time()
            
            logger.debug(f"收到 IMU 数据")
            
        except Exception as e:
            logger.error(f"IMU 数据处理失败: {e}")

    def odom_callback(self, msg: Odometry):
        """里程计回调"""
        try:
            self.sensor_data['odom'] = {
                'pose': {
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z,
                    'qx': msg.pose.pose.orientation.x,
                    'qy': msg.pose.pose.orientation.y,
                    'qz': msg.pose.pose.orientation.z,
                    'qw': msg.pose.pose.orientation.w
                },
                'twist': {
                    'linear_x': msg.twist.twist.linear.x,
                    'linear_y': msg.twist.twist.linear.y,
                    'linear_z': msg.twist.twist.linear.z,
                    'angular_x': msg.twist.twist.angular.x,
                    'angular_y': msg.twist.twist.angular.y,
                    'angular_z': msg.twist.twist.angular.z
                },
                'timestamp': time.time()
            }
            self.last_timestamps['odom'] = time.time()
            
            logger.debug(f"收到里程计数据: ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})")
            
        except Exception as e:
            logger.error(f"里程计数据处理失败: {e}")

    def laser_callback(self, msg: LaserScan):
        """激光雷达回调"""
        try:
            self.sensor_data['laser_scan'] = {
                'ranges': list(msg.ranges),
                'angles': [
                    msg.angle_min + i * msg.angle_increment
                    for i in range(len(msg.ranges))
                ],
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'timestamp': time.time()
            }
            self.last_timestamps['laser'] = time.time()
            
            logger.debug(f"收到激光扫描: {len(msg.ranges)} 个点")
            
        except Exception as e:
            logger.error(f"激光雷达数据处理失败: {e}")

    def has_sensor_data(self, sensor_name: str) -> bool:
        """检查是否有传感器数据"""
        data = self.sensor_data.get(sensor_name)
        return data is not None and len(data) > 0 if isinstance(data, (np.ndarray, list)) else data is not None

    def get_data_status(self) -> Dict[str, bool]:
        """获取数据状态"""
        return {
            'rgb': self.has_sensor_data('rgb_image'),
            'depth': self.has_sensor_data('depth_image'),
            'pointcloud': self.has_sensor_data('pointcloud'),
            'imu': self.has_sensor_data('imu'),
            'odom': self.has_sensor_data('odom'),
            'laser': self.has_sensor_data('laser_scan')
        }

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        """等待传感器数据"""
        logger.info(f"等待传感器数据 (超时 {timeout} 秒)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_data_status()
            if any(status.values()):
                logger.success(f"收到传感器数据: {status}")
                return True
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.1)
        
        logger.warning("等待传感器数据超时")
        return False


async def test_vlm_scene_understanding(
    node: PerceptionTestNode,
    test_duration: int = 60
) -> Dict[str, Any]:
    """测试 VLM 场景理解"""
    logger.info("=" * 80)
    logger.info("开始测试: VLM 场景理解")
    logger.info("=" * 80)
    
    test_results = {
        'test_name': 'VLM 场景理解',
        'start_time': datetime.now().isoformat(),
        'success': False,
        'results': [],
        'errors': []
    }
    
    try:
        # 等待 RGB 图像
        if not node.has_sensor_data('rgb_image'):
            logger.warning("没有 RGB 图像数据，跳过 VLM 测试")
            test_results['errors'].append("没有 RGB 图像数据")
            return test_results
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < test_duration:
            if node.has_sensor_data('rgb_image'):
                rgb_image = node.sensor_data['rgb_image']
                
                # 执行场景分析
                logger.info(f"执行 VLM 场景分析 (帧 {frame_count + 1})...")
                
                try:
                    # 场景描述
                    scene_description = await node.vlm.describe_scene(rgb_image)
                    logger.info(f"场景描述: {scene_description[:200]}...")
                    
                    # 查找物体
                    objects = await node.vlm.find_object(rgb_image, "obstacle")
                    logger.info(f"检测到 {len(objects)} 个障碍物")
                    
                    # 保存结果
                    test_results['results'].append({
                        'frame': frame_count,
                        'timestamp': datetime.now().isoformat(),
                        'scene_description': scene_description,
                        'detected_objects': len(objects),
                        'objects': [
                            {
                                'label': obj.label,
                                'confidence': obj.confidence,
                                'position': obj.position_description,
                                'description': obj.description
                            }
                            for obj in objects
                        ]
                    })
                    
                    frame_count += 1
                    
                    # 保存图像
                    output_dir = Path("/tmp/perception_test")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(output_dir / f"frame_{frame_count:04d}.jpg"),
                        rgb_image
                    )
                    
                except Exception as e:
                    logger.error(f"VLM 分析失败: {e}")
                    test_results['errors'].append(f"帧 {frame_count}: {str(e)}")
            
            # 等待下一帧
            await asyncio.sleep(3.5)  # VLM 分析间隔
            rclpy.spin_once(node, timeout_sec=0.1)
        
        test_results['success'] = len(test_results['errors']) == 0
        test_results['total_frames'] = frame_count
        test_results['end_time'] = datetime.now().isoformat()
        
        logger.success(f"VLM 测试完成: {frame_count} 帧, {len(test_results['errors'])} 个错误")
        
    except Exception as e:
        logger.error(f"VLM 测试异常: {e}")
        test_results['errors'].append(str(e))
        test_results['success'] = False
    
    return test_results


def test_sensor_fusion(node: PerceptionTestNode) -> Dict[str, Any]:
    """测试传感器数据融合"""
    logger.info("=" * 80)
    logger.info("开始测试: 传感器数据融合")
    logger.info("=" * 80)
    
    test_results = {
        'test_name': '传感器数据融合',
        'start_time': datetime.now().isoformat(),
        'success': False,
        'results': [],
        'errors': []
    }
    
    try:
        # 测试位姿融合
        if node.has_sensor_data('imu') and node.has_sensor_data('odom'):
            logger.info("测试 IMU + 里程计融合...")
            
            for i in range(10):
                # 更新里程计
                odom = node.sensor_data['odom']
                node.pose_fusion.update_odom({
                    'x': odom['pose']['x'],
                    'y': odom['pose']['y'],
                    'z': odom['pose']['z'],
                    'qx': odom['pose']['qx'],
                    'qy': odom['pose']['qy'],
                    'qz': odom['pose']['qz'],
                    'qw': odom['pose']['qw']
                })
                
                # 更新 IMU
                imu = node.sensor_data['imu']
                node.pose_fusion.update_imu(imu)
                
                # 获取融合位姿
                fused_pose = node.pose_fusion.get_pose()
                
                logger.info(f"融合位姿 {i+1}: "
                          f"({fused_pose.x:.3f}, {fused_pose.y:.3f}, {fused_pose.z:.3f})")
                
                test_results['results'].append({
                    'iteration': i,
                    'fused_pose': {
                        'x': fused_pose.x,
                        'y': fused_pose.y,
                        'z': fused_pose.z,
                        'roll': fused_pose.roll,
                        'pitch': fused_pose.pitch,
                        'yaw': fused_pose.yaw
                    }
                })
                
                time.sleep(0.1)
                rclpy.spin_once(node, timeout_sec=0.1)
            
            test_results['success'] = True
            logger.success("位姿融合测试完成")
        else:
            logger.warning("缺少 IMU 或里程计数据，跳过位姿融合测试")
            test_results['errors'].append("缺少 IMU 或里程计数据")
        
        # 测试障碍物检测
        if node.has_sensor_data('depth_image'):
            logger.info("测试从深度图检测障碍物...")
            
            depth_image = node.sensor_data['depth_image']
            obstacles = node.obstacle_detector.detect_from_depth(depth_image)
            
            logger.info(f"从深度图检测到 {len(obstacles)} 个障碍物")
            test_results['results'].append({
                'obstacle_detection': {
                    'source': 'depth',
                    'count': len(obstacles),
                    'obstacles': obstacles[:5]  # 只保存前5个
                }
            })
            
            test_results['success'] = True
            logger.success("障碍物检测测试完成")
        
        if node.has_sensor_data('laser_scan'):
            logger.info("测试从激光雷达检测障碍物...")
            
            laser = node.sensor_data['laser_scan']
            obstacles = node.obstacle_detector.detect_from_laser(
                laser['ranges'],
                laser['angles']
            )
            
            logger.info(f"从激光雷达检测到 {len(obstacles)} 个障碍物")
            test_results['results'].append({
                'obstacle_detection': {
                    'source': 'laser',
                    'count': len(obstacles),
                    'obstacles': obstacles[:5]
                }
            })
            
            test_results['success'] = True
            logger.success("激光雷达障碍物检测测试完成")
        
        test_results['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"传感器融合测试异常: {e}")
        test_results['errors'].append(str(e))
        test_results['success'] = False
    
    return test_results


def test_occupancy_mapping(node: PerceptionTestNode) -> Dict[str, Any]:
    """测试占据栅格地图生成"""
    logger.info("=" * 80)
    logger.info("开始测试: 占据栅格地图生成")
    logger.info("=" * 80)
    
    test_results = {
        'test_name': '占据栅格地图',
        'start_time': datetime.now().isoformat(),
        'success': False,
        'results': [],
        'errors': []
    }
    
    try:
        # 从深度图更新
        if node.has_sensor_data('depth_image'):
            logger.info("从深度图更新占据地图...")
            
            depth_image = node.sensor_data['depth_image']
            
            # 假设相机位姿
            camera_pose = {
                'x': 0.0, 'y': 0.0, 'z': 0.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
            
            node.mapper.update_from_depth(depth_image, camera_pose)
            
            grid = node.mapper.get_map()
            occupied_cells = np.sum(grid.data == 100)
            free_cells = np.sum(grid.data == 0)
            unknown_cells = np.sum(grid.data == -1)
            
            logger.info(f"占据地图统计:")
            logger.info(f"  - 占据单元格: {occupied_cells}")
            logger.info(f"  - 自由单元格: {free_cells}")
            logger.info(f"  - 未知单元格: {unknown_cells}")
            logger.info(f"  - 总单元格: {grid.width * grid.height}")
            
            test_results['results'].append({
                'source': 'depth',
                'occupied_cells': int(occupied_cells),
                'free_cells': int(free_cells),
                'unknown_cells': int(unknown_cells),
                'total_cells': grid.width * grid.height
            })
            
            # 保存地图可视化
            output_dir = Path("/tmp/perception_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建可视化
            vis_image = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
            vis_image[grid.data == -1] = [128, 128, 128]  # 灰色 - 未知
            vis_image[grid.data == 0] = [255, 255, 255]    # 白色 - 自由
            vis_image[grid.data == 100] = [0, 0, 0]        # 黑色 - 占据
            
            cv2.imwrite(str(output_dir / "occupancy_map.png"), vis_image)
            logger.success(f"占据地图已保存到 {output_dir / 'occupancy_map.png'}")
            
            test_results['success'] = True
        
        # 从点云更新
        if node.has_sensor_data('pointcloud'):
            logger.info("从点云更新占据地图...")
            
            pointcloud = node.sensor_data['pointcloud']
            
            # 假设激光雷达位姿
            lidar_pose = {
                'x': 0.0, 'y': 0.0, 'z': 1.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
            }
            
            node.mapper.update_from_lidar(pointcloud, lidar_pose)
            
            grid = node.mapper.get_map()
            occupied_cells = np.sum(grid.data == 100)
            
            logger.info(f"从点云更新后占据单元格: {occupied_cells}")
            
            test_results['results'].append({
                'source': 'pointcloud',
                'occupied_cells': int(occupied_cells)
            })
            
            test_results['success'] = True
            logger.success("从点云更新地图完成")
        
        test_results['end_time'] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"占据栅格地图测试异常: {e}")
        test_results['errors'].append(str(e))
        test_results['success'] = False
    
    return test_results


def print_test_summary(results: List[Dict[str, Any]]):
    """打印测试总结"""
    logger.info("\n" + "=" * 80)
    logger.info("测试总结")
    logger.info("=" * 80)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n测试 {i}: {result['test_name']}")
        logger.info(f"  状态: {'✅ 成功' if result['success'] else '❌ 失败'}")
        logger.info(f"  开始时间: {result.get('start_time', 'N/A')}")
        logger.info(f"  结束时间: {result.get('end_time', 'N/A')}")
        
        if result.get('errors'):
            logger.warning(f"  错误: {len(result['errors'])} 个")
            for error in result['errors'][:3]:  # 只显示前3个
                logger.warning(f"    - {error}")
        
        if result.get('total_frames'):
            logger.info(f"  总帧数: {result['total_frames']}")
        
        if result.get('results'):
            logger.info(f"  结果数: {len(result['results'])}")


async def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    logger.info("=" * 80)
    logger.info("感知模块测试 - 使用真实 ROS2 数据")
    logger.info("=" * 80)
    logger.info(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', 'default')}")
    logger.info(f"Ollama 模型: llava:7b")
    logger.info("=" * 80)
    
    # 初始化 ROS2
    rclpy.init()
    
    try:
        # 创建测试节点
        config_path = "/media/yangyuhui/CODES1/Brain/config/nova_carter_ros2.yaml"
        node = PerceptionTestNode(config_path)
        
        # 等待传感器数据
        if not node.wait_for_data(timeout=30.0):
            logger.error("未收到任何传感器数据，请检查 ROS2 节点是否运行")
            return
        
        # 打印数据状态
        status = node.get_data_status()
        logger.info(f"\n传感器数据状态:")
        for sensor, available in status.items():
            logger.info(f"  {sensor}: {'✅' if available else '❌'}")
        
        # 运行测试
        test_results = []
        
        # 测试 1: VLM 场景理解
        vlm_results = await test_vlm_scene_understanding(node, test_duration=30)
        test_results.append(vlm_results)
        
        # 测试 2: 传感器融合
        fusion_results = test_sensor_fusion(node)
        test_results.append(fusion_results)
        
        # 测试 3: 占据栅格地图
        mapping_results = test_occupancy_mapping(node)
        test_results.append(mapping_results)
        
        # 打印总结
        print_test_summary(test_results)
        
        # 保存结果
        output_dir = Path("/tmp/perception_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        result_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        logger.success(f"测试结果已保存到: {result_file}")
        
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        logger.info("清理资源...")
        rclpy.shutdown()
        logger.info("测试完成")


if __name__ == '__main__':
    asyncio.run(main())

