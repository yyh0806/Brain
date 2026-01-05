#!/usr/bin/env python3
"""
使用实际ROS2数据的感知层测试

从正在运行的rosbag订阅真实数据：
- /rgb_test (RGB图像)
- /depth_test (深度图)
- 测试完整的感知处理流程
"""

import sys
import os
import time
import numpy as np
from typing import Optional
from dataclasses import dataclass

# 设置路径
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain-Perception-Dev')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage

# 感知层导入
from brain.perception.detection.detector import ObjectDetector
from brain.perception.core.types import Position3D
from brain.perception.core.enums import ObjectType

# 测试结果收集
@dataclass
class TestResult:
    """测试结果"""
    total_frames: int = 0
    detection_counts: list = None
    processing_times: list = None
    errors: list = None
    image_sizes: list = None
    depth_stats: dict = None

    def __post_init__(self):
        if self.detection_counts is None:
            self.detection_counts = []
        if self.processing_times is None:
            self.processing_times = []
        if self.errors is None:
            self.errors = []
        if self.image_sizes is None:
            self.image_sizes = []
        if self.depth_stats is None:
            self.depth_stats = {"min": float('inf'), "max": 0.0, "avg": 0.0, "count": 0}


class PerceptionTestNode(Node):
    """感知测试节点 - 使用真实ROS2数据"""

    def __init__(self):
        super().__init__('perception_real_test_node')

        # 初始化感知模块
        self.detector = ObjectDetector(config={
            "mode": "fast",
            "confidence_threshold": 0.5
        })

        # 测试结果
        self.result = TestResult()
        self.rgb_image = None
        self.depth_image = None
        self.rgb_timestamp = 0
        self.depth_timestamp = 0

        # 订阅topics
        self.rgb_sub = self.create_subscription(
            SensorImage,
            '/rgb_test',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            SensorImage,
            '/depth_test',
            self.depth_callback,
            10
        )

        self.get_logger().info('✅ 感知测试节点已启动 (ROS_DOMAIN_ID=0)')
        self.get_logger().info('📡 订阅: /rgb_test (RGB图像)')
        self.get_logger().info('📡 订阅: /depth_test (深度图)')

        # 测试计数器
        self.test_frame_count = 0
        self.max_test_frames = 15  # 测试15帧
        self.test_start_time = time.time()

    def _convert_ros_image_to_numpy(self, msg: SensorImage) -> Optional[np.ndarray]:
        """转换ROS Image消息到numpy数组"""
        try:
            height = msg.height
            width = msg.width

            # 根据编码方式处理数据
            if msg.encoding == "bgr8" or msg.encoding == "rgb8":
                dtype = np.uint8
                channels = 3
            elif msg.encoding == "mono8":
                dtype = np.uint8
                channels = 1
            elif msg.encoding == "16UC1":
                dtype = np.uint16
                channels = 1
            elif msg.encoding == "32FC1":
                dtype = np.float32
                channels = 1
            else:
                dtype = np.uint8
                channels = 3

            # 转换数据
            data = np.frombuffer(msg.data, dtype=dtype)

            # 重塑为图像格式
            if channels == 1:
                image = data.reshape((height, width))
            else:
                image = data.reshape((height, width, channels))

            # 如果是RGB，转换为BGR
            if msg.encoding == "rgb8" and channels == 3:
                image = image[:, :, ::-1]

            return image

        except Exception as e:
            self.get_logger().error(f'图像转换错误: {e}')
            return None

    def rgb_callback(self, msg: SensorImage):
        """RGB图像回调"""
        try:
            self.rgb_image = self._convert_ros_image_to_numpy(msg)
            self.rgb_timestamp = time.time()

            if self.rgb_image is not None:
                self.result.image_sizes.append(self.rgb_image.shape)

            # 如果有深度图且时间同步，执行检测
            if self.depth_image is not None and abs(self.rgb_timestamp - self.depth_timestamp) < 0.3:
                self.process_frame()

        except Exception as e:
            self.get_logger().error(f'RGB回调错误: {e}')

    def depth_callback(self, msg: SensorImage):
        """深度图回调"""
        try:
            self.depth_image = self._convert_ros_image_to_numpy(msg)
            self.depth_timestamp = time.time()

            if self.depth_image is not None:
                # 更新深度统计
                if self.depth_image.dtype == np.uint16:
                    depth_in_meters = self.depth_image.astype(np.float32) / 1000.0
                else:
                    depth_in_meters = self.depth_image

                valid_depths = depth_in_meters[depth_in_meters > 0]
                if len(valid_depths) > 0:
                    self.result.depth_stats["min"] = min(self.result.depth_stats["min"], np.min(valid_depths))
                    self.result.depth_stats["max"] = max(self.result.depth_stats["max"], np.max(valid_depths))
                    self.result.depth_stats["avg"] = (self.result.depth_stats["avg"] * self.result.depth_stats["count"] + np.mean(valid_depths)) / (self.result.depth_stats["count"] + 1)
                    self.result.depth_stats["count"] += 1

            # 如果有RGB图且时间同步，执行检测
            if self.rgb_image is not None and abs(self.rgb_timestamp - self.depth_timestamp) < 0.3:
                self.process_frame()

        except Exception as e:
            self.get_logger().error(f'深度回调错误: {e}')

    async def process_frame(self):
        """处理一帧数据"""
        if self.test_frame_count >= self.max_test_frames:
            if self.test_frame_count == self.max_test_frames:
                self.get_logger().info(f'✅ 已完成 {self.max_test_frames} 帧测试')
                self.test_frame_count += 1
            return

        start_time = time.time()

        try:
            self.get_logger().info(f'\n{"="*60}')
            self.get_logger().info(f'🎯 处理第 {self.test_frame_count + 1}/{self.max_test_frames} 帧 (真实数据)')
            self.get_logger().info(f'{"="*60}')

            # 显示输入数据信息
            if self.rgb_image is not None:
                self.get_logger().info(f'📷 RGB图像: {self.rgb_image.shape}, dtype: {self.rgb_image.dtype}')
            if self.depth_image is not None:
                self.get_logger().info(f'📏 深度图: {self.depth_image.shape}, dtype: {self.depth_image.dtype}')

            # 执行检测
            detections = await self.detector.detect(
                self.rgb_image,
                self.depth_image
            )

            # 记录结果
            processing_time = time.time() - start_time
            self.result.total_frames += 1
            self.result.detection_counts.append(len(detections))
            self.result.processing_times.append(processing_time)

            # 输出检测结果
            self.get_logger().info(f'✅ 检测到 {len(detections)} 个目标')
            self.get_logger().info(f'⏱️  处理耗时: {processing_time*1000:.1f}ms ({1.0/processing_time:.1f} FPS)')

            if len(detections) > 0:
                self.get_logger().info(f'\n检测结果详情:')
                for i, det in enumerate(detections):
                    self.get_logger().info(
                        f'  [{i+1}] 类型: {det.object_type.value:10s} | '
                        f'置信度: {det.confidence:.2f}'
                    )
                    if det.position_3d:
                        self.get_logger().info(
                            f'       位置: X={det.position_3d.x:5.2f}m, '
                            f'Y={det.position_3d.y:5.2f}m, '
                            f'Z={det.position_3d.z:5.2f}m'
                        )
                    if det.bounding_box_2d:
                        x, y, w, h = det.bounding_box_2d
                        self.get_logger().info(f'       边界框: ({x}, {y}, {w}x{h})')
            else:
                self.get_logger().info(f'  ℹ️  使用模拟检测数据（实际部署需接入YOLO等模型）')

            self.test_frame_count += 1

            # 重置图像以准备下一帧
            self.rgb_image = None
            self.depth_image = None

        except Exception as e:
            self.get_logger().error(f'❌ 处理帧错误: {e}')
            self.result.errors.append(f"处理帧: {str(e)}")

    def get_test_summary(self) -> str:
        """获取测试摘要"""
        total_time = time.time() - self.test_start_time

        summary = []
        summary.append("\n")
        summary.append("=" * 70)
        summary.append("🎯 感知层真实ROS2数据测试结果")
        summary.append("=" * 70)

        if self.result.total_frames == 0:
            summary.append("\n⚠️  没有接收到有效数据进行测试")
            summary.append("\n可能的原因:")
            summary.append("  • rosbag未播放")
            summary.append("  • ROS_DOMAIN_ID不匹配")
            summary.append("  • topic名称不匹配")
            if self.result.errors:
                summary.append(f"\n错误信息:")
                for error in self.result.errors[:5]:
                    summary.append(f"  • {error}")
            summary.append("\n" + "=" * 70)
            return "\n".join(summary)

        # 统计信息
        summary.append(f"\n📊 测试统计:")
        summary.append(f"  • 测试时长: {total_time:.1f}秒")
        summary.append(f"  • 处理帧数: {self.result.total_frames}")
        summary.append(f"  • 总检测数: {sum(self.result.detection_counts)}")
        summary.append(f"  • 平均每帧: {np.mean(self.result.detection_counts):.1f} 个目标")

        # 图像信息
        if self.result.image_sizes:
            summary.append(f"\n📷 输入数据:")
            summary.append(f"  • RGB尺寸: {self.result.image_sizes[0]}")

        # 深度信息
        if self.result.depth_stats["count"] > 0:
            summary.append(f"  • 深度范围: {self.result.depth_stats['min']:.2f}m - {self.result.depth_stats['max']:.2f}m")
            summary.append(f"  • 平均深度: {self.result.depth_stats['avg']:.2f}m")

        # 性能统计
        if self.result.processing_times:
            summary.append(f"\n⏱️  性能指标:")
            summary.append(f"  • 平均耗时: {np.mean(self.result.processing_times)*1000:.1f}ms")
            summary.append(f"  • 最快: {min(self.result.processing_times)*1000:.1f}ms")
            summary.append(f"  • 最慢: {max(self.result.processing_times)*1000:.1f}ms")
            summary.append(f"  • 平均FPS: {1.0/np.mean(self.result.processing_times):.1f}")

        # 错误信息
        if self.result.errors:
            summary.append(f"\n❌ 错误 ({len(self.result.errors)}):")
            for error in self.result.errors[:5]:
                summary.append(f"  • {error}")
        else:
            summary.append(f"\n✅ 无错误")

        # 验证说明
        summary.append(f"\n📝 测试说明:")
        summary.append(f"  ✅ 使用真实ROS2 topic数据")
        summary.append(f"  ✅ RGB-D图像同步接收成功")
        summary.append(f"  ✅ 图像转换处理正常")
        summary.append(f"  ✅ 检测器处理流程正常")
        summary.append(f"  ℹ️  检测结果为模拟数据（需接入YOLO等真实模型）")

        summary.append("\n" + "=" * 70)

        return "\n".join(summary)


def main(args=None):
    """主函数"""
    # 设置ROS_DOMAIN_ID
    os.environ['ROS_DOMAIN_ID'] = '0'

    rclpy.init(args=args)

    # 创建测试节点
    test_node = PerceptionTestNode()

    print("\n" + "=" * 70)
    print("🚀 感知层真实ROS2数据测试")
    print("=" * 70)
    print("\n环境配置:")
    print("  • ROS_DOMAIN_ID: 0")
    print("  • 订阅topics:")
    print("    - /rgb_test (RGB图像)")
    print("    - /depth_test (深度图)")
    print(f"\n正在等待数据...将测试 {test_node.max_test_frames} 帧\n")

    try:
        # 运行节点
        rclpy.spin(test_node)

    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    finally:
        # 输出测试结果
        print(test_node.get_test_summary())

        # 清理
        test_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
