#!/usr/bin/env python3
"""
简化版真实ROS2数据测试

直接订阅并处理真实ROS2数据
"""

import sys
import os
import time
import asyncio
import numpy as np

sys.path.insert(0, '/media/yangyuhui/CODES1/Brain-Perception-Dev')

# 必须在import rclpy之前设置DOMAIN_ID
os.environ['ROS_DOMAIN_ID'] = '0'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as SensorImage

from brain.perception.detection.detector import ObjectDetector
from brain.perception.core.types import Position3D


class SimplePerceptionTest(Node):
    """简化的感知测试"""

    def __init__(self):
        super().__init__('simple_perception_test')

        self.detector = ObjectDetector(config={"mode": "fast"})
        self.frame_count = 0
        self.max_frames = 10
        self.results = []

        # 订阅RGB和深度图
        self.rgb_sub = self.create_subscription(SensorImage, '/rgb_test', self.callback, 10)
        self.depth_sub = self.create_subscription(SensorImage, '/depth_test', self.callback, 10)

        self.last_rgb = None
        self.last_depth = None

        self.get_logger().info("✅ 测试节点已启动")
        self.get_logger().info(f"📡 订阅: /rgb_test, /depth_test")
        self.get_logger().info(f"⏱️  将处理 {self.max_frames} 帧")

    def callback(self, msg):
        """统一回调处理"""
        if msg._type == 'sensor_msgs/msg/Image':
            # 简单转换（不使用cv_bridge）
            try:
                height = msg.height
                width = msg.width

                # 根据编码处理
                if msg.encoding in ['bgr8', 'rgb8', 'mono8']:
                    dtype = np.uint8
                elif msg.encoding in ['16UC1']:
                    dtype = np.uint16
                elif msg.encoding in ['32FC1']:
                    dtype = np.float32
                else:
                    dtype = np.uint8

                data = np.frombuffer(msg.data, dtype=dtype)

                # 重塑图像
                if msg.encoding in ['bgr8', 'rgb8']:
                    image = data.reshape((height, width, 3))
                else:
                    image = data.reshape((height, width))

                # 识别是RGB还是深度
                if 'rgb' in msg._type_name or msg.encoding in ['bgr8', 'rgb8']:
                    self.last_rgb = image
                else:
                    self.last_depth = image

                # 如果有RGB和深度，处理
                if self.last_rgb is not None and self.last_depth is not None:
                    self.process_frame()

            except Exception as e:
                self.get_logger().error(f"处理错误: {e}")

    def process_frame(self):
        """处理一帧"""
        if self.frame_count >= self.max_frames:
            return

        start = time.time()

        # 异步处理检测
        asyncio.run(self.detect_and_log())

        elapsed = time.time() - start
        self.results.append(elapsed)

        # 清空缓存
        self.last_rgb = None
        self.last_depth = None

    async def detect_and_log(self):
        """执行检测并记录"""
        try:
            detections = await self.detector.detect(self.last_rgb, self.last_depth)

            self.frame_count += 1

            print(f"\n{'─'*50}")
            print(f"帧 {self.frame_count}/{self.max_frames}")
            print(f"RGB: {self.last_rgb.shape}, 深度: {self.last_depth.shape}")
            print(f"检测到: {len(detections)} 个目标")
            print(f"耗时: {(time.time()-start)*1000:.1f}ms")

            for i, det in enumerate(detections):
                print(f"  [{i+1}] {det.object_type.value}: {det.confidence:.2f}")
                if det.position_3d:
                    print(f"      位置: ({det.position_3d.x:.2f}, {det.position_3d.y:.2f}, {det.position_3d.z:.2f})")

            if self.frame_count >= self.max_frames:
                print(f"\n✅ 测试完成！")
                self.print_summary()
                rclpy.shutdown()

        except Exception as e:
            print(f"检测错误: {e}")

    def print_summary(self):
        """打印摘要"""
        print(f"\n{'='*50}")
        print(f"📊 测试摘要")
        print(f"{'='*50}")
        print(f"处理帧数: {self.frame_count}")
        print(f"平均耗时: {np.mean(self.results)*1000:.1f}ms")
        print(f"平均FPS: {1.0/np.mean(self.results):.1f}")
        print(f"\n✅ 真实ROS2数据处理成功！")
        print(f"{'='*50}\n")


def main():
    """主函数"""
    rclpy.init()

    test = SimplePerceptionTest()

    print("\n" + "="*50)
    print("🚀 感知层真实数据测试")
    print("="*50)
    print("\n正在订阅 ROS2 topics (DOMAIN_ID=0)...")
    print("等待10帧数据...\n")

    try:
        rclpy.spin(test)
    except KeyboardInterrupt:
        print("\n测试被中断")
    finally:
        test.destroy_node()


if __name__ == '__main__':
    main()
