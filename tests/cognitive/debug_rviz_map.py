#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断OccupancyGrid消息发布

检查实际发布的map消息的元数据，确认坐标系和尺寸是否正确
"""
import sys
import os
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class MapDiagnostics(Node):
    """地图诊断节点"""

    def __init__(self):
        super().__init__('map_diagnostics')

        self.get_logger().info("=" * 80)
        self.get_logger().info("🔍 OccupancyGrid诊断工具")
        self.get_logger().info("=" * 80)

        # 订阅semantic_grid话题
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/world_model/semantic_grid',
            self.map_callback,
            10
        )

        self.get_logger().info("✅ 已订阅 /world_model/semantic_grid")
        self.get_logger().info("等待数据...")
        self.get_logger().info("=" * 80)

    def map_callback(self, msg: OccupancyGrid):
        """处理地图消息"""
        print("\n" + "=" * 80)
        print("📦 收到OccupancyGrid消息")
        print("=" * 80)

        # Header信息
        print(f"\n📋 Header:")
        print(f"   Frame ID: {msg.header.frame_id}")
        print(f"   Stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

        # Map info
        print(f"\n🗺️  Map Info:")
        print(f"   分辨率 (resolution): {msg.info.resolution} 米/格")
        print(f"   宽度 (width): {msg.info.width} 格")
        print(f"   高度 (height): {msg.info.height} 格")

        # 计算实际尺寸
        real_width = msg.info.width * msg.info.resolution
        real_height = msg.info.height * msg.info.resolution
        print(f"   实际尺寸: {real_width:.1f}m x {real_height:.1f}m")

        # Origin信息
        print(f"\n📍 Origin (地图原点):")
        print(f"   Position: x={msg.info.origin.position.x:.2f}, y={msg.info.origin.position.y:.2f}, z={msg.info.origin.position.z:.2f}")
        print(f"   Orientation: x={msg.info.origin.orientation.x:.4f}, y={msg.info.origin.orientation.y:.4f}, "
              f"z={msg.info.origin.orientation.z:.4f}, w={msg.info.origin.orientation.w:.4f}")

        # 计算地图边界
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        max_x = origin_x + real_width
        max_y = origin_y + real_height

        print(f"\n📏 地图边界 (世界坐标):")
        print(f"   X范围: [{origin_x:.2f}, {max_x:.2f}]")
        print(f"   Y范围: [{origin_y:.2f}, {max_y:.2f}]")

        # 数据统计
        if len(msg.data) > 0:
            data = list(msg.data)
            total = len(data)
            unknown = sum(1 for v in data if v == -1)
            free = sum(1 for v in data if v == 0)
            occupied = sum(1 for v in data if v == 100)

            print(f"\n📊 数据统计:")
            print(f"   总单元格: {total:,}")
            print(f"   未知 (-1): {unknown:,} ({100*unknown/total:.1f}%)")
            print(f"   空闲 (0): {free:,} ({100*free/total:.1f}%)")
            print(f"   占据 (100): {occupied:,} ({100*occupied/total:.1f}%)")

            # 检查是否有语义数据
            semantic = sum(1 for v in data if 101 <= v <= 199)
            if semantic > 0:
                print(f"   语义 (101-199): {semantic:,} ({100*semantic/total:.1f}%)")

            # 显示数据样本
            print(f"\n🔍 数据样本 (前20个值):")
            print(f"   {data[:20]}")

        # RViz配置建议
        print(f"\n💡 RViz配置建议:")
        print(f"   1. Fixed Frame 应设置为: {msg.header.frame_id}")
        print(f"   2. 地图中心世界坐标: ({(origin_x + real_width/2):.2f}, {(origin_y + real_height/2):.2f})")
        print(f"   3. 如果机器人位置约为(0,0)，则应显示在地图中心附近")

        # 检查是否有问题
        print(f"\n⚠️  诊断结果:")
        issues = []

        if msg.info.width == 0 or msg.info.height == 0:
            issues.append("❌ 地图尺寸为0！")

        if len(msg.data) != msg.info.width * msg.info.height:
            issues.append(f"❌ 数据长度不匹配！预期{msg.info.width * msg.info.height}，实际{len(msg.data)}")

        if msg.header.frame_id != "map":
            issues.append(f"⚠️  Frame ID不是'map'，而是'{msg.header.frame_id}'")

        if len(issues) == 0:
            print("   ✅ 未发现明显问题")
            print(f"   📌 如果RViz只显示1/4地图，可能原因:")
            print(f"      - RViz的Fixed Frame未设置为'map'")
            print(f"      - RViz相机位置不对，需要手动调整视角")
            print(f"      - 需要在RViz中点击'2D Pose Estimate'来重置视角")
        else:
            for issue in issues:
                print(f"   {issue}")

        print("\n" + "=" * 80 + "\n")


def main():
    rclpy.init()

    diagnostics = MapDiagnostics()

    try:
        rclpy.spin(diagnostics)
    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号")
    finally:
        diagnostics.destroy_node()
        rclpy.shutdown()
        print("\n✅ 诊断工具已关闭")


if __name__ == '__main__':
    main()
