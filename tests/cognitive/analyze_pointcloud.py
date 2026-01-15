#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析点云数据的分布
"""
import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import struct
import numpy as np
import math

class PointCloudAnalyzer(Node):
    def __init__(self):
        super().__init__('pointcloud_analyzer')
        self.sub = self.create_subscription(
            PointCloud2,
            '/front_3d_lidar/lidar_points',
            self.callback,
            10
        )
        self.count = 0
        self.max_count = 3  # 只分析前3帧

    def callback(self, msg):
        if self.count >= self.max_count:
            return

        self.count += 1

        # 解析点云
        points = self._read_pointcloud(msg)

        if len(points) == 0:
            self.get_logger().warning(f"帧#{self.count}: 没有点云数据")
            return

        # 转换为numpy数组
        points_array = np.array(points)

        # 提取x, y（只用x,y做2D分析）
        x = points_array[:, 0]
        y = points_array[:, 1]
        z = points_array[:, 2]

        # 计算极坐标（角度和距离）
        angles = np.arctan2(y, x) * 180 / np.pi  # 转换为度数
        distances = np.sqrt(x**2 + y**2)

        print("\n" + "=" * 80)
        print(f"📊 点云分析 - 帧#{self.count}")
        print("=" * 80)

        # 基本信息
        print(f"\n基本信息:")
        print(f"   点云数量: {len(points)}")
        print(f"   Frame ID: {msg.header.frame_id}")
        print(f"   Point Step: {msg.point_step} bytes")

        # 坐标范围
        print(f"\n坐标范围:")
        print(f"   X: [{x.min():.2f}, {x.max():.2f}] 米")
        print(f"   Y: [{y.min():.2f}, {y.max():.2f}] 米")
        print(f"   Z: [{z.min():.2f}, {z.max():.2f}] 米")

        # 角度分布
        print(f"\n角度分布:")
        print(f"   最小角度: {angles.min():.1f}°")
        print(f"   最大角度: {angles.max():.1f}°")
        print(f"   角度范围: {angles.max() - angles.min():.1f}°")

        # 统计四个象限的点数
        q1 = np.sum((x >= 0) & (y >= 0))
        q2 = np.sum((x < 0) & (y >= 0))
        q3 = np.sum((x < 0) & (y < 0))
        q4 = np.sum((x >= 0) & (y < 0))

        total = len(points)
        print(f"\n象限分布:")
        print(f"   第一象限 (X≥0, Y≥0): {q1:,} ({100*q1/total:.1f}%)")
        print(f"   第二象限 (X<0, Y≥0): {q2:,} ({100*q2/total:.1f}%)")
        print(f"   第三象限 (X<0, Y<0): {q3:,} ({100*q3/total:.1f}%)")
        print(f"   第四象限 (X≥0, Y<0): {q4:,} ({100*q4/total:.1f}%)")

        # 判断是否是完整扫描
        angle_range = angles.max() - angles.min()
        if angle_range < 100:
            print(f"\n⚠️  问题: 点云只覆盖了 {angle_range:.1f}° 的扇形区域！")
            print(f"   这不是360度全景扫描，可能是:")
            print(f"   1. 前向雷达（只能看到前方）")
            print(f"   2. 点云数据经过了角度过滤")
            print(f"   3. 3D激光雷达的有限视场角（FOV）")
        elif angle_range > 300:
            print(f"\n✅ 点云覆盖接近360度 ({angle_range:.1f}°)")

        # 距离统计
        print(f"\n距离统计:")
        print(f"   最小距离: {distances.min():.2f} 米")
        print(f"   最大距离: {distances.max():.2f} 米")
        print(f"   平均距离: {distances.mean():.2f} 米")

        print("=" * 80)

        if self.count >= self.max_count:
            print("\n分析完成！")
            self.destroy_node()

    def _read_pointcloud(self, msg):
        """读取点云数据"""
        point_step = msg.point_step

        # 查找x, y, z字段的偏移
        x_offset = y_offset = z_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset

        if x_offset is None or y_offset is None or z_offset is None:
            return []

        # 解析点云
        points = []
        data = msg.data

        for i in range(0, len(data), point_step):
            if i + point_step > len(data):
                break

            try:
                x_bytes = data[i + x_offset:i + x_offset + 4]
                y_bytes = data[i + y_offset:i + y_offset + 4]
                z_bytes = data[i + z_offset:i + z_offset + 4]

                x = struct.unpack('f', x_bytes)[0]
                y = struct.unpack('f', y_bytes)[0]
                z = struct.unpack('f', z_bytes)[0]

                if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                    if not (np.isinf(x) or np.isinf(y) or np.isinf(z)):
                        points.append([x, y, z])
            except:
                continue

        return points

def main():
    rclpy.init()
    analyzer = PointCloudAnalyzer()
    print("等待点云数据...")
    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
