#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析占据栅格地图的数据分布
"""
import sys
sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

import os
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class MapChecker(Node):
    def __init__(self):
        super().__init__('map_checker')
        self.sub = self.create_subscription(OccupancyGrid, '/world_model/semantic_grid', self.callback, 10)
        self.received = False

    def callback(self, msg):
        if self.received:
            return
        self.received = True

        data = np.array(msg.data)
        data = data.reshape(msg.info.height, msg.info.width)

        # 找到所有非-1的值（有数据的地方）
        occupied_mask = data != -1
        occupied_indices = np.argwhere(occupied_mask)

        print("\n" + "=" * 80)
        print("🗺️  占据栅格数据分布分析")
        print("=" * 80)

        if len(occupied_indices) > 0:
            min_y, min_x = occupied_indices.min(axis=0)
            max_y, max_x = occupied_indices.max(axis=0)

            # 计算实际占据区域的大小
            height = max_y - min_y + 1
            width = max_x - min_x + 1

            print(f"\n📐 地图尺寸:")
            print(f"   总地图: {msg.info.width} x {msg.info.height} ({msg.info.width*msg.info.resolution:.1f}m x {msg.info.height*msg.info.resolution:.1f}m)")
            print(f"   有数据区域: {width} x {height} ({width*msg.info.resolution:.1f}m x {height*msg.info.resolution:.1f}m)")
            print(f"   占据比例: {100*width*height/(msg.info.width*msg.info.height):.1f}%")

            print(f"\n📍 栅格坐标范围:")
            print(f"   X: [{min_x}, {max_x}] (共{width}格)")
            print(f"   Y: [{min_y}, {max_y}] (共{height}格)")

            # 计算中心点
            center_x = msg.info.width // 2
            center_y = msg.info.height // 2
            data_center_x = (min_x + max_x) // 2
            data_center_y = (min_y + max_y) // 2

            print(f"\n🎯 中心点对比:")
            print(f"   地图中心: ({center_x}, {center_y})")
            print(f"   数据中心: ({data_center_x}, {data_center_y})")
            print(f"   偏移: ({data_center_x - center_x:+d}, {data_center_y - center_y:+d})")

            # 判断数据是否在角落
            is_corner = False
            corner_name = "中心区域"
            if data_center_x < center_x * 0.5 and data_center_y < center_y * 0.5:
                is_corner = True
                corner_name = "左下角"
            elif data_center_x > center_x * 1.5 and data_center_y < center_y * 0.5:
                is_corner = True
                corner_name = "右下角"
            elif data_center_x < center_x * 0.5 and data_center_y > center_y * 1.5:
                is_corner = True
                corner_name = "左上角"
            elif data_center_x > center_x * 1.5 and data_center_y > center_y * 1.5:
                is_corner = True
                corner_name = "右上角"

            print(f"\n📌 数据位置: {corner_name}")

            # 计算世界坐标
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            resolution = msg.info.resolution

            world_min_x = origin_x + min_x * resolution
            world_max_x = origin_x + max_x * resolution
            world_min_y = origin_y + min_y * resolution
            world_max_y = origin_y + max_y * resolution

            print(f"\n🌍 世界坐标范围:")
            print(f"   X: [{world_min_x:.2f}, {world_max_x:.2f}] 米 (宽度: {world_max_x - world_min_x:.2f}m)")
            print(f"   Y: [{world_min_y:.2f}, {world_max_y:.2f}] 米 (高度: {world_max_y - world_min_y:.2f}m)")

            # 检查是否需要调整地图原点
            if is_corner or width < msg.info.width * 0.3 or height < msg.info.height * 0.3:
                print(f"\n⚠️  问题: 数据只覆盖了地图的小部分区域！")
                print(f"   建议: 这可能是因为:")
                print(f"   1. 点云数据范围有限（机器人周围小范围）")
                print(f"   2. 机器人还没有大范围移动")
                print(f"   3. 点云过滤条件太严格")

        # 统计值
        unique, counts = np.unique(data, return_counts=True)
        print(f"\n📊 数据统计:")
        for val, count in zip(unique, counts):
            pct = 100 * count / data.size
            if val == -1:
                name = '未知  '
            elif val == 0:
                name = '空闲  '
            elif val == 100:
                name = '占据  '
            else:
                name = f'语义{val:3d}'
            print(f"   {name}: {count:>7,} ({pct:>5.1f}%)")

        print("\n" + "=" * 80)

def main():
    rclpy.init()
    checker = MapChecker()
    print("等待地图数据...")
    try:
        rclpy.spin_once(checker, timeout_sec=10.0)
    except KeyboardInterrupt:
        pass
    finally:
        checker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
