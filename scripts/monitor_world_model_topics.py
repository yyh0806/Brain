#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldModel话题监控工具

监控所有WorldModel可视化的ROS2话题，显示发布频率和数据大小。

Usage:
    export ROS_DOMAIN_ID=42
    python3 scripts/monitor_world_model_topics.py
"""

import os
import sys
import time
from datetime import datetime
from collections import defaultdict, deque

# 设置ROS域ID
os.environ['ROS_DOMAIN_ID'] = '42'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray


class WorldModelTopicMonitor(Node):
    """WorldModel话题监控节点"""

    def __init__(self):
        super().__init__('worldmodel_topic_monitor')

        self.get_logger().info("=" * 80)
        self.get_logger().info("📡 WorldModel话题监控器启动")
        self.get_logger().info("=" * 80)

        # 统计数据
        self.topic_stats = {
            '/world_model/semantic_grid': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            },
            '/world_model/semantic_markers': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            },
            '/world_model/trajectory': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            },
            '/world_model/frontiers': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            },
            '/world_model/belief_markers': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            },
            '/world_model/change_events': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            },
            '/world_model/vlm_markers': {
                'count': 0,
                'last_timestamp': None,
                'data_sizes': deque(maxlen=100),
                'periods': deque(maxlen=10)
            }
        }

        # QoS配置（使用最佳努力策略）
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 创建订阅者
        self._create_subscribers(qos)

        # 显示信息
        self.display_interval = 5.0  # 每5秒显示一次
        self.last_display_time = time.time()
        self.start_time = time.time()

        self.get_logger().info("✅ 监控器初始化完成")
        self.get_logger().info("")
        self.get_logger().info("监控的话题:")
        for topic in self.topic_stats.keys():
            self.get_logger().info(f"  - {topic}")
        self.get_logger().info("")
        self.get_logger().info("等待话题数据...")

    def _create_subscribers(self, qos):
        """创建所有话题订阅者"""

        # 1. 语义占据栅格
        self.create_subscription(
            OccupancyGrid,
            '/world_model/semantic_grid',
            lambda msg: self._semantic_grid_callback(msg),
            qos
        )

        # 2. 语义物体标注
        self.create_subscription(
            MarkerArray,
            '/world_model/semantic_markers',
            lambda msg: self._semantic_markers_callback(msg),
            qos
        )

        # 3. 机器人轨迹
        self.create_subscription(
            Path,
            '/world_model/trajectory',
            lambda msg: self._trajectory_callback(msg),
            qos
        )

        # 4. 探索前沿
        self.create_subscription(
            MarkerArray,
            '/world_model/frontiers',
            lambda msg: self._frontiers_callback(msg),
            qos
        )

        # 5. 信念状态
        self.create_subscription(
            MarkerArray,
            '/world_model/belief_markers',
            lambda msg: self._belief_markers_callback(msg),
            qos
        )

        # 6. 变化事件
        self.create_subscription(
            MarkerArray,
            '/world_model/change_events',
            lambda msg: self._change_events_callback(msg),
            qos
        )

        # 7. VLM检测
        self.create_subscription(
            MarkerArray,
            '/vlm/detections',
            lambda msg: self._vlm_markers_callback(msg),
            qos
        )

    def _update_stats(self, topic_name, data_size):
        """更新话题统计信息"""
        stats = self.topic_stats[topic_name]
        stats['count'] += 1
        stats['last_timestamp'] = datetime.now()

        # 记录数据大小
        stats['data_sizes'].append(data_size)

        # 计算发布周期
        if len(stats['periods']) > 0:
            period = (datetime.now() - stats['last_timestamp']).total_seconds()
            stats['periods'].append(period)

    def _semantic_grid_callback(self, msg: OccupancyGrid):
        """处理语义占据栅格"""
        data_size = len(msg.data)
        self._update_stats('/world_model/semantic_grid', data_size)

    def _semantic_markers_callback(self, msg: MarkerArray):
        """处理语义物体标注"""
        data_size = len(msg.markers)
        self._update_stats('/world_model/semantic_markers', data_size)

    def _trajectory_callback(self, msg: Path):
        """处理机器人轨迹"""
        data_size = len(msg.poses)
        self._update_stats('/world_model/trajectory', data_size)

    def _frontiers_callback(self, msg: MarkerArray):
        """处理探索前沿"""
        data_size = len(msg.markers)
        self._update_stats('/world_model/frontiers', data_size)

    def _belief_markers_callback(self, msg: MarkerArray):
        """处理信念状态"""
        data_size = len(msg.markers)
        self._update_stats('/world_model/belief_markers', data_size)

    def _change_events_callback(self, msg: MarkerArray):
        """处理变化事件"""
        data_size = len(msg.markers)
        self._update_stats('/world_model/change_events', data_size)

    def _vlm_markers_callback(self, msg: MarkerArray):
        """处理VLM检测"""
        data_size = len(msg.markers)
        self._update_stats('/vlm/detections', data_size)

    def _display_stats(self):
        """显示统计信息"""
        current_time = time.time()
        elapsed = current_time - self.start_time

        print("\n" + "=" * 80)
        print(f"📊 WorldModel话题统计 (运行: {elapsed:.1f}秒)")
        print("=" * 80)

        # 按话题显示
        for topic, stats in self.topic_stats.items():
            count = stats['count']
            rate = count / elapsed if elapsed > 0 else 0

            # 计算平均数据大小
            if stats['data_sizes']:
                avg_size = sum(stats['data_sizes']) / len(stats['data_sizes'])
                min_size = min(stats['data_sizes'])
                max_size = max(stats['data_sizes'])
            else:
                avg_size = min_size = max_size = 0

            # 计算平均发布周期
            if stats['periods']:
                avg_period = sum(stats['periods']) / len(stats['periods'])
                freq = 1.0 / avg_period if avg_period > 0 else 0
            else:
                avg_period = freq = 0

            # 状态标记
            if count == 0:
                status = "❌ 未收到数据"
            elif rate < 0.1:
                status = "⚠️  发布频率过低"
            elif rate > 5.0:
                status = "✅ 发布频率高"
            else:
                status = "✅ 正常"

            print(f"\n{status} {topic}")
            print(f"  总消息数: {count}")
            print(f"  平均频率: {rate:.2f} Hz (预期: 0.5-2.0 Hz)")
            print(f"  实际频率: {freq:.2f} Hz")
            print(f"  数据大小: 平均={avg_size:.1f}, 最小={min_size:.1f}, 最大={max_size:.1f}")
            print(f"  发布周期: {avg_period:.3f} 秒")
            print(f"  最后更新: {stats['last_timestamp']}")

        # 总体统计
        print(f"\n📈 总体统计:")
        total_messages = sum(s['count'] for s in self.topic_stats.values())
        active_topics = sum(1 for s in self.topic_stats.values() if s['count'] > 0)
        print(f"  总消息数: {total_messages}")
        print(f"  活跃话题: {active_topics}/{len(self.topic_stats)}")

        # RViz建议
        print(f"\n💡 RViz配置建议:")
        print(f"  Fixed Frame: map")
        print(f"  确保以下话题正确订阅:")
        for topic, stats in self.topic_stats.items():
            if stats['count'] == 0:
                print(f"    ⚠️  {topic} (未收到数据)")
            else:
                print(f"    ✅ {topic}")

        print("\n" + "=" * 80)

    def spin_once(self):
        """执行一次spin并显示统计"""
        rclpy.spin_once(self, timeout_sec=0.1)

        # 定期显示统计
        current_time = time.time()
        if current_time - self.last_display_time >= self.display_interval:
            self._display_stats()
            self.last_display_time = current_time


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    monitor = WorldModelTopicMonitor()

    try:
        print("\n" + "=" * 80)
        print("🚀 WorldModel话题监控器已启动")
        print("=" * 80)
        print("\n提示:")
        print("  • 按 Ctrl+C 停止")
        print("  • 每5秒显示一次统计信息")
        print("=" * 80 + "\n")

        while rclpy.ok():
            monitor.spin_once()

    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号，正在停止...")
    finally:
        # 显示最终统计
        monitor._display_stats()

        monitor.destroy_node()
        rclpy.shutdown()

        print("\n" + "=" * 80)
        print("✅ 监控器已停止")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
