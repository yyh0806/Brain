#!/usr/bin/env python3
"""
测试不同domain ID的rosbag数据
"""
import os
import subprocess
import time

# 测试不同的domain IDs
domains_to_test = [0, 42]  # rosbag可能在这两个domain

for domain_id in domains_to_test:
    print(f"\n{'='*60}")
    print(f"测试 ROS_DOMAIN_ID={domain_id}")
    print(f"{'='*60}")

    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(domain_id)

    # 列出话题
    result = subprocess.run(
        ['ros2', 'topic', 'list'],
        env=env,
        capture_output=True,
        text=True,
        timeout=5
    )

    topics = result.stdout.strip()
    odom_topic = '/chassis/odom' in topics

    print(f"话题数量: {len(topics.splitlines())}")
    print(f"有/chassis/odom: {odom_topic}")

    if odom_topic:
        print(f"✅ rosbag在domain {domain_id}运行")
        print(f"\n建议: export ROS_DOMAIN_ID={domain_id}")
        break
else:
    print("\n❌ 在测试的domain IDs中都找不到rosbag数据")
    print("请确保rosbag正在运行")
