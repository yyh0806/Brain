#!/usr/bin/env python3
"""
检查里程计话题状态
"""
import sys
import subprocess

def check_topic(topic_name):
    """检查话题状态"""
    print(f"\n检查话题: {topic_name}")
    print("-" * 50)
    
    # 检查话题是否存在
    result = subprocess.run(
        ["ros2", "topic", "info", topic_name],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ 话题不存在或无法访问")
        print(result.stderr)
        return False
    
    info = result.stdout
    print(info)
    
    # 检查发布者数量
    if "Publisher count: 0" in info:
        print("⚠️  警告: 没有发布者（Publisher count = 0）")
        print("   这意味着没有节点在发布数据到这个话题")
        return False
    elif "Publisher count:" in info:
        print("✓ 有发布者在发布数据")
        return True
    
    return None

if __name__ == "__main__":
    import os
    os.system("source /opt/ros/galactic/setup.bash")
    
    topics_to_check = [
        "/car3/local_odom",
        "/car3/odom",
        "/odom",
        "/car0/local_odom"
    ]
    
    print("=" * 50)
    print("里程计话题诊断工具")
    print("=" * 50)
    
    for topic in topics_to_check:
        check_topic(topic)
    
    print("\n" + "=" * 50)
    print("建议:")
    print("1. 如果所有话题都没有发布者，请检查仿真环境是否启动")
    print("2. 确认car3是否在仿真中激活")
    print("3. 检查仿真环境的配置，确认里程计话题名称")
    print("=" * 50)



