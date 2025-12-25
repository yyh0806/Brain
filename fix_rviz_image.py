#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 RViz2 图像显示问题

这个脚本会：
1. 检查图像话题是否正常
2. 提供修复步骤
3. 创建正确的配置
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from loguru import logger

print("""
================================================================================
 RViz2 图像显示修复工具
================================================================================
""")

# 初始化 ROS2
if not rclpy.ok():
    rclpy.init()

node = rclpy.create_node('image_fix_checker')

image_received = False
image_data = None

def image_callback(msg):
    global image_received, image_data
    image_received = True
    image_data = msg

print("1. 检查图像话题...")
img_sub = node.create_subscription(Image, '/front_stereo_camera/left/image_raw', image_callback, 10)

# 等待数据
print("   等待图像数据（5秒）...")
for i in range(50):
    rclpy.spin_once(node, timeout_sec=0.1)
    if image_received:
        break

if image_received and image_data:
    print(f"   ✓ 图像数据正常接收")
    print(f"      - 分辨率: {image_data.width}x{image_data.height}")
    print(f"      - 编码: {image_data.encoding}")
    print(f"      - 数据大小: {len(image_data.data)} 字节")
    print(f"      - 步长: {image_data.step}")
    
    print("\n" + "="*70)
    print("图像数据正常！问题可能在 RViz2 配置")
    print("="*70)
    print("\n修复步骤：")
    print("\n【方法 1】在 RViz2 中手动修复（推荐）")
    print("-"*70)
    print("1. 在左侧 Displays 面板找到 'RGB Camera'")
    print("2. 确保已勾选 'Enabled'（启用）")
    print("3. 展开 'RGB Camera' 查看参数：")
    print("   - Image Topic: /front_stereo_camera/left/image_raw")
    print("   - Transport Hint: raw")
    print("   - Normalize Range: false")
    print("   - Min Value: 0")
    print("   - Max Value: 255")
    print("4. 如果话题不对，点击话题选择器，选择正确的話題")
    print("5. 点击左下角的 'Image' 按钮打开图像面板")
    print("   或者菜单栏: Panels → RGB Camera")
    
    print("\n【方法 2】重新添加图像显示")
    print("-"*70)
    print("1. 在 Displays 面板点击 'Add' 按钮")
    print("2. 选择 'Image'（不是 Camera！）")
    print("3. 设置参数：")
    print("   - Name: RGB Camera")
    print("   - Image Topic: /front_stereo_camera/left/image_raw")
    print("   - Transport Hint: raw")
    print("   - Normalize Range: false")
    print("   - Min Value: 0")
    print("   - Max Value: 255")
    print("4. 勾选 'Enabled'")
    print("5. 点击左下角的 'Image' 按钮打开图像面板")
    
    print("\n【方法 3】使用新配置文件")
    print("-"*70)
    print("1. 关闭当前 RViz2")
    print("2. 运行: rviz2 -d config/rviz2/nova_carter_perfect.rviz")
    print("3. 点击左下角的 'Image' 按钮打开图像面板")
    
    print("\n【布局调整】缩小左侧面板")
    print("-"*70)
    print("1. 找到左侧 Displays 面板和 3D 视图之间的分割线")
    print("2. 鼠标移到分割线上，光标变成双向箭头")
    print("3. 按住左键向左拖动，缩小左侧面板")
    print("4. 建议宽度：200-250 像素")
    print("5. 调整好后，File → Save Config As... 保存配置")
    
else:
    print("   ✗ 未收到图像数据")
    print("\n可能原因：")
    print("  1. Isaac Sim 未运行或未按 Play")
    print("  2. ROS2 Bridge 未启用")
    print("  3. 话题名称不匹配")
    print("\n检查命令：")
    print("  ros2 topic list | grep camera")
    print("  ros2 topic echo /front_stereo_camera/left/image_raw --once")

node.destroy_node()
rclpy.shutdown()

print("\n" + "="*70)
print("修复完成！")
print("="*70)






