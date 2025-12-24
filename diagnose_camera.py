#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断相机话题

检查相机话题的详细信息
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from loguru import logger


def diagnose_camera():
    """诊断相机话题"""
    
    if not rclpy.ok():
        rclpy.init()
    
    node = rclpy.create_node('camera_diagnostics')
    
    print("="*70)
    print("相机话题诊断")
    print("="*70)
    
    # 检查相机信息话题
    print("\n1. 检查相机信息话题...")
    
    camera_info_received = False
    camera_info = None
    
    def camera_info_callback(msg):
        nonlocal camera_info_received, camera_info
        camera_info_received = True
        camera_info = msg
    
    try:
        info_sub = node.create_subscription(
            CameraInfo,
            '/front_stereo_camera/left/camera_info',
            camera_info_callback,
            10
        )
        
        print("   等待相机信息...")
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
            if camera_info_received:
                break
        
        if camera_info_received and camera_info:
            print(f"   ✓ 相机信息已接收:")
            print(f"      - 宽度: {camera_info.width}")
            print(f"      - 高度: {camera_info.height}")
            print(f"      - 分辨率: {camera_info.width}x{camera_info.height}")
            print(f"      - 编码: {camera_info.distortion_model}")
            print(f"      - K矩阵: {camera_info.k.tolist()[:3]}")
        else:
            print("   ✗ 相机信息未接收")
    except Exception as e:
        print(f"   ✗ 相机信息检查失败: {e}")
    
    # 检查图像话题
    print("\n2. 检查图像话题...")
    
    image_received = False
    image_data = None
    
    def image_callback(msg):
        nonlocal image_received, image_data
        image_received = True
        image_data = msg
    
    try:
        img_sub = node.create_subscription(
            Image,
            '/front_stereo_camera/left/image_raw',
            image_callback,
            10
        )
        
        print("   等待图像数据...")
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
            if image_received:
                break
        
        if image_received and image_data:
            print(f"   ✓ 图像数据已接收:")
            print(f"      - 宽度: {image_data.width}")
            print(f"      - 高度: {image_data.height}")
            print(f"      - 分辨率: {image_data.width}x{image_data.height}")
            print(f"      - 编码: {image_data.encoding}")
            print(f"      - 步长: {image_data.step}")
            print(f"      - 数据大小: {len(image_data.data)} 字节")
            print(f"      - 大端: {image_data.is_bigendian}")
            
            # 验证数据完整性
            expected_size = image_data.height * image_data.step
            actual_size = len(image_data.data)
            
            if expected_size == actual_size:
                print(f"   ✓ 数据完整性验证通过")
            else:
                print(f"   ⚠ 数据大小不匹配: 预期 {expected_size}, 实际 {actual_size}")
            
            # 计算期望的像素数
            if image_data.encoding == "rgb8":
                bytes_per_pixel = 3
                expected_pixels = actual_size // bytes_per_pixel
                print(f"   ✓ RGB8 编码，期望像素数: {expected_pixels}")
            elif image_data.encoding == "bgr8":
                bytes_per_pixel = 3
                expected_pixels = actual_size // bytes_per_pixel
                print(f"   ✓ BGR8 编码，期望像素数: {expected_pixels}")
            elif image_data.encoding == "mono8":
                bytes_per_pixel = 1
                expected_pixels = actual_size // bytes_per_pixel
                print(f"   ✓ Mono8 编码，期望像素数: {expected_pixels}")
            else:
                print(f"   ⚠ 未知编码格式: {image_data.encoding}")
            
        else:
            print("   ✗ 图像数据未接收")
    except Exception as e:
        print(f"   ✗ 图像检查失败: {e}")
    
    # 检查压缩图像话题
    print("\n3. 检查压缩图像话题...")
    
    try:
        from sensor_msgs.msg import CompressedImage
        
        compressed_received = False
        compressed_data = None
        
        def compressed_callback(msg):
            nonlocal compressed_received, compressed_data
            compressed_received = True
            compressed_data = msg
        
        compressed_sub = node.create_subscription(
            CompressedImage,
            '/front_stereo_camera/left/image_raw/compressed',
            compressed_callback,
            10
        )
        
        print("   等待压缩图像...")
        for _ in range(50):
            rclpy.spin_once(node, timeout_sec=0.1)
            if compressed_received:
                break
        
        if compressed_received and compressed_data:
            print(f"   ✓ 压缩图像已接收:")
            print(f"      - 编码: {compressed_data.format}")
            print(f"      - 数据大小: {len(compressed_data.data)} 字节")
        else:
            print("   ℹ 压缩图像未发布（可能不需要）")
    except Exception as e:
        print(f"   ℹ 压缩图像检查失败（可能未发布）: {e}")
    
    print("\n" + "="*70)
    print("诊断完成")
    print("="*70)
    print("\nRViz2 配置建议：")
    print("  1. 使用 'Image' 显示，而不是 'Camera'")
    print("  2. Topic: /front_stereo_camera/left/image_raw")
    print("  3. Transport Hint: raw")
    print("  4. Normalize Range: false")
    print("  5. Min Value: 0, Max Value: 255")
    print("\n如果仍然不显示：")
    print("  - 检查 RViz2 的 'Camera Image' 面板是否打开")
    print("  - 在 Displays 面板中确保 'Camera Image' 已启用")
    print("  - 尝试禁用 'Stereo Camera' 显示（如果存在）")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    diagnose_camera()




