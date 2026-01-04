#!/usr/bin/env python3
"""
ROS2本地回环模式测试脚本

验证本地回环配置是否正常工作：
- 测试节点发现
- 测试话题发布/订阅
- 测试服务通信
"""

import sys
import time
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from example_interfaces.srv import Trigger
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("✗ ROS2 (rclpy) 不可用，请先安装ROS2")
    sys.exit(1)


class TestPublisher(Node):
    """测试发布者"""
    def __init__(self):
        super().__init__('test_publisher')
        self.publisher_ = self.create_publisher(String, 'test_topic', 10)
        self.timer = self.create_timer(1.0, self.publish_callback)
        self.count = 0
        self.get_logger().info('测试发布者已启动')

    def publish_callback(self):
        msg = String()
        msg.data = f'Hello from publisher - Count: {self.count}'
        self.publisher_.publish(msg)
        self.count += 1
        self.get_logger().info(f'发布消息: {msg.data}')


class TestSubscriber(Node):
    """测试订阅者"""
    def __init__(self):
        super().__init__('test_subscriber')
        self.subscription = self.create_subscription(
            String,
            'test_topic',
            self.listener_callback,
            10
        )
        self.get_logger().info('测试订阅者已启动')
        self.message_count = 0

    def listener_callback(self, msg):
        self.message_count += 1
        self.get_logger().info(f'收到消息 #{self.message_count}: {msg.data}')


class TestService(Node):
    """测试服务"""
    def __init__(self):
        super().__init__('test_service')
        self.srv = self.create_service(
            Trigger,
            'test_service',
            self.service_callback
        )
        self.get_logger().info('测试服务已启动')

    def service_callback(self, request, response):
        response.success = True
        response.message = 'Service is working!'
        self.get_logger().info('服务请求已处理')
        return response


def print_test_header():
    """打印测试标题"""
    print("=" * 80)
    print(" ROS2 本地回环模式测试")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # 打印当前环境配置
    print("【环境配置】")
    print(f"  ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', '未设置')}")
    print(f"  RMW_IMPLEMENTATION: {os.environ.get('RMW_IMPLEMENTATION', '未设置')}")
    print(f"  CYCLONEDDS_URI: {os.environ.get('CYCLONEDDS_URI', '未设置')}")
    print(f"  ROS_LOCALHOST_ONLY: {os.environ.get('ROS_LOCALHOST_ONLY', '未设置')}")
    print("")

    # 检查配置文件
    cyclone_config = "config/dds/cyclonedds_local.xml"
    fastdds_config = "config/dds/local_loopback_profile.xml"

    print("【配置文件检查】")
    print(f"  CycloneDDS配置: {'✓' if os.path.exists(cyclone_config) else '✗'} {cyclone_config}")
    print(f"  FastDDS配置: {'✓' if os.path.exists(fastdds_config) else '✗'} {fastdds_config}")
    print("")


def check_local_loopback():
    """检查是否已配置为本地回环模式"""
    warnings = []

    # 检查域ID
    if os.environ.get('ROS_DOMAIN_ID') != '42':
        warnings.append("⚠ ROS_DOMAIN_ID未设置为42")

    # 检查RMW实现
    rmw = os.environ.get('RMW_IMPLEMENTATION')
    if rmw not in ['rmw_cyclonedds_cpp', 'rmw_fastrtps_cpp']:
        warnings.append(f"⚠ RMW_IMPLEMENTATION未设置或使用默认值: {rmw}")

    # 检查本地限制
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        warnings.append("⚠ ROS_LOCALHOST_ONLY未设置")

    # 检查DDS配置
    cyclone_uri = os.environ.get('CYCLONEDDS_URI')
    if rmw == 'rmw_cyclonedds_cpp' and not cyclone_uri:
        warnings.append("⚠ 使用CycloneDDS但未设置CYCLONEDDS_URI")

    fastdds_uri = os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE')
    if rmw == 'rmw_fastrtps_cpp' and not fastdds_uri:
        warnings.append("⚠ 使用FastDDS但未设置FASTRTPS_DEFAULT_PROFILES_FILE")

    return warnings


def test_discovery():
    """测试节点发现"""
    print("【测试1: 节点发现】")
    print("  等待5秒，列出所有节点...")
    print("")

    # 延迟让节点有时间发现彼此
    time.sleep(5)

    import subprocess
    try:
        result = subprocess.run(
            ['ros2', 'node', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        nodes = result.stdout.strip().split('\n') if result.stdout.strip() else []

        print(f"  发现的节点数: {len(nodes)}")
        if nodes:
            for node in nodes:
                print(f"    - {node}")
            print("  ✓ 节点发现正常")
        else:
            print("  ✗ 未发现任何节点")
            print("  提示: 确保ROS2守护进程已启动")
            print("  运行: ros2 daemon stop && ros2 daemon start")
    except Exception as e:
        print(f"  ✗ 节点发现测试失败: {e}")

    print("")


def test_topics():
    """测试话题通信"""
    print("【测试2: 话题发布/订阅】")
    print("  创建发布者和订阅者...")
    print("")

    rclpy.init()

    # 创建发布者和订阅者
    publisher = TestPublisher()
    subscriber = TestSubscriber()

    # 创建executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(publisher)
    executor.add_node(subscriber)

    # 运行测试
    print("  运行10秒...")
    try:
        start_time = time.time()
        while time.time() - start_time < 10:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n  测试被用户中断")
    finally:
        publisher.destroy_node()
        subscriber.destroy_node()
        rclpy.shutdown()

    print(f"  发布消息数: {publisher.count}")
    print(f"  接收消息数: {subscriber.message_count}")

    if subscriber.message_count >= 5:
        print("  ✓ 话题通信正常")
    else:
        print("  ✗ 话题通信异常（接收消息数不足）")

    print("")


def test_services():
    """测试服务通信"""
    print("【测试3: 服务通信】")
    print("  创建服务...")
    print("")

    rclpy.init()

    # 创建服务节点
    service_node = TestService()

    # 等待服务可用
    time.sleep(2)

    # 创建客户端
    client_node = rclpy.create_node('test_client')
    client = client_node.create_client(Trigger, 'test_service')

    print("  等待服务连接...")
    if not client.wait_for_service(timeout_sec=5.0):
        print("  ✗ 服务未在超时时间内可用")
        service_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()
        print("")
        return

    print("  发送服务请求...")
    request = Trigger.Request()
    future = client.call_async(request)

    # 等待响应
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(service_node)
    executor.add_node(client_node)

    try:
        rclpy.spin_until_future_complete(client_node, future, timeout_sec=5.0)
        response = future.result()

        if response:
            print(f"  服务响应: {response.message}")
            print(f"  成功标志: {response.success}")
            print("  ✓ 服务通信正常")
        else:
            print("  ✗ 服务返回空响应")
    except Exception as e:
        print(f"  ✗ 服务通信失败: {e}")
    finally:
        service_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()

    print("")


def main():
    """主测试函数"""
    print_test_header()

    # 检查配置
    warnings = check_local_loopback()
    if warnings:
        print("【配置警告】")
        for warning in warnings:
            print(f"  {warning}")
        print("  提示: 请运行 ./scripts/start_ros2_local.sh 加载本地回环配置")
        print("")
    else:
        print("【配置检查】")
        print("  ✓ 所有配置项正确")
        print("")

    # 询问是否继续测试
    choice = input("是否继续功能测试？(y/n): ").lower()
    if choice != 'y':
        print("测试已取消")
        return

    try:
        # 测试节点发现
        test_discovery()

        # 测试话题通信
        test_topics()

        # 测试服务通信
        test_services()

        # 测试总结
        print("=" * 80)
        print("【测试总结】")
        print("  本地回环模式功能测试完成")
        print("")
        print("  如果所有测试都通过，说明本地回环配置正常工作。")
        print("  如果测试失败，请检查:")
        print("  1. 是否已加载本地回环配置")
        print("  2. ROS2守护进程是否运行: ros2 daemon status")
        print("  3. 端口是否被占用: netstat -an | grep 7400")
        print("  4. 防火墙是否阻止本地通信")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()




