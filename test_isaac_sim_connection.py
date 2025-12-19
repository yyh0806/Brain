#!/usr/bin/env python3
"""
测试Isaac Sim Web界面连接
"""

import requests
import time

def test_isaac_sim_connection():
    """测试Isaac Sim Web界面连接"""

    print("🔍 测试Isaac Sim Web界面连接...")
    print("=" * 50)

    # 测试的端口
    ports = [49000, 49001, 49002]
    base_url = "http://localhost"

    for port in ports:
        url = f"{base_url}:{port}"
        print(f"\n测试端口 {port}: {url}")

        try:
            # 尝试连接
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                print(f"✅ 端口 {port} 连接成功!")
                print(f"   状态码: {response.status_code}")
                if response.text:
                    print(f"   响应长度: {len(response.text)} 字符")
            else:
                print(f"⚠️  端口 {port} 有响应但状态码: {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"❌ 端口 {port} 连接被拒绝")
        except requests.exceptions.Timeout:
            print(f"⏰ 端口 {port} 连接超时")
        except Exception as e:
            print(f"❌ 端口 {port} 连接错误: {e}")

    print("\n" + "=" * 50)
    print("📋 Isaac Sim 访问信息:")
    print("🌐 Web界面: http://localhost:49000")
    print("📹 Livestream: http://localhost:49001")
    print("🔧 API端口: http://localhost:49002")
    print("💡 如果端口无法访问，请等待Isaac Sim完全启动")

if __name__ == "__main__":
    test_isaac_sim_connection()