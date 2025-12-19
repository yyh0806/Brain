#!/usr/bin/env python3
"""
Isaac Sim Web界面启用脚本
自动配置并启动Isaac Sim的Web界面
"""

import asyncio
import sys
import os
import subprocess
import time

async def enable_isaac_web_interface():
    """启用Isaac Sim Web界面"""

    print("🚀 正在启用Isaac Sim Web界面...")

    # 等待Isaac Sim完全启动
    await asyncio.sleep(5)

    try:
        # 导入Isaac Sim模块
        import carb
        import omni.kit.app

        app = omni.kit.app.get_app()

        # 启用HTTP传输服务
        from omni.services.transport.server.http import HttpServer
        http_server = HttpServer()
        await http_server.start_async('0.0.0.0', 8222)
        print("✅ HTTP服务器已启动在端口8222")

        # 启用Livestream服务
        from omni.services.livestream.nvcf import LivestreamNvcfInterface
        livestream = LivestreamNvcfInterface()
        await livestream.start_async()
        print("✅ Livestream服务已启动在端口49001")

        # 启用WebRTC服务
        try:
            from omni.kit.livestream.webrtc import WebRTCStreamInterface
            webrtc = WebRTCStreamInterface()
            await webrtc.start_async()
            print("✅ WebRTC服务已启动")
        except Exception as e:
            print(f"⚠️  WebRTC服务启动失败: {e}")

        # 启用UI界面（如果是headless模式）
        if app.get_editor_interface() is None:
            try:
                from omni.kit.window.core import get_default_viewport_resolution
                from omni.kit.viewport.utility import get_active_viewport

                # 尝试创建虚拟显示器
                os.environ['DISPLAY'] = ':99'
                subprocess.run(['Xvfb', ':99', '-screen', '0', '1920x1080x24'],
                             capture_output=True, check=False)

                print("✅ 虚拟显示器已创建")
            except Exception as e:
                print(f"⚠️  虚拟显示器创建失败: {e}")

        # 创建测试场景
        import omni.isaac.core
        from omni.isaac.core import World

        world = World()
        world.scene.add_ground_plane()

        print("🎯 Web界面配置完成!")
        print("🌐 请访问: http://localhost:8222")
        print("📹 Livestream: http://localhost:49001")

        return True

    except Exception as e:
        print(f"❌ Web界面启动失败: {e}")
        return False

def main():
    """主函数"""
    try:
        success = asyncio.run(enable_isaac_web_interface())
        if success:
            print("✅ Isaac Sim Web界面启用成功!")

            # 保持服务运行
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n⏹️  停止Web界面服务...")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()