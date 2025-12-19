#!/usr/bin/env python3
"""
Isaac Sim Web界面解决方案
提供多种方法来启用Isaac Sim的Web界面
"""

import subprocess
import time
import os

def solution_1_restart_with_web_support():
    """解决方案1: 重新启动容器并启用Web支持"""

    print("🔧 解决方案1: 重新启动带Web界面的Isaac Sim容器")

    # 停止当前容器
    subprocess.run(['docker', 'stop', 'isaac-sim-gui'], capture_output=True)
    subprocess.run(['docker', 'rm', 'isaac-sim-gui'], capture_output=True)

    # 创建配置文件
    config_content = """
[extensions]
"omni.services.transport.server.http" = {}
"omni.services.livestream.nvcf" = {}
"omni.kit.livestream.webrtc" = {}
"omni.kit.window.core" = {}

[livestream]
enabled = true
web_port = 8222
stream_port = 49001

[renderer]
raytracing.enabled = true
rtx.enabled = true

[app]
window.width = 1920
window.height = 1080
"""

    with open('./isaac_config.toml', 'w') as f:
        f.write(config_content)

    # 启动带Web支持的新容器
    cmd = [
        'docker', 'run', '-d', '--name', 'isaac-sim-web',
        '--gpus', 'all',
        '--runtime=nvidia',
        '--shm-size=16gb',
        '-e', 'ACCEPT_EULA=Y',
        '-e', 'OMNI_KIT_ACCEPT_EULA=Y',
        '-p', '8222:8222',
        '-p', '49001:49001',
        '-p', '49002:49002',
        '-v', f'{os.getcwd()}/isaac-sim-workspace:/workspace/isaac-sim',
        '-v', f'{os.path.expanduser("~")}/isaac-sim-cache/kit/cache:/root/.cache/kit',
        '-v', f'{os.path.expanduser("~")}/isaac-sim-cache/data:/root/.local/share/ov/data',
        '-v', f'{os.getcwd()}/isaac_config.toml:/isaac-sim/config/extra_config.toml',
        'nvcr.io/nvidia/isaac-sim:5.1.0',
        '/isaac-sim/isaac-sim.sh'
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0:
        print("✅ 新容器启动成功!")
        print("🌐 请访问: http://localhost:8222")
        return True
    else:
        print(f"❌ 容器启动失败: {result.stderr.decode()}")
        return False

def solution_2_enable_in_current_container():
    """解决方案2: 在当前容器中启用Web界面"""

    print("🔧 解决方案2: 在当前容器中启用Web界面")

    # 复制Web启用脚本到容器
    copy_cmd = [
        'docker', 'cp', 'enable_isaac_web.py',
        'isaac-sim-gui:/root/enable_isaac_web.py'
    ]

    result = subprocess.run(copy_cmd, capture_output=True)

    if result.returncode == 0:
        print("✅ 脚本已复制到容器")

        # 在容器中运行Web启用脚本
        exec_cmd = [
            'docker', 'exec', '-d', 'isaac-sim-gui',
            'python3', '/root/enable_isaac_web.py'
        ]

        result = subprocess.run(exec_cmd, capture_output=True)

        if result.returncode == 0:
            print("✅ Web界面启动脚本已执行")
            print("🌐 请在1分钟后访问: http://localhost:8222")
            return True
        else:
            print(f"❌ 脚本执行失败: {result.stderr.decode()}")

    return False

def solution_3_create_jupyter_access():
    """解决方案3: 通过Jupyter访问Isaac Sim"""

    print("🔧 解决方案3: 启带Jupyter的Isaac Sim容器")

    # 停止当前容器
    subprocess.run(['docker', 'stop', 'isaac-sim-gui'], capture_output=True)
    subprocess.run(['docker', 'rm', 'isaac-sim-gui'], capture_output=True)

    # 启动带Jupyter的容器
    cmd = [
        'docker', 'run', '-d', '--name', 'isaac-sim-jupyter',
        '--gpus', 'all',
        '--runtime=nvidia',
        '--shm-size=16gb',
        '-e', 'ACCEPT_EULA=Y',
        '-p', '8888:8888',
        '-p', '8222:8222',
        '-v', f'{os.getcwd()}/isaac-sim-workspace:/workspace/isaac-sim',
        '-v', f'{os.path.expanduser("~")}/isaac-sim-cache/kit/cache:/root/.cache/kit',
        'nvcr.io/nvidia/isaac-sim:5.1.0',
        'jupyter', 'lab', '--ip=0.0.0.0', '--port=8888', '--no-browser',
        '--NotebookApp.token=isaac2024', '--allow-root'
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0:
        print("✅ Jupyter容器启动成功!")
        print("🌐 请访问Jupyter: http://localhost:8888?token=isaac2024")
        print("🔧 在Jupyter中运行Isaac Sim代码")
        return True
    else:
        print(f"❌ Jupyter容器启动失败: {result.stderr.decode()}")
        return False

def main():
    """主函数 - 提供解决方案菜单"""

    print("🚨 Isaac Sim Web界面诊断结果:")
    print("❌ 当前容器运行在headless模式，没有Web界面")
    print("✅ 正在提供解决方案...")
    print("")

    solutions = [
        ("重新启动带Web界面的容器", solution_1_restart_with_web_support),
        ("在当前容器中启用Web界面", solution_2_enable_in_current_container),
        ("通过Jupyter访问Isaac Sim", solution_3_create_jupyter_access),
    ]

    print("请选择解决方案:")
    for i, (name, _) in enumerate(solutions, 1):
        print(f"{i}. {name}")

    print("4. 执行所有解决方案")
    print("")

def auto_solution():
    """自动尝试所有解决方案"""

    print("🤖 自动执行所有解决方案...")

    # 解决方案1
    if solution_1_restart_with_web_support():
        time.sleep(10)
        check_access()
        return

    # 解决方案2
    if solution_2_enable_in_current_container():
        time.sleep(30)
        check_access()
        return

    # 解决方案3
    if solution_3_create_jupyter_access():
        time.sleep(10)
        return

def check_access():
    """检查Web界面是否可访问"""

    import requests

    try:
        response = requests.get('http://localhost:8222', timeout=5)
        if response.status_code == 200:
            print("🎉 成功！Isaac Sim Web界面可访问!")
            print("🌐 访问地址: http://localhost:8222")
            return True
    except:
        pass

    try:
        response = requests.get('http://localhost:8888', timeout=5)
        if response.status_code == 200:
            print("🎉 成功！Isaac Sim Jupyter界面可访问!")
            print("🌐 访问地址: http://localhost:8888?token=isaac2024")
            return True
    except:
        pass

    return False

if __name__ == "__main__":
    auto_solution()