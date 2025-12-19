#!/usr/bin/env python3
"""
Isaac Sim 独立Web控制器
通过Flask创建Web界面，通过Docker exec控制Isaac Sim
"""

import subprocess
import threading
import time
import json
import os
from flask import Flask, render_template_string, jsonify, request
from datetime import datetime

app = Flask(__name__)

# Isaac Sim控制模板
ISAAK_CONTROL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Isaac Sim Web控制器</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }

        .header {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h2 {
            margin-bottom: 20px;
            color: #4CAF50;
            font-size: 1.5em;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 15px 0;
        }

        .status-item {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .status-ok {
            border-left: 4px solid #4CAF50;
        }

        .status-warning {
            border-left: 4px solid #ff9800;
        }

        .button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            margin: 5px;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }

        .button:hover {
            background: #45a049;
            transform: translateY(-2px);
        }

        .button-danger {
            background: #f44336;
        }

        .button-danger:hover {
            background: #d32f2f;
        }

        .console {
            background: #000;
            color: #0F0;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            height: 300px;
            overflow-y: auto;
            margin: 15px 0;
            font-size: 14px;
            line-height: 1.4;
        }

        .input-group {
            margin: 15px 0;
        }

        .input-group input {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-size: 16px;
            background: rgba(255,255,255,0.9);
            color: #333;
        }

        .alert {
            background: rgba(255,152,0,0.2);
            border-left: 4px solid #ff9800;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }

        .success {
            background: rgba(76,175,80,0.2);
            border-left: 4px solid #4CAF50;
        }

        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            .status-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Isaac Sim 5.1.0 Web控制器</h1>
        <p>通过Docker控制Isaac Sim仿真环境</p>
    </div>

    <div class="container">
        <!-- 系统状态 -->
        <div class="card">
            <h2>📊 系统状态</h2>
            <div class="status-grid">
                <div class="status-item status-ok">
                    <h3>容器状态</h3>
                    <p id="container-status">检查中...</p>
                </div>
                <div class="status-item status-ok">
                    <h3>GPU支持</h3>
                    <p>✅ NVIDIA CUDA</p>
                </div>
                <div class="status-item status-ok">
                    <h3>Isaac Sim版本</h3>
                    <p>5.1.0</p>
                </div>
                <div class="status-item status-warning">
                    <h3>Web界面</h3>
                    <p>独立控制器</p>
                </div>
            </div>
        </div>

        <!-- 控制面板 -->
        <div class="card">
            <h2>🎮 控制面板</h2>

            <div class="input-group">
                <input type="text" id="python-command" placeholder="输入Python命令..."
                       value="import omni.isaac.core; world = omni.isaac.core.World()">
            </div>

            <button class="button" onclick="executePython('python-command')">
                🐍 执行Python
            </button>
            <button class="button" onclick="executeExample('hello-world')">
                👋 Hello World
            </button>
            <button class="button" onclick="executeExample('create-scene')">
                🎬 创建场景
            </button>
            <button class="button" onclick="executeExample('add-robot')">
                🤖 添加机器人
            </button>

            <div id="alert-area"></div>
        </div>

        <!-- 控制台输出 -->
        <div class="card">
            <h2>📝 控制台输出</h2>
            <div id="console" class="console">
                > Isaac Sim Web控制器已启动<br>
                > 等待连接到Isaac Sim容器...<br>
                > 当前时间: {{ current_time }}<br>
                > 准备执行命令...<br>
            </div>

            <button class="button" onclick="clearConsole()">🗑️ 清空控制台</button>
            <button class="button" onclick="refreshStatus()">🔄 刷新状态</button>
        </div>

        <!-- 系统信息 -->
        <div class="card">
            <h2>🔧 系统信息</h2>
            <div class="status-grid">
                <div class="status-item">
                    <h3>Docker容器</h3>
                    <p id="docker-name">isaac-sim-ultimate</p>
                </div>
                <div class="status-item">
                    <h3>端口映射</h3>
                    <p>8222, 49001, 49002</p>
                </div>
                <div class="status-item">
                    <h3>内存使用</h3>
                    <p id="memory-usage">检查中...</p>
                </div>
                <div class="status-item">
                    <h3>GPU使用</h3>
                    <p id="gpu-usage">检查中...</p>
                </div>
            </div>

            <button class="button" onclick="showDockerLogs()">📋 查看日志</button>
            <button class="button button-danger" onclick="restartContainer()">🔄 重启容器</button>
        </div>

        <!-- 快速开始 -->
        <div class="card">
            <h2>🎯 快速开始</h2>
            <div class="alert success">
                <h3>✅ Isaac Sim已经运行!</h3>
                <p>您可以使用此Web界面控制Isaac Sim，或者通过以下方式直接访问：</p>
                <ul style="margin: 10px 0; padding-left: 20px;">
                    <li>Docker命令: <code>docker exec -it isaac-sim-ultimate python3</code></li>
                    <li>工作目录: <code>./isaac-sim-workspace/</code></li>
                </ul>
            </div>

            <button class="button" onclick="openWorkspace()">📁 打开工作目录</button>
            <button class="button" onclick="showExamples()">📚 查看示例</button>
        </div>
    </div>

    <script>
        function log(message) {
            const console = document.getElementById('console');
            const timestamp = new Date().toLocaleTimeString();
            console.innerHTML += `[${timestamp}] ${message}<br>`;
            console.scrollTop = console.scrollHeight;
        }

        function showAlert(message, type = 'info') {
            const alertArea = document.getElementById('alert-area');
            const alertClass = type === 'success' ? 'success' : 'alert';
            alertArea.innerHTML = `<div class="${alertClass}">${message}</div>`;
            setTimeout(() => {
                alertArea.innerHTML = '';
            }, 5000);
        }

        function executePython(inputId) {
            const command = document.getElementById(inputId).value;
            if (!command.trim()) {
                showAlert('请输入Python命令', 'warning');
                return;
            }

            log(`🐍 执行Python: ${command}`);

            fetch('/execute_python', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    log(`✅ 成功: ${data.output}`);
                    showAlert('命令执行成功', 'success');
                } else {
                    log(`❌ 错误: ${data.error}`);
                    showAlert('命令执行失败', 'warning');
                }
            })
            .catch(error => {
                log(`❌ 网络错误: ${error}`);
                showAlert('网络连接错误', 'warning');
            });
        }

        function executeExample(example) {
            const examples = {
                'hello-world': 'print("Hello from Isaac Sim!")',
                'create-scene': 'import omni.isaac.core; world = omni.isaac.core.World(); world.scene.add_ground_plane()',
                'add-robot': 'import omni.isaac.core; from omni.isaac.core import World; world = World(); print("机器人场景已准备")'
            };

            const command = examples[example];
            document.getElementById('python-command').value = command;
            executePython('python-command');
        }

        function clearConsole() {
            document.getElementById('console').innerHTML = '> 控制台已清空<br>';
        }

        function refreshStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('container-status').textContent = data.container_running ? '✅ 运行中' : '❌ 已停止';
                    log('🔄 状态已刷新');
                });
        }

        function showDockerLogs() {
            window.open('/logs', '_blank');
        }

        function restartContainer() {
            if (confirm('确定要重启Isaac Sim容器吗？这可能需要几分钟时间。')) {
                fetch('/restart_container', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            log('🔄 容器重启已启动');
                            showAlert('容器重启中，请稍候...', 'info');
                        } else {
                            log(`❌ 重启失败: ${data.error}`);
                        }
                    });
            }
        }

        function openWorkspace() {
            log('📁 工作目录: ./isaac-sim-workspace/');
            alert('工作目录位于: ./isaac-sim-workspace/');
        }

        function showExamples() {
            alert('Isaac Sim示例代码位于容器内的 /isaac-sim/apps/isaacsim/standalone_examples/');
        }

        // 定期刷新状态
        setInterval(refreshStatus, 10000);
        refreshStatus();

        log('🎉 Isaac Sim Web控制器已完全加载!');
        log('🌐 Web界面: http://localhost:5000');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页面"""
    return render_template_string(ISAAK_CONTROL_TEMPLATE,
                                 current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

@app.route('/status')
def status():
    """获取系统状态"""
    try:
        # 检查容器是否运行
        result = subprocess.run(['docker', 'ps', '--filter', 'name=isaac-sim-ultimate',
                               '--format', '{{.Status}}'], capture_output=True, text=True)
        container_running = bool(result.stdout.strip())

        return jsonify({
            'container_running': container_running,
            'status': 'running' if container_running else 'stopped',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/execute_python', methods=['POST'])
def execute_python():
    """执行Python命令"""
    try:
        data = request.json
        command = data.get('command', '')

        if not command:
            return jsonify({'success': False, 'error': '未提供命令'})

        # 在容器内执行Python命令
        exec_cmd = [
            'docker', 'exec', 'isaac-sim-ultimate',
            'python3', '-c', command
        ]

        result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            output = result.stdout.strip() if result.stdout.strip() else '命令执行成功'
            return jsonify({'success': True, 'output': output})
        else:
            error = result.stderr.strip() if result.stderr.strip() else '命令执行失败'
            return jsonify({'success': False, 'error': error})

    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': '命令执行超时'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/restart_container', methods=['POST'])
def restart_container():
    """重启容器"""
    try:
        # 重启容器
        subprocess.run(['docker', 'restart', 'isaac-sim-ultimate'], check=True)
        return jsonify({'success': True, 'message': '容器重启已启动'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logs')
def logs():
    """获取容器日志"""
    try:
        result = subprocess.run(['docker', 'logs', '--tail', '100', 'isaac-sim-ultimate'],
                               capture_output=True, text=True)
        logs = result.stdout
        return f"<pre>{logs}</pre>"
    except Exception as e:
        return f"错误获取日志: {str(e)}"

def run_isaac_sim_web_controller():
    """运行Isaac Sim Web控制器"""

    print("🚀 启动Isaac Sim独立Web控制器...")
    print("🌐 Web界面: http://localhost:5000")
    print("🔗 通过Flask控制Docker容器中的Isaac Sim")

    # 确保Isaac Sim容器正在运行
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=isaac-sim-ultimate'],
                               capture_output=True, text=True)

        if not result.stdout.strip():
            print("⚠️  Isaac Sim容器未运行，正在启动...")
            cmd = [
                'docker', 'run', '-d', '--name', 'isaac-sim-ultimate',
                '--gpus', 'all',
                '--runtime=nvidia',
                '--shm-size=16gb',
                '-e', 'ACCEPT_EULA=Y',
                '-p', '8222:8222',
                '-v', f'{os.getcwd()}/isaac-sim-workspace:/workspace/isaac-sim',
                'nvcr.io/nvidia/isaac-sim:5.1.0'
            ]
            subprocess.run(cmd, check=True)
            print("✅ Isaac Sim容器已启动")

    except Exception as e:
        print(f"❌ 容器操作错误: {e}")
        return

    print("🎯 Web控制器启动完成!")
    print("💡 提示: 这是一个独立Web界面，通过Docker exec控制Isaac Sim")

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    run_isaac_sim_web_controller()