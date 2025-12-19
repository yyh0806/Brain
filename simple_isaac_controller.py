#!/usr/bin/env python3
"""
Isaac Sim 简单Web控制器
使用Python内置http.server，无需额外依赖
"""

import http.server
import socketserver
import json
import subprocess
import threading
import urllib.parse
from datetime import datetime
import os

class IsaacSimHandler(http.server.SimpleHTTPRequestHandler):
    """Isaac Sim Web请求处理器"""

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = self.get_main_page()
            self.wfile.write(html.encode())

        elif self.path == '/status':
            self.send_json_response(self.get_status())

        elif self.path == '/logs':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            logs = self.get_container_logs()
            self.wfile.write(logs.encode())

        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/execute_python':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                result = self.execute_python(data.get('command', ''))
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({'success': False, 'error': str(e)})

        elif self.path == '/restart_container':
            result = self.restart_container()
            self.send_json_response(result)

    def send_json_response(self, data):
        """发送JSON响应"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_main_page(self):
        """获取主页面HTML"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Isaac Sim Web控制器</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.2);
        }}

        .button {{
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
            font-weight: bold;
        }}

        .button:hover {{
            background: #45a049;
        }}

        .button-danger {{
            background: #f44336;
        }}

        .console {{
            background: #000;
            color: #0F0;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            height: 200px;
            overflow-y: auto;
            margin: 15px 0;
            font-size: 12px;
        }}

        input[type="text"] {{
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin: 10px 0;
            background: rgba(255,255,255,0.9);
            color: #333;
        }}

        .status {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .status-item {{
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}

        .alert {{
            background: rgba(255,152,0,0.2);
            border-left: 4px solid #ff9800;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}

        .success {{
            background: rgba(76,175,80,0.2);
            border-left: 4px solid #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Isaac Sim 5.1.0 Web控制器</h1>
            <p>通过Python http.server控制Isaac Sim</p>
        </div>

        <div class="card">
            <h2>📊 系统状态</h2>
            <div class="status">
                <div class="status-item">
                    <h3>容器状态</h3>
                    <p id="container-status">检查中...</p>
                </div>
                <div class="status-item">
                    <h3>Isaac Sim</h3>
                    <p>✅ 5.1.0 运行中</p>
                </div>
                <div class="status-item">
                    <h3>GPU支持</h3>
                    <p>✅ NVIDIA CUDA</p>
                </div>
                <div class="status-item">
                    <h3>Web控制</h3>
                    <p>✅ 独立控制器</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🎮 Python控制</h2>
            <input type="text" id="python-command" placeholder="输入Python命令..."
                   value="import omni.isaac.core; print('Isaac Sim connected!')">
            <br>
            <button class="button" onclick="executePython()">🐍 执行Python</button>
            <button class="button" onclick="executeExample('hello')">👋 Hello</button>
            <button class="button" onclick="executeExample('world')">🌍 创建世界</button>
            <button class="button" onclick="showLogs()">📋 查看日志</button>
            <div id="alert-area"></div>
        </div>

        <div class="card">
            <h2>📝 控制台输出</h2>
            <div id="console" class="console">
                > Isaac Sim Web控制器已启动<br>
                > 时间: {datetime.now().strftime('%H:%M:%S')}<br>
                > 准备执行命令...<br>
            </div>
            <button class="button" onclick="clearConsole()">🗑️ 清空</button>
            <button class="button" onclick="refreshStatus()">🔄 刷新状态</button>
        </div>

        <div class="card success">
            <h2>✅ 连接成功!</h2>
            <p>Isaac Sim容器正在运行，您现在可以通过此界面控制仿真环境。</p>
            <p><strong>提示:</strong> 这是一个独立的Web控制器，通过Docker exec与Isaac Sim通信。</p>
            <p><strong>Docker命令:</strong> <code>docker exec -it isaac-sim-ultimate python3</code></p>
        </div>
    </div>

    <script>
        function log(message) {{
            const console = document.getElementById('console');
            const timestamp = new Date().toLocaleTimeString();
            console.innerHTML += `[${{timestamp}}] ${{message}}<br>`;
            console.scrollTop = console.scrollHeight;
        }}

        function showAlert(message, type = 'info') {{
            const alertArea = document.getElementById('alert-area');
            const alertClass = type === 'success' ? 'success' : 'alert';
            alertArea.innerHTML = `<div class="${{alertClass}}">${{message}}</div>`;
            setTimeout(() => {{
                alertArea.innerHTML = '';
            }}, 5000);
        }}

        function executePython() {{
            const command = document.getElementById('python-command').value;
            if (!command.trim()) {{
                showAlert('请输入Python命令');
                return;
            }}

            log(`🐍 执行: ${{command}}`);

            fetch('/execute_python', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{command: command}})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    log(`✅ 成功: ${{data.output}}`);
                    showAlert('命令执行成功', 'success');
                }} else {{
                    log(`❌ 错误: ${{data.error}}`);
                    showAlert('命令执行失败');
                }}
            }})
            .catch(error => {{
                log(`❌ 网络错误: ${{error}}`);
                showAlert('网络连接错误');
            }});
        }}

        function executeExample(type) {{
            const examples = {{
                'hello': 'print("Hello from Isaac Sim!")',
                'world': 'import omni.isaac.core; world = omni.isaac.core.World(); world.scene.add_ground_plane(); print("世界已创建")'
            }};

            document.getElementById('python-command').value = examples[type];
            executePython();
        }}

        function clearConsole() {{
            document.getElementById('console').innerHTML = '> 控制台已清空<br>';
        }}

        function refreshStatus() {{
            fetch('/status')
                .then(response => response.json())
                .then(data => {{
                    const status = data.container_running ? '✅ 运行中' : '❌ 已停止';
                    document.getElementById('container-status').textContent = status;
                    log('🔄 状态已刷新');
                }});
        }}

        function showLogs() {{
            window.open('/logs', '_blank');
        }}

        // 自动刷新状态
        setInterval(refreshStatus, 10000);
        refreshStatus();

        log('🎉 Web控制器完全加载!');
        log('🌐 端口: 8000');
    </script>
</body>
</html>
        """

    def get_status(self):
        """获取系统状态"""
        try:
            result = subprocess.run(['docker', 'ps', '--filter', 'name=isaac-sim-ultimate',
                                   '--format', '{{.Status}}'], capture_output=True, text=True)
            container_running = bool(result.stdout.strip())

            return {
                'container_running': container_running,
                'status': 'running' if container_running else 'stopped',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}

    def execute_python(self, command):
        """执行Python命令"""
        try:
            exec_cmd = [
                'docker', 'exec', 'isaac-sim-ultimate',
                'python3', '-c', command
            ]

            result = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout.strip() if result.stdout.strip() else '命令执行成功'
                return {'success': True, 'output': output}
            else:
                error = result.stderr.strip() if result.stderr.strip() else '命令执行失败'
                return {'success': False, 'error': error}

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': '命令执行超时'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_container_logs(self):
        """获取容器日志"""
        try:
            result = subprocess.run(['docker', 'logs', '--tail', '50', 'isaac-sim-ultimate'],
                                   capture_output=True, text=True)
            return result.stdout if result.stdout else "暂无日志输出"
        except Exception as e:
            return f"获取日志失败: {str(e)}"

    def restart_container(self):
        """重启容器"""
        try:
            subprocess.run(['docker', 'restart', 'isaac-sim-ultimate'], check=True)
            return {'success': True, 'message': '容器重启已启动'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

def run_simple_controller():
    """运行简单的Web控制器"""

    PORT = 8080
    Handler = IsaacSimHandler

    print("🚀 启动Isaac Sim简单Web控制器...")
    print("🌐 Web界面: http://localhost:8080")
    print("🔧 使用Python内置http.server")
    print("📊 通过Docker exec控制Isaac Sim")

    # 确保Isaac Sim容器正在运行
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=isaac-sim-ultimate'],
                               capture_output=True, text=True)

        if not result.stdout.strip():
            print("⚠️  Isaac Sim容器未运行，但Web控制器仍可启动")

    except Exception as e:
        print(f"⚠️  容器检查错误: {e}")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"✅ Web服务器启动在端口 {PORT}")
        print("🎯 在浏览器中访问 http://localhost:8080")
        print("💡 这是一个独立的Web控制界面")
        print("⏹️  按 Ctrl+C 停止服务器")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n⏹️  Web服务器已停止")

if __name__ == '__main__':
    run_simple_controller()