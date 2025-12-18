# 🤖 Claude Code 项目指南

## 📖 项目概述

**Brain** - 无人系统任务分解大脑，是一个先进的认知架构系统，结合了感知、融合、决策和执行模块，为无人系统提供智能化的任务管理和执行能力。

## 🏗️ 系统架构

### 核心模块结构
```
brain/
├── core/                    # 认知核心
│   ├── task_manager.py     # 任务调度器
│   ├── world_model/        # 世界模型
│   └── decision_engine/    # 决策引擎
├── perception/             # 感知系统
│   ├── sensors/           # 传感器管理
│   └── sensor_models/     # 传感器模型
├── fusion/                # 数据融合
├── communication/         # 通信系统
│   ├── command_queue.py   # 命令队列
│   └── ros2_interface.py  # ROS2接口
└── platforms/            # 平台适配
    ├── isaac_sim_interface.py  # Isaac Sim接口
    └── carla_interface.py      # Carla接口
```

### 支持系统
```
config/                    # 配置管理
├── environments/          # 环境配置
├── modules/              # 模块配置
└── platforms/            # 平台配置

data/                     # 数据存储
docs-reports-dev/         # 文档报告
```

## 🚀 快速开始

### 环境要求
- **Python**: 3.10+
- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **GPU**: NVIDIA GPU (可选，用于仿真环境)
- **内存**: 8GB+ RAM
- **存储**: 20GB+ 可用空间

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yyh0806/Brain.git
cd Brain
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate   # Windows
```

3. **安装依赖**
```bash
# 基础依赖
pip install -r requirements.txt

# CI/CD轻量依赖
pip install -r requirements-ci.txt
```

4. **配置环境**
```bash
# 复制配置模板
cp config/environments/development.yaml.template config/environments/development.yaml

# 编辑配置文件
nano config/environments/development.yaml
```

5. **运行系统**
```bash
python -m brain.main
```

## 🔧 Claude Code 集成

### 可用命令

在GitHub Issues或Pull Requests中使用`@claude`命令：

#### 基础命令
- `@claude help` - 显示帮助信息
- `@claude status` - 检查项目状态
- `@claude review` - 代码审查

#### 开发命令
- `@claude test` - 运行测试套件
- `@claude fix [issue]` - 修复问题
- `@claude implement [feature]` - 实现功能
- `@claude analyze` - 分析代码库

#### 高级功能
- `@claude optimize` - 性能优化
- `@claude document` - 生成文档
- `@claude security` - 安全检查

### 使用示例

```
@claude help
```
🤖 **Claude Code 可用命令**
- `@claude help` - 显示帮助信息
- `@claude status` - 检查项目状态和CI/CD
- `@claude review` - 代码审查和建议

```
@claude status
```
📊 **项目状态报告**
- CI/CD状态: ✅ 正常
- 最新提交: [commit_hash]
- 分支: master
- 工作流: [查看详情](link)

```
@claude fix the failing CI test
```
🔧 **正在修复CI测试问题**
- 分析测试失败原因
- 提供解决方案
- 自动应用修复（如果可能）

## 🔄 工作流程

### 开发流程

1. **创建功能分支**
```bash
git checkout -b feature/new-cognitive-module
```

2. **开发功能**
- 编写代码
- 添加测试
- 更新文档

3. **使用Claude协助**
```bash
# 在PR中使用Claude进行代码审查
@claude review my cognitive module implementation

# 请求帮助修复问题
@claude fix the import error in world_model.py

# 生成文档
@claude document the new perception module
```

4. **创建Pull Request**
- 填写PR模板
- 使用`@claude review`请求审查
- 等待CI/CD通过

5. **合并代码**
- 分支保护确保代码质量
- 自动化测试验证
- 代码审查通过

### CI/CD流程

#### 自动触发
- **Push**: 触发代码分析
- **PR**: 触发智能审查
- **Merge**: 触发部署检查

#### 手动触发
- **代码分析**: 手动运行深度分析
- **安全扫描**: 按需安全检查
- **性能测试**: 性能基准测试

## ⚙️ 配置管理

### 环境配置
```yaml
# config/environments/development.yaml
environment:
  name: "development"
  debug: true
  log_level: "DEBUG"

sensors:
  enabled: ["camera", "lidar", "imu"]
  config_path: "config/sensors/"

platforms:
  isaac_sim:
    enabled: false
    config: "config/platforms/isaac_sim.yaml"
```

### 模块配置
```yaml
# config/modules/perception.yaml
perception:
  sensor_fusion:
    algorithm: "kalman_filter"
    update_rate: 30.0

  object_detection:
    model: "yolov8"
    confidence_threshold: 0.7
```

### Claude配置
在GitHub仓库设置中配置以下secrets：
- `ANTHROPIC_BASE_URL`: API基础URL
- `ANTHROPIC_API_KEY`: Claude API密钥

## 🧪 测试策略

### 测试层级
1. **单元测试**: 测试单个模块
2. **集成测试**: 测试模块交互
3. **系统测试**: 测试完整流程
4. **仿真测试**: 测试虚拟环境

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest tests/perception/

# 生成覆盖率报告
pytest --cov=brain --cov-report=html
```

### 使用Claude协助测试
```
@claude test the perception module
```
🧪 **测试执行中**
- 运行感知模块测试
- 检查代码覆盖率
- 分析性能指标

## 📊 性能监控

### 关键指标
- **任务完成率**: 任务执行成功率
- **响应时间**: 系统响应延迟
- **资源使用**: CPU/内存/GPU使用率
- **错误率**: 系统错误频率

### 监控工具
- **Loguru**: 结构化日志
- **Prometheus**: 指标收集
- **Grafana**: 可视化面板

## 🛡️ 安全考虑

### 安全最佳实践
- ✅ 敏感信息使用环境变量
- ✅ 定期更新依赖项
- ✅ 启用分支保护
- ✅ 代码审查流程
- ✅ 自动化安全扫描

### 安全扫描
```bash
# 手动运行安全扫描
@claude security

# 检查敏感信息泄露
grep -r "password\|secret\|token" --include="*.py" .
```

## 🚀 部署指南

### Docker部署
```bash
# 构建镜像
docker build -t brain-system .

# 运行容器
docker run -p 8080:8080 brain-system
```

### Kubernetes部署
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brain-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brain-system
  template:
    metadata:
      labels:
        app: brain-system
    spec:
      containers:
      - name: brain
        image: brain-system:latest
        ports:
        - containerPort: 8080
```

## 📚 文档资源

### 内部文档
- [系统架构](docs/architecture.md)
- [API参考](docs/api.md)
- [配置指南](docs/configuration.md)
- [故障排除](docs/troubleshooting.md)

### 外部资源
- [Claude Code 文档](https://code.claude.com/docs)
- [GitHub Actions 指南](https://docs.github.com/actions)
- [Python最佳实践](https://peps.python.org/pep-0008/)

## 🤝 贡献指南

### 如何贡献
1. Fork项目
2. 创建功能分支
3. 编写代码和测试
4. 提交Pull Request
5. 使用`@claude review`请求审查

### 代码规范
- 遵循PEP 8编码规范
- 添加类型注解
- 编写文档字符串
- 保持测试覆盖率 > 80%

### 提交规范
```
feat: 添加新的认知模块
fix: 修复传感器数据处理错误
docs: 更新API文档
test: 添加感知模块单元测试
refactor: 重构任务调度器
```

## 🆘 故障排除

### 常见问题

**Q: Claude API调用失败**
```
A: 检查以下配置：
- ANTHROPIC_API_KEY是否正确设置
- ANTHROPIC_BASE_URL是否可访问
- API密钥是否有足够权限
```

**Q: 模块导入错误**
```
A: 确认以下事项：
- Python版本兼容性 (3.10+)
- 依赖项已正确安装
- PYTHONPATH配置正确
```

**Q: CI/CD工作流失败**
```
A: 检查以下方面：
- requirements.txt依赖冲突
- GitHub secrets配置
- 工作流语法错误
```

### 获取帮助
- 📖 查看文档: `docs/`目录
- 🤖 Claude协助: 使用`@claude help`
- 🐛 报告问题: 创建GitHub Issue
- 💬 社区讨论: 参与Discussions

---

## 📞 联系方式

- **项目维护者**: yyh0806
- **GitHub**: https://github.com/yyh0806/Brain
- **Claude Code**: 使用`@claude`在项目中获取协助

---

*本文档由 Claude Code 自动生成和维护 | 最后更新: $(date)*