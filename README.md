# Brain - 智能无人系统任务规划核心

Brain是一个基于三层架构（感知层、认知层、规划层）的智能任务规划系统，专为无人系统设计，提供从自然语言理解到任务执行的全流程支持。

## 架构概述

Brain系统采用分层架构设计，将功能模块按照职责划分为感知层、认知层、规划层、执行层和通信层等，各层职责清晰，依赖关系明确。

```
┌─────────────────────────────────────────────────────────────────┐
│                   Brain 核心控制器                        │
├─────────────────────────────────────────────────────────────────┤
│  认知层 (Cognitive)                                  │
│  ├── 世界模型 (WorldModel)                              │
│  ├── 对话管理 (DialogueManager)                         │
│  ├── 推理引擎 (CoTEngine)                              │
│  └── 感知监控 (PerceptionMonitor)                      │
├─────────────────────────────────────────────────────────────────┤
│  规划层 (Planning)                                    │
│  ├── 任务规划 (TaskPlanner)                              │
│  ├── 导航规划 (NavigationPlanner)                        │
│  └── 行为规划 (BehaviorPlanner)                         │
├─────────────────────────────────────────────────────────────────┤
│  感知层 (Perception)                                  │
│  ├── 传感器管理 (SensorManager)                          │
│  ├── 环境感知 (EnvironmentPerception)                    │
│  ├── 地图构建 (Mapping)                                 │
│  └── VLM感知 (VLMPerception)                             │
├─────────────────────────────────────────────────────────────────┤
│  执行层 (Execution)                                    │
│  ├── 执行器 (Executor)                                   │
│  └── 操作库 (Operations)                                │
├─────────────────────────────────────────────────────────────────┤
│  通信层 (Communication)                                 │
│  ├── 机器人接口 (RobotInterface)                         │
│  ├── ROS2接口 (ROS2Interface)                            │
│  └── 控制适配器 (ControlAdapter)                        │
├─────────────────────────────────────────────────────────────────┤
│  模型层 (Models)                                       │
│  ├── LLM接口 (LLMInterface)                              │
│  ├── 提示模板 (PromptTemplates)                          │
│  └── 任务解析器 (TaskParser)                              │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
brain/
├── __init__.py
├── cognitive/               # 认知层 - 负责环境理解、世界建模和决策推理
│   ├── __init__.py
│   ├── dialogue/              # 对话管理
│   │   ├── __init__.py
│   │   └── dialogue_manager.py
│   ├── monitoring/            # 认知监控
│   │   ├── __init__.py
│   │   └── perception_monitor.py
│   ├── reasoning/             # 推理引擎
│   │   ├── __init__.py
│   │   └── cot_engine.py
│   └── world_model/           # 世界模型管理
│       ├── __init__.py
│       └── world_model.py
├── communication/           # 通信层 - 负责与外部系统通信
│   ├── __init__.py
│   ├── control_adapter.py     # 控制适配器
│   ├── message_types.py       # 消息类型定义
│   ├── robot_interface.py     # 机器人接口
│   └── ros2_interface.py     # ROS2接口
├── core/                   # 核心控制器
│   ├── __init__.py
│   ├── brain.py             # 主控制器
│   └── monitor.py           # 系统监控
├── execution/               # 执行层 - 负责指令执行和控制
│   ├── __init__.py
│   ├── executor.py           # 执行器
│   └── operations/          # 操作库
│       ├── __init__.py
│       ├── base.py
│       ├── drone.py
│       ├── ros2_ugv.py
│       ├── ugv.py
│       └── usv.py
├── models/                 # 模型层 - LLM和AI模型接口
│   ├── __init__.py
│   ├── cot_prompts.py       # CoT提示模板
│   ├── llm_interface.py     # LLM接口
│   ├── ollama_client.py     # Ollama客户端
│   ├── prompt_templates.py   # 提示模板
│   └── task_parser.py       # 任务解析器
├── perception/             # 感知层 - 负责所有传感器数据处理和环境感知
│   ├── __init__.py
│   ├── environment.py        # 环境模型感知
│   ├── mapping/             # 地图构建
│   │   ├── __init__.py
│   │   └── occupancy_mapper.py
│   ├── object_detector.py    # 物体检测
│   ├── sensors/             # 传感器管理
│   │   ├── __init__.py
│   │   ├── ros2_sensor_manager.py
│   │   ├── sensor_fusion.py
│   │   └── sensor_manager.py
│   └── vlm/                # 视觉语言模型感知
│       ├── __init__.py
│       └── vlm_perception.py
├── planning/               # 规划层 - 负责任务规划、路径规划和行为规划
│   ├── __init__.py
│   ├── behavior/            # 行为规划
│   │   └── __init__.py
│   ├── navigation/          # 导航规划
│   │   ├── __init__.py
│   │   ├── exploration_planner.py
│   │   ├── intersection_navigator.py
│   │   ├── local_planner.py
│   │   └── smooth_executor.py
│   └── task/               # 任务规划
│       ├── __init__.py
│       └── task_planner.py
├── platforms/              # 平台支持
│   ├── __init__.py
│   └── robot_capabilities.py
├── recovery/               # 错误恢复
│   ├── __init__.py
│   ├── error_handler.py
│   ├── replanner.py
│   └── rollback.py
├── state/                 # 状态管理
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── mission_state.py
│   └── world_state.py
├── utils/                 # 工具类
│   ├── __init__.py
│   └── config.py
└── visualization/          # 可视化
    ├── __init__.py
    └── rviz2_visualizer.py
```

## 各层职责

### 感知层 (Perception Layer)
- 负责所有传感器数据的采集和处理
- 进行数据融合和初步分析
- 构建和维护环境地图
- 提供环境感知的统一接口

### 认知层 (Cognitive Layer)
- 维护和更新世界模型
- 进行环境理解和语义分析
- 执行推理和决策过程
- 管理人机对话和交互

### 规划层 (Planning Layer)
- 进行任务分解和规划
- 生成导航路径和行为序列
- 处理动态重规划
- 协调不同规划子模块

### 执行层 (Execution Layer)
- 执行规划层生成的操作序列
- 管理不同平台的操作接口
- 处理执行过程中的状态反馈
- 提供统一的执行控制接口

### 通信层 (Communication Layer)
- 管理与外部系统的通信
- 提供统一的通信接口
- 处理不同通信协议的适配
- 管理消息类型和格式转换

### 模型层 (Models Layer)
- 提供LLM和AI模型的统一接口
- 管理提示模板和任务解析
- 处理模型配置和连接
- 提供模型能力的抽象

## 依赖关系

- 感知层 → 无底层依赖
- 认知层 → 依赖感知层提供的环境数据
- 规划层 → 依赖认知层提供的世界模型和推理结果
- 执行层 → 依赖规划层生成的计划
- 核心控制器 → 协调各层之间的交互

## 开发指南

### 环境要求

- Python 3.8+
- 依赖项见 `requirements.txt`

### 开发流程

1. 克隆主仓库
2. 创建功能分支
3. 使用Git Worktree管理不同层的开发
4. 提交代码并创建PR
5. 代码审查和合并

### Git Worktree 开发模式

为了支持并行开发不同层级，我们使用Git Worktree模式，每个层级可以有独立的开发分支：

```bash
# 1. 创建并初始化主仓库
git clone <repository-url>
cd Brain
git checkout main

# 2. 为每个层级创建worktree
git worktree add ../brain-perception perception
git worktree add ../brain-cognitive cognitive
git worktree add ../brain-planning planning
git worktree add ../brain-execution execution
git worktree add ../brain-communication communication
git worktree add ../brain-models models

# 3. 开发者可以根据负责的层级进入对应的worktree
cd ../brain-perception  # 感知层开发
cd ../brain-cognitive  # 认知层开发
cd ../brain-planning   # 规划层开发
cd ../brain-execution   # 执行层开发
cd ../brain-communication # 通信层开发
cd ../brain-models      # 模型层开发
```

### 分支管理策略

```bash
# 主分支
main                    # 主开发分支，包含所有层级

# 层级分支
perception-dev          # 感知层开发分支
cognitive-dev          # 认知层开发分支
planning-dev           # 规划层开发分支
execution-dev          # 执行层开发分支
communication-dev      # 通信层开发分支
models-dev             # 模型层开发分支

# 功能分支
feature/xxx-perception  # 感知层功能分支
feature/xxx-cognitive  # 认知层功能分支
feature/xxx-planning   # 规划层功能分支
feature/xxx-execution   # 执行层功能分支
feature/xxx-communication # 通信层功能分支
feature/xxx-models      # 模型层功能分支
```

### 开发工作流

```bash
# 1. 开发者从main分支创建功能分支
git checkout main
git pull origin main
git checkout -b feature/new-functionality

# 2. 在对应的worktree中进行开发
cd ../brain-<layer>
# 进行代码修改...

# 3. 提交代码
git add .
git commit -m "feat: add new functionality"

# 4. 推送到远程仓库
git push origin feature/new-functionality

# 5. 创建PR到main分支
# 在GitHub/GitLab上创建Pull Request
```

### 代码合并流程

```bash
# 1. 各层级开发完成后，合并到对应的层级开发分支
git checkout perception-dev
git merge feature/xxx-perception
git push origin perception-dev

# 2. 定期将层级开发分支合并到main分支
git checkout main
git merge perception-dev
git merge cognitive-dev
git merge planning-dev
git merge execution-dev
git merge communication-dev
git merge models-dev
git push origin main
```

### 测试策略

```bash
# 1. 单元测试
cd ../brain-<layer>
python -m pytest tests/<layer>/

# 2. 集成测试
cd ../brain
python -m pytest tests/integration/

# 3. 系统测试
python -m pytest tests/system/
```

## 贡献指南

1. 遵循各层的职责边界，避免跨层级的直接依赖
2. 保持接口稳定，如需修改需经过讨论和审查
3. 添加适当的单元测试和文档
4. 遵循代码风格指南
5. 提交前确保所有测试通过

## 许可证

[请在此处添加许可证信息]

## 联系方式

[请在此处添加联系信息]
