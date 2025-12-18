# Brain系统配置管理

统一的配置管理架构，支持多层级配置覆盖和模块化配置。

## 目录结构

```
config/
├── README.md                 # 配置说明文档
├── config_loader.py          # 配置加载器
├── validator.py              # 配置验证工具
├── global/                   # 全局配置
│   ├── system.yaml           # 系统全局配置
│   ├── defaults.yaml         # 默认配置值
│   └── schemas/              # 配置文件模式定义
│       ├── system_schema.json
│       ├── perception_schema.json
│       └── ...
├── modules/                  # 模块配置
│   ├── perception/           # 感知模块配置
│   │   ├── sensors.yaml
│   │   ├── vlm.yaml
│   │   └── fusion.yaml
│   ├── planning/             # 规划模块配置
│   │   ├── path_planning.yaml
│   │   ├── task_planning.yaml
│   │   └── exploration.yaml
│   ├── control/              # 控制模块配置
│   │   ├── motion_control.yaml
│   │   └── trajectory.yaml
│   ├── communication/        # 通信模块配置
│   │   ├── ros2.yaml
│   │   ├── zmq.yaml
│   │   └── protocols.yaml
│   ├── safety/               # 安全模块配置
│   │   ├── constraints.yaml
│   │   └── emergency.yaml
│   └── llm/                  # LLM配置
│       ├── providers.yaml
│       └── models.yaml
├── platforms/                # 平台特定配置
│   ├── drone/                # 无人机配置
│   │   ├── quadrotor.yaml
│   │   └── fixed_wing.yaml
│   ├── ugv/                  # 无人车配置
│   │   ├── car3.yaml
│   │   └── ackermann.yaml
│   └── usv/                  # 无人船配置
│       └── surface.yaml
├── environments/             # 环境配置
│   ├── simulation/           # 仿真环境
│   │   ├── isaac_sim.yaml
│   │   ├── gazebo.yaml
│   │   └── carla.yaml
│   └── real_world/           # 真实环境
│       ├── indoor.yaml
│       └── outdoor.yaml
└── users/                    # 用户特定配置
    ├── user_defaults.yaml    # 用户默认配置
    └── [username]/           # 个性化配置目录
```

## 配置加载优先级

配置系统按以下优先级加载（高优先级覆盖低优先级）：

1. **用户配置** (`users/[username]/`)
   - 用户个人配置文件
   - 环境变量覆盖
   - 命令行参数

2. **平台配置** (`platforms/[platform]/`)
   - 特定平台（drone/ugv/usv）的配置

3. **环境配置** (`environments/[env]/`)
   - 仿真环境或真实环境的配置

4. **模块配置** (`modules/`)
   - 各功能模块的配置

5. **全局配置** (`global/`)
   - 系统全局配置
   - 默认值配置

## 配置覆盖规则

- 相同路径的配置项，高优先级覆盖低优先级
- 数组类型的配置默认替换，可配置合并策略
- 支持配置继承和引用

## 使用方法

```python
from config.config_loader import ConfigLoader

# 初始化配置加载器
loader = ConfigLoader()

# 加载配置
config = loader.load_config(
    platform="ugv",
    environment="real_world",
    user="yangyuhui"
)

# 访问配置
llm_config = config.get("llm")
perception_config = config.get("modules.perception")
```

## 配置验证

```python
from config.validator import ConfigValidator

validator = ConfigValidator()
is_valid, errors = validator.validate(config)

if not is_valid:
    print("配置验证失败：", errors)
```