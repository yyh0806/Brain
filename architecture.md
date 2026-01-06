# Brain项目架构与数据流向

## 项目概述

Brain是一个**感知驱动的智能无人系统**，以World Model为核心，融合多传感器数据，实现环境感知、智能决策和自主控制。

---

## 1. 系统整体架构

```mermaid
graph TB
    subgraph "交互层"
        USER[用户指令/反馈]
        NLP[自然语言处理]
    end

    subgraph "感知层 Perception"
        SENSORS[传感器管理器<br/>ROS2SensorManager]
        VLM[VLM视觉感知<br/>VLMPerception]
        OCCUPANCY[占据栅格地图<br/>OccupancyMapper]
        FUSION[多传感器融合<br/>EKF/UKF]
    end

    subgraph "认知层 Cognitive"
        WM[世界模型<br/>WorldModel]
        COT[CoT推理引擎<br/>CoTEngine]
        DIALOGUE[对话管理器<br/>DialogueManager]
        PERCEPTION_MONITOR[感知监控器<br/>PerceptionMonitor]
    end

    subgraph "规划层 Planning"
        TASK_PLANNER[任务级规划器<br/>TaskPlanner]
        SKILL_PLANNER[技能级规划器<br/>SkillLevelPlanner]
        ACTION_PLANNER[动作级规划器<br/>ActionLevelPlanner]
        DYNAMIC_PLANNER[动态规划器<br/>DynamicPlanner]
    end

    subgraph "执行层 Execution"
        EXECUTOR[执行引擎<br/>Executor]
        OPERATIONS[操作库<br/>移动/传感器/工具]
    end

    subgraph "通信层 Communication"
        ROS2[ROS2接口<br/>ROS2Interface]
        ROBOT[机器人接口<br/>RobotInterface]
    end

    subgraph "状态管理 State"
        CONFIG[配置管理]
        LOGS[日志系统]
        STATE[系统状态]
    end

    subgraph "外部系统"
        ISAAC_SIM[Isaac Sim仿真]
        REAL_ROBOT[真实机器人]
        SENSOR_HW[传感器硬件]
    end

    %% 用户交互
    USER --> NLP
    NLP --> DIALOGUE
    DIALOGUE --> USER

    %% 感知数据流
    SENSOR_HW --> ROS2
    ISAAC_SIM --> ROS2
    ROS2 --> SENSORS
    SENSORS --> FUSION
    SENSORS --> VLM
    SENSORS --> OCCUPANCY
    FUSION --> WM
    VLM --> WM
    OCCUPANCY --> WM

    %% 认知层
    WM --> PERCEPTION_MONITOR
    PERCEPTION_MONITOR --> DYNAMIC_PLANNER
    WM --> COT
    DIALOGUE --> COT
    COT --> TASK_PLANNER

    %% 规划层
    TASK_PLANNER --> SKILL_PLANNER
    SKILL_PLANNER --> ACTION_PLANNER
    ACTION_PLANNER --> EXECUTOR
    DYNAMIC_PLANNER --> TASK_PLANNER

    %% 执行层
    EXECUTOR --> OPERATIONS
    OPERATIONS --> ROS2
    ROS2 --> ROBOT
    ROBOT --> REAL_ROBOT
    ROBOT --> ISAAC_SIM

    %% 状态管理
    CONFIG --> SENSORS
    CONFIG --> WM
    CONFIG --> COT
    CONFIG --> EXECUTOR
    STATE --> DIALOGUE

    %% 样式
    classDef perceptionStyle fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    classDef cognitiveStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef planningStyle fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef executionStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:3px
    classDef communicationStyle fill:#fce4ec,stroke:#880e4f,stroke-width:3px

    class SENSORS,VLM,OCCUPANCY,FUSION perceptionStyle
    class WM,COT,DIALOGUE,PERCEPTION_MONITOR cognitiveStyle
    class TASK_PLANNER,SKILL_PLANNER,ACTION_PLANNER,DYNAMIC_PLANNER planningStyle
    class EXECUTOR,OPERATIONS executionStyle
    class ROS2,ROBOT communicationStyle
```

---

## 2. 数据流向图

### 2.1 感知数据流

```mermaid
flowchart LR
    subgraph "传感器数据源"
        CAM[RGB相机<br/>Image]
        DEPTH[深度相机<br/>Depth]
        LIDAR[激光雷达<br/>PointCloud]
        GPS[GPS<br/>Position]
        IMU[IMU<br/>Orientation]
    end

    subgraph "数据预处理"
        SYNC[时间同步]
        CALIB[标定补偿]
        TRANSFORM[坐标转换]
    end

    subgraph "特征提取与融合"
        VLM_PROC[VLM分析<br/>场景理解]
        OBJ_DET[物体检测]
        MAP_BUILD[地图构建]
        SENSOR_FUSION[传感器融合<br/>EKF/UKF]
    end

    subgraph "世界模型"
        OBJ_TRACK[物体追踪]
        ENV_REP[环境表示]
        CHANGE_DET[变化检测]
    end

    CAM --> SYNC
    DEPTH --> SYNC
    LIDAR --> SYNC
    GPS --> SYNC
    IMU --> SYNC

    SYNC --> CALIB
    CALIB --> TRANSFORM

    TRANSFORM --> VLM_PROC
    TRANSFORM --> OBJ_DET
    TRANSFORM --> MAP_BUILD
    TRANSFORM --> SENSOR_FUSION

    VLM_PROC --> ENV_REP
    OBJ_DET --> OBJ_TRACK
    MAP_BUILD --> ENV_REP
    SENSOR_FUSION --> OBJ_TRACK

    OBJ_TRACK --> CHANGE_DET
    ENV_REP --> CHANGE_DET

    CHANGE_DET -->|显著变化| REPLAN[触发重规划]
```

### 2.2 决策与控制数据流

```mermaid
flowchart TB
    START([用户指令输入])

    START --> PARSE[指令解析<br/>DialogueManager]

    PARSE --> UNDERSTAND[语义理解<br/>CoT推理]

    UNDERSTAND --> PLAN[任务规划<br/>TaskPlanner]

    PLAN --> DECOMPOSE[任务分解]

    DECOMPOSE --> SKILL[技能规划<br/>SkillLevelPlanner]

    SKILL --> ACTION[动作规划<br/>ActionLevelPlanner]

    ACTION --> EXECUTE[执行<br/>Executor]

    EXECUTE --> MONITOR{监控}

    MONITOR -->|正常| COMPLETE([任务完成])
    MONITOR -->|异常| RECOVER[错误恢复]

    RECOVER --> RETRY{重试}

    RETRY -->|成功| COMPLETE
    RETRY -->|失败| REPORT[报告用户]

    REPORT --> COMPLETE

    %% 感知驱动重规划
    PERCEPTION[感知监控] -.->|环境变化| REPLAN[重规划触发]
    EXECUTE -.->|状态反馈| PERCEPTION
    PERCEPTION --> CHANGE_DETECTED{变化检测}
    CHANGE_DETECTED -->|显著| REEVALUATE[重新评估计划]
    CHANGE_DETECTED -->|无关| MONITOR
    REEVALUATE --> PLAN

    %% 用户交互
    COMPLETE --> DIALOGUE[对话管理]
    DIALOGUE --> USER_FEEDBACK[用户反馈]
    USER_FEEDBACK --> START

    style PERCEPTION fill:#e1f5ff,stroke:#01579b
    style PLAN fill:#fff3e0,stroke:#e65100
    style EXECUTE fill:#e8f5e9,stroke:#1b5e20
    style REEVALUATE fill:#fff3e0,stroke:#e65100
```

### 2.3 端到端数据流

```mermaid
sequenceDiagram
    participant User as 用户
    participant DLG as 对话管理器
    participant COT as CoT推理引擎
    participant Plan as 规划层
    participant WM as 世界模型
    participant Perc as 感知层
    participant Exec as 执行器
    participant Robot as 机器人/仿真
    participant Monitor as 感知监控

    User->>DLG: 自然语言指令
    DLG->>COT: 解析指令
    COT->>WM: 获取当前状态
    WM-->>COT: 环境信息

    COT->>Plan: 生成任务计划
    Plan->>Exec: 执行序列

    loop 执行循环
        Exec->>Perc: 获取感知数据
        Perc->>Robot: 读取传感器
        Robot-->>Perc: 原始数据
        Perc-->>Exec: 处理后数据
        Exec->>Robot: 发送控制指令

        Perc->>WM: 更新世界模型
        WM->>Monitor: 检测变化

        alt 环境显著变化
            Monitor->>COT: 触发重规划
            COT->>Plan: 重新规划
            Plan-->>Exec: 新计划
        end
    end

    Exec->>DLG: 任务完成/状态更新
    DLG->>User: 反馈结果
```

---

## 3. 核心模块交互图

```mermaid
graph LR
    subgraph "感知驱动重规划机制"
        PERC_DATA[感知数据] --> WM_UPDATE[世界模型更新]
        WM_UPDATE --> CHANGE[变化检测]
        CHANGE --> SIG{显著性?}
        SIG -->|是| REPLAN[重规划]
        SIG -->|否| CONTINUE[继续执行]
        REPLAN --> PLAN_ADJUST[计划调整]
        PLAN_ADJUST --> EXEC_NEW[新执行]
    end

    subgraph "CoT推理流程"
        TASK[任务] --> COT_START[CoT推理开始]
        COT_START --> DECOMP[任务分解]
        DECOMP --> REASON[逐步推理]
        REASON --> VALIDATE[验证]
        VALIDATE -->|通过| PLAN_OUT[计划输出]
        VALIDATE -->|失败| REASON
    end

    subgraph "对话交互"
        USER_Q[用户问题] --> NLP[自然语言处理]
        NLP --> INTENT[意图识别]
        INTENT --> CTX[上下文管理]
        CTX --> RESPONSE[生成响应]
        RESPONSE --> USER_A[用户回答]
    end

    style PERC_DATA fill:#e1f5ff,stroke:#01579b
    style WM_UPDATE fill:#e1f5ff,stroke:#01579b
    style CHANGE fill:#f3e5f5,stroke:#4a148c
    style REPLAN fill:#fff3e0,stroke:#e65100
    style EXEC_NEW fill:#e8f5e9,stroke:#1b5e20
```

---

## 4. 分层架构详解

### 4.1 感知层 (Perception Layer)

**位置**: `/brain/perception/`

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| ROS2SensorManager | 传感器数据采集与管理 | 传感器硬件 | 原始数据流 |
| VLMPerception | 视觉语言模型场景理解 | RGB图像 | 语义信息 |
| OccupancyMapper | 占据栅格地图构建 | 点云数据 | 2D/3D地图 |
| SensorFusion | 多传感器数据融合 | 异构传感器 | 融合状态估计 |

**关键技术**:
- EKF/UKF滤波算法
- 点云处理 (PCL)
- VLM集成 (Ollama)
- 坐标系转换 (TF)

### 4.2 认知层 (Cognitive Layer)

**位置**: `/brain/cognitive/`

| 模块 | 功能 | 输入 | 输出 |
|------|------|------|------|
| WorldModel | 环境表示与状态维护 | 感知数据 | 世界状态 |
| CoTEngine | 链式思维推理 | 任务/上下文 | 推理结果 |
| DialogueManager | 对话交互管理 | 用户输入 | 响应输出 |
| PerceptionMonitor | 感知变化监控 | 感知流 | 重规划信号 |

**关键技术**:
- Chain-of-Thought推理
- 物体追踪与状态估计
- 变化检测算法
- 对话状态管理

### 4.3 规划层 (Planning Layer)

**位置**: `/brain/planning/`

| 规划器 | 粒度 | 功能 |
|--------|------|------|
| TaskPlanner | 任务级 | 高层任务分解与策略 |
| SkillLevelPlanner | 技能级 | 可重用技能组合 |
| ActionLevelPlanner | 动作级 | 具体动作序列生成 |
| DynamicPlanner | 动态 | 实时调整与重规划 |

**规划流程**:
```
任务 → 技能序列 → 动作序列 → 原子操作
  ↓        ↓          ↓         ↓
战略    战术      操作      执行
```

### 4.4 执行层 (Execution Layer)

**位置**: `/brain/execution/`

| 组件 | 功能 |
|------|------|
| Executor | 执行引擎,管理操作队列 |
| 移动操作 | 起飞、降落、导航 |
| 传感器操作 | 数据采集控制 |
| 工具操作 | 机械臂、抓取等 |

**执行特性**:
- 异步执行
- 状态监控
- 超时处理
- 错误恢复

### 4.5 通信层 (Communication Layer)

**位置**: `/brain/communication/`

| 接口 | 功能 |
|------|------|
| ROS2Interface | ROS2话题/服务/动作 |
| RobotInterface | 机器人控制抽象 |
| 紧急停止 | 安全机制 |

**支持的通信模式**:
- 发布/订阅 (Topic)
- 请求/响应 (Service)
- 目标/反馈 (Action)
- QoS配置

---

## 5. 配置系统架构

```mermaid
graph TD
    CONFIG_ROOT[配置根目录]

    CONFIG_ROOT --> GLOBAL[全局配置<br/>global/]
    CONFIG_ROOT --> MODULES[模块配置<br/>modules/]
    CONFIG_ROOT --> PLATFORMS[平台配置<br/>platforms/]
    CONFIG_ROOT --> ENVIRONMENTS[环境配置<br/>environments/]
    CONFIG_ROOT --> USERS[用户配置<br/>users/]

    GLOBAL --> SYS[system.yaml<br/>系统配置]
    GLOBAL --> DEF[defaults.yaml<br/>默认值]
    GLOBAL --> SCH[schemas/<br/>配置模式]

    MODULES --> PERC[perception/<br/>感知配置]
    MODULES --> COMM[communication/<br/>通信配置]
    MODULES --> COG[cognitive/<br/>认知配置]
    MODULES --> PLAN[planning/<br/>规划配置]
    MODULES --> EXEC[execution/<br/>执行配置]

    PERC --> SENS[sensors.yaml]
    PERC --> FUS[fusion.yaml]

    COMM --> ROS[ros2.yaml]

    style CONFIG_ROOT fill:#f5f5f5,stroke:#424242
    style SYS fill:#e3f2fd,stroke:#1976d2
    style SENS fill:#e1f5fe,stroke:#0277bd
    style ROS fill:#fce4ec,stroke:#c2185b
```

---

## 6. 部署模式

```mermaid
graph LR
    subgraph "开发环境"
        DEV[开发模式<br/>单元测试/集成测试]
    end

    subgraph "仿真环境"
        SIM[仿真模式<br/>Isaac Sim]
    end

    subgraph "生产环境"
        PROD[生产模式<br/>真实机器人]
    end

    DEV --> TEST[测试套件]
    TEST --> SIM

    SIM --> VALIDATE[验证]
    VALIDATE --> PROD

    PROD --> MONITORING[运行监控]
    MONITORING -.反馈优化.-> DEV
```

---

## 7. 技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | Python 3.8+ |
| **机器人框架** | ROS2 Humble |
| **AI/ML** | Ollama (VLM), OpenAI LLM |
| **数据处理** | NumPy, OpenCV, PCL |
| **异步处理** | asyncio |
| **日志系统** | Loguru |
| **仿真** | Isaac Sim |
| **配置管理** | YAML, Pydantic |

---

## 8. 关键设计模式

1. **分层架构**: 感知→认知→规划→执行
2. **世界模型**: 中央状态管理
3. **感知驱动**: 环境变化触发重规划
4. **CoT推理**: 可追溯的决策过程
5. **对话式交互**: 自然语言接口
6. **模块化配置**: 灵活的多层级配置系统

---

## 9. 数据流关键路径

### 正常执行路径
```
指令 → CoT推理 → 任务规划 → 技能规划 → 动作规划 → 执行 → 反馈
```

### 感知驱动重规划路径
```
感知数据 → 世界模型 → 变化检测 → 显著性评估 → 重规划 → 新计划 → 执行
```

### 错误恢复路径
```
执行异常 → 错误检测 → 恢复策略 → 重试/回退 → 报告用户
```

---

## 10. 扩展性设计

- **传感器扩展**: 插件式传感器管理器
- **规划器扩展**: 可替换的规划算法
- **执行器扩展**: 可扩展的操作库
- **通信扩展**: 支持多种机器人接口
- **AI模型扩展**: 可切换的VLM/LLM后端

---

*文档版本: 1.0*
*生成日期: 2026-01-06*
*项目: Brain - 感知驱动的智能无人系统*
