# Brain 项目完整架构图 - Mermaid代码

## 使用方法
1. 访问 https://mermaid.live
2. 将下面的代码复制到编辑器
3. 导出为 PNG/SVG 格式

---

## 图1：完整系统架构图（主图）

```mermaid
graph TB
    subgraph Sensors["传感器层"]
        LIDAR[LiDAR<br/>激光雷达]
        CAM[摄像头<br/>RGB/深度/热成像]
        IMU[IMU<br/>惯性测量]
        GPS[GPS<br/>定位]
    end

    subgraph Perception["感知层 Perception Layer<br/>多模态数据融合"]
        SensorInput[SensorInput<br/>传感器输入]
        PointCloud[PointCloudProcessor<br/>点云处理]
        ObjectDet[ObjectDetector<br/>YOLO目标检测]
        VLM[VLM视觉语言模型<br/>LLaVA/MiniCPM-V]
        Fusion[FusionEngine<br/>融合引擎]
        Situational[SituationalMap<br/>态势图生成]
    end

    subgraph Cognitive["认知层 Cognitive Layer<br/>世界理解与推理"]
        PerceptParser[PerceptionParser<br/>感知解析器]
        WorldModel[World Model<br/>世界模型]
        Semantic[SemanticUnderstanding<br/>语义理解]
        ContextMgr[ContextManager<br/>上下文管理]
        CoT[CoT Engine<br/>思维链推理引擎<br/>GPT-4/Claude/Llama]
    end

    subgraph Planning["规划层 Planning Layer<br/>HTN分层规划"]
        TaskLevel[TaskLevelPlanner<br/>任务层规划<br/>自然语言→技能序列]
        SkillLevel[SkillLevelPlanner<br/>技能层规划<br/>技能→动作序列]
        ActionLevel[ActionLevelPlanner<br/>动作层规划<br/>动作→参数化操作]
        Dynamic[DynamicPlanner<br/>动态规划器<br/>运行时插入前置条件]
        Replanning[ReplanningRules<br/>重规划规则<br/>失败恢复策略]
    end

    subgraph Execution["执行层 Execution Layer<br/>自适应执行"]
        Executor[Executor<br/>执行器]
        Adaptive[AdaptiveExecutor<br/>自适应执行器<br/>实时监控]
        Monitor[ExecutionMonitor<br/>执行监控器]
        FailureDet[FailureDetector<br/>失败检测器]
        Recovery[RecoveryEngine<br/>恢复引擎]
    end

    subgraph Platforms["平台层 Platform Layer"]
        Drone[无人机 Drone<br/>巡航/搜索/投送]
        UGV[无人车 UGV<br/>巡逻/运输/抓取]
        USV[无人船 USV<br/>水域搜索/检测]
    end

    %% 传感器到感知层
    LIDAR --> SensorInput
    CAM --> SensorInput
    CAM --> VLM
    IMU --> SensorInput
    GPS --> SensorInput

    %% 感知层内部流
    SensorInput --> PointCloud
    SensorInput --> ObjectDet
    PointCloud --> Fusion
    ObjectDet --> Fusion
    VLM --> Fusion
    Fusion --> Situational

    %% 感知层到认知层
    Fusion -->|PerceptionData| PerceptParser
    PerceptParser --> WorldModel
    WorldModel --> Semantic
    WorldModel --> ContextMgr
    ContextMgr --> CoT

    %% 认知层到规划层
    WorldModel -->|PlanningContext<br/>robot_state<br/>world_objects<br/>spatial_relations| TaskLevel
    CoT -->|ReasoningResult<br/>推理链+决策| TaskLevel

    %% 规划层内部流
    TaskLevel --> SkillLevel
    SkillLevel --> ActionLevel
    ActionLevel --> Dynamic
    Dynamic --> Replanning

    %% 规划层到执行层
    ActionLevel -->|PlanState<br/>HTN任务树| Executor
    Dynamic -->|插入节点| Executor
    Replanning -->|重规划| TaskLevel

    %% 执行层内部流
    Executor --> Monitor
    Monitor --> FailureDet
    FailureDet -->|失败| Recovery
    Recovery -->|重试| Executor
    Recovery -->|重新规划| TaskLevel

    %% 执行层到平台
    Adaptive -->|execute_action| Drone
    Adaptive -->|execute_action| UGV
    Adaptive -->|execute_action| USV

    Drone -->|ObservationResult| Monitor
    UGV -->|ObservationResult| Monitor
    USV -->|ObservationResult| Monitor

    %% 样式
    classDef sensorStyle fill:#FFE0B2,stroke:#FF9800,stroke-width:2px
    classDef perceptionStyle fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    classDef cognitiveStyle fill:#B3E5FC,stroke:#0288D1,stroke-width:2px
    classDef planningStyle fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    classDef executionStyle fill:#E1BEE7,stroke:#7B1FA2,stroke-width:2px
    classDef platformStyle fill:#FFCCBC,stroke:#FF5722,stroke-width:2px
    classDef vlmStyle fill:#FFEBEE,stroke:#E53935,stroke-width:3px
    classDef llmStyle fill:#E8EAF6,stroke:#3F51B5,stroke-width:3px

    class LIDAR,CAM,IMU,GPS sensorStyle
    class SensorInput,PointCloud,ObjectDet,Fusion,Situational perceptionStyle
    class PerceptParser,WorldModel,Semantic,ContextMgr cognitiveStyle
    class TaskLevel,SkillLevel,ActionLevel,Dynamic,Replanning planningStyle
    class Executor,Adaptive,Monitor,FailureDet,Recovery executionStyle
    class Drone,UGV,USV platformStyle
    class VLM vlmStyle
    class CoT llmStyle
```

---

## 图2：数据流详解图

```mermaid
graph LR
    subgraph Layer1["感知层输出"]
        P1[PerceptionData]
        P1_1[point_cloud<br/>点云数据]
        P1_2[detections<br/>目标检测结果]
        P1_3[semantic_objects<br/>VLM语义理解]
        P1_4[sensors_data<br/>传感器原始数据]
    end

    subgraph Layer2["认知层输出"]
        C1[CognitiveOutput]
        C1_1[planning_context<br/>PlanningContext]
        C1_1_a[robot_state<br/>机器人状态]
        C1_1_b[world_objects<br/>世界物体列表]
        C1_1_c[spatial_relations<br/>空间关系]
        C1_1_d[tracked_objects<br/>追踪对象]
        C1_2[environment_changes<br/>环境变化]
    end

    subgraph Layer3["规划层输出"]
        PL1[PlanState]
        PL1_1[roots<br/>根节点列表]
        PL1_2[nodes<br/>所有节点索引]
        PL1_3[execution_history<br/>执行历史]
        PL1_4[PlanNode节点<br/>id/name/action<br/>preconditions/effects<br/>parameters/status/children]
    end

    subgraph Layer4["执行层输出"]
        E1[ExecutionResult]
        E1_1[success<br/>成功/失败]
        E1_2[action_id<br/>动作ID]
        E1_3[result<br/>执行结果]
        E1_4[error<br/>错误信息]
    end

    P1 -->|process_perception| C1
    C1 -->|get_planning_context| PL1
    PL1 -->|execute_plan| E1
    E1 -.->|失败反馈| PL1

    classDef dataStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    class P1,P1_1,P1_2,P1_3,P1_4,C1,C1_1,C1_1_a,C1_1_b,C1_1_c,C1_1_d,C1_2,PL1,PL1_1,PL1_2,PL1_3,PL1_4,E1,E1_1,E1_2,E1_3,E1_4 dataStyle
```

---

## 图3：大模型应用位置图

```mermaid
graph TB
    subgraph VLM_App["VLM应用位置 - 感知层"]
        VLM_Input[输入]
        VLM_I1[摄像头图像RGB]
        VLM_I2[目标描述文字]
        VLM_I3[问题文字]

        VLM_Model[模型]
        VLM_M1[LLaVA:7b]
        VLM_M2[MiniCPM-V]
        VLM_M3[Ollama本地部署]

        VLM_Func[功能]
        VLM_F1[场景理解<br/>场景描述+物体列表+空间关系]
        VLM_F2[目标搜索<br/>找到/未找到+位置+建议动作]
        VLM_F3[空间问答<br/>方向+距离+描述]

        VLM_Input --> VLM_Model
        VLM_Model --> VLM_Func
    end

    subgraph LLM_App["LLM应用位置 - 认知层"]
        LLM_Input[输入]
        LLM_I1[任务指令]
        LLM_I2[环境上下文]
        LLM_I3[失败信息]

        LLM_Model[模型]
        LLM_M1[GPT-4 API]
        LLM_M2[Claude API]
        LLM_M3[Llama3.1 本地]

        LLM_Func[功能]
        LLM_F1[链式思维推理<br/>推理链+决策+建议]
        LLM_F2[任务分解<br/>自然语言→HTN任务树]
        LLM_F3[异常处理<br/>原因分析+恢复策略]

        LLM_Input --> LLM_Model
        LLM_Model --> LLM_Func
    end

    subgraph DataFlow["完整数据流"]
        Step1[步骤1: VLM场景理解<br/>图像→VLM→场景描述]
        Step2[步骤2: VLM目标搜索<br/>图像+指令→VLM→目标位置]
        Step3[步骤3: 感知数据融合<br/>点云+VLM→PerceptionData]
        Step4[步骤4: LLM推理决策<br/>PerceptionData+任务→CoT→ReasoningResult]
        Step5[步骤5: HTN任务分解<br/>ReasoningResult→PlanState]
        Step6[步骤6: 执行与反馈<br/>PlanState→Executor→ObservationResult]
    end

    VLM_Func --> Step3
    LLM_Func --> Step5

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
    Step4 --> Step5
    Step5 --> Step6

    classDef vlmStyle fill:#FFEBEE,stroke:#E53935,stroke-width:3px
    classDef llmStyle fill:#E8EAF6,stroke:#3F51B5,stroke-width:3px
    classDef flowStyle fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px

    class VLM_Model,VLM_M1,VLM_M2,VLM_M3 vlmStyle
    class LLM_Model,LLM_M1,LLM_M2,LLM_M3 llmStyle
    class Step1,Step2,Step3,Step4,Step5,Step6 flowStyle
```

---

## 图4：HTN分层规划详解

```mermaid
graph TB
    subgraph Level1["任务层 Task Level"]
        Input1[自然语言指令]
        T1["搜索灾区<br/>发现被困人员后报告"]
        T1_Task[任务分解]
        T1_Skills[技能序列]
        T1_S["起飞<br/>巡航<br/>搜索<br/>检测<br/>返航"]

        Input1 --> T1
        T1 --> T1_Task
        T1_Task --> T1_Skills
        T1_Skills --> T1_S
    end

    subgraph Level2["技能层 Skill Level"]
        S1[巡航技能]
        S2[搜索技能]
        S3[检测技能]

        S1_Actions[动作序列]
        S1_A["goto(location)<br/>monitor_battery<br/>avoid_obstacles"]

        S2_Actions[动作序列]
        S2_A["search_pattern<br/>detect_human<br/>track_target"]

        S3_Actions[动作序列]
        S3_A["thermal_scan<br/>take_photo<br/>record_position"]

        T1_S --> S1
        T1_S --> S2
        T1_S --> S3

        S1 --> S1_Actions
        S2 --> S2_Actions
        S3 --> S3_Actions

        S1_Actions --> S1_A
        S2_Actions --> S2_A
        S3_Actions --> S3_A
    end

    subgraph Level3["动作层 Action Level"]
        A1[goto动作]
        A2[detect_human动作]
        A3[thermal_scan动作]

        A1_Params[参数化]
        A1_P["location: (x,y,z)<br/>speed: 5m/s<br/>altitude: 50m"]

        A2_Params[参数化]
        A2_P["mode: thermal+visual<br/>threshold: 0.8<br/>range: 100m"]

        A3_Params[参数化]
        A3_P["target: human<br/>resolution: high<br/>duration: 30s"]

        S1_A --> A1
        S2_A --> A2
        S3_A --> A3

        A1 --> A1_Params
        A2 --> A2_Params
        A3 --> A3_Params

        A1_Params --> A1_P
        A2_Params --> A2_P
        A3_Params --> A3_P
    end

    subgraph Dynamic["动态推理"]
        Check[检查前置条件]
        Insert[插入动作]

        Check -.->|门关闭| Insert
        Insert --> I1[open_door]

        Check -.->|目标丢失| Insert
        Insert --> I2[search_area]

        Check -.->|电量低| Insert
        Insert --> I3[return_home]
    end

    A1 -.->|检查| Check
    A2 -.->|检查| Check
    A3 -.->|检查| Check

    classDef taskStyle fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    classDef skillStyle fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    classDef actionStyle fill:#B3E5FC,stroke:#0288D1,stroke-width:2px
    classDef dynamicStyle fill:#FFCCBC,stroke:#FF5722,stroke-width:2px

    class Input1,T1,T1_Task,T1_Skills,T1_S taskStyle
    class S1,S2,S3,S1_Actions,S2_Actions,S3_Actions,S1_A,S2_A,S3_A skillStyle
    class A1,A2,A3,A1_Params,A2_Params,A3_Params,A1_P,A2_P,A3_P actionStyle
    class Check,Insert,I1,I2,I3 dynamicStyle
```

---

## 图5：自适应执行流程

```mermaid
graph TB
    Start[接收PlanState] --> Execute[Executor执行节点]

    Execute --> Monitor[ExecutionMonitor<br/>实时监控]

    Monitor --> Check{检查执行状态}

    Check -->|成功| Update[更新WorldModel]
    Check -->|失败| Detect[FailureDetector<br/>失败检测]

    Detect --> Classify{失败分类}

    Classify -->|临时故障| Retry[重试<br/>retry_count < 3]
    Classify -->|环境变化| Insert[动态插入<br/>插入新动作]
    Classify -->|连续失败| Replan[重新规划<br/>触发ReplanningRules]

    Retry --> Execute
    Insert --> Execute
    Replan --> ReplanEngine[ReplanningEngine<br/>重规划引擎]

    ReplanEngine --> Strategy{选择策略}
    Strategy -->|替代路径| AltPath[选择替代路径]
    Strategy -->|回滚| Rollback[回滚到安全状态]
    Strategy -->|求助| Human[请求人工介入]

    AltPath --> Execute
    Rollback --> Execute
    Human --> End[任务终止]

    Update --> Next{有下一个节点?}
    Next -->|是| Execute
    Next -->|否| Success[任务完成]

    Success --> End

    classDef normalStyle fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    classDef errorStyle fill:#FFCCBC,stroke:#FF5722,stroke-width:2px
    classDef decisionStyle fill:#FFF9C4,stroke:#FBC02D,stroke-width:2px
    classDef successStyle fill:#B3E5FC,stroke:#0288D1,stroke-width:2px

    class Start,Execute,Monitor,Update,Next,Execute normalStyle
    class Detect,Classify,Retry,Insert,Replan,ReplanEngine errorStyle
    class Check,Strategy decisionStyle
    class Success,End successStyle
```

---

## 图6：三平台统一架构对比

```mermaid
graph TB
    subgraph Brain["Brain系统 - 统一核心"]
        Perception[感知层<br/>VLM多模态融合]
        Cognitive[认知层<br/>World Model + CoT]
        Planning[规划层<br/>HTN分层规划]
        Execution[执行层<br/>自适应执行]
    end

    subgraph DroneApp["无人机应用 Drone"]
        D_Cap[能力集<br/>fly/goto/search<br/>thermal_scan/air_drop]
        D_Task[典型任务<br/>灾区搜救<br/>电力巡检<br/>测绘勘探]
        D_Adv[优势<br/>空中视角<br/>快速部署<br/>大范围覆盖]
    end

    subgraph UGVApp["无人车应用 UGV"]
        G_Cap[能力集<br/>drive/grasp/carry<br/>patrol/inspect]
        G_Task[典型任务<br/>物流运输<br/>安保巡逻<br/>危化品处理]
        G_Adv[优势<br/>载重能力<br/>精确操作<br/>持续作业]
    end

    subgraph USVApp["无人船应用 USV"]
        S_Cap[能力集<br/>navigate/sample<br/>water_test/detect]
        S_Task[典型任务<br/>水质监测<br/>水下检测<br/>水上救援]
        S_Adv[优势<br/>水域适应<br/>样品采集<br/>双模检测]
    end

    Perception --> P1[Capability接口]
    Cognitive --> P1
    Planning --> P1
    Execution --> P1

    P1 -->|适配| D_Cap
    P1 -->|适配| G_Cap
    P1 -->|适配| S_Cap

    D_Cap --> D_Task
    G_Cap --> G_Task
    S_Cap --> S_Task

    D_Task --> D_Adv
    G_Task --> G_Adv
    S_Task --> S_Adv

    classDef brainStyle fill:#E1BEE7,stroke:#7B1FA2,stroke-width:3px
    classDef droneStyle fill:#FFCCBC,stroke:#FF5722,stroke-width:2px
    classDef ugvStyle fill:#C8E6C9,stroke:#388E3C,stroke-width:2px
    classDef usvStyle fill:#B3E5FC,stroke:#0288D1,stroke-width:2px

    class Perception,Cognitive,Planning,Execution,P1 brainStyle
    class D_Cap,D_Task,D_Adv droneStyle
    class G_Cap,G_Task,G_Adv ugvStyle
    class S_Cap,S_Task,S_Adv usvStyle
```

---

## 快速使用指南

### 方案1：在线预览（推荐）
1. 打开 https://mermaid.live
2. 复制上面任意图的代码（从```mermaid到```）
3. 粘贴到左侧编辑器
4. 右侧实时预览
5. 点击"Download PNG/SVG"下载

### 方案2：在Markdown中使用
如果你使用Typora、VS Code等支持Mermaid的编辑器：
- 直接复制代码块到Markdown文件
- 编辑器会自动渲染成图

### 方案3：生成到PPT
1. 在Mermaid Live生成图片
2. 下载为PNG/SVG
3. 插入到你的PPT中

---

## 建议

**用于路演PPT**：
- 使用图1（完整系统架构图）作为主图
- 配合图3（大模型应用位置）展示AI能力

**用于技术文档**：
- 图2（数据流详解）+ 图4（HTN分层规划）
- 图5（自适应执行流程）

**用于应用场景展示**：
- 图6（三平台统一架构对比）

需要我调整任何图的样式或内容吗？
