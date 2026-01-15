# 规划层 API 文档

## 概述

规划层（Planning Layer, L4）是 Brain 架构中的核心组件，负责将高层任务指令转换为可执行的动作序列。本文档描述规划层的公共 API 和使用方法。

## 架构

```
认知层 (L3)                    规划层 (L4)                    执行层 (L5)
    │                              │                              │
    │  PlanningInput               │  PlanState                   │  Action
    ├─────────────────────────────>│  (任务树)                     │
    │                              ├─────────────────────────────>│
    │  ReasoningResult             │  Action Sequence             │
    │  PlanningContext             │                              │
    │  Belief[]                    │                              │
    │                              │  ReplanningOutput            │
    │                              │<─────────────────────────────┤
    │                              │  Execution Feedback          │
```

## 核心组件

### 1. PlanningOrchestrator

规划编排器，提供统一的规划与执行接口。

#### 初始化

```python
from brain.planning.orchestrator import PlanningOrchestrator

# 独立模式（使用 WorldModelMock）
orchestrator = PlanningOrchestrator(
    platform="ugv",              # 平台类型: drone, ugv, usv
    use_cognitive_layer=False,   # 不使用认知层
    enable_replanning=True       # 启用动态重规划
)

# 集成模式（使用 CognitiveWorldAdapter）
orchestrator = PlanningOrchestrator(
    platform="ugv",
    use_cognitive_layer=True,    # 使用认知层
    enable_replanning=True
)
```

#### 主要方法

##### plan()

规划（不执行），返回 `PlanningOutput`。

```python
from brain.planning.interfaces import PlanningInput

input_data = PlanningInput(
    command="去厨房拿杯水"
)

output = orchestrator.plan(input_data)

if output.is_successful():
    print(f"规划成功，预计时长: {output.estimated_duration}秒")
    print(f"节点数量: {len(output.plan_state.nodes)}")
```

##### plan_and_execute_async()

异步规划并执行。

```python
output = await orchestrator.plan_and_execute_async(
    planning_input=input_data,
    robot_interface=robot  # 可选
)
```

##### replan()

执行重规划。

```python
from brain.planning.interfaces import ReplanningInput

replan_input = ReplanningInput(
    current_plan=current_plan_state,
    failed_actions=["open_door"],
    trigger_reason="动作执行失败"
)

replan_output = orchestrator.replan(replan_input)
```

##### update_context()

更新认知上下文（集成模式）。

```python
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief

context = PlanningContext(
    current_position={"x": 1.0, "y": 2.0, "z": 0.0},
    current_heading=90.0,
    # ... 其他字段
)

beliefs = [
    Belief(id="1", content="杯子在厨房", confidence=0.9)
]

orchestrator.update_context(context, beliefs)
```

---

### 2. PlanningInput

规划输入数据类。

```python
from brain.planning.interfaces import PlanningInput
from brain.cognitive.reasoning.reasoning_result import ReasoningResult
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief

input_data = PlanningInput(
    # 必填
    command="去厨房拿杯水",

    # 可选
    reasoning_result=reasoning_result,  # ReasoningResult
    planning_context=context,          # PlanningContext
    beliefs=[belief1, belief2],        # List[Belief]

    # 元数据（自动生成）
    timestamp=datetime.now(),
    metadata={"key": "value"}
)
```

#### 方法

- `has_reasoning() -> bool`: 是否有推理结果
- `get_reasoning_mode() -> ReasoningMode`: 获取推理模式
- `get_high_confidence_beliefs(threshold=0.7) -> List[Belief]`: 获取高置信度信念
- `to_summary() -> str`: 生成输入摘要

---

### 3. PlanningOutput

规划输出数据类。

```python
from brain.planning.interfaces import PlanningOutput, PlanningStatus
from brain.planning.state import PlanState

output = PlanningOutput(
    # 必填
    plan_state=plan_state,                      # PlanState
    planning_status=PlanningStatus.SUCCESS,     # PlanningStatus
    estimated_duration=15.5,                    # float (秒)
    success_rate=0.95,                          # float (0-1)

    # 可选
    resource_requirements=["arm", "mobile"],   # List[str]
    clarification_request="请问哪个杯子？",     # str
    rejection_reason="指令无法理解",            # str
    planning_log=["规划成功", "3个节点"],        # List[str]

    # 元数据
    timestamp=datetime.now(),
    metadata={"node_count": 3}
)
```

#### 方法

- `is_successful() -> bool`: 是否成功
- `needs_clarification() -> bool`: 是否需要澄清
- `is_rejected() -> bool`: 是否被拒绝
- `get_plan_summary() -> str`: 获取计划摘要
- `to_dict() -> Dict`: 转换为字典

---

### 4. PlanningStatus

规划状态枚举。

```python
from brain.planning.interfaces import PlanningStatus

PlanningStatus.SUCCESS                  # 成功
PlanningStatus.FAILURE                  # 失败
PlanningStatus.PARTIAL                  # 部分成功
PlanningStatus.CLARIFICATION_NEEDED     # 需要澄清
PlanningStatus.REJECTED                 # 被拒绝
```

---

### 5. ReplanningInput

重规划输入数据类。

```python
from brain.planning.interfaces import ReplanningInput

replan_input = ReplanningInput(
    # 必填
    current_plan=plan_state,           # PlanState
    trigger_reason="环境变化",          # str

    # 可选
    current_node_id="node_1",          # str
    environment_changes=[...],          # List[Dict]
    failed_actions=["action_1"],        # List[str]
    new_beliefs=[belief1],             # List[Belief]
    urgency="high",                     # str: low, normal, high, critical

    # 元数据
    timestamp=datetime.now(),
    metadata={"context": context}
)
```

---

### 6. ReplanningOutput

重规划输出数据类。

```python
from brain.planning.interfaces import ReplanningOutput

replan_output = ReplanningOutput(
    # 必填
    new_plan=new_plan_state,           # PlanState
    replanning_type="repair",          # str: repair, replan, insert, retry
    success=True,                      # bool
    reason="修复了2个节点",             # str

    # 可选
    modified_nodes=["node_1"],         # List[str]
    added_nodes=["node_2"],            # List[str]
    removed_nodes=[],                  # List[str]

    # 元数据
    timestamp=datetime.now(),
    metadata={"requires_full_replanning": False}
)
```

#### 方法

- `get_change_summary() -> str`: 获取变化摘要

---

### 7. CognitiveWorldAdapter

认知世界适配器，将认知层输出转换为 IWorldModel 接口。

```python
from brain.planning.interfaces import CognitiveWorldAdapter

adapter = CognitiveWorldAdapter(
    planning_context=context,
    beliefs=beliefs
)

# 实现 IWorldModel 接口
location = adapter.get_location("kitchen")
robot_pos = adapter.get_robot_position()
obstacles = adapter.get_obstacles()
constraints = adapter.get_constraints()
```

#### 扩展方法

- `get_obstacles() -> List[Dict]`: 获取障碍物
- `get_targets() -> List[Dict]`: 获取目标
- `get_constraints() -> List[str]`: 获取约束
- `get_battery_level() -> float`: 获取电池电量
- `get_recent_changes() -> List[Dict]`: 获取最近变化

---

### 8. ReplanningManager

重规划管理器，处理环境变化和计划修复。

```python
from brain.planning.intelligent import ReplanningManager

manager = ReplanningManager(
    world_model=world_model,
    config={
        "max_insertions": 3,
        "max_retries": 3,
        "enable_plan_mending": True,
        "enable_full_replanning": True
    }
)
```

#### 主要方法

##### detect_environment_changes()

检测环境变化。

```python
changes = manager.detect_environment_changes(
    current_plan=plan_state,
    current_context=context,
    new_beliefs=new_beliefs,
    failed_actions=["action_1"]
)
```

##### make_replanning_decision()

做出重规划决策。

```python
from brain.planning.intelligent import ReplanningDecision

decision = manager.make_replanning_decision(
    replanning_input=replan_input,
    changes=changes
)

# decision.should_replan: bool
# decision.strategy: ReplanningStrategy
# decision.reason: str
# decision.urgency: str
# decision.confidence: float
```

##### replan()

执行重规划。

```python
output = manager.replan(
    replanning_input=replan_input,
    decision=decision
)
```

##### validate_plan()

验证计划。

```python
valid, issues = manager.validate_plan(plan_state)
```

##### reset_counters()

重置计数器。

```python
manager.reset_counters()
```

##### get_statistics()

获取统计信息。

```python
stats = manager.get_statistics()
# {
#     "insertion_count": 2,
#     "retry_count": 1,
#     "replan_count": 0,
#     "environment_changes": 3
# }
```

---

### 9. 重规划策略

```python
from brain.planning.intelligent import ReplanningStrategy

ReplanningStrategy.INSERT   # 动态插入前置操作
ReplanningStrategy.RETRY    # 重试当前动作
ReplanningStrategy.REPAIR   # 局部修复计划
ReplanningStrategy.REPLAN   # 完全重新规划
ReplanningStrategy.ABORT    # 中止执行
```

---

### 10. 重规划触发原因

```python
from brain.planning.intelligent import ReplanningTrigger

ReplanningTrigger.ACTION_FAILURE         # 动作执行失败
ReplanningTrigger.ENVIRONMENT_CHANGE     # 环境变化
ReplanningTrigger.BELIEF_CONTRADICTION   # 信念矛盾
ReplanningTrigger.PLAN_INVALID           # 计划无效
ReplanningTrigger.USER_INTERRUPT         # 用户中断
ReplanningTrigger.TIMEOUT                # 超时
ReplanningTrigger.OBSTACLE_DETECTED      # 检测到障碍
ReplanningTrigger.GOAL_UNREACHABLE       # 目标不可达
```

---

## 配置

### 规划层配置

```python
config = {
    # 重规划配置
    "max_insertions": 3,              # 最大插入次数
    "max_retries": 3,                 # 最大重试次数
    "enable_plan_mending": True,      # 启用计划修复
    "enable_full_replanning": True,   # 启用完全重规划

    # 平台配置
    "platform": "ugv",

    # 日志配置
    "log_level": "INFO"
}
```

### 能力配置

配置文件：`config/planning/capability_config.yaml`

```yaml
capabilities:
  move_to:
    type: movement
    platforms: [drone, ugv, usv]
    preconditions:
      - path_clear
    effects:
      - robot_at_location
    default_duration: 5.0

  grasp:
    type: manipulation
    platforms: [ugv, drone]
    preconditions:
      - object_visible
      - robot_at_object
    effects:
      - holding_object
    default_duration: 2.0
```

---

## 错误处理

### PlanningStatus.FAILURE

当规划失败时：

```python
output = orchestrator.plan(input_data)

if output.planning_status == PlanningStatus.FAILURE:
    print(f"规划失败: {output.rejection_reason}")
```

### PlanningStatus.CLARIFICATION_NEEDED

当需要澄清时：

```python
if output.needs_clarification():
    print(f"需要澄清: {output.clarification_request}")
    # 向用户请求澄清
```

### PlanningStatus.REJECTED

当被拒绝时（如低置信度）：

```python
if output.is_rejected():
    print(f"执行被拒绝: {output.rejection_reason}")
    # 可能需要重新评估或请求新指令
```

---

## 最佳实践

1. **始终检查 PlanningStatus**
   ```python
   if output.is_successful():
       # 处理成功情况
   ```

2. **使用认知层集成模式获取更好的结果**
   ```python
   orchestrator = PlanningOrchestrator(
       use_cognitive_layer=True
   )
   ```

3. **定期更新认知上下文**
   ```python
   orchestrator.update_context(context, beliefs)
   ```

4. **启用重规划以应对环境变化**
   ```python
   orchestrator = PlanningOrchestrator(
       enable_replanning=True
   )
   ```

5. **记录规划日志**
   ```python
   for log in output.planning_log:
       logger.info(log)
   ```
