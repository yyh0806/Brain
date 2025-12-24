# 认知层架构重构文档

## 概述

本文档说明认知层架构重构后的新架构和接口使用方式。

## 核心架构原则

**架构警句（设计准则）：**
> 感知层看到世界，认知层相信世界，规划层改变世界。

认知层作为系统的"信念维护与推理"核心，**绝不输出行动决策**，只输出认知结果。

## 新的文件结构

```
brain/cognitive/
├── __init__.py                    # 统一导出接口
├── interface.py                   # 认知层统一接口（新增）
│
├── world_model/                   # 世界模型模块
│   ├── __init__.py
│   ├── world_model.py             # 核心世界模型
│   ├── planning_context.py        # PlanningContext定义
│   ├── environment_change.py      # EnvironmentChange相关
│   ├── belief/                    # 信念管理（新增核心模块）
│   │   ├── __init__.py
│   │   ├── belief.py              # 信念定义
│   │   └── belief_update_policy.py # 信念修正策略（核心）
│   ├── object_tracking/            # 物体追踪（新增子模块）
│   │   ├── __init__.py
│   │   └── tracked_object.py
│   └── semantic/                   # 语义理解（新增子模块）
│       ├── __init__.py
│       └── semantic_object.py
│
├── reasoning/                     # 推理模块
│   ├── __init__.py
│   ├── cot_engine.py              # CoT推理引擎
│   └── reasoning_result.py        # 推理结果定义（从cot_engine.py拆分）
│
├── dialogue/                      # 对话模块
│   ├── __init__.py
│   ├── dialogue_manager.py
│   └── dialogue_types.py          # 对话类型定义（从dialogue_manager.py拆分）
│
└── monitoring/                     # 监控模块
    ├── __init__.py
    ├── perception_monitor.py
    └── monitor_events.py           # 监控事件定义（从perception_monitor.py拆分）
```

## 统一接口使用

### 推荐方式：使用 CognitiveLayer

```python
from brain.cognitive import CognitiveLayer
from brain.perception.sensors.ros2_sensor_manager import PerceptionData
from brain.cognitive.interface import ObservationResult, ObservationStatus

# 初始化认知层
cognitive_layer = CognitiveLayer(
    world_model=world_model,
    cot_engine=cot_engine,
    dialogue_manager=dialogue_manager,
    perception_monitor=perception_monitor,
    config=config
)

# 处理感知数据
perception_data = await sensor_manager.get_fused_perception()
cognitive_output = await cognitive_layer.process_perception(perception_data)

# 获取规划上下文
planning_context = cognitive_layer.get_planning_context()

# 执行推理
reasoning_result = await cognitive_layer.reason(
    query="环境发生变化，是否需要调整计划？",
    context={"changes": changes_desc},
    mode=ReasoningMode.REPLANNING
)

# 更新信念（根据执行结果）
observation_result = ObservationResult(
    operation_id="op_123",
    operation_type="search",
    status=ObservationStatus.FAILURE,
    location={"x": 10.0, "y": 20.0}
)
belief_update = await cognitive_layer.update_belief(observation_result)

# 获取已证伪的信念
falsified_beliefs = cognitive_layer.get_falsified_beliefs()
```

### 直接使用各模块（不推荐，但支持向后兼容）

```python
from brain.cognitive.world_model import WorldModel, PlanningContext
from brain.cognitive.reasoning import CoTEngine, ReasoningMode
from brain.cognitive.dialogue import DialogueManager
from brain.cognitive.monitoring import PerceptionMonitor

# 直接使用各模块
world_model = WorldModel(config=config)
cot_engine = CoTEngine(llm_interface=llm)
dialogue_manager = DialogueManager(llm_interface=llm)
perception_monitor = PerceptionMonitor(world_model=world_model)
```

## 关键变更

### 1. 传感器相关文件已迁移到感知层

以下文件已从 `brain/cognitive/world_model/` 迁移到 `brain/perception/`：

- `sensor_manager.py` → `brain/perception/sensors/sensor_manager.py`
- `sensor_interface.py` → `brain/perception/sensors/sensor_interface.py`
- `data_converter.py` → `brain/perception/data_converter.py`
- `sensor_input_types.py` → `brain/perception/sensor_input_types.py`

**更新导入路径：**
```python
# 旧导入（已废弃）
from brain.cognitive.world_model.sensor_manager import MultiSensorManager

# 新导入
from brain.perception.sensors.sensor_manager import MultiSensorManager
```

### 2. 类型定义已拆分

**推理相关类型：**
```python
# 旧导入（已废弃）
from brain.cognitive.reasoning.cot_engine import ReasoningResult, ReasoningMode

# 新导入
from brain.cognitive.reasoning.reasoning_result import ReasoningResult, ReasoningMode
from brain.cognitive.reasoning import ReasoningResult, ReasoningMode  # 或使用统一导出
```

**对话相关类型：**
```python
# 旧导入（已废弃）
from brain.cognitive.dialogue.dialogue_manager import DialogueType, DialogueContext

# 新导入
from brain.cognitive.dialogue.dialogue_types import DialogueType, DialogueContext
from brain.cognitive.dialogue import DialogueType, DialogueContext  # 或使用统一导出
```

**监控相关类型：**
```python
# 旧导入（已废弃）
from brain.cognitive.monitoring.perception_monitor import MonitorEvent, ReplanTrigger

# 新导入
from brain.cognitive.monitoring.monitor_events import MonitorEvent, ReplanTrigger
from brain.cognitive.monitoring import MonitorEvent, ReplanTrigger  # 或使用统一导出
```

### 3. 新增信念修正机制

**BeliefUpdatePolicy** 是认知层实现"自我否定"能力的核心：

```python
from brain.cognitive.world_model.belief import BeliefUpdatePolicy
from brain.cognitive.interface import ObservationResult, ObservationStatus

# 创建信念修正策略
belief_policy = BeliefUpdatePolicy(config=config)

# 根据观测结果更新信念
observation_result = ObservationResult(
    operation_id="search_kitchen",
    operation_type="search",
    status=ObservationStatus.FAILURE,
    location={"x": 5.0, "y": 3.0}
)

belief_update = belief_policy.update_belief(observation_result)

# 获取已证伪的信念
falsified_beliefs = belief_policy.get_falsified_beliefs()
```

## 输出边界约束（重要）

认知层**绝不输出**：
- ❌ 行动决策（"应该去厨房"）
- ❌ 操作序列（"先移动，再搜索"）
- ❌ 执行指令（这些属于规划层）

认知层**只输出**：
- ✅ 状态（belief）："杯子在厨房的概率已下降至 0.2"
- ✅ 变化（event）："检测到新障碍物"
- ✅ 推理结论（why/what changed）："搜索失败3次，建议重新评估假设"
- ✅ 建议（suggestion）："建议重新评估杯子位置假设"（**不是决策**）

## 迁移指南

### 对于使用认知层的代码

1. **推荐迁移到统一接口：**
   ```python
   # 旧方式
   world_model = WorldModel()
   changes = world_model.update_from_perception(data)
   context = world_model.get_context_for_planning()
   
   # 新方式（推荐）
   cognitive_layer = CognitiveLayer(world_model=world_model, ...)
   output = await cognitive_layer.process_perception(data)
   context = cognitive_layer.get_planning_context()
   ```

2. **更新导入路径：**
   - 检查所有 `from brain.cognitive.world_model.sensor_*` 导入
   - 更新为 `from brain.perception.sensors.*` 或 `from brain.perception.*`
   - 检查所有 `from brain.cognitive.cot_engine` 导入
   - 更新为 `from brain.cognitive.reasoning.*`

### 对于测试代码

更新测试文件中的导入路径：
```python
# tests/unit/test_sensor_input.py
# 旧导入
from brain.cognitive.world_model.sensor_input_types import ...

# 新导入
from brain.perception.sensor_input_types import ...
```

## 向后兼容性

为了保持向后兼容性，以下导入仍然可用（通过 `__init__.py` 重新导出）：

```python
# 这些导入仍然有效
from brain.cognitive import WorldModel, CoTEngine, DialogueManager
from brain.cognitive.world_model import PlanningContext, EnvironmentChange
```

但建议使用新的统一接口 `CognitiveLayer`。

## 测试

运行测试确保功能正常：

```bash
# 运行认知层相关测试
python -m pytest brain/cognitive/ -v

# 运行所有测试
python -m pytest tests/ -v
```

## 常见问题

### Q: 为什么传感器相关文件要迁移到感知层？

A: 认知层应该专注于"认知"（理解、推理、决策），而不是数据处理。传感器数据处理属于感知层的职责。

### Q: 为什么要拆分类型定义？

A: 提高代码可维护性，每个模块的类型定义独立，便于理解和修改。

### Q: BeliefUpdatePolicy 的作用是什么？

A: 这是认知层实现"自我否定"能力的核心。它根据执行结果（成功/失败）更新信念，维护"哪些假设已经被证伪"，防止规划震荡和隐性循环。

### Q: 如何确保认知层不输出行动决策？

A: 
1. 使用 `CognitiveLayer` 统一接口，接口中明确标注了输出边界
2. 代码审查时检查是否有输出行动决策的逻辑
3. `ReasoningResult.decision` 字段只包含解释性判断，不包含动作指令

## 参考

- [认知层架构重构计划](../plans/认知层架构重构_73501bd9.plan.md)
- [认知层分析报告](COGNITIVE_LAYER_ANALYSIS_REPORT.md)
- [认知层优化计划](COGNITIVE_LAYER_OPTIMIZATION_PLAN.md)






