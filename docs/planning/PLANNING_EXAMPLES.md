# 规划层使用示例

本文档提供规划层的实际使用示例。

## 目录

1. [基本规划](#基本规划)
2. [认知层集成](#认知层集成)
3. [动态重规划](#动态重规划)
4. [完整示例](#完整示例)

---

## 基本规划

### 示例 1：简单任务规划

```python
from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.interfaces import PlanningInput, PlanningStatus

# 初始化规划器
orchestrator = PlanningOrchestrator(
    platform="ugv",
    use_cognitive_layer=False
)

# 创建规划输入
input_data = PlanningInput(
    command="去厨房拿杯水"
)

# 执行规划
output = orchestrator.plan(input_data)

# 检查结果
if output.is_successful():
    print(f"✓ 规划成功")
    print(f"  预计时长: {output.estimated_duration:.1f}秒")
    print(f"  成功率: {output.success_rate:.1%}")
    print(f"  节点数量: {len(output.plan_state.nodes)}")
elif output.needs_clarification():
    print(f"? 需要澄清: {output.clarification_request}")
elif output.is_rejected():
    print(f"✗ 被拒绝: {output.rejection_reason}")
else:
    print(f"✗ 规划失败")
```

### 示例 2：查看计划详情

```python
output = orchestrator.plan(PlanningInput(command="去厨房拿杯水"))

# 打印计划摘要
print(output.get_plan_summary())
# 输出:
# 规划状态: success
# 节点数量: 5
# 预计时长: 15.0秒
# 预期成功率: 90.0%

# 遍历计划节点
for node_id, node in output.plan_state.nodes.items():
    print(f"  [{node.status.value}] {node.name}: {node.action or 'task'}")
    if node.parameters:
        print(f"    参数: {node.parameters}")
```

---

## 认知层集成

### 示例 3：使用认知层推理结果

```python
from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.interfaces import PlanningInput
from brain.cognitive.reasoning.reasoning_result import (
    ReasoningResult,
    ReasoningMode,
    ComplexityLevel
)

# 初始化（认知层模式）
orchestrator = PlanningOrchestrator(
    platform="ugv",
    use_cognitive_layer=True
)

# 创建推理结果
reasoning_result = ReasoningResult(
    mode=ReasoningMode.PLANNING,
    query="去厨房拿杯水",
    context_summary="用户在客厅，想去厨房拿杯子",
    complexity=ComplexityLevel.SIMPLE,
    chain=[],  # 推理链
    decision="可以执行，需要先移动到厨房，然后抓取杯子",
    suggestion="建议检查厨房门是否打开",
    confidence=0.9,
    raw_response="..."
)

# 带推理结果的规划输入
input_data = PlanningInput(
    command="去厨房拿杯水",
    reasoning_result=reasoning_result
)

# 执行规划
output = orchestrator.plan(input_data)

# 成功率会基于推理置信度调整
print(f"成功率: {output.success_rate:.1%}")  # 约 90%
```

### 示例 4：使用信念和上下文

```python
from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.interfaces import PlanningInput
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief

# 初始化
orchestrator = PlanningOrchestrator(
    platform="ugv",
    use_cognitive_layer=True
)

# 创建规划上下文
context = PlanningContext(
    current_position={"x": 0.0, "y": 0.0, "z": 0.0},
    current_heading=0.0,
    obstacles=[
        {"type": "static", "position": {"x": 2.0, "y": 0.0}, "description": "椅子"}
    ],
    targets=[
        {"type": "object", "name": "cup", "position": {"x": 5.0, "y": 3.0}}
    ],
    points_of_interest=[
        {"name": "kitchen", "position": {"x": 5.0, "y": 3.0}, "type": "room"}
    ],
    weather={"condition": "clear", "visibility": "good"},
    battery_level=85.0,
    signal_strength=90.0,
    available_paths=[
        {"from": "living_room", "to": "kitchen", "cost": 10.0}
    ],
    constraints=["avoid_obstacles", "minimize_energy"],
    recent_changes=[],
    risk_areas=[]
)

# 创建信念
beliefs = [
    Belief(id="1", content="杯子在厨房", confidence=0.95),
    Belief(id="2", content="厨房门已打开", confidence=0.90),
    Belief(id="3", content="路径畅通", confidence=0.85)
]

# 更新认知上下文
orchestrator.update_context(context, beliefs)

# 执行规划
input_data = PlanningInput(
    command="去厨房拿杯水",
    planning_context=context,
    beliefs=beliefs
)

output = orchestrator.plan(input_data)

# 高置信度信念会提高成功率
print(f"成功率: {output.success_rate:.1%}")  # 通常 > 90%
```

### 示例 5：处理低置信度推理

```python
from brain.planning.interfaces import PlanningInput, PlanningStatus

# 低置信度推理结果
reasoning_result = ReasoningResult(
    mode=ReasoningMode.PLANNING,
    query="不清楚的指令",
    context_summary="指令不够明确",
    complexity=ComplexityLevel.COMPLEX,
    chain=[],
    decision="指令不够明确",
    suggestion="需要澄清",
    confidence=0.2,  # 低于阈值 0.3
    raw_response="..."
)

input_data = PlanningInput(
    command="不清楚的指令",
    reasoning_result=reasoning_result
)

output = orchestrator.plan(input_data)

# 低置信度会被拒绝
if output.is_rejected():
    print(f"执行被拒绝: {output.rejection_reason}")
    # 输出: "执行被拒绝: 推理置信度过低: 0.20"
```

---

## 动态重规划

### 示例 6：处理执行失败

```python
from brain.planning.interfaces import ReplanningInput

# 假设已有执行中的计划
plan_output = orchestrator.plan(PlanningInput(command="去厨房拿杯水"))
current_plan = plan_output.plan_state

# 模拟执行失败
failed_actions = ["open_door"]  # 开门动作失败

# 创建重规划输入
replan_input = ReplanningInput(
    current_plan=current_plan,
    current_node_id="open_door",
    failed_actions=failed_actions,
    trigger_reason="动作执行失败"
)

# 执行重规划
replan_output = orchestrator.replan(replan_input)

if replan_output.success:
    print(f"✓ 重规划成功: {replan_output.reason}")
    print(f"  策略: {replan_output.replanning_type}")
    print(f"  修改节点: {len(replan_output.modified_nodes)}")
    print(f"  新增节点: {len(replan_output.added_nodes)}")
else:
    print(f"✗ 重规划失败: {replan_output.reason}")
```

### 示例 7：环境变化触发重规划

```python
from brain.planning.interfaces import ReplanningInput
from brain.cognitive.world_model.belief.belief import Belief

# 当前执行中的计划
current_plan = plan_output.plan_state

# 环境变化：新障碍物出现
environment_changes = [
    {
        "type": "obstacle",
        "description": "检测到新障碍物",
        "position": {"x": 3.0, "y": 1.5},
        "severity": "high"
    }
]

# 信念被证伪
falsified_belief = Belief(
    id="1",
    content="路径畅通",
    confidence=0.0,
    falsified=True
)

# 创建重规划输入
replan_input = ReplanningInput(
    current_plan=current_plan,
    environment_changes=environment_changes,
    new_beliefs=[falsified_belief],
    trigger_reason="环境变化",
    urgency="high"
)

# 执行重规划
replan_output = orchestrator.replan(replan_input)

print(replan_output.get_change_summary())
# 输出:
# 重规划类型: repair
# 修改节点: 2
# 新增节点: 1
# 删除节点: 0
```

### 示例 8：获取重规划统计

```python
# 初始化时启用重规划
orchestrator = PlanningOrchestrator(
    platform="ugv",
    enable_replanning=True
)

# 执行一些操作...
# orchestrator.replan(...)

# 获取统计信息
stats = orchestrator.get_replanning_statistics()
print(f"插入次数: {stats['insertion_count']}")
print(f"重试次数: {stats['retry_count']}")
print(f"重规划次数: {stats['replan_count']}")
print(f"环境变化: {stats['environment_changes']}")
```

### 示例 9：便捷的重规划方法

```python
# 使用便捷方法
replan_output = orchestrator.check_and_replan(
    current_plan=current_plan,
    failed_actions=["open_door"],
    current_context=context,  # 可选
    new_beliefs=[]             # 可选
)

if replan_output.success:
    # 使用新计划继续执行
    new_plan = replan_output.new_plan
```

---

## 完整示例

### 示例 10：完整的任务执行流程

```python
import asyncio
from brain.planning.orchestrator import PlanningOrchestrator
from brain.planning.interfaces import PlanningInput
from brain.cognitive.world_model.planning_context import PlanningContext
from brain.cognitive.world_model.belief.belief import Belief

async def execute_task():
    # 1. 初始化
    orchestrator = PlanningOrchestrator(
        platform="ugv",
        use_cognitive_layer=True,
        enable_replanning=True
    )

    # 2. 设置初始环境
    context = PlanningContext(
        current_position={"x": 0.0, "y": 0.0, "z": 0.0},
        current_heading=0.0,
        obstacles=[],
        targets=[
            {"type": "object", "name": "cup", "position": {"x": 5.0, "y": 3.0}}
        ],
        points_of_interest=[
            {"name": "kitchen", "position": {"x": 5.0, "y": 3.0}}
        ],
        weather={"condition": "clear"},
        battery_level=85.0,
        signal_strength=90.0,
        available_paths=[],
        constraints=[],
        recent_changes=[],
        risk_areas=[]
    )

    beliefs = [
        Belief(id="1", content="杯子在厨房", confidence=0.95)
    ]

    orchestrator.update_context(context, beliefs)

    # 3. 规划任务
    input_data = PlanningInput(
        command="去厨房拿杯水",
        planning_context=context,
        beliefs=beliefs
    )

    output = orchestrator.plan(input_data)

    if not output.is_successful():
        print(f"规划失败: {output.rejection_reason}")
        return

    print(f"规划成功！")
    print(f"预计时长: {output.estimated_duration:.1f}秒")

    # 4. 模拟执行（简化）
    plan = output.plan_state
    for node_id, node in plan.nodes.items():
        print(f"执行: {node.name}")

        # 模拟失败
        if node.name == "open_door":
            print(f"  ✗ 失败: 门卡住了")

            # 触发重规划
            from brain.planning.interfaces import ReplanningInput
            replan_output = orchestrator.check_and_replan(
                current_plan=plan,
                failed_actions=[node_id],
                current_context=context
            )

            if replan_output.success:
                print(f"  ✓ 重规划成功: {replan_output.reason}")
                # 使用新计划继续...
            else:
                print(f"  ✗ 重规划失败: {replan_output.reason}")
                break

    # 5. 获取统计
    stats = orchestrator.get_replanning_statistics()
    print(f"\n统计: {stats}")

# 运行
asyncio.run(execute_task())
```

### 示例 11：带错误处理的完整流程

```python
from brain.planning.interfaces import PlanningStatus

async def robust_execution(orchestrator, command, context, beliefs):
    """带错误处理的执行函数"""

    # 更新上下文
    orchestrator.update_context(context, beliefs)

    # 规划
    input_data = PlanningInput(
        command=command,
        planning_context=context,
        beliefs=beliefs
    )

    output = orchestrator.plan(input_data)

    # 处理各种状态
    if output.is_successful():
        print(f"✓ 规划成功")
        return output.plan_state

    elif output.needs_clarification():
        print(f"? 需要澄清: {output.clarification_request}")
        # 向用户请求澄清...
        return None

    elif output.is_rejected():
        print(f"✗ 被拒绝: {output.rejection_reason}")
        # 可能需要重新评估
        return None

    else:  # FAILURE
        print(f"✗ 规划失败")
        return None
```

### 示例 12：监控环境变化

```python
class EnvironmentMonitor:
    """环境变化监控器"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.last_context = None

    def check_changes(self, current_context, current_plan):
        """检查环境变化并触发重规划"""

        if self.last_context is None:
            self.last_context = current_context
            return None

        # 检测变化
        changes = []

        # 检查障碍物变化
        current_obstacles = set(o.get('type', '') for o in current_context.obstacles)
        last_obstacles = set(o.get('type', '') for o in self.last_context.obstacles)

        if current_obstacles != last_obstacles:
            changes.append({
                "type": "obstacles_changed",
                "description": "障碍物数量变化"
            })

        # 检查电量变化
        if current_context.battery_level < 20.0:
            changes.append({
                "type": "low_battery",
                "description": "电量低于20%",
                "severity": "high"
            })

        # 如果有变化，触发重规划
        if changes:
            from brain.planning.interfaces import ReplanningInput
            replan_input = ReplanningInput(
                current_plan=current_plan,
                environment_changes=changes,
                trigger_reason="环境变化",
                urgency="high"
            )

            return self.orchestrator.replan(replan_input)

        self.last_context = current_context
        return None

# 使用
monitor = EnvironmentMonitor(orchestrator)
replan_output = monitor.check_changes(new_context, current_plan)
```

---

## 调试技巧

### 查看计划详情

```python
# 打印所有节点
def print_plan(plan_state):
    for node_id, node in plan_state.nodes.items():
        indent = "  " * node.depth
        print(f"{indent}[{node.status.value}] {node.name}")
        if node.action:
            print(f"{indent}  action: {node.action}")
        if node.parameters:
            print(f"{indent}  params: {node.parameters}")

print_plan(output.plan_state)
```

### 查看输入摘要

```python
# 打印规划输入摘要
input_data = PlanningInput(command="去厨房拿杯水", ...)
print(input_data.to_summary())

# 输出:
# 指令: 去厨房拿杯水
# 时间: 10:30:45
# 推理模式: planning
# 推理置信度: 0.90
# 信念数量: 3
```

### 查看重规划决策

```python
from brain.planning.intelligent import ReplanningManager

manager = ReplanningManager(world_model=...)

# ...检测变化...

decision = manager.make_replanning_decision(replan_input, changes)

print(f"决策: {decision.strategy.value}")
print(f"原因: {decision.reason}")
print(f"置信度: {decision.confidence:.2f}")
print(f"紧急度: {decision.urgency}")
```

---

## 最佳实践

1. **始终检查返回状态**
   ```python
   if output.is_successful():
       # 处理成功
   ```

2. **使用认知层集成模式获取更好的结果**
   ```python
   orchestrator = PlanningOrchestrator(use_cognitive_layer=True)
   ```

3. **定期更新上下文**
   ```python
   orchestrator.update_context(context, beliefs)
   ```

4. **启用重规划应对变化**
   ```python
   orchestrator = PlanningOrchestrator(enable_replanning=True)
   ```

5. **记录日志用于调试**
   ```python
   for log in output.planning_log:
       logger.info(log)
   ```
