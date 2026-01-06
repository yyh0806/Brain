# 异步测试说明

## 什么是异步测试？

异步测试是测试**异步操作**的测试方法。在认知层中，很多操作都是异步的，因为它们需要：
- 等待LLM响应（推理引擎）
- 处理感知数据流
- 更新世界模型
- 与外部系统通信

---

## 为什么需要异步测试？

### 1. **真实的异步行为**
认知层的很多方法都是异步的（`async def`），需要用异步测试来验证：

```python
# 同步测试无法测试异步方法
def test_wrong():
    result = cot_engine.reason(...)  # ❌ 错误！返回的是协程对象，不是结果

# 异步测试正确处理
async def test_correct():
    result = await cot_engine.reason(...)  # ✅ 正确！等待结果
```

### 2. **并发测试**
异步测试可以验证多个并发操作的正确性：
- 同时处理多个感知数据
- 并发的信念更新
- 多个推理任务并行执行

### 3. **性能验证**
异步测试可以测量真实的异步性能：
```python
start = time.time()
result = await cot_engine.reason(...)
elapsed = time.time() - start
assert elapsed < 1.0  # 验证响应时间
```

---

## 认知层的异步测试分类

### 单元测试异步测试 (23个)

#### 1. WorldModel异步测试 (3个)

**位置**: `tests/unit/cognitive/test_world_model.py`

**测试内容**:
```python
async def test_update_from_perception_basic():
    """测试：感知数据基本更新"""
    # 验证机器人位置更新
    # 验证变化检测

async def test_update_occupancy_grid():
    """测试：占据栅格更新"""
    # 验证地图数据更新
    # 验证地图分辨率

async def test_update_tracked_objects():
    """测试：物体跟踪更新"""
    # 验证障碍物跟踪
    # 验证物体位置
```

**为什么是异步？**
- WorldModel的`update_from_perception()`可能触发异步操作
- 未来的版本可能需要异步处理感知数据流

#### 2. CognitiveInterface异步测试 (20个)

**位置**: `tests/unit/cognitive/test_cognitive_interface.py`

**测试内容**:

**a) 感知处理接口 (3个)**
```python
async def test_process_perception_basic():
    """测试：基本感知处理"""
    output = await cognitive_layer.process_perception(data)
    # 验证输出结构
    # 验证规划上下文

async def test_process_perception_updates_world_model():
    """测试：感知更新世界模型"""
    await cognitive_layer.process_perception(data)
    # 验证世界模型状态更新

async def test_process_perception_with_none_data():
    """测试：处理空数据"""
    # 验证错误处理
```

**b) 信念更新接口 (2个)**
```python
async def test_update_belief_success():
    """测试：成功观察后更新信念"""
    observation = ObservationResult(SUCCESS)
    await cognitive_layer.update_belief(observation)
    # 验证置信度增加

async def test_update_belief_failure():
    """测试：失败观察后更新信念"""
    observation = ObservationResult(FAILURE)
    await cognitive_layer.update_belief(observation)
    # 验证置信度降低
```

**c) 推理接口 (6个)**
```python
async def test_reason_basic_query():
    """测试：基本推理查询"""
    result = await cognitive_layer.reason("前进")
    # 验证推理结果

async def test_reason_with_complex_context():
    """测试：复杂上下文推理"""
    result = await cognitive_layer.reason("规划路径", context=...)
    # 验证复杂度评估
```

**d) 对话接口 (6个)**
```python
async def test_dialogue_clarification():
    """测试：澄清对话"""
    response = await cognitive_layer.start_dialogue("去那边")
    # 验证澄清逻辑

async def test_dialogue_confirmation():
    """测试：确认对话"""
    response = await cognitive_layer.confirm_action("跳跃")
    # 验证确认流程
```

**e) 监控控制接口 (3个)**
```python
async def test_start_monitoring():
    """测试：启动监控"""
    await cognitive_layer.start_monitoring()
    # 验证监控状态

async def test_stop_monitoring():
    """测试：停止监控"""
    await cognitive_layer.stop_monitoring()
    # 验证清理
```

**为什么是异步？**
- `process_perception()` 需要等待感知数据处理完成
- `update_belief()` 可能触发异步信念修正
- `reason()` 调用异步的CoT推理引擎
- `start_dialogue()` 等待LLM响应
- 监控启动/停止是异步操作

#### 3. ReasoningEngine异步测试 (6个)

**位置**: `tests/unit/cognitive/test_reasoning_engine.py`

**测试内容**:
```python
async def test_quick_reasoning():
    """测试：快速推理"""
    result = await cot_engine.reason("前进", ...)
    # 验证简单任务快速处理

async def test_full_cot_reasoning():
    """测试：完整CoT推理"""
    result = await cot_engine.reason("规划路径", ...)
    # 验证完整推理链
```

**为什么是异步？**
- CoT引擎调用LLM是异步的（网络请求）
- `cot_engine.reason()` 返回协程，需要await

#### 4. DialogueManager异步测试 (4个)

**位置**: `tests/unit/cognitive/test_dialogue_manager.py`

**测试内容**:
```python
async def test_clarify_ambiguous_command():
    """测试：澄清模糊命令"""
    response = await manager.clarify("去那边")
    # 验证澄清问题

async def test_request_confirmation():
    """测试：请求确认"""
    response = await manager.request_confirm("危险操作")
    # 验证确认提示
```

**为什么是异步？**
- 对话管理器调用LLM生成对话
- 需要等待LLM响应

---

### 功能测试异步测试 (28个)

#### 1. 感知流程测试 (5个)

**位置**: `tests/functional/cognitive/test_perception_flow.py`

**测试内容**:
```python
async def test_basic_perception_flow():
    """测试：感知→认知完整流程"""
    # 感知数据 → 认知处理 → 规划输出
    output = await cognitive_layer.process_perception(data)
    # 验证完整数据流

async def test_continuous_updates():
    """测试：连续感知更新"""
    # 模拟时间序列的感知数据
    for data in perception_sequence:
        await cognitive_layer.process_perception(data)
    # 验证状态连续性
```

**为什么是异步？**
- 测试完整的异步数据流
- 验证异步状态更新

#### 2. 信念更新流程测试 (4个)

**位置**: `tests/functional/cognitive/test_belief_update_flow.py`

**测试内容**:
```python
async def test_observation_success_increases_confidence():
    """测试：成功观察增加置信度"""
    # 执行操作 → 成功 → 置信度上升
    await cognitive_layer.execute_and_observe(action)
    # 验证信念状态

async def test_belief_falsification_flow():
    """测试：信念证伪流程"""
    # 多次失败 → 信念被移除
    for _ in range(5):
        await cognitive_layer.execute_and_observe(failing_action)
    # 验证信念被清理
```

**为什么是异步？**
- 执行操作和观察是异步的
- 需要等待操作完成

#### 3. 推理流程测试 (6个)

**位置**: `tests/functional/cognitive/test_reasoning_flow.py`

**测试内容**:
```python
async def test_simple_reasoning_flow():
    """测试：简单推理流程"""
    # 简单任务 → 快速推理
    result = await cognitive_layer.reason("前进")
    # 验证推理效率

async def test_complex_reasoning_flow():
    """测试：复杂推理流程"""
    # 复杂任务 → 完整CoT推理
    result = await cognitive_layer.reason("规划避障路径")
    # 验证推理深度
```

**为什么是异步？**
- 测试完整的异步推理流程
- 验证LLM调用和响应

#### 4. 对话流程测试 (5个)

**位置**: `tests/functional/cognitive/test_dialogue_flow.py`

**测试内容**:
```python
async def test_clarification_flow():
    """测试：澄清流程"""
    # 模糊命令 → 澄清对话 → 理解意图
    response = await cognitive_layer.handle_command("过去")
    # 验证对话交互

async def test_progress_reporting_flow():
    """测试：进度报告流程"""
    # 长时间操作 → 进度更新
    async for update in cognitive_layer.execute_with_progress(action):
        # 验证进度报告
```

**为什么是异步？**
- 对话交互是异步的
- 进度报告使用异步迭代器

#### 5. 变化检测流程测试 (8个)

**位置**: `tests/functional/cognitive/test_change_detection_flow.py`

**测试内容**:
```python
async def test_new_obstacle_detection():
    """测试：新障碍物检测"""
    # 环境变化 → 检测 → 触发重规划
    await cognitive_layer.process_perception(new_obstacle_data)
    # 验证检测逻辑

async def test_significant_changes_trigger_replan():
    """测试：显著变化触发重规划"""
    # 重大变化 → 重规划信号
    changes = await cognitive_layer.detect_changes(significant_data)
    # 验证重规划触发
```

**为什么是异步？**
- 变化检测可能需要异步处理
- 重规划触发是异步操作

---

## 异步测试的关键概念

### 1. async/await语法

```python
# 定义异步测试
@pytest.mark.asyncio
async def test_something():
    # 使用await等待异步操作
    result = await some_async_function()
    assert result is not None
```

### 2. 异步Fixture

```python
# 异步fixture
@pytest.fixture
async def async_resource():
    # 异步初始化
    resource = await create_resource_async()
    yield resource
    # 异步清理
    await resource.cleanup_async()
```

### 3. pytest-asyncio配置

```ini
# pytest.ini
[tool:pytest]
asyncio_mode = auto  # 自动识别async测试
asyncio_default_fixture_loop_scope = function  # fixture作用域
```

---

## 当前状态

### 单元测试异步测试 (23个)
- **状态**: ⏭️ 被跳过
- **原因**: pytest-asyncio未正确配置
- **影响**: 无法测试异步功能
- **优先级**: P0 - 高

### 功能测试异步测试 (28个)
- **状态**: ⏭️ 被跳过
- **原因**: 同上
- **影响**: 无法验证端到端流程
- **优先级**: P0 - 高

---

## 如何启用异步测试

### 方法1: 配置pytest.ini (推荐)

```ini
# pytest.ini
[tool:pytest]
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
markers =
    asyncio: Async tests
    unit: Unit tests
    integration: Integration tests
```

### 方法2: 命令行参数

```bash
pytest tests/unit/cognitive/ -v --asyncio-mode=auto
```

### 方法3: 转换为同步测试 (不推荐)

如果确实不需要异步，可以转换为同步：

```python
# 修改前 (异步)
@pytest.mark.asyncio
async def test_update():
    result = await world_model.update(data)

# 修改后 (同步)
def test_update():
    result = world_model.update(data)
```

---

## 异步测试的价值

### 1. **测试真实性**
异步测试能更真实地反映实际运行情况，包括：
- 并发处理
- 异步I/O
- 事件循环行为

### 2. **发现竞态条件**
```python
async def test_concurrent_updates():
    """测试：并发更新不会冲突"""
    # 同时更新多个信念
    tasks = [
        cognitive_layer.update_belief(obs1),
        cognitive_layer.update_belief(obs2),
        cognitive_layer.update_belief(obs3),
    ]
    await asyncio.gather(*tasks)
    # 验证没有数据竞争
```

### 3. **验证异步资源管理**
```python
async def test_cleanup_on_error():
    """测试：错误时正确清理资源"""
    with pytest.raises(Exception):
        await cognitive_layer.process_perception(invalid_data)
    # 验证资源被正确释放
```

---

## 总结

**异步测试测试什么？**

1. **异步方法调用** - `async def`定义的方法
2. **LLM交互** - 等待LLM响应
3. **感知数据流** - 异步处理感知数据
4. **世界模型更新** - 异步状态更新
5. **信念修正** - 异步置信度调整
6. **对话管理** - 异步对话交互
7. **并发操作** - 多个异步操作同时执行
8. **端到端流程** - 完整的异步数据流

**为什么重要？**

- ✅ 真实性：反映实际异步行为
- ✅ 完整性：测试完整的功能流程
- ✅ 可靠性：发现异步相关的问题
- ✅ 性能：验证异步性能指标

**当前状态**: 51个异步测试已编写完成，等待配置启用

---

**文档创建**: 2026-01-05
**相关文件**:
- pytest.ini - 配置文件
- tests/unit/cognitive/ - 单元异步测试
- tests/functional/cognitive/ - 功能异步测试
