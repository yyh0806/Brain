# 认知层故障排查指南

> **版本**: v2.0
> **最后更新**: 2026-01-14

---

## 目录

1. [快速诊断](#快速诊断)
2. [SLAM集成问题](#slam集成问题)
3. [性能问题](#性能问题)
4. [内存问题](#内存问题)
5. [推理引擎问题](#推理引擎问题)
6. [测试失败](#测试失败)
7. [调试工具](#调试工具)

---

## 快速诊断

### 健康检查脚本

```bash
#!/bin/bash
# 快速健康检查

echo "=== 认知层健康检查 ==="

# 1. Python环境
echo "[1/6] 检查Python环境..."
python3 --version
if [ $? -eq 0 ]; then
    echo "  ✓ Python环境正常"
else
    echo "  ✗ Python环境异常"
    exit 1
fi

# 2. 依赖检查
echo "[2/6] 检查依赖..."
python3 -c "import numpy, rclpy, asyncio"
if [ $? -eq 0 ]; then
    echo "  ✓ 核心依赖正常"
else
    echo "  ✗ 依赖缺失"
    exit 1
fi

# 3. 认知层模块
echo "[3/6] 检查认知层模块..."
python3 -c "from brain.cognitive.interface import CognitiveLayer"
if [ $? -eq 0 ]; then
    echo "  ✓ 认知层模块正常"
else
    echo "  ✗ 认知层模块异常"
    exit 1
fi

# 4. SLAM集成
echo "[4/6] 检查SLAM集成..."
python3 -c "from slam_integration.src import SLAMManager"
if [ $? -eq 0 ]; then
    echo "  ✓ SLAM集成正常"
else
    echo "  ⚠ SLAM集成不可用（这是正常的，如果SLAM未安装）"
fi

# 5. 配置文件
echo "[5/6] 检查配置文件..."
if [ -f "config/slam/slam_config.yaml" ]; then
    echo "  ✓ 配置文件存在"
else
    echo "  ⚠ 配置文件缺失"
fi

# 6. 测试覆盖
echo "[6/6] 检查测试..."
if [ -d "tests/cognitive" ]; then
    echo "  ✓ 测试文件存在"
else
    echo "  ⚠ 测试文件缺失"
fi

echo ""
echo "=== 健康检查完成 ==="
```

保存为`scripts/health_check.sh`并运行：

```bash
chmod +x scripts/health_check.sh
./scripts/health_check.sh
```

---

## SLAM集成问题

### 问题1: SLAM地图不可用

**症状**：
```
ValueError: SLAM地图尚未可用
```

**诊断步骤**：

1. 检查SLAM节点是否运行：

```bash
ros2 node list | grep slam
```

2. 检查地图话题：

```bash
ros2 topic list | grep map
ros2 topic hz /map
```

3. 查看地图消息：

```bash
ros2 topic echo /map --once
```

**解决方案**：

**方案A：启动SLAM节点**

```bash
# 启动FAST-LIVO
ros2 launch slam_integration slam_integration.launch.py
```

**方案B：使用MockSLAM进行测试**

```python
from tests.cognitive.mock_slam import MockSLAMManager

# 使用MockSLAM
slam_manager = MockSLAMManager()
```

### 问题2: 坐标转换失败

**症状**：
```
ValueError: 无法进行坐标转换
```

**诊断步骤**：

1. 检查TF树：

```bash
ros2 run tf2_tools view_frames
```

2. 检查坐标变换：

```bash
ros2 run tf2_ros tf2_echo map base_link
```

**解决方案**：

确保TF树正确配置：

```yaml
# config/slam/tf_config.yaml
frames:
  map_frame: "map"
  odom_frame: "odom"
  base_link_frame: "base_link"
  sensor_frames:
    - "camera_link"
    - "lidar_link"
```

### 问题3: SLAM更新频率低

**症状**：地图更新频率<1Hz

**诊断步骤**：

```bash
# 监控地图话题频率
ros2 topic hz /map
```

**解决方案**：

调整SLAM参数：

```yaml
# config/slam/slam_config.yaml
slam:
  mapping:
    publish_frequency: 10.0  # Hz
    update_frequency: 10.0   # Hz
```

---

## 性能问题

### 问题1: WorldModel更新慢

**症状**：更新时间>10ms

**诊断步骤**：

```python
import time
from brain.cognitive.world_model.modular_world_model import ModularWorldModel

model = ModularWorldModel()
await model.initialize()

# 测量更新时间
start = time.time()
await model.update_from_perception(perception_data)
duration = (time.time() - start) * 1000

print(f"更新时间: {duration:.2f} ms")
```

**解决方案**：

**方案A：启用增量更新**

```python
config = {
    "enable_incremental_update": True,
    "hash_based_detection": True
}
model = ModularWorldModel(config=config)
```

**方案B：减少物体数量**

```python
config = {
    "max_semantic_objects": 300,  # 减少最大物体数
    "object_ttl": 180.0  # 缩短TTL
}
```

### 问题2: 内存持续增长

**症状**：长时间运行内存持续增长

**诊断步骤**：

```python
import psutil
import os

process = psutil.Process(os.getpid())

# 监控内存
for i in range(100):
    await model.update_from_perception(perception_data)
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"内存: {memory_mb:.1f} MB")
```

**解决方案**：

**方案A：启用自动清理**

```python
config = {
    "auto_cleanup": True,
    "cleanup_threshold": 0.8,  # 使用率达到80%时清理
    "cleanup_interval": 60.0  # 每60秒清理一次
}
```

**方案B：手动触发清理**

```python
# 定期清理
import asyncio

async def periodic_cleanup():
    while True:
        await asyncio.sleep(60)
        expired_count = model.semantic_layer.object_manager.cleanup_expired()
        print(f"清理了{expired_count}个过期物体")
```

### 问题3: 推理缓存命中率低

**症状**：缓存命中率<50%

**诊断步骤**：

```python
from brain.cognitive.reasoning.async_cot_engine import AsyncCoTEngine

engine = AsyncCoTEngine()
engine.start()

# 执行推理
await engine.reason("查询1", {}, "default")
await engine.reason("查询1", {}, "default")  # 应该命中缓存

# 查看统计
stats = engine.get_statistics()
print(f"缓存命中率: {stats['cache_hit_rate']:.1%}")
```

**解决方案**：

**方案A：增加缓存大小**

```python
engine = AsyncCoTEngine(cache_size=200)  # 增加缓存
```

**方案B：优化缓存键**

```python
# 只使用关键字段计算缓存键
def compute_cache_key(query, mode, context):
    key_data = {
        "query": query,
        "mode": mode
        # 移除context的hash，只使用关键字段
    }
    return hashlib.md5(str(key_data).encode()).hexdigest()
```

---

## 内存问题

### 问题1: 内存泄漏

**症状**：内存持续增长，不会释放

**诊断工具**：

```bash
# 使用memory_profiler
pip3 install memory_profiler

python3 -m memory_profiler your_script.py
```

**解决方案**：

检查常见泄漏点：

```python
# 1. 全局列表/字典
self.objects = []  # ❌ 无界增长
self.objects = LRUCache(max_size=1000)  # ✅ 有界

# 2. 循环引用
class A:
    def __init__(self):
        self.b = B(self)

class B:
    def __init__(self, a):
        self.a = a  # ❌ 循环引用

# 3. 未清理的回调
def callback(data):
    self.cache.append(data)  # ❌ 无界增长
```

### 问题2: 对象过早清理

**症状**：物体在TTL前被清理

**诊断步骤**：

```python
manager = SemanticObjectManager(object_ttl=300.0)

obj_id = manager.add_or_update(obj)

# 立即检查
obj = manager.get(obj_id)
print(f"对象存在: {obj is not None}")

# 查看统计
stats = manager.get_statistics()
print(f"缓存大小: {stats['size']}")
```

**解决方案**：

**方案A：增加TTL**

```python
manager = SemanticObjectManager(object_ttl=600.0)  # 10分钟
```

**方案B：禁用LRU淘汰**

```python
manager = SemanticObjectManager(
    max_objects=10000,  # 设置很大的上限
    object_ttl=300.0
)
```

---

## 推理引擎问题

### 问题1: 推理队列阻塞

**症状**：推理请求长时间无响应

**诊断步骤**：

```python
engine = AsyncCoTEngine(max_queue_size=10)

# 查看队列状态
stats = engine.get_statistics()
print(f"队列大小: {stats['queue_size']}")
print(f"总请求: {stats['total_requests']}")
print(f"已处理: {stats['total_processed']}")
```

**解决方案**：

**方案A：增加工作线程**

```python
engine = AsyncCoTEngine(num_workers=4)  # 增加工作线程
```

**方案B：增加队列大小**

```python
engine = AsyncCoTEngine(max_queue_size=50)  # 增加队列
```

**方案C：使用优先级**

```python
# 高优先级请求优先处理
result = await engine.reason(
    query="紧急查询",
    context={},
    mode="default",
    priority=0  # 数字越小优先级越高
)
```

### 问题2: 推理结果错误

**症状**：推理结论不合理

**诊断步骤**：

```python
result = await engine.reason("查询", {}, "default")

# 查看推理链
print("推理链:")
for i, step in enumerate(result.chain):
    print(f"  {i+1}. {step}")

print(f"\n结论: {result.conclusion}")
print(f"置信度: {result.confidence}")
```

**解决方案**：

**方案A：实现自定义推理逻辑**

```python
def custom_reasoning_chain(query, context, mode):
    """自定义推理链生成"""
    chain = []

    if "位置" in query:
        chain.append("分析位置查询")
        # ... 自定义逻辑
    else:
        chain.append("分析一般查询")

    return chain

# 注入到引擎
engine._generate_reasoning_chain = custom_reasoning_chain
```

---

## 测试失败

### 问题1: 导入错误

**症状**：
```
ImportError: No module named 'brain.cognitive...'
```

**解决方案**：

```bash
# 添加项目路径到PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/media/yangyuhui/CODES1/Brain"

# 或者使用python3 -m
python3 -m pytest tests/cognitive/
```

### 问题2: ROS2相关测试失败

**症状**：
```
NameError: name 'rclpy' is not defined
```

**解决方案**：

```bash
# Source ROS2环境
source /opt/ros/humble/setup.bash

# 运行测试
python3 -m pytest tests/cognitive/
```

### 问题3: SLAM测试超时

**症状**：测试等待SLAM响应超时

**解决方案**：

```bash
# 使用MockSLAM进行测试
export USE_MOCK_SLAM=1

python3 -m pytest tests/cognitive/
```

---

## 调试工具

### 1. 性能分析器

```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()

    # ... 要分析的代码 ...

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(20)  # 打印前20个最耗时的函数

profile_function()
```

### 2. 内存分析器

```bash
# 安装memory_profiler
pip3 install memory_profiler

# 分析内存使用
python3 -m memory_profiler your_script.py
```

### 3. 日志调试

```python
import logging

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d: %(message)s'
)

# 使用日志
logger = logging.getLogger(__name__)
logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告")
logger.error("错误")
```

### 4. 可视化工具

```bash
# 使用RViz2可视化
rviz2 -d config/rviz2/cognitive_debug.rviz

# 查看世界模型状态
ros2 topic echo /cognitive/world_model --once

# 查看变化事件
ros2 topic echo /cognitive/changes
```

---

## 常见错误代码

| 错误代码 | 说明 | 解决方案 |
|---------|------|---------|
| `E001` | SLAM地图不可用 | 启动SLAM节点 |
| `E002` | 坐标转换失败 | 检查TF配置 |
| `E003` | 内存不足 | 增加内存或优化配置 |
| `E004` | 推理队列满 | 增加队列大小 |
| `E005` | 对象未找到 | 检查物体ID |
| `E006` | 缓存键冲突 | 优化缓存键计算 |
| `E007` | 配置文件缺失 | 创建配置文件 |
| `E008` | 依赖缺失 | 安装依赖 |

---

## 获取帮助

### 社区支持

- GitHub Issues: [Brain项目](https://github.com/your-org/Brain/issues)
- 文档: [完整文档](./COGNITIVE_LAYER_GUIDE.md)

### 日志收集

提交问题时，请提供：

1. 完整的错误堆栈
2. 系统环境信息
3. 配置文件
4. 相关日志

```bash
# 收集诊断信息
bash scripts/collect_diagnostic_info.sh > diagnostic_info.txt
```

---

**维护者**: Claude (ultrathink mode)
**版本**: v2.0
**最后更新**: 2026-01-14
