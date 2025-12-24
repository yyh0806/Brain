# 视觉语言模型感知文档

Brain感知模块集成视觉语言模型(VLM)，提供基于自然语言的高级场景理解和目标搜索能力。

## VLM感知器使用

### 基本使用

```python
from brain.perception.vlm.vlm_perception import VLMPerception

# 创建VLM感知器
vlm = VLMPerception(
    model="llava:latest",        # Ollama模型名称
    ollama_host="http://localhost:11434",  # Ollama服务地址
    use_yolo=False              # 是否使用YOLO辅助检测
)

# 获取RGB图像
rgb_image = get_rgb_image()  # (H, W, 3) numpy数组

# 描述当前场景
scene = await vlm.describe_scene(rgb_image)

# 访问场景描述
print(f"场景概要: {scene.summary}")
print(f"检测到的物体: {len(scene.objects)}")
print(f"空间关系: {scene.spatial_relations}")
print(f"导航提示: {scene.navigation_hints}")
```

### 目标搜索

```python
# 根据描述搜索目标
target = "红色的门"
search_result = await vlm.find_target(rgb_image, target)

if search_result.found:
    print(f"找到目标: {target}")
    print(f"位置: {search_result.best_match.position_description}")
    print(f"建议动作: {search_result.suggested_action}")
    print(f"置信度: {search_result.confidence}")
else:
    print(f"未找到目标: {target}")
    print(f"说明: {search_result.explanation}")
```

### 空间查询

```python
# 询问空间相关问题
query = "门在哪个方向?"
answer = await vlm.answer_spatial_query(rgb_image, query)
print(f"回答: {answer}")

query = "左侧是否有障碍物?"
answer = await vlm.answer_spatial_query(rgb_image, query)
print(f"回答: {answer}")
```

## 数据结构

### DetectedObject

```python
from brain.perception.vlm.vlm_perception import DetectedObject, DetectionSource

obj = DetectedObject(
    id="obj_1",                    # 物体ID
    label="门",                     # 物体标签
    confidence=0.85,                # 置信度
    bbox=BoundingBox(x=0.5, y=0.6, width=0.2, height=0.3),  # 边界框(归一化)
    description="建筑入口的玻璃门",    # 详细描述
    position_description="前方中央偏右", # 位置描述
    estimated_distance=10.5,        # 估计距离(米)
    estimated_direction="前方偏右",    # 方向描述
    source=DetectionSource.VLM,      # 检测来源
    attributes={"color": "透明"}      # 其他属性
)

# 转换为像素坐标
x1, y1, x2, y2 = obj.bbox.to_pixel(640, 480)  # 图像尺寸640x480

# 获取中心像素坐标
cx, cy = obj.bbox.center_pixel(640, 480)

# 转换为字典
obj_dict = obj.to_dict()
```

### SceneDescription

```python
from brain.perception.vlm.vlm_perception import SceneDescription

scene = SceneDescription(
    summary="这是一个室内大厅环境，前方有一个玻璃门",
    objects=[
        DetectedObject(id="obj_1", label="门", ...),
        DetectedObject(id="obj_2", label="植物", ...)
    ],
    spatial_relations=["门在前方中央", "植物在门的右侧"],
    navigation_hints=["前方可以通行", "左侧有空间"],
    potential_targets=["门", "入口"],
    raw_response="原始VLM响应文本"
)

# 转换为字典
scene_dict = scene.to_dict()
```

### TargetSearchResult

```python
from brain.perception.vlm.vlm_perception import TargetSearchResult

result = TargetSearchResult(
    found=True,                          # 是否找到目标
    target_description="红色的门",           # 目标描述
    matched_objects=[detected_obj],        # 匹配的物体列表
    best_match=detected_obj,              # 最佳匹配
    suggested_action="直行10米然后左转",    # 建议动作
    confidence=0.85,                     # 置信度
    explanation="目标位于前方，可以直接前进"  # 说明
)
```

## VLM后端配置

### Ollama配置

```python
# 使用本地Ollama模型
vlm = VLMPerception(
    model="llava:latest",
    ollama_host="http://localhost:11434"
)

# 使用特定模型
vlm = VLMPerception(
    model="llava:13b",
    ollama_host="http://192.168.1.100:11434"
)
```

### YOLO辅助

```python
# 使用YOLO辅助检测
vlm = VLMPerception(
    model="llava:latest",
    use_yolo=True,
    yolo_model="yolov8n.pt"  # YOLO模型文件路径
)

# 获取检测历史
history = vlm.get_detection_history(count=10)
for obj in history:
    print(f"检测到: {obj.label} (来源: {obj.source.value})")
```

## 提示词定制

### 场景分析提示词

```python
# 自定义场景分析提示词
custom_scene_prompt = """分析这张图像，特别注意：
1. 可通行路径
2. 障碍物位置
3. 目标位置(门、入口等)
4. 安全距离

请以JSON格式回复..."""

# 在VLM类中重写提示词
class CustomVLMPerception(VLMPerception):
    SCENE_ANALYSIS_PROMPT = custom_scene_prompt
```

### 目标搜索提示词

```python
# 自定义目标搜索提示词
custom_target_prompt = """在这张图像中寻找 {target}。
特别注意：
1. 目标的精确位置
2. 与目标的距离
3. 到达目标的路径
4. 可能的障碍物

请以JSON格式回复..."""

class CustomVLMPerception(VLMPerception):
    TARGET_SEARCH_PROMPT = custom_target_prompt
```

## 边界框处理

### 边界框估算

```python
# 根据位置描述估算边界框
from brain.perception.vlm.vlm_perception import BoundingBox

bbox = BoundingBox(x=0.5, y=0.6, width=0.2, height=0.3)  # 归一化坐标

# 转换为像素坐标
x1, y1, x2, y2 = bbox.to_pixel(640, 480)  # 图像尺寸640x480

# 转换为中心点坐标
cx, cy = bbox.center_pixel(640, 480)
```

### 检测融合

```python
# 融合VLM和YOLO检测结果
vlm = VLMPerception(use_yolo=True)
scene = await vlm.describe_scene(image)

# 检查融合结果
for obj in scene.objects:
    if obj.source == DetectionSource.FUSED:
        print(f"融合检测: {obj.label}, 置信度: {obj.confidence}")
    elif obj.source == DetectionSource.VLM:
        print(f"VLM检测: {obj.label}, 置信度: {obj.confidence}")
    elif obj.source == DetectionSource.YOLO:
        print(f"YOLO检测: {obj.label}, 置信度: {obj.confidence}")
```

## 性能优化

### 缓存使用

```python
# 获取最近场景描述
last_scene = vlm.get_last_scene()
if last_scene:
    print(f"最近场景: {last_scene.summary}")

# 比较场景变化
if last_scene and scene.summary == last_scene.summary:
    print("场景无显著变化")
else:
    print("场景已更新")
```

### 异步处理

```python
import asyncio

# 并行处理多个查询
async def process_queries(image, queries):
    tasks = []
    for query in queries:
        task = vlm.answer_spatial_query(image, query)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

queries = ["门在哪里?", "左侧有什么?", "前方有障碍物吗?"]
answers = await process_queries(image, queries)
for query, answer in zip(queries, answers):
    print(f"问: {query}")
    print(f"答: {answer}")
```

## 故障处理

### 模型不可用

```python
try:
    scene = await vlm.describe_scene(image)
except Exception as e:
    print(f"VLM调用失败: {e}")
    # 使用备用方案
    scene = SceneDescription(
        summary="VLM不可用，使用备用检测",
        objects=[],
        spatial_relations=[],
        navigation_hints=["使用基本传感器数据导航"]
    )
```

### 结果解析失败

```python
# 检查原始响应
scene = await vlm.describe_scene(image)
raw_response = scene.raw_response

# 如果解析失败，可以使用原始响应
if not scene.objects:
    print("解析失败，使用原始响应:")
    print(raw_response)
```

## 应用示例

### 场景理解与导航

```python
async def understand_and_navigate(image, target):
    # 理解场景
    scene = await vlm.describe_scene(image)
    print(f"当前场景: {scene.summary}")
    
    # 搜索目标
    result = await vlm.find_target(image, target)
    
    if result.found:
        print(f"找到 {target}: {result.best_match.position_description}")
        print(f"建议: {result.suggested_action}")
        
        # 询问导航路径
        query = f"如何到达{target}?"
        nav_answer = await vlm.answer_spatial_query(image, query)
        print(f"导航: {nav_answer}")
    else:
        print(f"未找到 {target}: {result.explanation}")

# 使用示例
await understand_and_navigate(image, "门的入口")
```

### 交互式探索

```python
async def interactive_explore():
    while True:
        # 获取当前场景图像
        image = get_current_image()
        
        # 描述场景
        scene = await vlm.describe_scene(image)
        print("\n" + "="*50)
        print("场景描述:")
        print(f"  概要: {scene.summary}")
        print(f"  物体: {', '.join([obj.label for obj in scene.objects])}")
        print(f"  导航提示: {', '.join(scene.navigation_hints)}")
        
        # 用户查询
        query = input("\n请输入查询(或'退出'): ")
        if query.lower() in ['退出', 'exit', 'quit']:
            break
            
        # 回答查询
        answer = await vlm.answer_spatial_query(image, query)
        print(f"\n回答: {answer}")

# 启动交互式探索
# asyncio.run(interactive_explore())
```

## 相关文档

- [目标检测](object_detection.md)
- [多传感器融合](sensor_fusion.md)
- [Isaac Sim集成](isaac_sim_integration.md)




