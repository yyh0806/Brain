# 感知层测试体系

## 概述

这是Brain感知层的完整测试体系，包括单元测试、功能测试和端到端测试。

## 测试结构

```
tests/perception/
├── __init__.py           # 测试模块初始化
├── conftest.py           # pytest配置和fixtures
├── run_tests.py          # 测试运行脚本
├── unit/                 # 单元测试
│   └── test_types.py     # 核心数据类型测试
├── functional/           # 功能测试
│   └── test_detection.py # 目标检测功能测试
└── integration/          # 端到端测试
    └── test_world_model.py # 世界模型集成测试
```

## 测试类型

### 1. 单元测试 (unit/)
测试单个类、函数的最小可测试单元。

**覆盖范围:**
- `test_types.py`: 核心数据类型
  - Position2D, Position3D
  - Pose2D, Pose3D
  - Velocity
  - BoundingBox
  - DetectedObject
  - OccupancyGrid

### 2. 功能测试 (functional/)
测试模块级别的功能，确保组件按预期工作。

**覆盖范围:**
- `test_detection.py`: 目标检测器
  - 检测功能
  - 跟踪功能
  - 置信度过滤

### 3. 端到端测试 (integration/)
测试完整的工作流程和模块间的集成。

**覆盖范围:**
- `test_world_model.py`: 世界模型集成
  - 传感器数据更新
  - 多传感器融合
  - 查询接口

## 运行测试

### 运行所有测试
```bash
cd tests/perception
python run_tests.py
```

### 运行特定类型的测试
```bash
# 只运行单元测试
python run_tests.py unit

# 只运行功能测试
python run_tests.py functional

# 只运行集成测试
python run_tests.py integration
```

### 详细输出
```bash
python run_tests.py -v
```

### 运行特定测试
```bash
# 运行匹配关键字的测试
python run_tests.py -k "test_position"

# 运行特定文件的测试
pytest tests/perception/unit/test_types.py::TestPosition3D
```

### 生成代码覆盖率报告
```bash
# 终端输出
python run_tests.py --cov

# HTML报告
python run_tests.py --cov --cov-report=html

# 打开HTML报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Fixtures

测试fixtures在`conftest.py`中定义，包括：

### 数据类Fixtures
- `sample_position_2d`: 示例2D位置
- `sample_position_3d`: 示例3D位置
- `sample_pose_2d`: 示例2D位姿
- `sample_pose_3d`: 示例3D位姿
- `sample_velocity`: 示例速度
- `sample_bounding_box`: 示例边界框
- `sample_detected_object`: 示例检测物体
- `sample_occupancy_grid`: 示例占据栅格
- `sample_scene_description`: 示例场景描述

### 模拟数据Fixtures
- `mock_image`: 模拟图像数据 (480x640x3)
- `mock_depth_map`: 模拟深度图 (480x640)
- `mock_laser_scan`: 模拟激光扫描数据 (360个点)
- `mock_pointcloud`: 模拟点云数据 (1000个点)

### 配置Fixtures
- `perception_config`: 感知层配置字典

### 辅助函数
- `assert_position_equal`: 断言两个位置相等
- `assert_bbox_equal`: 断言两个边界框相等
- `create_mock_perception_data`: 创建模拟感知数据

## 添加新测试

### 1. 单元测试示例
```python
# tests/perception/unit/test_new_module.py
import pytest
from brain.perception.new_module import NewClass

class TestNewClass:
    def test_initialization(self):
        obj = NewClass()
        assert obj is not None

    def test_some_method(self):
        obj = NewClass()
        result = obj.some_method()
        assert result == expected_value
```

### 2. 功能测试示例
```python
# tests/perception/functional/test_new_feature.py
import pytest
from brain.perception.new_feature import NewFeature

class TestNewFeature:
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_data):
        feature = NewFeature()
        result = await feature.process(mock_data)
        assert result is not None
```

### 3. 集成测试示例
```python
# tests/perception/integration/test_new_pipeline.py
import pytest
from brain.perception.module1 import Module1
from brain.perception.module2 import Module2

class TestNewPipeline:
    def test_end_to_end_flow(self, mock_input):
        m1 = Module1()
        m2 = Module2()

        # 完整流程
        intermediate = m1.process(mock_input)
        final = m2.process(intermediate)

        assert final is not None
```

## 最佳实践

1. **测试隔离**: 每个测试应该独立运行，不依赖其他测试的状态
2. **使用Fixtures**: 重复使用fixtures来减少代码重复
3. **清晰的命名**: 测试名称应该清楚地描述测试的内容
4. **一个断言**: 每个测试 ideally 只测试一个东西
5. **异步测试**: 使用`@pytest.mark.asyncio`装饰器测试异步函数
6. **Mock外部依赖**: 使用mock来隔离外部依赖
7. **覆盖率目标**: 单元测试目标覆盖率 > 80%

## 持续集成

这些测试应该集成到CI/CD流程中：

```yaml
# .github/workflows/test.yml
- name: Run perception tests
  run: |
    cd tests/perception
    python run_tests.py --cov --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## 故障排除

### 导入错误
如果遇到导入错误，确保在项目根目录运行测试，或设置PYTHONPATH:
```bash
export PYTHONPATH=/media/yangyuhui/CODES1/Brain:$PYTHONPATH
```

### 依赖问题
确保安装了所有依赖:
```bash
pip install pytest pytest-asyncio pytest-cov
```

### 测试超时
某些测试可能需要更多时间，可以在测试文件中设置:
```python
@pytest.mark.timeout(10)  # 10秒超时
def test_slow_operation():
    ...
```
