# Isaac Sim仿真环境集成指南

本指南详细介绍了如何将NVIDIA Isaac Sim仿真环境集成到Brain项目中，替代CARLA仿真，提供高保真的物理仿真和机器人仿真支持。

## 📋 目录

- [系统要求](#系统要求)
- [安装配置](#安装配置)
- [核心接口](#核心接口)
- [配置文件](#配置文件)
- [使用示例](#使用示例)
- [World Model集成](#world-model集成)
- [故障排除](#故障排除)

## 💻 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with VOLTA架构或更新版本
- **VRAM**: 最少4GB（推荐8GB+）
- **内存**: 最少16GB RAM
- **存储**: 至少50GB可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04/22.04（推荐）或Windows
- **Python版本**:
  - Isaac Sim 4.x: Python 3.10
  - Isaac Sim 5.x: Python 3.11
- **GLIBC**: 2.35+
- **NVIDIA驱动**: 515.65+

### 系统兼容性检查
```bash
# 检查GLIBC版本
ldd --version

# 检查NVIDIA驱动
nvidia-smi

# 检查Python版本
python3 --version
```

## 🚀 安装配置

### 1. 自动安装（推荐）
```bash
# 运行环境配置脚本
cd /media/yangyuhui/CODES1/Brain
./scripts/setup_isaac_sim.sh
```

### 2. 手动安装

#### 安装Isaac Sim
```bash
# 创建虚拟环境
python3 -m venv isaac_sim_env
source isaac_sim_env/bin/activate

# 升级pip
pip install --upgrade pip

# 安装Isaac Sim
pip install isaacsim
```

#### 验证安装
```bash
# 运行测试脚本
python test_isaac_sim.py
```

#### 安装项目依赖
```bash
# 安装项目依赖
pip install -r requirements.txt
```

## 🔧 核心接口

### IsaacSimInterface类

主要接口类，提供与Isaac Sim的完整集成：

```python
from brain.platforms.isaac_sim_interface import IsaacSimInterface, RobotConfig, SensorConfig

# 创建接口实例
interface = IsaacSimInterface(
    simulation_mode=SimulationMode.HEADLESS,
    headless=True
)

# 初始化
await interface.initialize()

# 创建机器人
robot_config = RobotConfig(
    robot_type="franka",
    robot_id="franka_emika",
    position=(0.0, 0.0, 0.0)
)
robot_id = await interface.create_robot(robot_config)

# 创建传感器
sensor_config = SensorConfig(
    sensor_type="camera",
    sensor_name="main_camera",
    attach_to_robot="franka_emika",
    sensor_params={"resolution": [640, 480]}
)
sensor_id = await interface.create_sensor(sensor_config)

# 启动仿真
await interface.start_simulation()

# 运行仿真循环
for _ in range(1000):
    await interface.step_simulation()

    # 获取传感器数据
    sensor_data = await interface.get_sensor_data(sensor_id)

    # 控制机器人
    command = {"joint_positions": {"panda_joint1": 0.5}}
    await interface.set_robot_command(robot_id, command)

# 关闭仿真
await interface.shutdown()
```

### 支持的功能

1. **机器人仿真**
   - Franka机械臂
   - UR10机械臂
   - 移动机器人（Husky）
   - 四旋翼无人机
   - 自定义USD模型

2. **传感器仿真**
   - RGB相机
   - 深度相机
   - 激光雷达
   - IMU
   - 自定义传感器

3. **物理仿真**
   - NVIDIA PhysX高精度物理引擎
   - 碰撞检测
   - 关节动力学
   - 材料属性

4. **场景管理**
   - 地面平面
   - 障碍物
   - 抓取对象
   - 环境光照

## 📄 配置文件

### 主配置文件：`config/isaac_sim_config.yaml`

```yaml
# 全局仿真配置
simulation:
  mode: "headless"  # headless, gui, render
  physics_dt: 0.016666  # 60Hz
  rendering_dt: 0.016666

# 机器人配置
robots:
  franka_emika:
    type: "franka"
    position: [0.0, 0.0, 0.0]
    default_joints:
      "panda_joint1": 0.0
      "panda_joint2": 0.0

# 传感器配置
sensors:
  rgb_camera:
    type: "camera"
    resolution: [640, 480]
    attach_to: "franka_emika"
```

### 配置参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `simulation.mode` | string | 仿真模式 | "headless" |
| `simulation.physics_dt` | float | 物理时间步长 | 0.016666 |
| `robot.type` | string | 机器人类型 | - |
| `robot.position` | array | 初始位置 | [0,0,0] |
| `sensor.type` | string | 传感器类型 | - |
| `sensor.resolution` | array | 相机分辨率 | [640,480] |

## 🎮 使用示例

### 1. 基础仿真演示

```bash
# 运行基础演示
python examples/isaac_sim_demo.py --demo basic

# GUI模式
python examples/isaac_sim_demo.py --demo basic --mode gui

# 抓取演示
python examples/isaac_sim_demo.py --demo pick_and_place
```

### 2. World Model集成演示

```bash
# 运行集成演示
python examples/world_model_isaac_integration.py --duration 60

# GUI模式运行
python examples/world_model_isaac_integration.py --duration 120 --mode gui
```

### 3. 自定义场景

```python
import asyncio
from brain.platforms.isaac_sim_interface import create_isaac_sim_interface

async def custom_simulation():
    # 创建仿真接口
    interface = await create_isaac_sim_interface(headless=True)

    # 添加自定义机器人
    robot_config = RobotConfig(
        robot_type="custom_robot",
        usd_path="/path/to/custom_robot.usd",
        position=(1.0, 1.0, 0.0)
    )
    await interface.create_robot(robot_config)

    # 运行仿真
    await interface.start_simulation()
    for _ in range(1000):
        await interface.step_simulation()

    await interface.shutdown()

asyncio.run(custom_simulation())
```

## 🧠 World Model集成

### 集成架构

```
感知层 → World Model → 规划层 → 执行层 → Isaac Sim
  ↑                                    ↓
  ←←←←←←←←← 传感器反馈 ←←←←←←←←←←←←←←←←←←←←←←←
```

### 关键特性

1. **感知集成**
   - 传感器数据处理
   - 目标检测和识别
   - 环境地图构建

2. **规划集成**
   - 任务规划
   - 路径规划
   - 运动规划

3. **执行集成**
   - 机器人控制
   - 实时监控
   - 错误处理

4. **闭环反馈**
   - 执行结果评估
   - 在线调整
   - 学习优化

### 集成示例代码

```python
from examples.world_model_isaac_integration import WorldModelIsaacIntegration

# 创建集成系统
integration = WorldModelIsaacIntegration()

# 初始化
await integration.initialize()

# 添加任务
from examples.world_model_isaac_integration import Task
task = Task(
    task_id="sample_task",
    task_type="grasp",
    goal={"target_object": "cube"}
)
await integration.add_task(task)

# 运行控制循环
await integration.run_control_loop(60.0)

# 关闭系统
await integration.shutdown()
```

## 🔧 故障排除

### 常见问题

#### 1. Isaac Sim导入失败

**错误信息**: `ImportError: No module named 'isaacsim'`

**解决方案**:
```bash
# 检查Python版本兼容性
python --version

# 重新安装Isaac Sim
pip uninstall isaacsim
pip install isaacsim

# 使用虚拟环境
python3 -m venv isaac_env
source isaac_env/bin/activate
pip install isaacsim
```

#### 2. GLIBC版本不兼容

**错误信息**: `GLIBC version not compatible`

**解决方案**:
```bash
# 检查GLIBC版本
ldd --version

# 使用Docker容器
docker pull nvcr.io/isaac/sim:2023.1.1
docker run -it --gpus all nvcr.io/isaac/sim:2023.1.1
```

#### 3. GPU内存不足

**错误信息**: `CUDA out of memory`

**解决方案**:
```python
# 使用headless模式
interface = IsaacSimInterface(
    simulation_mode=SimulationMode.HEADLESS,
    headless=True
)

# 降低分辨率
sensor_config = SensorConfig(
    sensor_type="camera",
    sensor_params={"resolution": [320, 240]}  # 降低分辨率
)
```

#### 4. 仿真性能问题

**解决方案**:
```yaml
# 降低物理精度
physics:
  physx:
    num_iterations: 5  # 降低迭代次数
    enable_gpu_dynamics: false  # 禁用GPU加速

# 降低仿真频率
simulation:
  physics_dt: 0.033333  # 30Hz
  rendering_dt: 0.033333
```

### 性能优化建议

1. **硬件优化**
   - 使用SSD存储
   - 增加系统内存
   - 使用高性能GPU

2. **软件优化**
   - 使用headless模式
   - 降低渲染质量
   - 减少传感器数量

3. **配置优化**
   - 调整物理参数
   - 简化场景复杂度
   - 使用合适的仿真频率

### 调试工具

```bash
# 检查系统状态
./scripts/setup_isaac_sim.sh

# 测试Isaac Sim
python test_isaac_sim.py

# 监控性能
nvidia-smi -l 1  # GPU监控
htop  # CPU和内存监控
```

## 📚 参考资料

### 官方文档
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html)
- [Isaac Sim Python API](https://docs.omniverse.nvidia.com/isaacsim/latest/core_api.html)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

### 示例项目
- [Isaac Sim Examples](https://docs.omniverse.nvidia.com/isaacsim/latest/introduction/examples.html)
- [Robot Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)

### 社区资源
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/omniverse/omniverse-sim-applications/274)
- [GitHub Repositories](https://github.com/search?q=isaac+sim+python)

---

## 🤝 贡献

如果您遇到问题或有改进建议，请：

1. 检查现有的故障排除指南
2. 查看GitHub Issues
3. 提交新的Issue或Pull Request

---

**注意**: Isaac Sim需要NVIDIA GPU和兼容的系统环境。如果您的系统不满足要求，可以考虑使用轻量级的PyBullet替代方案（已在代码中实现）。