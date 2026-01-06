"""
Pytest Fixtures for Planning Tests

规划层测试的pytest fixtures

使用猴子补丁技术，在加载前预先设置sys.modules，完全绕过 brain/__init__.py
"""

import sys
from pathlib import Path
import importlib.util
from types import ModuleType
from enum import Enum
from typing import Any, Dict, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
planning_root = project_root / "brain" / "planning"


def create_fake_package(package_name):
    """创建一个假的包模块，用于阻止Python加载真实的__init__.py"""
    fake_module = ModuleType(package_name)
    fake_module.__path__ = []
    sys.modules[package_name] = fake_module
    return fake_module


def load_module_isolated(module_name, file_path, dependent_modules=None):
    """隔离加载模块，直接从文件加载，不经过包的__init__.py"""
    # 预先加载依赖模块
    if dependent_modules:
        for dep_name, dep_path in dependent_modules.items():
            if dep_name not in sys.modules:
                load_module_isolated(dep_name, dep_path)

    # 创建父包的假模块（如果还没有）
    parts = module_name.split('.')
    for i in range(len(parts) - 1):
        parent_name = '.'.join(parts[:i+1])
        if parent_name not in sys.modules:
            create_fake_package(parent_name)

    # 加载当前模块
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为模块 {module_name} 创建spec: {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # 如果加载失败，清理sys.modules
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise e

    return module


# ==================== 关键步骤：在加载任何模块之前，创建假的 brain 包 ====================
# 这将阻止Python执行真实的 brain/__init__.py，从而避免感知层的依赖问题
fake_brain = create_fake_package("brain")
fake_brain_planning = create_fake_package("brain.planning")
fake_brain_planning_state = create_fake_package("brain.planning.state")
fake_brain_planning_models = create_fake_package("brain.planning.models")
fake_brain_planning_interfaces = create_fake_package("brain.planning.interfaces")
fake_brain_planning_capability = create_fake_package("brain.planning.capability")
fake_brain_planners = create_fake_package("brain.planning.planners")
fake_brain_intelligent = create_fake_package("brain.planning.intelligent")
fake_brain_execution_executor = create_fake_package("brain.execution.executor")
fake_brain_state = create_fake_package("brain.state")
fake_brain_state_world_state = create_fake_package("brain.state.world_state")

fake_brain_orchestrator = create_fake_package("brain.planning.orchestrator")
fake_brain_action_level = create_fake_package("brain.planning.action_level")
fake_brain_execution = create_fake_package("brain.execution")
fake_brain_execution_monitor = create_fake_package("brain.execution.monitor")
# =======================================================================================


# 创建简化的 FailureType 枚举（避免加载完整的 execution 模块）
class FailureType(Enum):
    """失败类型（简化版，仅用于测试）"""
    PRECONDITION_FAILED = "precondition_failed"
    EXECUTION_FAILED = "execution_failed"
    WORLD_STATE_CHANGED = "world_state_changed"
    PERCEPTION_FAILED = "perception_failed"


# 创建简化的 AdaptiveExecutor 类（仅用于测试）
class AdaptiveExecutor:
    """AdaptiveExecutor mock（仅用于测试）"""
    def __init__(self, *args, **kwargs):
        pass



# 创建简化的 Executor 类（仅用于测试）
class Executor:
    """Executor mock（仅用于测试）"""
    def __init__(self, *args, **kwargs):
        pass


# 创建简化的 WorldState 类（仅用于测试）
class WorldState:
    """WorldState mock（仅用于测试）"""
    def __init__(self, *args, **kwargs):
        pass

# 创建简化的 ExecutionMonitor 类（仅用于测试）
class ExecutionMonitor:
    """ExecutionMonitor mock（仅用于测试）"""
    def __init__(self, *args, **kwargs):
        pass

sys.modules["brain.execution.executor"] = ModuleType("brain.execution.executor")
sys.modules["brain.execution.executor"].Executor = Executor
sys.modules["brain.state.world_state"] = ModuleType("brain.state.world_state")
sys.modules["brain.state.world_state"].WorldState = WorldState


# 创建简化的 FailureClassifier 类（仅用于测试）
class FailureClassifier:
    """FailureClassifier mock（仅用于测试）"""
    def __init__(self, *args, **kwargs):
        pass


# 将这些类添加到假模块中
sys.modules["brain.execution.monitor"].FailureType = FailureType
sys.modules["brain.execution.monitor"].AdaptiveExecutor = AdaptiveExecutor
sys.modules["brain.execution.monitor"].ExecutionMonitor = ExecutionMonitor
sys.modules["brain.execution.monitor"].FailureClassifier = FailureClassifier

# 首先加载基础模块（无依赖的）
plan_node_module = load_module_isolated(
    "brain.planning.state.plan_node",
    planning_root / "state" / "plan_node.py"
)
NodeStatus = plan_node_module.NodeStatus
CommitLevel = plan_node_module.CommitLevel
PlanNode = plan_node_module.PlanNode

plan_state_module = load_module_isolated(
    "brain.planning.state.plan_state",
    planning_root / "state" / "plan_state.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py"
    }
)
PlanState = plan_state_module.PlanState

# 更新 brain.planning.state 模块，使其包含导出的类
sys.modules["brain.planning.state"].PlanNode = PlanNode
sys.modules["brain.planning.state"].PlanState = PlanState
sys.modules["brain.planning.state"].NodeStatus = NodeStatus
sys.modules["brain.planning.state"].CommitLevel = CommitLevel

# 加载models
location_module = load_module_isolated(
    "brain.planning.models.location",
    planning_root / "models" / "location.py"
)
Location = location_module.Location
Door = location_module.Door
ObjectInfo = location_module.ObjectInfo

# 更新 brain.planning.models 模块，使其包含导出的类
sys.modules["brain.planning.models"].Location = Location
sys.modules["brain.planning.models"].Door = Door
sys.modules["brain.planning.models"].ObjectInfo = ObjectInfo

# 加载interfaces（依赖models）
world_model_interface = load_module_isolated(
    "brain.planning.interfaces.world_model",
    planning_root / "interfaces" / "world_model.py",
    dependent_modules={
        "brain.planning.models.location": planning_root / "models" / "location.py"
    }
)
IWorldModel = world_model_interface.IWorldModel

# 更新 brain.planning.interfaces 模块，使其包含导出的类
sys.modules["brain.planning.interfaces"].IWorldModel = IWorldModel

# 加载action_level（依赖interfaces）
world_model_mock_module = load_module_isolated(
    "brain.planning.action_level.world_model_mock",
    planning_root / "action_level" / "world_model_mock.py",
    dependent_modules={
        "brain.planning.interfaces.world_model": planning_root / "interfaces" / "world_model.py",
        "brain.planning.models.location": planning_root / "models" / "location.py"
    }
)
WorldModelMock = world_model_mock_module.WorldModelMock

# 更新 brain.planning.action_level 模块
sys.modules["brain.planning.action_level"].WorldModelMock = WorldModelMock

# 加载capability
capability_registry_module = load_module_isolated(
    "brain.planning.capability.capability_registry",
    planning_root / "capability" / "capability_registry.py"
)
Capability = capability_registry_module.Capability
CapabilityRegistry = capability_registry_module.CapabilityRegistry

platform_adapter_module = load_module_isolated(
    "brain.planning.capability.platform_adapter",
    planning_root / "capability" / "platform_adapter.py",
    dependent_modules={
        "brain.planning.capability.capability_registry": planning_root / "capability" / "capability_registry.py"
    }
)
PlatformAdapter = platform_adapter_module.PlatformAdapter

# 更新 brain.planning.capability 模块
sys.modules["brain.planning.capability"].Capability = Capability
sys.modules["brain.planning.capability"].CapabilityRegistry = CapabilityRegistry
sys.modules["brain.planning.capability"].PlatformAdapter = PlatformAdapter

# 加载planners
action_level_planner_module = load_module_isolated(
    "brain.planning.planners.action_level_planner",
    planning_root / "planners" / "action_level_planner.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py",
        "brain.planning.capability.capability_registry": planning_root / "capability" / "capability_registry.py",
        "brain.planning.capability.platform_adapter": planning_root / "capability" / "platform_adapter.py",
        "brain.planning.action_level.world_model_mock": planning_root / "action_level" / "world_model_mock.py"
    }
)
ActionLevelPlanner = action_level_planner_module.ActionLevelPlanner

skill_level_planner_module = load_module_isolated(
    "brain.planning.planners.skill_level_planner",
    planning_root / "planners" / "skill_level_planner.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py",
        "brain.planning.planners.action_level_planner": planning_root / "planners" / "action_level_planner.py"
    }
)
SkillLevelPlanner = skill_level_planner_module.SkillLevelPlanner

task_level_planner_module = load_module_isolated(
    "brain.planning.planners.task_level_planner",
    planning_root / "planners" / "task_level_planner.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py"
    }
)
TaskLevelPlanner = task_level_planner_module.TaskLevelPlanner

# 更新 brain.planning.planners 模块
sys.modules["brain.planning.planners"].ActionLevelPlanner = ActionLevelPlanner
sys.modules["brain.planning.planners"].SkillLevelPlanner = SkillLevelPlanner
sys.modules["brain.planning.planners"].TaskLevelPlanner = TaskLevelPlanner

# 加载intelligent（有execution依赖，使用mock）
dynamic_planner_module = load_module_isolated(
    "brain.planning.intelligent.dynamic_planner",
    planning_root / "intelligent" / "dynamic_planner.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py",
        "brain.planning.action_level.world_model_mock": planning_root / "action_level" / "world_model_mock.py"
    }
)
DynamicPlanner = dynamic_planner_module.DynamicPlanner

replanning_rules_module = load_module_isolated(
    "brain.planning.intelligent.replanning_rules",
    planning_root / "intelligent" / "replanning_rules.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py"
    }
)
ReplanningRules = replanning_rules_module.ReplanningRules

plan_validator_module = load_module_isolated(
    "brain.planning.intelligent.plan_validator",
    planning_root / "intelligent" / "plan_validator.py",
    dependent_modules={
        "brain.planning.state.plan_node": planning_root / "state" / "plan_node.py",
        "brain.planning.state.plan_state": planning_root / "state" / "plan_state.py"
    }
)
PlanValidation = plan_validator_module.PlanValidation
PlanValidator = plan_validator_module.PlanValidator

# 更新 brain.planning.intelligent 模块
sys.modules["brain.planning.intelligent"].DynamicPlanner = DynamicPlanner
sys.modules["brain.planning.intelligent"].ReplanningRules = ReplanningRules
sys.modules["brain.planning.intelligent"].PlanValidation = PlanValidation
sys.modules["brain.planning.intelligent"].PlanValidator = PlanValidator

# 加载orchestrator（依赖很多模块）
orchestrator_module = load_module_isolated(
    "brain.planning.orchestrator.planning_orchestrator",
    planning_root / "orchestrator" / "planning_orchestrator.py",
    dependent_modules={
        "brain.planning.state.plan_state": planning_root / "state" / "plan_state.py",
        "brain.planning.capability.capability_registry": planning_root / "capability" / "capability_registry.py",
        "brain.planning.capability.platform_adapter": planning_root / "capability" / "platform_adapter.py",
        "brain.planning.action_level.world_model_mock": planning_root / "action_level" / "world_model_mock.py",
        "brain.planning.planners.task_level_planner": planning_root / "planners" / "task_level_planner.py",
        "brain.planning.planners.skill_level_planner": planning_root / "planners" / "skill_level_planner.py",
        "brain.planning.planners.action_level_planner": planning_root / "planners" / "action_level_planner.py"
    }
)
PlanningOrchestrator = orchestrator_module.PlanningOrchestrator

# 更新 brain.planning.orchestrator 模块
sys.modules["brain.planning.orchestrator"].PlanningOrchestrator = PlanningOrchestrator

import pytest


@pytest.fixture
def config_path():
    """能力配置文件路径"""
    return project_root / "config" / "planning" / "capability_config.yaml"


@pytest.fixture
def capability_registry(config_path):
    """CapabilityRegistry 实例"""
    return CapabilityRegistry(str(config_path))


@pytest.fixture
def platform_adapter(capability_registry):
    """PlatformAdapter 实例"""
    return PlatformAdapter(capability_registry)


@pytest.fixture
def world_model_mock():
    """WorldModelMock 实例"""
    return WorldModelMock()


@pytest.fixture
def action_level_planner(capability_registry, platform_adapter, world_model_mock):
    """ActionLevelPlanner 实例"""
    return ActionLevelPlanner(
        capability_registry=capability_registry,
        platform_adapter=platform_adapter,
        world_model=world_model_mock,
        platform="ugv"
    )


@pytest.fixture
def skill_level_planner(action_level_planner):
    """SkillLevelPlanner 实例"""
    return SkillLevelPlanner(action_planner=action_level_planner)


@pytest.fixture
def task_level_planner():
    """TaskLevelPlanner 实例"""
    return TaskLevelPlanner()


@pytest.fixture
def planning_orchestrator():
    """PlanningOrchestrator 实例"""
    return PlanningOrchestrator(platform="ugv")


@pytest.fixture
def dynamic_planner(world_model_mock):
    """DynamicPlanner 实例"""
    return DynamicPlanner(world_model=world_model_mock)


@pytest.fixture
def replanning_rules():
    """ReplanningRules 实例"""
    return ReplanningRules()


@pytest.fixture
def plan_validator():
    """PlanValidator 实例"""
    return PlanValidator()


@pytest.fixture
def sample_plan_node():
    """示例 PlanNode"""
    return PlanNode(
        id="test_node",
        name="test_action",
        action="move_to",
        skill="go_to_location",
        task="test_task",
        parameters={"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
        preconditions=["robot.ready == True"],
        expected_effects=["robot.position.near(target)"]
    )


@pytest.fixture
def sample_plan_state(sample_plan_node):
    """示例 PlanState"""
    state = PlanState()
    state.add_root(sample_plan_node)
    return state


@pytest.fixture
def sample_locations():
    """示例位置数据"""
    return {
        "kitchen": {"x": 5.0, "y": 3.0, "z": 0.0},
        "living_room": {"x": 0.0, "y": 0.0, "z": 0.0},
        "table": {"x": 2.0, "y": 0.0, "z": 0.8}
    }


# Pytest markers 配置
def pytest_configure(config):
    """配置pytest markers"""
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "e2e: 端到端测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
