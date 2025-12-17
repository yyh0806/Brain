# Planning Layer Analysis Report

## Executive Summary

This report provides a comprehensive analysis of the planning layer code in the brain-planning worktree, focusing on code structure, optimization opportunities, and architectural improvements. The planning layer consists of three main components: task planning, navigation planning, and behavior planning.

## 1. Code Structure Analysis

### 1.1 Directory Structure
```
brain/planning/
├── task/
│   ├── __init__.py
│   └── task_planner.py          # Main task decomposition and planning
├── navigation/
│   ├── __init__.py
│   ├── local_planner.py         # Local path planning and obstacle avoidance
│   ├── exploration_planner.py   # Exploration-based navigation
│   ├── intersection_navigator.py # Intersection handling
│   └── smooth_executor.py       # Smooth execution with perception integration
└── behavior/
    └── __init__.py              # Currently empty placeholder

brain/recovery/
└── replanner.py                 # Dynamic replanning capabilities
```

### 1.2 Component Overview

#### Task Planner (`task_planner.py`)
- **Lines of Code**: 891
- **Primary Responsibilities**:
  - Task decomposition using hierarchical planning
  - Operation template management
  - Perception-driven planning with CoT integration
  - Constraint application and validation
- **Design Patterns**: Template Method, Strategy Pattern

#### Navigation Components
- **Local Planner** (290 lines): DWA-based local planning with Pure Pursuit control
- **Exploration Planner** (587 lines): VLM-guided exploration and target search
- **Intersection Navigator** (394 lines): VLM-enhanced intersection detection and turning
- **Smooth Executor** (261 lines): Continuous execution with perception loops

#### Replanner (`replanner.py`)
- **Lines of Code**: 200+ (partial view)
- **Features**: LLM-based and rule-based replanning strategies

## 2. Optimization Opportunities

### 2.1 Task Planning Bottlenecks

#### Current Issues:
1. **Sequential Task Decomposition**: The `_decompose_task_tree` method processes tasks sequentially without parallelization
2. **Redundant Validation**: Plan validation occurs after full decomposition, catching issues late
3. **Static Operation Templates**: Templates are hardcoded, limiting runtime flexibility
4. **Limited Parallel Planning**: The `find_parallel_groups` method uses simple heuristics

#### Optimization Recommendations:
```python
# 1. Implement parallel task decomposition
async def _decompose_task_tree_parallel(self, task_node: TaskNode, platform_type: str) -> List[Operation]:
    """Decompose task tree with parallel processing of independent branches"""
    if not task_node.children:
        return self._task_to_operations(task_node, platform_type)

    # Create independent tasks for parallel processing
    independent_groups = self._identify_independent_subtasks(task_node.children)

    # Process groups in parallel
    tasks = [
        self._decompose_task_tree(child, platform_type)
        for group in independent_groups
        for child in group
    ]

    results = await asyncio.gather(*tasks)
    operations = [op for sublist in results for op in sublist]
    return self._resolve_dependencies(operations, task_node.children)

# 2. Early constraint validation during decomposition
def _validate_task_node_early(self, task_node: TaskNode, platform_type: str) -> bool:
    """Validate task feasibility before decomposition"""
    template = self.operation_templates.get(task_node.task_type)
    if not template:
        return False

    if platform_type not in template.get("platform", ["drone", "ugv", "usv"]):
        return False

    return True
```

### 2.2 Navigation Algorithm Optimization

#### Current Limitations:
1. **Local Planner**: Uses simplified DWA without considering robot dynamics
2. **Exploration Efficiency**: Random and spiral patterns lack environment awareness
3. **Path Smoothing**: No path smoothing or optimization
4. **Memoryless Planning**: No learning from past navigation experiences

#### Advanced Algorithm Integration:
```python
# 1. Enhanced A* implementation
class AStarPlanner:
    def __init__(self, resolution=0.1, robot_radius=0.3):
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.motion_model = self._init_motion_model()

    def plan(self, start, goal, occupancy_grid):
        """A* with dynamic constraints and path smoothing"""
        # Initialize open and closed sets
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        while not open_set.empty():
            current = open_set.get()[1]

            if self._goal_reached(current, goal):
                path = self._reconstruct_path(came_from, current)
                return self._smooth_path(path, occupancy_grid)

            for neighbor, motion in self._get_neighbors(current):
                tentative_g = g_score[current] + self._motion_cost(motion)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, motion)
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    open_set.put((f_score[neighbor], neighbor))

        return None  # No path found

# 2. RRT* for high-DOF planning
class RRTStarPlanner:
    def __init__(self, max_iter=5000, step_size=0.5, goal_bias=0.1):
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.nodes = []
        self.parent = {}
        self.cost = {}

    def plan(self, start, goal, state_valid_fn):
        """RRT* with asymptotic optimality"""
        self.nodes = [start]
        self.parent[start] = None
        self.cost[start] = 0

        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_bias:
                x_rand = goal
            else:
                x_rand = self._sample_free(state_valid_fn)

            # Find nearest node
            x_nearest = self._nearest_neighbor(x_rand)

            # Steer towards random point
            x_new = self._steer(x_nearest, x_rand, self.step_size)

            if state_valid_fn(x_new):
                # Find best parent (RRT* improvement)
                X_near = self._near_neighbors(x_new)
                x_min = x_nearest
                c_min = self.cost[x_nearest] + self._distance(x_nearest, x_new)

                for x_near in X_near:
                    if self._collision_free(x_near, x_new):
                        c = self.cost[x_near] + self._distance(x_near, x_new)
                        if c < c_min:
                            x_min = x_near
                            c_min = c

                self.nodes.append(x_new)
                self.parent[x_new] = x_min
                self.cost[x_new] = c_min

                # Rewire tree (RRT* improvement)
                for x_near in X_near:
                    if self._collision_free(x_new, x_near):
                        c = self.cost[x_new] + self._distance(x_new, x_near)
                        if c < self.cost[x_near]:
                            self.parent[x_near] = x_new
                            self.cost[x_near] = c

                # Check if goal is reached
                if self._distance(x_new, goal) < self.step_size:
                    return self._extract_path(start, x_new, goal)

        return None  # No path found

# 3. D* Lite for dynamic replanning
class DStarLite:
    def __init__(self, map_width, map_height):
        self.map_width = map_width
        self.map_height = map_height
        self.km = 0
        self.rhs = {}
        self.g = {}
        self.U = {}  # Priority queue
        self.s_last = None
        self.start = None
        self.goal = None

    def update_vertex(self, u):
        """Update vertex costs for D* Lite"""
        if u != self.goal:
            self.rhs[u] = float('inf')
            for v in self._successors(u):
                if self._edge_cost(u, v) < float('inf'):
                    self.rhs[u] = min(self.rhs[u], self.g[v] + self._edge_cost(u, v))

        if u in self.U:
            del self.U[u]

        if self.g[u] != self.rhs[u]:
            self._insert(u, self._calculate_key(u))

    def compute_shortest_path(self):
        """Compute shortest path with dynamic updates"""
        while True:
            u = self._top_key()
            if u is None or (self._compare_keys(self._calculate_key(self.start), u) and
                            self.g[self.start] == self.rhs[self.start]):
                break

            k_old = self._top_key()
            u = self._pop()

            if self._compare_keys(k_old, self._calculate_key(u)):
                self._insert(u, self._calculate_key(u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v in self._predecessors(u):
                    self.update_vertex(v)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for v in self._predecessors(u):
                    self.update_vertex(v)
```

### 2.3 Behavior Planning Improvements

#### Current State:
- The behavior planning module is essentially empty (only an `__init__.py` file)
- Behavior decisions are distributed across navigation components
- No unified behavior coordination

#### Proposed Architecture:
```python
# behavior_planner.py
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

class BehaviorState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    EXPLORING = "exploring"
    APPROACHING_GOAL = "approaching_goal"
    RECOVERING = "recovering"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class BehaviorContext:
    current_state: BehaviorState
    goal: Optional[Tuple[float, float]]
    obstacles: List[Dict[str, Any]]
    robot_pose: Tuple[float, float, float]
    battery_level: float
    time_remaining: float
    perception_confidence: float

class BehaviorPlanner:
    """Unified behavior planning and decision making"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_machine = BehaviorStateMachine()
        self.decision_trees = self._init_decision_trees()
        self.behavior_history = []

    def update_behavior(
        self,
        context: BehaviorContext,
        high_level_command: str
    ) -> BehaviorDecision:
        """Update behavior based on context and high-level command"""

        # Evaluate behavior conditions
        conditions = self._evaluate_conditions(context)

        # Check for emergency conditions first
        if conditions["emergency_stop"]:
            return BehaviorDecision(
                state=BehaviorState.EMERGENCY_STOP,
                action="stop",
                parameters={},
                priority=10,
                reason="Emergency condition detected"
            )

        # State machine based decision
        behavior = self.state_machine.update(context, conditions)

        # Override with high-level command if appropriate
        if self._should_override(context, high_level_command):
            behavior = self._create_command_behavior(high_level_command, context)

        # Log behavior change
        if behavior.state != context.current_state:
            self.behavior_history.append({
                "timestamp": time.time(),
                "from_state": context.current_state,
                "to_state": behavior.state,
                "context": context,
                "reason": behavior.reason
            })

        return behavior

# behavior_state_machine.py
class BehaviorStateMachine:
    """Hierarchical behavior state machine"""

    def __init__(self):
        self.current_state = BehaviorState.IDLE
        self.state_handlers = {
            BehaviorState.IDLE: self._handle_idle,
            BehaviorState.NAVIGATING: self._handle_navigating,
            BehaviorState.AVOIDING_OBSTACLE: self._handle_avoiding,
            BehaviorState.EXPLORING: self._handle_exploring,
            BehaviorState.APPROACHING_GOAL: self._handle_approaching,
            BehaviorState.RECOVERING: self._handle_recovering,
            BehaviorState.EMERGENCY_STOP: self._handle_emergency
        }
        self.transitions = self._init_transitions()

    def update(self, context: BehaviorContext, conditions: Dict[str, bool]) -> BehaviorDecision:
        """Update state machine based on context and conditions"""

        # Check for state transitions
        new_state = self._evaluate_transitions(context, conditions)

        if new_state != self.current_state:
            self.current_state = new_state

        # Execute state handler
        handler = self.state_handlers[self.current_state]
        return handler(context, conditions)
```

## 3. Architecture Improvements

### 3.1 Multi-Layer Planning Architecture

```python
# planning_hierarchy.py
class PlanningHierarchy:
    """Hierarchical planning with multiple abstraction levels"""

    def __init__(self):
        self.layers = {
            "strategic": StrategicPlanner(),     # High-level mission planning
            "tactical": TacticalPlanner(),       # Task decomposition and allocation
            "operational": OperationalPlanner(), # Motion planning and control
            "reactive": ReactivePlanner()        # Real-time reaction to obstacles
        }
        self.planning_budget = PlanningBudget()

    async def plan_hierarchical(self, mission: Mission) -> PlanResult:
        """Execute hierarchical planning"""

        # Budget allocation
        budget = self.planning_budget.allocate(mission)

        # Strategic planning (high level)
        strategic_result = await self.layers["strategic"].plan(
            mission,
            time_limit=budget.strategic_time
        )

        # Tactical planning (task level)
        tactical_result = await self.layers["tactical"].plan(
            strategic_result.tasks,
            time_limit=budget.tactical_time
        )

        # Operational planning (motion level)
        operational_result = await self.layers["operational"].plan(
            tactical_result.waypoints,
            time_limit=budget.operational_time
        )

        # Reactive planning (always active)
        reactive_planner = self.layers["reactive"]
        reactive_planner.initialize(operational_result.trajectory)

        return PlanResult(
            strategic=strategic_result,
            tactical=tactical_result,
            operational=operational_result,
            reactive=reactive_planner
        )

# planning_budget.py
@dataclass
class PlanningBudget:
    """Time and resource budgeting for planning"""
    total_time: float = 2.0  # Total planning time budget
    strategic_ratio: float = 0.2  # 20% for strategic
    tactical_ratio: float = 0.3   # 30% for tactical
    operational_ratio: float = 0.4 # 40% for operational
    reactive_ratio: float = 0.1   # 10% for reactive setup

    def allocate(self, mission: Mission) -> Dict[str, float]:
        """Allocate planning time based on mission complexity"""
        complexity = self._calculate_complexity(mission)

        return {
            "strategic_time": self.total_time * self.strategic_ratio * complexity,
            "tactical_time": self.total_time * self.tactical_ratio * complexity,
            "operational_time": self.total_time * self.operational_ratio,
            "reactive_time": self.total_time * self.reactive_ratio
        }
```

### 3.2 Multi-Robot Coordination

```python
# multi_robot_planner.py
class MultiRobotPlanner:
    """Coordination planner for multiple robots"""

    def __init__(self, robot_registry):
        self.robots = robot_registry
        self.task_allocator = TaskAllocator()
        self.coordination_graph = CoordinationGraph()
        self.conflict_resolver = ConflictResolver()

    async def plan_multi_robot(self, mission: Mission) -> MultiRobotPlan:
        """Plan for multiple robots with coordination"""

        # Task decomposition and allocation
        tasks = await self._decompose_mission(mission)
        allocations = await self.task_allocator.allocate(tasks, self.robots)

        # Build coordination graph
        self.coordination_graph.build(allocations)

        # Plan individual robot tasks
        robot_plans = {}
        for robot_id, robot_tasks in allocations.items():
            robot = self.robots[robot_id]
            planner = robot.get_planner()

            # Get robot constraints from coordination graph
            constraints = self.coordination_graph.get_constraints(robot_id)

            plan = await planner.plan_with_constraints(robot_tasks, constraints)
            robot_plans[robot_id] = plan

        # Resolve conflicts
        conflicts = self._detect_conflicts(robot_plans)
        if conflicts:
            robot_plans = await self.conflict_resolver.resolve(
                conflicts,
                robot_plans,
                self.coordination_graph
            )

        # Synchronize execution
        synchronized_plan = self._synchronize_plans(robot_plans)

        return MultiRobotPlan(
            robot_plans=robot_plans,
            coordination_graph=self.coordination_graph,
            synchronized_plan=synchronized_plan
        )

# conflict_resolver.py
class ConflictResolver:
    """Resolve conflicts in multi-robot plans"""

    def __init__(self):
        self.resolution_strategies = {
            "spatial": SpatialConflictResolver(),
            "temporal": TemporalConflictResolver(),
            "resource": ResourceConflictResolver()
        }

    async def resolve(
        self,
        conflicts: List[Conflict],
        plans: Dict[str, Plan],
        coordination_graph: CoordinationGraph
    ) -> Dict[str, Plan]:
        """Resolve all detected conflicts"""

        resolved_plans = plans.copy()

        # Prioritize conflicts
        sorted_conflicts = sorted(conflicts, key=lambda c: c.severity, reverse=True)

        for conflict in sorted_conflicts:
            strategy = self.resolution_strategies[conflict.type]

            if conflict.type == "spatial":
                resolution = await strategy.resolve_spatial_conflict(
                    conflict, resolved_plans
                )
            elif conflict.type == "temporal":
                resolution = await strategy.resolve_temporal_conflict(
                    conflict, resolved_plans
                )
            elif conflict.type == "resource":
                resolution = await strategy.resolve_resource_conflict(
                    conflict, resolved_plans
                )

            # Apply resolution
            for robot_id, plan in resolution.updated_plans.items():
                resolved_plans[robot_id] = plan

            # Update coordination graph
            coordination_graph.update(resolution.constraints)

        return resolved_plans
```

### 3.3 Real-Time Planning Optimizations

```python
# realtime_planner.py
class RealTimePlanner:
    """Real-time planning with incremental updates"""

    def __init__(self, config):
        self.config = config
        self.incremental_planner = IncrementalAStar()
        self.trajectory_optimizer = TrajectoryOptimizer()
        self.perception_buffer = CircularBuffer(maxlen=100)
        self.planning_thread = None
        self.latest_plan = None
        self.plan_lock = threading.Lock()

    def start_continuous_planning(self):
        """Start continuous planning in background thread"""
        self.planning_thread = threading.Thread(
            target=self._continuous_planning_loop,
            daemon=True
        )
        self.planning_thread.start()

    def _continuous_planning_loop(self):
        """Background planning loop"""
        while True:
            try:
                # Get latest perception
                perception = self.perception_buffer.get_latest()

                if perception:
                    # Incremental replanning
                    new_plan = self.incremental_planner.replan(
                        current_plan=self.latest_plan,
                        new_obstacles=perception.obstacles,
                        new_goal=perception.goal
                    )

                    # Optimize trajectory
                    optimized_plan = self.trajectory_optimizer.optimize(new_plan)

                    # Update plan atomically
                    with self.plan_lock:
                        self.latest_plan = optimized_plan

                # Dynamic sleep based on computation load
                sleep_time = self._calculate_sleep_time()
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Real-time planning error: {e}")
                time.sleep(0.1)

    def get_latest_plan(self) -> Plan:
        """Thread-safe access to latest plan"""
        with self.plan_lock:
            return self.latest_plan.copy() if self.latest_plan else None

# incremental_astar.py
class IncrementalAStar:
    """Incremental A* for fast replanning"""

    def __init__(self):
        self.expanded_nodes = {}  # Cache of expanded nodes
        self.heuristic_cache = {}  # Cache of heuristic values
        self.last_plan = None

    def replan(self, current_plan, new_obstacles, new_goal=None):
        """Incremental replanning with cached information"""

        # Check if major changes require full replanning
        if self._should_replan_fully(new_obstacles, new_goal):
            return self._full_replan(new_goal)

        # Otherwise, use incremental update
        return self._incremental_update(current_plan, new_obstacles)

    def _incremental_update(self, current_plan, new_obstacles):
        """Update plan incrementally"""
        # Identify affected nodes
        affected_nodes = self._find_affected_nodes(new_obstacles)

        # Invalidate affected nodes from cache
        for node in affected_nodes:
            if node in self.expanded_nodes:
                del self.expanded_nodes[node]

        # Replan only affected segments
        updated_segments = {}
        for segment in current_plan.segments:
            if self._segment_affected(segment, affected_nodes):
                updated_segments[segment.id] = self._replan_segment(
                    segment.start, segment.end, new_obstacles
                )

        # Reconstruct plan
        return self._reconstruct_plan(current_plan, updated_segments)
```

## 4. Development Guidelines

### 4.1 Coding Standards

#### Code Organization
1. **Single Responsibility**: Each planner class should have a single, well-defined purpose
2. **Interface Segregation**: Define clear interfaces between planning components
3. **Dependency Injection**: Use dependency injection for better testability
4. **Async/Await**: Use async patterns for all I/O-bound operations

#### Naming Conventions
```python
# Class names: PascalCase
class TaskPlanner:
    pass

# Method names: snake_case with clear verbs
async def decompose_task_tree(self, task_node: TaskNode) -> List[Operation]:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_PLANNING_TIME = 2.0
DEFAULT_GOAL_TOLERANCE = 0.5

# Private methods: prefix with underscore
def _validate_constraints(self, operations: List[Operation]) -> bool:
    pass
```

#### Documentation Standards
```python
class AdvancedPathPlanner:
    """
    Advanced path planner with A*, RRT*, and D* Lite algorithms.

    Features:
    - Multi-algorithm support based on environment complexity
    - Dynamic replanning capabilities
    - Path smoothing and optimization
    - Real-time performance optimization

    Attributes:
        algorithm: Current planning algorithm
        config: Configuration dictionary
        stats: Planning statistics

    Example:
        >>> planner = AdvancedPathPlanner(config={
        ...     'default_algorithm': 'a_star',
        ...     'resolution': 0.1,
        ...     'max_planning_time': 1.0
        ... })
        >>> path = planner.plan(start=(0, 0), goal=(10, 10),
        ...                     occupancy_grid=grid)
    """

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        occupancy_grid: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan a path from start to goal.

        Args:
            start: Start coordinates (x, y)
            goal: Goal coordinates (x, y)
            occupancy_grid: 2D numpy array of occupancy
            constraints: Optional planning constraints

        Returns:
            List of waypoints if path found, None otherwise

        Raises:
            PlanningTimeoutError: If planning exceeds time limit
            InvalidInputError: If inputs are invalid
        """
        pass
```

### 4.2 Testing Strategies

#### Unit Testing
```python
# test_task_planner.py
import pytest
from unittest.mock import Mock, AsyncMock

class TestTaskPlanner:
    """Test suite for TaskPlanner"""

    @pytest.fixture
    def task_planner(self):
        """Create test fixture for TaskPlanner"""
        world_state = Mock()
        config = {
            "strategy": "perception_driven",
            "max_planning_time": 1.0
        }
        return TaskPlanner(world_state, config)

    @pytest.mark.asyncio
    async def test_simple_task_decomposition(self, task_planner):
        """Test simple task decomposition"""
        parsed_task = {
            "task_type": "patrol",
            "parameters": {
                "area": [(0, 0), (10, 0), (10, 10), (0, 10)]
            }
        }

        operations = await task_planner.plan(parsed_task, "drone")

        assert len(operations) > 0
        assert operations[0].name == "takeoff"
        assert operations[-1].name == "land"

    @pytest.mark.asyncio
    async def test_perception_driven_planning(self, task_planner):
        """Test perception-driven planning with obstacles"""
        # Mock planning context with obstacles
        planning_context = Mock()
        planning_context.obstacles = [
            {"position": (5, 5), "distance": 2.0, "id": "obs1"}
        ]
        planning_context.battery_level = 80

        parsed_task = {
            "task_type": "goto",
            "parameters": {"position": (10, 10)}
        }

        operations = await task_planner.plan_with_perception(
            parsed_task,
            "drone",
            planning_context
        )

        # Check that obstacle avoidance is included
        has_avoidance = any(
            op.name == "detect_objects" or
            op.metadata.get("requires_avoidance", False)
            for op in operations
        )
        assert has_avoidance
```

#### Integration Testing
```python
# test_planning_integration.py
import pytest
from brain.planning.task import TaskPlanner
from brain.planning.navigation import LocalPlanner
from brain.execution.operations import ROS2UGVOperations

class TestPlanningIntegration:
    """Integration tests for planning components"""

    @pytest.mark.asyncio
    async def test_end_to_end_planning(self):
        """Test complete planning pipeline"""

        # Initialize components
        world_state = WorldState()
        task_planner = TaskPlanner(world_state)
        local_planner = LocalPlanner()
        executor = ROS2UGVOperations()

        # Define mission
        mission = {
            "task_type": "delivery",
            "parameters": {
                "pickup": (2, 3),
                "destination": (8, 7)
            }
        }

        # Plan tasks
        operations = await task_planner.plan(mission, "ugv")

        # Navigate to pickup location
        for op in operations:
            if op.name == "goto":
                path = [(0, 0), op.parameters["position"]]
                local_planner.set_path(path)

                # Simulate execution
                pose = (0, 0, 0)
                while not local_planner.is_goal_reached():
                    cmd = local_planner.compute_velocity(pose)
                    # Execute command
                    pose = self._simulate_motion(pose, cmd)

        assert True  # Test passes if no exceptions
```

#### Performance Testing
```python
# test_planning_performance.py
import time
import pytest
from brain.planning.navigation import AStarPlanner, RRTStarPlanner

class TestPlanningPerformance:
    """Performance tests for planning algorithms"""

    def test_planning_time_scalability(self):
        """Test planning time scales appropriately"""
        planner = AStarPlanner(resolution=0.1)

        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            # Generate random map
            grid = np.random.choice([0, 1], size=(size, size))
            start = (0, 0)
            goal = (size-1, size-1)

            # Time planning
            start_time = time.time()
            path = planner.plan(start, goal, grid)
            planning_time = time.time() - start_time

            times.append(planning_time)

        # Check that planning time scales sub-quadratically
        for i in range(1, len(times)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times[i] / times[i-1]
            assert time_ratio < size_ratio ** 1.5  # Sub-quadratic scaling
```

### 4.3 Documentation Requirements

#### Architecture Documentation
1. **System Architecture Diagram**: High-level component relationships
2. **Data Flow Diagrams**: How information flows through planning components
3. **State Machine Diagrams**: Behavior planning states and transitions
4. **API Documentation**: Detailed interface specifications

#### Design Decisions Document
```markdown
# Planning Layer Design Decisions

## ADR-001: Hierarchical Planning Architecture
- **Status**: Accepted
- **Date**: 2024-01-15
- **Decision**: Adopt three-layer hierarchical planning
- **Rationale**:
  - Separates concerns across abstraction levels
  - Enables parallel planning where possible
  - Improves scalability for complex missions
- **Alternatives Considered**: Monolithic planning, behavior-based planning
- **Consequences**: Increased complexity, better performance

## ADR-002: Perception-Driven Replanning
- **Status**: Accepted
- **Date**: 2024-01-20
- **Decision**: Integrate real-time perception into planning loop
- **Rationale**:
  - Enables adaptation to dynamic environments
  - Reduces planning failures
  - Improves safety
- **Consequences**: Higher computational load, need for robust perception
```

#### API Documentation (OpenAPI style)
```yaml
# planning_api.yaml
openapi: 3.0.0
info:
  title: Planning Layer API
  version: 1.0.0
  description: RESTful API for planning operations

paths:
  /plan:
    post:
      summary: Generate a plan for given task
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PlanningRequest'
      responses:
        '200':
          description: Plan generated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PlanningResponse'
        '400':
          description: Invalid request
        '500':
          description: Planning error

components:
  schemas:
    PlanningRequest:
      type: object
      required:
        - task_type
        - platform_type
      properties:
        task_type:
          type: string
          enum: [patrol, delivery, inspection, search_and_rescue]
        parameters:
          type: object
          additionalProperties: true
        platform_type:
          type: string
          enum: [drone, ugv, usv]
        constraints:
          type: object
          properties:
            max_speed:
              type: number
            safe_distance:
              type: number
```

## 5. Integration Patterns

### 5.1 Layer Integration

```python
# planning_facade.py
class PlanningFacade:
    """Facade pattern for planning layer integration"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._init_components()
        self._setup_interfaces()

    def _init_components(self):
        """Initialize all planning components"""
        self.task_planner = TaskPlanner(
            world_state=self.world_state,
            config=self.config.get("task_planning", {})
        )
        self.navigation_planner = NavigationPlanner(
            config=self.config.get("navigation", {})
        )
        self.behavior_planner = BehaviorPlanner(
            config=self.config.get("behavior", {})
        )
        self.replanner = Replanner(
            planner=self.task_planner,
            llm=self.llm,
            config=self.config.get("replanning", {})
        )

    async def execute_mission(self, mission: Mission) -> MissionResult:
        """Execute complete mission through planning hierarchy"""

        try:
            # 1. Task decomposition
            tasks = await self.task_planner.plan_with_perception(
                parsed_task=mission.to_dict(),
                platform_type=mission.platform_type,
                planning_context=mission.planning_context
            )

            # 2. Navigation planning
            navigation_plan = await self.navigation_planner.plan(
                tasks=tasks,
                initial_pose=mission.initial_pose,
                map_data=mission.map_data
            )

            # 3. Behavior planning
            behavior_plan = self.behavior_planner.generate_behavior(
                navigation_plan=navigation_plan,
                mission_objectives=mission.objectives
            )

            # 4. Execute with monitoring
            result = await self._execute_with_monitoring(
                tasks=tasks,
                navigation_plan=navigation_plan,
                behavior_plan=behavior_plan,
                mission=mission
            )

            return result

        except PlanningError as e:
            # Attempt recovery through replanning
            recovery_plan = await self.replanner.replan_with_perception(
                original_command=mission.command,
                completed_operations=[],
                changes=[e.environment_change],
                planning_context=mission.planning_context
            )
            # Execute recovery plan...
```

### 5.2 Event-Driven Integration

```python
# planning_events.py
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict

class PlanningEventType(Enum):
    PLAN_GENERATED = "plan_generated"
    PLAN_FAILED = "plan_failed"
    OBSTACLE_DETECTED = "obstacle_detected"
    GOAL_REACHED = "goal_reached"
    REPLAN_REQUIRED = "replan_required"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"

@dataclass
class PlanningEvent:
    event_type: PlanningEventType
    data: Dict[str, Any]
    timestamp: float
    source: str

class PlanningEventBus:
    """Event bus for planning component communication"""

    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event_type: PlanningEventType, callback):
        """Subscribe to planning events"""
        self.subscribers[event_type].append(callback)

    async def publish(self, event: PlanningEvent):
        """Publish planning event"""
        for callback in self.subscribers[event.event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

# Usage example
class MonitoringSystem:
    def __init__(self, event_bus: PlanningEventBus):
        self.event_bus = event_bus
        self.subscribe_to_events()

    def subscribe_to_events(self):
        self.event_bus.subscribe(
            PlanningEventType.PLAN_GENERATED,
            self.on_plan_generated
        )
        self.event_bus.subscribe(
            PlanningEventType.REPLAN_REQUIRED,
            self.on_replan_required
        )

    async def on_plan_generated(self, event: PlanningEvent):
        """Handle new plan generation"""
        plan = event.data["plan"]
        await self.analyze_plan(plan)
        self.update_monitoring(plan)

    async def on_replan_required(self, event: PlanningEvent):
        """Handle replanning request"""
        reason = event.data["reason"]
        await self.trigger_replanning(reason)
```

## 6. Recommendations Summary

### 6.1 High Priority Improvements
1. **Implement Advanced Pathfinding**: Replace simplified algorithms with A*, RRT*, and D* Lite
2. **Add Real-Time Planning**: Implement continuous replanning with incremental updates
3. **Complete Behavior Planning**: Implement the missing behavior planning module
4. **Parallel Task Decomposition**: Enable parallel processing of independent tasks

### 6.2 Medium Priority Improvements
1. **Multi-Robot Coordination**: Add support for coordinated multi-robot planning
2. **Learning Capabilities**: Implement learning from past planning experiences
3. **Path Optimization**: Add trajectory optimization and smoothing
4. **Performance Monitoring**: Add comprehensive metrics and profiling

### 6.3 Low Priority Improvements
1. **GPU Acceleration**: Explore GPU-based planning for large environments
2. **Distributed Planning**: Implement distributed planning for very large missions
3. **Planning Database**: Store and reuse successful planning patterns
4. **Visualization Tools**: Add planning visualization and debugging tools

### 6.4 Implementation Roadmap

#### Phase 1 (1-2 months)
- Implement A* and D* Lite planners
- Add real-time incremental replanning
- Complete behavior planning module
- Improve unit test coverage to 80%

#### Phase 2 (2-3 months)
- Add RRT* for complex environments
- Implement multi-robot coordination
- Add path optimization and smoothing
- Create comprehensive integration tests

#### Phase 3 (3-4 months)
- Add learning capabilities
- Implement distributed planning for large missions
- Create planning visualization tools
- Optimize for real-time performance

## 7. Conclusion

The planning layer shows a solid foundation with good separation of concerns between task, navigation, and behavior planning. However, there are significant opportunities for optimization and architectural improvements. The recommendations in this report focus on implementing advanced planning algorithms, improving real-time performance, and enhancing the overall architecture for scalability and maintainability.

By following the proposed development guidelines and implementing the suggested improvements, the planning layer can achieve significantly better performance, reliability, and flexibility for complex autonomous missions.