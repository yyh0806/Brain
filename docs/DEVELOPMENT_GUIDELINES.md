# Brain Project Development Guidelines

## Overview

This document provides comprehensive development guidelines for the Brain intelligent unmanned system task planning core. The project uses a parallel development approach with Git worktrees to enable simultaneous development across different architectural layers.

## Architecture Layers

### 1. Perception Layer (`brain/perception/`)
**Responsibility**: Sensor data processing, environment understanding, and mapping
**Key Technologies**: ROS2, OpenCV, sensor fusion algorithms

### 2. Cognitive Layer (`brain/cognitive/`)
**Responsibility**: World modeling, reasoning, dialogue management
**Key Technologies**: LLM integration, CoT reasoning, state management

### 3. Planning Layer (`brain/planning/`)
**Responsibility**: Task planning, navigation, behavior generation
**Key Technologies**: Pathfinding algorithms, task decomposition

### 4. Execution Layer (`brain/execution/`)
**Responsibility**: Command execution, operation management
**Key Technologies**: Async execution, robot control interfaces

### 5. Communication Layer (`brain/communication/`)
**Responsibility**: External system communication, protocol management
**Key Technologies**: ROS2, network protocols, message handling

### 6. Models Layer (`brain/models/`)
**Responsibility**: AI model interfaces, prompt management, task parsing
**Key Technologies**: LLM APIs, prompt engineering, model optimization

## Code Standards

### General Guidelines

1. **Python Version**: Use Python 3.8+ syntax features
2. **Code Style**: Follow PEP 8 with 100-character line length
3. **Type Hints**: Use type annotations for all public APIs
4. **Documentation**: Docstrings for all classes and public methods
5. **Testing**: Minimum 80% test coverage for new code

### Layer-Specific Guidelines

#### Perception Layer
```python
# Example: Sensor data processing
class SensorProcessor:
    """Processes sensor data for environment perception."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize sensor processor with configuration."""
        self.config = config
        self._sensor_queue = asyncio.Queue()

    async def process_sensor_data(self, data: SensorData) -> ProcessedData:
        """Process incoming sensor data asynchronously."""
        try:
            # Process data with error handling
            processed = await self._process_data(data)
            await self._update_environment_model(processed)
            return processed
        except Exception as e:
            logger.error(f"Sensor processing failed: {e}")
            raise PerceptionError(f"Failed to process sensor data: {e}")
```

#### Cognitive Layer
```python
# Example: World model management
class WorldModel:
    """Manages the cognitive world model state."""

    def __init__(self, llm_interface: LLMInterface) -> None:
        """Initialize world model with LLM interface."""
        self.llm = llm_interface
        self._state = WorldState()
        self._reasoning_cache = LRUCache(maxsize=1000)

    async def update_world_state(self, perception_data: PerceptionData) -> WorldState:
        """Update world state based on perception input."""
        try:
            # Apply reasoning with caching
            cache_key = self._generate_cache_key(perception_data)
            if cache_key in self._reasoning_cache:
                reasoning_result = self._reasoning_cache[cache_key]
            else:
                reasoning_result = await self._reason_about_data(perception_data)
                self._reasoning_cache[cache_key] = reasoning_result

            self._state.update(reasoning_result)
            return self._state
        except Exception as e:
            logger.error(f"World model update failed: {e}")
            raise CognitiveError(f"Failed to update world state: {e}")
```

#### Planning Layer
```python
# Example: Task planning
class TaskPlanner:
    """Plans and decomposes high-level tasks."""

    def __init__(self, world_model: WorldModel) -> None:
        """Initialize task planner with world model."""
        self.world_model = world_model
        self._planning_strategies = {
            'navigation': NavigationPlanner(),
            'manipulation': ManipulationPlanner(),
            'communication': CommunicationPlanner()
        }

    async def plan_task(self, task: Task) -> Plan:
        """Generate execution plan for given task."""
        try:
            # Decompose task based on type
            strategy = self._select_planning_strategy(task)
            subtasks = await strategy.decompose(task)

            # Generate execution sequence
            plan = await self._generate_execution_sequence(subtasks)
            await self._validate_plan(plan)

            return plan
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            raise PlanningError(f"Failed to plan task {task.id}: {e}")
```

#### Execution Layer
```python
# Example: Operation execution
class AsyncExecutor:
    """Executes operations asynchronously with monitoring."""

    def __init__(self, operation_registry: OperationRegistry) -> None:
        """Initialize executor with operation registry."""
        self.operations = operation_registry
        self._execution_semaphore = asyncio.Semaphore(10)
        self._active_operations = {}

    async def execute_operation(self, operation: Operation) -> ExecutionResult:
        """Execute operation with async monitoring."""
        async with self._execution_semaphore:
            try:
                # Start execution with monitoring
                task_id = str(uuid.uuid4())
                execution_task = asyncio.create_task(
                    self._execute_with_monitoring(operation, task_id)
                )
                self._active_operations[task_id] = execution_task

                result = await execution_task
                return result
            except Exception as e:
                logger.error(f"Operation execution failed: {e}")
                raise ExecutionError(f"Failed to execute {operation.name}: {e}")
            finally:
                self._active_operations.pop(task_id, None)
```

#### Communication Layer
```python
# Example: Protocol management
class ProtocolManager:
    """Manages communication protocols and message routing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize protocol manager with configuration."""
        self.config = config
        self._protocols = {}
        self._message_handlers = defaultdict(list)
        self._connection_pool = ConnectionPool()

    async def send_message(self, message: Message, protocol: str) -> bool:
        """Send message using specified protocol."""
        try:
            protocol_handler = self._protocols.get(protocol)
            if not protocol_handler:
                raise ProtocolError(f"Unsupported protocol: {protocol}")

            # Send with connection reuse
            async with self._connection_pool.get_connection(protocol) as conn:
                await protocol_handler.send(conn, message)
                return True
        except Exception as e:
            logger.error(f"Message sending failed: {e}")
            raise CommunicationError(f"Failed to send message: {e}")
```

#### Models Layer
```python
# Example: LLM interface optimization
class OptimizedLLMInterface:
    """Optimized LLM interface with caching and batching."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize LLM interface with optimization features."""
        self.config = config
        self._request_cache = TTLCache(maxsize=1000, ttl=3600)
        self._batch_queue = asyncio.Queue()
        self._batch_processor_task = asyncio.create_task(self._process_batches())

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response with caching and optimization."""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(prompt, kwargs)
            if cache_key in self._request_cache:
                return self._request_cache[cache_key]

            # Add to batch queue
            future = asyncio.Future()
            await self._batch_queue.put((prompt, kwargs, future))

            # Wait for batch processing
            response = await future

            # Cache result
            self._request_cache[cache_key] = response
            return response
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise ModelError(f"Failed to generate response: {e}")
```

## Testing Guidelines

### Unit Testing
```python
# Example test structure
import pytest
from unittest.mock import AsyncMock, patch

class TestSensorProcessor:
    """Test cases for SensorProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create test processor instance."""
        config = {"sensor_type": "lidar", "update_rate": 10.0}
        return SensorProcessor(config)

    @pytest.mark.asyncio
    async def test_process_sensor_data_success(self, processor):
        """Test successful sensor data processing."""
        # Arrange
        test_data = SensorData(timestamp=time.time(), data=[1, 2, 3])

        # Act
        result = await processor.process_sensor_data(test_data)

        # Assert
        assert result is not None
        assert result.processed_timestamp > test_data.timestamp

    @pytest.mark.asyncio
    async def test_process_sensor_data_failure(self, processor):
        """Test sensor data processing failure handling."""
        # Arrange
        invalid_data = None

        # Act & Assert
        with pytest.raises(PerceptionError):
            await processor.process_sensor_data(invalid_data)
```

### Integration Testing
```python
# Example integration test
class TestPerceptionCognitiveIntegration:
    """Test integration between perception and cognitive layers."""

    @pytest.mark.asyncio
    async def test_perception_to_cognitive_flow(self):
        """Test data flow from perception to cognitive layer."""
        # Arrange
        sensor_processor = SensorProcessor(test_config)
        world_model = WorldModel(test_llm_interface)

        # Act
        sensor_data = await sensor_processor.get_sensor_data()
        perception_result = await sensor_processor.process_sensor_data(sensor_data)
        world_state = await world_model.update_world_state(perception_result)

        # Assert
        assert world_state is not None
        assert world_state.last_updated > 0
```

## Git Workflow

### Worktree Development
```bash
# Switch to layer-specific worktree
cd ../brain-perception  # or brain-cognitive, brain-planning, etc.

# Create feature branch
git checkout -b feature/new-sensor-algorithm

# Make changes and commit
git add .
git commit -m "feat: implement new sensor fusion algorithm

- Added Kalman filter implementation
- Improved noise reduction by 30%
- Enhanced real-time performance

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to remote
git push origin feature/new-sensor-algorithm
```

### Branch Naming Conventions
- `feature/layer-description` - New features
- `fix/layer-description` - Bug fixes
- `refactor/layer-description` - Code refactoring
- `test/layer-description` - Test additions
- `docs/layer-description` - Documentation updates

### Commit Message Format
```
type(scope): brief description

optional detailed explanation
- bullet points for specific changes

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

## Performance Guidelines

### Perception Layer
- Use async/await for sensor data processing
- Implement sensor fusion with optimized algorithms
- Cache processed results when appropriate
- Monitor processing latency and memory usage

### Cognitive Layer
- Implement reasoning result caching
- Use efficient data structures for world state
- Batch LLM requests when possible
- Optimize prompt templates for token efficiency

### Planning Layer
- Use efficient pathfinding algorithms (A*, RRT*)
- Implement dynamic replanning capabilities
- Cache planning results for recurring scenarios
- Optimize task decomposition algorithms

### Execution Layer
- Use async/await for operation execution
- Implement operation queuing and prioritization
- Monitor resource usage and performance
- Handle execution failures gracefully

### Communication Layer
- Use connection pooling for network operations
- Implement message batching when appropriate
- Optimize serialization/deserialization
- Monitor communication latency and throughput

### Models Layer
- Implement intelligent caching for LLM responses
- Use request batching to reduce API calls
- Optimize prompt templates for token efficiency
- Monitor token usage and API costs

## Security Guidelines

### General Security
- Validate all external inputs
- Use secure communication protocols
- Implement proper error handling without information leakage
- Regular security audits and dependency updates

### API Security
- Use authentication and authorization for external APIs
- Implement rate limiting for API calls
- Validate and sanitize all API inputs
- Use HTTPS for all external communications

### Data Security
- Encrypt sensitive data at rest and in transit
- Implement proper access controls
- Regular data backup and recovery testing
- Compliance with data protection regulations

## Documentation Guidelines

### Code Documentation
- Comprehensive docstrings for all classes and methods
- Type hints for better code understanding
- Usage examples in docstrings
- API documentation for external interfaces

### Architecture Documentation
- High-level architecture diagrams
- Data flow diagrams between layers
- Interface specifications
- Deployment and configuration guides

## Monitoring and Debugging

### Logging
```python
# Example logging setup
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use contextual logging
logger.info("Processing sensor data", extra={
    'sensor_type': sensor_type,
    'data_size': len(data),
    'processing_time': elapsed_time
})
```

### Performance Monitoring
- Monitor key metrics: latency, throughput, error rates
- Use distributed tracing for cross-layer operations
- Implement health checks for critical components
- Set up alerting for performance degradation

### Debugging Guidelines
- Use debug logging for troubleshooting
- Implement comprehensive error reporting
- Use structured logging for better analysis
- Provide debugging tools and utilities

## Deployment Guidelines

### Environment Setup
- Use containerization (Docker) for consistent deployments
- Implement environment-specific configurations
- Use infrastructure as code (Terraform) for cloud deployments
- Implement proper secret management

### CI/CD Pipeline
- Automated testing for all code changes
- Code quality checks (linting, formatting)
- Security scanning for vulnerabilities
- Automated deployment with rollback capabilities

### Production Monitoring
- Comprehensive logging and monitoring
- Performance metrics and alerting
- Error tracking and reporting
- Regular health checks and maintenance