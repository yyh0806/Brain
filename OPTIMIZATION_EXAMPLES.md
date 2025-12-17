# Communication Layer Optimization Examples

This document provides concrete code examples for implementing the optimizations described in the analysis report.

## 1. Async Command Queue Implementation

```python
import asyncio
import heapq
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import time

class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class QueuedCommand:
    priority: Priority
    timestamp: float
    command: str
    parameters: Dict[str, Any]
    timeout: float
    future: asyncio.Future

    def __lt__(self, other):
        # For heap queue - lower priority number = higher priority
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp

class AsyncCommandQueue:
    """
    High-performance async command queue with priority support
    """
    def __init__(self, max_size: int = 1000):
        self._queue = []
        self._queue_lock = asyncio.Lock()
        self._max_size = max_size
        self._processing = False
        self._stats = {
            "queued": 0,
            "processed": 0,
            "failed": 0,
            "average_wait_time": 0.0
        }

    async def put(
        self,
        command: str,
        parameters: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0
    ) -> asyncio.Future:
        """Queue a command for processing"""
        future = asyncio.get_event_loop().create_future()

        async with self._queue_lock:
            if len(self._queue) >= self._max_size:
                raise QueueFullError(f"Command queue is full (max: {self._max_size})")

            queued_cmd = QueuedCommand(
                priority=priority,
                timestamp=time.time(),
                command=command,
                parameters=parameters,
                timeout=timeout,
                future=future
            )

            heapq.heappush(self._queue, queued_cmd)
            self._stats["queued"] += 1

        if not self._processing:
            asyncio.create_task(self._process_queue())

        return future

    async def _process_queue(self):
        """Process queued commands"""
        if self._processing:
            return

        self._processing = True
        logger.info("Started command queue processing")

        try:
            while self._queue:
                async with self._queue_lock:
                    if not self._queue:
                        break

                    queued_cmd = heapq.heappop(self._queue)

                start_time = time.time()

                try:
                    # Process command with timeout
                    await asyncio.wait_for(
                        self._execute_command(queued_cmd),
                        timeout=queued_cmd.timeout
                    )

                    wait_time = time.time() - start_time
                    self._update_average_wait(wait_time)
                    self._stats["processed"] += 1

                except Exception as e:
                    logger.error(f"Command failed: {queued_cmd.command} - {e}")
                    queued_cmd.future.set_exception(e)
                    self._stats["failed"] += 1

        finally:
            self._processing = False
            logger.info("Stopped command queue processing")

    async def _execute_command(self, queued_cmd: QueuedCommand):
        """Execute a single command"""
        # This would delegate to the actual robot interface
        # For now, simulate execution
        await asyncio.sleep(0.1)
        queued_cmd.future.set_result({"success": True})

    def _update_average_wait(self, wait_time: float):
        """Update average wait time"""
        total = self._stats["processed"] + self._stats["failed"]
        if total > 0:
            self._stats["average_wait_time"] = (
                (self._stats["average_wait_time"] * (total - 1) + wait_time) / total
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return self._stats.copy()

    async def clear(self):
        """Clear all pending commands"""
        async with self._queue_lock:
            for queued_cmd in self._queue:
                queued_cmd.future.cancel()
            self._queue.clear()

class QueueFullError(Exception):
    """Raised when command queue is full"""
    pass
```

## 2. Differential Telemetry Updates

```python
import json
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class TelemetryDelta:
    """Represents changes in telemetry data"""
    timestamp: float
    changed_fields: Dict[str, Any]
    removed_fields: Set[str]

class TelemetryDiffer:
    """
    Computes and applies differential updates to telemetry data
    """
    def __init__(self):
        self._last_state = {}
        self._field_thresholds = {
            "position": 0.01,  # 1 cm
            "orientation": 0.01,  # ~0.5 degrees
            "velocity": 0.1,  # 0.1 m/s
            "battery": 0.5,  # 0.5%
        }

    def compute_delta(self, current_telemetry: Dict[str, Any]) -> Optional[TelemetryDelta]:
        """
        Compute delta between current and last telemetry state
        """
        delta = TelemetryDelta(
            timestamp=datetime.now().timestamp(),
            changed_fields={},
            removed_fields=set()
        )

        # Check for changes
        for key, current_value in current_telemetry.items():
            last_value = self._last_state.get(key)

            if last_value is None:
                # New field
                delta.changed_fields[key] = current_value
            elif self._has_significant_change(key, current_value, last_value):
                # Significant change detected
                delta.changed_fields[key] = current_value

        # Check for removed fields
        for key in self._last_state:
            if key not in current_telemetry:
                delta.removed_fields.add(key)

        # Update last state
        self._last_state = current_telemetry.copy()

        # Return delta only if there are changes
        if delta.changed_fields or delta.removed_fields:
            return delta

        return None

    def _has_significant_change(self, key: str, current: Any, last: Any) -> bool:
        """
        Check if a field has changed significantly
        """
        if key in self._field_thresholds:
            threshold = self._field_thresholds[key]

            if isinstance(current, (int, float)) and isinstance(last, (int, float)):
                return abs(current - last) > threshold
            elif isinstance(current, dict) and isinstance(last, dict):
                # Handle nested dictionaries (e.g., position, orientation)
                return self._dict_has_significant_change(current, last, threshold)

        # For other fields, any change is significant
        return current != last

    def _dict_has_significant_change(self, current: Dict, last: Dict, threshold: float) -> bool:
        """
        Check if dictionary has significant changes
        """
        for key in current:
            if key not in last:
                return True

            current_val = current[key]
            last_val = last[key]

            if isinstance(current_val, (int, float)) and isinstance(last_val, (int, float)):
                if abs(current_val - last_val) > threshold:
                    return True

        return False

    def apply_delta(self, base_telemetry: Dict[str, Any], delta: TelemetryDelta) -> Dict[str, Any]:
        """
        Apply delta to base telemetry to get current state
        """
        result = base_telemetry.copy()

        # Apply changes
        result.update(delta.changed_fields)

        # Remove fields
        for key in delta.removed_fields:
            result.pop(key, None)

        return result

    def compress_delta(self, delta: TelemetryDelta) -> bytes:
        """
        Compress delta for transmission
        """
        # Use more compact representation
        data = {
            "t": delta.timestamp,
            "c": delta.changed_fields,
            "r": list(delta.removed_fields)
        }

        # Use msgpack for binary serialization (more compact than JSON)
        try:
            import msgpack
            return msgpack.packb(data, use_bin_type=True)
        except ImportError:
            # Fallback to JSON with compression
            json_str = json.dumps(data, separators=(',', ':'))
            import gzip
            return gzip.compress(json_str.encode())

    def decompress_delta(self, compressed_data: bytes) -> TelemetryDelta:
        """
        Decompress delta from transmission format
        """
        try:
            import msgpack
            data = msgpack.unpackb(compressed_data, raw=False, strict_map_key=False)
        except ImportError:
            import gzip
            json_str = gzip.decompress(compressed_data).decode()
            data = json.loads(json_str)

        return TelemetryDelta(
            timestamp=data["t"],
            changed_fields=data["c"],
            removed_fields=set(data["r"])
        )
```

## 3. Connection Pool with Health Monitoring

```python
import asyncio
import time
import weakref
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
from loguru import logger

class ConnectionStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    CLOSING = "closing"

@dataclass
class ConnectionMetrics:
    created_at: float
    last_used: float
    usage_count: int
    error_count: int
    last_health_check: float
    consecutive_failures: int

class PooledConnection:
    """
    Wrapper for individual connections with health tracking
    """
    def __init__(self, connection_id: str, create_fn: Callable):
        self.id = connection_id
        self._create_fn = create_fn
        self._connection = None
        self.status = ConnectionStatus.IDLE
        self.metrics = ConnectionMetrics(
            created_at=time.time(),
            last_used=time.time(),
            usage_count=0,
            error_count=0,
            last_health_check=time.time(),
            consecutive_failures=0
        )
        self._lock = asyncio.Lock()

    async def get(self):
        """Get the underlying connection"""
        async with self._lock:
            if self._connection is None:
                self._connection = await self._create_fn()
                self.status = ConnectionStatus.ACTIVE

            self.metrics.last_used = time.time()
            self.metrics.usage_count += 1
            return self._connection

    async def health_check(self) -> bool:
        """Perform health check on connection"""
        try:
            conn = await self.get()
            # Implement actual health check based on connection type
            # For TCP: check if socket is still connected
            # For WebSocket: send ping
            # For ROS2: check node status

            # Placeholder health check
            await asyncio.sleep(0.01)

            self.metrics.last_health_check = time.time()
            self.metrics.consecutive_failures = 0
            self.status = ConnectionStatus.ACTIVE
            return True

        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.consecutive_failures += 1

            if self.metrics.consecutive_failures > 3:
                self.status = ConnectionStatus.UNHEALTHY
                await self._recreate()

            logger.warning(f"Connection {self.id} health check failed: {e}")
            return False

    async def _recreate(self):
        """Recreate the connection"""
        async with self._lock:
            try:
                if self._connection:
                    # Close old connection if possible
                    if hasattr(self._connection, 'close'):
                        await self._connection.close()

                self._connection = await self._create_fn()
                self.metrics.consecutive_failures = 0
                self.status = ConnectionStatus.ACTIVE

            except Exception as e:
                logger.error(f"Failed to recreate connection {self.id}: {e}")
                self.status = ConnectionStatus.UNHEALTHY

    async def close(self):
        """Close the connection"""
        self.status = ConnectionStatus.CLOSING

        async with self._lock:
            if self._connection and hasattr(self._connection, 'close'):
                await self._connection.close()
            self._connection = None

class AdvancedConnectionPool:
    """
    Advanced connection pool with health monitoring and auto-recovery
    """
    def __init__(
        self,
        create_fn: Callable,
        min_connections: int = 2,
        max_connections: int = 10,
        health_check_interval: float = 30.0,
        max_idle_time: float = 300.0,
        recovery_retry_delay: float = 5.0
    ):
        self._create_fn = create_fn
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.health_check_interval = health_check_interval
        self.max_idle_time = max_idle_time
        self.recovery_retry_delay = recovery_retry_delay

        self._connections: Dict[str, PooledConnection] = {}
        self._available = asyncio.Queue()
        self._total_connections = 0
        self._lock = asyncio.Lock()

        # Health monitoring
        self._health_monitor_task = None
        self._running = False

        # Statistics
        self._stats = {
            "created": 0,
            "destroyed": 0,
            "acquired": 0,
            "released": 0,
            "health_checks": 0,
            "recoveries": 0
        }

    async def initialize(self):
        """Initialize the connection pool"""
        self._running = True

        # Create initial connections
        for _ in range(self.min_connections):
            await self._create_connection()

        # Start health monitoring
        self._health_monitor_task = asyncio.create_task(self._health_monitor())

        logger.info(f"Connection pool initialized with {self.min_connections} connections")

    async def acquire(self, timeout: Optional[float] = None) -> Any:
        """
        Acquire a connection from the pool
        """
        # Try to get from available queue
        try:
            if timeout:
                connection_id = await asyncio.wait_for(
                    self._available.get(),
                    timeout=timeout
                )
            else:
                connection_id = await self._available.get()
        except asyncio.TimeoutError:
            # Try to create new connection
            async with self._lock:
                if self._total_connections < self.max_connections:
                    connection = await self._create_connection()
                    connection_id = connection.id
                else:
                    raise PoolExhaustedError("Connection pool exhausted")

        # Get actual connection
        pooled_conn = self._connections.get(connection_id)
        if not pooled_conn or pooled_conn.status == ConnectionStatus.UNHEALTHY:
            # Connection is unhealthy, get a different one
            if pooled_conn:
                await self._remove_connection(pooled_conn)
            return await self.acquire(timeout)

        self._stats["acquired"] += 1
        return await pooled_conn.get()

    async def release(self, connection: Any):
        """
        Release a connection back to the pool
        """
        # Find the pooled connection for this actual connection
        pooled_conn = None
        for conn in self._connections.values():
            if conn._connection == connection:
                pooled_conn = conn
                break

        if pooled_conn:
            pooled_conn.status = ConnectionStatus.IDLE
            await self._available.put(pooled_conn.id)
            self._stats["released"] += 1
        else:
            # Unknown connection, close it
            if hasattr(connection, 'close'):
                await connection.close()

    async def _create_connection(self) -> PooledConnection:
        """Create a new connection"""
        connection_id = f"conn_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        pooled_conn = PooledConnection(connection_id, self._create_fn)

        async with self._lock:
            self._connections[connection_id] = pooled_conn
            self._total_connections += 1
            self._stats["created"] += 1

        # Initialize the connection
        await pooled_conn.get()

        # Add to available queue
        await self._available.put(connection_id)

        return pooled_conn

    async def _remove_connection(self, pooled_conn: PooledConnection):
        """Remove a connection from the pool"""
        await pooled_conn.close()

        async with self._lock:
            self._connections.pop(pooled_conn.id, None)
            self._total_connections -= 1
            self._stats["destroyed"] += 1

    async def _health_monitor(self):
        """Monitor connection health"""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
                await self._cleanup_idle_connections()
                await self._maintain_min_connections()

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all connections"""
        connections_to_check = list(self._connections.values())

        tasks = [conn.health_check() for conn in connections_to_check]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for conn, result in zip(connections_to_check, results):
            self._stats["health_checks"] += 1

            if isinstance(result, Exception) or not result:
                logger.warning(f"Connection {conn.id} failed health check")
                self._stats["recoveries"] += 1

    async def _cleanup_idle_connections(self):
        """Remove idle connections beyond minimum"""
        current_time = time.time()
        idle_connections = []

        for conn in self._connections.values():
            if (conn.status == ConnectionStatus.IDLE and
                current_time - conn.metrics.last_used > self.max_idle_time and
                self._total_connections > self.min_connections):
                idle_connections.append(conn)

        for conn in idle_connections:
            await self._remove_connection(conn)
            logger.debug(f"Removed idle connection: {conn.id}")

    async def _maintain_min_connections(self):
        """Maintain minimum number of connections"""
        while self._total_connections < self.min_connections:
            try:
                await self._create_connection()
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                break

    async def close(self):
        """Close all connections and shutdown pool"""
        self._running = False

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        connections = list(self._connections.values())
        tasks = [conn.close() for conn in connections]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._connections.clear()
        self._total_connections = 0

        logger.info("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = self._stats.copy()
        stats.update({
            "total_connections": self._total_connections,
            "active_connections": sum(
                1 for c in self._connections.values()
                if c.status == ConnectionStatus.ACTIVE
            ),
            "idle_connections": sum(
                1 for c in self._connections.values()
                if c.status == ConnectionStatus.IDLE
            ),
            "unhealthy_connections": sum(
                1 for c in self._connections.values()
                if c.status == ConnectionStatus.UNHEALTHY
            )
        })
        return stats

class PoolExhaustedError(Exception):
    """Raised when connection pool is exhausted"""
    pass
```

## 4. Binary Protocol with Protocol Buffers

```python
# First, create the .proto file
"""
syntax = "proto3";

package brain;

// Base message
message BaseMessage {
  string message_id = 1;
  MessageType message_type = 2;
  int64 timestamp = 3;
  string source = 4;
  string target = 5;
}

enum MessageType {
  COMMAND = 0;
  RESPONSE = 1;
  TELEMETRY = 2;
  HEARTBEAT = 3;
}

// Command message
message CommandMessage {
  BaseMessage base = 1;
  string command = 2;
  map<string, Value> parameters = 3;
  int32 priority = 4;
  float timeout = 5;
  bool requires_ack = 6;
}

// Response message
message ResponseMessage {
  BaseMessage base = 1;
  bool success = 2;
  string request_id = 3;
  string error = 4;
  string error_code = 5;
  map<string, Value> data = 6;
}

// Telemetry message
message TelemetryMessage {
  BaseMessage base = 1;

  // Position
  double latitude = 2;
  double longitude = 3;
  double altitude = 4;

  // Attitude
  double roll = 5;
  double pitch = 6;
  double yaw = 7;

  // Velocity
  double velocity_x = 8;
  double velocity_y = 9;
  double velocity_z = 10;
  double ground_speed = 11;

  // System status
  double battery = 12;
  double signal_strength = 13;
  int32 gps_satellites = 14;
  string gps_fix_type = 15;
  string flight_mode = 16;
  bool armed = 17;
}

// Generic value type for parameters
message Value {
  oneof value {
    string string_value = 1;
    double double_value = 2;
    int32 int_value = 3;
    bool bool_value = 4;
    string json_value = 5;
  }
}
"""

# Then the Python implementation
from brain.protos import brain_pb2
from google.protobuf import json_format
from google.protobuf.message import Message
import gzip
import asyncio
from typing import Dict, Any, Union

class ProtobufAdapter:
    """
    Adapter for Protocol Buffers serialization
    """
    @staticmethod
    def command_to_protobuf(command_msg: 'CommandMessage') -> bytes:
        """Convert CommandMessage to protobuf"""
        pb_msg = brain_pb2.CommandMessage()

        # Base message
        pb_msg.base.message_id = command_msg.message_id
        pb_msg.base.message_type = brain_pb2.MessageType.COMMAND
        pb_msg.base.timestamp = int(command_msg.timestamp.timestamp() * 1000)
        pb_msg.base.source = command_msg.source
        pb_msg.base.target = command_msg.target

        # Command specific
        pb_msg.command = command_msg.command
        pb_msg.priority = command_msg.priority
        pb_msg.timeout = command_msg.timeout
        pb_msg.requires_ack = command_msg.requires_ack

        # Parameters
        for key, value in command_msg.parameters.items():
            pb_msg.parameters[key].CopyFrom(
                ProtobufAdapter._value_to_protobuf(value)
            )

        return pb_msg.SerializeToString()

    @staticmethod
    def command_from_protobuf(data: bytes) -> 'CommandMessage':
        """Convert protobuf to CommandMessage"""
        pb_msg = brain_pb2.CommandMessage()
        pb_msg.ParseFromString(data)

        parameters = {}
        for key, value in pb_msg.parameters.items():
            parameters[key] = ProtobufAdapter._value_from_protobuf(value)

        from datetime import datetime
        return CommandMessage(
            message_id=pb_msg.base.message_id,
            timestamp=datetime.fromtimestamp(pb_msg.base.timestamp / 1000),
            source=pb_msg.base.source,
            target=pb_msg.base.target,
            command=pb_msg.command,
            parameters=parameters,
            priority=pb_msg.priority,
            timeout=pb_msg.timeout,
            requires_ack=pb_msg.requires_ack
        )

    @staticmethod
    def _value_to_protobuf(value: Any) -> brain_pb2.Value:
        """Convert Python value to protobuf Value"""
        pb_value = brain_pb2.Value()

        if isinstance(value, str):
            pb_value.string_value = value
        elif isinstance(value, (int, float)):
            pb_value.double_value = float(value)
        elif isinstance(value, bool):
            pb_value.bool_value = value
        elif isinstance(value, (dict, list)):
            import json
            pb_value.json_value = json.dumps(value)
        else:
            pb_value.string_value = str(value)

        return pb_value

    @staticmethod
    def _value_from_protobuf(pb_value: brain_pb2.Value) -> Any:
        """Convert protobuf Value to Python"""
        which = pb_value.WhichOneof("value")

        if which == "string_value":
            return pb_value.string_value
        elif which == "double_value":
            return pb_value.double_value
        elif which == "int_value":
            return pb_value.int_value
        elif which == "bool_value":
            return pb_value.bool_value
        elif which == "json_value":
            import json
            return json.loads(pb_value.json_value)
        else:
            return None

class BinaryProtocolHandler:
    """
    Handles binary protocol communication with compression
    """
    def __init__(self, use_compression: bool = True):
        self.use_compression = use_compression
        self.protobuf_adapter = ProtobufAdapter()
        self._stats = {
            "messages_sent": 0,
            "bytes_sent": 0,
            "messages_received": 0,
            "bytes_received": 0,
            "compression_ratio": 1.0
        }

    async def serialize_message(self, message: Union['CommandMessage', 'ResponseMessage', 'TelemetryMessage']) -> bytes:
        """
        Serialize message to binary format
        """
        # Convert to protobuf
        if isinstance(message, CommandMessage):
            proto_data = self.protobuf_adapter.command_to_protobuf(message)
        else:
            # Handle other message types
            proto_data = b""

        # Add message type prefix
        msg_type = message.__class__.__name__.encode()
        data = len(msg_type).to_bytes(1, 'big') + msg_type + proto_data

        # Compress if enabled
        if self.use_compression:
            compressed = gzip.compress(data)
            compression_ratio = len(compressed) / len(data)

            # Only use compression if it actually helps
            if compression_ratio < 0.9:
                data = b'\x01' + compressed  # Prefix with compression flag
                self._update_compression_stats(len(data), len(compressed))
            else:
                data = b'\x00' + data  # No compression

        self._stats["messages_sent"] += 1
        self._stats["bytes_sent"] += len(data)

        return data

    async def deserialize_message(self, data: bytes) -> Union['CommandMessage', 'ResponseMessage', 'TelemetryMessage']:
        """
        Deserialize message from binary format
        """
        self._stats["messages_received"] += 1
        self._stats["bytes_received"] += len(data)

        # Check compression flag
        if data[0] == 1:
            data = gzip.decompress(data[1:])
        else:
            data = data[1:]

        # Extract message type
        msg_type_len = data[0]
        msg_type = data[1:1+msg_type_len].decode()
        proto_data = data[1+msg_type_len:]

        # Deserialize based on type
        if msg_type == "CommandMessage":
            return self.protobuf_adapter.command_from_protobuf(proto_data)
        else:
            # Handle other message types
            raise ValueError(f"Unknown message type: {msg_type}")

    def _update_compression_stats(self, original_size: int, compressed_size: int):
        """Update compression statistics"""
        total_compression = (
            self._stats.get("total_compressed", 0) + compressed_size
        )
        total_original = (
            self._stats.get("total_original", 0) + original_size
        )

        self._stats["total_compressed"] = total_compression
        self._stats["total_original"] = total_original
        self._stats["compression_ratio"] = total_compression / total_original

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics"""
        return self._stats.copy()
```

## 5. WebSocket Real-time Communication

```python
import asyncio
import json
import websockets
from typing import Dict, Any, Set, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
from loguru import logger

@dataclass
class StreamSubscription:
    """Represents a subscription to a data stream"""
    subscription_id: str
    stream_name: str
    filter_fn: Optional[Callable] = None
    last_sent: Optional[float] = None

class WebSocketManager:
    """
    Manages WebSocket connections for real-time communication
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, Set[StreamSubscription]] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self._server = None
        self._running = False

        # Statistics
        self._stats = {
            "connections_opened": 0,
            "connections_closed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "active_subscriptions": 0
        }

    async def start(self):
        """Start the WebSocket server"""
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        self._running = True
        logger.info(f"WebSocket server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server"""
        self._running = False

        # Close all connections
        close_tasks = [
            conn.close() for conn in self.connections.values()
        ]
        await asyncio.gather(*close_tasks, return_exceptions=True)

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("WebSocket server stopped")

    async def _handle_client(self, websocket, path):
        """
        Handle a new WebSocket client connection
        """
        connection_id = str(uuid.uuid4())
        self.connections[connection_id] = websocket
        self._stats["connections_opened"] += 1

        logger.info(f"Client connected: {connection_id} from {websocket.remote_address}")

        try:
            # Send welcome message
            await self._send_message(websocket, {
                "type": "welcome",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            })

            # Handle messages from client
            async for message in websocket:
                await self._handle_message(connection_id, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling client {connection_id}: {e}")
        finally:
            # Cleanup
            await self._cleanup_connection(connection_id)

    async def _handle_message(self, connection_id: str, message: str):
        """
        Handle message from client
        """
        self._stats["messages_received"] += 1

        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "subscribe":
                await self._handle_subscribe(connection_id, data)
            elif msg_type == "unsubscribe":
                await self._handle_unsubscribe(connection_id, data)
            elif msg_type == "ping":
                await self._handle_ping(connection_id, data)
            else:
                # Call custom message handler
                handler = self.message_handlers.get(msg_type)
                if handler:
                    await handler(connection_id, data)
                else:
                    logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {connection_id}")
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")

    async def _handle_subscribe(self, connection_id: str, data: Dict):
        """Handle subscription request"""
        stream_name = data.get("stream")
        subscription_id = data.get("subscription_id", str(uuid.uuid4()))

        # Create subscription
        subscription = StreamSubscription(
            subscription_id=subscription_id,
            stream_name=stream_name
        )

        # Add to subscriptions
        if stream_name not in self.subscriptions:
            self.subscriptions[stream_name] = set()

        self.subscriptions[stream_name].add(subscription)
        self._stats["active_subscriptions"] += 1

        # Send confirmation
        websocket = self.connections[connection_id]
        await self._send_message(websocket, {
            "type": "subscription_confirmed",
            "subscription_id": subscription_id,
            "stream": stream_name
        })

        logger.info(f"Client {connection_id} subscribed to {stream_name}")

    async def _handle_unsubscribe(self, connection_id: str, data: Dict):
        """Handle unsubscription request"""
        subscription_id = data.get("subscription_id")

        # Find and remove subscription
        for stream_subs in self.subscriptions.values():
            for sub in stream_subs.copy():
                if sub.subscription_id == subscription_id:
                    stream_subs.remove(sub)
                    self._stats["active_subscriptions"] -= 1
                    break

        # Send confirmation
        websocket = self.connections[connection_id]
        await self._send_message(websocket, {
            "type": "unsubscription_confirmed",
            "subscription_id": subscription_id
        })

    async def _handle_ping(self, connection_id: str, data: Dict):
        """Handle ping request"""
        websocket = self.connections[connection_id]
        await self._send_message(websocket, {
            "type": "pong",
            "timestamp": data.get("timestamp")
        })

    async def _cleanup_connection(self, connection_id: str):
        """Clean up when client disconnects"""
        self.connections.pop(connection_id, None)
        self._stats["connections_closed"] += 1

        # Remove all subscriptions for this connection
        removed_count = 0
        for stream_subs in self.subscriptions.values():
            for sub in list(stream_subs):
                stream_subs.remove(sub)
                removed_count += 1

        self._stats["active_subscriptions"] -= removed_count

    async def _send_message(self, websocket, message: Dict):
        """Send message to WebSocket client"""
        try:
            await websocket.send(json.dumps(message))
            self._stats["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast_to_stream(self, stream_name: str, data: Any):
        """
        Broadcast data to all subscribers of a stream
        """
        if stream_name not in self.subscriptions:
            return

        message = {
            "type": "stream_data",
            "stream": stream_name,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        # Send to all subscribers
        for subscription in self.subscriptions[stream_name].copy():
            try:
                # Find websocket for subscription
                for websocket in self.connections.values():
                    await self._send_message(websocket, message)
                    break
            except Exception as e:
                logger.error(f"Error broadcasting to {stream_name}: {e}")

    def register_handler(self, message_type: str, handler: Callable):
        """Register custom message handler"""
        self.message_handlers[message_type] = handler

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        stats = self._stats.copy()
        stats.update({
            "active_connections": len(self.connections),
            "active_streams": len(self.subscriptions)
        })
        return stats

# Example usage:
"""
# Create WebSocket manager
ws_manager = WebSocketManager()

# Register custom handlers
async def handle_command(connection_id: str, data: Dict):
    command = data.get("command")
    # Process command
    await ws_manager.broadcast_to_stream("command_response", {
        "status": "completed",
        "command": command
    })

ws_manager.register_handler("command", handle_command)

# Start server
await ws_manager.start()

# Broadcast telemetry data
async def broadcast_telemetry():
    while True:
        telemetry = await get_robot_telemetry()
        await ws_manager.broadcast_to_stream("telemetry", telemetry)
        await asyncio.sleep(0.1)

asyncio.create_task(broadcast_telemetry())
"""