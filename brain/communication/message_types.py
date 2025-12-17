"""
消息类型定义 - Message Types

定义Brain系统与机器人平台之间的通信消息格式
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class MessageType(Enum):
    """消息类型"""
    COMMAND = "command"             # 指令消息
    RESPONSE = "response"           # 响应消息
    TELEMETRY = "telemetry"         # 遥测消息
    STATUS = "status"               # 状态消息
    EVENT = "event"                 # 事件消息
    HEARTBEAT = "heartbeat"         # 心跳消息
    ERROR = "error"                 # 错误消息
    ACK = "ack"                     # 确认消息


class CommandType(Enum):
    """指令类型"""
    # 飞行控制
    TAKEOFF = "takeoff"
    LAND = "land"
    GOTO = "goto"
    HOVER = "hover"
    RETURN_TO_HOME = "return_to_home"
    EMERGENCY_STOP = "emergency_stop"
    
    # 任务控制
    START_MISSION = "start_mission"
    PAUSE_MISSION = "pause_mission"
    RESUME_MISSION = "resume_mission"
    ABORT_MISSION = "abort_mission"
    
    # 系统控制
    ARM = "arm"
    DISARM = "disarm"
    REBOOT = "reboot"
    SET_MODE = "set_mode"
    
    # 感知控制
    CAPTURE_IMAGE = "capture_image"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    SCAN_AREA = "scan_area"
    
    # 通用
    CUSTOM = "custom"


@dataclass
class BaseMessage:
    """消息基类"""
    message_id: str
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    target: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class CommandMessage(BaseMessage):
    """指令消息"""
    command: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: float = 30.0
    requires_ack: bool = True
    
    def __post_init__(self):
        self.message_type = MessageType.COMMAND
        if not self.command:
            raise ValueError("command is required")
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "command": self.command,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout": self.timeout,
            "requires_ack": self.requires_ack
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommandMessage':
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data.get("message_type", "command")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            source=data.get("source", ""),
            target=data.get("target", ""),
            command=data["command"],
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 1),
            timeout=data.get("timeout", 30.0),
            requires_ack=data.get("requires_ack", True)
        )


@dataclass
class ResponseMessage(BaseMessage):
    """响应消息"""
    success: bool = True
    request_id: str = ""          # 对应的请求ID
    error: Optional[str] = None
    error_code: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.message_type = MessageType.RESPONSE
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "request_id": self.request_id,
            "success": self.success,
            "error": self.error,
            "error_code": self.error_code,
            "data": self.data
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseMessage':
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data.get("message_type", "response")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            source=data.get("source", ""),
            target=data.get("target", ""),
            request_id=data.get("request_id", ""),
            success=data.get("success", True),
            error=data.get("error"),
            error_code=data.get("error_code"),
            data=data.get("data", {})
        )


@dataclass
class TelemetryMessage(BaseMessage):
    """遥测消息"""
    # 位置
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    
    # 姿态
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    # 速度
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    ground_speed: float = 0.0
    
    # 系统状态
    battery: float = 100.0
    signal_strength: float = 100.0
    gps_satellites: int = 0
    gps_fix_type: str = "none"
    
    # 模式
    flight_mode: str = "manual"
    armed: bool = False
    
    # 额外数据
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.message_type = MessageType.TELEMETRY
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "position": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "altitude": self.altitude
            },
            "attitude": {
                "roll": self.roll,
                "pitch": self.pitch,
                "yaw": self.yaw
            },
            "velocity": {
                "x": self.velocity_x,
                "y": self.velocity_y,
                "z": self.velocity_z,
                "ground_speed": self.ground_speed
            },
            "system": {
                "battery": self.battery,
                "signal_strength": self.signal_strength,
                "gps_satellites": self.gps_satellites,
                "gps_fix_type": self.gps_fix_type,
                "flight_mode": self.flight_mode,
                "armed": self.armed
            },
            "extra_data": self.extra_data
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelemetryMessage':
        pos = data.get("position", {})
        att = data.get("attitude", {})
        vel = data.get("velocity", {})
        sys = data.get("system", {})
        
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data.get("message_type", "telemetry")),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            source=data.get("source", ""),
            target=data.get("target", ""),
            latitude=pos.get("latitude", 0.0),
            longitude=pos.get("longitude", 0.0),
            altitude=pos.get("altitude", 0.0),
            roll=att.get("roll", 0.0),
            pitch=att.get("pitch", 0.0),
            yaw=att.get("yaw", 0.0),
            velocity_x=vel.get("x", 0.0),
            velocity_y=vel.get("y", 0.0),
            velocity_z=vel.get("z", 0.0),
            ground_speed=vel.get("ground_speed", 0.0),
            battery=sys.get("battery", 100.0),
            signal_strength=sys.get("signal_strength", 100.0),
            gps_satellites=sys.get("gps_satellites", 0),
            gps_fix_type=sys.get("gps_fix_type", "none"),
            flight_mode=sys.get("flight_mode", "manual"),
            armed=sys.get("armed", False),
            extra_data=data.get("extra_data", {})
        )


@dataclass
class StatusMessage(BaseMessage):
    """状态消息"""
    status: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'message_type'):
            self.message_type = MessageType.STATUS
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "status": self.status,
            "details": self.details
        })
        return data


@dataclass
class EventMessage(BaseMessage):
    """事件消息"""
    event_type: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    
    def __post_init__(self):
        self.message_type = MessageType.EVENT
        if not self.event_type:
            raise ValueError("event_type is required")
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "event_type": self.event_type,
            "event_data": self.event_data,
            "severity": self.severity
        })
        return data


@dataclass
class HeartbeatMessage(BaseMessage):
    """心跳消息"""
    sequence: int = 0
    system_status: str = "ok"
    
    def __post_init__(self):
        self.message_type = MessageType.HEARTBEAT
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "sequence": self.sequence,
            "system_status": self.system_status
        })
        return data


def parse_message(data: Dict[str, Any]) -> BaseMessage:
    """解析消息"""
    message_type = MessageType(data.get("message_type", "status"))
    
    parsers = {
        MessageType.COMMAND: CommandMessage.from_dict,
        MessageType.RESPONSE: ResponseMessage.from_dict,
        MessageType.TELEMETRY: TelemetryMessage.from_dict
    }
    
    parser = parsers.get(message_type)
    if parser:
        return parser(data)
    
    # 默认返回基础消息
    return BaseMessage(
        message_id=data.get("message_id", ""),
        message_type=message_type,
        source=data.get("source", ""),
        target=data.get("target", "")
    )

