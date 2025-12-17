"""
世界状态 - World State

负责:
- 维护全局状态
- 状态查询与更新
- 条件评估
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
from loguru import logger

from brain.execution.operations.base import Operation, OperationResult


@dataclass
class RobotState:
    """机器人状态"""
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0, "lat": 0, "lon": 0, "alt": 0})
    velocity: Dict[str, float] = field(default_factory=lambda: {"vx": 0, "vy": 0, "vz": 0})
    orientation: Dict[str, float] = field(default_factory=lambda: {"roll": 0, "pitch": 0, "yaw": 0})
    battery: float = 100.0
    fuel: float = 100.0
    status: str = "idle"
    armed: bool = False
    airborne: bool = False
    on_ground: bool = True
    moving: bool = False
    ready: bool = True
    home_position: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "velocity": self.velocity,
            "orientation": self.orientation,
            "battery": self.battery,
            "fuel": self.fuel,
            "status": self.status,
            "armed": self.armed,
            "airborne": self.airborne,
            "on_ground": self.on_ground,
            "moving": self.moving,
            "ready": self.ready,
            "home_position": self.home_position
        }


@dataclass
class SystemState:
    """系统状态"""
    gps: Dict[str, Any] = field(default_factory=lambda: {"status": "available", "satellites": 12, "accuracy": 0.01})
    imu: Dict[str, Any] = field(default_factory=lambda: {"status": "ok"})
    camera: Dict[str, Any] = field(default_factory=lambda: {"status": "ok", "recording": False})
    lidar: Dict[str, Any] = field(default_factory=lambda: {"status": "ok"})
    motors: Dict[str, Any] = field(default_factory=lambda: {"status": "ok", "ready": True})
    communication: Dict[str, Any] = field(default_factory=lambda: {"status": "connected", "signal_strength": 95})
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gps": self.gps,
            "imu": self.imu,
            "camera": self.camera,
            "lidar": self.lidar,
            "motors": self.motors,
            "communication": self.communication
        }


class WorldState:
    """
    世界状态
    
    维护系统的全局状态，包括机器人状态、环境状态等
    """
    
    def __init__(self):
        # 机器人状态
        self.robot = RobotState()
        
        # 系统状态
        self.system = SystemState()
        
        # 环境状态
        self.environment: Dict[str, Any] = {
            "weather": "clear",
            "wind_speed": 0.0,
            "visibility": "good",
            "temperature": 25.0
        }
        
        # 任务相关状态
        self.mission: Dict[str, Any] = {
            "id": None,
            "status": "idle",
            "progress": 0.0,
            "current_operation": None
        }
        
        # 检测到的物体
        self.objects: List[Dict[str, Any]] = []
        
        # 自定义状态变量
        self.custom: Dict[str, Any] = {}
        
        # 状态历史
        self.history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # 最后更新时间
        self.last_update = datetime.now()
        
        logger.info("WorldState 初始化完成")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取状态值
        
        Args:
            path: 状态路径 (如 "robot.battery", "system.gps.status")
            default: 默认值
            
        Returns:
            状态值
        """
        try:
            parts = path.split(".")
            value = self._get_value(parts)
            return value if value is not None else default
        except (KeyError, AttributeError, TypeError):
            return default
    
    def _get_value(self, parts: List[str]) -> Any:
        """递归获取值"""
        if not parts:
            return None
        
        first = parts[0]
        rest = parts[1:]
        
        # 根级别
        if first == "robot":
            obj = self.robot
        elif first == "system":
            obj = self.system
        elif first == "environment":
            obj = self.environment
        elif first == "mission":
            obj = self.mission
        elif first == "custom":
            obj = self.custom
        else:
            return None
        
        # 继续遍历
        for part in rest:
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
            
            if obj is None:
                return None
        
        return obj
    
    def set(self, path: str, value: Any):
        """
        设置状态值
        
        Args:
            path: 状态路径
            value: 值
        """
        parts = path.split(".")
        
        if len(parts) < 2:
            return
        
        first = parts[0]
        
        if first == "robot":
            self._set_nested(self.robot, parts[1:], value)
        elif first == "system":
            self._set_nested(self.system, parts[1:], value)
        elif first == "environment":
            self._set_nested(self.environment, parts[1:], value)
        elif first == "mission":
            self._set_nested(self.mission, parts[1:], value)
        elif first == "custom":
            self._set_nested(self.custom, parts[1:], value)
        
        self.last_update = datetime.now()
    
    def _set_nested(self, obj: Any, parts: List[str], value: Any):
        """递归设置值"""
        if not parts:
            return
        
        if len(parts) == 1:
            if isinstance(obj, dict):
                obj[parts[0]] = value
            elif hasattr(obj, parts[0]):
                setattr(obj, parts[0], value)
        else:
            first = parts[0]
            rest = parts[1:]
            
            if isinstance(obj, dict):
                if first not in obj:
                    obj[first] = {}
                self._set_nested(obj[first], rest, value)
            elif hasattr(obj, first):
                self._set_nested(getattr(obj, first), rest, value)
    
    def update(
        self, 
        operation: Operation,
        result: OperationResult,
        timestamp: datetime
    ):
        """
        根据操作结果更新状态
        
        Args:
            operation: 执行的操作
            result: 操作结果
            timestamp: 时间戳
        """
        # 保存历史
        self._save_history()
        
        # 根据操作类型更新状态
        if operation.name == "takeoff":
            self.robot.airborne = True
            self.robot.on_ground = False
            self.robot.armed = True
            if "altitude" in operation.parameters:
                self.robot.position["alt"] = operation.parameters["altitude"]
                self.robot.position["z"] = operation.parameters["altitude"]
        
        elif operation.name == "land":
            self.robot.airborne = False
            self.robot.on_ground = True
            self.robot.armed = False
            self.robot.position["alt"] = 0
            self.robot.position["z"] = 0
        
        elif operation.name == "goto":
            if "position" in operation.parameters:
                pos = operation.parameters["position"]
                if "lat" in pos:
                    self.robot.position["lat"] = pos.get("lat", self.robot.position["lat"])
                    self.robot.position["lon"] = pos.get("lon", self.robot.position["lon"])
                if "x" in pos:
                    self.robot.position["x"] = pos.get("x", self.robot.position["x"])
                    self.robot.position["y"] = pos.get("y", self.robot.position["y"])
                if "alt" in pos:
                    self.robot.position["alt"] = pos.get("alt", self.robot.position["alt"])
                    self.robot.position["z"] = pos.get("alt", self.robot.position["z"])
        
        elif operation.name == "hover":
            self.robot.moving = False
        
        elif operation.name == "start" or operation.name == "start_engine":
            self.robot.ready = True
        
        elif operation.name == "stop" or operation.name == "stop_engine":
            self.robot.moving = False
            self.robot.ready = False
        
        # 更新任务状态
        self.mission["current_operation"] = operation.name
        
        # 从结果中提取数据
        if result.data:
            for key, value in result.data.items():
                self.set(f"custom.{key}", value)
        
        self.last_update = timestamp
    
    def _save_history(self):
        """保存状态历史"""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "robot": self.robot.to_dict(),
            "mission": dict(self.mission)
        }
        
        self.history.append(snapshot)
        
        # 限制历史长度
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def evaluate_condition(self, condition: str) -> bool:
        """
        评估条件表达式
        
        Args:
            condition: 条件表达式 (如 "robot.battery > 20")
            
        Returns:
            bool: 条件是否满足
        """
        try:
            # 简单的条件解析
            # 支持的格式: "path.to.value operator value"
            # operator: ==, !=, >, <, >=, <=
            
            # 替换路径引用
            def replace_path(match):
                path = match.group(0)
                value = self.get(path)
                if value is None:
                    return "None"
                elif isinstance(value, str):
                    return f"'{value}'"
                elif isinstance(value, bool):
                    return str(value)
                else:
                    return str(value)
            
            # 匹配形如 xxx.yyy.zzz 的路径
            pattern = r'[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+' 
            evaluated = re.sub(pattern, replace_path, condition)
            
            # 安全评估
            # 只允许基本的比较操作
            allowed_names = {
                "True": True,
                "False": False,
                "None": None
            }
            
            result = eval(evaluated, {"__builtins__": {}}, allowed_names)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"条件评估失败: {condition} - {e}")
            return False
    
    def restore(self, state_dict: Dict[str, Any]):
        """
        从字典恢复状态
        
        Args:
            state_dict: 状态字典
        """
        if "robot" in state_dict:
            robot_data = state_dict["robot"]
            for key, value in robot_data.items():
                if hasattr(self.robot, key):
                    setattr(self.robot, key, value)
        
        if "system" in state_dict:
            system_data = state_dict["system"]
            for key, value in system_data.items():
                if hasattr(self.system, key):
                    setattr(self.system, key, value)
        
        if "environment" in state_dict:
            self.environment.update(state_dict["environment"])
        
        if "mission" in state_dict:
            self.mission.update(state_dict["mission"])
        
        if "custom" in state_dict:
            self.custom.update(state_dict["custom"])
        
        self.last_update = datetime.now()
        logger.info("世界状态已恢复")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "robot": self.robot.to_dict(),
            "system": self.system.to_dict(),
            "environment": self.environment,
            "mission": self.mission,
            "custom": self.custom,
            "last_update": self.last_update.isoformat()
        }
    
    def summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "robot_status": self.robot.status,
            "battery": self.robot.battery,
            "position": self.robot.position,
            "airborne": self.robot.airborne,
            "mission_status": self.mission.get("status"),
            "gps_status": self.system.gps.get("status"),
            "last_update": self.last_update.isoformat()
        }
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        return deepcopy(self.to_dict())

