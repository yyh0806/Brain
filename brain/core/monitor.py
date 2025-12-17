"""
系统监控 - System Monitor

负责:
- 系统健康监控
- 安全检查
- 资源监控
- 告警管理
"""

import asyncio
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger

if TYPE_CHECKING:
    from brain.core.brain import Brain

from brain.execution.operations.base import Operation


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警"""
    id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    passed: bool
    reason: Optional[str] = None
    checks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """系统健康状态"""
    healthy: bool
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    battery_level: float = 100.0
    communication_status: str = "connected"
    gps_status: str = "available"
    sensor_status: Dict[str, str] = field(default_factory=dict)
    last_heartbeat: Optional[datetime] = None
    issues: List[str] = field(default_factory=list)


class SystemMonitor:
    """
    系统监控器
    
    监控系统状态、执行安全检查、管理告警
    """
    
    def __init__(
        self, 
        brain: 'Brain',
        config: Optional[Dict[str, Any]] = None
    ):
        self.brain = brain
        self.config = config or {}
        
        # 告警列表
        self.alerts: List[Alert] = []
        self.max_alerts = config.get("max_alerts", 100)
        
        # 健康状态
        self.health = SystemHealth(healthy=True)
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 安全阈值
        self.thresholds = {
            "battery_warning": config.get("battery_warning", 20),
            "battery_critical": config.get("battery_critical", 10),
            "cpu_warning": 80,
            "cpu_critical": 95,
            "memory_warning": 80,
            "memory_critical": 95,
            "heartbeat_timeout": config.get("heartbeat_timeout", 5.0)
        }
        
        logger.info("SystemMonitor 初始化完成")
    
    async def start(self):
        """启动监控"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("系统监控已启动")
    
    async def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("系统监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        interval = self.config.get("heartbeat_interval", 1.0)
        
        while self._running:
            try:
                # 更新健康状态
                await self._update_health()
                
                # 检查告警条件
                await self._check_alerts()
                
                # 记录心跳
                self.health.last_heartbeat = datetime.now()
                
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
            
            await asyncio.sleep(interval)
    
    async def _update_health(self):
        """更新健康状态"""
        issues = []
        
        # 获取机器人状态
        try:
            robot_status = await self.brain.robot_interface.get_status()
            
            self.health.battery_level = robot_status.get("battery", 100.0)
            self.health.gps_status = robot_status.get("gps_status", "unknown")
            self.health.communication_status = "connected"
            self.health.sensor_status = robot_status.get("sensors", {})
            
            # 检查电池
            if self.health.battery_level < self.thresholds["battery_critical"]:
                issues.append(f"电池电量严重不足: {self.health.battery_level}%")
            elif self.health.battery_level < self.thresholds["battery_warning"]:
                issues.append(f"电池电量偏低: {self.health.battery_level}%")
            
            # 检查GPS
            if self.health.gps_status != "available":
                issues.append(f"GPS状态异常: {self.health.gps_status}")
            
            # 检查传感器
            for sensor, status in self.health.sensor_status.items():
                if status != "ok":
                    issues.append(f"传感器 {sensor} 状态异常: {status}")
                    
        except Exception as e:
            self.health.communication_status = "disconnected"
            issues.append(f"通信异常: {e}")
        
        self.health.issues = issues
        self.health.healthy = len(issues) == 0
    
    async def _check_alerts(self):
        """检查告警条件"""
        # 电池告警
        if self.health.battery_level < self.thresholds["battery_critical"]:
            await self.add_alert(
                level=AlertLevel.CRITICAL,
                message=f"电池电量严重不足: {self.health.battery_level}%",
                source="battery_monitor"
            )
        elif self.health.battery_level < self.thresholds["battery_warning"]:
            await self.add_alert(
                level=AlertLevel.WARNING,
                message=f"电池电量偏低: {self.health.battery_level}%",
                source="battery_monitor"
            )
        
        # 通信告警
        if self.health.communication_status != "connected":
            await self.add_alert(
                level=AlertLevel.ERROR,
                message="与机器人通信断开",
                source="comm_monitor"
            )
        
        # GPS告警
        if self.health.gps_status != "available":
            await self.add_alert(
                level=AlertLevel.WARNING,
                message=f"GPS信号异常: {self.health.gps_status}",
                source="gps_monitor"
            )
    
    async def add_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加告警"""
        # 检查是否有重复的未确认告警
        for alert in self.alerts:
            if (alert.message == message and 
                alert.source == source and 
                not alert.acknowledged):
                return  # 跳过重复告警
        
        alert = Alert(
            id=f"alert_{len(self.alerts)}_{datetime.now().timestamp()}",
            level=level,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # 限制告警数量
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        logger.log(
            level.value.upper(), 
            f"[{source}] {message}"
        )
        
        # 严重告警触发紧急处理
        if level == AlertLevel.CRITICAL:
            await self._handle_critical_alert(alert)
    
    async def _handle_critical_alert(self, alert: Alert):
        """处理严重告警"""
        logger.critical(f"严重告警: {alert.message}")
        
        # 根据告警类型采取行动
        if "电池" in alert.message:
            # 触发返航
            logger.warning("电池严重不足，触发自动返航")
            # await self.brain.emergency_return_to_home()
        
        elif "通信" in alert.message:
            # 进入安全模式
            logger.warning("通信断开，进入悬停等待模式")
    
    async def safety_check(self, operation: Operation) -> SafetyCheckResult:
        """
        执行安全检查
        
        Args:
            operation: 要检查的操作
            
        Returns:
            SafetyCheckResult: 检查结果
        """
        checks = {}
        
        # 1. 电池检查
        checks["battery"] = self.health.battery_level > self.thresholds["battery_critical"]
        
        # 2. 通信检查
        checks["communication"] = self.health.communication_status == "connected"
        
        # 3. GPS检查 (移动操作需要)
        if operation.type.value == "movement":
            checks["gps"] = self.health.gps_status == "available"
        else:
            checks["gps"] = True
        
        # 4. 地理围栏检查
        if "position" in operation.parameters:
            checks["geofence"] = await self._check_geofence(
                operation.parameters["position"]
            )
        else:
            checks["geofence"] = True
        
        # 5. 禁飞区检查
        if "position" in operation.parameters:
            checks["no_fly_zone"] = await self._check_no_fly_zones(
                operation.parameters["position"]
            )
        else:
            checks["no_fly_zone"] = True
        
        # 6. 系统健康检查
        checks["system_health"] = self.health.healthy or len(self.health.issues) < 3
        
        # 汇总结果
        all_passed = all(checks.values())
        failed_checks = [k for k, v in checks.items() if not v]
        
        return SafetyCheckResult(
            passed=all_passed,
            reason=f"检查失败: {', '.join(failed_checks)}" if failed_checks else None,
            checks=checks
        )
    
    async def _check_geofence(self, position: Dict[str, float]) -> bool:
        """检查地理围栏"""
        geofence_config = self.brain.config.get("safety.geofence", {})
        
        if not geofence_config.get("enabled", False):
            return True
        
        # 获取围栏中心和半径
        center = geofence_config.get("center", {"lat": 0, "lon": 0})
        radius = geofence_config.get("default_radius", 500)
        
        # 计算距离 (简化计算)
        lat_diff = abs(position.get("lat", 0) - center.get("lat", 0))
        lon_diff = abs(position.get("lon", 0) - center.get("lon", 0))
        
        # 简化的距离估算 (实际应使用大圆距离)
        distance = ((lat_diff * 111000) ** 2 + (lon_diff * 111000) ** 2) ** 0.5
        
        return distance <= radius
    
    async def _check_no_fly_zones(self, position: Dict[str, float]) -> bool:
        """检查禁飞区"""
        no_fly_zones = self.brain.config.get("safety.no_fly_zones", [])
        
        for zone in no_fly_zones:
            zone_center = zone.get("center", {})
            zone_radius = zone.get("radius", 0)
            
            lat_diff = abs(position.get("lat", 0) - zone_center.get("lat", 0))
            lon_diff = abs(position.get("lon", 0) - zone_center.get("lon", 0))
            
            distance = ((lat_diff * 111000) ** 2 + (lon_diff * 111000) ** 2) ** 0.5
            
            if distance <= zone_radius:
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return {
            "health": {
                "healthy": self.health.healthy,
                "battery": self.health.battery_level,
                "communication": self.health.communication_status,
                "gps": self.health.gps_status,
                "issues": self.health.issues
            },
            "alerts": {
                "total": len(self.alerts),
                "unacknowledged": sum(1 for a in self.alerts if not a.acknowledged),
                "critical": sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL),
                "recent": [
                    {
                        "level": a.level.value,
                        "message": a.message,
                        "time": a.timestamp.isoformat()
                    }
                    for a in self.alerts[-5:]
                ]
            },
            "last_heartbeat": (
                self.health.last_heartbeat.isoformat()
                if self.health.last_heartbeat else None
            )
        }
    
    def acknowledge_alert(self, alert_id: str):
        """确认告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"告警已确认: {alert_id}")
                return
    
    def clear_acknowledged_alerts(self):
        """清除已确认的告警"""
        self.alerts = [a for a in self.alerts if not a.acknowledged]
        logger.info("已清除确认的告警")

