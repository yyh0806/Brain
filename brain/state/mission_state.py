"""
任务状态 - Mission State

负责:
- 任务生命周期管理
- 任务状态跟踪
- 进度计算
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class MissionStatus(Enum):
    """任务状态"""
    PENDING = "pending"         # 待执行
    PLANNED = "planned"         # 已规划
    EXECUTING = "executing"     # 执行中
    PAUSED = "paused"           # 暂停
    COMPLETED = "completed"     # 完成
    FAILED = "failed"           # 失败
    CANCELLED = "cancelled"     # 取消
    ROLLBACK = "rollback"       # 回滚中


class MissionPhase(Enum):
    """任务阶段"""
    INITIALIZATION = "initialization"   # 初始化
    PREPARATION = "preparation"         # 准备
    EXECUTION = "execution"             # 执行
    MONITORING = "monitoring"           # 监控
    COMPLETION = "completion"           # 完成
    CLEANUP = "cleanup"                 # 清理


@dataclass
class OperationProgress:
    """操作进度"""
    operation_id: str
    operation_name: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    error: Optional[str] = None


@dataclass
class MissionProgress:
    """任务进度"""
    total_operations: int
    completed_operations: int
    current_operation_index: int
    current_operation: Optional[str] = None
    percentage: float = 0.0
    estimated_remaining_time: float = 0.0
    operations_progress: List[OperationProgress] = field(default_factory=list)


class MissionState:
    """
    任务状态管理
    
    跟踪任务的执行状态和进度
    """
    
    def __init__(self):
        # 当前任务
        self.current_mission_id: Optional[str] = None
        self.current_status: MissionStatus = MissionStatus.PENDING
        self.current_phase: MissionPhase = MissionPhase.INITIALIZATION
        
        # 任务详情
        self.missions: Dict[str, Dict[str, Any]] = {}
        
        # 任务历史
        self.history: List[Dict[str, Any]] = []
        
        logger.info("MissionState 初始化完成")
    
    def create_mission(
        self,
        mission_id: str,
        command: str,
        platform: str,
        operations_count: int = 0
    ):
        """
        创建新任务
        
        Args:
            mission_id: 任务ID
            command: 原始指令
            platform: 平台类型
            operations_count: 操作数量
        """
        self.missions[mission_id] = {
            "id": mission_id,
            "command": command,
            "platform": platform,
            "status": MissionStatus.PENDING.value,
            "phase": MissionPhase.INITIALIZATION.value,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "progress": MissionProgress(
                total_operations=operations_count,
                completed_operations=0,
                current_operation_index=0,
                percentage=0.0,
                estimated_remaining_time=0.0
            ),
            "operations": [],
            "errors": [],
            "metadata": {}
        }
        
        logger.info(f"任务 [{mission_id}] 创建")
    
    def start_mission(self, mission_id: str):
        """开始任务"""
        if mission_id not in self.missions:
            return
        
        self.current_mission_id = mission_id
        self.missions[mission_id]["status"] = MissionStatus.EXECUTING.value
        self.missions[mission_id]["phase"] = MissionPhase.EXECUTION.value
        self.missions[mission_id]["started_at"] = datetime.now().isoformat()
        
        self.current_status = MissionStatus.EXECUTING
        self.current_phase = MissionPhase.EXECUTION
        
        logger.info(f"任务 [{mission_id}] 开始执行")
    
    def update_operation_progress(
        self,
        mission_id: str,
        operation_index: int,
        operation_name: str,
        status: str,
        progress: float = 0.0,
        error: Optional[str] = None
    ):
        """更新操作进度"""
        if mission_id not in self.missions:
            return
        
        mission = self.missions[mission_id]
        mission_progress = mission["progress"]
        
        # 更新当前操作
        mission_progress.current_operation_index = operation_index
        mission_progress.current_operation = operation_name
        
        # 计算总体进度
        if mission_progress.total_operations > 0:
            base_progress = operation_index / mission_progress.total_operations
            op_progress = progress / mission_progress.total_operations
            mission_progress.percentage = (base_progress + op_progress) * 100
        
        # 更新操作记录
        op_progress_record = OperationProgress(
            operation_id=f"op_{operation_index}",
            operation_name=operation_name,
            status=status,
            start_time=datetime.now() if status == "executing" else None,
            end_time=datetime.now() if status in ["success", "failed"] else None,
            progress=progress,
            error=error
        )
        
        if operation_index < len(mission_progress.operations_progress):
            mission_progress.operations_progress[operation_index] = op_progress_record
        else:
            mission_progress.operations_progress.append(op_progress_record)
        
        # 更新完成计数
        if status == "success":
            mission_progress.completed_operations = operation_index + 1
    
    def complete_mission(self, mission_id: str, success: bool = True):
        """完成任务"""
        if mission_id not in self.missions:
            return
        
        mission = self.missions[mission_id]
        mission["status"] = (
            MissionStatus.COMPLETED.value if success 
            else MissionStatus.FAILED.value
        )
        mission["phase"] = MissionPhase.COMPLETION.value
        mission["completed_at"] = datetime.now().isoformat()
        mission["progress"].percentage = 100.0 if success else mission["progress"].percentage
        
        self.current_status = MissionStatus.COMPLETED if success else MissionStatus.FAILED
        self.current_phase = MissionPhase.COMPLETION
        
        # 添加到历史
        self._add_to_history(mission_id)
        
        logger.info(f"任务 [{mission_id}] {'完成' if success else '失败'}")
    
    def pause_mission(self, mission_id: str):
        """暂停任务"""
        if mission_id not in self.missions:
            return
        
        self.missions[mission_id]["status"] = MissionStatus.PAUSED.value
        self.current_status = MissionStatus.PAUSED
        
        logger.info(f"任务 [{mission_id}] 已暂停")
    
    def resume_mission(self, mission_id: str):
        """恢复任务"""
        if mission_id not in self.missions:
            return
        
        self.missions[mission_id]["status"] = MissionStatus.EXECUTING.value
        self.current_status = MissionStatus.EXECUTING
        
        logger.info(f"任务 [{mission_id}] 已恢复")
    
    def cancel_mission(self, mission_id: str):
        """取消任务"""
        if mission_id not in self.missions:
            return
        
        self.missions[mission_id]["status"] = MissionStatus.CANCELLED.value
        self.missions[mission_id]["completed_at"] = datetime.now().isoformat()
        
        self.current_status = MissionStatus.CANCELLED
        
        self._add_to_history(mission_id)
        
        logger.info(f"任务 [{mission_id}] 已取消")
    
    def add_error(self, mission_id: str, error: str):
        """添加错误记录"""
        if mission_id not in self.missions:
            return
        
        self.missions[mission_id]["errors"].append({
            "message": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_mission_status(self, mission_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if mission_id not in self.missions:
            return None
        
        mission = self.missions[mission_id]
        progress = mission["progress"]
        
        return {
            "id": mission_id,
            "status": mission["status"],
            "phase": mission["phase"],
            "progress": progress.percentage,
            "current_operation": progress.current_operation,
            "completed_operations": f"{progress.completed_operations}/{progress.total_operations}",
            "errors_count": len(mission["errors"]),
            "started_at": mission["started_at"],
            "completed_at": mission["completed_at"]
        }
    
    def get_current_mission_status(self) -> Optional[Dict[str, Any]]:
        """获取当前任务状态"""
        if self.current_mission_id:
            return self.get_mission_status(self.current_mission_id)
        return None
    
    def get_progress(self, mission_id: str) -> Optional[MissionProgress]:
        """获取任务进度"""
        if mission_id in self.missions:
            return self.missions[mission_id]["progress"]
        return None
    
    def _add_to_history(self, mission_id: str):
        """添加到历史"""
        if mission_id in self.missions:
            self.history.append({
                "id": mission_id,
                "status": self.missions[mission_id]["status"],
                "started_at": self.missions[mission_id]["started_at"],
                "completed_at": self.missions[mission_id]["completed_at"],
                "progress": self.missions[mission_id]["progress"].percentage
            })
            
            # 限制历史长度
            if len(self.history) > 100:
                self.history = self.history[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = {}
        for mission in self.missions.values():
            status = mission["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_missions": len(self.missions),
            "current_mission": self.current_mission_id,
            "current_status": self.current_status.value,
            "by_status": status_counts,
            "history_count": len(self.history)
        }

