"""
检查点管理 - Checkpoint Manager

负责:
- 状态快照保存
- 检查点恢复
- 持久化存储
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger


@dataclass
class Checkpoint:
    """检查点"""
    id: str
    mission_id: str
    stage: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "stage": self.stage,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        return cls(
            id=data["id"],
            mission_id=data["mission_id"],
            stage=data["stage"],
            data=data["data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


class CheckpointManager:
    """
    检查点管理器
    
    管理任务执行过程中的状态检查点，支持故障恢复
    """
    
    def __init__(self, storage_path: str = "./data/checkpoints"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 内存缓存
        self.checkpoints: Dict[str, List[Checkpoint]] = {}
        
        # 配置
        self.max_checkpoints_per_mission = 20
        self.auto_persist = True
        
        logger.info(f"CheckpointManager 初始化完成, 存储路径: {self.storage_path}")
    
    async def create_checkpoint(
        self,
        mission_id: str,
        stage: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        创建检查点
        
        Args:
            mission_id: 任务ID
            stage: 阶段标识
            data: 状态数据
            metadata: 元数据
            
        Returns:
            Checkpoint: 创建的检查点
        """
        checkpoint = Checkpoint(
            id=f"cp_{mission_id}_{stage}_{datetime.now().timestamp()}",
            mission_id=mission_id,
            stage=stage,
            data=data,
            metadata=metadata or {}
        )
        
        # 添加到内存
        if mission_id not in self.checkpoints:
            self.checkpoints[mission_id] = []
        
        self.checkpoints[mission_id].append(checkpoint)
        
        # 限制数量
        if len(self.checkpoints[mission_id]) > self.max_checkpoints_per_mission:
            self.checkpoints[mission_id] = self.checkpoints[mission_id][-self.max_checkpoints_per_mission:]
        
        # 持久化
        if self.auto_persist:
            await self._persist_checkpoint(checkpoint)
        
        logger.debug(f"检查点创建: {checkpoint.id}")
        
        return checkpoint
    
    async def get_checkpoint(
        self, 
        checkpoint_id: str
    ) -> Optional[Checkpoint]:
        """获取指定检查点"""
        # 先查内存
        for mission_checkpoints in self.checkpoints.values():
            for cp in mission_checkpoints:
                if cp.id == checkpoint_id:
                    return cp
        
        # 查文件
        return await self._load_checkpoint(checkpoint_id)
    
    async def get_latest_checkpoint(
        self, 
        mission_id: str
    ) -> Optional[Checkpoint]:
        """获取任务的最新检查点"""
        if mission_id in self.checkpoints and self.checkpoints[mission_id]:
            return self.checkpoints[mission_id][-1]
        
        # 从文件加载
        return await self._load_latest_checkpoint(mission_id)
    
    async def get_nearest_checkpoint(
        self,
        mission_id: str,
        target_index: int
    ) -> Optional[Checkpoint]:
        """
        获取最接近目标索引的检查点
        
        Args:
            mission_id: 任务ID
            target_index: 目标操作索引
            
        Returns:
            Checkpoint: 最近的检查点
        """
        if mission_id not in self.checkpoints:
            # 尝试从文件加载
            await self._load_mission_checkpoints(mission_id)
        
        if mission_id not in self.checkpoints:
            return None
        
        # 查找最接近的检查点
        best_checkpoint = None
        best_distance = float('inf')
        
        for cp in self.checkpoints[mission_id]:
            cp_index = cp.data.get("operation_index", 0)
            
            # 只考虑目标之前的检查点
            if cp_index <= target_index:
                distance = target_index - cp_index
                if distance < best_distance:
                    best_distance = distance
                    best_checkpoint = cp
        
        return best_checkpoint
    
    async def get_mission_checkpoints(
        self, 
        mission_id: str
    ) -> List[Checkpoint]:
        """获取任务的所有检查点"""
        if mission_id not in self.checkpoints:
            await self._load_mission_checkpoints(mission_id)
        
        return self.checkpoints.get(mission_id, [])
    
    async def delete_checkpoint(self, checkpoint_id: str):
        """删除检查点"""
        # 从内存删除
        for mission_id, checkpoints in self.checkpoints.items():
            self.checkpoints[mission_id] = [
                cp for cp in checkpoints if cp.id != checkpoint_id
            ]
        
        # 从文件删除
        file_path = self.storage_path / f"{checkpoint_id}.json"
        if file_path.exists():
            file_path.unlink()
        
        logger.debug(f"检查点删除: {checkpoint_id}")
    
    async def delete_mission_checkpoints(self, mission_id: str):
        """删除任务的所有检查点"""
        # 从内存删除
        if mission_id in self.checkpoints:
            del self.checkpoints[mission_id]
        
        # 从文件删除
        mission_dir = self.storage_path / mission_id
        if mission_dir.exists():
            import shutil
            shutil.rmtree(mission_dir)
        
        logger.info(f"任务 [{mission_id}] 所有检查点已删除")
    
    async def save_all(self):
        """保存所有检查点"""
        for mission_id, checkpoints in self.checkpoints.items():
            for cp in checkpoints:
                await self._persist_checkpoint(cp)
        
        logger.info("所有检查点已保存")
    
    async def _persist_checkpoint(self, checkpoint: Checkpoint):
        """持久化检查点"""
        try:
            mission_dir = self.storage_path / checkpoint.mission_id
            mission_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = mission_dir / f"{checkpoint.id}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"检查点持久化失败: {e}")
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """从文件加载检查点"""
        try:
            # 搜索所有任务目录
            for mission_dir in self.storage_path.iterdir():
                if mission_dir.is_dir():
                    file_path = mission_dir / f"{checkpoint_id}.json"
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        return Checkpoint.from_dict(data)
            return None
            
        except Exception as e:
            logger.error(f"检查点加载失败: {e}")
            return None
    
    async def _load_latest_checkpoint(self, mission_id: str) -> Optional[Checkpoint]:
        """加载任务的最新检查点"""
        mission_dir = self.storage_path / mission_id
        
        if not mission_dir.exists():
            return None
        
        try:
            # 获取所有检查点文件
            files = list(mission_dir.glob("*.json"))
            if not files:
                return None
            
            # 按修改时间排序
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            with open(files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Checkpoint.from_dict(data)
            
        except Exception as e:
            logger.error(f"最新检查点加载失败: {e}")
            return None
    
    async def _load_mission_checkpoints(self, mission_id: str):
        """加载任务的所有检查点"""
        mission_dir = self.storage_path / mission_id
        
        if not mission_dir.exists():
            return
        
        try:
            checkpoints = []
            
            for file_path in mission_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    checkpoints.append(Checkpoint.from_dict(data))
                except Exception as e:
                    logger.warning(f"检查点文件加载失败: {file_path} - {e}")
            
            # 按时间排序
            checkpoints.sort(key=lambda cp: cp.created_at)
            
            self.checkpoints[mission_id] = checkpoints
            
        except Exception as e:
            logger.error(f"任务检查点加载失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_checkpoints = sum(len(cps) for cps in self.checkpoints.values())
        
        return {
            "total_missions": len(self.checkpoints),
            "total_checkpoints": total_checkpoints,
            "storage_path": str(self.storage_path),
            "missions": {
                mission_id: len(cps) 
                for mission_id, cps in self.checkpoints.items()
            }
        }

