# -*- coding: utf-8 -*-
"""
Python路径配置模块
用于统一管理多分支并行开发时的模块导入路径
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class PathConfig:
    """路径配置管理器"""

    def __init__(self, brain_root: Optional[str] = None):
        """初始化路径配置

        Args:
            brain_root: Brain项目根目录路径，如果为None则自动检测
        """
        if brain_root is None:
            self.brain_root = self._detect_brain_root()
        else:
            self.brain_root = Path(brain_root).resolve()

        self.current_worktree = self._detect_current_worktree()
        self._configure_paths()

    def _detect_brain_root(self) -> Path:
        """自动检测Brain项目根目录"""
        current = Path(__file__).resolve()

        # 向上查找包含setup.py或pyproject.toml的目录
        while current.parent != current:
            if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
                return current
            if current.name == "Brain" and (current / "brain").exists():
                return current
            current = current.parent

        raise RuntimeError("无法检测到Brain项目根目录")

    def _detect_current_worktree(self) -> Optional[str]:
        """检测当前所在的git worktree"""
        git_dir = self.brain_root / ".git"

        if git_dir.is_file():
            # 这是git worktree，读取实际git目录路径
            with open(git_dir, 'r') as f:
                content = f.read().strip()
                if content.startswith("gitdir: "):
                    git_dir = Path(content[8:]).parent

        # 读取HEAD文件获取当前分支
        head_file = git_dir / "HEAD"
        if head_file.exists():
            with open(head_file, 'r') as f:
                content = f.read().strip()
                if content.startswith("ref: refs/heads/"):
                    return content[16:]

        return None

    def _configure_paths(self):
        """配置Python路径"""
        # 确保主Brain目录在Python路径中
        brain_path = str(self.brain_root)
        if brain_path not in sys.path:
            sys.path.insert(0, brain_path)
            logger.info(f"添加主Brain目录到Python路径: {brain_path}")

        # 添加brain包路径
        brain_package_path = str(self.brain_root / "brain")
        if brain_package_path not in sys.path:
            sys.path.insert(0, brain_package_path)
            logger.info(f"添加brain包目录到Python路径: {brain_package_path}")

    def get_worktree_path(self, worktree_name: str) -> Optional[Path]:
        """获取指定worktree的路径"""
        worktree_path = self.brain_root.parent / f"Brain-{worktree_name}"
        if worktree_path.exists():
            return worktree_path
        return None

    def add_worktree_to_path(self, worktree_name: str):
        """添加指定worktree到Python路径"""
        worktree_path = self.get_worktree_path(worktree_name)
        if worktree_path:
            worktree_str = str(worktree_path)
            if worktree_str not in sys.path:
                sys.path.insert(0, worktree_str)
                logger.info(f"添加worktree到Python路径: {worktree_str}")

    def configure_all_worktrees(self):
        """配置所有worktree到Python路径（按优先级）"""
        # 优先级：当前worktree > master > 其他分支
        worktrees = [
            "docs-reports-dev",
            "fusion-engine-dev",
            "integration-dev",
            "preprocessing-dev",
            "sensor-input-dev",
            "situational-map-dev",
            "testing-framework-dev"
        ]

        # 添加其他worktree（但优先级较低）
        for worktree in worktrees:
            if worktree != self.current_worktree:
                self.add_worktree_to_path(worktree)

    def validate_import_structure(self) -> List[str]:
        """验证导入结构，返回发现的问题列表"""
        issues = []

        # 检查关键模块是否可以导入
        try:
            from brain.core.brain import Brain
        except ImportError as e:
            issues.append(f"无法导入Brain类: {e}")

        try:
            from brain.planning.task.task_planner import TaskPlanner
        except ImportError as e:
            issues.append(f"无法导入TaskPlanner: {e}")

        try:
            from brain.execution.executor import Executor
        except ImportError as e:
            issues.append(f"无法导入Executor: {e}")

        try:
            from brain.core.monitor import SystemMonitor
        except ImportError as e:
            issues.append(f"无法导入SystemMonitor: {e}")

        return issues

    def print_path_status(self):
        """打印当前路径状态"""
        print(f"Brain项目根目录: {self.brain_root}")
        print(f"当前worktree: {self.current_worktree or 'master'}")
        print("\nPython路径 (前10个):")
        for i, path in enumerate(sys.path[:10]):
            marker = " (当前)" if path == str(self.brain_root) else ""
            print(f"  {i+1}. {path}{marker}")


# 全局路径配置实例
_path_config: Optional[PathConfig] = None


def configure_brain_paths(brain_root: Optional[str] = None) -> PathConfig:
    """配置Brain项目的Python路径

    Args:
        brain_root: Brain项目根目录路径，如果为None则自动检测

    Returns:
        PathConfig实例
    """
    global _path_config

    if _path_config is None:
        _path_config = PathConfig(brain_root)

    return _path_config


def get_path_config() -> Optional[PathConfig]:
    """获取全局路径配置实例"""
    return _path_config


# 模块导入时自动配置路径
try:
    configure_brain_paths()
except Exception as e:
    logger.warning(f"自动配置Brain路径时出错: {e}")