#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM异步分析服务

独立运行的VLM分析服务，避免阻塞主感知循环。

核心特性：
- 异步分析：使用线程池处理VLM请求
- 请求队列：VLM分析请求排队处理
- 结果缓存：相同图像的分析结果缓存，避免重复计算
- 超时控制：VLM分析超时机制
"""

import asyncio
import hashlib
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import numpy as np
import hashlib

from brain.perception.vlm.vlm_perception import VLMPerception, SceneDescription, DetectedObject


@dataclass
class VLMRequest:
    """VLM分析请求"""
    id: str
    rgb_image: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 优先级，高优先级先处理


@dataclass
class VLMResult:
    """VLM分析结果"""
    request_id: str
    scene_description: Optional[SceneDescription] = None
    processing_time: float = 0.0  # 处理时间（秒）
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VLMCacheEntry:
    """VLM缓存条目"""
    image_hash: str
    result: VLMResult
    hit_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)


class VLMService:
    """
    VLM异步分析服务
    
    管理VLM分析的异步处理，包括：
    - 请求队列
    - 工作线程池
    - 结果缓存
    - 超时控制
    """
    
    def __init__(
        self,
        vlm: VLMPerception,
        max_workers: int = 1,
        cache_size: int = 10,
        timeout: float = 30.0,
        enable_cache: bool = True
    ):
        """
        Args:
            vlm: VLMPerception实例
            max_workers: 最大工作线程数
            cache_size: 缓存大小
            timeout: 超时时间（秒）
            enable_cache: 是否启用缓存
        """
        self.vlm = vlm
        self.max_workers = max_workers
        self.timeout = timeout
        self.enable_cache = enable_cache
        
        # 请求队列
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._request_id_counter = 0
        
        # 结果缓存
        self._cache: Dict[str, VLMCacheEntry] = {}
        self._cache_lock = asyncio.Lock()
        self._max_cache_size = cache_size
        
        # 工作线程池
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # 统计
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"VLMService 初始化: workers={max_workers}, cache_size={cache_size}, timeout={timeout}s")
    
    def start(self) -> None:
        """启动VLM服务"""
        if self._running:
            logger.warning("VLMService已经在运行")
            return
        
        self._running = True
        
        # 创建工作线程
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        
        logger.info(f"VLMService已启动: {len(self._workers)}个工作线程")
    
    async def stop(self) -> None:
        """停止VLM服务"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消所有工作线程
        for worker in self._workers:
            worker.cancel()
        
        # 等待所有工作线程结束
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # 清空队列
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self._workers.clear()
        logger.info("VLMService已停止")
    
    async def _worker(self, worker_id: int) -> None:
        """
        工作线程：处理VLM分析请求
        
        Args:
            worker_id: 工作线程ID
        """
        logger.debug(f"VLM工作线程#{worker_id}已启动")
        
        while self._running:
            try:
                # 获取请求（带超时）
                request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0
                )
                
                if request is None:
                    # 哨兵值，退出
                    break
                
                start_time = datetime.now()
                logger.debug(f"Worker#{worker_id}处理请求#{request.id}")
                
                # 检查缓存
                if self.enable_cache:
                    cache_key = self._compute_image_hash(request.rgb_image)
                    cached = await self._get_cache(cache_key)
                    
                    if cached is not None:
                        self._cache_hits += 1
                        result = cached.result
                        logger.debug(f"缓存命中: {cache_key}")
                    else:
                        self._cache_misses += 1
                        # 执行VLM分析
                        result = await self._analyze_scene(request)
                else:
                    # 不使用缓存，直接分析
                    result = await self._analyze_scene(request)
                
                # 处理时间
                processing_time = (datetime.now() - start_time).total_seconds()
                result.processing_time = processing_time
                
                # 更新缓存（如果启用）
                if self.enable_cache and result.scene_description is not None:
                    cache_key = self._compute_image_hash(request.rgb_image)
                    await self._set_cache(cache_key, result)
                
                logger.debug(f"Worker#{worker_id}完成请求#{request.id}: {processing_time:.2f}s")
                
            except asyncio.TimeoutError:
                logger.warning(f"Worker#{worker_id}获取请求超时")
            except Exception as e:
                logger.error(f"Worker#{worker_id}处理请求失败: {e}")
    
    async def _analyze_scene(self, request: VLMRequest) -> VLMResult:
        """
        执行VLM场景分析
        
        Args:
            request: VLM请求
        
        Returns:
            VLMResult
        """
        try:
            # 调用VLM
            scene_description = await asyncio.wait_for(
                self.vlm.describe_scene(request.rgb_image),
                timeout=self.timeout
            )
            
            return VLMResult(
                request_id=request.id,
                scene_description=scene_description,
                processing_time=0.0  # 会在worker中更新
            )
        
        except asyncio.TimeoutError:
            logger.error(f"VLM分析超时（{self.timeout}s）: {request.id}")
            return VLMResult(
                request_id=request.id,
                scene_description=None,
                error=f"Timeout after {self.timeout}s"
            )
        except Exception as e:
            logger.error(f"VLM分析失败: {e}")
            return VLMResult(
                request_id=request.id,
                scene_description=None,
                error=str(e)
            )
    
    async def analyze_image(
        self,
        rgb_image: np.ndarray,
        priority: int = 0,
        callback: Optional[Callable] = None
    ) -> Optional[str]:
        """
        异步分析图像
        
        Args:
            rgb_image: RGB图像（numpy数组）
            priority: 优先级（高值优先）
            callback: 完成回调函数（可选）
        
        Returns:
            请求ID（可以用于查询结果）
        """
        if not self._running:
            logger.warning("VLMService未运行，无法处理请求")
            return None
        
        # 创建请求
        self._request_id_counter += 1
        request_id = f"req_{self._request_id_counter}"
        
        request = VLMRequest(
            id=request_id,
            rgb_image=rgb_image,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # 添加到队列
        try:
            self._request_queue.put_nowait(request)
        except asyncio.QueueFull:
            logger.error(f"VLM请求队列已满: {request_id}")
            return None
        
        self._total_requests += 1
        logger.debug(f"VLM请求已入队: {request_id}, priority={priority}, 队列长度={self._request_queue.qsize()}")
        
        # 如果有回调，不等待结果
        if callback:
            # 可以在未来实现异步回调机制
            pass
        
        return request_id
    
    async def get_result(self, request_id: str, timeout: float = 5.0) -> Optional[VLMResult]:
        """
        获取VLM分析结果
        
        Args:
            request_id: 请求ID
            timeout: 超时时间（秒）
        
        Returns:
            VLM结果或None
        """
        # 简单实现：当前结果是通过队列处理的
        # 在实际实现中，可能需要结果队列或回调机制
        logger.warning(f"get_result方法需要实现: {request_id}")
        return None
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """
        计算图像哈希（用于缓存）
        
        Args:
            image: RGB图像
        
        Returns:
            哈希字符串
        """
        # 使用图像的简单哈希（降采样后）
        if image.size > 10000:
            # 降采样到最大100x100
            h, w = image.shape[:2]
            scale = min(1.0, 100.0 / max(h, w))
            small_image = (
                (image[:int(h*scale), :int(w*scale)] * 255).astype(np.uint8)
                if image.dtype != np.uint8 else image
            )
        else:
            small_image = image
        
        # 计算MD5哈希
        md5 = hashlib.md5()
        md5.update(small_image.tobytes())
        return md5.hexdigest()
    
    async def _get_cache(self, cache_key: str) -> Optional[VLMResult]:
        """
        获取缓存结果
        
        Args:
            cache_key: 缓存键
        
        Returns:
            缓存结果或None
        """
        async with self._cache_lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.hit_count += 1
                entry.last_used = datetime.now()
                return entry.result
            return None
    
    async def _set_cache(self, cache_key: str, result: VLMResult) -> None:
        """
        设置缓存结果
        
        Args:
            cache_key: 缓存键
            result: VLM结果
        """
        async with self._cache_lock:
            # 如果缓存已满，移除最旧的条目
            if len(self._cache) >= self._max_cache_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].last_used
                )
                del self._cache[oldest_key]
                logger.debug(f"缓存已满，移除最旧条目: {oldest_key}")
            
            # 添加或更新缓存
            self._cache[cache_key] = VLMCacheEntry(
                image_hash=cache_key,
                result=result,
                hit_count=0,
                last_used=datetime.now()
            )
    
    async def clear_cache(self) -> None:
        """清空缓存"""
        async with self._cache_lock:
            self._cache.clear()
            logger.info("VLM缓存已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取服务统计信息
        
        Returns:
            统计字典
        """
        cache_hit_rate = 0.0
        if self._total_requests > 0:
            cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
        
        return {
            "running": self._running,
            "total_requests": self._total_requests,
            "queue_size": self._request_queue.qsize(),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "worker_count": len(self._workers)
        }

