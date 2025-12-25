"""
异步处理器 - Async Processor

用于将CPU密集型任务异步化，避免阻塞事件循环
"""

import asyncio
from typing import Callable, Any, Optional, TypeVar, Coroutine
import numpy as np
from loguru import logger

T = TypeVar('T')


async def process_cpu_intensive(
    func: Callable[..., T],
    *args,
    **kwargs
) -> T:
    """
    在线程池中执行CPU密集型函数
    
    Args:
        func: 要执行的函数
        *args, **kwargs: 函数参数
        
    Returns:
        函数返回值
    """
    return await asyncio.to_thread(func, *args, **kwargs)


async def process_image_async(
    image: np.ndarray,
    processor: Callable[[np.ndarray], np.ndarray],
    *args,
    **kwargs
) -> np.ndarray:
    """
    异步处理图像
    
    Args:
        image: 输入图像（numpy数组）
        processor: 图像处理函数
        *args, **kwargs: 处理函数参数
        
    Returns:
        处理后的图像
    """
    def _process():
        return processor(image, *args, **kwargs)
    
    return await process_cpu_intensive(_process)


async def process_batch_async(
    items: list,
    processor: Callable,
    max_workers: Optional[int] = None,
    *args,
    **kwargs
) -> list:
    """
    异步批量处理
    
    Args:
        items: 要处理的项列表
        processor: 处理函数
        max_workers: 最大并发数（None表示无限制）
        *args, **kwargs: 处理函数参数
        
    Returns:
        处理结果列表
    """
    if not items:
        return []
    
    # 创建处理任务
    tasks = []
    for item in items:
        task = process_cpu_intensive(processor, item, *args, **kwargs)
        tasks.append(task)
    
    # 限制并发数
    if max_workers:
        semaphore = asyncio.Semaphore(max_workers)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        tasks = [bounded_task(task) for task in tasks]
    
    # 并发执行
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 过滤异常
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"批量处理项 {i} 失败: {result}")
        else:
            valid_results.append(result)
    
    return valid_results


class AsyncImageProcessor:
    """
    异步图像处理器
    
    提供常用的异步图像处理操作
    """
    
    @staticmethod
    async def resize(
        image: np.ndarray,
        size: tuple,
        interpolation: int = 1
    ) -> np.ndarray:
        """异步调整图像大小"""
        def _resize():
            try:
                import cv2
                return cv2.resize(image, size, interpolation=interpolation)
            except ImportError:
                # 如果没有cv2，使用PIL
                from PIL import Image
                pil_image = Image.fromarray(image)
                resized = pil_image.resize(size)
                return np.array(resized)
        
        return await process_cpu_intensive(_resize)
    
    @staticmethod
    async def normalize(
        image: np.ndarray,
        mean: Optional[tuple] = None,
        std: Optional[tuple] = None
    ) -> np.ndarray:
        """异步归一化图像"""
        def _normalize():
            img = image.astype(np.float32)
            if mean:
                img -= np.array(mean).reshape(1, 1, -1)
            if std:
                img /= np.array(std).reshape(1, 1, -1)
            return img
        
        return await process_cpu_intensive(_normalize)
    
    @staticmethod
    async def convert_color_space(
        image: np.ndarray,
        conversion_code: int
    ) -> np.ndarray:
        """异步转换颜色空间"""
        def _convert():
            try:
                import cv2
                return cv2.cvtColor(image, conversion_code)
            except ImportError:
                logger.warning("OpenCV不可用，跳过颜色空间转换")
                return image
        
        return await process_cpu_intensive(_convert)


class AsyncCoordinateTransformer:
    """
    异步坐标变换器
    
    用于批量坐标变换操作
    """
    
    @staticmethod
    async def transform_points_batch(
        points: np.ndarray,
        transform_matrix: np.ndarray
    ) -> np.ndarray:
        """
        异步批量变换点
        
        Args:
            points: 点数组 (N, 3) 或 (N, 4)
            transform_matrix: 变换矩阵 (4, 4)
            
        Returns:
            变换后的点数组
        """
        def _transform():
            # 转换为齐次坐标
            if points.shape[1] == 3:
                homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
            else:
                homogeneous = points
            
            # 应用变换
            transformed = (transform_matrix @ homogeneous.T).T
            
            # 转换回3D坐标
            if points.shape[1] == 3:
                return transformed[:, :3]
            return transformed
        
        return await process_cpu_intensive(_transform)
    
    @staticmethod
    async def compute_distances_batch(
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """
        异步批量计算距离
        
        Args:
            points1: 第一组点 (N, 3)
            points2: 第二组点 (N, 3)
            
        Returns:
            距离数组 (N,)
        """
        def _compute():
            diff = points1 - points2
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
            return distances
        
        return await process_cpu_intensive(_compute)







