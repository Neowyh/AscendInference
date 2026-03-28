#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行预处理器模块

提供并行化的图像预处理功能，支持：
- 多线程并行预处理
- 批量图像处理优化
- 内存高效处理
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Any, Callable, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from PIL.Image import Image as PILImage

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from utils.logger import LoggerConfig, get_logger
from utils.exceptions import PreprocessError

logger = LoggerConfig.setup_logger('ascend_inference.parallel_preprocess', format_type='text')


@dataclass
class PreprocessResult:
    """预处理结果"""
    index: int
    data: Optional[np.ndarray]
    success: bool
    error: Optional[str] = None
    latency: float = 0.0


class ParallelPreprocessor:
    """并行预处理器
    
    使用线程池并行处理图像预处理任务。
    """
    
    def __init__(
        self,
        input_width: int,
        input_height: int,
        num_workers: int = 4
    ):
        """初始化并行预处理器
        
        Args:
            input_width: 输入宽度
            input_height: 输入高度
            num_workers: 工作线程数
        """
        self.input_width = input_width
        self.input_height = input_height
        self.num_workers = num_workers
        
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._stats = {
            'total_processed': 0,
            'total_errors': 0,
            'total_latency': 0.0
        }
    
    def start(self) -> None:
        """启动预处理器"""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
                logger.info(f"并行预处理器已启动，工作线程数: {self.num_workers}")
    
    def stop(self) -> None:
        """停止预处理器"""
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
                logger.info("并行预处理器已停止")
    
    def _preprocess_single(
        self,
        image_data: Union[str, np.ndarray, PILImage],
        backend: str,
        index: int
    ) -> PreprocessResult:
        """预处理单张图像
        
        Args:
            image_data: 图像数据
            backend: 图像处理后端
            index: 图像索引
            
        Returns:
            PreprocessResult: 预处理结果
        """
        start_time = time.time()
        
        try:
            if isinstance(image_data, str):
                if backend == 'opencv' and HAS_OPENCV:
                    image = cv2.imread(image_data)
                    if image is None:
                        return PreprocessResult(
                            index=index,
                            data=None,
                            success=False,
                            error=f"无法读取图像: {image_data}"
                        )
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    from PIL import Image
                    image = np.array(Image.open(image_data).convert('RGB'))
            elif isinstance(image_data, np.ndarray):
                image = image_data
            else:
                image = np.array(image_data)
            
            if backend == 'opencv' and HAS_OPENCV:
                image = cv2.resize(image, (self.input_width, self.input_height))
            else:
                from PIL import Image
                pil_image = Image.fromarray(image)
                image = np.array(pil_image.resize((self.input_width, self.input_height)))
            
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1)).flatten()
            
            latency = time.time() - start_time
            
            return PreprocessResult(
                index=index,
                data=image,
                success=True,
                latency=latency
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return PreprocessResult(
                index=index,
                data=None,
                success=False,
                error=str(e),
                latency=latency
            )
    
    def process_batch(
        self,
        image_list: List[Union[str, np.ndarray, PILImage]],
        backend: str = 'opencv'
    ) -> Tuple[List[np.ndarray], List[int]]:
        """并行批量预处理图像
        
        Args:
            image_list: 图像数据列表
            backend: 图像处理后端
            
        Returns:
            Tuple[List[np.ndarray], List[int]]: (成功处理的结果列表, 失败的索引列表)
        """
        if self._executor is None:
            self.start()
        
        start_time = time.time()
        
        futures = []
        for i, image_data in enumerate(image_list):
            future = self._executor.submit(
                self._preprocess_single,
                image_data,
                backend,
                i
            )
            futures.append(future)
        
        results: List[Optional[np.ndarray]] = [None] * len(image_list)
        failed_indices: List[int] = []
        
        for future in as_completed(futures):
            result: PreprocessResult = future.result()
            
            if result.success:
                results[result.index] = result.data
            else:
                failed_indices.append(result.index)
                logger.warning(f"图像 {result.index} 预处理失败: {result.error}")
        
        successful_results = [r for r in results if r is not None]
        
        total_latency = time.time() - start_time
        with self._lock:
            self._stats['total_processed'] += len(successful_results)
            self._stats['total_errors'] += len(failed_indices)
            self._stats['total_latency'] += total_latency
        
        logger.debug(
            f"批量预处理完成: {len(successful_results)}/{len(image_list)} 成功, "
            f"耗时: {total_latency*1000:.2f}ms"
        )
        
        return successful_results, failed_indices
    
    def process_batch_with_callback(
        self,
        image_list: List[Union[str, np.ndarray, PILImage]],
        callback: Callable[[int, np.ndarray], None],
        backend: str = 'opencv'
    ) -> Tuple[int, List[int]]:
        """并行批量预处理并回调
        
        Args:
            image_list: 图像数据列表
            callback: 回调函数，参数为 (index, result)
            backend: 图像处理后端
            
        Returns:
            Tuple[int, List[int]]: (成功数量, 失败索引列表)
        """
        if self._executor is None:
            self.start()
        
        failed_indices: List[int] = []
        success_count = 0
        
        futures = {}
        for i, image_data in enumerate(image_list):
            future = self._executor.submit(
                self._preprocess_single,
                image_data,
                backend,
                i
            )
            futures[future] = i
        
        for future in as_completed(futures):
            result: PreprocessResult = future.result()
            
            if result.success and result.data is not None:
                callback(result.index, result.data)
                success_count += 1
            else:
                failed_indices.append(result.index)
        
        return success_count, failed_indices
    
    def process_generator(
        self,
        image_generator,
        batch_size: int = 8,
        backend: str = 'opencv'
    ):
        """生成器模式的批量预处理
        
        Args:
            image_generator: 图像生成器
            batch_size: 批大小
            backend: 图像处理后端
            
        Yields:
            Tuple[int, np.ndarray]: (索引, 预处理结果)
        """
        batch = []
        batch_indices = []
        index = 0
        
        for image_data in image_generator:
            batch.append(image_data)
            batch_indices.append(index)
            index += 1
            
            if len(batch) >= batch_size:
                results, failed = self.process_batch(batch, backend)
                
                for i, result in enumerate(results):
                    original_index = batch_indices[i] if i < len(batch_indices) else -1
                    if original_index >= 0:
                        yield original_index, result
                
                batch = []
                batch_indices = []
        
        if batch:
            results, failed = self.process_batch(batch, backend)
            for i, result in enumerate(results):
                original_index = batch_indices[i] if i < len(batch_indices) else -1
                if original_index >= 0:
                    yield original_index, result
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            stats = self._stats.copy()
        
        stats.update({
            'num_workers': self.num_workers,
            'input_size': (self.input_width, self.input_height),
            'running': self._executor is not None
        })
        
        if stats['total_processed'] > 0:
            stats['avg_latency_per_image'] = (
                stats['total_latency'] / stats['total_processed']
            )
        else:
            stats['avg_latency_per_image'] = 0
        
        return stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._stats = {
                'total_processed': 0,
                'total_errors': 0,
                'total_latency': 0.0
            }
    
    def __enter__(self) -> 'ParallelPreprocessor':
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()


class PreprocessPipeline:
    """预处理流水线
    
    支持多阶段流水线处理。
    """
    
    def __init__(
        self,
        stages: List[Callable[[np.ndarray], np.ndarray]],
        num_workers: int = 4
    ):
        """初始化预处理流水线
        
        Args:
            stages: 处理阶段列表，每个阶段是一个函数
            num_workers: 工作线程数
        """
        self.stages = stages
        self.num_workers = num_workers
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def start(self) -> None:
        """启动流水线"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
    
    def stop(self) -> None:
        """停止流水线"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """处理单张图像
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 处理后的图像
        """
        for stage in self.stages:
            image = stage(image)
        return image
    
    def process_batch(
        self,
        images: List[np.ndarray]
    ) -> List[np.ndarray]:
        """批量处理图像
        
        Args:
            images: 输入图像列表
            
        Returns:
            List[np.ndarray]: 处理后的图像列表
        """
        if self._executor is None:
            self.start()
        
        futures = [
            self._executor.submit(self.process, img)
            for img in images
        ]
        
        return [f.result() for f in futures]
    
    def __enter__(self) -> 'PreprocessPipeline':
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
