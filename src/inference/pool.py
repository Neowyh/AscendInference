#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理池模块

提供线程池模式的推理功能，支持：
- 推理实例复用
- 线程池管理
- 批量推理优化
"""

import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Any, Dict, Callable
from dataclasses import dataclass

try:
    import acl
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.exceptions import ThreadError, InferenceError
from .base import Inference

logger = LoggerConfig.setup_logger('ascend_inference.pool', format_type='text')


@dataclass
class InferenceTask:
    """推理任务"""
    task_id: int
    image_data: Any
    backend: str = 'opencv'
    callback: Optional[Callable] = None


class InferencePool:
    """推理实例池
    
    管理多个推理实例，支持线程池模式的并行推理。
    复用推理实例以减少初始化开销。
    """
    
    def __init__(self, config: Config, pool_size: int = 4):
        """初始化推理池
        
        Args:
            config: 配置实例
            pool_size: 池大小（推理实例数量）
        """
        self.config = config
        self.pool_size = pool_size
        
        self._instances: List[Inference] = []
        self._instance_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._initialized = False
        self._shutdown = False
        
        self._executor: Optional[ThreadPoolExecutor] = None
        self._task_counter = 0
        self._task_lock = threading.Lock()
    
    def init(self) -> None:
        """初始化推理实例池
        
        Raises:
            ThreadError: 初始化失败
        """
        if self._initialized:
            return
        
        logger.info(f"初始化推理池，池大小: {self.pool_size}")
        
        for i in range(self.pool_size):
            device_id = i % Config.MAX_AI_CORES
            config = Config(
                model_path=self.config.model_path,
                device_id=device_id,
                resolution=self.config.resolution
            )
            
            try:
                instance = Inference(config)
                instance.init()
                self._instances.append(instance)
                self._instance_queue.put(instance)
                logger.info(f"推理实例 {i} 初始化成功 (device: {device_id})")
            except Exception as e:
                logger.error(f"推理实例 {i} 初始化失败: {e}")
        
        if not self._instances:
            raise ThreadError(
                "推理池初始化失败：无法创建任何推理实例",
                error_code=4001,
                details={"pool_size": self.pool_size}
            )
        
        self._executor = ThreadPoolExecutor(max_workers=self.pool_size)
        self._initialized = True
        logger.info(f"推理池初始化完成，成功创建 {len(self._instances)} 个实例")
    
    def _get_instance(self, timeout: float = 30.0) -> Optional[Inference]:
        """获取一个推理实例
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            Inference: 推理实例，超时返回 None
        """
        try:
            return self._instance_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("获取推理实例超时")
            return None
    
    def _return_instance(self, instance: Inference) -> None:
        """归还推理实例
        
        Args:
            instance: 推理实例
        """
        self._instance_queue.put(instance)
    
    def _generate_task_id(self) -> int:
        """生成任务ID
        
        Returns:
            int: 任务ID
        """
        with self._task_lock:
            self._task_counter += 1
            return self._task_counter
    
    def infer(self, image_data: Any, backend: str = 'opencv') -> Any:
        """执行单次推理
        
        Args:
            image_data: 图像数据
            backend: 图像处理后端
            
        Returns:
            Any: 推理结果
            
        Raises:
            ThreadError: 推理失败
        """
        if not self._initialized:
            raise ThreadError("推理池未初始化", error_code=4002)
        
        if self._shutdown:
            raise ThreadError("推理池已关闭", error_code=4003)
        
        instance = self._get_instance()
        if instance is None:
            raise ThreadError("无法获取推理实例", error_code=4004)
        
        try:
            return instance.run_inference(image_data, backend)
        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            raise ThreadError(
                "推理执行失败",
                error_code=4005,
                original_error=e
            ) from e
        finally:
            self._return_instance(instance)
    
    def infer_batch(
        self,
        image_list: List[Any],
        backend: str = 'opencv',
        callback: Optional[Callable[[int, Any], None]] = None
    ) -> List[Any]:
        """批量推理
        
        Args:
            image_list: 图像数据列表
            backend: 图像处理后端
            callback: 回调函数，参数为 (index, result)
            
        Returns:
            List[Any]: 推理结果列表
        """
        if not self._initialized:
            raise ThreadError("推理池未初始化", error_code=4002)
        
        if self._shutdown:
            raise ThreadError("推理池已关闭", error_code=4003)
        
        results = [None] * len(image_list)
        futures: List[Future] = []
        
        def _infer_one(index: int, image_data: Any) -> None:
            result = self.infer(image_data, backend)
            results[index] = result
            if callback:
                callback(index, result)
        
        for i, image_data in enumerate(image_list):
            future = self._executor.submit(_infer_one, i, image_data)
            futures.append(future)
        
        for future in futures:
            future.result()
        
        return results
    
    def submit(
        self,
        image_data: Any,
        backend: str = 'opencv',
        callback: Optional[Callable[[int, Any], None]] = None
    ) -> Future:
        """提交异步推理任务
        
        Args:
            image_data: 图像数据
            backend: 图像处理后端
            callback: 回调函数
            
        Returns:
            Future: 异步任务对象
        """
        if not self._initialized:
            raise ThreadError("推理池未初始化", error_code=4002)
        
        if self._shutdown:
            raise ThreadError("推理池已关闭", error_code=4003)
        
        task_id = self._generate_task_id()
        
        def _execute():
            result = self.infer(image_data, backend)
            if callback:
                callback(task_id, result)
            return result
        
        return self._executor.submit(_execute)
    
    def map(
        self,
        image_list: List[Any],
        backend: str = 'opencv',
        timeout: Optional[float] = None
    ) -> List[Any]:
        """映射推理（类似 map 函数）
        
        Args:
            image_list: 图像数据列表
            backend: 图像处理后端
            timeout: 超时时间
            
        Returns:
            List[Any]: 推理结果列表
        """
        if not self._initialized:
            raise ThreadError("推理池未初始化", error_code=4002)
        
        futures = [
            self._executor.submit(self.infer, img, backend)
            for img in image_list
        ]
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                results.append(None)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理池统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'pool_size': self.pool_size,
            'active_instances': len(self._instances),
            'available_instances': self._instance_queue.qsize(),
            'initialized': self._initialized,
            'shutdown': self._shutdown,
            'total_tasks': self._task_counter
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """关闭推理池
        
        Args:
            wait: 是否等待所有任务完成
        """
        if self._shutdown:
            return
        
        logger.info("关闭推理池...")
        self._shutdown = True
        
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
        
        for instance in self._instances:
            try:
                instance.destroy()
            except Exception as e:
                logger.warning(f"销毁推理实例失败: {e}")
        
        self._instances.clear()
        self._initialized = False
        logger.info("推理池已关闭")
    
    def __enter__(self) -> 'InferencePool':
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
    
    def __del__(self) -> None:
        if self._initialized and not self._shutdown:
            logger.warning("推理池未正确关闭，正在自动清理...")
            try:
                self.shutdown(wait=False)
            except Exception as e:
                logger.error(f"自动关闭推理池失败: {e}")
