#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理策略组件

支持动态批处理，提高吞吐量
"""

from typing import Dict, Any, Optional, List
import time
import threading
import queue
from .base import Strategy, InferenceContext
from config.strategy_config import BatchStrategyConfig


class BatchStrategy(Strategy):
    """批处理策略组件
    
    支持动态批处理，收集请求后批量执行
    """
    
    name = "batch"
    
    def __init__(self, config: Optional[BatchStrategyConfig] = None):
        """初始化批处理策略
        
        Args:
            config: 批处理策略配置
        """
        super().__init__(config or BatchStrategyConfig())
        self._batch_queue: queue.Queue = queue.Queue()
        self._results: Dict[int, Any] = {}
        self._lock = threading.Lock()
        self._task_id_counter = 0
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._inference_instance = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchStrategy':
        """从字典创建策略实例
        
        Args:
            data: 配置字典
            
        Returns:
            BatchStrategy: 策略实例
        """
        config = BatchStrategyConfig.from_dict(data)
        return cls(config)
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        """应用批处理策略
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        if not self.enabled:
            return context
        
        self._inference_instance = context.inference
        
        context.set_state('batch_strategy', self)
        context.set_state('batch_size', self.config.batch_size)
        context.set_metadata('strategy_type', 'batch')
        
        return context
    
    def start(self) -> bool:
        """启动批处理工作线程
        
        Returns:
            bool: 是否成功
        """
        if self._running:
            return True
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self._worker_thread.start()
        return True
    
    def stop(self) -> None:
        """停止批处理工作线程"""
        self._running = False
        self._batch_queue.put(None)
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None
    
    def submit(self, image_data: Any, backend: str = 'pil') -> int:
        """提交推理任务
        
        Args:
            image_data: 图像数据
            backend: 图像处理后端
            
        Returns:
            int: 任务ID
        """
        with self._lock:
            task_id = self._task_id_counter
            self._task_id_counter += 1
        
        self._batch_queue.put((task_id, image_data, backend))
        return task_id
    
    def get_result(self, task_id: int, timeout: float = 10.0) -> Optional[Any]:
        """获取任务结果
        
        Args:
            task_id: 任务ID
            timeout: 超时时间
            
        Returns:
            推理结果
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if task_id in self._results:
                    return self._results.pop(task_id)
            time.sleep(0.01)
        return None
    
    def _batch_worker(self) -> None:
        """批处理工作线程"""
        batch: List[tuple] = []
        timeout_start = None
        
        while self._running:
            try:
                task = self._batch_queue.get(timeout=0.1)
                
                if task is None:
                    break
                
                batch.append(task)
                timeout_start = time.time()
                
                if len(batch) >= self.config.batch_size:
                    self._process_batch(batch)
                    batch.clear()
                    timeout_start = None
                    
            except queue.Empty:
                if batch and timeout_start:
                    elapsed = (time.time() - timeout_start) * 1000
                    if elapsed >= self.config.timeout_ms:
                        self._process_batch(batch)
                        batch.clear()
                        timeout_start = None
        
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[tuple]) -> None:
        """处理一批任务
        
        Args:
            batch: 任务列表
        """
        if not self._inference_instance:
            return
        
        try:
            images = [task[1] for task in batch]
            
            if hasattr(self._inference_instance, 'run_inference_batch'):
                results = self._inference_instance.run_inference_batch(images)
            else:
                results = [self._inference_instance.run_inference(img) for img in images]
            
            with self._lock:
                for i, task in enumerate(batch):
                    task_id = task[0]
                    self._results[task_id] = results[i] if i < len(results) else None
                    
        except Exception as e:
            with self._lock:
                for task in batch:
                    self._results[task[0]] = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略指标
        
        Returns:
            Dict: 策略指标
        """
        return {
            'batch_size': self.config.batch_size,
            'timeout_ms': self.config.timeout_ms,
            'dynamic_batch': self.config.dynamic_batch,
            'queue_size': self._batch_queue.qsize(),
            'running': self._running
        }
