#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应批处理策略模块

提供智能的动态批处理功能，支持：
- 自适应批大小调整
- 优先级队列
- 超时处理
- 性能监控
"""

import time
import heapq
import threading
from typing import List, Any, Optional, Dict, Callable, Tuple
from dataclasses import dataclass, field
from queue import Queue, Empty

from utils.logger import LoggerConfig, get_logger
from utils.exceptions import InferenceError

logger = LoggerConfig.setup_logger('ascend_inference.adaptive_batch', format_type='text')


@dataclass(order=True)
class PriorityItem:
    """优先级队列项"""
    priority: int
    sequence: int
    item: Any = field(compare=False)


class AdaptiveBatchQueue:
    """自适应批处理队列
    
    支持优先级的批处理队列，能够自动调整批大小。
    """
    
    def __init__(
        self,
        batch_size: int = 4,
        max_batch_size: int = 16,
        min_batch_size: int = 1,
        timeout_ms: float = 10.0
    ):
        """初始化自适应批处理队列
        
        Args:
            batch_size: 初始批大小
            max_batch_size: 最大批大小
            min_batch_size: 最小批大小
            timeout_ms: 批处理超时时间（毫秒）
        """
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.timeout_ms = timeout_ms
        
        self._queue: List[PriorityItem] = []
        self._counter = 0
        self._lock = threading.Lock()
        
        self._performance_history: List[float] = []
        self._history_size = 100
        self._adjustment_interval = 10
        self._adjustment_counter = 0
    
    def push(self, item: Any, priority: int = 0) -> None:
        """添加项目到队列
        
        Args:
            item: 项目数据
            priority: 优先级（数值越小优先级越高）
        """
        with self._lock:
            heapq.heappush(
                self._queue,
                PriorityItem(priority, self._counter, item)
            )
            self._counter += 1
    
    def pop_batch(self, batch_size: Optional[int] = None) -> List[Tuple[int, Any]]:
        """弹出一批项目
        
        Args:
            batch_size: 批大小，None 则使用当前批大小
            
        Returns:
            List[Tuple[int, Any]]: 项目列表，每个元素为 (sequence, item)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = []
        timeout = self.timeout_ms / 1000.0
        start_time = time.time()
        
        with self._lock:
            while len(batch) < batch_size:
                if not self._queue:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout or len(batch) > 0:
                        break
                    time.sleep(0.001)
                    continue
                
                priority_item = heapq.heappop(self._queue)
                batch.append((priority_item.sequence, priority_item.item))
        
        return batch
    
    def size(self) -> int:
        """获取队列大小
        
        Returns:
            int: 队列中的项目数量
        """
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """检查队列是否为空
        
        Returns:
            bool: 是否为空
        """
        return self.size() == 0
    
    def record_latency(self, latency: float) -> None:
        """记录延迟以用于自适应调整
        
        Args:
            latency: 延迟时间（秒）
        """
        self._performance_history.append(latency)
        if len(self._performance_history) > self._history_size:
            self._performance_history.pop(0)
        
        self._adjustment_counter += 1
        if self._adjustment_counter >= self._adjustment_interval:
            self._adjust_batch_size()
            self._adjustment_counter = 0
    
    def _adjust_batch_size(self) -> None:
        """根据性能历史调整批大小"""
        if len(self._performance_history) < 10:
            return
        
        avg_latency = sum(self._performance_history[-10:]) / 10
        
        if avg_latency < 0.01:
            new_size = min(self.batch_size + 1, self.max_batch_size)
        elif avg_latency > 0.05:
            new_size = max(self.batch_size - 1, self.min_batch_size)
        else:
            return
        
        if new_size != self.batch_size:
            logger.debug(f"批大小调整: {self.batch_size} -> {new_size} (延迟: {avg_latency*1000:.2f}ms)")
            self.batch_size = new_size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取队列统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'queue_size': self.size(),
            'batch_size': self.batch_size,
            'max_batch_size': self.max_batch_size,
            'min_batch_size': self.min_batch_size,
            'timeout_ms': self.timeout_ms,
            'avg_latency': sum(self._performance_history) / len(self._performance_history)
                if self._performance_history else 0
        }


class AdaptiveBatchProcessor:
    """自适应批处理器
    
    自动管理批处理流程，包括：
    - 动态批大小调整
    - 超时处理
    - 性能监控
    """
    
    def __init__(
        self,
        process_fn: Callable[[List[Any]], List[Any]],
        batch_size: int = 4,
        max_batch_size: int = 16,
        timeout_ms: float = 10.0,
        target_latency_ms: float = 20.0
    ):
        """初始化自适应批处理器
        
        Args:
            process_fn: 批处理函数，接收列表返回列表
            batch_size: 初始批大小
            max_batch_size: 最大批大小
            timeout_ms: 批处理超时时间
            target_latency_ms: 目标延迟
        """
        self.process_fn = process_fn
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.target_latency_ms = target_latency_ms
        
        self._queue = AdaptiveBatchQueue(
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            timeout_ms=timeout_ms
        )
        
        self._results: Dict[int, Any] = {}
        self._results_lock = threading.Lock()
        self._sequence = 0
        self._sequence_lock = threading.Lock()
        
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
    
    def submit(self, item: Any, priority: int = 0) -> int:
        """提交项目进行处理
        
        Args:
            item: 项目数据
            priority: 优先级
            
        Returns:
            int: 序列号，用于获取结果
        """
        with self._sequence_lock:
            seq = self._sequence
            self._sequence += 1
        
        self._queue.push((seq, item), priority)
        return seq
    
    def get_result(self, seq: int, timeout: float = 30.0) -> Optional[Any]:
        """获取处理结果
        
        Args:
            seq: 序列号
            timeout: 超时时间
            
        Returns:
            Any: 处理结果，超时返回 None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._results_lock:
                if seq in self._results:
                    return self._results.pop(seq)
            time.sleep(0.001)
        return None
    
    def start(self) -> None:
        """启动批处理器"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("自适应批处理器已启动")
    
    def stop(self) -> None:
        """停止批处理器"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            self._worker_thread = None
        logger.info("自适应批处理器已停止")
    
    def _worker_loop(self) -> None:
        """工作线程循环"""
        while self._running:
            try:
                batch = self._queue.pop_batch()
                if not batch:
                    continue
                
                start_time = time.time()
                
                items = [(seq, item) for seq, item in batch]
                inputs = [item for _, item in items]
                
                try:
                    results = self.process_fn(inputs)
                    
                    latency = time.time() - start_time
                    self._queue.record_latency(latency)
                    
                    with self._results_lock:
                        for (seq, _), result in zip(items, results):
                            self._results[seq] = result
                    
                except Exception as e:
                    logger.error(f"批处理执行失败: {e}")
                    with self._results_lock:
                        for seq, _ in items:
                            self._results[seq] = None
                
            except Exception as e:
                logger.error(f"批处理工作线程异常: {e}")
                time.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self._queue.get_stats()
        stats.update({
            'running': self._running,
            'pending_results': len(self._results)
        })
        return stats
    
    def __enter__(self) -> 'AdaptiveBatchProcessor':
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
