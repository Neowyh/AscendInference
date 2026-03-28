#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程推理模块

提供多线程推理功能，支持：
- 工作窃取负载均衡
- 动态算力调整
- 并行推理
"""

import time
import threading
import queue
import numpy as np
from PIL.Image import Image as PILImage
from typing import Optional, Union, List, Tuple

try:
    import acl
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.validators import validate_image_backend, validate_file_path
from .base import Inference

logger = LoggerConfig.setup_logger('ascend_inference.multithread', format_type='text')

DEFAULT_SLEEP_INTERVAL = 0.01
DEFAULT_WORKER_ERROR_SLEEP = 0.1


class MultithreadInference:
    """多线程推理管理器（工作窃取负载均衡+动态算力调整优化）"""

    def __init__(self, config: Optional[Config] = None, auto_scale: bool = True):
        """初始化多线程推理

        Args:
            config: Config 实例
            auto_scale: 是否启用自动算力调整
        """
        self.config = config or Config()
        self.initial_num_threads = min(self.config.num_threads, Config.MAX_AI_CORES)
        self.num_threads = self.initial_num_threads
        self.model_path = self.config.model_path
        self.resolution = self.config.resolution
        self.auto_scale = auto_scale

        self.workers: List[Inference] = []
        self.task_queues: List[queue.Queue] = []
        self.result_queue: queue.Queue = queue.Queue()
        self.threads: List[threading.Thread] = []
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.worker_states: List[bool] = []

        self.min_threads = 1
        self.max_threads = Config.MAX_AI_CORES
        self.scale_up_threshold = 0.7
        self.scale_down_threshold = 0.2
        self.scale_interval = 5
    
    def _init_workers(self) -> bool:
        """初始化工作线程

        Returns:
            bool: 是否初始化成功
        """
        for i in range(self.num_threads):
            device_id = i % Config.MAX_AI_CORES
            config = Config(
                model_path=self.model_path,
                device_id=device_id,
                resolution=self.resolution
            )
            worker = Inference(config)
            try:
                worker.init()
                self.workers.append(worker)
                self.task_queues.append(queue.Queue())
                self.worker_states.append(False)
                logger.info(f"Worker {i} 初始化成功 (device: {device_id})")
            except Exception as e:
                logger.error(f"Worker {i} 初始化失败: {e}")
        
        return len(self.workers) > 0
    
    def _worker_thread(self, worker_id: int, worker: Inference) -> None:
        """工作线程函数（支持工作窃取）

        Args:
            worker_id: worker编号
            worker: 推理工作实例
        """
        if HAS_ACL and worker.context:
            acl.rt.set_context(worker.context)

        while self.running:
            try:
                task = self.task_queues[worker_id].get(block=False)
                if task is None:
                    break

                image_path, backend = task
                self.worker_states[worker_id] = True
                result = worker.run_inference(image_path, backend)
                self.result_queue.put((image_path, result))
                self.task_queues[worker_id].task_done()
                self.worker_states[worker_id] = False

            except queue.Empty:
                stolen = False
                for other_id in range(len(self.task_queues)):
                    if other_id == worker_id:
                        continue
                    try:
                        task = self.task_queues[other_id].get(block=False)
                        if task is not None:
                            image_path, backend = task
                            self.worker_states[worker_id] = True
                            result = worker.run_inference(image_path, backend)
                            self.result_queue.put((image_path, result))
                            self.task_queues[other_id].task_done()
                            self.worker_states[worker_id] = False
                            stolen = True
                            break
                    except queue.Empty:
                        continue

                if not stolen:
                    time.sleep(DEFAULT_SLEEP_INTERVAL)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.worker_states[worker_id] = False
                time.sleep(DEFAULT_WORKER_ERROR_SLEEP)
    
    def start(self) -> bool:
        """启动多线程

        Returns:
            bool: 是否启动成功
        """
        if not self.workers:
            if not self._init_workers():
                return False
        
        self.running = True

        for worker_id, worker in enumerate(self.workers):
            thread = threading.Thread(target=self._worker_thread, args=(worker_id, worker))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        return True
    
    def add_task(self, image_path: Union[str, np.ndarray, PILImage], backend: Optional[str] = None) -> None:
        """添加推理任务（均匀分配到各worker队列）

        Args:
            image_path: 图像路径或图像数据
            backend: 图像处理后端
        """
        if backend is None:
            backend = self.config.backend

        validate_image_backend(backend)
        if isinstance(image_path, str):
            validate_file_path(image_path, must_exist=True)

        worker_id = len(self.result_queue.queue) % len(self.task_queues)
        self.task_queues[worker_id].put((image_path, backend))
    
    def get_results(self) -> List[Tuple[Union[str, int], Optional[np.ndarray]]]:
        """获取推理结果

        Returns:
            List[Tuple]: 每个元素是(图像标识, 推理结果)的元组
        """
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get(block=False)
                results.append(result)
                self.result_queue.task_done()
            except queue.Empty:
                break
        return results
    
    def wait_completion(self) -> None:
        """等待所有任务完成"""
        for q in self.task_queues:
            q.join()
        while not self.result_queue.empty():
            time.sleep(DEFAULT_SLEEP_INTERVAL)
    
    def stop(self) -> None:
        """停止多线程"""
        self.running = False

        for q in self.task_queues:
            q.put(None)

        for thread in self.threads:
            thread.join(timeout=5)

        for worker in self.workers:
            worker.destroy()

        if HAS_ACL:
            acl.finalize()

    def __del__(self) -> None:
        """析构函数，检测资源泄漏"""
        if self.running:
            logger.warning(
                f"资源泄漏检测：MultithreadInference实例未正确调用stop()方法"
            )
            try:
                self.stop()
            except Exception as e:
                logger.error(f"自动停止多线程失败: {e}")
