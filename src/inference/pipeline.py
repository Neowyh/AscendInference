#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线推理模块

提供流水线推理功能，支持：
- 预处理+推理+后处理并行执行
- 批处理推理
- 流量控制
"""

import time
import threading
import queue
import numpy as np
from PIL.Image import Image as PILImage
from typing import Optional, Union, List, Dict, Any, Callable

try:
    import acl
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.validators import validate_positive_integer, validate_file_path
from .base import Inference

logger = LoggerConfig.setup_logger('ascend_inference.pipeline', format_type='text')


class PipelineInference:
    """流水线推理管理器（预处理+推理+后处理并行执行）"""

    def __init__(self, config: Optional[Config] = None, batch_size: int = 4, queue_size: int = 10):
        """初始化流水线推理

        Args:
            config: 配置实例
            batch_size: 批处理大小
            queue_size: 队列最大长度，用于流量控制
        """
        self.config = config or Config()
        self.batch_size = batch_size
        self.queue_size = queue_size

        self.preprocess_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.infer_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.postprocess_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        self.running = False
        self.preprocess_threads: List[threading.Thread] = []
        self.infer_threads: List[threading.Thread] = []
        self.postprocess_thread: Optional[threading.Thread] = None

        self.infer_instances: List[Inference] = []

    def start(self, num_preprocess_threads: int = 2, num_infer_threads: int = 1) -> bool:
        """启动流水线

        Args:
            num_preprocess_threads: 预处理线程数
            num_infer_threads: 推理线程数

        Returns:
            bool: 是否启动成功
        """
        self.running = True

        for i in range(num_infer_threads):
            device_id = i % Config.MAX_AI_CORES
            config = Config(
                model_path=self.config.model_path,
                device_id=device_id,
                resolution=self.config.resolution
            )
            infer = Inference(config, batch_size=self.batch_size)
            try:
                infer.init()
                self.infer_instances.append(infer)
            except Exception as e:
                logger.error(f"推理实例{i}初始化失败: {e}")
                self.stop()
                return False

        for i in range(num_preprocess_threads):
            t = threading.Thread(target=self._preprocess_worker, args=(i,))
            t.daemon = True
            t.start()
            self.preprocess_threads.append(t)
            logger.info(f"预处理线程{i}已启动")

        for i in range(num_infer_threads):
            t = threading.Thread(target=self._infer_worker, args=(i,))
            t.daemon = True
            t.start()
            self.infer_threads.append(t)
            logger.info(f"推理线程{i}已启动")

        self.postprocess_thread = threading.Thread(target=self._postprocess_worker)
        self.postprocess_thread.daemon = True
        self.postprocess_thread.start()
        logger.info("后处理线程已启动")

        logger.info(f"流水线启动成功：预处理线程={num_preprocess_threads}, 推理线程={num_infer_threads}, 批大小={self.batch_size}")
        return True

    def _preprocess_worker(self, worker_id: int) -> None:
        """预处理工作线程"""
        while self.running:
            try:
                task = self.preprocess_queue.get(timeout=0.1)
                if task is None:
                    break

                batch_id, image_list, callback = task
                logger.debug(f"预处理线程{worker_id}处理批次{batch_id}，共{len(image_list)}张图像")

                for i in range(0, len(image_list), self.batch_size):
                    batch = image_list[i:i+self.batch_size]
                    while len(batch) < self.batch_size:
                        batch.append(image_list[0])

                    self.infer_queue.put((batch_id, i // self.batch_size, batch, callback))

                self.preprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"预处理线程{worker_id}异常: {e}")
                time.sleep(0.1)

    def _infer_worker(self, worker_id: int) -> None:
        """推理工作线程"""
        infer = self.infer_instances[worker_id]
        if HAS_ACL and infer.context:
            acl.rt.set_context(infer.context)

        while self.running:
            try:
                task = self.infer_queue.get(timeout=0.1)
                if task is None:
                    break

                batch_id, sub_batch_id, image_batch, callback = task
                logger.debug(f"推理线程{worker_id}处理批次{batch_id}-{sub_batch_id}")

                results = infer.run_inference_batch(image_batch, self.config.backend)
                if results:
                    self.postprocess_queue.put((batch_id, sub_batch_id, results, callback))

                self.infer_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"推理线程{worker_id}异常: {e}")
                time.sleep(0.1)

    def _postprocess_worker(self) -> None:
        """后处理工作线程"""
        batch_results: Dict[int, Dict[int, List[np.ndarray]]] = {}

        while self.running:
            try:
                task = self.postprocess_queue.get(timeout=0.1)
                if task is None:
                    break

                batch_id, sub_batch_id, results, callback = task
                logger.debug(f"后处理批次{batch_id}-{sub_batch_id}")

                if batch_id not in batch_results:
                    batch_results[batch_id] = {}
                batch_results[batch_id][sub_batch_id] = results

                if callback:
                    callback(batch_id, sub_batch_id, results)

                self.postprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"后处理线程异常: {e}")
                time.sleep(0.1)

    def submit(self, image_list: List[Union[str, np.ndarray, PILImage]], callback: Optional[Callable] = None) -> int:
        """提交推理任务

        Args:
            image_list: 图像列表
            callback: 结果回调函数，参数为 (batch_id, sub_batch_id, results)

        Returns:
            int: 批次ID
        """
        validate_positive_integer(len(image_list), "image_list length", min_val=1)
        for image_data in image_list:
            if isinstance(image_data, str):
                validate_file_path(image_data, must_exist=True)

        batch_id = int(time.time() * 1000) % 1000000
        self.preprocess_queue.put((batch_id, image_list, callback))
        return batch_id

    def wait_for_completion(self) -> None:
        """等待所有任务完成"""
        self.preprocess_queue.join()
        self.infer_queue.join()
        self.postprocess_queue.join()

    def stop(self) -> None:
        """停止流水线"""
        self.running = False

        for _ in range(len(self.preprocess_threads)):
            self.preprocess_queue.put(None)
        for _ in range(len(self.infer_threads)):
            self.infer_queue.put(None)
        self.postprocess_queue.put(None)

        for t in self.preprocess_threads:
            t.join(timeout=3)
        for t in self.infer_threads:
            t.join(timeout=3)
        if self.postprocess_thread:
            self.postprocess_thread.join(timeout=3)

        for infer in self.infer_instances:
            infer.destroy()

        logger.info("流水线已停止")

    def __del__(self) -> None:
        """析构函数，检测资源泄漏"""
        if self.running:
            logger.warning(
                f"资源泄漏检测：PipelineInference实例未正确调用stop()方法"
            )
            try:
                self.stop()
            except Exception as e:
                logger.error(f"自动停止流水线失败: {e}")
