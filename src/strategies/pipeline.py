#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流水线策略组件

支持预处理、推理、后处理的流水线并行
"""

from typing import Dict, Any, Optional, List, Callable
import time
import threading
import queue
from .base import Strategy, InferenceContext
from config.strategy_config import PipelineStrategyConfig


class PipelineStrategy(Strategy):
    """流水线策略组件
    
    支持预处理、推理、后处理的流水线并行执行
    """
    
    name = "pipeline"
    
    def __init__(self, config: Optional[PipelineStrategyConfig] = None):
        """初始化流水线策略
        
        Args:
            config: 流水线策略配置
        """
        super().__init__(config or PipelineStrategyConfig())
        self._pipeline_instance = None
        self._running = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStrategy':
        """从字典创建策略实例
        
        Args:
            data: 配置字典
            
        Returns:
            PipelineStrategy: 策略实例
        """
        config = PipelineStrategyConfig.from_dict(data)
        return cls(config)
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        """应用流水线策略
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        if not self.enabled:
            return context
        
        from src.inference import PipelineInference
        from config import Config
        
        inference_config = context.config or Config()
        
        self._pipeline_instance = PipelineInference(
            config=inference_config,
            batch_size=self.config.batch_size,
            queue_size=self.config.queue_size
        )
        
        context.set_state('pipeline_instance', self._pipeline_instance)
        context.set_state('batch_size', self.config.batch_size)
        context.set_state('queue_size', self.config.queue_size)
        context.set_metadata('strategy_type', 'pipeline')
        
        return context
    
    def start(self) -> bool:
        """启动流水线
        
        Returns:
            bool: 是否成功
        """
        if not self._pipeline_instance:
            return False
        
        result = self._pipeline_instance.start(
            num_preprocess_threads=self.config.num_preprocess_threads,
            num_infer_threads=self.config.num_infer_threads,
            num_postprocess_threads=self.config.num_postprocess_threads,
        )
        
        if result:
            self._running = True
        
        return result
    
    def stop(self) -> None:
        """停止流水线"""
        if self._pipeline_instance:
            self._pipeline_instance.stop()
            self._pipeline_instance = None
        self._running = False
    
    def submit(self, image_list: List[Any], callback: Optional[Callable] = None) -> int:
        """提交推理任务
        
        Args:
            image_list: 图像列表
            callback: 回调函数
            
        Returns:
            int: 批次ID
        """
        if self._pipeline_instance:
            return self._pipeline_instance.submit(image_list, callback)
        return -1
    
    def wait_for_completion(self) -> None:
        """等待所有任务完成"""
        if self._pipeline_instance:
            self._pipeline_instance.wait_for_completion()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略指标
        
        Returns:
            Dict: 策略指标
        """
        return {
            'queue_size': self.config.queue_size,
            'batch_size': self.config.batch_size,
            'num_preprocess_threads': self.config.num_preprocess_threads,
            'num_infer_threads': self.config.num_infer_threads,
            'num_postprocess_threads': self.config.num_postprocess_threads,
            'running': self._running
        }
    
    def get_theoretical_speedup(self) -> float:
        """获取理论加速比
        
        Returns:
            float: 理论加速比
        """
        total_threads = (
            self.config.num_preprocess_threads + 
            self.config.num_infer_threads + 
            self.config.num_postprocess_threads
        )
        return float(max(1, total_threads))
