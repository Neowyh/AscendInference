#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程策略组件

封装多线程推理功能，支持：
- 工作窃取负载均衡
- 动态算力调整
"""

from typing import Dict, Any, Optional
from .base import Strategy, InferenceContext, BaseStrategyConfig
from config.strategy_config import MultithreadStrategyConfig


class MultithreadStrategy(Strategy):
    """多线程策略组件
    
    封装 MultithreadInference 功能，支持工作窃取和动态调整
    """
    
    name = "multithread"
    
    def __init__(self, config: Optional[MultithreadStrategyConfig] = None):
        """初始化多线程策略
        
        Args:
            config: 多线程策略配置
        """
        super().__init__(config or MultithreadStrategyConfig())
        self._multithread_instance = None
        self._baseline_fps: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultithreadStrategy':
        """从字典创建策略实例
        
        Args:
            data: 配置字典
            
        Returns:
            MultithreadStrategy: 策略实例
        """
        config = MultithreadStrategyConfig.from_dict(data)
        return cls(config)
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        """应用多线程策略
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        if not self.enabled:
            return context
        
        from src.inference import MultithreadInference
        from config import Config
        
        inference_config = context.config or Config()
        inference_config.num_threads = self.config.num_threads
        
        self._multithread_instance = MultithreadInference(inference_config)
        
        context.set_state('multithread_instance', self._multithread_instance)
        context.set_state('num_threads', self.config.num_threads)
        context.set_metadata('strategy_type', 'multithread')
        
        return context
    
    def start(self) -> bool:
        """启动多线程推理
        
        Returns:
            bool: 是否成功
        """
        if self._multithread_instance:
            return self._multithread_instance.start()
        return False
    
    def stop(self) -> None:
        """停止多线程推理"""
        if self._multithread_instance:
            self._multithread_instance.stop()
            self._multithread_instance = None
    
    def add_task(self, image_data: Any, backend: str = 'pil') -> None:
        """添加推理任务
        
        Args:
            image_data: 图像数据
            backend: 图像处理后端
        """
        if self._multithread_instance:
            self._multithread_instance.add_task(image_data, backend)
    
    def get_results(self) -> list:
        """获取推理结果
        
        Returns:
            list: 结果列表
        """
        if self._multithread_instance:
            return self._multithread_instance.get_results()
        return []
    
    def wait_completion(self) -> None:
        """等待所有任务完成"""
        if self._multithread_instance:
            self._multithread_instance.wait_completion()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略指标
        
        Returns:
            Dict: 策略指标
        """
        return {
            'num_threads': self.config.num_threads,
            'work_stealing': self.config.work_stealing,
            'dynamic_scaling': self.config.dynamic_scaling,
            'baseline_fps': self._baseline_fps,
            'instance_active': self._multithread_instance is not None
        }
    
    def set_baseline_fps(self, fps: float) -> None:
        """设置基准FPS
        
        Args:
            fps: 基准FPS
        """
        self._baseline_fps = fps
    
    def get_theoretical_speedup(self) -> float:
        """获取理论加速比
        
        Returns:
            float: 理论加速比
        """
        return float(self.config.num_threads)
