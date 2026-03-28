#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存池策略组件

支持内存复用，减少内存分配开销
"""

from typing import Dict, Any, Optional
from .base import Strategy, InferenceContext
from config.strategy_config import MemoryPoolStrategyConfig


class MemoryPoolStrategy(Strategy):
    """内存池策略组件
    
    支持内存复用，减少内存分配开销
    """
    
    name = "memory_pool"
    
    def __init__(self, config: Optional[MemoryPoolStrategyConfig] = None):
        """初始化内存池策略
        
        Args:
            config: 内存池策略配置
        """
        super().__init__(config or MemoryPoolStrategyConfig())
        self._memory_pool = None
        self._stats = {
            'allocations': 0,
            'reuses': 0,
            'growth_events': 0
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryPoolStrategy':
        """从字典创建策略实例
        
        Args:
            data: 配置字典
            
        Returns:
            MemoryPoolStrategy: 策略实例
        """
        config = MemoryPoolStrategyConfig.from_dict(data)
        return cls(config)
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        """应用内存池策略
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        if not self.enabled:
            return context
        
        from utils.memory_pool import MemoryPool
        
        inference = context.inference
        if inference and hasattr(inference, 'input_size'):
            buffer_size = inference.input_size
            self._memory_pool = MemoryPool(
                buffer_size=buffer_size,
                device=self.config.device,
                max_buffers=self.config.max_buffers
            )
            
            context.set_state('memory_pool', self._memory_pool)
            context.set_state('pool_size', self.config.pool_size)
        
        context.set_metadata('strategy_type', 'memory_pool')
        
        return context
    
    def allocate(self) -> Optional[Any]:
        """分配内存
        
        Returns:
            内存缓冲区
        """
        if self._memory_pool:
            self._stats['allocations'] += 1
            return self._memory_pool.allocate()
        return None
    
    def free(self, buffer: Any) -> None:
        """释放内存
        
        Args:
            buffer: 内存缓冲区
        """
        if self._memory_pool:
            self._memory_pool.free(buffer)
            self._stats['reuses'] += 1
    
    def cleanup(self) -> None:
        """清理内存池"""
        if self._memory_pool:
            self._memory_pool.cleanup()
            self._memory_pool = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略指标
        
        Returns:
            Dict: 策略指标
        """
        reuse_rate = 0.0
        if self._stats['allocations'] > 0:
            reuse_rate = self._stats['reuses'] / self._stats['allocations'] * 100
        
        return {
            'pool_size': self.config.pool_size,
            'max_buffers': self.config.max_buffers,
            'device': self.config.device,
            'growth_factor': self.config.growth_factor,
            'stats': self._stats.copy(),
            'reuse_rate': reuse_rate,
            'active': self._memory_pool is not None
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            'allocations': 0,
            'reuses': 0,
            'growth_events': 0
        }
