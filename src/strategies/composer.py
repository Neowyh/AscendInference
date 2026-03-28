#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组合器

支持灵活组合多种策略，提供：
- 策略添加/移除
- 策略链式调用
- 配置导入/导出
"""

from typing import List, Dict, Any, Optional, Type
from .base import Strategy, InferenceContext


class StrategyComposer:
    """策略组合器
    
    支持灵活组合多种策略，按顺序应用
    
    Example:
        composer = StrategyComposer()
        composer.add_strategy(MultithreadStrategy(threads=4))
        composer.add_strategy(BatchStrategy(batch_size=8))
        
        context = InferenceContext(inference_instance)
        context = composer.apply_all(context)
        
        metrics = composer.get_all_metrics()
    """
    
    _strategy_registry: Dict[str, Type[Strategy]] = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[Strategy]) -> None:
        """注册策略类
        
        Args:
            name: 策略名称
            strategy_class: 策略类
        """
        cls._strategy_registry[name] = strategy_class
    
    @classmethod
    def get_registered_strategies(cls) -> List[str]:
        """获取已注册的策略列表
        
        Returns:
            list: 策略名称列表
        """
        return list(cls._strategy_registry.keys())
    
    def __init__(self):
        """初始化策略组合器"""
        self.strategies: List[Strategy] = []
        self.execution_order: List[str] = []
        self._context: Optional[InferenceContext] = None
    
    def add_strategy(self, strategy: Strategy) -> 'StrategyComposer':
        """添加策略
        
        Args:
            strategy: 策略实例
            
        Returns:
            StrategyComposer: 支持链式调用
        """
        self.strategies.append(strategy)
        self.execution_order.append(strategy.name)
        return self
    
    def remove_strategy(self, name: str) -> 'StrategyComposer':
        """移除策略
        
        Args:
            name: 策略名称
            
        Returns:
            StrategyComposer: 支持链式调用
        """
        self.strategies = [s for s in self.strategies if s.name != name]
        self.execution_order = [n for n in self.execution_order if n != name]
        return self
    
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """获取策略实例
        
        Args:
            name: 策略名称
            
        Returns:
            Strategy: 策略实例，未找到返回None
        """
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None
    
    def has_strategy(self, name: str) -> bool:
        """检查是否包含指定策略
        
        Args:
            name: 策略名称
            
        Returns:
            bool: 是否包含
        """
        return any(s.name == name for s in self.strategies)
    
    def enable_strategy(self, name: str) -> bool:
        """启用指定策略
        
        Args:
            name: 策略名称
            
        Returns:
            bool: 是否成功
        """
        strategy = self.get_strategy(name)
        if strategy:
            strategy.enable()
            return True
        return False
    
    def disable_strategy(self, name: str) -> bool:
        """禁用指定策略
        
        Args:
            name: 策略名称
            
        Returns:
            bool: 是否成功
        """
        strategy = self.get_strategy(name)
        if strategy:
            strategy.disable()
            return True
        return False
    
    def apply_all(self, context: InferenceContext) -> InferenceContext:
        """按顺序应用所有启用的策略
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        self._context = context
        
        for strategy in self.strategies:
            if strategy.enabled:
                context = strategy.apply(context)
                context.update_metrics({strategy.name: strategy.get_metrics()})
        
        return context
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """收集所有策略的统计指标
        
        Returns:
            Dict: 所有策略的指标
        """
        metrics = {}
        for strategy in self.strategies:
            if strategy.enabled:
                metrics[strategy.name] = strategy.get_metrics()
        return metrics
    
    def get_config(self) -> Dict[str, Any]:
        """导出当前配置
        
        Returns:
            Dict: 配置字典
        """
        return {
            'strategies': [
                strategy.get_info()
                for strategy in self.strategies
            ],
            'execution_order': self.execution_order.copy()
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StrategyComposer':
        """从配置创建策略组合器
        
        Args:
            config: 配置字典
            
        Returns:
            StrategyComposer: 策略组合器实例
        """
        composer = cls()
        
        strategies_config = config.get('strategies', [])
        
        for strategy_config in strategies_config:
            name = strategy_config.get('name')
            if not name:
                continue
            
            strategy_class = cls._strategy_registry.get(name)
            if strategy_class:
                strategy = strategy_class.from_dict(strategy_config.get('config', {}))
                if not strategy_config.get('enabled', True):
                    strategy.disable()
                composer.add_strategy(strategy)
        
        return composer
    
    def clear(self) -> None:
        """清空所有策略"""
        self.strategies.clear()
        self.execution_order.clear()
        self._context = None
    
    def get_enabled_strategies(self) -> List[str]:
        """获取已启用的策略列表
        
        Returns:
            list: 已启用的策略名称列表
        """
        return [s.name for s in self.strategies if s.enabled]
    
    def get_disabled_strategies(self) -> List[str]:
        """获取已禁用的策略列表
        
        Returns:
            list: 已禁用的策略名称列表
        """
        return [s.name for s in self.strategies if not s.enabled]
    
    def __len__(self) -> int:
        """返回策略数量"""
        return len(self.strategies)
    
    def __iter__(self):
        """迭代策略"""
        return iter(self.strategies)
    
    def __repr__(self) -> str:
        """字符串表示"""
        strategies_str = ', '.join(f'{s.name}({"enabled" if s.enabled else "disabled"})' 
                                   for s in self.strategies)
        return f"StrategyComposer([{strategies_str}])"


def register_builtin_strategies() -> None:
    """注册内置策略
    
    延迟导入以避免循环依赖
    """
    try:
        from .multithread import MultithreadStrategy
        StrategyComposer.register_strategy('multithread', MultithreadStrategy)
    except ImportError:
        pass
    
    try:
        from .batch import BatchStrategy
        StrategyComposer.register_strategy('batch', BatchStrategy)
    except ImportError:
        pass
    
    try:
        from .pipeline import PipelineStrategy
        StrategyComposer.register_strategy('pipeline', PipelineStrategy)
    except ImportError:
        pass
    
    try:
        from .memory_pool import MemoryPoolStrategy
        StrategyComposer.register_strategy('memory_pool', MemoryPoolStrategy)
    except ImportError:
        pass
    
    try:
        from .high_res import HighResStrategy
        StrategyComposer.register_strategy('high_res', HighResStrategy)
    except ImportError:
        pass
