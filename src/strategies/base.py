#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组件基类

提供策略组件的基础接口和实现
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class BaseStrategyConfig:
    """策略配置基类"""
    enabled: bool = True


class InferenceContext:
    """推理上下文
    
    封装推理实例和相关状态，支持策略链式处理
    """
    
    def __init__(self, inference_instance: Any = None, config: Any = None):
        """初始化推理上下文
        
        Args:
            inference_instance: 推理实例
            config: 配置对象
        """
        self.inference = inference_instance
        self.config = config
        self.state: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
    
    def set_state(self, key: str, value: Any) -> None:
        """设置状态
        
        Args:
            key: 状态键
            value: 状态值
        """
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态
        
        Args:
            key: 状态键
            default: 默认值
            
        Returns:
            状态值
        """
        return self.state.get(key, default)
    
    def set_metric(self, key: str, value: Any) -> None:
        """设置指标
        
        Args:
            key: 指标键
            value: 指标值
        """
        self.metrics[key] = value
    
    def get_metric(self, key: str, default: Any = None) -> Any:
        """获取指标
        
        Args:
            key: 指标键
            default: 默认值
            
        Returns:
            指标值
        """
        return self.metrics.get(key, default)
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """更新指标
        
        Args:
            metrics: 指标字典
        """
        self.metrics.update(metrics)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据
        
        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据
        
        Args:
            key: 元数据键
            default: 默认值
            
        Returns:
            元数据值
        """
        return self.metadata.get(key, default)


class Strategy(ABC):
    """策略组件基类
    
    所有优化策略都应继承此类，实现 apply 和 get_metrics 方法
    """
    
    name: str = "base"
    
    def __init__(self, config: Optional[BaseStrategyConfig] = None):
        """初始化策略
        
        Args:
            config: 策略配置
        """
        self.config = config or BaseStrategyConfig()
        self.enabled = self.config.enabled
        self._metrics: Dict[str, Any] = {}
    
    @abstractmethod
    def apply(self, context: InferenceContext) -> InferenceContext:
        """应用策略到推理上下文
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略相关的统计指标
        
        Returns:
            Dict: 策略指标
        """
        pass
    
    def enable(self) -> None:
        """启用策略"""
        self.enabled = True
    
    def disable(self) -> None:
        """禁用策略"""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """检查策略是否启用
        
        Returns:
            bool: 是否启用
        """
        return self.enabled
    
    def set_metric(self, key: str, value: Any) -> None:
        """设置指标
        
        Args:
            key: 指标键
            value: 指标值
        """
        self._metrics[key] = value
    
    def get_metric(self, key: str, default: Any = None) -> Any:
        """获取指标
        
        Args:
            key: 指标键
            default: 默认值
            
        Returns:
            指标值
        """
        return self._metrics.get(key, default)
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self._metrics.clear()
    
    def get_info(self) -> Dict[str, Any]:
        """获取策略信息
        
        Returns:
            Dict: 策略信息
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        }


class NoOpStrategy(Strategy):
    """空操作策略
    
    用于测试或占位
    """
    
    name = "noop"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'NoOpStrategy':
        """从字典创建策略实例
        
        Args:
            config: 配置字典
            
        Returns:
            NoOpStrategy: 策略实例
        """
        return cls(config=BaseStrategyConfig(**config))
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        """不做任何操作
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 原上下文
        """
        return context
    
    def get_metrics(self) -> Dict[str, Any]:
        """返回空指标
        
        Returns:
            Dict: 空字典
        """
        return {}
