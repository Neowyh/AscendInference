#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组件模块

提供可组合的优化策略组件，支持：
- 策略基类和上下文
- 策略组合器
- 多线程策略
- 批处理策略
- 流水线策略
- 内存池策略
- 高分辨率策略
"""

from .base import Strategy, InferenceContext, BaseStrategyConfig, NoOpStrategy
from .base_unit import StrategyUnit, ValidationResult
from .composer import StrategyComposer, register_builtin_strategies
from .composition import StrategyCompositionEngine

__all__ = [
    'Strategy',
    'InferenceContext',
    'BaseStrategyConfig',
    'NoOpStrategy',
    'StrategyUnit',
    'ValidationResult',
    'StrategyComposer',
    'StrategyCompositionEngine',
    'register_builtin_strategies'
]

register_builtin_strategies()
