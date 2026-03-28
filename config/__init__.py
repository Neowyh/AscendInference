#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块

提供配置管理功能，包括：
- 基础配置 (Config)
- 策略配置 (StrategyConfig)
- 评测配置 (BenchmarkConfig)
"""

from config.config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES
from config.strategy_config import (
    StrategyConfig,
    BenchmarkConfig,
    ModelInfoConfig,
    MultithreadStrategyConfig,
    BatchStrategyConfig,
    PipelineStrategyConfig,
    MemoryPoolStrategyConfig,
    HighResStrategyConfig,
    AsyncIOStrategyConfig,
    CacheStrategyConfig
)

__all__ = [
    'Config',
    'SUPPORTED_RESOLUTIONS',
    'MAX_AI_CORES',
    'StrategyConfig',
    'BenchmarkConfig',
    'ModelInfoConfig',
    'MultithreadStrategyConfig',
    'BatchStrategyConfig',
    'PipelineStrategyConfig',
    'MemoryPoolStrategyConfig',
    'HighResStrategyConfig',
    'AsyncIOStrategyConfig',
    'CacheStrategyConfig'
]
