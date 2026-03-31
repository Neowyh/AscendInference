#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测模块

提供三层评测场景：
1. ModelSelectionScenario - 模型选型评测
2. StrategyValidationScenario - 策略验证评测
3. ExtremePerformanceScenario - 极限性能评测

提供多种报告格式：
- TextReporter - 文本格式
- JsonReporter - JSON格式
- HtmlReporter - HTML格式
"""

from .scenarios import (
    BenchmarkScenario,
    BenchmarkResult,
    ModelInfo,
    ModelSelectionScenario,
    RouteExperimentScenario,
    StrategyValidationScenario,
    ExtremePerformanceScenario
)

from .reporters import (
    Reporter,
    TextReporter,
    JsonReporter,
    HtmlReporter,
    create_reporter,
    save_report
)

__all__ = [
    'BenchmarkScenario',
    'BenchmarkResult',
    'ModelInfo',
    'ModelSelectionScenario',
    'RouteExperimentScenario',
    'StrategyValidationScenario',
    'ExtremePerformanceScenario',
    'Reporter',
    'TextReporter',
    'JsonReporter',
    'HtmlReporter',
    'create_reporter',
    'save_report'
]
