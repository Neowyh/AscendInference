#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略单元基础模型

定义策略单元、执行器类型与组合校验结果，供评测编排层复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class StrategyUnit:
    """可组合的策略单元定义"""

    name: str
    executor_kind: str = "simple"
    supported_routes: Tuple[str, ...] = field(default_factory=tuple)
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    conflicts: Tuple[str, ...] = field(default_factory=tuple)

    def supports_route(self, route_type: str | None) -> bool:
        """检查策略是否支持指定路线"""
        if route_type is None:
            return True
        if not self.supported_routes:
            return True
        return route_type in self.supported_routes


@dataclass(frozen=True)
class ValidationResult:
    """策略组合校验结果"""

    is_valid: bool
    normalized_strategies: Tuple[str, ...] = field(default_factory=tuple)
    errors: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
