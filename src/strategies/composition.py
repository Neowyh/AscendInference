#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组合与校验引擎

为策略评测提供统一的规范化、路线兼容性检查和执行器声明。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .base_unit import StrategyUnit, ValidationResult


class StrategyCompositionEngine:
    """策略组合引擎"""

    def __init__(self) -> None:
        self._units: Dict[str, StrategyUnit] = {}
        self._aliases: Dict[str, str] = {}
        self._register_builtin_units()

    def _register_builtin_units(self) -> None:
        self.register(
            StrategyUnit(
                name="multithread",
                executor_kind="multithread",
            )
        )
        self.register(
            StrategyUnit(
                name="batch",
                executor_kind="batch",
            )
        )
        self.register(
            StrategyUnit(
                name="pipeline",
                executor_kind="pipeline",
            )
        )
        self.register(
            StrategyUnit(
                name="memory_pool",
                executor_kind="simple",
            )
        )
        self.register(
            StrategyUnit(
                name="high_res_tiling",
                executor_kind="high_res",
                supported_routes=("tiled_route",),
                aliases=("high_res",),
            )
        )
        self.register(
            StrategyUnit(
                name="async_io",
                executor_kind="simple",
            )
        )
        self.register(
            StrategyUnit(
                name="cache",
                executor_kind="simple",
            )
        )

    def register(self, unit: StrategyUnit) -> None:
        """注册策略单元"""
        self._units[unit.name] = unit
        for alias in unit.aliases:
            self._aliases[alias] = unit.name

    def get_unit(self, strategy_name: str) -> Optional[StrategyUnit]:
        """获取规范化后的策略单元"""
        canonical_name = self._aliases.get(strategy_name, strategy_name)
        return self._units.get(canonical_name)

    def normalize_strategies(self, strategy_names: Iterable[str]) -> List[str]:
        """规范化策略名称并去重"""
        normalized: List[str] = []
        seen = set()
        for strategy_name in strategy_names:
            unit = self.get_unit(strategy_name)
            canonical_name = unit.name if unit else strategy_name
            if canonical_name not in seen:
                normalized.append(canonical_name)
                seen.add(canonical_name)
        return normalized

    def validate(
        self,
        strategy_names: Iterable[str],
        route_type: Optional[str] = None,
    ) -> ValidationResult:
        """校验策略集合在指定路线下是否合法"""
        normalized = []
        errors = []

        for strategy_name in strategy_names:
            unit = self.get_unit(strategy_name)
            if unit is None:
                errors.append(f"未知策略单元: {strategy_name}")
                continue

            if unit.name not in normalized:
                normalized.append(unit.name)

            if not unit.supports_route(route_type):
                errors.append(
                    f"{unit.name} 与路线 {route_type} 不兼容"
                )

        normalized_set = set(normalized)
        for unit_name in normalized:
            unit = self._units[unit_name]
            for conflict_name in unit.conflicts:
                if conflict_name in normalized_set:
                    errors.append(
                        f"{unit.name} 与 {conflict_name} 不能同时启用"
                    )

        return ValidationResult(
            is_valid=not errors,
            normalized_strategies=tuple(normalized),
            errors=tuple(errors),
        )
