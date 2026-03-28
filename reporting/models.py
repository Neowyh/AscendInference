#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的执行记录模型。

把模型执行指标和系统端到端指标分开保存，同时提供兼容旧报告代码的
合并视图。
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


def _deep_merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，保留嵌套指标结构。"""
    merged: Dict[str, Any] = deepcopy(left)
    for key, value in right.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _split_legacy_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """把旧的扁平指标拆成模型指标和系统指标。"""
    model_metrics: Dict[str, Any] = {}
    system_metrics: Dict[str, Any] = {}

    model_stage_keys = {"preprocess", "execute", "postprocess", "queue_wait", "total", "ratios"}
    system_stage_keys = {
        "iterations",
        "duration",
        "strategy",
        "throughput_fps",
        "total_tasks",
        "completed_tasks",
        "test_duration",
    }

    for key, value in metrics.items():
        if key == "fps" and isinstance(value, dict):
            if "pure" in value:
                model_metrics.setdefault("fps", {})["pure"] = value["pure"]
            if "e2e" in value:
                system_metrics.setdefault("fps", {})["e2e"] = value["e2e"]
            remaining = {k: v for k, v in value.items() if k not in {"pure", "e2e"}}
            if remaining:
                model_metrics.setdefault("fps", {}).update(remaining)
            continue

        if key in model_stage_keys:
            model_metrics[key] = value
        elif key in system_stage_keys:
            system_metrics[key] = value
        else:
            system_metrics[key] = value

    return {
        "model_metrics": model_metrics,
        "system_metrics": system_metrics,
    }


@dataclass
class ExecutionRecord:
    """统一执行记录。

    Attributes:
        task_name: 任务或场景名称
        route_type: 路径类型
        model_name: 模型名称
        model_metrics: 纯模型执行相关指标
        system_metrics: 端到端和系统级指标
        resource_stats: 资源统计
        config: 记录对应的配置
        strategies: 启用的策略列表
        timestamp: 记录时间戳
    """

    task_name: str = ""
    route_type: str = ""
    model_name: str = ""
    model_info: Any = None
    model_metrics: Dict[str, Any] = field(default_factory=dict)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_stats: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    strategies: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        if name == "model_name":
            model_info = self.__dict__.get("model_info")
            if model_info is not None and hasattr(model_info, "name"):
                object.__setattr__(model_info, "name", value)
        elif name == "model_info":
            model_info = value
            if model_info is not None and hasattr(model_info, "name"):
                object.__setattr__(self, "model_name", getattr(model_info, "name", ""))

    def to_legacy_metrics(self) -> Dict[str, Any]:
        """返回兼容旧代码的合并指标视图。"""
        return _deep_merge_dicts(self.model_metrics, self.system_metrics)

    @classmethod
    def from_legacy_metrics(
        cls,
        metrics: Optional[Dict[str, Any]] = None,
        *,
        task_name: str = "",
        route_type: str = "",
        model_name: str = "",
        model_info: Any = None,
        resource_stats: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        strategies: Optional[list] = None,
        timestamp: Optional[float] = None,
        ) -> "ExecutionRecord":
        """从旧指标字典构造记录。"""
        metrics = metrics or {}
        split_metrics = _split_legacy_metrics(metrics)
        return cls(
            task_name=task_name,
            route_type=route_type,
            model_name=model_name,
            model_info=model_info,
            model_metrics=split_metrics["model_metrics"],
            system_metrics=split_metrics["system_metrics"],
            resource_stats=dict(resource_stats or {}),
            config=dict(config or {}),
            strategies=list(strategies or []),
            timestamp=time.time() if timestamp is None else timestamp,
        )
