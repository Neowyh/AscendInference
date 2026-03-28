#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting.models 的最小契约测试
"""

from reporting.models import ExecutionRecord


def test_execution_record_separates_model_and_system_metrics():
    record = ExecutionRecord(
        task_name="model_selection",
        route_type="standard",
        model_name="yolov8n.om",
        model_metrics={
            "execute": {"avg": 12.0},
            "fps": {"pure": 83.3},
        },
        system_metrics={
            "fps": {"e2e": 55.5},
            "iterations": {"test": 100},
        },
    )

    assert record.model_metrics["execute"]["avg"] == 12.0
    assert record.system_metrics["fps"]["e2e"] == 55.5
    assert record.to_legacy_metrics()["fps"]["pure"] == 83.3
    assert record.to_legacy_metrics()["fps"]["e2e"] == 55.5
    assert record.to_legacy_metrics()["iterations"]["test"] == 100


def test_benchmark_result_uses_legacy_metrics_conversion_path():
    from benchmark.scenarios import BenchmarkResult, ModelInfo

    result = BenchmarkResult(
        scenario_name="model_selection",
        model_info=ModelInfo(name="legacy_model"),
        metrics={
            "preprocess": {"avg": 10.0},
            "execute": {"avg": 12.0},
            "fps": {"pure": 83.3, "e2e": 55.5},
            "iterations": {"test": 100},
        },
    )

    assert result.model_metrics["execute"]["avg"] == 12.0
    assert result.system_metrics["fps"]["e2e"] == 55.5
    assert result.execution_record.task_name == "model_selection"
    assert result.execution_record.model_name == "legacy_model"
