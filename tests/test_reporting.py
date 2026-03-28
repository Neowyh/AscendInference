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


def test_benchmark_result_compat_views_follow_execution_record():
    from benchmark.scenarios import BenchmarkResult, ModelInfo

    result = BenchmarkResult(
        scenario_name="model_selection",
        model_info=ModelInfo(name="original", resolution="640x640"),
        metrics={
            "execute": {"avg": 12.0},
            "fps": {"pure": 83.3, "e2e": 55.5},
        },
        config={"iterations": 100},
        resource_stats={"cpu": {"avg": 50.0}},
        timestamp=123.0,
    )

    result.execution_record.task_name = "updated"
    result.execution_record.model_info.name = "updated_model"
    result.execution_record.config["iterations"] = 200
    result.execution_record.resource_stats["cpu"]["avg"] = 75.0
    result.execution_record.timestamp = 456.0
    result.execution_record.model_metrics["execute"]["avg"] = 15.0

    assert result.scenario_name == "updated"
    assert result.config["iterations"] == 200
    assert result.resource_stats["cpu"]["avg"] == 75.0
    assert result.timestamp == 456.0
    assert result.model_info.name == "updated_model"
    assert result.metrics["execute"]["avg"] == 15.0
    assert result.metrics["fps"]["pure"] == 83.3


def test_legacy_metrics_snapshot_does_not_persist_in_place_changes():
    from benchmark.scenarios import BenchmarkResult, ModelInfo

    result = BenchmarkResult(
        scenario_name="model_selection",
        model_info=ModelInfo(name="snapshot_model"),
        metrics={"execute": {"avg": 12.0}, "fps": {"pure": 83.3}},
    )

    snapshot = result.metrics
    snapshot["execute"]["avg"] = 99.0

    assert result.metrics["execute"]["avg"] == 12.0


def test_execution_record_model_name_keeps_model_info_in_sync():
    from benchmark.scenarios import BenchmarkResult, ModelInfo

    result = BenchmarkResult(
        scenario_name="model_selection",
        model_info=ModelInfo(name="snapshot_model"),
        metrics={"execute": {"avg": 12.0}, "fps": {"pure": 83.3}},
    )

    result.execution_record.model_name = "renamed_model"

    assert result.model_info.name == "renamed_model"
    assert result.execution_record.model_info.name == "renamed_model"


def test_model_info_name_updates_execution_record_model_name():
    from benchmark.scenarios import BenchmarkResult, ModelInfo

    result = BenchmarkResult(
        scenario_name="model_selection",
        model_info=ModelInfo(name="snapshot_model"),
        metrics={"execute": {"avg": 12.0}, "fps": {"pure": 83.3}},
    )

    result.model_info.name = "renamed_model"

    assert result.model_info.name == "renamed_model"
    assert result.execution_record.model_name == "renamed_model"


def test_execution_record_model_info_name_updates_model_name_view():
    from benchmark.scenarios import BenchmarkResult, ModelInfo

    result = BenchmarkResult(
        scenario_name="model_selection",
        model_info=ModelInfo(name="snapshot_model"),
        metrics={"execute": {"avg": 12.0}, "fps": {"pure": 83.3}},
    )

    result.execution_record.model_info.name = "renamed_model"

    assert result.model_info.name == "renamed_model"
    assert result.execution_record.model_name == "renamed_model"
