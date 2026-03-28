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
