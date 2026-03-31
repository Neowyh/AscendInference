#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告归档工具
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _sanitize_segment(value: str) -> str:
    cleaned = str(value or "standard").strip().replace("\\", "_").replace("/", "_")
    return cleaned or "standard"


def build_archive_path(root: Path, task_name: str, route_type: str | None) -> Path:
    """根据任务名和路线构建归档目录"""
    return Path(root) / _sanitize_segment(task_name) / _sanitize_segment(route_type or "standard")


def archive_result(
    archive_root: Path,
    task_metadata: Dict[str, Any],
    report_body: str,
    raw_results: Dict[str, Any],
    report_extension: str = ".md",
) -> Dict[str, Path]:
    """把报告和原始结果归档到统一目录"""
    archive_dir = build_archive_path(
        Path(archive_root),
        task_metadata.get("task_name", "evaluation"),
        task_metadata.get("route_type", "standard"),
    )
    archive_dir.mkdir(parents=True, exist_ok=True)

    report_path = archive_dir / f"report{report_extension}"
    raw_results_path = archive_dir / "raw_results.json"
    metadata_path = archive_dir / "metadata.json"

    report_path.write_text(report_body, encoding="utf-8")
    raw_results_path.write_text(
        json.dumps(raw_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(task_metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return {
        "archive_dir": archive_dir,
        "report_path": report_path,
        "raw_results_path": raw_results_path,
        "metadata_path": metadata_path,
    }
