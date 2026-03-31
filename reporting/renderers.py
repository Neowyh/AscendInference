#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告渲染器

提供 Markdown 与 JSON 报告渲染能力。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable


class MarkdownReportRenderer:
    """Markdown 报告渲染器"""

    def render(self, report_model: Dict[str, Any]) -> str:
        lines = [
            f"# {report_model.get('title', 'Evaluation Report')}",
            "",
            f"- Generated At: {report_model.get('generated_at', datetime.now().isoformat())}",
            f"- Task: {report_model.get('task_name', 'evaluation')}",
            f"- Result Count: {report_model.get('result_count', 0)}",
            "",
        ]

        route_comparison = report_model.get("route_comparison", [])
        if route_comparison:
            lines.extend([
                "## Route Comparison",
                "",
                "| Route | Results | Models | Strategies |",
                "| --- | ---: | --- | --- |",
            ])
            for item in route_comparison:
                lines.append(
                    f"| {item.get('route', 'standard')} | {item.get('result_count', 0)} | "
                    f"{', '.join(item.get('models', [])) or '-'} | "
                    f"{', '.join(item.get('strategies', [])) or '-'} |"
                )
            lines.append("")

        results = report_model.get("results", [])
        if results:
            lines.extend([
                "## Results",
                "",
            ])
            for result in results:
                lines.extend(self._render_result(result))

        return "\n".join(lines).rstrip() + "\n"

    def _render_result(self, result: Dict[str, Any]) -> Iterable[str]:
        metrics = result.get("metrics", {})
        model_name = result.get("model_name", "unknown")
        route_type = result.get("route_type", "standard")
        strategies = ", ".join(result.get("strategies", [])) or "baseline"
        fps = metrics.get("fps", {})

        return [
            f"### {model_name}",
            "",
            f"- Route: {route_type}",
            f"- Strategies: {strategies}",
            f"- Pure FPS: {fps.get('pure', 0)}",
            f"- E2E FPS: {fps.get('e2e', 0)}",
            "",
        ]


class JsonReportRenderer:
    """JSON 报告渲染器"""

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def render(self, report_model: Dict[str, Any]) -> str:
        return json.dumps(report_model, indent=self.indent, ensure_ascii=False, default=str)
