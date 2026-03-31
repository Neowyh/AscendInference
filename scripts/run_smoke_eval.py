#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend 硬件 smoke 评测脚本

根据样例配置拼装 CLI 命令；默认仅打印命令，加 --run 时实际执行。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CONFIGS = {
    "standard": "config/evaluation/smoke_standard_eval.json",
    "remote": "config/evaluation/smoke_remote_eval.json",
    "strategy": "config/evaluation/smoke_strategy_eval.json",
}


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_optional_args(command: List[str], args_map: Dict[str, Any], option_name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, list):
        if value:
            command.append(option_name)
            command.extend(str(item) for item in value)
        return
    if isinstance(value, bool):
        if value:
            command.append(option_name)
        return
    command.extend([option_name, str(value)])


def build_command(mode: str, config: Dict[str, Any]) -> List[str]:
    if mode == "standard":
        command = [
            sys.executable,
            "main.py",
            "model-bench",
            *config["models"],
            "--images",
            *config["images"],
        ]
        _append_optional_args(command, config, "--input-tiers", config.get("input_tiers"))
        _append_optional_args(command, config, "--iterations", config.get("iterations"))
        _append_optional_args(command, config, "--warmup", config.get("warmup"))
        _append_optional_args(command, config, "--device", config.get("device"))
        _append_optional_args(command, config, "--backend", config.get("backend"))
        _append_optional_args(command, config, "--output", config.get("output"))
        return command

    if mode == "remote":
        command = [
            sys.executable,
            "main.py",
            "model-bench",
            *config["models"],
            "--images",
            *config["images"],
        ]
        _append_optional_args(command, config, "--routes", config.get("routes"))
        _append_optional_args(command, config, "--image-size-tiers", config.get("image_size_tiers"))
        _append_optional_args(command, config, "--iterations", config.get("iterations"))
        _append_optional_args(command, config, "--warmup", config.get("warmup"))
        _append_optional_args(command, config, "--device", config.get("device"))
        _append_optional_args(command, config, "--backend", config.get("backend"))
        _append_optional_args(command, config, "--output", config.get("output"))
        return command

    if mode == "strategy":
        command = [
            sys.executable,
            "main.py",
            "strategy-bench",
            "--model",
            config["model"],
            "--image",
            config["image"],
        ]
        _append_optional_args(command, config, "--strategies", config.get("strategies"))
        _append_optional_args(command, config, "--routes", config.get("routes"))
        _append_optional_args(command, config, "--image-size-tiers", config.get("image_size_tiers"))
        _append_optional_args(command, config, "--iterations", config.get("iterations"))
        _append_optional_args(command, config, "--warmup", config.get("warmup"))
        _append_optional_args(command, config, "--threads", config.get("threads"))
        _append_optional_args(command, config, "--batch-size", config.get("batch_size"))
        _append_optional_args(command, config, "--device", config.get("device"))
        _append_optional_args(command, config, "--backend", config.get("backend"))
        _append_optional_args(command, config, "--output", config.get("output"))
        return command

    raise ValueError(f"Unsupported mode: {mode}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建或执行 Ascend smoke 评测命令")
    parser.add_argument("--mode", choices=sorted(DEFAULT_CONFIGS), required=True, help="smoke 类型")
    parser.add_argument("--config", help="配置文件路径，默认使用内置样例配置")
    parser.add_argument("--run", action="store_true", help="实际执行命令；默认仅打印")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / (args.config or DEFAULT_CONFIGS[args.mode])

    if not config_path.exists():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        return 1

    config = _load_config(config_path)
    command = build_command(args.mode, config)
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)

    print(f"[smoke] mode={args.mode}")
    print(f"[smoke] config={config_path}")
    print(f"[smoke] command={printable}")

    if not args.run:
        return 0

    completed = subprocess.run(command, cwd=root, env=os.environ.copy())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
