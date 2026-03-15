#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理命令实现
"""
import os
import json
from pathlib import Path
from config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES


def cmd_config(args):
    """配置管理命令"""
    if args.show:
        print("=" * 60)
        print("当前配置")
        print("=" * 60)

        config = Config()
        if args.config:
            config = Config.from_json(args.config)

        print(f"\n模型配置:")
        print(f"  model_path: {config.model_path}")
        print(f"  device_id: {config.device_id}")
        print(f"  resolution: {config.resolution}")

        print(f"\n推理配置:")
        print(f"  tile_size: {config.tile_size}")
        print(f"  overlap: {config.overlap}")
        print(f"  num_threads: {config.num_threads}")
        print(f"  backend: {config.backend}")

        print(f"\n检测配置:")
        print(f"  conf_threshold: {config.conf_threshold}")
        print(f"  iou_threshold: {config.iou_threshold}")
        print(f"  max_detections: {config.max_detections}")

        print(f"\n日志配置:")
        print(f"  enable_logging: {config.enable_logging}")
        print(f"  log_level: {config.log_level}")
        print(f"  enable_profiling: {config.enable_profiling}")

        print(f"\n系统配置:")
        print(f"  MAX_AI_CORES: {MAX_AI_CORES}")

        print(f"\n支持的分辨率:")
        for res, (w, h) in SUPPORTED_RESOLUTIONS.items():
            print(f"  {res}: {w}x{h}")

        return 0

    if args.validate:
        print("=" * 60)
        print("配置验证")
        print("=" * 60)

        errors = []

        config = Config()
        if args.config:
            config = Config.from_json(args.config)

        if not os.path.exists(config.model_path):
            errors.append(f"模型文件不存在：{config.model_path}")

        if config.resolution not in SUPPORTED_RESOLUTIONS:
            errors.append(f"不支持的分辨率：{config.resolution}")

        if config.device_id < 0:
            errors.append(f"无效的设备 ID：{config.device_id}")

        if config.num_threads < 1 or config.num_threads > MAX_AI_CORES:
            errors.append(f"线程数应在 1-{MAX_AI_CORES} 之间：{config.num_threads}")

        if config.backend not in ['pil', 'opencv']:
            errors.append(f"不支持的后端：{config.backend}")

        if config.conf_threshold < 0 or config.conf_threshold > 1:
            errors.append(f"置信度阈值应在 0-1 之间：{config.conf_threshold}")

        if config.iou_threshold < 0 or config.iou_threshold > 1:
            errors.append(f"IOU 阈值应在 0-1 之间：{config.iou_threshold}")

        if errors:
            print("\n❌ 配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("\n✅ 配置验证通过")
            return 0

    if args.generate:
        output_path = Path(args.generate)
        config = Config()

        config_dict = {
            "model_path": config.model_path,
            "device_id": config.device_id,
            "resolution": config.resolution,
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "num_threads": config.num_threads,
            "backend": config.backend,
            "conf_threshold": config.conf_threshold,
            "iou_threshold": config.iou_threshold,
            "max_detections": config.max_detections,
            "enable_logging": config.enable_logging,
            "log_level": config.log_level,
            "enable_profiling": config.enable_profiling,
            "warmup": config.warmup,
            "warmup_iterations": config.warmup_iterations
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        print(f"配置文件已生成：{output_path}")
        return 0

    print("请指定操作：--show, --validate 或 --generate")
    return 1
