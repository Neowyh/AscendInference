#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

简化的两层配置系统：
1. JSON 文件 - 基础配置（最全）
2. 命令行参数 - 覆盖配置（优先级更高）
"""

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """配置类 - 所有配置项都在 JSON 文件中定义"""
    
    # 推理核心配置
    model_path: str = "models/yolov8s.om"
    device_id: int = 0
    resolution: str = "640x640"
    
    # 高分辨率配置
    tile_size: int = 640
    overlap: int = 100
    
    # 多线程配置
    num_threads: int = 4
    
    # 图像处理配置
    backend: str = "pil"
    
    # 检测配置
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    max_detections: int = 100
    
    # 日志和性能配置
    enable_logging: bool = True
    log_level: str = "info"
    enable_profiling: bool = False
    
    # 类常量
    SUPPORTED_RESOLUTIONS = {
        "640x640": (640, 640),
        "1k": (1024, 1024),
        "1k2k": (1024, 2048),
        "2k": (2048, 2048),
        "2k4k": (2048, 4096),
        "4k": (4096, 4096),
        "4k6k": (4096, 6144),
        "3k6k": (3072, 6144),
        "6k": (6144, 6144)
    }
    MAX_AI_CORES = 4
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """从 JSON 文件加载配置（基础配置）"""
        if not os.path.exists(path):
            print(f"警告：配置文件不存在 {path}，使用默认配置")
            return cls()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"警告：加载配置文件失败 {e}，使用默认配置")
            return cls()
    
    def apply_overrides(self, **kwargs):
        """应用命令行参数覆盖（优先级更高）"""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def get_resolution(cls, resolution_name: str) -> tuple:
        """获取分辨率尺寸"""
        return cls.SUPPORTED_RESOLUTIONS.get(resolution_name, (640, 640))
