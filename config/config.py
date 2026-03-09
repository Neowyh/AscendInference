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
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any


_logger = logging.getLogger('ascend_inference.config')


_SUPPORTED_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
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


_MAX_AI_CORES: int = 4


def get_supported_resolutions() -> Dict[str, Tuple[int, int]]:
    """获取支持的分辨率列表"""
    return _SUPPORTED_RESOLUTIONS.copy()


def get_max_ai_cores() -> int:
    """获取最大 AI 核心数"""
    return _MAX_AI_CORES


@dataclass
class Config:
    """配置类 - 所有配置项都在 JSON 文件中定义"""

    model_path: str = "models/yolov8s.om"
    device_id: int = 0
    resolution: str = "640x640"
    
    tile_size: int = 640
    overlap: int = 100
    
    num_threads: int = 4
    
    backend: str = "pil"
    
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    max_detections: int = 100
    
    enable_logging: bool = True
    log_level: str = "info"
    enable_profiling: bool = False
    
    SUPPORTED_RESOLUTIONS = _SUPPORTED_RESOLUTIONS
    MAX_AI_CORES = _MAX_AI_CORES
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """从 JSON 文件加载配置（基础配置）
        
        Args:
            path: JSON 配置文件路径
            
        Returns:
            Config 实例
        """
        if not os.path.exists(path):
            _logger.warning(f"配置文件不存在 {path}，使用默认配置")
            return cls()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            _logger.warning(f"加载配置文件失败 {e}，使用默认配置")
            return cls()
    
    def apply_overrides(self, **kwargs: Any) -> None:
        """应用命令行参数覆盖（优先级更高）
        
        Args:
            **kwargs: 需要覆盖的配置项
        """
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    @staticmethod
    def get_resolution(resolution_name: str) -> Tuple[int, int]:
        """获取分辨率尺寸
        
        Args:
            resolution_name: 分辨率名称
            
        Returns:
            (width, height) 元组
        """
        return _SUPPORTED_RESOLUTIONS.get(resolution_name, (640, 640))
    
    @staticmethod
    def is_supported_resolution(resolution_name: str) -> bool:
        """检查分辨率是否支持
        
        Args:
            resolution_name: 分辨率名称
            
        Returns:
            是否支持
        """
        return resolution_name in _SUPPORTED_RESOLUTIONS


SUPPORTED_RESOLUTIONS = _SUPPORTED_RESOLUTIONS
MAX_AI_CORES = _MAX_AI_CORES
