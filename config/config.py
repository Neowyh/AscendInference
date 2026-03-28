#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

简化的两层配置系统：
1. JSON 文件 - 基础配置（最全）
2. 命令行参数 - 覆盖配置（优先级更高）

支持策略配置和评测配置
"""

import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

from config.strategy_config import StrategyConfig, BenchmarkConfig, ModelInfoConfig, EvaluationConfig


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
    """配置类 - 所有配置项都在 JSON 文件中定义
    
    支持策略配置和评测配置，保持向后兼容
    """
    
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
    
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    model_info: ModelInfoConfig = field(default_factory=ModelInfoConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    warmup: int = 5
    warmup_iterations: int = 5
    
    SUPPORTED_RESOLUTIONS = _SUPPORTED_RESOLUTIONS
    MAX_AI_CORES = _MAX_AI_CORES

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, EvaluationConfig):
            raise TypeError("evaluation must be an EvaluationConfig instance")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "evaluation" and not isinstance(value, EvaluationConfig):
            raise TypeError("evaluation must be an EvaluationConfig instance")
        super().__setattr__(name, value)
    
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
            
            config = cls(
                model_path=data.get('model_path', cls.model_path),
                device_id=data.get('device_id', cls.device_id),
                resolution=data.get('resolution', cls.resolution),
                tile_size=data.get('tile_size', cls.tile_size),
                overlap=data.get('overlap', cls.overlap),
                num_threads=data.get('num_threads', cls.num_threads),
                backend=data.get('backend', cls.backend),
                conf_threshold=data.get('conf_threshold', cls.conf_threshold),
                iou_threshold=data.get('iou_threshold', cls.iou_threshold),
                max_detections=data.get('max_detections', cls.max_detections),
                enable_logging=data.get('enable_logging', cls.enable_logging),
                log_level=data.get('log_level', cls.log_level),
                enable_profiling=data.get('enable_profiling', cls.enable_profiling),
                warmup=data.get('warmup', cls.warmup),
                warmup_iterations=data.get('warmup_iterations', cls.warmup_iterations)
            )
            
            if 'strategies' in data:
                config.strategies = StrategyConfig.from_dict(data['strategies'])
            
            if 'benchmark' in data:
                config.benchmark = BenchmarkConfig.from_dict(data['benchmark'])
            
            if 'model_info' in data:
                config.model_info = ModelInfoConfig.from_dict(data['model_info'])

            if 'evaluation' in data:
                config.evaluation = EvaluationConfig.from_dict(data['evaluation'])
            
            return config
        except Exception as e:
            _logger.warning(f"加载配置文件失败 {e}，使用默认配置")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict: 配置字典
        """
        return {
            'model_path': self.model_path,
            'device_id': self.device_id,
            'resolution': self.resolution,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'num_threads': self.num_threads,
            'backend': self.backend,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'enable_profiling': self.enable_profiling,
            'warmup': self.warmup,
            'warmup_iterations': self.warmup_iterations,
            'strategies': self.strategies.to_dict(),
            'benchmark': self.benchmark.to_dict(),
            'model_info': self.model_info.to_dict(),
            'evaluation': self.evaluation.to_dict()
        }
    
    def apply_overrides(self, **kwargs: Any) -> None:
        """应用命令行参数覆盖（优先级更高）
        
        Args:
            **kwargs: 需要覆盖的配置项
        """
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    def get_enabled_strategies(self) -> list:
        """获取已启用的策略列表
        
        Returns:
            list: 已启用的策略名称列表
        """
        return self.strategies.get_enabled_strategies()
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """检查指定策略是否启用
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 是否启用
        """
        strategy_map = {
            'multithread': self.strategies.multithread,
            'batch': self.strategies.batch,
            'pipeline': self.strategies.pipeline,
            'memory_pool': self.strategies.memory_pool,
            'high_res': self.strategies.high_res,
            'async_io': self.strategies.async_io,
            'cache': self.strategies.cache
        }
        strategy = strategy_map.get(strategy_name)
        return strategy.enabled if strategy else False
    
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
