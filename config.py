#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理模块

集中管理项目的所有配置参数，包括：
- 模型路径
- 设备ID
- 支持的分辨率
- 其他配置参数

使用JSON文件存储配置，支持从配置文件加载和更新配置
"""

import json
import os


class Config:
    """配置类"""
    
    # 配置文件路径
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
    DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "default.json")
    
    # 支持的输入分辨率
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
    
    # 昇腾310B有4个AI Core
    MAX_AI_CORES = 4
    
    # 支持的图像读取后端
    SUPPORTED_BACKENDS = ["pil", "opencv"]
    
    # 支持的推理模式
    SUPPORTED_MODES = ["basic", "fast", "multithread", "high_res"]
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.DEFAULT_CONFIG_FILE):
                with open(self.DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                # 如果配置文件不存在，使用默认配置
                self._config = {
                    "model_path": "models/yolov8s.om",
                    "device_id": 0,
                    "resolution": "640x640",
                    "tile_size": 640,
                    "overlap": 100,
                    "num_threads": 4,
                    "backend": "pil",
                    "conf_threshold": 0.4,
                    "iou_threshold": 0.5,
                    "max_detections": 100,
                    "enable_logging": True,
                    "log_level": "info",
                    "enable_profiling": False
                }
                # 保存默认配置
                self._save_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            # 使用默认配置
            self._config = {
                "model_path": "models/yolov8s.om",
                "device_id": 0,
                "resolution": "640x640",
                "tile_size": 640,
                "overlap": 100,
                "num_threads": 4,
                "backend": "pil",
                "conf_threshold": 0.4,
                "iou_threshold": 0.5,
                "max_detections": 100,
                "enable_logging": True,
                "log_level": "info",
                "enable_profiling": False
            }
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            # 确保配置目录存在
            os.makedirs(self.CONFIG_DIR, exist_ok=True)
            with open(self.DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    @classmethod
    def get_instance(cls):
        """获取配置实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def model_path(self):
        """模型路径"""
        return self._config.get("model_path", "models/yolov8s.om")
    
    @property
    def device_id(self):
        """设备ID"""
        return self._config.get("device_id", 0)
    
    @property
    def resolution(self):
        """输入分辨率"""
        return self._config.get("resolution", "640x640")
    
    @property
    def tile_size(self):
        """高分辨率推理的分块大小"""
        return self._config.get("tile_size", 640)
    
    @property
    def overlap(self):
        """高分辨率推理的重叠区域"""
        return self._config.get("overlap", 100)
    
    @property
    def num_threads(self):
        """多线程推理的线程数"""
        return self._config.get("num_threads", 4)
    
    @property
    def backend(self):
        """图像读取后端"""
        return self._config.get("backend", "pil")
    
    @property
    def conf_threshold(self):
        """置信度阈值"""
        return self._config.get("conf_threshold", 0.4)
    
    @property
    def iou_threshold(self):
        """IOU阈值"""
        return self._config.get("iou_threshold", 0.5)
    
    @property
    def max_detections(self):
        """最大检测数量"""
        return self._config.get("max_detections", 100)
    
    @property
    def enable_logging(self):
        """是否启用日志"""
        return self._config.get("enable_logging", True)
    
    @property
    def log_level(self):
        """日志级别"""
        return self._config.get("log_level", "info")
    
    @property
    def enable_profiling(self):
        """是否启用性能分析"""
        return self._config.get("enable_profiling", False)
    
    def update(self, **kwargs):
        """更新配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if key in self._config:
                self._config[key] = value
        # 保存更新后的配置
        self._save_config()
    
    def get(self, key, default=None):
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._config.get(key, default)
    
    @classmethod
    def get_resolution(cls, resolution_name):
        """获取分辨率尺寸
        
        Args:
            resolution_name: 分辨率名称
            
        Returns:
            tuple: (宽度, 高度)
        """
        return cls.SUPPORTED_RESOLUTIONS.get(resolution_name, cls.SUPPORTED_RESOLUTIONS["640x640"])
    
    @classmethod
    def is_supported_resolution(cls, resolution_name):
        """检查分辨率是否支持
        
        Args:
            resolution_name: 分辨率名称
            
        Returns:
            bool: 是否支持
        """
        return resolution_name in cls.SUPPORTED_RESOLUTIONS
    
    @classmethod
    def is_supported_backend(cls, backend):
        """检查后端是否支持
        
        Args:
            backend: 后端名称
            
        Returns:
            bool: 是否支持
        """
        return backend in cls.SUPPORTED_BACKENDS
    
    @classmethod
    def is_supported_mode(cls, mode):
        """检查推理模式是否支持
        
        Args:
            mode: 推理模式
            
        Returns:
            bool: 是否支持
        """
        return mode in cls.SUPPORTED_MODES
