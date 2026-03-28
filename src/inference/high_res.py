#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率推理模块

提供高分辨率图像推理功能，支持：
- 图像分块处理
- 权重融合
- 并行推理
"""

import time
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Optional, Union, Dict, Any, List, Tuple

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.validators import validate_image_backend, validate_file_path
from .postprocessor import split_image, merge_results
from .multithread import MultithreadInference

logger = LoggerConfig.setup_logger('ascend_inference.high_res', format_type='text')


class HighResInference:
    """高分辨率图像推理管理器"""
    
    def __init__(self, config: Optional[Config] = None):
        """初始化高分辨率推理

        Args:
            config: Config 实例
        """
        self.config = config or Config()
        self.num_threads = min(self.config.num_threads, Config.MAX_AI_CORES)
        self.model_path = self.config.model_path
        self.tile_size = (self.config.tile_size, self.config.tile_size)
        self.overlap = self.config.overlap / self.config.tile_size if self.config.overlap > 1 else self.config.overlap
        
        self.multithread: Optional[MultithreadInference] = None
    
    def _init_multithread(self) -> bool:
        """初始化多线程推理实例
        
        Returns:
            bool: 是否初始化成功
        """
        self.multithread = MultithreadInference(
            Config(
                model_path=self.model_path,
                num_threads=self.num_threads,
                resolution=f"{self.tile_size[1]}x{self.tile_size[0]}"
            )
        )
        return self.multithread.start()
    
    def _load_image(self, image_path: str, backend: str) -> Optional[np.ndarray]:
        """加载图像
        
        Args:
            image_path: 图像路径
            backend: 图像处理后端
            
        Returns:
            np.ndarray: 图像数组，失败返回None
        """
        if backend == 'opencv' and HAS_OPENCV:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像：{image_path}")
                return None
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                return np.array(image)
            except Exception as e:
                logger.error(f"无法读取图像：{e}")
                logger.debug(f"图像路径：{image_path}")
                return None
    
    def process_image(self, image_path: str, backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """处理高分辨率图像

        Args:
            image_path: 图像路径
            backend: 图像读取后端

        Returns:
            dict: 推理结果，失败返回None
        """
        if backend is None:
            backend = self.config.backend

        validate_image_backend(backend)
        validate_file_path(image_path, must_exist=True)

        image_array = self._load_image(image_path, backend)
        if image_array is None:
            return None
        
        logger.info(f"处理图像：{image_path}, 形状：{image_array.shape}")
        
        start_time = time.time()
        tiles, positions, weight_map = split_image(image_array, self.tile_size, self.overlap)
        logger.debug(f"分割完成：{len(tiles)} 个子块，耗时：{time.time() - start_time:.2f} 秒")
        
        if not self._init_multithread():
            logger.error("无法启动推理")
            return None
        
        for i, tile in enumerate(tiles):
            self.multithread.add_task(tile, backend)
        
        self.multithread.wait_completion()
        results = self.multithread.get_results()
        
        merged_result = merge_results(results, positions, image_array.shape)
        
        logger.info(f"推理完成：{len(merged_result['sub_results'])}/{len(tiles)} 个子块成功")
        
        self.multithread.stop()
        self.multithread = None
        return merged_result
    
    def stop(self) -> None:
        """停止推理"""
        if self.multithread:
            self.multithread.stop()
            self.multithread = None
