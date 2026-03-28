#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率策略组件

支持大分辨率图像的分块推理和结果融合
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from .base import Strategy, InferenceContext
from config.strategy_config import HighResStrategyConfig


class HighResStrategy(Strategy):
    """高分辨率策略组件
    
    支持大分辨率图像的分块推理和结果融合
    """
    
    name = "high_res"
    
    def __init__(self, config: Optional[HighResStrategyConfig] = None):
        """初始化高分辨率策略
        
        Args:
            config: 高分辨率策略配置
        """
        super().__init__(config or HighResStrategyConfig())
        self._highres_instance = None
        self._tile_count = 0
        self._stats = {
            'total_tiles': 0,
            'total_images': 0,
            'avg_tiles_per_image': 0.0
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HighResStrategy':
        """从字典创建策略实例
        
        Args:
            data: 配置字典
            
        Returns:
            HighResStrategy: 策略实例
        """
        config = HighResStrategyConfig.from_dict(data)
        return cls(config)
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        """应用高分辨率策略
        
        Args:
            context: 推理上下文
            
        Returns:
            InferenceContext: 处理后的上下文
        """
        if not self.enabled:
            return context
        
        from src.inference import HighResInference
        from config import Config
        
        inference_config = context.config or Config()
        inference_config.tile_size = self.config.tile_size
        inference_config.overlap = self.config.overlap
        
        self._highres_instance = HighResInference(inference_config)
        
        context.set_state('highres_instance', self._highres_instance)
        context.set_state('tile_size', self.config.tile_size)
        context.set_state('overlap', self.config.overlap)
        context.set_state('weight_fusion', self.config.weight_fusion)
        context.set_metadata('strategy_type', 'high_res')
        
        return context
    
    def process_image(self, image_path: str, backend: str = 'pil') -> Optional[Dict[str, Any]]:
        """处理高分辨率图像
        
        Args:
            image_path: 图像路径
            backend: 图像处理后端
            
        Returns:
            推理结果
        """
        if not self._highres_instance:
            return None
        
        result = self._highres_instance.process_image(image_path, backend)
        
        if result:
            num_tiles = result.get('num_tiles', 0)
            self._stats['total_tiles'] += num_tiles
            self._stats['total_images'] += 1
            self._stats['avg_tiles_per_image'] = (
                self._stats['total_tiles'] / self._stats['total_images']
            )
        
        return result
    
    def split_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray]:
        """分割图像为子块
        
        Args:
            image: 输入图像
            
        Returns:
            Tuple: (子块列表, 位置列表, 权重矩阵)
        """
        from src.inference import split_image
        
        tile_size = (self.config.tile_size, self.config.tile_size)
        overlap_ratio = self.config.overlap_ratio
        
        return split_image(image, tile_size, overlap_ratio)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略指标
        
        Returns:
            Dict: 策略指标
        """
        return {
            'tile_size': self.config.tile_size,
            'overlap': self.config.overlap,
            'overlap_ratio': self.config.overlap_ratio,
            'weight_fusion': self.config.weight_fusion,
            'stats': self._stats.copy(),
            'active': self._highres_instance is not None
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._stats = {
            'total_tiles': 0,
            'total_images': 0,
            'avg_tiles_per_image': 0.0
        }
    
    def estimate_tiles(self, image_width: int, image_height: int) -> int:
        """估算图像分块数量
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            int: 估算的分块数量
        """
        tile_size = self.config.tile_size
        overlap = int(tile_size * self.config.overlap_ratio)
        step = tile_size - overlap
        
        tiles_x = max(1, (image_width - overlap + step - 1) // step)
        tiles_y = max(1, (image_height - overlap + step - 1) // step)
        
        return tiles_x * tiles_y
