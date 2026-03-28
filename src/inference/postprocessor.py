#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理器模块

提供推理结果后处理功能，支持：
- 结果解析
- 结果转换
- 高分辨率图像融合
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from utils.logger import LoggerConfig, get_logger

logger = LoggerConfig.setup_logger('ascend_inference.postprocessor', format_type='text')


class Postprocessor:
    """推理结果后处理器"""
    
    def __init__(self):
        """初始化后处理器"""
        pass
    
    def process(self, output: np.ndarray) -> np.ndarray:
        """处理单张推理结果
        
        Args:
            output: 原始推理输出
            
        Returns:
            np.ndarray: 处理后的结果
        """
        return output
    
    def process_batch(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        """批量处理推理结果
        
        Args:
            outputs: 原始推理输出列表
            
        Returns:
            List[np.ndarray]: 处理后的结果列表
        """
        return [self.process(o) for o in outputs]


def create_hann_window(tile_size: Tuple[int, int]) -> np.ndarray:
    """创建汉宁窗权重矩阵
    
    Args:
        tile_size: 子块大小 (height, width)
        
    Returns:
        np.ndarray: 汉宁窗权重矩阵
    """
    tile_h, tile_w = tile_size
    return np.outer(np.hanning(tile_h), np.hanning(tile_w)).astype(np.float32)


def split_image(
    image: np.ndarray,
    tile_size: Tuple[int, int],
    overlap: float
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray]:
    """将图像划分为带重叠的子块（支持权重融合）
    
    Args:
        image: 输入图像
        tile_size: 子块大小 (height, width)
        overlap: 重叠比例
        
    Returns:
        tiles: 子块列表
        positions: 子块位置列表 (x1, y1, w, h)
        weight_map: 权重矩阵，用于结果融合时消除重叠边缘效应
    """
    h, w = image.shape[:2]
    tile_h, tile_w = tile_size
    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)
    step_h = tile_h - overlap_h
    step_w = tile_w - overlap_w

    tiles = []
    positions = []
    weight_map = np.zeros((h, w), dtype=np.float32)

    hann_2d = create_hann_window((tile_h, tile_w))

    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            x1 = x
            y1 = y
            x2 = min(x + tile_w, w)
            y2 = min(y + tile_h, h)

            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)
            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)

            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
            positions.append((x1, y1, x2 - x1, y2 - y1))

            weight_map[y1:y2, x1:x2] += hann_2d[:y2-y1, :x2-x1]

    weight_map[weight_map < 1e-6] = 1.0
    return tiles, positions, weight_map


def merge_results(
    results: List[Tuple[int, Optional[np.ndarray]]],
    positions: List[Tuple[int, int, int, int]],
    image_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    """合并高分辨率图像的推理结果
    
    Args:
        results: 推理结果列表，每个元素为 (tile_id, result)
        positions: 子块位置列表
        image_shape: 原始图像形状
        
    Returns:
        dict: 合并后的结果
    """
    merged_result = {
        "sub_results": [],
        "image_shape": image_shape,
        "num_tiles": len(positions)
    }
    
    for tile_id, result in sorted(results, key=lambda x: x[0]):
        if result is not None:
            x, y, w, h = positions[tile_id]
            merged_result["sub_results"].append({
                "position": (x, y, w, h),
                "result": result.tolist()[:10]
            })
    
    return merged_result
