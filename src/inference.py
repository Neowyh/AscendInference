#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理核心类（向后兼容模块）

本模块保留原有导入路径，从拆分后的模块导入所有类和函数。
所有功能已迁移到 src/inference/ 目录下的各个模块中。

模块结构：
- base.py: 基础推理类 Inference
- preprocessor.py: 图像预处理器 Preprocessor
- executor.py: 推理执行器 Executor
- postprocessor.py: 后处理器 Postprocessor, split_image, merge_results
- multithread.py: 多线程推理 MultithreadInference
- pipeline.py: 流水线推理 PipelineInference
- high_res.py: 高分辨率推理 HighResInference
"""

from src.inference import (
    Inference,
    Preprocessor,
    Executor,
    Postprocessor,
    split_image,
    merge_results,
    MultithreadInference,
    PipelineInference,
    HighResInference,
)

__all__ = [
    'Inference',
    'Preprocessor',
    'Executor',
    'Postprocessor',
    'split_image',
    'merge_results',
    'MultithreadInference',
    'PipelineInference',
    'HighResInference',
]
