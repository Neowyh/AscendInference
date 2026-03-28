#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模块

提供统一的模型推理功能，支持：
- 标准推理
- 多线程推理
- 流水线推理
- 高分辨率图像分块推理
"""

from .base import Inference
from .preprocessor import Preprocessor
from .executor import Executor
from .postprocessor import Postprocessor, split_image, merge_results
from .multithread import MultithreadInference
from .pipeline import PipelineInference
from .high_res import HighResInference

try:
    import acl
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

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
    'HAS_ACL',
]
