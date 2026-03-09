#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils 模块

提供工具函数
"""

from utils.profiler import profile_context, profile_decorator, profile_func, profile
from utils.logger import LoggerConfig, get_logger
from utils.memory_pool import MemoryPool, MultiSizeMemoryPool
from utils.exceptions import (
    InferenceError,
    ModelLoadError,
    DeviceError,
    PreprocessError,
    PostprocessError,
    ConfigurationError
)

try:
    from utils.acl_utils import (
        init_acl,
        destroy_acl,
        load_model,
        unload_model,
        malloc_device,
        malloc_host,
        free_device,
        free_host
    )
    HAS_ACL = True
except ImportError:
    HAS_ACL = False
    init_acl = None
    destroy_acl = None
    load_model = None
    unload_model = None
    malloc_device = None
    malloc_host = None
    free_device = None
    free_host = None

__all__ = [
    'init_acl',
    'destroy_acl',
    'load_model',
    'unload_model',
    'malloc_device',
    'malloc_host',
    'free_device',
    'free_host',
    'HAS_ACL',
    'profile_context',
    'profile_decorator',
    'profile_func',
    'profile',
    'LoggerConfig',
    'get_logger',
    'MemoryPool',
    'MultiSizeMemoryPool',
    'InferenceError',
    'ModelLoadError',
    'DeviceError',
    'PreprocessError',
    'PostprocessError',
    'ConfigurationError'
]
