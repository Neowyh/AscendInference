#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils 模块

提供工具函数
"""

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

from utils.profiler import profile

__all__ = [
    'init_acl',
    'destroy_acl',
    'load_model',
    'unload_model',
    'malloc_device',
    'malloc_host',
    'free_device',
    'free_host',
    'profile'
]
