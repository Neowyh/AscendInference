#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能评测工具

提供简洁的性能测量功能
"""

import time
from functools import wraps
from contextlib import contextmanager


@contextmanager
def profile(name=""):
    """性能分析上下文管理器
    
    Args:
        name: 性能分析名称
        
    Yields:
        None
        
    Example:
        with profile("预处理"):
            # 要测量的代码
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[{name}] 耗时：{elapsed:.4f} 秒")


def profile_decorator(name=None):
    """性能分析装饰器
    
    Args:
        name: 性能分析名称，None 则使用函数名
        
    Returns:
        装饰器函数
        
    Example:
        @profile_decorator("推理")
        def inference():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                print(f"[{func_name}] 耗时：{elapsed:.4f} 秒")
        return wrapper
    return decorator


profile = profile_decorator
