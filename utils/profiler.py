#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能评测工具

提供简洁的性能测量功能
"""

import time
from functools import wraps
from contextlib import contextmanager
from typing import Callable, Any, Optional

try:
    from utils.logger import LoggerConfig
    _default_logger = LoggerConfig.setup_logger('ascend_inference.profiler')
except Exception:
    _default_logger = None


@contextmanager
def profile_context(name: str = "", logger=None):
    """性能分析上下文管理器
    
    Args:
        name: 性能分析名称
        logger: 日志记录器，None 则使用默认 logger
        
    Yields:
        None
        
    Example:
        with profile_context("预处理"):
            # 要测量的代码
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    msg = f"[{name}] 耗时：{elapsed:.4f} 秒"
    if logger:
        logger.info(msg)
    elif _default_logger:
        _default_logger.info(msg)
    else:
        print(msg)


def profile_decorator(name: Optional[str] = None, logger=None) -> Callable:
    """性能分析装饰器
    
    Args:
        name: 性能分析名称，None 则使用函数名
        logger: 日志记录器，None 则使用默认 logger
        
    Returns:
        装饰器函数
        
    Example:
        @profile_decorator("推理")
        def inference():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = name or func.__name__
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                msg = f"[{func_name}] 耗时：{elapsed:.4f} 秒"
                if logger:
                    logger.info(msg)
                elif _default_logger:
                    _default_logger.info(msg)
                else:
                    print(msg)
        return wrapper
    return decorator


def profile_func(name: Optional[str] = None, logger=None):
    """性能分析装饰器（兼容性别名）
    
    Args:
        name: 性能分析名称，None 则使用函数名
        logger: 日志记录器，None 则使用默认 logger
        
    Returns:
        装饰器函数
    """
    return profile_decorator(name, logger)


# 为了兼容性，profile 可以作为上下文管理器或装饰器使用
profile = profile_context
