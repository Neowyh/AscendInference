#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块

提供统一的日志管理功能，支持：
- 可配置的日志级别
- 统一的日志格式（文本/JSON结构化）
- 支持输出到文件和控制台
- 日志采样功能，避免高负载下日志过多
- 上下文日志支持，便于问题追踪
- 便于生产环境调试
"""

import logging
import sys
import json
import time
import random
from typing import Optional, Dict, Any, Union
from pathlib import Path
from threading import local

# 线程本地存储，用于保存请求上下文
_thread_local = local()


class JsonFormatter(logging.Formatter):
    """JSON结构化日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 添加线程上下文信息
        if hasattr(_thread_local, "log_context"):
            log_data.update(_thread_local.log_context)

        # 添加自定义字段
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


class SamplingFilter(logging.Filter):
    """日志采样过滤器"""

    def __init__(self, sample_rate: float = 1.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.rng = random.Random(time.time())

    def filter(self, record: logging.LogRecord) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if record.levelno >= logging.ERROR:
            return True  # 错误级别日志总是输出
        return self.rng.random() < self.sample_rate


class LoggerConfig:
    """日志配置类"""

    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    JSON_FORMAT = '%(message)s'  # JSON格式由JsonFormatter处理
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    # 日志级别使用规范
    LEVEL_SPEC = {
        "DEBUG": "调试信息，开发阶段使用，生产环境默认关闭",
        "INFO": "正常运行信息，记录关键流程节点",
        "WARNING": "警告信息，不影响正常运行但需要关注",
        "ERROR": "错误信息，单个请求或操作失败",
        "CRITICAL": "严重错误，系统级故障，需要立即处理"
    }

    _global_sample_rate = 1.0
    _global_log_context: Dict[str, Any] = {}
    
    @staticmethod
    def setup_logger(
        name: str = 'ascend_inference',
        level: str = 'info',
        log_file: Optional[str] = None,
        format_type: str = 'text',  # 'text' 或 'json'
        sample_rate: float = 1.0,
        format_str: Optional[str] = None
    ) -> logging.Logger:
        """设置并返回日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
            log_file: 日志文件路径，None 则只输出到控制台
            format_type: 日志格式类型，'text' 或 'json'
            sample_rate: 日志采样率，0-1之间，默认1.0即全部采样，ERROR及以上级别总是输出
            format_str: 自定义日志格式字符串，仅对text格式有效

        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False

        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for log_filter in list(logger.filters):
            logger.removeFilter(log_filter)

        # 选择格式化器
        if format_type == 'json':
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                format_str or LoggerConfig.DEFAULT_FORMAT,
                datefmt=LoggerConfig.DATE_FORMAT
            )

        # 添加采样过滤器
        if sample_rate < 1.0:
            sampling_filter = SamplingFilter(sample_rate)
            logger.addFilter(sampling_filter)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def set_global_sample_rate(rate: float) -> None:
        """设置全局日志采样率"""
        LoggerConfig._global_sample_rate = max(0.0, min(1.0, rate))

    @staticmethod
    def add_global_context(key: str, value: Any) -> None:
        """添加全局日志上下文，所有日志都会包含这些字段"""
        LoggerConfig._global_log_context[key] = value
        _thread_local.log_context = LoggerConfig._global_log_context.copy()

    @staticmethod
    def add_request_context(key: str, value: Any) -> None:
        """添加请求级日志上下文，仅当前线程生效"""
        if not hasattr(_thread_local, "log_context"):
            _thread_local.log_context = LoggerConfig._global_log_context.copy()
        _thread_local.log_context[key] = value

    @staticmethod
    def clear_request_context() -> None:
        """清空当前线程的请求上下文"""
        if hasattr(_thread_local, "log_context"):
            delattr(_thread_local, "log_context")

    @staticmethod
    def log_with_context(logger: logging.Logger, level: str, message: str, **kwargs) -> None:
        """输出带自定义字段的日志

        Args:
            logger: 日志记录器实例
            level: 日志级别
            message: 日志消息
            **kwargs: 自定义字段，会包含在结构化日志中
        """
        log_method = getattr(logger, level.lower(), logger.info)
        if kwargs:
            extra = {"extra_fields": kwargs}
            log_method(message, extra=extra)
        else:
            log_method(message)


def get_logger(name: str = 'ascend_inference') -> logging.Logger:
    """获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)
