#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块

提供统一的日志管理功能，支持：
- 可配置的日志级别
- 统一的日志格式
- 支持输出到文件和控制台
- 便于生产环境调试
"""

import logging
import sys
from typing import Optional
from pathlib import Path


class LoggerConfig:
    """日志配置类"""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    @staticmethod
    def setup_logger(
        name: str = 'ascend_inference',
        level: str = 'info',
        log_file: Optional[str] = None,
        format_str: Optional[str] = None
    ) -> logging.Logger:
        """设置并返回日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
            log_file: 日志文件路径，None 则只输出到控制台
            format_str: 日志格式字符串，None 则使用默认格式
            
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(name)
        
        if logger.handlers:
            return logger
        
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        formatter = logging.Formatter(
            format_str or LoggerConfig.DEFAULT_FORMAT,
            datefmt=LoggerConfig.DATE_FORMAT
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


def get_logger(name: str = 'ascend_inference') -> logging.Logger:
    """获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    return logging.getLogger(name)
