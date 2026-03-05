#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块单元测试
"""

import pytest
import logging
import os
from pathlib import Path
from utils.logger import LoggerConfig, get_logger


class TestLoggerConfig:
    """LoggerConfig 类测试"""
    
    def test_setup_logger_default(self):
        """测试设置默认日志记录器"""
        logger = LoggerConfig.setup_logger("test_default")
        
        assert logger.name == "test_default"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_setup_logger_with_level(self):
        """测试设置不同日志级别"""
        logger = LoggerConfig.setup_logger("test_debug", level="debug")
        assert logger.level == logging.DEBUG
        
        logger = LoggerConfig.setup_logger("test_warning", level="warning")
        assert logger.level == logging.WARNING
        
        logger = LoggerConfig.setup_logger("test_error", level="error")
        assert logger.level == logging.ERROR
    
    def test_setup_logger_with_file(self, tmp_path):
        """测试设置文件日志输出"""
        log_file = tmp_path / "test.log"
        logger = LoggerConfig.setup_logger(
            "test_file",
            log_file=str(log_file)
        )
        
        assert len(logger.handlers) == 2  # 控制台 + 文件
        assert os.path.exists(log_file)
        
        # 测试日志写入
        logger.info("Test message")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Test message" in content
    
    def test_setup_logger_custom_format(self):
        """测试自定义日志格式"""
        custom_format = "%(levelname)s - %(message)s"
        logger = LoggerConfig.setup_logger(
            "test_format",
            format_str=custom_format
        )
        
        handler = logger.handlers[0]
        assert handler.formatter._fmt == custom_format
    
    def test_get_logger(self):
        """测试获取日志记录器"""
        logger = get_logger("ascend_inference")
        assert logger.name == "ascend_inference"
        assert isinstance(logger, logging.Logger)
    
    def test_logger_singleton(self):
        """测试日志记录器单例特性"""
        logger1 = LoggerConfig.setup_logger("test_singleton")
        logger2 = LoggerConfig.setup_logger("test_singleton")
        
        assert logger1 is logger2  # 应该返回同一个实例
    
    def test_logger_output(self, caplog):
        """测试日志输出"""
        logger = LoggerConfig.setup_logger("test_output", level="debug")
        
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
        
        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text


class TestLoggerIntegration:
    """日志集成测试"""
    
    def test_logger_with_config(self, tmp_path):
        """测试日志与配置集成"""
        from config import Config
        
        # 创建配置
        config = Config(
            log_level="debug",
            enable_logging=True
        )
        
        # 设置日志
        log_file = tmp_path / "config_test.log"
        logger = LoggerConfig.setup_logger(
            "ascend_inference",
            level=config.log_level,
            log_file=str(log_file) if config.enable_logging else None
        )
        
        assert logger.level == logging.DEBUG
        
        # 记录日志
        logger.info("Inference started")
        logger.debug("Debug info")
        
        # 验证文件日志
        assert log_file.exists()
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Inference started" in content
    
    def test_multiple_loggers(self):
        """测试多个日志记录器"""
        logger1 = LoggerConfig.setup_logger("logger1", level="info")
        logger2 = LoggerConfig.setup_logger("logger2", level="debug")
        
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
        assert logger1.level == logging.INFO
        assert logger2.level == logging.DEBUG
        
        # 不同的日志记录器应该独立
        assert logger1 is not logger2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
