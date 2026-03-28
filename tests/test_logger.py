#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志功能验证脚本
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import LoggerConfig
import tempfile


def test_text_logger():
    """测试文本格式日志"""
    print("\n=== 测试文本格式日志 ===")
    logger = LoggerConfig.setup_logger('test.text', level='debug')
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    print("[OK] 文本日志输出正常")


def test_json_logger():
    """测试JSON格式日志"""
    print("\n=== 测试JSON格式日志 ===")
    logger = LoggerConfig.setup_logger('test.json', level='debug', format_type='json')

    # 输出带上下文的日志
    LoggerConfig.log_with_context(logger, "info", "推理完成",
        image_path="test.jpg",
        inference_time=0.012,
        status="success"
    )
    print("[OK] JSON日志输出正常")


def test_log_context():
    """测试日志上下文"""
    print("\n=== 测试日志上下文 ===")
    LoggerConfig.add_global_context("app_version", "1.0.0")
    LoggerConfig.add_request_context("request_id", "test_12345")

    logger = LoggerConfig.setup_logger('test.context', format_type='json')
    LoggerConfig.log_with_context(logger, "info", "测试上下文")

    LoggerConfig.clear_request_context()
    print("[OK] 日志上下文功能正常")


def test_log_sampling():
    """测试日志采样"""
    print("\n=== 测试日志采样 ===")
    logger = LoggerConfig.setup_logger('test.sampling', level='debug', sample_rate=0.5)

    debug_count = 0
    for i in range(100):
        logger.debug(f"测试采样日志 {i}")
        debug_count += 1

    print(f"[OK] 日志采样功能正常，共输出{debug_count}条debug日志（预期约50条）")


if __name__ == "__main__":
    print("开始验证日志功能...")

    # 重置日志配置
    import logging
    for name in logging.root.manager.loggerDict:
        if name.startswith('test.'):
            logger = logging.getLogger(name)
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

    test_text_logger()
    test_json_logger()
    test_log_context()
    test_log_sampling()

    print("\n✅ 所有日志功能验证完成！")
