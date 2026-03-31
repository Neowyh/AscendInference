#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成单元测试

包含所有模块的单元测试：
- Config 配置模块
- Inference 推理模块
- API 接口模块
- Logger 日志模块

运行方式:
    python -m pytest tests/test_all.py -v
    python -m pytest tests/test_all.py -v --tb=short
    python -m pytest tests/test_all.py::TestConfig -v  # 只运行 Config 测试
"""

import pytest
import logging
import os
import queue
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Config 配置模块测试
# ============================================================================

class TestConfig:
    """Config 类测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        from config import Config
        
        config = Config()
        assert config.model_path == "models/yolov8s.om"
        assert config.device_id == 0
        assert config.resolution == "640x640"
        assert config.tile_size == 640
        assert config.overlap == 100
        assert config.num_threads == 4
        assert config.backend == "pil"
        assert config.conf_threshold == 0.4
        assert config.iou_threshold == 0.5
        assert config.max_detections == 100
        assert config.enable_logging is True
        assert config.log_level == "info"
        assert config.enable_profiling is False
    
    def test_from_json(self, tmp_path):
        """测试从 JSON 文件加载配置"""
        from config import Config
        import json
        
        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "model_path": "test_model.om",
                "device_id": 1,
                "resolution": "1k"
            }, f)
        
        config = Config.from_json(str(config_file))
        assert config.model_path == "test_model.om"
        assert config.device_id == 1
        assert config.resolution == "1k"
    
    def test_from_json_nonexistent_file(self):
        """测试加载不存在的 JSON 文件"""
        from config import Config
        
        config = Config.from_json("nonexistent.json")
        assert config.model_path == "models/yolov8s.om"
        assert config.device_id == 0
    
    def test_apply_overrides(self):
        """测试应用命令行参数覆盖"""
        from config import Config
        
        config = Config()
        config.apply_overrides(
            device_id=2,
            resolution="2k",
            backend="opencv"
        )
        assert config.device_id == 2
        assert config.resolution == "2k"
        assert config.backend == "opencv"
    
    def test_get_resolution(self):
        """测试获取分辨率"""
        from config import Config
        
        assert Config.get_resolution("640x640") == (640, 640)
        assert Config.get_resolution("1k") == (1024, 1024)
        assert Config.get_resolution("2k") == (2048, 2048)
        assert Config.get_resolution("4k") == (4096, 4096)
        assert Config.get_resolution("unknown") == (640, 640)
    
    def test_is_supported_resolution(self):
        """测试分辨率是否支持"""
        from config import Config
        
        assert Config.is_supported_resolution("640x640") is True
        assert Config.is_supported_resolution("1k") is True
        assert Config.is_supported_resolution("unknown") is False
    
    def test_max_ai_cores(self):
        """测试 AI 核心数常量"""
        from config import MAX_AI_CORES
        assert MAX_AI_CORES == 4
    
    def test_supported_resolutions(self):
        """测试支持的分辨率列表"""
        from config import SUPPORTED_RESOLUTIONS
        
        assert "640x640" in SUPPORTED_RESOLUTIONS
        assert "1k" in SUPPORTED_RESOLUTIONS
        assert "2k" in SUPPORTED_RESOLUTIONS
        assert "4k" in SUPPORTED_RESOLUTIONS
        assert len(SUPPORTED_RESOLUTIONS) == 9


# ============================================================================
# Inference 推理模块测试
# ============================================================================

try:
    from src.inference import Inference, MultithreadInference, HighResInference
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False


@pytest.mark.skipif(not HAS_INFERENCE, reason="推理模块不可用")
class TestInference:
    """Inference 类测试"""
    
    def test_init_default_config(self):
        """测试初始化使用默认配置"""
        inference = Inference()
        assert inference.config is not None
        assert inference.model_path == "models/yolov8s.om"
        assert inference.device_id == 0
    
    def test_init_custom_config(self):
        """测试初始化使用自定义配置"""
        from config import Config
        
        config = Config(
            model_path="test.om",
            device_id=1,
            resolution="1k"
        )
        inference = Inference(config)
        assert inference.config == config
        assert inference.model_path == "test.om"
        assert inference.device_id == 1
    
    def test_init_resolution(self):
        """测试分辨率解析"""
        from config import Config
        
        config = Config(resolution="1k")
        inference = Inference(config)
        assert inference.input_width == 1024
        assert inference.input_height == 1024
    
    def test_context_manager_enter(self):
        """测试上下文管理器入口"""
        with patch.object(Inference, 'init', return_value=True):
            inference = Inference()
            with inference as inf:
                assert inf == inference
    
    def test_context_manager_exit(self):
        """测试上下文管理器出口"""
        with patch.object(Inference, 'init', return_value=True):
            with patch.object(Inference, 'destroy') as mock_destroy:
                inference = Inference()
                with inference:
                    pass
                assert mock_destroy.called
    
    @patch('src.inference.HAS_ACL', False)
    def test_init_no_acl(self):
        """测试无 ACL 库时的初始化"""
        inference = Inference()
        from utils.exceptions import ACLError

        with pytest.raises(ACLError):
            inference.init()
    
    @patch('src.inference.HAS_ACL', False)
    def test_preprocess_no_acl(self):
        """测试无 ACL 库时的预处理"""
        inference = Inference()
        from utils.exceptions import ACLError

        with patch('src.inference.base.validate_file_path'):
            with pytest.raises(ACLError):
                inference.preprocess("test.jpg")
    
    @patch('src.inference.HAS_ACL', False)
    def test_execute_no_acl(self):
        """测试无 ACL 库时的推理执行"""
        inference = Inference()
        from utils.exceptions import ACLError

        with pytest.raises(ACLError):
            inference.execute()


@pytest.mark.skipif(not HAS_INFERENCE, reason="推理模块不可用")
class TestMultithreadInference:
    """MultithreadInference 类测试"""
    
    def test_init_default_config(self):
        """测试初始化默认配置"""
        mt_inference = MultithreadInference()
        assert mt_inference.config is not None
        assert mt_inference.num_threads <= 4  # MAX_AI_CORES
    
    def test_init_custom_config(self):
        """测试初始化自定义配置"""
        from config import Config
        
        config = Config(num_threads=8)
        mt_inference = MultithreadInference(config)
        assert mt_inference.num_threads <= 4  # 会被限制在 MAX_AI_CORES
    
    def test_task_queue(self):
        """测试任务队列"""
        mt_inference = MultithreadInference()
        mt_inference.task_queues = [queue.Queue()]
        with patch('src.inference.multithread.validate_file_path'):
            mt_inference.add_task("test.jpg", "pil")
        assert mt_inference.task_queues
        task = mt_inference.task_queues[0].get()
        assert task == ("test.jpg", "pil")


@pytest.mark.skipif(not HAS_INFERENCE, reason="推理模块不可用")
class TestHighResInference:
    """HighResInference 类测试"""
    
    def test_init_default_config(self):
        """测试初始化默认配置"""
        hr_inference = HighResInference()
        assert hr_inference.config is not None
        assert hr_inference.tile_size == (640, 640)
    
    def test_init_custom_config(self):
        """测试初始化自定义配置"""
        from config import Config
        
        config = Config(tile_size=512, overlap=50)
        hr_inference = HighResInference(config)
        assert hr_inference.tile_size == (512, 512)


# ============================================================================
# API 接口模块测试
# ============================================================================

try:
    from src.api import InferenceAPI
    HAS_API = True
except ImportError:
    HAS_API = False


@pytest.mark.skipif(not HAS_API, reason="API 模块不可用")
class TestInferenceAPI:
    """InferenceAPI 类测试"""
    
    @patch('src.api.HAS_INFERENCE', False)
    def test_inference_image_no_inference_module(self):
        """测试无推理模块时的异常"""
        with patch('src.api.validate_file_path'):
            with pytest.raises(ImportError, match="推理模块不可用"):
                InferenceAPI.inference_image('base', 'test.jpg')
    
    @patch('src.api.HAS_INFERENCE', False)
    def test_inference_batch_no_inference_module(self):
        """测试无推理模块时的批量推理异常"""
        with patch('src.api.validate_file_path'):
            with pytest.raises(ImportError, match="推理模块不可用"):
                InferenceAPI.inference_batch('base', ['test.jpg'])
    
    @patch('src.api.HAS_INFERENCE', True)
    @patch('src.api.Inference')
    def test_inference_image_base_mode(self, mock_inference_class):
        """测试 base 模式推理"""
        mock_inference = Mock()
        mock_inference.__enter__ = Mock(return_value=mock_inference)
        mock_inference.__exit__ = Mock(return_value=None)
        mock_inference.run_inference = Mock(return_value=np.array([1, 2, 3]))
        mock_inference_class.return_value = mock_inference
        
        from config import Config
        config = Config()
        with patch('src.api.validate_file_path'):
            result = InferenceAPI.inference_image('base', 'test.jpg', config)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_inference.run_inference.assert_called_once()
    
    @patch('src.api.HAS_INFERENCE', True)
    @patch('src.api.MultithreadInference')
    def test_inference_image_multithread_mode(self, mock_mt_inference_class):
        """测试 multithread 模式推理"""
        mock_mt_inference = Mock()
        mock_mt_inference.start = Mock(return_value=True)
        mock_mt_inference.add_task = Mock()
        mock_mt_inference.wait_completion = Mock()
        mock_mt_inference.get_results = Mock(return_value=[('test.jpg', np.array([1, 2, 3]))])
        mock_mt_inference.stop = Mock()
        mock_mt_inference_class.return_value = mock_mt_inference
        
        from config import Config
        config = Config()
        with patch('src.api.validate_file_path'):
            result = InferenceAPI.inference_image('multithread', 'test.jpg', config)
        
        assert result is not None
        mock_mt_inference.start.assert_called_once()
        mock_mt_inference.add_task.assert_called_once()
        mock_mt_inference.wait_completion.assert_called_once()
        mock_mt_inference.stop.assert_called_once()
    
    @patch('src.api.HAS_INFERENCE', True)
    @patch('src.api.HighResInference')
    def test_inference_image_high_res_mode(self, mock_hr_inference_class):
        """测试 high_res 模式推理"""
        mock_hr_inference = Mock()
        mock_hr_inference.process_image = Mock(return_value={"result": "test"})
        mock_hr_inference_class.return_value = mock_hr_inference
        
        from config import Config
        config = Config()
        with patch('src.api.validate_file_path'):
            result = InferenceAPI.inference_image('high_res', 'test.jpg', config)
        
        assert result is not None
        mock_hr_inference.process_image.assert_called_once()
    
    @patch('src.api.HAS_INFERENCE', True)
    @patch('src.api.Inference')
    def test_inference_batch_base_mode(self, mock_inference_class):
        """测试 base 模式批量推理"""
        mock_inference = Mock()
        mock_inference.init = Mock(return_value=True)
        mock_inference.run_inference = Mock(return_value=np.array([1, 2, 3]))
        mock_inference.destroy = Mock()
        mock_inference_class.return_value = mock_inference
        
        from config import Config
        config = Config()
        image_paths = ['test1.jpg', 'test2.jpg']
        with patch('src.api.validate_file_path'):
            results = InferenceAPI.inference_batch('base', image_paths, config)
        
        assert len(results) == 2
        assert mock_inference.init.called
        assert mock_inference.run_inference.call_count == 2
        assert mock_inference.destroy.called
    
    @patch('src.api.HAS_INFERENCE', True)
    @patch('src.api.MultithreadInference')
    def test_inference_batch_multithread_mode(self, mock_mt_inference_class):
        """测试 multithread 模式批量推理"""
        mock_mt_inference = Mock()
        mock_mt_inference.start = Mock(return_value=True)
        mock_mt_inference.add_task = Mock()
        mock_mt_inference.wait_completion = Mock()
        mock_mt_inference.get_results = Mock(return_value=[
            ('test1.jpg', np.array([1, 2, 3])),
            ('test2.jpg', np.array([4, 5, 6]))
        ])
        mock_mt_inference.stop = Mock()
        mock_mt_inference_class.return_value = mock_mt_inference
        
        from config import Config
        config = Config()
        image_paths = ['test1.jpg', 'test2.jpg']
        with patch('src.api.validate_file_path'):
            results = InferenceAPI.inference_batch('multithread', image_paths, config)
        
        assert len(results) == 2
        mock_mt_inference.start.assert_called_once()
        assert mock_mt_inference.add_task.call_count == 2
        mock_mt_inference.stop.assert_called_once()
    
    def test_default_config(self):
        """测试使用默认配置"""
        with patch('src.api.HAS_INFERENCE', True):
            with patch('src.api.Inference') as mock_inference_class:
                mock_inference = Mock()
                mock_inference.__enter__ = Mock(return_value=mock_inference)
                mock_inference.__exit__ = Mock(return_value=None)
                mock_inference.run_inference = Mock(return_value=np.array([1, 2, 3]))
                mock_inference_class.return_value = mock_inference
                
                with patch('src.api.validate_file_path'):
                    result = InferenceAPI.inference_image('base', 'test.jpg')
                assert result is not None


# ============================================================================
# Logger 日志模块测试
# ============================================================================

class TestLoggerConfig:
    """LoggerConfig 类测试"""
    
    def test_setup_logger_default(self):
        """测试设置默认日志记录器"""
        from utils.logger import LoggerConfig
        
        logger = LoggerConfig.setup_logger("test_default")
        
        assert logger.name == "test_default"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_setup_logger_with_level(self):
        """测试设置不同日志级别"""
        from utils.logger import LoggerConfig
        
        logger = LoggerConfig.setup_logger("test_debug", level="debug")
        assert logger.level == logging.DEBUG
        
        logger = LoggerConfig.setup_logger("test_warning", level="warning")
        assert logger.level == logging.WARNING
        
        logger = LoggerConfig.setup_logger("test_error", level="error")
        assert logger.level == logging.ERROR
    
    def test_setup_logger_with_file(self, tmp_path):
        """测试设置文件日志输出"""
        from utils.logger import LoggerConfig
        
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
        from utils.logger import LoggerConfig
        
        custom_format = "%(levelname)s - %(message)s"
        logger = LoggerConfig.setup_logger(
            "test_format",
            format_str=custom_format
        )
        
        handler = logger.handlers[0]
        assert handler.formatter._fmt == custom_format
    
    def test_get_logger(self):
        """测试获取日志记录器"""
        from utils.logger import get_logger
        
        logger = get_logger("ascend_inference")
        assert logger.name == "ascend_inference"
        assert isinstance(logger, logging.Logger)
    
    def test_logger_singleton(self):
        """测试日志记录器单例特性"""
        from utils.logger import LoggerConfig
        
        logger1 = LoggerConfig.setup_logger("test_singleton")
        logger2 = LoggerConfig.setup_logger("test_singleton")
        
        assert logger1 is logger2  # 应该返回同一个实例
    
    def test_logger_output(self, capsys):
        """测试日志输出"""
        from utils.logger import LoggerConfig
        
        logger = LoggerConfig.setup_logger("test_output", level="debug")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        captured = capsys.readouterr().out
        assert "Debug message" in captured
        assert "Info message" in captured
        assert "Warning message" in captured
        assert "Error message" in captured


class TestLoggerIntegration:
    """日志集成测试"""
    
    def test_logger_with_config(self, tmp_path):
        """测试日志与配置集成"""
        from utils.logger import LoggerConfig
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
        from utils.logger import LoggerConfig
        
        logger1 = LoggerConfig.setup_logger("logger1", level="info")
        logger2 = LoggerConfig.setup_logger("logger2", level="debug")
        
        assert logger1.name == "logger1"
        assert logger2.name == "logger2"
        assert logger1.level == logging.INFO
        assert logger2.level == logging.DEBUG
        
        # 不同的日志记录器应该独立
        assert logger1 is not logger2


def run_mock_standard_evaluation(tmp_path):
    from benchmark.reporters import render_report
    from benchmark.scenarios import BenchmarkResult, ModelInfo
    from reporting.archive import archive_result

    results = [
        BenchmarkResult(
            scenario_name="model_selection",
            model_info=ModelInfo(name="yolov8n.om", resolution="640x640"),
            metrics={
                "execute": {"avg": 12.0},
                "fps": {"pure": 83.3, "e2e": 55.5},
                "iterations": {"test": 10},
            },
            config={"input_tier": "720p"},
        )
    ]

    report, report_model, report_extension = render_report(
        results,
        task_name="model_selection",
        output_format="text",
    )
    archived = archive_result(
        tmp_path,
        {"task_name": "model_selection", "route_type": "standard"},
        report,
        report_model,
        report_extension=report_extension,
    )

    return {
        "report_path": str(archived["report_path"]),
        "archive_dir": str(archived["archive_dir"]),
        "raw_results": len(report_model["results"]),
    }


def test_end_to_end_standard_evaluation_produces_archive_metadata(tmp_path):
    result = run_mock_standard_evaluation(tmp_path)

    assert result["report_path"].endswith(".md")
    assert result["raw_results"] == 1


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
