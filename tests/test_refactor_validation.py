#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构验证测试模块

验证本轮重构更改的正确性，包括：
- 推理模块拆分验证
- 异常处理统一验证
- 配置验证器验证
- 推理池验证
- 自适应批处理验证
- 并行预处理器验证
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInferenceModuleSplit:
    """验证推理模块拆分"""
    
    def test_inference_module_imports(self):
        """测试推理模块导入"""
        from src.inference import (
            Inference,
            Preprocessor,
            Executor,
            Postprocessor,
            split_image,
            merge_results,
            MultithreadInference,
            PipelineInference,
            HighResInference,
        )
        
        assert Inference is not None
        assert Preprocessor is not None
        assert Executor is not None
        assert Postprocessor is not None
        assert MultithreadInference is not None
        assert PipelineInference is not None
        assert HighResInference is not None
    
    def test_backward_compatibility(self):
        """测试向后兼容性 - 原有导入路径"""
        from src.inference import Inference as InferenceFromNew
        
        assert InferenceFromNew is not None
    
    def test_preprocessor_class(self):
        """测试预处理器类"""
        from src.inference.preprocessor import Preprocessor
        
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        assert preprocessor.input_width == 640
        assert preprocessor.input_height == 640
        assert preprocessor.batch_size == 1
    
    def test_executor_class(self):
        """测试执行器类"""
        from src.inference.executor import Executor
        
        executor = Executor(
            model_id=12345,
            stream=Mock(),
            input_dataset=Mock(),
            output_dataset=Mock(),
            output_buffer=0x1000,
            output_size=8400 * 4,
            batch_size=1
        )
        
        assert executor.model_id == 12345
        assert executor.batch_size == 1
    
    def test_postprocessor_class(self):
        """测试后处理器类"""
        from src.inference.postprocessor import Postprocessor
        
        postprocessor = Postprocessor()
        sample = np.random.randn(100).astype(np.float32)
        
        result = postprocessor.process(sample)
        np.testing.assert_array_equal(result, sample)


class TestExceptionHandling:
    """验证异常处理统一"""
    
    def test_exception_classes_exist(self):
        """测试异常类存在"""
        from utils.exceptions import (
            InferenceError,
            ModelLoadError,
            DeviceError,
            PreprocessError,
            PostprocessError,
            ConfigurationError,
            MemoryError,
            ACLError,
            ThreadError,
            InputValidationError,
            BenchmarkError,
            ResourceError,
        )
        
        assert issubclass(ModelLoadError, InferenceError)
        assert issubclass(DeviceError, InferenceError)
        assert issubclass(PreprocessError, InferenceError)
        assert issubclass(PostprocessError, InferenceError)
        assert issubclass(ConfigurationError, InferenceError)
        assert issubclass(ACLError, InferenceError)
        assert issubclass(ThreadError, InferenceError)
        assert issubclass(InputValidationError, InferenceError)
        assert issubclass(BenchmarkError, InferenceError)
        assert issubclass(ResourceError, InferenceError)
    
    def test_exception_message_format(self):
        """测试异常消息格式"""
        from utils.exceptions import InferenceError
        
        error = InferenceError(
            "测试错误",
            error_code=1001,
            details={"key": "value"}
        )
        
        assert error.message == "测试错误"
        assert error.error_code == 1001
        assert error.details == {"key": "value"}
        assert "[1001]" in str(error)
    
    def test_preprocess_error_on_invalid_file(self):
        """测试预处理异常 - 无效文件"""
        from src.inference.preprocessor import Preprocessor
        from utils.exceptions import PreprocessError
        
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        with pytest.raises(PreprocessError):
            preprocessor.load_image("/nonexistent/path.jpg", "opencv")


class TestConfigValidator:
    """验证配置验证器"""
    
    def test_validator_import(self):
        """测试验证器导入"""
        from config.validator import ConfigValidator, ValidationResult, validate_config
        
        assert ConfigValidator is not None
        assert ValidationResult is not None
        assert validate_config is not None
    
    def test_validation_result_dataclass(self):
        """测试验证结果数据类"""
        from config.validator import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["test warning"]
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert bool(result) is True
    
    def test_validator_supported_resolutions(self):
        """测试支持的分辨率列表"""
        from config.validator import ConfigValidator
        
        assert '640x640' in ConfigValidator.SUPPORTED_RESOLUTIONS
        assert '320x320' in ConfigValidator.SUPPORTED_RESOLUTIONS
    
    def test_validator_invalid_model_path(self):
        """测试无效模型路径验证"""
        from config.validator import ConfigValidator
        from config import Config
        
        config = Config(model_path="/nonexistent/model.om")
        result = ConfigValidator.validate(config)
        
        assert result.is_valid is False
        assert any("不存在" in e for e in result.errors)
    
    def test_validator_invalid_device_id(self):
        """测试无效设备ID验证"""
        from config.validator import ConfigValidator
        from config import Config
        
        config = Config(device_id=-1)
        result = ConfigValidator.validate(config)
        
        assert result.is_valid is False
        assert any("设备ID" in e for e in result.errors)


class TestInferencePool:
    """验证推理池"""
    
    def test_pool_import(self):
        """测试推理池导入"""
        from src.inference.pool import InferencePool, InferenceTask
        
        assert InferencePool is not None
        assert InferenceTask is not None
    
    def test_pool_initialization(self):
        """测试推理池初始化"""
        from src.inference.pool import InferencePool
        from config import Config
        
        config = Config()
        pool = InferencePool(config, pool_size=2)
        
        assert pool.pool_size == 2
        assert pool._initialized is False
        assert pool._shutdown is False
    
    def test_pool_stats(self):
        """测试推理池统计"""
        from src.inference.pool import InferencePool
        from config import Config
        
        pool = InferencePool(Config(), pool_size=4)
        stats = pool.get_stats()
        
        assert stats['pool_size'] == 4
        assert stats['initialized'] is False
        assert stats['shutdown'] is False


class TestAdaptiveBatch:
    """验证自适应批处理"""
    
    def test_adaptive_batch_import(self):
        """测试自适应批处理导入"""
        from src.strategies.adaptive_batch import (
            AdaptiveBatchQueue,
            AdaptiveBatchProcessor,
            PriorityItem
        )
        
        assert AdaptiveBatchQueue is not None
        assert AdaptiveBatchProcessor is not None
        assert PriorityItem is not None
    
    def test_priority_item(self):
        """测试优先级项"""
        from src.strategies.adaptive_batch import PriorityItem
        
        item1 = PriorityItem(priority=1, sequence=0, item="a")
        item2 = PriorityItem(priority=2, sequence=1, item="b")
        
        assert item1 < item2
    
    def test_adaptive_batch_queue(self):
        """测试自适应批处理队列"""
        from src.strategies.adaptive_batch import AdaptiveBatchQueue
        
        queue = AdaptiveBatchQueue(batch_size=4, max_batch_size=16)
        
        assert queue.batch_size == 4
        assert queue.max_batch_size == 16
        
        queue.push("item1")
        queue.push("item2")
        
        assert queue.size() == 2
    
    def test_adaptive_batch_queue_pop(self):
        """测试队列弹出"""
        from src.strategies.adaptive_batch import AdaptiveBatchQueue
        
        queue = AdaptiveBatchQueue(batch_size=2, timeout_ms=100)
        queue.push("item1", priority=1)
        queue.push("item2", priority=0)
        
        batch = queue.pop_batch(batch_size=2)
        
        assert len(batch) == 2
    
    def test_adaptive_batch_processor(self):
        """测试自适应批处理器"""
        from src.strategies.adaptive_batch import AdaptiveBatchProcessor
        
        def process_fn(items):
            return [item * 2 for item in items]
        
        processor = AdaptiveBatchProcessor(
            process_fn=process_fn,
            batch_size=2
        )
        
        assert processor.batch_size == 2
        assert processor._running is False


class TestParallelPreprocessor:
    """验证并行预处理器"""
    
    def test_parallel_preprocessor_import(self):
        """测试并行预处理器导入"""
        from src.preprocessing.parallel_preprocessor import (
            ParallelPreprocessor,
            PreprocessResult,
            PreprocessPipeline
        )
        
        assert ParallelPreprocessor is not None
        assert PreprocessResult is not None
        assert PreprocessPipeline is not None
    
    def test_preprocess_result(self):
        """测试预处理结果"""
        from src.preprocessing.parallel_preprocessor import PreprocessResult
        
        result = PreprocessResult(
            index=0,
            data=np.zeros(100),
            success=True,
            latency=0.01
        )
        
        assert result.index == 0
        assert result.success is True
        assert result.latency == 0.01
    
    def test_parallel_preprocessor_init(self):
        """测试并行预处理器初始化"""
        from src.preprocessing.parallel_preprocessor import ParallelPreprocessor
        
        preprocessor = ParallelPreprocessor(
            input_width=640,
            input_height=640,
            num_workers=4
        )
        
        assert preprocessor.input_width == 640
        assert preprocessor.input_height == 640
        assert preprocessor.num_workers == 4
    
    def test_parallel_preprocessor_stats(self):
        """测试并行预处理器统计"""
        from src.preprocessing.parallel_preprocessor import ParallelPreprocessor
        
        preprocessor = ParallelPreprocessor(
            input_width=640,
            input_height=640,
            num_workers=4
        )
        
        stats = preprocessor.get_stats()
        
        assert stats['num_workers'] == 4
        assert stats['running'] is False
        assert stats['total_processed'] == 0
    
    def test_parallel_preprocessor_batch(self):
        """测试并行批量预处理"""
        from src.preprocessing.parallel_preprocessor import ParallelPreprocessor
        
        preprocessor = ParallelPreprocessor(
            input_width=640,
            input_height=640,
            num_workers=2
        )
        
        images = [
            np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            for _ in range(4)
        ]
        
        results, failed = preprocessor.process_batch(images, backend='pil')
        
        assert len(results) == 4
        assert len(failed) == 0
        
        for result in results:
            assert result.dtype == np.float32
            assert len(result.shape) == 1


class TestImageSplitting:
    """验证图像分割功能"""
    
    def test_split_image(self):
        """测试图像分割"""
        from src.inference.postprocessor import split_image
        
        image = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        tile_size = (640, 640)
        overlap = 0.25
        
        tiles, positions, weight_map = split_image(image, tile_size, overlap)
        
        assert len(tiles) > 0
        assert len(tiles) == len(positions)
        assert weight_map.shape == image.shape[:2]
    
    def test_merge_results(self):
        """测试结果合并"""
        from src.inference.postprocessor import split_image, merge_results
        
        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        tile_size = (640, 640)
        overlap = 0.25
        
        tiles, positions, _ = split_image(image, tile_size, overlap)
        results = [(i, np.random.randn(100).astype(np.float32)) for i in range(len(tiles))]
        
        merged = merge_results(results, positions, image.shape)
        
        assert 'sub_results' in merged
        assert 'image_shape' in merged
        assert 'num_tiles' in merged


class TestScenariosExceptionHandling:
    """验证评测场景异常处理"""
    
    def test_benchmark_error_import(self):
        """测试评测异常导入"""
        from utils.exceptions import BenchmarkError
        
        assert BenchmarkError is not None
    
    def test_scenarios_import(self):
        """测试评测场景导入"""
        from benchmark.scenarios import (
            BenchmarkScenario,
            ModelSelectionScenario,
            StrategyValidationScenario,
            ExtremePerformanceScenario,
            BenchmarkResult,
            ModelInfo
        )
        
        assert BenchmarkScenario is not None
        assert ModelSelectionScenario is not None
        assert StrategyValidationScenario is not None
        assert ExtremePerformanceScenario is not None
        assert BenchmarkResult is not None
        assert ModelInfo is not None


class TestModuleLineCounts:
    """验证模块行数限制"""
    
    def test_preprocessor_line_count(self):
        """测试预处理器行数 < 500"""
        import src.inference.preprocessor as module
        import inspect
        
        source = inspect.getsource(module)
        lines = source.count('\n')
        
        assert lines < 500, f"preprocessor.py has {lines} lines, should be < 500"
    
    def test_executor_line_count(self):
        """测试执行器行数 < 300"""
        import src.inference.executor as module
        import inspect
        
        source = inspect.getsource(module)
        lines = source.count('\n')
        
        assert lines < 300, f"executor.py has {lines} lines, should be < 300"
    
    def test_postprocessor_line_count(self):
        """测试后处理器行数 < 300"""
        import src.inference.postprocessor as module
        import inspect
        
        source = inspect.getsource(module)
        lines = source.count('\n')
        
        assert lines < 300, f"postprocessor.py has {lines} lines, should be < 300"
    
    def test_base_line_count(self):
        """测试基础推理类行数 < 600"""
        import src.inference.base as module
        import inspect
        
        source = inspect.getsource(module)
        lines = source.count('\n')
        
        assert lines < 600, f"base.py has {lines} lines, should be < 600"
    
    def test_multithread_line_count(self):
        """测试多线程推理行数 < 300"""
        import src.inference.multithread as module
        import inspect
        
        source = inspect.getsource(module)
        lines = source.count('\n')
        
        assert lines < 300, f"multithread.py has {lines} lines, should be < 300"
    
    def test_pipeline_line_count(self):
        """测试流水线推理行数 < 300"""
        import src.inference.pipeline as module
        import inspect
        
        source = inspect.getsource(module)
        lines = source.count('\n')
        
        assert lines < 300, f"pipeline.py has {lines} lines, should be < 300"


class TestIntegration:
    """集成测试"""
    
    def test_full_preprocessing_pipeline(self):
        """测试完整预处理流水线"""
        from src.inference.preprocessor import Preprocessor
        
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        sample_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        result = preprocessor.process_single(sample_image, 'pil')
        
        assert result.dtype == np.float32
        assert len(result.shape) == 1
        assert result.max() <= 1.0
        assert result.min() >= 0.0
    
    def test_config_validation_flow(self):
        """测试配置验证流程"""
        from config import Config
        from config.validator import ConfigValidator
        
        config = Config(
            model_path="models/yolov8n.om",
            device_id=0,
            resolution="640x640",
            num_threads=4
        )
        
        result = ConfigValidator.validate(config)
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
