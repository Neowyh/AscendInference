#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略配置模块测试

测试 config/strategy_config.py 中的核心功能
"""

import pytest

from config.strategy_config import (
    MultithreadStrategyConfig,
    BatchStrategyConfig,
    PipelineStrategyConfig,
    MemoryPoolStrategyConfig,
    HighResStrategyConfig,
    AsyncIOStrategyConfig,
    CacheStrategyConfig,
    StrategyConfig,
    BenchmarkConfig,
    ModelInfoConfig
)


class TestMultithreadStrategyConfig:
    """MultithreadStrategyConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = MultithreadStrategyConfig()
        assert config.enabled is False
        assert config.num_threads == 4
        assert config.work_stealing is True
        assert config.dynamic_scaling is False
        assert config.min_threads == 1
        assert config.max_threads == 4
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'enabled': True,
            'num_threads': 8,
            'work_stealing': False,
            'dynamic_scaling': True
        }
        config = MultithreadStrategyConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.num_threads == 8
        assert config.work_stealing is False
        assert config.dynamic_scaling is True
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = MultithreadStrategyConfig(
            enabled=True,
            num_threads=8,
            work_stealing=False
        )
        data = config.to_dict()
        
        assert data['enabled'] is True
        assert data['num_threads'] == 8
        assert data['work_stealing'] is False
        assert 'dynamic_scaling' in data
    
    def test_roundtrip(self):
        """测试字典往返转换"""
        original = MultithreadStrategyConfig(
            enabled=True,
            num_threads=16,
            work_stealing=False,
            dynamic_scaling=True,
            min_threads=2,
            max_threads=16
        )
        
        data = original.to_dict()
        restored = MultithreadStrategyConfig.from_dict(data)
        
        assert restored.enabled == original.enabled
        assert restored.num_threads == original.num_threads
        assert restored.work_stealing == original.work_stealing
        assert restored.dynamic_scaling == original.dynamic_scaling


class TestBatchStrategyConfig:
    """BatchStrategyConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = BatchStrategyConfig()
        assert config.enabled is False
        assert config.batch_size == 4
        assert config.timeout_ms == 10.0
        assert config.dynamic_batch is False
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'enabled': True,
            'batch_size': 8,
            'timeout_ms': 20.0,
            'dynamic_batch': True
        }
        config = BatchStrategyConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.batch_size == 8
        assert config.timeout_ms == 20.0
        assert config.dynamic_batch is True
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = BatchStrategyConfig(enabled=True, batch_size=16)
        data = config.to_dict()
        
        assert data['enabled'] is True
        assert data['batch_size'] == 16


class TestPipelineStrategyConfig:
    """PipelineStrategyConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = PipelineStrategyConfig()
        assert config.enabled is False
        assert config.queue_size == 10
        assert config.num_preprocess_threads == 2
        assert config.num_infer_threads == 1
        assert config.num_postprocess_threads == 1
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'enabled': True,
            'queue_size': 20,
            'num_preprocess_threads': 4,
            'num_infer_threads': 2
        }
        config = PipelineStrategyConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.queue_size == 20
        assert config.num_preprocess_threads == 4
        assert config.num_infer_threads == 2


class TestMemoryPoolStrategyConfig:
    """MemoryPoolStrategyConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = MemoryPoolStrategyConfig()
        assert config.enabled is False
        assert config.pool_size == 10
        assert config.growth_factor == 1.5
        assert config.max_buffers == 20
        assert config.device == 'host'
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'enabled': True,
            'pool_size': 20,
            'device': 'device'
        }
        config = MemoryPoolStrategyConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.pool_size == 20
        assert config.device == 'device'


class TestHighResStrategyConfig:
    """HighResStrategyConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = HighResStrategyConfig()
        assert config.enabled is False
        assert config.tile_size == 640
        assert config.overlap == 100
        assert config.weight_fusion is True
        assert config.overlap_ratio == 0.25
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'enabled': True,
            'tile_size': 1024,
            'overlap': 200,
            'weight_fusion': False
        }
        config = HighResStrategyConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.tile_size == 1024
        assert config.overlap == 200
        assert config.weight_fusion is False


class TestStrategyConfig:
    """StrategyConfig 集合类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = StrategyConfig()
        
        assert isinstance(config.multithread, MultithreadStrategyConfig)
        assert isinstance(config.batch, BatchStrategyConfig)
        assert isinstance(config.pipeline, PipelineStrategyConfig)
        assert isinstance(config.memory_pool, MemoryPoolStrategyConfig)
        assert isinstance(config.high_res, HighResStrategyConfig)
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'multithread': {'enabled': True, 'num_threads': 8},
            'batch': {'enabled': True, 'batch_size': 16},
            'pipeline': {'enabled': False}
        }
        config = StrategyConfig.from_dict(data)
        
        assert config.multithread.enabled is True
        assert config.multithread.num_threads == 8
        assert config.batch.enabled is True
        assert config.batch.batch_size == 16
        assert config.pipeline.enabled is False
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = StrategyConfig()
        config.multithread.enabled = True
        config.batch.enabled = True
        
        data = config.to_dict()
        
        assert 'multithread' in data
        assert 'batch' in data
        assert 'pipeline' in data
        assert data['multithread']['enabled'] is True
        assert data['batch']['enabled'] is True
    
    def test_get_enabled_strategies(self):
        """测试获取已启用的策略列表"""
        config = StrategyConfig()
        assert config.get_enabled_strategies() == []
        
        config.multithread.enabled = True
        config.batch.enabled = True
        
        enabled = config.get_enabled_strategies()
        assert 'multithread' in enabled
        assert 'batch' in enabled
        assert len(enabled) == 2
    
    def test_is_any_enabled(self):
        """测试是否有策略启用"""
        config = StrategyConfig()
        assert config.is_any_enabled() is False
        
        config.multithread.enabled = True
        assert config.is_any_enabled() is True
    
    def test_roundtrip(self):
        """测试字典往返转换"""
        original = StrategyConfig()
        original.multithread.enabled = True
        original.multithread.num_threads = 8
        original.batch.enabled = True
        original.batch.batch_size = 16
        
        data = original.to_dict()
        restored = StrategyConfig.from_dict(data)
        
        assert restored.multithread.enabled == original.multithread.enabled
        assert restored.multithread.num_threads == original.multithread.num_threads
        assert restored.batch.enabled == original.batch.enabled
        assert restored.batch.batch_size == original.batch.batch_size


class TestBenchmarkConfig:
    """BenchmarkConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = BenchmarkConfig()
        assert config.iterations == 100
        assert config.warmup == 5
        assert config.enable_profiling is True
        assert config.enable_monitoring is True
        assert config.output_format == "text"
        assert config.output_path is None
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'iterations': 200,
            'warmup': 10,
            'enable_profiling': False,
            'output_format': 'json',
            'output_path': 'report.json'
        }
        config = BenchmarkConfig.from_dict(data)
        
        assert config.iterations == 200
        assert config.warmup == 10
        assert config.enable_profiling is False
        assert config.output_format == 'json'
        assert config.output_path == 'report.json'
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = BenchmarkConfig(iterations=50, warmup=3)
        data = config.to_dict()
        
        assert data['iterations'] == 50
        assert data['warmup'] == 3


class TestModelInfoConfig:
    """ModelInfoConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = ModelInfoConfig()
        assert config.collect_input_size is True
        assert config.collect_output_size is True
        assert config.collect_params is False
        assert config.collect_flops is False
    
    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            'collect_input_size': False,
            'collect_params': True
        }
        config = ModelInfoConfig.from_dict(data)
        
        assert config.collect_input_size is False
        assert config.collect_params is True


class TestConfigIntegration:
    """配置集成测试"""
    
    def test_config_with_strategies(self):
        """测试 Config 类集成策略配置"""
        from config import Config
        
        config = Config()
        
        assert hasattr(config, 'strategies')
        assert hasattr(config, 'benchmark')
        assert hasattr(config, 'model_info')
        
        assert isinstance(config.strategies, StrategyConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
    
    def test_config_to_dict(self):
        """测试 Config 转换为字典"""
        from config import Config
        
        config = Config()
        config.strategies.multithread.enabled = True
        
        data = config.to_dict()
        
        assert 'strategies' in data
        assert 'benchmark' in data
        assert data['strategies']['multithread']['enabled'] is True
    
    def test_config_get_enabled_strategies(self):
        """测试 Config 获取启用的策略"""
        from config import Config
        
        config = Config()
        assert config.get_enabled_strategies() == []
        
        config.strategies.multithread.enabled = True
        assert 'multithread' in config.get_enabled_strategies()
    
    def test_config_is_strategy_enabled(self):
        """测试 Config 检查策略是否启用"""
        from config import Config
        
        config = Config()
        assert config.is_strategy_enabled('multithread') is False
        
        config.strategies.multithread.enabled = True
        assert config.is_strategy_enabled('multithread') is True
        assert config.is_strategy_enabled('batch') is False


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
