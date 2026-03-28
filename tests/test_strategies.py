#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略组件模块测试

测试 src/strategies/ 中的核心功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.strategies import (
    Strategy,
    InferenceContext,
    BaseStrategyConfig,
    NoOpStrategy,
    StrategyComposer,
    register_builtin_strategies
)
from src.strategies.base import BaseStrategyConfig
from config.strategy_config import (
    MultithreadStrategyConfig,
    BatchStrategyConfig,
    PipelineStrategyConfig,
    MemoryPoolStrategyConfig,
    HighResStrategyConfig
)


class TestInferenceContext:
    """InferenceContext 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        context = InferenceContext()
        assert context.inference is None
        assert context.config is None
        assert context.state == {}
        assert context.metrics == {}
        assert context.metadata == {}
    
    def test_init_with_params(self):
        """测试带参数初始化"""
        inference = Mock()
        config = Mock()
        
        context = InferenceContext(inference_instance=inference, config=config)
        assert context.inference is inference
        assert context.config is config
    
    def test_state_operations(self):
        """测试状态操作"""
        context = InferenceContext()
        
        context.set_state('key1', 'value1')
        assert context.get_state('key1') == 'value1'
        assert context.get_state('nonexistent') is None
        assert context.get_state('nonexistent', 'default') == 'default'
    
    def test_metrics_operations(self):
        """测试指标操作"""
        context = InferenceContext()
        
        context.set_metric('fps', 100.0)
        assert context.get_metric('fps') == 100.0
        
        context.update_metrics({'latency': 10.0, 'throughput': 50.0})
        assert context.get_metric('latency') == 10.0
        assert context.get_metric('throughput') == 50.0
    
    def test_metadata_operations(self):
        """测试元数据操作"""
        context = InferenceContext()
        
        context.set_metadata('model', 'yolov8')
        assert context.get_metadata('model') == 'yolov8'
        assert context.get_metadata('nonexistent') is None


class TestNoOpStrategy:
    """NoOpStrategy 类测试"""
    
    def test_init(self):
        """测试初始化"""
        strategy = NoOpStrategy()
        assert strategy.name == "noop"
        assert strategy.enabled is True
    
    def test_apply(self):
        """测试应用策略"""
        strategy = NoOpStrategy()
        context = InferenceContext()
        
        result = strategy.apply(context)
        assert result is context
    
    def test_get_metrics(self):
        """测试获取指标"""
        strategy = NoOpStrategy()
        assert strategy.get_metrics() == {}
    
    def test_enable_disable(self):
        """测试启用/禁用"""
        strategy = NoOpStrategy()
        
        strategy.disable()
        assert strategy.enabled is False
        
        strategy.enable()
        assert strategy.enabled is True


class TestStrategyComposer:
    """StrategyComposer 类测试"""
    
    def test_init(self):
        """测试初始化"""
        composer = StrategyComposer()
        assert len(composer.strategies) == 0
        assert len(composer.execution_order) == 0
    
    def test_add_strategy(self):
        """测试添加策略"""
        composer = StrategyComposer()
        strategy = NoOpStrategy()
        
        result = composer.add_strategy(strategy)
        
        assert result is composer
        assert len(composer.strategies) == 1
        assert composer.strategies[0] is strategy
        assert 'noop' in composer.execution_order
    
    def test_remove_strategy(self):
        """测试移除策略"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        composer.remove_strategy('noop')
        
        assert len(composer.strategies) == 0
        assert 'noop' not in composer.execution_order
    
    def test_get_strategy(self):
        """测试获取策略"""
        composer = StrategyComposer()
        strategy = NoOpStrategy()
        composer.add_strategy(strategy)
        
        result = composer.get_strategy('noop')
        assert result is strategy
        
        assert composer.get_strategy('nonexistent') is None
    
    def test_has_strategy(self):
        """测试检查策略是否存在"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        assert composer.has_strategy('noop') is True
        assert composer.has_strategy('nonexistent') is False
    
    def test_enable_disable_strategy(self):
        """测试启用/禁用策略"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        result = composer.disable_strategy('noop')
        assert result is True
        assert composer.get_strategy('noop').enabled is False
        
        result = composer.enable_strategy('noop')
        assert result is True
        assert composer.get_strategy('noop').enabled is True
        
        result = composer.disable_strategy('nonexistent')
        assert result is False
    
    def test_apply_all(self):
        """测试应用所有策略"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        context = InferenceContext()
        result = composer.apply_all(context)
        
        assert result is context
    
    def test_apply_all_disabled(self):
        """测试应用时跳过禁用的策略"""
        composer = StrategyComposer()
        strategy = NoOpStrategy()
        strategy.disable()
        composer.add_strategy(strategy)
        
        context = InferenceContext()
        result = composer.apply_all(context)
        
        assert result is context
    
    def test_get_all_metrics(self):
        """测试获取所有指标"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        metrics = composer.get_all_metrics()
        
        assert 'noop' in metrics
    
    def test_get_config(self):
        """测试获取配置"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        config = composer.get_config()
        
        assert 'strategies' in config
        assert 'execution_order' in config
        assert len(config['strategies']) == 1
    
    def test_clear(self):
        """测试清空"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        composer.clear()
        
        assert len(composer.strategies) == 0
        assert len(composer.execution_order) == 0
    
    def test_get_enabled_strategies(self):
        """测试获取已启用的策略列表"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        enabled = composer.get_enabled_strategies()
        assert 'noop' in enabled
        
        composer.disable_strategy('noop')
        enabled = composer.get_enabled_strategies()
        assert 'noop' not in enabled
    
    def test_get_disabled_strategies(self):
        """测试获取已禁用的策略列表"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        composer.disable_strategy('noop')
        
        disabled = composer.get_disabled_strategies()
        assert 'noop' in disabled
    
    def test_len(self):
        """测试长度"""
        composer = StrategyComposer()
        assert len(composer) == 0
        
        composer.add_strategy(NoOpStrategy())
        assert len(composer) == 1
    
    def test_iter(self):
        """测试迭代"""
        composer = StrategyComposer()
        strategy1 = NoOpStrategy()
        strategy2 = NoOpStrategy()
        strategy2.name = "noop2"
        
        composer.add_strategy(strategy1)
        composer.add_strategy(strategy2)
        
        strategies = list(composer)
        assert len(strategies) == 2
    
    def test_repr(self):
        """测试字符串表示"""
        composer = StrategyComposer()
        composer.add_strategy(NoOpStrategy())
        
        repr_str = repr(composer)
        assert 'StrategyComposer' in repr_str
        assert 'noop' in repr_str


class TestMultithreadStrategy:
    """MultithreadStrategy 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        from src.strategies.multithread import MultithreadStrategy
        
        strategy = MultithreadStrategy()
        assert strategy.name == "multithread"
        assert strategy.config.num_threads == 4
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from src.strategies.multithread import MultithreadStrategy
        
        config = MultithreadStrategyConfig(num_threads=8, enabled=True)
        strategy = MultithreadStrategy(config)
        
        assert strategy.config.num_threads == 8
        assert strategy.enabled is True
    
    def test_from_dict(self):
        """测试从字典创建"""
        from src.strategies.multithread import MultithreadStrategy
        
        data = {'num_threads': 16, 'enabled': True}
        strategy = MultithreadStrategy.from_dict(data)
        
        assert strategy.config.num_threads == 16
        assert strategy.enabled is True
    
    def test_get_metrics(self):
        """测试获取指标"""
        from src.strategies.multithread import MultithreadStrategy
        
        strategy = MultithreadStrategy()
        metrics = strategy.get_metrics()
        
        assert 'num_threads' in metrics
        assert 'work_stealing' in metrics
    
    def test_get_theoretical_speedup(self):
        """测试获取理论加速比"""
        from src.strategies.multithread import MultithreadStrategy
        
        config = MultithreadStrategyConfig(num_threads=8)
        strategy = MultithreadStrategy(config)
        
        assert strategy.get_theoretical_speedup() == 8.0


class TestBatchStrategy:
    """BatchStrategy 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        from src.strategies.batch import BatchStrategy
        
        strategy = BatchStrategy()
        assert strategy.name == "batch"
        assert strategy.config.batch_size == 4
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from src.strategies.batch import BatchStrategy
        
        config = BatchStrategyConfig(batch_size=8, enabled=True)
        strategy = BatchStrategy(config)
        
        assert strategy.config.batch_size == 8
        assert strategy.enabled is True
    
    def test_from_dict(self):
        """测试从字典创建"""
        from src.strategies.batch import BatchStrategy
        
        data = {'batch_size': 16, 'enabled': True}
        strategy = BatchStrategy.from_dict(data)
        
        assert strategy.config.batch_size == 16
    
    def test_get_metrics(self):
        """测试获取指标"""
        from src.strategies.batch import BatchStrategy
        
        strategy = BatchStrategy()
        metrics = strategy.get_metrics()
        
        assert 'batch_size' in metrics
        assert 'timeout_ms' in metrics


class TestPipelineStrategy:
    """PipelineStrategy 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        from src.strategies.pipeline import PipelineStrategy
        
        strategy = PipelineStrategy()
        assert strategy.name == "pipeline"
        assert strategy.config.queue_size == 10
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from src.strategies.pipeline import PipelineStrategy
        
        config = PipelineStrategyConfig(queue_size=20, enabled=True)
        strategy = PipelineStrategy(config)
        
        assert strategy.config.queue_size == 20
        assert strategy.enabled is True
    
    def test_get_theoretical_speedup(self):
        """测试获取理论加速比"""
        from src.strategies.pipeline import PipelineStrategy
        
        config = PipelineStrategyConfig(
            num_preprocess_threads=2,
            num_infer_threads=1,
            num_postprocess_threads=1
        )
        strategy = PipelineStrategy(config)
        
        assert strategy.get_theoretical_speedup() == 4.0


class TestMemoryPoolStrategy:
    """MemoryPoolStrategy 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        from src.strategies.memory_pool import MemoryPoolStrategy
        
        strategy = MemoryPoolStrategy()
        assert strategy.name == "memory_pool"
        assert strategy.config.pool_size == 10
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from src.strategies.memory_pool import MemoryPoolStrategy
        
        config = MemoryPoolStrategyConfig(pool_size=20, enabled=True)
        strategy = MemoryPoolStrategy(config)
        
        assert strategy.config.pool_size == 20
        assert strategy.enabled is True
    
    def test_get_metrics(self):
        """测试获取指标"""
        from src.strategies.memory_pool import MemoryPoolStrategy
        
        strategy = MemoryPoolStrategy()
        metrics = strategy.get_metrics()
        
        assert 'pool_size' in metrics
        assert 'reuse_rate' in metrics


class TestHighResStrategy:
    """HighResStrategy 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        from src.strategies.high_res import HighResStrategy
        
        strategy = HighResStrategy()
        assert strategy.name == "high_res"
        assert strategy.config.tile_size == 640
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        from src.strategies.high_res import HighResStrategy
        
        config = HighResStrategyConfig(tile_size=512, enabled=True)
        strategy = HighResStrategy(config)
        
        assert strategy.config.tile_size == 512
        assert strategy.enabled is True
    
    def test_estimate_tiles(self):
        """测试估算分块数量"""
        from src.strategies.high_res import HighResStrategy
        
        strategy = HighResStrategy()
        
        tiles = strategy.estimate_tiles(1920, 1080)
        assert tiles > 0
        
        tiles_small = strategy.estimate_tiles(640, 640)
        assert tiles_small >= 1
    
    def test_get_metrics(self):
        """测试获取指标"""
        from src.strategies.high_res import HighResStrategy
        
        strategy = HighResStrategy()
        metrics = strategy.get_metrics()
        
        assert 'tile_size' in metrics
        assert 'overlap' in metrics


class TestStrategyComposerFromConfig:
    """StrategyComposer.from_config 测试"""
    
    def test_from_config_empty(self):
        """测试空配置"""
        config = {'strategies': []}
        composer = StrategyComposer.from_config(config)
        
        assert len(composer) == 0
    
    def test_from_config_with_noop(self):
        """测试带 NoOp 策略的配置"""
        StrategyComposer.register_strategy('noop', NoOpStrategy)
        
        config = {
            'strategies': [
                {'name': 'noop', 'enabled': True}
            ]
        }
        composer = StrategyComposer.from_config(config)
        
        assert len(composer) == 1
        assert composer.has_strategy('noop')


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
