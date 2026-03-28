#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略配置模块

提供各种优化策略的配置类，支持：
- 多线程策略配置
- 批处理策略配置
- 流水线策略配置
- 内存池策略配置
- 高分辨率策略配置
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class MultithreadStrategyConfig:
    """多线程策略配置"""
    enabled: bool = False
    num_threads: int = 4
    work_stealing: bool = True
    dynamic_scaling: bool = False
    min_threads: int = 1
    max_threads: int = 4
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.2
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultithreadStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            num_threads=data.get('num_threads', 4),
            work_stealing=data.get('work_stealing', True),
            dynamic_scaling=data.get('dynamic_scaling', False),
            min_threads=data.get('min_threads', 1),
            max_threads=data.get('max_threads', 4),
            scale_up_threshold=data.get('scale_up_threshold', 0.7),
            scale_down_threshold=data.get('scale_down_threshold', 0.2)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'num_threads': self.num_threads,
            'work_stealing': self.work_stealing,
            'dynamic_scaling': self.dynamic_scaling,
            'min_threads': self.min_threads,
            'max_threads': self.max_threads,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold
        }


@dataclass
class BatchStrategyConfig:
    """批处理策略配置"""
    enabled: bool = False
    batch_size: int = 4
    timeout_ms: float = 10.0
    dynamic_batch: bool = False
    max_batch_size: int = 16
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            batch_size=data.get('batch_size', 4),
            timeout_ms=data.get('timeout_ms', 10.0),
            dynamic_batch=data.get('dynamic_batch', False),
            max_batch_size=data.get('max_batch_size', 16)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'batch_size': self.batch_size,
            'timeout_ms': self.timeout_ms,
            'dynamic_batch': self.dynamic_batch,
            'max_batch_size': self.max_batch_size
        }


@dataclass
class PipelineStrategyConfig:
    """流水线策略配置"""
    enabled: bool = False
    queue_size: int = 10
    num_preprocess_threads: int = 2
    num_infer_threads: int = 1
    num_postprocess_threads: int = 1
    batch_size: int = 4
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            queue_size=data.get('queue_size', 10),
            num_preprocess_threads=data.get('num_preprocess_threads', 2),
            num_infer_threads=data.get('num_infer_threads', 1),
            num_postprocess_threads=data.get('num_postprocess_threads', 1),
            batch_size=data.get('batch_size', 4)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'queue_size': self.queue_size,
            'num_preprocess_threads': self.num_preprocess_threads,
            'num_infer_threads': self.num_infer_threads,
            'num_postprocess_threads': self.num_postprocess_threads,
            'batch_size': self.batch_size
        }


@dataclass
class MemoryPoolStrategyConfig:
    """内存池策略配置"""
    enabled: bool = False
    pool_size: int = 10
    growth_factor: float = 1.5
    max_buffers: int = 20
    device: str = 'host'
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryPoolStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            pool_size=data.get('pool_size', 10),
            growth_factor=data.get('growth_factor', 1.5),
            max_buffers=data.get('max_buffers', 20),
            device=data.get('device', 'host')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'pool_size': self.pool_size,
            'growth_factor': self.growth_factor,
            'max_buffers': self.max_buffers,
            'device': self.device
        }


@dataclass
class HighResStrategyConfig:
    """高分辨率策略配置"""
    enabled: bool = False
    tile_size: int = 640
    overlap: int = 100
    weight_fusion: bool = True
    overlap_ratio: float = 0.25
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HighResStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            tile_size=data.get('tile_size', 640),
            overlap=data.get('overlap', 100),
            weight_fusion=data.get('weight_fusion', True),
            overlap_ratio=data.get('overlap_ratio', 0.25)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'weight_fusion': self.weight_fusion,
            'overlap_ratio': self.overlap_ratio
        }


@dataclass
class AsyncIOStrategyConfig:
    """异步IO策略配置"""
    enabled: bool = False
    prefetch_size: int = 5
    num_workers: int = 2
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AsyncIOStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            prefetch_size=data.get('prefetch_size', 5),
            num_workers=data.get('num_workers', 2)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'prefetch_size': self.prefetch_size,
            'num_workers': self.num_workers
        }


@dataclass
class CacheStrategyConfig:
    """缓存策略配置"""
    enabled: bool = False
    max_size: int = 100
    ttl_seconds: float = 300.0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheStrategyConfig':
        return cls(
            enabled=data.get('enabled', False),
            max_size=data.get('max_size', 100),
            ttl_seconds=data.get('ttl_seconds', 300.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


@dataclass
class StrategyConfig:
    """策略配置集合
    
    聚合所有优化策略的配置
    """
    multithread: MultithreadStrategyConfig = field(default_factory=MultithreadStrategyConfig)
    batch: BatchStrategyConfig = field(default_factory=BatchStrategyConfig)
    pipeline: PipelineStrategyConfig = field(default_factory=PipelineStrategyConfig)
    memory_pool: MemoryPoolStrategyConfig = field(default_factory=MemoryPoolStrategyConfig)
    high_res: HighResStrategyConfig = field(default_factory=HighResStrategyConfig)
    async_io: AsyncIOStrategyConfig = field(default_factory=AsyncIOStrategyConfig)
    cache: CacheStrategyConfig = field(default_factory=CacheStrategyConfig)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        """从字典创建配置
        
        Args:
            data: 配置字典
            
        Returns:
            StrategyConfig: 配置实例
        """
        config = cls()
        
        if 'multithread' in data:
            config.multithread = MultithreadStrategyConfig.from_dict(data['multithread'])
        if 'batch' in data:
            config.batch = BatchStrategyConfig.from_dict(data['batch'])
        if 'pipeline' in data:
            config.pipeline = PipelineStrategyConfig.from_dict(data['pipeline'])
        if 'memory_pool' in data:
            config.memory_pool = MemoryPoolStrategyConfig.from_dict(data['memory_pool'])
        if 'high_res' in data:
            config.high_res = HighResStrategyConfig.from_dict(data['high_res'])
        if 'async_io' in data:
            config.async_io = AsyncIOStrategyConfig.from_dict(data['async_io'])
        if 'cache' in data:
            config.cache = CacheStrategyConfig.from_dict(data['cache'])
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict: 配置字典
        """
        return {
            'multithread': self.multithread.to_dict(),
            'batch': self.batch.to_dict(),
            'pipeline': self.pipeline.to_dict(),
            'memory_pool': self.memory_pool.to_dict(),
            'high_res': self.high_res.to_dict(),
            'async_io': self.async_io.to_dict(),
            'cache': self.cache.to_dict()
        }
    
    def get_enabled_strategies(self) -> list:
        """获取已启用的策略列表
        
        Returns:
            list: 已启用的策略名称列表
        """
        enabled = []
        if self.multithread.enabled:
            enabled.append('multithread')
        if self.batch.enabled:
            enabled.append('batch')
        if self.pipeline.enabled:
            enabled.append('pipeline')
        if self.memory_pool.enabled:
            enabled.append('memory_pool')
        if self.high_res.enabled:
            enabled.append('high_res')
        if self.async_io.enabled:
            enabled.append('async_io')
        if self.cache.enabled:
            enabled.append('cache')
        return enabled
    
    def is_any_enabled(self) -> bool:
        """检查是否有任何策略启用
        
        Returns:
            bool: 是否有策略启用
        """
        return len(self.get_enabled_strategies()) > 0


@dataclass
class BenchmarkConfig:
    """评测配置"""
    iterations: int = 100
    warmup: int = 5
    enable_profiling: bool = True
    enable_monitoring: bool = True
    output_format: str = "text"
    output_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        return cls(
            iterations=data.get('iterations', 100),
            warmup=data.get('warmup', 5),
            enable_profiling=data.get('enable_profiling', True),
            enable_monitoring=data.get('enable_monitoring', True),
            output_format=data.get('output_format', 'text'),
            output_path=data.get('output_path')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'iterations': self.iterations,
            'warmup': self.warmup,
            'enable_profiling': self.enable_profiling,
            'enable_monitoring': self.enable_monitoring,
            'output_format': self.output_format,
            'output_path': self.output_path
        }


@dataclass
class ModelInfoConfig:
    """模型信息配置"""
    collect_input_size: bool = True
    collect_output_size: bool = True
    collect_params: bool = False
    collect_flops: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfoConfig':
        return cls(
            collect_input_size=data.get('collect_input_size', True),
            collect_output_size=data.get('collect_output_size', True),
            collect_params=data.get('collect_params', False),
            collect_flops=data.get('collect_flops', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'collect_input_size': self.collect_input_size,
            'collect_output_size': self.collect_output_size,
            'collect_params': self.collect_params,
            'collect_flops': self.collect_flops
        }
