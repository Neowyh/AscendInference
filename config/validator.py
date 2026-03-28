#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证器模块

提供配置验证功能，支持：
- 参数类型验证
- 参数范围验证
- 策略冲突检测
- 配置完整性检查
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.exceptions import ConfigurationError

logger = LoggerConfig.setup_logger('ascend_inference.validator', format_type='text')


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.is_valid


class ConfigValidator:
    """配置验证器"""
    
    SUPPORTED_RESOLUTIONS = [
        '320x320', '416x416', '512x512', '608x608', '640x640',
        '768x768', '800x800', '1024x1024', '1280x1280'
    ]
    
    SUPPORTED_BACKENDS = ['opencv', 'pil', 'cv2']
    
    @staticmethod
    def validate(config: Config) -> ValidationResult:
        """验证配置
        
        Args:
            config: 配置实例
            
        Returns:
            ValidationResult: 验证结果
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        ConfigValidator._validate_model_path(config, errors, warnings)
        ConfigValidator._validate_device(config, errors, warnings)
        ConfigValidator._validate_resolution(config, errors, warnings)
        ConfigValidator._validate_threads(config, errors, warnings)
        ConfigValidator._validate_backend(config, errors, warnings)
        ConfigValidator._validate_strategies(config, errors, warnings)
        ConfigValidator._validate_batch(config, errors, warnings)
        ConfigValidator._validate_tile(config, errors, warnings)
        
        is_valid = len(errors) == 0
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        if errors:
            for error in errors:
                logger.error(error)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def _validate_model_path(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证模型路径"""
        if not config.model_path:
            errors.append("模型路径不能为空")
            return
        
        if not os.path.exists(config.model_path):
            errors.append(f"模型文件不存在: {config.model_path}")
            return
        
        if not config.model_path.endswith('.om'):
            warnings.append(f"模型文件扩展名不是 .om: {config.model_path}")
    
    @staticmethod
    def _validate_device(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证设备配置"""
        if config.device_id < 0:
            errors.append(f"设备ID无效: {config.device_id}，必须 >= 0")
        
        if config.device_id >= Config.MAX_AI_CORES:
            warnings.append(
                f"设备ID ({config.device_id}) 可能超出可用设备数量 "
                f"(最大: {Config.MAX_AI_CORES})"
            )
    
    @staticmethod
    def _validate_resolution(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证分辨率"""
        if not config.resolution:
            errors.append("分辨率不能为空")
            return
        
        try:
            width, height = Config.get_resolution(config.resolution)
            if width <= 0 or height <= 0:
                errors.append(f"分辨率尺寸无效: {config.resolution}")
        except Exception as e:
            errors.append(f"分辨率格式无效: {config.resolution}, 错误: {e}")
            return
        
        if config.resolution not in ConfigValidator.SUPPORTED_RESOLUTIONS:
            warnings.append(
                f"分辨率 {config.resolution} 不在推荐列表中，"
                f"推荐: {', '.join(ConfigValidator.SUPPORTED_RESOLUTIONS)}"
            )
    
    @staticmethod
    def _validate_threads(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证线程配置"""
        if config.num_threads < 1:
            errors.append(f"线程数必须 >= 1: {config.num_threads}")
        elif config.num_threads > Config.MAX_AI_CORES:
            warnings.append(
                f"线程数 ({config.num_threads}) 超过最大AI核数 "
                f"({Config.MAX_AI_CORES})，可能无法充分利用"
            )
        elif config.num_threads > 16:
            warnings.append(
                f"线程数 ({config.num_threads}) 过大，可能导致性能下降"
            )
    
    @staticmethod
    def _validate_backend(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证后端配置"""
        if config.backend not in ConfigValidator.SUPPORTED_BACKENDS:
            errors.append(
                f"不支持的后端: {config.backend}，"
                f"支持: {', '.join(ConfigValidator.SUPPORTED_BACKENDS)}"
            )
    
    @staticmethod
    def _validate_strategies(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证策略配置"""
        enabled = config.get_enabled_strategies()
        
        if 'multithread' in enabled and 'pipeline' in enabled:
            warnings.append(
                "多线程策略和流水线策略同时启用可能导致资源竞争，"
                "建议只启用其中一个"
            )
        
        if config.strategies.multithread.enabled:
            if config.strategies.multithread.num_threads < 1:
                errors.append("多线程策略的线程数必须 >= 1")
            elif config.strategies.multithread.num_threads > Config.MAX_AI_CORES:
                warnings.append(
                    f"多线程策略的线程数 ({config.strategies.multithread.num_threads}) "
                    f"超过最大AI核数 ({Config.MAX_AI_CORES})"
                )
        
        if config.strategies.batch.enabled:
            if config.strategies.batch.batch_size < 1:
                errors.append("批大小必须 >= 1")
            if config.strategies.batch.timeout_ms < 0:
                errors.append("批处理超时时间不能为负")
            if config.strategies.batch.batch_size > 32:
                warnings.append("批大小过大可能导致内存不足")
        
        if config.strategies.pipeline.enabled:
            if config.strategies.pipeline.queue_size < 1:
                errors.append("流水线队列大小必须 >= 1")
            if config.strategies.pipeline.queue_size > 100:
                warnings.append("流水线队列过大可能导致延迟增加")
        
        if config.strategies.memory_pool.enabled:
            if config.strategies.memory_pool.pool_size < 1:
                errors.append("内存池大小必须 >= 1")
            if config.strategies.memory_pool.max_buffers < 1:
                errors.append("最大缓冲区数必须 >= 1")
        
        if config.strategies.high_res.enabled:
            if config.strategies.high_res.tile_size < 64:
                errors.append("分块大小必须 >= 64")
            if config.strategies.high_res.overlap < 0:
                errors.append("重叠区域不能为负")
            if config.strategies.high_res.overlap >= config.strategies.high_res.tile_size:
                errors.append("重叠区域必须小于分块大小")
    
    @staticmethod
    def _validate_batch(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证批处理配置"""
        if hasattr(config, 'batch_size'):
            if config.batch_size < 1:
                errors.append(f"批处理大小必须 >= 1: {config.batch_size}")
    
    @staticmethod
    def _validate_tile(config: Config, errors: List[str], warnings: List[str]) -> None:
        """验证分块配置"""
        if config.tile_size < 64:
            errors.append(f"分块大小必须 >= 64: {config.tile_size}")
        if config.tile_size > 2048:
            warnings.append(f"分块大小 ({config.tile_size}) 过大可能导致内存问题")
        
        if config.overlap < 0:
            errors.append(f"重叠区域不能为负: {config.overlap}")
        if config.overlap >= config.tile_size:
            errors.append(f"重叠区域 ({config.overlap}) 必须小于分块大小 ({config.tile_size})")


def validate_config(config: Config, raise_on_error: bool = True) -> ValidationResult:
    """验证配置的便捷函数
    
    Args:
        config: 配置实例
        raise_on_error: 验证失败时是否抛出异常
        
    Returns:
        ValidationResult: 验证结果
        
    Raises:
        ConfigurationError: 配置验证失败（当 raise_on_error=True）
    """
    result = ConfigValidator.validate(config)
    
    if not result.is_valid and raise_on_error:
        raise ConfigurationError(
            f"配置验证失败: {'; '.join(result.errors)}",
            error_code=5001,
            details={"errors": result.errors, "warnings": result.warnings}
        )
    
    return result
