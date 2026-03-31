#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration validation helpers.
"""

import os
from dataclasses import dataclass
from typing import Any, List

from config import Config
from config.strategy_config import EvaluationConfig
from evaluations.routes import RouteType
from evaluations.tiers import InputTier
from utils.exceptions import ConfigurationError
from utils.logger import LoggerConfig

logger = LoggerConfig.setup_logger("ascend_inference.validator", format_type="text")


@dataclass
class ValidationResult:
    """Validation result."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __bool__(self) -> bool:
        return self.is_valid


class ConfigValidator:
    """Validate runtime configuration."""

    SUPPORTED_RESOLUTIONS = [
        "320x320",
        "416x416",
        "512x512",
        "608x608",
        "640x640",
        "768x768",
        "800x800",
        "1024x1024",
        "1280x1280",
    ]

    SUPPORTED_BACKENDS = ["opencv", "pil", "cv2"]

    @staticmethod
    def validate(config: Config) -> ValidationResult:
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
        ConfigValidator._validate_evaluation(config, errors, warnings)

        for warning in warnings:
            logger.warning(warning)
        for error in errors:
            logger.error(error)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def _validate_model_path(config: Config, errors: List[str], warnings: List[str]) -> None:
        if not config.model_path:
            errors.append("model_path cannot be empty")
            return

        if not os.path.exists(config.model_path):
            errors.append(f"model file does not exist: {config.model_path}")
            return

        if not config.model_path.endswith(".om"):
            warnings.append(f"model file should usually end with .om: {config.model_path}")

    @staticmethod
    def _validate_device(config: Config, errors: List[str], warnings: List[str]) -> None:
        if config.device_id < 0:
            errors.append(f"device_id must be >= 0: {config.device_id}")

        if config.device_id >= Config.MAX_AI_CORES:
            warnings.append(
                f"device_id {config.device_id} may exceed available AI cores (max {Config.MAX_AI_CORES})"
            )

    @staticmethod
    def _validate_resolution(config: Config, errors: List[str], warnings: List[str]) -> None:
        if not config.resolution:
            errors.append("resolution cannot be empty")
            return

        try:
            width, height = Config.get_resolution(config.resolution)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"invalid resolution {config.resolution}: {exc}")
            return

        if width <= 0 or height <= 0:
            errors.append(f"invalid resolution size: {config.resolution}")

        if config.resolution not in ConfigValidator.SUPPORTED_RESOLUTIONS:
            warnings.append(
                f"resolution {config.resolution} is not in the recommended list: "
                f"{', '.join(ConfigValidator.SUPPORTED_RESOLUTIONS)}"
            )

    @staticmethod
    def _validate_threads(config: Config, errors: List[str], warnings: List[str]) -> None:
        if config.num_threads < 1:
            errors.append(f"num_threads must be >= 1: {config.num_threads}")
        elif config.num_threads > Config.MAX_AI_CORES:
            warnings.append(
                f"num_threads {config.num_threads} is greater than the available AI cores "
                f"({Config.MAX_AI_CORES})"
            )
        elif config.num_threads > 16:
            warnings.append(f"num_threads {config.num_threads} may reduce performance")

    @staticmethod
    def _validate_backend(config: Config, errors: List[str], warnings: List[str]) -> None:
        if config.backend not in ConfigValidator.SUPPORTED_BACKENDS:
            errors.append(
                f"unsupported backend: {config.backend}; supported: "
                f"{', '.join(ConfigValidator.SUPPORTED_BACKENDS)}"
            )

    @staticmethod
    def _validate_strategies(config: Config, errors: List[str], warnings: List[str]) -> None:
        enabled = config.get_enabled_strategies()

        if "multithread" in enabled and "pipeline" in enabled:
            warnings.append(
                "multithread and pipeline enabled together may compete for resources"
            )

        if config.strategies.multithread.enabled:
            if config.strategies.multithread.num_threads < 1:
                errors.append("multithread.num_threads must be >= 1")
            elif config.strategies.multithread.num_threads > Config.MAX_AI_CORES:
                warnings.append(
                    f"multithread.num_threads {config.strategies.multithread.num_threads} exceeds "
                    f"available AI cores ({Config.MAX_AI_CORES})"
                )

        if config.strategies.batch.enabled:
            if config.strategies.batch.batch_size < 1:
                errors.append("batch_size must be >= 1")
            if config.strategies.batch.timeout_ms < 0:
                errors.append("batch timeout cannot be negative")
            if config.strategies.batch.batch_size > 32:
                warnings.append("batch_size may be too large")

        if config.strategies.pipeline.enabled:
            if config.strategies.pipeline.queue_size < 1:
                errors.append("pipeline.queue_size must be >= 1")
            if config.strategies.pipeline.queue_size > 100:
                warnings.append("pipeline queue may be too large")

        if config.strategies.memory_pool.enabled:
            if config.strategies.memory_pool.pool_size < 1:
                errors.append("memory_pool.pool_size must be >= 1")
            if config.strategies.memory_pool.max_buffers < 1:
                errors.append("memory_pool.max_buffers must be >= 1")

        if config.strategies.high_res.enabled:
            if config.strategies.high_res.tile_size < 64:
                errors.append("high_res.tile_size must be >= 64")
            if config.strategies.high_res.overlap < 0:
                errors.append("high_res.overlap cannot be negative")
            if config.strategies.high_res.overlap >= config.strategies.high_res.tile_size:
                errors.append("high_res.overlap must be smaller than tile_size")

    @staticmethod
    def _validate_batch(config: Config, errors: List[str], warnings: List[str]) -> None:
        if hasattr(config, "batch_size") and config.batch_size < 1:
            errors.append(f"batch_size must be >= 1: {config.batch_size}")

    @staticmethod
    def _validate_tile(config: Config, errors: List[str], warnings: List[str]) -> None:
        if config.tile_size < 64:
            errors.append(f"tile_size must be >= 64: {config.tile_size}")
        if config.tile_size > 2048:
            warnings.append(f"tile_size {config.tile_size} may cause memory pressure")

        if config.overlap < 0:
            errors.append(f"overlap cannot be negative: {config.overlap}")
        if config.overlap >= config.tile_size:
            errors.append(f"overlap {config.overlap} must be smaller than tile_size {config.tile_size}")

    @staticmethod
    def _validate_evaluation(config: Config, errors: List[str], warnings: List[str]) -> None:
        evaluation = getattr(config, "evaluation", None)
        if not isinstance(evaluation, EvaluationConfig):
            errors.append("evaluation must be an EvaluationConfig instance")
            return

        try:
            input_tier = InputTier.from_value(evaluation.input_tier)
        except Exception as exc:
            errors.append(f"unsupported input tier: {evaluation.input_tier}, error: {exc}")
            return

        try:
            RouteType.from_value(evaluation.route_type)
        except Exception as exc:
            errors.append(f"unsupported route type: {evaluation.route_type}, error: {exc}")
            return

        if not evaluation.report_format:
            errors.append("report_format cannot be empty")
        elif evaluation.report_format not in {"text", "json", "markdown"}:
            errors.append(f"unsupported report_format: {evaluation.report_format}")

        if not isinstance(evaluation.archive_enabled, bool):
            errors.append("archive_enabled must be a boolean")

        expected_width, expected_height = Config.get_resolution(input_tier.runtime_resolution)
        width, height = Config.get_resolution(config.resolution)
        if (width, height) != (expected_width, expected_height):
            errors.append(
                f"resolution {config.resolution} must match input tier "
                f"{input_tier.value} ({input_tier.runtime_resolution})"
            )


def validate_config(config: Config, raise_on_error: bool = True) -> ValidationResult:
    result = ConfigValidator.validate(config)

    if not result.is_valid and raise_on_error:
        raise ConfigurationError(
            f"configuration validation failed: {'; '.join(result.errors)}",
            error_code=5001,
            details={"errors": result.errors, "warnings": result.warnings},
        )

    return result
