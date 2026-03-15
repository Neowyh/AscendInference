#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数验证工具模块

提供统一的参数验证功能，提前拦截非法输入
"""
import os
from typing import Any, Union, List, Optional
from pathlib import Path

from utils.exceptions import InputValidationError
from config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES


def validate_file_path(path: Union[str, Path], must_exist: bool = True, allowed_extensions: Optional[List[str]] = None) -> None:
    """验证文件路径合法性

    Args:
        path: 文件路径
        must_exist: 是否必须存在
        allowed_extensions: 允许的文件扩展名列表，None表示不限制

    Raises:
        InputValidationError: 路径不合法时抛出
    """
    if not path:
        raise InputValidationError(
            "文件路径不能为空",
            error_code=3001,
            details={"path": str(path)}
        )

    path_str = str(path)
    path_obj = Path(path_str).resolve()

    # 路径遍历安全检查
    try:
        # 检查是否是相对路径跳出当前目录
        work_dir = Path.cwd().resolve()
        path_obj.relative_to(work_dir)
    except ValueError:
        raise InputValidationError(
            "路径不允许超出工作目录",
            error_code=3002,
            details={"path": path_str, "work_dir": str(work_dir)}
        ) from None

    if must_exist and not path_obj.exists():
        raise InputValidationError(
            "文件不存在",
            error_code=3003,
            details={"path": path_str}
        )

    if must_exist and not path_obj.is_file():
        raise InputValidationError(
            "路径不是文件",
            error_code=3004,
            details={"path": path_str}
        )

    if allowed_extensions:
        ext = path_obj.suffix.lower()
        allowed_exts = [e.lower() for e in allowed_extensions]
        if ext not in allowed_exts:
            raise InputValidationError(
                f"文件类型不支持，允许的类型: {allowed_extensions}",
                error_code=3005,
                details={"path": path_str, "allowed_extensions": allowed_extensions}
            )


def validate_directory_path(path: Union[str, Path], must_exist: bool = True, create_if_not_exists: bool = False) -> None:
    """验证目录路径合法性

    Args:
        path: 目录路径
        must_exist: 是否必须存在
        create_if_not_exists: 不存在时是否自动创建

    Raises:
        InputValidationError: 路径不合法时抛出
    """
    if not path:
        raise InputValidationError(
            "目录路径不能为空",
            error_code=3010,
            details={"path": str(path)}
        )

    path_str = str(path)
    path_obj = Path(path_str).resolve()

    # 路径遍历安全检查
    try:
        work_dir = Path.cwd().resolve()
        path_obj.relative_to(work_dir)
    except ValueError:
        raise InputValidationError(
            "路径不允许超出工作目录",
            error_code=3011,
            details={"path": path_str, "work_dir": str(work_dir)}
        ) from None

    if path_obj.exists():
        if not path_obj.is_dir():
            raise InputValidationError(
                "路径不是目录",
                error_code=3012,
                details={"path": path_str}
            )
    else:
        if must_exist:
            raise InputValidationError(
                "目录不存在",
                error_code=3013,
                details={"path": path_str}
            )
        if create_if_not_exists:
            path_obj.mkdir(parents=True, exist_ok=True)


def validate_numeric_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], name: str = "value") -> None:
    """验证数值范围

    Args:
        value: 数值
        min_val: 最小值（包含）
        max_val: 最大值（包含）
        name: 参数名称，用于错误提示

    Raises:
        InputValidationError: 数值超出范围时抛出
    """
    if not isinstance(value, (int, float)):
        raise InputValidationError(
            f"{name}必须是数字类型",
            error_code=3020,
            details={name: value, "type": str(type(value))}
        )

    if value < min_val or value > max_val:
        raise InputValidationError(
            f"{name}超出范围，必须在[{min_val}, {max_val}]之间",
            error_code=3021,
            details={name: value, "min": min_val, "max": max_val}
        )


def validate_positive_integer(value: int, name: str = "value", min_val: int = 1) -> None:
    """验证正整数

    Args:
        value: 数值
        name: 参数名称
        min_val: 最小值（包含）

    Raises:
        InputValidationError: 不是正整数时抛出
    """
    if not isinstance(value, int) or value < min_val:
        raise InputValidationError(
            f"{name}必须是大于等于{min_val}的正整数",
            error_code=3022,
            details={name: value, "min": min_val}
        )


def validate_enum(value: Any, allowed_values: List[Any], name: str = "value") -> None:
    """验证枚举值

    Args:
        value: 待验证值
        allowed_values: 允许的值列表
        name: 参数名称

    Raises:
        InputValidationError: 值不在允许列表中时抛出
    """
    if value not in allowed_values:
        raise InputValidationError(
            f"{name}无效，允许的值: {allowed_values}",
            error_code=3030,
            details={name: value, "allowed_values": allowed_values}
        )


def validate_resolution(resolution: str) -> None:
    """验证分辨率是否支持

    Args:
        resolution: 分辨率字符串

    Raises:
        InputValidationError: 分辨率不支持时抛出
    """
    validate_enum(resolution, list(SUPPORTED_RESOLUTIONS.keys()), "resolution")


def validate_device_id(device_id: int) -> None:
    """验证设备ID合法性

    Args:
        device_id: 设备ID

    Raises:
        InputValidationError: 设备ID不合法时抛出
    """
    validate_numeric_range(device_id, 0, MAX_AI_CORES - 1, "device_id")


def validate_batch_size(batch_size: int) -> None:
    """验证批处理大小合法性

    Args:
        batch_size: 批处理大小

    Raises:
        InputValidationError: 批处理大小不合法时抛出
    """
    validate_positive_integer(batch_size, "batch_size", min_val=1)


def validate_thread_count(thread_count: int) -> None:
    """验证线程数合法性

    Args:
        thread_count: 线程数

    Raises:
        InputValidationError: 线程数不合法时抛出
    """
    validate_numeric_range(thread_count, 1, MAX_AI_CORES * 2, "thread_count")


def validate_image_backend(backend: str) -> None:
    """验证图像处理后端合法性

    Args:
        backend: 后端名称

    Raises:
        InputValidationError: 后端不支持时抛出
    """
    validate_enum(backend, ["pil", "opencv"], "backend")


def validate_inference_mode(mode: str) -> None:
    """验证推理模式合法性

    Args:
        mode: 推理模式

    Raises:
        InputValidationError: 模式不支持时抛出
    """
    validate_enum(mode, ["base", "multithread", "high_res"], "inference_mode")
