#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常处理模块

定义项目级自定义异常类
"""


class InferenceError(Exception):
    """推理基础异常

    Attributes:
        message: 错误信息
        error_code: 错误码
        original_error: 原始异常对象
        details: 附加详细信息
    """
    def __init__(self, message: str, error_code: int = 1000, original_error: Exception = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.original_error = original_error
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """格式化错误信息，包含所有详细内容"""
        parts = [f"[{self.error_code}] {self.message}"]

        if self.original_error:
            parts.append(f"\n原始错误: {type(self.original_error).__name__}: {str(self.original_error)}")

        if self.details:
            parts.append("\n详细信息:")
            for k, v in self.details.items():
                parts.append(f"  {k}: {v}")

        return "".join(parts)


class ModelLoadError(InferenceError):
    """模型加载异常"""
    def __init__(self, message: str, error_code: int = 1001, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class DeviceError(InferenceError):
    """设备异常"""
    def __init__(self, message: str, error_code: int = 1002, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class PreprocessError(InferenceError):
    """预处理异常"""
    def __init__(self, message: str, error_code: int = 1003, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class PostprocessError(InferenceError):
    """后处理异常"""
    def __init__(self, message: str, error_code: int = 1004, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class ConfigurationError(InferenceError):
    """配置异常"""
    def __init__(self, message: str, error_code: int = 1005, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class MemoryError(InferenceError):
    """内存操作异常"""
    def __init__(self, message: str, error_code: int = 1006, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class ACLError(InferenceError):
    """ACL操作异常"""
    def __init__(self, message: str, error_code: int = 1007, original_error: Exception = None, acl_ret: int = 0, details: dict = None):
        if details is None:
            details = {}
        details["acl_return_code"] = acl_ret
        super().__init__(message, error_code, original_error, details)


class ThreadError(InferenceError):
    """多线程操作异常"""
    def __init__(self, message: str, error_code: int = 1008, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)


class InputValidationError(InferenceError):
    """输入参数验证异常"""
    def __init__(self, message: str, error_code: int = 1009, original_error: Exception = None, details: dict = None):
        super().__init__(message, error_code, original_error, details)
