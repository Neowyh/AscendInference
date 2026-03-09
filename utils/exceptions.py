#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常处理模块

定义项目级自定义异常类
"""


class InferenceError(Exception):
    """推理基础异常"""
    pass


class ModelLoadError(InferenceError):
    """模型加载异常"""
    pass


class DeviceError(InferenceError):
    """设备异常"""
    pass


class PreprocessError(InferenceError):
    """预处理异常"""
    pass


class PostprocessError(InferenceError):
    """后处理异常"""
    pass


class ConfigurationError(InferenceError):
    """配置异常"""
    pass
