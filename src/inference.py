#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理核心类

提供统一的模型推理功能，支持：
- 标准推理
- 多线程推理
- 高分辨率图像分块推理
"""

import os
import time
import threading
import queue
import ctypes
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Optional, Union, Tuple, List, Dict, Any

try:
    import acl
    from utils.acl_utils import (
        init_acl, destroy_acl, 
        load_model, unload_model,
        malloc_device, malloc_host,
        free_device, free_host,
        create_dataset, destroy_dataset,
        get_last_error_msg,
        MEMCPY_HOST_TO_DEVICE, MEMCPY_DEVICE_TO_HOST
    )
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.memory_pool import MemoryPool
from utils.exceptions import (
    InferenceError,
    ModelLoadError,
    DeviceError,
    PreprocessError,
    PostprocessError,
    MemoryError,
    ACLError,
    InputValidationError
)
from utils.validators import (
    validate_positive_integer,
    validate_image_backend,
    validate_file_path,
    validate_enum
)

# 常量定义
DEFAULT_SLEEP_INTERVAL = 0.01
DEFAULT_WORKER_ERROR_SLEEP = 0.1
DEFAULT_OVERLAP_RATIO = 0.25

# 获取日志记录器
logger = LoggerConfig.setup_logger('ascend_inference.inference', format_type='text')


class Inference:
    """统一推理类"""
    
    def __init__(self, config: Optional[Config] = None, batch_size: int = 1):
        """初始化推理类

        Args:
            config: Config 实例，None 则使用默认配置
            batch_size: 批处理大小，默认1
        """
        # 验证参数
        validate_positive_integer(batch_size, "batch_size", min_val=1)

        self.config = config or Config()
        self.model_path = self.config.model_path
        self.device_id = self.config.device_id
        self.resolution = self.config.resolution
        self.input_width, self.input_height = Config.get_resolution(self.resolution)
        self.batch_size = batch_size  # 批处理大小

        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        self.input_size = 0  # 单张输入大小
        self.output_size = 0  # 单张输出大小
        self.batch_input_size = 0  # 批量输入总大小
        self.batch_output_size = 0  # 批量输出总大小
        self.input_buffer = None
        self.output_buffer = None
        self.output_host = None
        self.input_dataset = None
        self.output_dataset = None

        self.initialized = False
        self.model_loaded = False
        self.input_host_pool = None
    
    def init(self) -> bool:
        """初始化 ACL 和加载模型

        Returns:
            bool: 是否成功
        """
        if not HAS_ACL:
            error_msg = "ACL 库不可用，仅在昇腾设备上可用。如在非昇腾设备上测试，请安装 ACL 库或跳过 ACL 相关功能。"
            logger.error(error_msg)
            raise ACLError(
                error_msg,
                error_code=2001,
                details={"device_id": self.device_id}
            )

        self.context, self.stream = init_acl(self.device_id)
        if not self.context:
            error_msg = f"ACL 初始化失败 (device_id={self.device_id})"
            logger.error(error_msg)
            logger.error("可能原因：1. 未在昇腾设备上运行 2. 设备 ID 不正确 3. ACL 环境未正确配置")
            raise DeviceError(
                error_msg,
                error_code=2002,
                details={
                    "device_id": self.device_id,
                    "possible_causes": ["未在昇腾设备上运行", "设备 ID 不正确", "ACL 环境未正确配置"]
                }
            )
        
        self.initialized = True

        try:
            if not self._load_model():
                error_msg = f"模型加载失败"
                logger.error(error_msg)
                logger.error(f"可能原因：1. 模型文件不存在：{self.model_path} 2. 模型文件损坏")
                self.destroy()
                raise ModelLoadError(
                    error_msg,
                    error_code=2003,
                    details={
                        "model_path": self.model_path,
                        "possible_causes": ["模型文件不存在", "模型文件损坏"]
                    }
                )
        except Exception as e:
            self.destroy()
            raise e

        # 初始化输入内存池，复用主机内存
        self.input_host_pool = MemoryPool(self.batch_input_size, device='host', max_buffers=5)
        self.output_host = malloc_host(self.batch_output_size)
        if not self.output_host:
            error_msg = "分配主机输出内存失败"
            logger.error(error_msg)
            self.destroy()
            raise MemoryError(
                error_msg,
                error_code=2004,
                details={"size": self.batch_output_size}
            )

        return True
    

    
    def _load_model(self) -> bool:
        """加载模型

        Returns:
            bool: 是否加载成功
        """
        if self.model_loaded:
            return True
        
        if not os.path.exists(self.model_path):
            error_msg = f"模型文件不存在：{self.model_path}"
            logger.error(error_msg)
            raise ModelLoadError(
                error_msg,
                error_code=2101,
                details={"model_path": self.model_path}
            )

        try:
            result = load_model(self.model_path)
            if result[0] is None:
                error_msg = "ACL加载模型失败"
                logger.error(error_msg)
                raise ACLError(
                    error_msg,
                    error_code=2102,
                    details={"model_path": self.model_path}
                )
        except Exception as e:
            error_msg = f"加载模型异常：{str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(
                error_msg,
                error_code=2103,
                original_error=e,
                details={"model_path": self.model_path}
            ) from e
        
        self.model_id, self.model_desc, self.input_size, self.output_size = result

        # 一致性检查：验证配置分辨率与模型输入尺寸是否匹配
        self._validate_model_input_consistency()

        # 计算批处理总大小
        self.batch_input_size = self.input_size * self.batch_size
        self.batch_output_size = self.output_size * self.batch_size

        self.input_buffer = malloc_device(self.batch_input_size)
        if not self.input_buffer:
            error_msg = "分配输入设备内存失败"
            logger.error(error_msg)
            raise MemoryError(
                error_msg,
                error_code=2104,
                details={"size": self.batch_input_size, "type": "device"}
            )

        self.output_buffer = malloc_device(self.batch_output_size)
        if not self.output_buffer:
            error_msg = "分配输出设备内存失败"
            logger.error(error_msg)
            raise MemoryError(
                error_msg,
                error_code=2105,
                details={"size": self.batch_output_size, "type": "device"}
            )
        
        self.input_dataset = create_dataset(self.input_buffer, self.batch_input_size, "输入数据集")
        if not self.input_dataset:
            error_msg = "创建输入数据集失败"
            logger.error(error_msg)
            logger.error("可能原因：1. 输入缓冲区是否分配成功 2. 输入数据大小是否正确 3. ACL 库是否正常工作")
            raise ACLError(
                error_msg,
                error_code=2106,
                details={
                    "buffer_size": self.batch_input_size,
                    "possible_causes": ["输入缓冲区分配失败", "输入数据大小不正确", "ACL 库异常"]
                }
            )
        
        self.output_dataset = create_dataset(self.output_buffer, self.batch_output_size, "输出数据集")
        if not self.output_dataset:
            error_msg = "创建输出数据集失败"
            logger.error(error_msg)
            logger.error("可能原因：1. 输出缓冲区是否分配成功 2. 输出数据大小是否正确 3. ACL 库是否正常工作")
            raise ACLError(
                error_msg,
                error_code=2107,
                details={
                    "buffer_size": self.batch_output_size,
                    "possible_causes": ["输出缓冲区分配失败", "输出数据大小不正确", "ACL 库异常"]
                }
            )
        
        self.model_loaded = True
        logger.info(f"模型加载成功：{self.model_path}")
        return True

    def _validate_model_input_consistency(self) -> None:
        """验证配置分辨率与模型输入尺寸的一致性

        Raises:
            InputValidationError: 分辨率不匹配时抛出
        """
        if not HAS_ACL:
            return

        try:
            from utils.acl_utils import get_model_input_info

            # 获取模型输入信息
            model_batch, model_channels, model_height, model_width, model_data_type, bytes_per_element = \
                get_model_input_info(self.model_desc, 0)

            # 计算预期的输入大小
            expected_input_size = self.input_width * self.input_height * model_channels * bytes_per_element

            # 检查大小是否匹配
            if self.input_size != expected_input_size:
                error_msg = "配置分辨率与模型输入尺寸不匹配"
                logger.error(error_msg)
                logger.error(f"配置: {self.input_width}x{self.input_height}, 模型: {model_width}x{model_height}")

                raise InputValidationError(
                    error_msg,
                    error_code=3120,
                    details={
                        "config_resolution": self.resolution,
                        "config_width": self.input_width,
                        "config_height": self.input_height,
                        "model_batch": model_batch,
                        "model_channels": model_channels,
                        "model_width": model_width,
                        "model_height": model_height,
                        "model_data_type": model_data_type,
                        "bytes_per_element": bytes_per_element,
                        "expected_input_size": expected_input_size,
                        "actual_model_input_size": self.input_size,
                        "calculation": f"{self.input_width} × {self.input_height} × {model_channels} × {bytes_per_element} = {expected_input_size}",
                        "model_shape": f"NCHW({model_batch}, {model_channels}, {model_height}, {model_width})",
                        "suggestion": f"请检查配置的 resolution 参数。模型期望输入为 {model_width}x{model_height} 的 {model_channels}-通道图像"
                    }
                )

            # 检查具体的尺寸是否匹配（更严格的检查）
            if self.input_width != model_width or self.input_height != model_height:
                error_msg = "配置分辨率尺寸与模型输入尺寸不匹配"
                logger.error(error_msg)
                logger.error(f"配置: {self.input_width}x{self.input_height}, 模型: {model_width}x{model_height}")

                raise InputValidationError(
                    error_msg,
                    error_code=3121,
                    details={
                        "config_resolution": self.resolution,
                        "config_width": self.input_width,
                        "config_height": self.input_height,
                        "model_width": model_width,
                        "model_height": model_height,
                        "suggestion": f"请将 resolution 设置为 '{model_width}x{model_height}'"
                    }
                )

            logger.info(
                f"模型输入一致性检查通过: 配置({self.input_width}x{self.input_height}) "
                f"== 模型({model_width}x{model_height}), 通道数={model_channels}, 数据类型={model_data_type}"
            )

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
            # 检查失败时抛出异常，强制用户解决问题
            raise InputValidationError(
                "无法验证模型输入一致性",
                error_code=3122,
                original_error=e,
                details={"original_error": str(e)}
            ) from e

    def _load_image(self, image_data: Union[str, np.ndarray, PILImage], backend: str = 'opencv') -> Union[PILImage, np.ndarray]:
        """加载图像

        Args:
            image_data: 图像路径或 numpy 数组或 PIL 图像
            backend: 图像读取后端，默认优先使用opencv

        Returns:
            PIL.Image.Image 或 numpy.ndarray: 图像数据（保持原始格式以便后续处理）

        Raises:
            PreprocessError: 图像加载失败时抛出
        """
        try:
            if isinstance(image_data, str) and not os.path.exists(image_data):
                error_msg = f"图像文件不存在：{image_data}"
                logger.error(error_msg)
                raise PreprocessError(
                    error_msg,
                    error_code=2201,
                    details={"image_path": image_data, "backend": backend}
                )

            if isinstance(image_data, str) and backend == 'opencv' and HAS_OPENCV:
                image = cv2.imread(image_data)
                if image is None:
                    error_msg = f"OpenCV 读取图像失败：{image_data}"
                    logger.error(error_msg)
                    raise PreprocessError(
                        error_msg,
                        error_code=2202,
                        details={"image_path": image_data, "backend": backend}
                    )
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_data, str):
                image = Image.open(image_data).convert('RGB')
            else:
                # 已经是numpy数组或PIL图像，直接返回
                image = image_data

            return image
        except Exception as e:
            if isinstance(e, InferenceError):
                raise e
            error_msg = f"读取图像异常：{str(e)}"
            logger.error(error_msg)
            logger.debug(f"异常详情：{image_data}")
            raise PreprocessError(
                error_msg,
                error_code=2203,
                original_error=e,
                details={"image_data": str(image_data), "backend": backend}
            ) from e
    
    def _resize_image(self, image: Union[PILImage, np.ndarray], backend: str = 'opencv') -> np.ndarray:
        """调整图像大小并转换为 numpy 数组

        Args:
            image: PIL Image 或 numpy 数组
            backend: 图像处理后端，默认优先使用opencv

        Returns:
            numpy.ndarray: resize 后的 RGB 图像数组

        Raises:
            PreprocessError: 图像缩放失败时抛出
        """
        try:
            if backend == 'opencv' and HAS_OPENCV:
                return cv2.resize(image, (self.input_width, self.input_height))
            else:
                resized_image = image.resize((self.input_width, self.input_height))
                return np.array(resized_image)
        except Exception as e:
            error_msg = f"调整图像大小异常：{str(e)}"
            logger.error(error_msg)
            logger.debug(f"图像形状：{getattr(image, 'shape', 'N/A')}, 目标尺寸：({self.input_width}, {self.input_height})")
            raise PreprocessError(
                error_msg,
                error_code=2204,
                original_error=e,
                details={
                    "image_shape": getattr(image, 'shape', 'N/A'),
                    "target_size": (self.input_width, self.input_height),
                    "backend": backend
                }
            ) from e
    
    def preprocess(self, image_data: Union[str, np.ndarray, PILImage], backend: str = 'opencv') -> None:
        """预处理图像

        Args:
            image_data: 图像数据
            backend: 图像读取后端，默认优先使用opencv

        Raises:
            ACLError: ACL库不可用
            PreprocessError: 预处理失败时抛出
        """
        # 验证参数
        validate_image_backend(backend)
        if isinstance(image_data, str):
            validate_file_path(image_data, must_exist=True)

        if not HAS_ACL:
            error_msg = "ACL 库不可用"
            logger.error(error_msg)
            raise ACLError(
                error_msg,
                error_code=2301
            )

        image = self._load_image(image_data, backend)
        
        image = self._resize_image(image, backend)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).flatten()

        try:
            input_host = self.input_host_pool.allocate()
            if not input_host:
                error_msg = "分配主机输入内存失败"
                logger.error(error_msg)
                raise MemoryError(
                    error_msg,
                    error_code=2302,
                    details={"size": self.input_size}
                )

            # 内存边界检查：确保数据不会溢出
            input_buffer_size = self.input_size
            actual_data_size = image.nbytes if hasattr(image, 'nbytes') else input_buffer_size

            if actual_data_size > input_buffer_size:
                error_msg = f"图像数据大小({actual_data_size})超过预期输入大小({input_buffer_size})，存在内存越界风险"
                logger.error(error_msg)
                logger.debug(f"图像形状：{image.shape}, 预期输入大小：{input_buffer_size}")
                raise PreprocessError(
                    error_msg,
                    error_code=2305,
                    details={
                        "actual_data_size": actual_data_size,
                        "expected_size": input_buffer_size,
                        "image_shape": getattr(image, 'shape', 'N/A')
                    }
                )

            # 安全拷贝数据到主机内存，使用min确保不越界
            safe_copy_size = min(input_buffer_size, actual_data_size)
            ctypes.memmove(input_host, image.ctypes.data, safe_copy_size)

            ret = acl.rt.memcpy(self.input_buffer, self.batch_input_size, input_host, self.input_size, MEMCPY_HOST_TO_DEVICE)
            self.input_host_pool.free(input_host)

            if ret != 0:
                err_msg = get_last_error_msg()
                error_msg = f"内存拷贝失败，错误码：{ret}，错误信息：{err_msg}"
                logger.error(error_msg)
                raise ACLError(
                    error_msg,
                    error_code=2303,
                    acl_ret=ret,
                    details={"error_msg": err_msg}
                )
        except Exception as e:
            if isinstance(e, InferenceError):
                raise e
            error_msg = f"预处理异常：{str(e)}"
            logger.error(error_msg)
            logger.debug(f"异常详情：{image_data}")
            raise PreprocessError(
                error_msg,
                error_code=2304,
                original_error=e,
                details={"image_data": str(image_data)}
            ) from e

        logger.debug("预处理完成")

    def preprocess_batch(self, image_data_list: List[Union[str, np.ndarray, PILImage]], backend: str = 'opencv') -> bool:
        """批量预处理图像

        Args:
            image_data_list: 图像数据列表
            backend: 图像读取后端

        Returns:
            bool: 是否成功
        """
        # 验证参数
        validate_image_backend(backend)
        validate_positive_integer(len(image_data_list), "image_data_list length", min_val=1)
        if len(image_data_list) > self.batch_size:
            logger.error(f"输入图像数量({len(image_data_list)})超过批处理大小({self.batch_size})")
            return False

        for i, image_data in enumerate(image_data_list):
            if isinstance(image_data, str):
                validate_file_path(image_data, must_exist=True)

        if not HAS_ACL:
            logger.error("ACL 库不可用")
            return False

        try:
            input_host = self.input_host_pool.allocate()
            if not input_host:
                logger.error("分配主机输入内存失败")
                return False

            # 逐个处理图像并拷贝到连续内存
            offset = 0
            for i, image_data in enumerate(image_data_list):
                image = self._load_image(image_data, backend)
                if image is None:
                    logger.error(f"图像加载失败：{image_data}")
                    self.input_host_pool.free(input_host)
                    return False

                image = self._resize_image(image, backend)
                if image is None:
                    logger.error(f"图像缩放失败：{image_data}")
                    self.input_host_pool.free(input_host)
                    return False

                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=2)

                image = image.astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1)).flatten()

                # 内存边界检查
                actual_data_size = image.nbytes if hasattr(image, 'nbytes') else self.input_size
                if actual_data_size > self.input_size:
                    error_msg = f"批处理图像数据大小({actual_data_size})超过预期输入大小({self.input_size})，存在内存越界风险"
                    logger.error(error_msg)
                    raise PreprocessError(
                        error_msg,
                        error_code=2306,
                        details={
                            "actual_data_size": actual_data_size,
                            "expected_size": self.input_size,
                            "image_index": i
                        }
                    )

                # 检查总偏移量是否会越界
                if offset + self.input_size > self.batch_input_size:
                    error_msg = f"批处理总偏移量({offset + self.input_size})超过批量输入大小({self.batch_input_size}), 存在内存越界风险"
                    logger.error(error_msg)
                    raise PreprocessError(
                        error_msg,
                        error_code=2307,
                        details={
                            "total_offset": offset + self.input_size,
                            "batch_input_size": self.batch_input_size,
                            "image_index": i
                        }
                    )

                # 安全拷贝到偏移位置
                safe_copy_size = min(self.input_size, actual_data_size)
                ctypes.memmove(
                    ctypes.c_void_p(int(input_host) + offset),
                    image.ctypes.data,
                    safe_copy_size
                )
                offset += self.input_size

            # 拷贝到设备
            ret = acl.rt.memcpy(
                self.input_buffer, self.batch_input_size,
                input_host, self.batch_input_size,
                MEMCPY_HOST_TO_DEVICE
            )
            self.input_host_pool.free(input_host)

            if ret != 0:
                err_msg = get_last_error_msg()
                logger.error(f"批量内存拷贝失败，错误码：{ret}，错误信息：{err_msg}")
                return False

        except Exception as e:
            logger.error(f"批量预处理异常：{e}")
            return False

        logger.debug(f"批量预处理完成，共{len(image_data_list)}张图像")
        return True
    
    def execute(self) -> None:
        """执行模型推理

        Raises:
            ACLError: ACL操作失败时抛出
            RuntimeError: 模型未加载时抛出
        """
        if not HAS_ACL:
            error_msg = "ACL 库不可用"
            logger.error(error_msg)
            raise ACLError(
                error_msg,
                error_code=2401
            )

        if not self.model_loaded:
            error_msg = "模型未加载"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)

            if ret != 0:
                err_msg = get_last_error_msg()
                error_msg = f"推理执行失败，错误码：{ret}，错误信息：{err_msg}"
                logger.error(error_msg)
                raise ACLError(
                    error_msg,
                    error_code=2402,
                    acl_ret=ret,
                    details={"error_msg": err_msg}
                )

            ret = acl.rt.synchronize_stream(self.stream)
            if ret != 0:
                err_msg = get_last_error_msg()
                error_msg = f"Stream 同步失败，错误码：{ret}，错误信息：{err_msg}"
                logger.error(error_msg)
                raise ACLError(
                    error_msg,
                    error_code=2403,
                    acl_ret=ret,
                    details={"error_msg": err_msg}
                )

            logger.debug("推理执行成功")
        except Exception as e:
            if isinstance(e, InferenceError):
                raise e
            error_msg = f"推理执行异常：{str(e)}"
            logger.error(error_msg)
            raise ACLError(
                error_msg,
                error_code=2404,
                original_error=e
            ) from e
    
    def get_result(self) -> np.ndarray:
        """获取推理结果

        Returns:
            np.ndarray: 推理结果

        Raises:
            PostprocessError: 获取结果失败时抛出
            RuntimeError: 模型未加载或输出内存未分配时抛出
        """
        if not self.model_loaded or not self.output_host:
            error_msg = "模型未加载或输出内存未分配"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        ret = acl.rt.memcpy(self.output_host, self.batch_output_size, self.output_buffer,
                          self.output_size, MEMCPY_DEVICE_TO_HOST)
        if ret != 0:
            err_msg = get_last_error_msg()
            error_msg = f"获取结果失败，错误码：{ret}，错误信息：{err_msg}"
            logger.error(error_msg)
            raise PostprocessError(
                error_msg,
                error_code=2501,
                acl_ret=ret,
                details={"error_msg": err_msg}
            )

        # Convert memory address to numpy array using ctypes
        buffer = ctypes.cast(self.output_host, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(buffer, shape=(self.output_size // 4,))

    def get_result_batch(self) -> Optional[List[np.ndarray]]:
        """获取批量推理结果

        Returns:
            List[np.ndarray]: 每个元素是一张图像的推理结果
        """
        if not self.model_loaded or not self.output_host:
            return None

        ret = acl.rt.memcpy(
            self.output_host, self.batch_output_size,
            self.output_buffer, self.batch_output_size,
            MEMCPY_DEVICE_TO_HOST
        )
        if ret != 0:
            err_msg = get_last_error_msg()
            logger.error(f"获取批量结果失败，错误码：{ret}，错误信息：{err_msg}")
            return None

        # 逐个解析每个batch的结果
        results = []
        single_output_size = self.output_size // 4  # float是4字节
        buffer = ctypes.cast(self.output_host, ctypes.POINTER(ctypes.c_float))

        for i in range(self.batch_size):
            offset = i * single_output_size
            result = np.ctypeslib.as_array(
                ctypes.cast(ctypes.addressof(buffer[offset]), ctypes.POINTER(ctypes.c_float)),
                shape=(single_output_size,)
            )
            results.append(result)

        return results
    
    def run_inference(self, image_data: Union[str, np.ndarray, PILImage], backend: str = 'opencv') -> np.ndarray:
        """执行完整推理流程

        Args:
            image_data: 图像数据
            backend: 图像读取后端，默认优先使用opencv

        Returns:
            np.ndarray: 推理结果

        Raises:
            PreprocessError: 预处理失败
            ACLError: 推理执行失败
            PostprocessError: 结果获取失败
        """
        # 验证参数
        validate_image_backend(backend)
        if isinstance(image_data, str):
            validate_file_path(image_data, must_exist=True)

        self.preprocess(image_data, backend)
        self.execute()
        return self.get_result()

    def run_inference_batch(self, image_data_list: List[Union[str, np.ndarray, PILImage]], backend: str = 'opencv') -> Optional[List[np.ndarray]]:
        """执行完整批量推理流程

        Args:
            image_data_list: 图像数据列表
            backend: 图像读取后端

        Returns:
            List[np.ndarray]: 推理结果列表，失败返回None
        """
        # 验证参数
        validate_image_backend(backend)
        validate_positive_integer(len(image_data_list), "image_data_list length", min_val=1)
        for image_data in image_data_list:
            if isinstance(image_data, str):
                validate_file_path(image_data, must_exist=True)

        if not self.preprocess_batch(image_data_list, backend):
            return None

        self.execute()

        return self.get_result_batch()
    
    def destroy(self) -> None:
        """销毁资源"""
        if not HAS_ACL:
            return
        
        if self.stream and self.context:
            try:
                acl.rt.set_context(self.context)
                acl.rt.synchronize_stream(self.stream)
            except Exception as e:
                logger.warning(f"流同步失败：{e}")
        
        if self.input_dataset:
            if not destroy_dataset(self.input_dataset, self.context):
                logger.warning("输入数据集销毁失败")
            self.input_dataset = None
        
        if self.output_dataset:
            if not destroy_dataset(self.output_dataset, self.context):
                logger.warning("输出数据集销毁失败")
            self.output_dataset = None
        
        if self.output_host:
            free_host(self.output_host)
            self.output_host = None
        
        if self.input_buffer:
            free_device(self.input_buffer)
            self.input_buffer = None
        
        if self.output_buffer:
            free_device(self.output_buffer)
            self.output_buffer = None
        
        if self.model_id:
            if not unload_model(self.model_id, self.model_desc):
                logger.warning("模型卸载失败")
            self.model_id = None
            self.model_desc = None
        
        if self.initialized:
            if not destroy_acl(self.context, self.stream, self.device_id):
                logger.warning("ACL 资源销毁失败")
            self.context = None
            self.stream = None
        
        self.initialized = False
        # 清理内存池
        if self.input_host_pool:
            self.input_host_pool.cleanup()
            self.input_host_pool = None

        logger.debug("资源销毁完成")
        self.model_loaded = False
    
    def __enter__(self) -> 'Inference':
        if not self.init():
            raise RuntimeError("初始化失败，请查看上方的详细错误信息")
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        self.destroy()

    def __del__(self) -> None:
        """析构函数，检测资源泄漏"""
        if self.initialized or self.model_loaded:
            logger.warning(
                f"资源泄漏检测：Inference实例未正确调用destroy()方法，资源可能泄漏。"
                f"model_path={self.model_path}, device_id={self.device_id}"
            )
            # 尝试自动释放
            try:
                self.destroy()
            except Exception as e:
                logger.error(f"自动释放资源失败: {e}")


class MultithreadInference:
    """多线程推理管理器（工作窃取负载均衡+动态算力调整优化）"""

    def __init__(self, config: Optional[Config] = None, auto_scale: bool = True):
        """初始化多线程推理

        Args:
            config: Config 实例
            auto_scale: 是否启用自动算力调整
        """
        self.config = config or Config()
        self.initial_num_threads = min(self.config.num_threads, Config.MAX_AI_CORES)
        self.num_threads = self.initial_num_threads
        self.model_path = self.config.model_path
        self.resolution = self.config.resolution
        self.auto_scale = auto_scale

        self.workers = []
        self.task_queues: List[queue.Queue] = []  # 每个worker独立的任务队列
        self.result_queue = queue.Queue()
        self.threads = []
        self.running = False
        self.monitor_thread = None
        self.worker_states: List[bool] = []  # worker运行状态，True表示忙碌

        # 自动调整参数
        self.min_threads = 1
        self.max_threads = Config.MAX_AI_CORES
        self.scale_up_threshold = 0.7  # 队列负载超过70%时扩容
        self.scale_down_threshold = 0.2  # 队列负载低于20%时缩容
        self.scale_interval = 5  # 调整间隔（秒）
    
    def _init_workers(self) -> bool:
        """初始化工作线程

        Returns:
            bool: 是否初始化成功
        """
        for i in range(self.num_threads):
            device_id = i % Config.MAX_AI_CORES
            config = Config(
                model_path=self.model_path,
                device_id=device_id,
                resolution=self.resolution
            )
            worker = Inference(config)
            if worker.init():
                self.workers.append(worker)
                logger.info(f"Worker {i} 初始化成功 (device: {device_id})")
            else:
                logger.error(f"Worker {i} 初始化失败")
        
        return len(self.workers) > 0
    
    def _worker_thread(self, worker_id: int, worker: Inference) -> None:
        """工作线程函数（支持工作窃取）

        Args:
            worker_id: worker编号
            worker: 推理工作实例
        """
        if HAS_ACL and worker.context:
            acl.rt.set_context(worker.context)

        while self.running:
            try:
                # 先尝试从自己的队列取任务
                task = self.task_queues[worker_id].get(block=False)
                if task is None:
                    break

                image_path, backend = task
                result = worker.run_inference(image_path, backend)
                self.result_queue.put((image_path, result))
                self.task_queues[worker_id].task_done()

            except queue.Empty:
                # 自己队列为空，尝试从其他worker队列偷取任务
                stolen = False
                for other_id in range(len(self.task_queues)):
                    if other_id == worker_id:
                        continue
                    try:
                        task = self.task_queues[other_id].get(block=False)
                        if task is not None:
                            image_path, backend = task
                            result = worker.run_inference(image_path, backend)
                            self.result_queue.put((image_path, result))
                            self.task_queues[other_id].task_done()
                            stolen = True
                            break
                    except queue.Empty:
                        continue

                if not stolen:
                    time.sleep(DEFAULT_SLEEP_INTERVAL)
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                logger.debug(f"Worker: {worker}, task: {task if 'task' in locals() else 'unknown'}")
                time.sleep(DEFAULT_WORKER_ERROR_SLEEP)
    
    def start(self) -> bool:
        """启动多线程

        Returns:
            bool: 是否启动成功
        """
        if not self.workers:
            if not self._init_workers():
                return False
        
        self.running = True

        for worker_id, worker in enumerate(self.workers):
            thread = threading.Thread(target=self._worker_thread, args=(worker_id, worker))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        return True
    
    def add_task(self, image_path: Union[str, np.ndarray, PILImage], backend: Optional[str] = None) -> None:
        """添加推理任务（均匀分配到各worker队列）

        Args:
            image_path: 图像路径或图像数据
            backend: 图像处理后端，None则使用配置默认值
        """
        if backend is None:
            backend = self.config.backend

        # 验证参数
        validate_image_backend(backend)
        if isinstance(image_path, str):
            validate_file_path(image_path, must_exist=True)

        # 轮询分配任务到各个worker队列，实现初始负载均衡
        worker_id = len(self.result_queue.queue) % len(self.task_queues)
        self.task_queues[worker_id].put((image_path, backend))
    
    def get_results(self) -> List[Tuple[Union[str, int], Optional[np.ndarray]]]:
        """获取推理结果

        Returns:
            List[Tuple]: 每个元素是(图像标识, 推理结果)的元组
        """
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get(block=False)
                results.append(result)
                self.result_queue.task_done()
            except queue.Empty:
                break
        return results
    
    def wait_completion(self) -> None:
        """等待所有任务完成"""
        # 等待所有worker队列的任务完成
        for q in self.task_queues:
            q.join()
        while not self.result_queue.empty():
            time.sleep(DEFAULT_SLEEP_INTERVAL)
    
    def stop(self) -> None:
        """停止多线程"""
        self.running = False

        # 向所有队列发送结束信号
        for q in self.task_queues:
            q.put(None)

        for thread in self.threads:
            thread.join(timeout=5)

        for worker in self.workers:
            worker.destroy()

        if HAS_ACL:
            acl.finalize()

    def __del__(self) -> None:
        """析构函数，检测资源泄漏"""
        if self.running:
            logger.warning(
                f"资源泄漏检测：MultithreadInference实例未正确调用stop()方法，资源可能泄漏。"
                f"model_path={self.model_path}, num_threads={self.num_threads}"
            )
            # 尝试自动停止
            try:
                self.stop()
            except Exception as e:
                logger.error(f"自动停止多线程失败: {e}")


class PipelineInference:
    """流水线推理管理器（预处理+推理+后处理并行执行）"""

    def __init__(self, config: Optional[Config] = None, batch_size: int = 4, queue_size: int = 10):
        """初始化流水线推理

        Args:
            config: 配置实例
            batch_size: 批处理大小
            queue_size: 队列最大长度，用于流量控制
        """
        self.config = config or Config()
        self.batch_size = batch_size
        self.queue_size = queue_size

        # 阶段队列
        self.preprocess_queue = queue.Queue(maxsize=queue_size)
        self.infer_queue = queue.Queue(maxsize=queue_size)
        self.postprocess_queue = queue.Queue(maxsize=queue_size)

        # 线程控制
        self.running = False
        self.preprocess_threads: List[threading.Thread] = []
        self.infer_threads: List[threading.Thread] = []
        self.postprocess_thread: Optional[threading.Thread] = None

        # 推理实例
        self.infer_instances: List[Inference] = []

    def start(self, num_preprocess_threads: int = 2, num_infer_threads: int = 1) -> bool:
        """启动流水线

        Args:
            num_preprocess_threads: 预处理线程数
            num_infer_threads: 推理线程数

        Returns:
            bool: 是否启动成功
        """
        self.running = True

        # 初始化推理实例
        for i in range(num_infer_threads):
            device_id = i % Config.MAX_AI_CORES
            config = Config(
                model_path=self.config.model_path,
                device_id=device_id,
                resolution=self.config.resolution
            )
            infer = Inference(config, batch_size=self.batch_size)
            if not infer.init():
                logger.error(f"推理实例{i}初始化失败")
                self.stop()
                return False
            self.infer_instances.append(infer)

        # 启动预处理线程
        for i in range(num_preprocess_threads):
            t = threading.Thread(target=self._preprocess_worker, args=(i,))
            t.daemon = True
            t.start()
            self.preprocess_threads.append(t)
            logger.info(f"预处理线程{i}已启动")

        # 启动推理线程
        for i in range(num_infer_threads):
            t = threading.Thread(target=self._infer_worker, args=(i,))
            t.daemon = True
            t.start()
            self.infer_threads.append(t)
            logger.info(f"推理线程{i}已启动")

        # 启动后处理线程
        self.postprocess_thread = threading.Thread(target=self._postprocess_worker)
        self.postprocess_thread.daemon = True
        self.postprocess_thread.start()
        logger.info("后处理线程已启动")

        logger.info(f"流水线启动成功：预处理线程={num_preprocess_threads}, 推理线程={num_infer_threads}, 批大小={self.batch_size}")
        return True

    def _preprocess_worker(self, worker_id: int) -> None:
        """预处理工作线程"""
        while self.running:
            try:
                task = self.preprocess_queue.get(timeout=0.1)
                if task is None:
                    break

                batch_id, image_list, callback = task
                logger.debug(f"预处理线程{worker_id}处理批次{batch_id}，共{len(image_list)}张图像")

                # 拆分批次
                for i in range(0, len(image_list), self.batch_size):
                    batch = image_list[i:i+self.batch_size]
                    # 填充到批大小
                    while len(batch) < self.batch_size:
                        batch.append(image_list[0])

                    # 送到推理队列
                    self.infer_queue.put((batch_id, i // self.batch_size, batch, callback))

                self.preprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"预处理线程{worker_id}异常: {e}")
                time.sleep(0.1)

    def _infer_worker(self, worker_id: int) -> None:
        """推理工作线程"""
        infer = self.infer_instances[worker_id]
        if HAS_ACL and infer.context:
            acl.rt.set_context(infer.context)

        while self.running:
            try:
                task = self.infer_queue.get(timeout=0.1)
                if task is None:
                    break

                batch_id, sub_batch_id, image_batch, callback = task
                logger.debug(f"推理线程{worker_id}处理批次{batch_id}-{sub_batch_id}")

                results = infer.run_inference_batch(image_batch, self.config.backend)
                if results:
                    # 送到后处理队列
                    self.postprocess_queue.put((batch_id, sub_batch_id, results, callback))

                self.infer_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"推理线程{worker_id}异常: {e}")
                time.sleep(0.1)

    def _postprocess_worker(self) -> None:
        """后处理工作线程"""
        # 按批次聚合结果
        batch_results: Dict[int, Dict[int, List[np.ndarray]]] = {}

        while self.running:
            try:
                task = self.postprocess_queue.get(timeout=0.1)
                if task is None:
                    break

                batch_id, sub_batch_id, results, callback = task
                logger.debug(f"后处理批次{batch_id}-{sub_batch_id}")

                # 保存子批次结果
                if batch_id not in batch_results:
                    batch_results[batch_id] = {}
                batch_results[batch_id][sub_batch_id] = results

                # 检查是否所有子批次都完成
                # 这里简化处理，实际应用中需要知道总子批次数量
                # 目前直接调用回调函数
                if callback:
                    callback(batch_id, sub_batch_id, results)

                self.postprocess_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"后处理线程异常: {e}")
                time.sleep(0.1)

    def submit(self, image_list: List[Union[str, np.ndarray, PILImage]], callback: Optional[callable] = None) -> int:
        """提交推理任务

        Args:
            image_list: 图像列表
            callback: 结果回调函数，参数为 (batch_id, sub_batch_id, results)

        Returns:
            int: 批次ID
        """
        # 验证参数
        validate_positive_integer(len(image_list), "image_list length", min_val=1)
        for image_data in image_list:
            if isinstance(image_data, str):
                validate_file_path(image_data, must_exist=True)

        batch_id = int(time.time() * 1000) % 1000000
        self.preprocess_queue.put((batch_id, image_list, callback))
        return batch_id

    def wait_for_completion(self) -> None:
        """等待所有任务完成"""
        self.preprocess_queue.join()
        self.infer_queue.join()
        self.postprocess_queue.join()

    def stop(self) -> None:
        """停止流水线"""
        self.running = False

        # 发送停止信号
        for _ in range(len(self.preprocess_threads)):
            self.preprocess_queue.put(None)
        for _ in range(len(self.infer_threads)):
            self.infer_queue.put(None)
        self.postprocess_queue.put(None)

        # 等待线程结束
        for t in self.preprocess_threads:
            t.join(timeout=3)
        for t in self.infer_threads:
            t.join(timeout=3)
        if self.postprocess_thread:
            self.postprocess_thread.join(timeout=3)

        # 销毁推理实例
        for infer in self.infer_instances:
            infer.destroy()

        logger.info("流水线已停止")

    def __del__(self) -> None:
        """析构函数，检测资源泄漏"""
        if self.running:
            logger.warning(
                f"资源泄漏检测：PipelineInference实例未正确调用stop()方法，资源可能泄漏。"
                f"batch_size={self.batch_size}, queue_size={self.queue_size}"
            )
            # 尝试自动停止
            try:
                self.stop()
            except Exception as e:
                logger.error(f"自动停止流水线失败: {e}")


def split_image(image: np.ndarray, tile_size: Tuple[int, int], overlap: float) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray]:
    """将图像划分为带重叠的子块（优化版：支持权重融合）

    Args:
        image: 输入图像
        tile_size: 子块大小
        overlap: 重叠比例

    Returns:
        tiles: 子块列表
        positions: 子块位置列表 (x1, y1, w, h)
         : 权重矩阵，用于结果融合时消除重叠边缘效应
    """
    h, w = image.shape[:2]
    tile_h, tile_w = tile_size
    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)
    step_h = tile_h - overlap_h
    step_w = tile_w - overlap_w

    tiles = []
    positions = []
    weight_map = np.zeros((h, w), dtype=np.float32)

    # 创建汉宁窗权重，消除边缘硬拼接效应
    hann_2d = np.outer(np.hanning(tile_h), np.hanning(tile_w)).astype(np.float32)

    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            x1 = x
            y1 = y
            x2 = min(x + tile_w, w)
            y2 = min(y + tile_h, h)

            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)
            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)

            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
            positions.append((x1, y1, x2 - x1, y2 - y1))

            # 累积权重
            weight_map[y1:y2, x1:x2] += hann_2d[:y2-y1, :x2-x1]

    # 归一化权重，避免除以0
    weight_map[weight_map < 1e-6] = 1.0
    return tiles, positions, weight_map


class HighResInference:
    """高分辨率图像推理管理器"""
    
    def __init__(self, config: Optional[Config] = None):
        """初始化高分辨率推理

        Args:
            config: Config 实例
        """
        self.config = config or Config()
        self.num_threads = min(self.config.num_threads, Config.MAX_AI_CORES)
        self.model_path = self.config.model_path
        self.tile_size = (self.config.tile_size, self.config.tile_size)
        self.overlap = self.config.overlap / self.config.tile_size if self.config.overlap > 1 else self.config.overlap
        
        self.multithread = MultithreadInference(
            Config(
                model_path=self.model_path,
                num_threads=self.num_threads,
                resolution=f"{self.tile_size[1]}x{self.tile_size[0]}"
            )
        )
    
    def process_image(self, image_path: str, backend: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """处理高分辨率图像

        Args:
            image_path: 图像路径
            backend: 图像读取后端

        Returns:
            dict: 推理结果，失败返回None
        """
        if backend is None:
            backend = self.config.backend

        # 验证参数
        validate_image_backend(backend)
        validate_file_path(image_path, must_exist=True)

        if backend == 'opencv' and HAS_OPENCV:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像：{image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.error(f"无法读取图像：{e}")
                logger.debug(f"图像路径：{image_path}")
                return None
        
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        logger.info(f"处理图像：{image_path}, 形状：{image_array.shape}")
        
        start_time = time.time()
        tiles, positions, weight_map = split_image(image_array, self.tile_size, self.overlap)
        logger.debug(f"分割完成：{len(tiles)} 个子块，耗时：{time.time() - start_time:.2f} 秒")
        
        if not self.multithread.start():
            logger.error("无法启动推理")
            return None
        
        for i, tile in enumerate(tiles):
            self.multithread.add_task(tile, backend)
        
        self.multithread.wait_completion()
        results = self.multithread.get_results()
        
        merged_result = {
            "sub_results": [],
            "image_shape": image_array.shape,
            "num_tiles": len(tiles)
        }
        
        for tile_id, result in sorted(results, key=lambda x: x[0]):
            if result is not None:
                x, y, w, h = positions[tile_id]
                merged_result["sub_results"].append({
                    "position": (x, y, w, h),
                    "result": result.tolist()[:10]
                })
        
        logger.info(f"推理完成：{len(merged_result['sub_results'])}/{len(tiles)} 个子块成功")
        
        self.multithread.stop()
        return merged_result
