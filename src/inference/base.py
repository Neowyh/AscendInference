#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础推理类模块

提供统一的模型推理功能，支持：
- ACL初始化和销毁
- 模型加载和卸载
- 单张和批量推理
"""

import os
import ctypes
import numpy as np
from PIL.Image import Image as PILImage
from typing import Optional, Union, List, Any

try:
    import acl
    from utils.acl_utils import (
        init_acl, destroy_acl, 
        load_model, unload_model,
        malloc_device, malloc_host, free_host,
        create_dataset, destroy_dataset,
        get_last_error_msg,
        MEMCPY_HOST_TO_DEVICE
    )
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

from config import Config
from utils.logger import LoggerConfig, get_logger
from utils.exceptions import (
    InferenceError,
    ModelLoadError,
    DeviceError,
    PreprocessError,
    ACLError,
    InputValidationError
)
from utils.validators import (
    validate_positive_integer,
    validate_image_backend,
    validate_file_path
)

from .preprocessor import Preprocessor
from .executor import Executor
from .postprocessor import Postprocessor

logger = LoggerConfig.setup_logger('ascend_inference.base', format_type='text')


class Inference:
    """统一推理类"""
    
    def __init__(self, config: Optional[Config] = None, batch_size: int = 1):
        """初始化推理类

        Args:
            config: Config 实例，None 则使用默认配置
            batch_size: 批处理大小，默认1
        """
        validate_positive_integer(batch_size, "batch_size", min_val=1)

        self.config = config or Config()
        self.model_path = self.config.model_path
        self.device_id = self.config.device_id
        self.resolution = self.config.resolution
        self.input_width, self.input_height = Config.get_resolution(self.resolution)
        self.batch_size = batch_size

        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        self.input_size = 0
        self.output_size = 0
        self.batch_input_size = 0
        self.batch_output_size = 0
        self.input_buffer = None
        self.output_buffer = None
        self.output_host = None
        self.input_dataset = None
        self.output_dataset = None

        self.initialized = False
        self.model_loaded = False
        
        self.preprocessor: Optional[Preprocessor] = None
        self.executor: Optional[Executor] = None
        self.postprocessor: Optional[Postprocessor] = None
    
    def init(self) -> bool:
        """初始化 ACL 和加载模型

        Returns:
            bool: 是否成功
            
        Raises:
            ACLError: ACL库不可用
            DeviceError: 设备初始化失败
            ModelLoadError: 模型加载失败
            MemoryError: 内存分配失败
        """
        if not HAS_ACL:
            error_msg = "ACL 库不可用，仅在昇腾设备上可用。"
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
            self._load_model()
        except Exception as e:
            self.destroy()
            raise e

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

        self._init_components()
        return True
    
    def _init_components(self) -> None:
        """初始化预处理器、执行器和后处理器"""
        max_buffers = 5
        if self.config.is_strategy_enabled('memory_pool'):
            max_buffers = self.config.strategies.memory_pool.max_buffers

        self.preprocessor = Preprocessor(
            input_width=self.input_width,
            input_height=self.input_height,
            input_size=self.input_size,
            batch_size=self.batch_size
        )
        self.preprocessor.init_pool(max_buffers=max_buffers)
        
        self.executor = Executor(
            model_id=self.model_id,
            stream=self.stream,
            input_dataset=self.input_dataset,
            output_dataset=self.output_dataset,
            output_buffer=self.output_buffer,
            output_size=self.output_size,
            batch_size=self.batch_size
        )
        self.executor.init_output_buffer(self.output_host)
        
        self.postprocessor = Postprocessor()
    
    def _load_model(self) -> None:
        """加载模型

        Raises:
            ModelLoadError: 模型加载失败
            MemoryError: 内存分配失败
            ACLError: ACL操作失败
        """
        if self.model_loaded:
            return
        
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
            if isinstance(e, (ACLError, ModelLoadError)):
                raise e
            error_msg = f"加载模型异常：{str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(
                error_msg,
                error_code=2103,
                original_error=e,
                details={"model_path": self.model_path}
            ) from e
        
        self.model_id, self.model_desc, self.input_size, self.output_size = result

        self._validate_model_input_consistency()

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
            raise ACLError(
                error_msg,
                error_code=2106,
                details={"buffer_size": self.batch_input_size}
            )
        
        self.output_dataset = create_dataset(self.output_buffer, self.batch_output_size, "输出数据集")
        if not self.output_dataset:
            error_msg = "创建输出数据集失败"
            logger.error(error_msg)
            raise ACLError(
                error_msg,
                error_code=2107,
                details={"buffer_size": self.batch_output_size}
            )
        
        self.model_loaded = True
        logger.info(f"模型加载成功：{self.model_path}")

    def _validate_model_input_consistency(self) -> None:
        """验证配置分辨率与模型输入尺寸的一致性

        Raises:
            InputValidationError: 分辨率不匹配时抛出
        """
        if not HAS_ACL:
            return

        try:
            from utils.acl_utils import get_model_input_info

            model_batch, model_channels, model_height, model_width, model_data_type, bytes_per_element = \
                get_model_input_info(self.model_desc, 0)

            expected_input_size = self.input_width * self.input_height * model_channels * bytes_per_element

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
                        "model_width": model_width,
                        "model_height": model_height,
                        "suggestion": f"请将 resolution 设置为 '{model_width}x{model_height}'"
                    }
                )

            if self.input_width != model_width or self.input_height != model_height:
                error_msg = "配置分辨率尺寸与模型输入尺寸不匹配"
                logger.error(error_msg)

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
                f"== 模型({model_width}x{model_height})"
            )

        except InputValidationError:
            raise
        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
            raise InputValidationError(
                "无法验证模型输入一致性",
                error_code=3122,
                original_error=e,
                details={"original_error": str(e)}
            ) from e

    def preprocess(self, image_data: Union[str, np.ndarray, PILImage], backend: str = 'opencv') -> None:
        """预处理图像

        Args:
            image_data: 图像数据
            backend: 图像读取后端

        Raises:
            PreprocessError: 预处理失败
        """
        validate_image_backend(backend)
        if isinstance(image_data, str):
            validate_file_path(image_data, must_exist=True)

        if not HAS_ACL:
            raise ACLError("ACL 库不可用", error_code=2301)

        image = self.preprocessor.process_single(image_data, backend)
        self.preprocessor.copy_to_device(image, self.input_buffer, self.context)

    def preprocess_batch(self, image_data_list: List[Union[str, np.ndarray, PILImage]], backend: str = 'opencv') -> bool:
        """批量预处理图像

        Args:
            image_data_list: 图像数据列表
            backend: 图像读取后端

        Returns:
            bool: 是否成功
        """
        validate_image_backend(backend)
        validate_positive_integer(len(image_data_list), "image_data_list length", min_val=1)
        
        for i, image_data in enumerate(image_data_list):
            if isinstance(image_data, str):
                validate_file_path(image_data, must_exist=True)

        return self.preprocessor.process_batch(
            image_data_list, backend, self.input_buffer, self.context
        )
    
    def execute(self) -> None:
        """执行模型推理

        Raises:
            ACLError: ACL操作失败
            RuntimeError: 模型未加载
        """
        if not HAS_ACL:
            raise ACLError("ACL 库不可用", error_code=2401)

        if not self.model_loaded:
            raise RuntimeError("模型未加载")

        self.executor.execute()
    
    def get_result(self) -> np.ndarray:
        """获取推理结果

        Returns:
            np.ndarray: 推理结果

        Raises:
            RuntimeError: 模型未加载或输出内存未分配
        """
        if not self.model_loaded or not self.output_host:
            raise RuntimeError("模型未加载或输出内存未分配")

        return self.executor.get_result()

    def get_result_batch(self) -> Optional[List[np.ndarray]]:
        """获取批量推理结果

        Returns:
            List[np.ndarray]: 每个元素是一张图像的推理结果
        """
        if not self.model_loaded or not self.output_host:
            return None

        return self.executor.get_result_batch()
    
    def run_inference(self, image_data: Union[str, np.ndarray, PILImage], backend: str = 'opencv') -> np.ndarray:
        """执行完整推理流程

        Args:
            image_data: 图像数据
            backend: 图像读取后端

        Returns:
            np.ndarray: 推理结果

        Raises:
            PreprocessError: 预处理失败
            ACLError: 推理执行失败
            PostprocessError: 结果获取失败
        """
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
        
        if self.preprocessor:
            self.preprocessor.cleanup()
        
        if self.stream and self.context:
            try:
                acl.rt.set_context(self.context)
                acl.rt.synchronize_stream(self.stream)
            except Exception as e:
                logger.warning(f"流同步失败：{e}")
        
        if self.input_dataset:
            destroy_dataset(self.input_dataset, self.context)
            self.input_dataset = None
        
        if self.output_dataset:
            destroy_dataset(self.output_dataset, self.context)
            self.output_dataset = None
        
        if self.output_host:
            free_host(self.output_host)
            self.output_host = None
        
        if self.input_buffer:
            from utils.acl_utils import free_device
            free_device(self.input_buffer)
            self.input_buffer = None
        
        if self.output_buffer:
            from utils.acl_utils import free_device
            free_device(self.output_buffer)
            self.output_buffer = None
        
        if self.model_id:
            unload_model(self.model_id, self.model_desc)
            self.model_id = None
            self.model_desc = None
        
        if self.initialized:
            destroy_acl(self.context, self.stream, self.device_id)
            self.context = None
            self.stream = None
        
        self.initialized = False
        self.model_loaded = False
        logger.debug("资源销毁完成")
    
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
            try:
                self.destroy()
            except Exception as e:
                logger.error(f"自动释放资源失败: {e}")
