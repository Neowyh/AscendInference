#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理器模块

提供统一的图像预处理功能，支持：
- 图像加载（多种格式）
- 图像缩放
- 归一化处理
- 批量预处理
"""

import os
import ctypes
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Optional, Union, List, Tuple

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import acl
    from utils.acl_utils import malloc_host, free_host, MEMCPY_HOST_TO_DEVICE
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

from utils.logger import LoggerConfig, get_logger
from utils.exceptions import PreprocessError, MemoryError, InferenceError
from utils.validators import validate_image_backend, validate_file_path
from utils.memory_pool import MemoryPool

logger = LoggerConfig.setup_logger('ascend_inference.preprocessor', format_type='text')


class Preprocessor:
    """图像预处理器"""
    
    def __init__(
        self,
        input_width: int,
        input_height: int,
        input_size: int,
        batch_size: int = 1
    ):
        """初始化预处理器
        
        Args:
            input_width: 输入宽度
            input_height: 输入高度
            input_size: 单张输入大小（字节）
            batch_size: 批处理大小
        """
        self.input_width = input_width
        self.input_height = input_height
        self.input_size = input_size
        self.batch_size = batch_size
        self.batch_input_size = input_size * batch_size
        self.input_host_pool: Optional[MemoryPool] = None
    
    def init_pool(self, max_buffers: int = 5) -> None:
        """初始化内存池
        
        Args:
            max_buffers: 最大缓冲区数量
        """
        self.input_host_pool = MemoryPool(self.batch_input_size, device='host', max_buffers=max_buffers)
    
    def load_image(
        self,
        image_data: Union[str, np.ndarray, PILImage],
        backend: str = 'opencv'
    ) -> Union[PILImage, np.ndarray]:
        """加载图像
        
        Args:
            image_data: 图像路径或 numpy 数组或 PIL 图像
            backend: 图像读取后端，默认优先使用opencv
            
        Returns:
            PIL.Image.Image 或 numpy.ndarray: 图像数据
            
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
    
    def resize_image(
        self,
        image: Union[PILImage, np.ndarray],
        backend: str = 'opencv'
    ) -> np.ndarray:
        """调整图像大小并转换为 numpy 数组
        
        Args:
            image: PIL Image 或 numpy 数组
            backend: 图像处理后端
            
        Returns:
            numpy.ndarray: resize 后的 RGB 图像数组
            
        Raises:
            PreprocessError: 图像缩放失败时抛出
        """
        try:
            if backend == 'opencv' and HAS_OPENCV:
                return cv2.resize(image, (self.input_width, self.input_height))
            else:
                from PIL import Image as PILImageModule
                if isinstance(image, PILImageModule.Image):
                    resized_image = image.resize((self.input_width, self.input_height))
                    return np.array(resized_image)
                else:
                    return np.array(PILImageModule.fromarray(image).resize((self.input_width, self.input_height)))
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
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化图像并转换为模型输入格式
        
        Args:
            image: RGB 图像数组
            
        Returns:
            np.ndarray: 归一化后的一维数组
        """
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        image = image.astype(np.float32) / 255.0
        return np.transpose(image, (2, 0, 1)).flatten()
    
    def process_single(
        self,
        image_data: Union[str, np.ndarray, PILImage],
        backend: str = 'opencv'
    ) -> np.ndarray:
        """处理单张图像（加载、缩放、归一化）
        
        Args:
            image_data: 图像数据
            backend: 图像处理后端
            
        Returns:
            np.ndarray: 处理后的图像数据
        """
        image = self.load_image(image_data, backend)
        image = self.resize_image(image, backend)
        return self.normalize(image)
    
    def copy_to_device(
        self,
        image: np.ndarray,
        input_buffer: int,
        acl_context
    ) -> None:
        """将图像数据拷贝到设备内存
        
        Args:
            image: 处理后的图像数据
            input_buffer: 设备输入缓冲区指针
            acl_context: ACL上下文
            
        Raises:
            MemoryError: 内存分配失败
            PreprocessError: 拷贝失败
        """
        if not HAS_ACL:
            raise PreprocessError("ACL 库不可用", error_code=2301)
        
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

            safe_copy_size = min(input_buffer_size, actual_data_size)
            ctypes.memmove(input_host, image.ctypes.data, safe_copy_size)

            ret = acl.rt.memcpy(input_buffer, self.batch_input_size, input_host, self.input_size, MEMCPY_HOST_TO_DEVICE)
            self.input_host_pool.free(input_host)

            if ret != 0:
                from utils.acl_utils import get_last_error_msg
                err_msg = get_last_error_msg()
                error_msg = f"内存拷贝失败，错误码：{ret}，错误信息：{err_msg}"
                logger.error(error_msg)
                raise PreprocessError(
                    error_msg,
                    error_code=2303,
                    details={"error_msg": err_msg}
                )
        except Exception as e:
            if isinstance(e, InferenceError):
                raise e
            error_msg = f"预处理异常：{str(e)}"
            logger.error(error_msg)
            raise PreprocessError(
                error_msg,
                error_code=2304,
                original_error=e,
                details={"image_data": str(image)}
            ) from e

        logger.debug("预处理完成")
    
    def process_batch(
        self,
        image_data_list: List[Union[str, np.ndarray, PILImage]],
        backend: str,
        input_buffer: int,
        acl_context
    ) -> bool:
        """批量预处理图像
        
        Args:
            image_data_list: 图像数据列表
            backend: 图像处理后端
            input_buffer: 设备输入缓冲区指针
            acl_context: ACL上下文
            
        Returns:
            bool: 是否成功
        """
        validate_image_backend(backend)
        
        if len(image_data_list) > self.batch_size:
            logger.error(f"输入图像数量({len(image_data_list)})超过批处理大小({self.batch_size})")
            return False

        if not HAS_ACL:
            logger.error("ACL 库不可用")
            return False

        try:
            input_host = self.input_host_pool.allocate()
            if not input_host:
                logger.error("分配主机输入内存失败")
                return False

            offset = 0
            for i, image_data in enumerate(image_data_list):
                image = self.process_single(image_data, backend)
                
                actual_data_size = image.nbytes if hasattr(image, 'nbytes') else self.input_size
                if actual_data_size > self.input_size:
                    error_msg = f"批处理图像数据大小({actual_data_size})超过预期输入大小({self.input_size})"
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

                if offset + self.input_size > self.batch_input_size:
                    error_msg = f"批处理总偏移量({offset + self.input_size})超过批量输入大小({self.batch_input_size})"
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

                safe_copy_size = min(self.input_size, actual_data_size)
                ctypes.memmove(
                    ctypes.c_void_p(int(input_host) + offset),
                    image.ctypes.data,
                    safe_copy_size
                )
                offset += self.input_size

            ret = acl.rt.memcpy(
                input_buffer, self.batch_input_size,
                input_host, self.batch_input_size,
                MEMCPY_HOST_TO_DEVICE
            )
            self.input_host_pool.free(input_host)

            if ret != 0:
                from utils.acl_utils import get_last_error_msg
                err_msg = get_last_error_msg()
                logger.error(f"批量内存拷贝失败，错误码：{ret}，错误信息：{err_msg}")
                return False

        except Exception as e:
            logger.error(f"批量预处理异常：{e}")
            return False

        logger.debug(f"批量预处理完成，共{len(image_data_list)}张图像")
        return True
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.input_host_pool:
            self.input_host_pool.cleanup()
            self.input_host_pool = None
