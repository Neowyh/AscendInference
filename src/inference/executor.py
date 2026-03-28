#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理执行器模块

提供模型推理执行功能，支持：
- ACL模型执行
- 流同步
- 错误处理
"""

import ctypes
import numpy as np
from typing import Optional, List

try:
    import acl
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

from utils.logger import LoggerConfig, get_logger
from utils.exceptions import ACLError, InferenceError

logger = LoggerConfig.setup_logger('ascend_inference.executor', format_type='text')


class Executor:
    """推理执行器"""
    
    def __init__(
        self,
        model_id: int,
        stream,
        input_dataset,
        output_dataset,
        output_buffer: int,
        output_size: int,
        batch_size: int = 1
    ):
        """初始化执行器
        
        Args:
            model_id: 模型ID
            stream: ACL流
            input_dataset: 输入数据集
            output_dataset: 输出数据集
            output_buffer: 输出缓冲区指针
            output_size: 单张输出大小
            batch_size: 批处理大小
        """
        self.model_id = model_id
        self.stream = stream
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.output_buffer = output_buffer
        self.output_size = output_size
        self.batch_size = batch_size
        self.batch_output_size = output_size * batch_size
        self.output_host: Optional[int] = None
    
    def init_output_buffer(self, output_host: int) -> None:
        """初始化输出主机缓冲区
        
        Args:
            output_host: 主机输出缓冲区指针
        """
        self.output_host = output_host
    
    def execute(self) -> None:
        """执行模型推理
        
        Raises:
            ACLError: ACL操作失败时抛出
            RuntimeError: 模型未加载时抛出
        """
        if not HAS_ACL:
            error_msg = "ACL 库不可用"
            logger.error(error_msg)
            raise ACLError(error_msg, error_code=2401)

        try:
            ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)

            if ret != 0:
                from utils.acl_utils import get_last_error_msg
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
                from utils.acl_utils import get_last_error_msg
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
            RuntimeError: 输出内存未分配时抛出
        """
        if not self.output_host:
            error_msg = "输出内存未分配"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        ret = acl.rt.memcpy(
            self.output_host, self.batch_output_size,
            self.output_buffer, self.output_size,
            acl.rt.MEMCPY_DEVICE_TO_HOST
        )
        if ret != 0:
            from utils.acl_utils import get_last_error_msg
            err_msg = get_last_error_msg()
            error_msg = f"获取结果失败，错误码：{ret}，错误信息：{err_msg}"
            logger.error(error_msg)
            from utils.exceptions import PostprocessError
            raise PostprocessError(
                error_msg,
                error_code=2501,
                acl_ret=ret,
                details={"error_msg": err_msg}
            )

        buffer = ctypes.cast(self.output_host, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(buffer, shape=(self.output_size // 4,))
    
    def get_result_batch(self) -> Optional[List[np.ndarray]]:
        """获取批量推理结果
        
        Returns:
            List[np.ndarray]: 每个元素是一张图像的推理结果
        """
        if not self.output_host:
            return None

        ret = acl.rt.memcpy(
            self.output_host, self.batch_output_size,
            self.output_buffer, self.batch_output_size,
            acl.rt.MEMCPY_DEVICE_TO_HOST
        )
        if ret != 0:
            from utils.acl_utils import get_last_error_msg
            err_msg = get_last_error_msg()
            logger.error(f"获取批量结果失败，错误码：{ret}，错误信息：{err_msg}")
            return None

        results = []
        single_output_size = self.output_size // 4
        buffer = ctypes.cast(self.output_host, ctypes.POINTER(ctypes.c_float))

        for i in range(self.batch_size):
            offset = i * single_output_size
            result = np.ctypeslib.as_array(
                ctypes.cast(ctypes.addressof(buffer[offset]), ctypes.POINTER(ctypes.c_float)),
                shape=(single_output_size,)
            )
            results.append(result)

        return results
