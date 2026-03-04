#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理基类

提取所有推理类的公共逻辑，包括：
- ACL初始化和资源管理
- 模型加载和管理
- 内存分配和管理
- 图像预处理
- 推理执行
- 结果后处理
- 资源释放
"""

import os
import sys
import numpy as np
import logging
from PIL import Image
import acl

# 导入配置和工具类
from config import Config
from utils.acl_utils import AclManager, ModelManager, MemoryManager

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseInference:
    """推理基类"""
    
    def __init__(self, model_path=None, device_id=None, resolution=None):
        """初始化推理实例
        
        Args:
            model_path: 模型路径
            device_id: 设备ID
            resolution: 输入分辨率
        """
        # 配置参数
        config = Config.get_instance()
        self.model_path = model_path or config.model_path
        self.device_id = device_id or config.device_id
        self.resolution = resolution or config.resolution
        self.input_width, self.input_height = Config.get_resolution(self.resolution)
        
        # 设备相关
        self.acl_manager = AclManager(self.device_id)
        
        # 模型相关
        self.model_manager = None
        
        # 内存相关
        self.memory_manager = MemoryManager()
        self.input_buffer = None
        self.output_buffer = None
        
        # 初始化状态
        self.initialized = False
        self.model_loaded = False
    
    def init_acl(self):
        """初始化ACL环境
        
        Returns:
            bool: 初始化是否成功
        """
        if self.initialized:
            logger.info("ACL已经初始化")
            return True
        
        # 初始化ACL
        if not self.acl_manager.init():
            logger.error("ACL初始化失败")
            return False
        
        self.initialized = True
        logger.info("ACL初始化成功")
        return True
    
    def load_model(self):
        """加载OM模型
        
        Returns:
            bool: 加载是否成功
        """
        if self.model_loaded:
            logger.info("模型已经加载")
            return True
        
        # 检查模型文件
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在: {self.model_path}")
            return False
        
        # 加载模型
        try:
            self.model_manager = ModelManager(self.model_path)
            if not self.model_manager.load():
                logger.error("模型加载失败")
                return False
        except Exception as e:
            logger.error(f"模型加载异常: {str(e)}")
            return False
        
        # 分配输入输出内存
        try:
            input_size = self.model_manager.get_input_size()
            output_size = self.model_manager.get_output_size()
            
            self.input_buffer = self.memory_manager.malloc_device(input_size)
            if not self.input_buffer:
                logger.error("分配输入设备内存失败")
                return False
            
            self.output_buffer = self.memory_manager.malloc_device(output_size)
            if not self.output_buffer:
                logger.error("分配输出设备内存失败")
                return False
        except Exception as e:
            logger.error(f"内存分配异常: {str(e)}")
            return False
        
        self.model_loaded = True
        logger.info(f"模型加载成功: {self.model_path}")
        return True
    
    def preprocess(self, image_path, backend='pil'):
        """预处理图像
        
        Args:
            image_path: 图像文件路径
            backend: 图像读取后端 ('pil' 或 'opencv')
            
        Returns:
            bool: 预处理是否成功
        """
        # 检查图像文件
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return False
        
        # 读取图像
        try:
            if backend == 'opencv' and HAS_OPENCV:
                # 使用OpenCV读取图像
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"无法读取图像: {image_path}")
                    return False
                # OpenCV默认读取为BGR格式，转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # 使用PIL读取图像
                image = Image.open(image_path)
                image = np.array(image)
        except Exception as e:
            logger.error(f"读取图像异常: {str(e)}")
            return False
        
        # 调整大小
        try:
            if backend == 'opencv' and HAS_OPENCV:
                image = cv2.resize(image, (self.input_width, self.input_height))
            else:
                image = Image.fromarray(image).resize((self.input_width, self.input_height))
                image = np.array(image)
        except Exception as e:
            logger.error(f"调整图像大小异常: {str(e)}")
            return False
        
        # 处理灰度图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)  # 增加通道维度
        
        # 归一化
        image = image.astype(np.float32)
        image = image / 255.0  # 归一化到 [0, 1]
        
        # 通道转换 (HWC -> CHW)
        image = np.transpose(image, (2, 0, 1))
        
        # 转换为一维数组
        image = image.flatten()
        
        # 分配主机内存
        try:
            input_size = self.model_manager.get_input_size()
            input_host, ret = acl.rt.malloc_host(input_size)
            if ret != 0:
                logger.error(f"分配主机内存失败，错误码: {ret}")
                return False
            
            # 复制数据到主机内存
            acl.util.vector_to_ptr(image.tobytes(), input_host, input_size)
            
            # 复制到设备内存
            ret = acl.rt.memcpy(self.input_buffer, input_size, input_host, input_size, acl.rt.MEMCPY_HOST_TO_DEVICE)
            
            # 释放主机内存
            acl.rt.free_host(input_host)
            
            if ret != 0:
                logger.error(f"内存拷贝失败，错误码: {ret}")
                return False
        except Exception as e:
            logger.error(f"预处理异常: {str(e)}")
            return False
        
        return True
    
    def inference(self):
        """执行模型推理
        
        Returns:
            bool: 推理是否成功
        """
        if not self.model_loaded:
            logger.error("模型未加载，无法执行推理")
            return False
        
        try:
            # 准备输入输出缓冲区
            input_data = np.array([self.input_buffer], dtype=np.uintptr)
            output_data = np.array([self.output_buffer], dtype=np.uintptr)
            
            # 执行推理
            model_id = self.model_manager.get_model_id()
            ret = acl.mdl.execute(model_id, input_data, output_data)
            
            if ret != 0:
                logger.error(f"推理执行失败，错误码: {ret}")
                return False
        except Exception as e:
            logger.error(f"推理执行异常: {str(e)}")
            return False
        
        logger.info("推理执行成功")
        return True
    
    def postprocess(self):
        """后处理推理结果
        
        Returns:
            np.ndarray: 推理结果
        """
        if not self.model_loaded:
            logger.error("模型未加载，无法后处理结果")
            return None
        
        try:
            # 分配主机内存
            output_size = self.model_manager.get_output_size()
            output_host = self.memory_manager.malloc_host(output_size)
            if not output_host:
                logger.error("分配输出主机内存失败")
                return None
            
            # 复制数据到主机内存
            ret = acl.rt.memcpy(output_host, output_size, self.output_buffer, output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                logger.error(f"内存拷贝失败，错误码: {ret}")
                self.memory_manager.free(output_host)
                return None
            
            # 解析输出数据
            output = np.frombuffer(output_host, dtype=np.float32)
            
            # 释放内存
            self.memory_manager.free(output_host)
            
            logger.info("后处理成功")
            return output
        except Exception as e:
            logger.error(f"后处理异常: {str(e)}")
            return None
    
    def destroy(self):
        """销毁资源"""
        try:
            # 释放内存
            self.memory_manager.free_all()
            logger.info("内存资源释放成功")
            
            # 卸载模型
            if self.model_manager:
                self.model_manager.unload()
                logger.info("模型卸载成功")
            
            # 销毁ACL资源
            if self.initialized:
                self.acl_manager.destroy()
                logger.info("ACL资源销毁成功")
            
            # 重置状态
            self.initialized = False
            self.model_loaded = False
        except Exception as e:
            logger.error(f"销毁资源异常: {str(e)}")
    
    def __enter__(self):
        """上下文管理器进入方法"""
        # 初始化ACL
        if not self.init_acl():
            raise Exception("Failed to initialize ACL")
        
        # 加载模型
        if not self.load_model():
            self.destroy()
            raise Exception("Failed to load model")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出方法"""
        self.destroy()
