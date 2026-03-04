#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AscendCL YOLO模型快速推理脚本

此脚本为简易版，专注于最高推理效率：
- 移除了所有打印输出
- 简化了错误处理
- 优化了内存管理
- 只保留核心推理流程

支持的模型：
- YOLOv5 (s, n)
- YOLOv8 (s, n)
- YOLOv10 (s, n)

支持的输入分辨率：
- 640x640 (默认)
- 1024x1024 (1k)
- 1024x2048 (1k×2k)
- 2048x2048 (2k)
- 2048x4096 (2k×4k)
- 4096x4096 (4k)
- 4096x6144 (4k×6k)
- 3072x6144 (3k×6k)
- 6144x6144 (6k)
"""

import os
import sys
import numpy as np
from PIL import Image
import acl

# 导入配置和基类
from config import Config
from base_inference import BaseInference

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class FastAscendInference(BaseInference):
    """高效AscendCL推理类"""
    
    def __init__(self, model_path=None, device_id=None, resolution=None):
        """初始化推理实例
        
        Args:
            model_path: 模型路径
            device_id: 设备ID
            resolution: 输入分辨率
        """
        super().__init__(model_path, device_id, resolution)
        self.output_host = None
    
    def init(self):
        """初始化ACL和加载模型"""
        # 初始化ACL
        if not self.init_acl():
            return False
        
        # 加载模型
        if not self.load_model():
            return False
        
        # 预分配主机内存用于输出
        output_size = self.model_manager.get_output_size()
        self.output_host = self.memory_manager.malloc_host(output_size)
        if not self.output_host:
            return False
        
        return True
    
    def preprocess(self, image_data, backend='pil'):
        """预处理图像数据
        
        Args:
            image_data: 图像路径或PIL图像对象
            backend: 图像读取后端 ('pil' 或 'opencv')
            
        Returns:
            bool: 预处理是否成功
        """
        # 读取和处理图像
        if isinstance(image_data, str):
            # 从路径读取图像
            if backend == 'opencv' and HAS_OPENCV:
                # 使用OpenCV读取图像
                image = cv2.imread(image_data)
                if image is None:
                    return False
                # 转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 调整大小
                image = cv2.resize(image, (self.input_width, self.input_height))
            else:
                # 使用PIL读取图像
                try:
                    image = Image.open(image_data)
                    image = image.resize((self.input_width, self.input_height))
                    image = np.array(image)
                except Exception:
                    return False
        else:
            # 已经是PIL图像对象
            try:
                image = image_data.resize((self.input_width, self.input_height))
                image = np.array(image)
            except Exception:
                return False
        
        # 处理灰度图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # 归一化和通道转换
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).flatten()
        
        # 分配临时主机内存
        input_size = self.model_manager.get_input_size()
        input_host, ret = acl.rt.malloc_host(input_size)
        if ret != 0:
            return False
        
        # 复制数据
        acl.util.vector_to_ptr(image.tobytes(), input_host, input_size)
        
        # 复制到设备
        ret = acl.rt.memcpy(self.input_buffer, input_size, input_host, input_size, acl.rt.MEMCPY_HOST_TO_DEVICE)
        
        # 释放临时内存
        acl.rt.free_host(input_host)
        
        return ret == 0
    
    def get_result(self):
        """获取结果"""
        if not self.model_loaded or not self.output_host:
            return None
        
        output_size = self.model_manager.get_output_size()
        if acl.rt.memcpy(self.output_host, output_size, self.output_buffer, output_size, acl.rt.MEMCPY_DEVICE_TO_HOST) != 0:
            return None
        
        return np.frombuffer(self.output_host, dtype=np.float32)
    
    def destroy(self):
        """释放资源"""
        super().destroy()


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型快速推理')
    parser.add_argument('image_path', help='图像文件路径')
    parser.add_argument('--model', default=Config.DEFAULT_MODEL_PATH, help='OM模型文件路径')
    parser.add_argument('--resolution', default=Config.DEFAULT_RESOLUTION, choices=Config.SUPPORTED_RESOLUTIONS.keys(),
                        help='输入分辨率')
    parser.add_argument('--device', type=int, default=Config.DEFAULT_DEVICE_ID, help='设备ID')
    parser.add_argument('--backend', default=Config.DEFAULT_BACKEND, choices=Config.SUPPORTED_BACKENDS,
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    # 创建推理实例
    inference = FastAscendInference(args.model, args.device, args.resolution)
    
    try:
        # 初始化
        if not inference.init():
            sys.exit(1)
        
        # 预处理
        if not inference.preprocess(args.image_path, args.backend):
            sys.exit(1)
        
        # 推理
        if not inference.inference():
            sys.exit(1)
        
        # 获取结果
        output = inference.get_result()
        if output is not None:
            # 这里可以添加后处理逻辑
            pass
    
    finally:
        # 释放资源
        inference.destroy()


if __name__ == "__main__":
    main()
