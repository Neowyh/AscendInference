#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用AscendCL Python接口执行YOLO模型推理示例
参考文档: https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/aclcppdevg/aclcppdevg_0000.html

本脚本实现了完整的YOLO模型推理流程，包括：
1. ACL初始化与资源管理
2. OM模型加载
3. 图像预处理
4. 模型推理
5. 结果后处理
6. 资源释放

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


class AscendInference(BaseInference):
    """AscendCL推理类，封装了完整的推理流程"""
    
    def __init__(self, model_path=None, device_id=None, resolution=None):
        """初始化推理实例
        
        Args:
            model_path: 模型路径
            device_id: 设备ID
            resolution: 输入分辨率
        """
        super().__init__(model_path, device_id, resolution)
    
    def init_acl(self):
        """初始化ACL环境
        
        Returns:
            bool: 初始化是否成功
        """
        print("开始初始化ACL...")
        
        if not super().init_acl():
            print("初始化ACL失败")
            return False
        
        print("ACL初始化成功")
        return True
    
    def load_model(self):
        """加载OM模型
        
        Returns:
            bool: 加载是否成功
        """
        print(f"加载模型: {self.model_path}")
        
        if not super().load_model():
            print("加载模型失败")
            return False
        
        input_size = self.model_manager.get_input_size()
        output_size = self.model_manager.get_output_size()
        print(f"输入大小: {input_size} bytes")
        print(f"输出大小: {output_size} bytes")
        
        print("模型加载成功")
        return True
    
    def preprocess(self, image_path, backend='pil'):
        """预处理图像
        
        Args:
            image_path: 图像文件路径
            backend: 图像读取后端 ('pil' 或 'opencv')
            
        Returns:
            bool: 预处理是否成功
        """
        print(f"预处理图像: {image_path} (backend: {backend})")
        
        if not super().preprocess(image_path, backend):
            print(f"图像预处理失败: {image_path}")
            return False
        
        print("图像预处理完成")
        return True
    
    def inference(self):
        """执行模型推理
        
        Returns:
            bool: 推理是否成功
        """
        print("开始推理...")
        
        if not super().inference():
            print("执行推理失败")
            return False
        
        print("推理完成")
        return True
    
    def postprocess(self):
        """后处理推理结果
        
        Returns:
            np.ndarray: 推理结果
        """
        print("开始后处理...")
        
        output = super().postprocess()
        if output is None:
            print("后处理失败")
            return None
        
        print(f"输出形状: {output.shape}")
        print("后处理完成")
        return output
    
    def destroy(self):
        """销毁资源"""
        print("释放资源...")
        super().destroy()
        print("资源释放完成")


def main():
    """主函数
    
    流程：
    1. 解析命令行参数
    2. 创建推理实例
    3. 初始化ACL
    4. 加载模型
    5. 预处理图像
    6. 执行推理
    7. 后处理结果
    8. 释放资源
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型推理')
    parser.add_argument('image_path', help='图像文件路径')
    parser.add_argument('--model', default=Config.DEFAULT_MODEL_PATH, help='OM模型文件路径')
    parser.add_argument('--resolution', default=Config.DEFAULT_RESOLUTION, choices=Config.SUPPORTED_RESOLUTIONS.keys(),
                        help='输入分辨率')
    parser.add_argument('--device', type=int, default=Config.DEFAULT_DEVICE_ID, help='设备ID')
    parser.add_argument('--backend', default=Config.DEFAULT_BACKEND, choices=Config.SUPPORTED_BACKENDS,
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    print(f"使用模型: {args.model}")
    print(f"输入分辨率: {Config.get_resolution(args.resolution)}")
    print(f"设备ID: {args.device}")
    
    # 创建推理实例
    inference = AscendInference(args.model, args.device, args.resolution)
    
    try:
        # 初始化ACL
        if not inference.init_acl():
            sys.exit(1)
        
        # 加载模型
        if not inference.load_model():
            sys.exit(1)
        
        # 预处理
        if not inference.preprocess(args.image_path, args.backend):
            sys.exit(1)
        
        # 执行推理
        if not inference.inference():
            sys.exit(1)
        
        # 后处理结果
        output = inference.postprocess()
        if output is not None:
            print(f"推理结果: {output[:10]}...")  # 只打印前10个值
    
    finally:
        # 释放资源
        inference.destroy()


if __name__ == "__main__":
    main()
