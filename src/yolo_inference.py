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
from PIL import Image
import acl

# 导入ACL工具类
from utils.acl_utils import AclManager, ModelManager, MemoryManager

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 全局配置参数
MODEL_PATH = "yolov5s.om"  # OM模型文件路径
DEVICE_ID = 0  # 设备ID，通常为0

# 支持的输入分辨率
SUPPORTED_RESOLUTIONS = {
    "640x640": (640, 640),
    "1k": (1024, 1024),
    "1k2k": (1024, 2048),
    "2k": (2048, 2048),
    "2k4k": (2048, 4096),
    "4k": (4096, 4096),
    "4k6k": (4096, 6144),
    "3k6k": (3072, 6144),
    "6k": (6144, 6144)
}

# 默认分辨率
DEFAULT_RESOLUTION = "640x640"
INPUT_WIDTH, INPUT_HEIGHT = SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION]


class AscendInference:
    """AscendCL推理类，封装了完整的推理流程"""
    
    def __init__(self):
        """初始化推理实例，设置默认值"""
        # 设备相关
        self.device_id = DEVICE_ID  # 设备ID
        self.acl_manager = AclManager(self.device_id)  # ACL管理器
        
        # 模型相关
        self.model_manager = None  # 模型管理器
        
        # 内存相关
        self.memory_manager = MemoryManager()  # 内存管理器
        self.input_buffer = None  # 输入缓冲区（设备端）
        self.output_buffer = None  # 输出缓冲区（设备端）
    
    def init_acl(self):
        """初始化ACL环境
        
        步骤：
        1. 初始化ACL
        2. 设置设备
        3. 创建上下文
        4. 创建流
        
        返回：
            bool: 初始化是否成功
        """
        print("开始初始化ACL...")
        
        if not self.acl_manager.init():
            print("初始化ACL失败")
            return False
        
        print("ACL初始化成功")
        return True
    
    def load_model(self, model_path):
        """加载OM模型
        
        步骤：
        1. 检查模型文件是否存在
        2. 从文件加载模型
        3. 创建模型描述
        4. 获取模型输入输出信息
        5. 分配输入输出内存
        
        参数：
            model_path (str): 模型文件路径
            
        返回：
            bool: 加载是否成功
        """
        print(f"加载模型: {model_path}")
        
        # 1. 检查模型文件
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        # 2. 加载模型
        self.model_manager = ModelManager(model_path)
        if not self.model_manager.load():
            print("加载模型失败")
            return False
        
        input_size = self.model_manager.get_input_size()
        output_size = self.model_manager.get_output_size()
        print(f"输入大小: {input_size} bytes")
        print(f"输出大小: {output_size} bytes")
        
        # 3. 分配输入内存
        self.input_buffer = self.memory_manager.malloc_device(input_size)
        if not self.input_buffer:
            print("申请输入内存失败")
            return False
        
        # 4. 分配输出内存
        self.output_buffer = self.memory_manager.malloc_device(output_size)
        if not self.output_buffer:
            print("申请输出内存失败")
            return False
        
        print("模型加载成功")
        return True
    
    def preprocess(self, image_path, backend='pil'):
        """预处理图像
        
        步骤：
        1. 读取图像
        2. 调整大小
        3. 转换为NCHW格式
        4. 归一化
        5. 通道转换 (HWC -> CHW)
        6. 复制数据到设备内存
        
        参数：
            image_path (str): 图像文件路径
            backend (str): 图像读取后端 ('pil' 或 'opencv')
            
        返回：
            bool: 预处理是否成功
        """
        print(f"预处理图像: {image_path} (backend: {backend})")
        
        # 1. 检查图像文件
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return None
        
        # 2. 读取图像
        if backend == 'opencv' and HAS_OPENCV:
            # 使用OpenCV读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"OpenCV无法读取图像: {image_path}")
                return None
            # OpenCV默认读取为BGR格式，转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 使用PIL读取图像
            image = Image.open(image_path)
            image = np.array(image)
        
        # 3. 调整大小
        if backend == 'opencv' and HAS_OPENCV:
            image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        else:
            image = Image.fromarray(image).resize((INPUT_WIDTH, INPUT_HEIGHT))
            image = np.array(image)
        
        # 4. 处理灰度图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)  # 增加通道维度
        
        # 5. 归一化
        image = image.astype(np.float32)
        image = image / 255.0  # 归一化到 [0, 1]
        
        # 6. 通道转换 (HWC -> CHW)
        image = np.transpose(image, (2, 0, 1))
        
        # 7. 转换为一维数组
        image = image.flatten()
        
        # 8. 分配主机内存
        input_host, ret = acl.rt.malloc_host(self.input_size)
        if ret != 0:
            print(f"分配主机内存失败: {ret}")
            return None
        
        # 9. 复制数据到主机内存
        acl.util.vector_to_ptr(image.tobytes(), input_host, self.input_size)
        
        # 10. 复制到设备内存
        ret = acl.rt.memcpy(self.input_buffer, self.input_size, input_host, self.input_size, acl.rt.MEMCPY_HOST_TO_DEVICE)
        if ret != 0:
            print(f"内存复制失败: {ret}")
            acl.rt.free_host(input_host)
            return None
        
        # 11. 释放主机内存
        acl.rt.free_host(input_host)
        
        print("图像预处理完成")
        return True
    
    def inference(self):
        """执行模型推理
        
        步骤：
        1. 准备输入输出缓冲区
        2. 调用模型执行推理
        
        返回：
            bool: 推理是否成功
        """
        print("开始推理...")
        
        # 1. 准备输入输出缓冲区
        input_data = np.array([self.input_buffer], dtype=np.uintptr)
        output_data = np.array([self.output_buffer], dtype=np.uintptr)
        
        # 2. 执行推理
        model_id = self.model_manager.get_model_id()
        ret = acl.mdl.execute(model_id, input_data, output_data)
        if ret != 0:
            print(f"执行推理失败: {ret}")
            return False
        
        print("推理完成")
        return True
    
    def postprocess(self):
        """后处理推理结果
        
        步骤：
        1. 分配主机内存
        2. 从设备内存复制数据到主机
        3. 解析输出数据
        4. 释放内存
        
        返回：
            np.ndarray: 推理结果
        """
        print("开始后处理...")
        
        # 1. 分配主机内存
        output_size = self.model_manager.get_output_size()
        output_host = self.memory_manager.malloc_host(output_size)
        if not output_host:
            print("分配主机内存失败")
            return None
        
        # 2. 复制数据到主机内存
        ret = acl.rt.memcpy(output_host, output_size, self.output_buffer, output_size, acl.rt.MEMCPY_DEVICE_TO_HOST)
        if ret != 0:
            print(f"内存复制失败: {ret}")
            self.memory_manager.free(output_host)
            return None
        
        # 3. 解析输出数据
        # 假设输出是 [batch, num_classes, grid_h, grid_w] 格式
        output = np.frombuffer(output_host, dtype=np.float32)
        
        # 4. 释放内存
        self.memory_manager.free(output_host)
        
        print(f"输出形状: {output.shape}")
        print("后处理完成")
        return output
    
    def destroy(self):
        """销毁资源
        
        步骤：
        1. 释放内存
        2. 卸载模型
        3. 销毁ACL资源
        """
        print("释放资源...")
        
        # 1. 释放内存
        self.memory_manager.free_all()
        
        # 2. 卸载模型
        if self.model_manager:
            self.model_manager.unload()
        
        # 3. 销毁ACL资源
        self.acl_manager.destroy()
        
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
    parser.add_argument('--model', default=MODEL_PATH, help='OM模型文件路径')
    parser.add_argument('--resolution', default=DEFAULT_RESOLUTION, choices=SUPPORTED_RESOLUTIONS.keys(),
                        help='输入分辨率')
    parser.add_argument('--device', type=int, default=DEVICE_ID, help='设备ID')
    parser.add_argument('--backend', default='pil', choices=['pil', 'opencv'],
                        help='图像读取后端 (pil 或 opencv)')
    
    args = parser.parse_args()
    
    # 更新全局变量
    global MODEL_PATH, DEVICE_ID, INPUT_WIDTH, INPUT_HEIGHT
    MODEL_PATH = args.model
    DEVICE_ID = args.device
    INPUT_WIDTH, INPUT_HEIGHT = SUPPORTED_RESOLUTIONS[args.resolution]
    
    print(f"使用模型: {MODEL_PATH}")
    print(f"输入分辨率: {INPUT_WIDTH}x{INPUT_HEIGHT}")
    print(f"设备ID: {DEVICE_ID}")
    
    # 创建推理实例
    inference = AscendInference()
    
    try:
        # 初始化ACL
        if not inference.init_acl():
            sys.exit(1)
        
        # 加载模型
        if not inference.load_model(MODEL_PATH):
            sys.exit(1)
        
        # 预处理
        if not inference.preprocess(image_path, args.backend):
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
