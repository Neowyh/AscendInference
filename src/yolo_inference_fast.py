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

# 导入ACL工具类
from utils.acl_utils import AclManager, ModelManager, MemoryManager

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 配置参数
MODEL_PATH = "yolov5s.om"
DEVICE_ID = 0

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


class FastAscendInference:
    """高效AscendCL推理类"""
    
    def __init__(self):
        # 设备相关
        self.device_id = DEVICE_ID  # 设备ID
        self.acl_manager = AclManager(self.device_id)  # ACL管理器
        
        # 模型相关
        self.model_manager = ModelManager(MODEL_PATH)  # 模型管理器
        
        # 内存相关
        self.memory_manager = MemoryManager()  # 内存管理器
        self.input_buffer = None
        self.output_buffer = None
        self.output_host = None
    
    def init(self):
        """初始化ACL和加载模型"""
        # 初始化ACL
        if not self.acl_manager.init():
            return False
        
        # 加载模型
        if not os.path.exists(MODEL_PATH):
            return False
        
        if not self.model_manager.load():
            return False
        
        # 分配内存
        input_size = self.model_manager.get_input_size()
        output_size = self.model_manager.get_output_size()
        
        self.input_buffer = self.memory_manager.malloc_device(input_size)
        if not self.input_buffer:
            return False
        
        self.output_buffer = self.memory_manager.malloc_device(output_size)
        if not self.output_buffer:
            return False
        
        # 预分配主机内存用于输出
        self.output_host = self.memory_manager.malloc_host(output_size)
        if not self.output_host:
            return False
        
        return True
    
    def preprocess(self, image_data, backend='pil'):
        """预处理图像数据
        
        参数：
            image_data: 图像路径或PIL图像对象
            backend: 图像读取后端 ('pil' 或 'opencv')
            
        返回：
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
                image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
            else:
                # 使用PIL读取图像
                image = Image.open(image_data)
                image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))
                image = np.array(image)
        else:
            # 已经是PIL图像对象
            image = image_data.resize((INPUT_WIDTH, INPUT_HEIGHT))
            image = np.array(image)
        
        # 处理灰度图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # 归一化和通道转换
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).flatten()
        
        # 分配临时主机内存
        input_host, ret = acl.rt.malloc_host(self.input_size)
        if ret != 0:
            return False
        
        # 复制数据
        acl.util.vector_to_ptr(image.tobytes(), input_host, self.input_size)
        
        # 复制到设备
        ret = acl.rt.memcpy(self.input_buffer, self.input_size, input_host, self.input_size, acl.rt.MEMCPY_HOST_TO_DEVICE)
        
        # 释放临时内存
        acl.rt.free_host(input_host)
        
        return ret == 0
    
    def inference(self):
        """执行推理"""
        input_data = np.array([self.input_buffer], dtype=np.uintptr)
        output_data = np.array([self.output_buffer], dtype=np.uintptr)
        
        model_id = self.model_manager.get_model_id()
        return acl.mdl.execute(model_id, input_data, output_data) == 0
    
    def get_result(self):
        """获取结果"""
        output_size = self.model_manager.get_output_size()
        if acl.rt.memcpy(self.output_host, output_size, self.output_buffer, output_size, acl.rt.MEMCPY_DEVICE_TO_HOST) != 0:
            return None
        
        return np.frombuffer(self.output_host, dtype=np.float32)
    
    def destroy(self):
        """释放资源"""
        # 释放内存
        self.memory_manager.free_all()
        
        # 卸载模型
        self.model_manager.unload()
        
        # 销毁ACL资源
        self.acl_manager.destroy()


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO模型快速推理')
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
    
    image_path = args.image_path
    
    # 创建推理实例
    inference = FastAscendInference()
    
    try:
        # 初始化
        if not inference.init():
            sys.exit(1)
        
        # 预处理
        if not inference.preprocess(image_path, args.backend):
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
