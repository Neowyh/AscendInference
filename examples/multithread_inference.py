#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多线程推理示例

使用多线程并行处理来提高推理性能，适用于批量处理多张图片
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import InferenceAPI
import time

# 配置参数
config = {
    'model_path': 'models/yolov8s.om',  # 模型路径
    'device_id': 0,  # 设备ID
    'resolution': '640x640',  # 输入分辨率
    'num_threads': 4  # 线程数
}

# 测试图片路径列表
image_paths = [
    'test1.jpg',
    'test2.jpg',
    'test3.jpg',
    'test4.jpg'
]

print("===== 多线程推理示例 =====")
print(f"模型路径: {config['model_path']}")
print(f"设备ID: {config['device_id']}")
print(f"输入分辨率: {config['resolution']}")
print(f"线程数: {config['num_threads']}")
print(f"测试图片数量: {len(image_paths)}")
print()

# 记录开始时间
start_time = time.time()

# 使用统一API进行批量推理
try:
    results = InferenceAPI.inference_batch(
        inference_type='multithread',
        image_paths=image_paths,
        **config
    )
    
    # 计算推理时间
    inference_time = time.time() - start_time
    
    print(f"推理完成，总耗时: {inference_time:.2f}秒")
    print(f"平均每张图片耗时: {inference_time/len(image_paths):.2f}秒")
    print()
    
    # 打印推理结果
    for i, (image_path, result) in enumerate(zip(image_paths, results)):
        print(f"图片 {i+1}: {image_path}")
        if result:
            for j, obj in enumerate(result):
                print(f"  目标 {j+1}: {obj}")
        else:
            print("  未检测到目标")
        print()
        
except Exception as e:
    print(f"推理失败: {str(e)}")

print("===== 示例完成 =====")
