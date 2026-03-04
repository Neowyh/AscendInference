#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率推理示例

使用分块处理来处理大分辨率图片，适用于需要检测大场景的应用
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
    'tile_size': 640,  # 分块大小
    'overlap': 100  # 重叠区域
}

# 高分辨率测试图片路径
image_path = 'high_res_image.jpg'

print("===== 高分辨率推理示例 =====")
print(f"模型路径: {config['model_path']}")
print(f"设备ID: {config['device_id']}")
print(f"输入分辨率: {config['resolution']}")
print(f"分块大小: {config['tile_size']}")
print(f"重叠区域: {config['overlap']}")
print(f"测试图片: {image_path}")
print()

# 记录开始时间
start_time = time.time()

# 使用统一API进行高分辨率推理
try:
    results = InferenceAPI.inference_image(
        inference_type='high_res',
        image_path=image_path,
        **config
    )
    
    # 计算推理时间
    inference_time = time.time() - start_time
    
    print(f"推理完成，总耗时: {inference_time:.2f}秒")
    print()
    
    # 打印推理结果
    print("推理结果:")
    if results:
        for i, result in enumerate(results):
            print(f"目标 {i+1}: {result}")
    else:
        print("未检测到目标")
        
except Exception as e:
    print(f"推理失败: {str(e)}")

print()
print("===== 示例完成 =====")
