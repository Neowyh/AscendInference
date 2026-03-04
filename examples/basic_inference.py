#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础推理示例

使用基础YOLO推理模式进行单张图片的目标检测
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import InferenceAPI

# 配置参数
config = {
    'model_path': 'models/yolov8s.om',  # 模型路径
    'device_id': 0,  # 设备ID
    'resolution': '640x640'  # 输入分辨率
}

# 测试图片路径
image_path = 'test.jpg'

print("===== 基础推理示例 =====")
print(f"模型路径: {config['model_path']}")
print(f"设备ID: {config['device_id']}")
print(f"输入分辨率: {config['resolution']}")
print(f"测试图片: {image_path}")
print()

# 使用统一API进行推理
try:
    results = InferenceAPI.inference_image(
        inference_type='base',
        image_path=image_path,
        **config
    )
    
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
