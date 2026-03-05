#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强工具

功能：
1. 根据给定图像进行图像增强
2. 将分辨率扩充到推理脚本中支持的分辨率
3. 支持多种插值方法
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def enhance_image(image_path, output_dir, resolutions=None, backend='pil', interpolation='bilinear'):
    """增强图像并保存到不同分辨率
    
    参数：
        image_path (str): 输入图像路径
        output_dir (str): 输出目录
        resolutions (list): 要生成的分辨率列表
        backend (str): 处理后端 ('pil' 或 'opencv')
        interpolation (str): 插值方法
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if resolutions is None:
        resolutions = list(Config.SUPPORTED_RESOLUTIONS.keys())
    
    if backend == 'opencv' and HAS_OPENCV:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return False
    else:
        image = Image.open(image_path)
    
    for res_name in resolutions:
        if res_name not in Config.SUPPORTED_RESOLUTIONS:
            print(f"不支持的分辨率: {res_name}")
            continue
        
        width, height = Config.SUPPORTED_RESOLUTIONS[res_name]
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{res_name}.jpg")
        
        if backend == 'opencv' and HAS_OPENCV:
            if interpolation == 'nearest':
                inter = cv2.INTER_NEAREST
            elif interpolation == 'bilinear':
                inter = cv2.INTER_LINEAR
            elif interpolation == 'bicubic':
                inter = cv2.INTER_CUBIC
            else:
                inter = cv2.INTER_LINEAR
            
            resized = cv2.resize(image, (width, height), interpolation=inter)
            cv2.imwrite(output_path, resized)
        else:
            if interpolation == 'nearest':
                inter = Image.NEAREST
            elif interpolation == 'bilinear':
                inter = Image.BILINEAR
            elif interpolation == 'bicubic':
                inter = Image.BICUBIC
            else:
                inter = Image.BILINEAR
            
            resized = image.resize((width, height), inter)
            resized.save(output_path)
        
        print(f"生成图像: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='图像增强工具')
    parser.add_argument('image_path', help='输入图像路径')
    parser.add_argument('--output', default='enhanced_images', help='输出目录')
    parser.add_argument('--resolutions', nargs='+', choices=Config.SUPPORTED_RESOLUTIONS.keys(),
                        help='要生成的分辨率列表')
    parser.add_argument('--backend', default='pil', choices=['pil', 'opencv'],
                        help='处理后端')
    parser.add_argument('--interpolation', default='bilinear', choices=['nearest', 'bilinear', 'bicubic'],
                        help='插值方法')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    if enhance_image(args.image_path, args.output, args.resolutions, args.backend, args.interpolation):
        print(f"图像增强完成，结果保存在: {args.output}")
    else:
        print("图像增强失败")


if __name__ == "__main__":
    main()
