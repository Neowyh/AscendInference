#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据生成工具

功能：
- 生成测试图像
- 创建不同分辨率的测试数据
- 支持批量生成
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2

def generate_test_image(output_path, width, height, color=(255, 255, 255)):
    """
    生成测试图像
    
    参数：
        output_path: 输出路径
        width: 宽度
        height: 高度
        color: 背景颜色
    """
    # 创建图像
    if HAS_OPENCV:
        # 使用OpenCV
        image = np.full((height, width, 3), color, dtype=np.uint8)
        # 添加测试内容
        cv2.putText(image, f"{width}x{height}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(output_path, image)
    else:
        # 使用PIL
        image = Image.new('RGB', (width, height), color)
        # 添加测试内容
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        draw.text((50, 50), f"{width}x{height}", fill=(0, 0, 0))
        image.save(output_path)
    
    print(f"生成测试图像: {output_path}")

def generate_batch_images(output_dir, resolutions):
    """
    批量生成测试图像
    
    参数：
        output_dir: 输出目录
        resolutions: 分辨率列表 [(width, height), ...]
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成不同分辨率的图像
    for i, (width, height) in enumerate(resolutions):
        output_path = os.path.join(output_dir, f"test_{width}x{height}.jpg")
        generate_test_image(output_path, width, height)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试数据生成工具')
    parser.add_argument('--output', default='data/test_images', help='输出目录')
    parser.add_argument('--resolutions', nargs='+', type=int, default=[640, 640, 1024, 1024, 2048, 2048],
                        help='分辨率列表，格式为 width1 height1 width2 height2 ...')
    
    args = parser.parse_args()
    
    # 解析分辨率
    if len(args.resolutions) % 2 != 0:
        print("分辨率参数必须成对出现")
        sys.exit(1)
    
    resolutions = []
    for i in range(0, len(args.resolutions), 2):
        resolutions.append((args.resolutions[i], args.resolutions[i+1]))
    
    # 生成测试图像
    generate_batch_images(args.output, resolutions)
    print(f"测试图像生成完成，保存在: {args.output}")


# 检查OpenCV
HAS_OPENCV = False
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    pass


if __name__ == "__main__":
    main()
