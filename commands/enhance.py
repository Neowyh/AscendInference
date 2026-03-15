#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像增强命令实现
"""
import os
from typing import Optional
from config import SUPPORTED_RESOLUTIONS


def cmd_enhance(args):
    """图像增强命令"""
    if not os.path.exists(args.image_path):
        print(f"图像文件不存在：{args.image_path}")
        return 1

    output_dir = args.output or 'enhanced-images'
    os.makedirs(output_dir, exist_ok=True)

    resolutions = args.resolutions or list(SUPPORTED_RESOLUTIONS.keys())
    count = args.count or 1

    try:
        import cv2
        HAS_OPENCV = True
    except ImportError:
        HAS_OPENCV = False

    backend = args.backend or 'pil'

    if backend == 'opencv' and HAS_OPENCV:
        image = cv2.imread(args.image_path)
        if image is None:
            print(f"无法读取图像：{args.image_path}")
            return 1
    else:
        from PIL import Image
        image = Image.open(args.image_path)

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    ext = os.path.splitext(args.image_path)[1] or '.jpg'

    print(f"图像增强配置:")
    print(f"  输入图像：{args.image_path}")
    print(f"  输出目录：{output_dir}")
    print(f"  扩增数量：{count}")
    print(f"  分辨率：{', '.join(resolutions)}")
    print(f"  后端：{backend}")
    print(f"  插值方法：{args.interpolation}")
    print()

    generated = 0

    for i in range(count):
        for res_name in resolutions:
            if res_name not in SUPPORTED_RESOLUTIONS:
                print(f"  [SKIP] 不支持的分辨率：{res_name}")
                continue

            width, height = SUPPORTED_RESOLUTIONS[res_name]

            if count > 1:
                output_path = os.path.join(output_dir, f"{base_name}_{i+1}_{res_name}{ext}")
            else:
                output_path = os.path.join(output_dir, f"{base_name}_{res_name}{ext}")

            if backend == 'opencv' and HAS_OPENCV:
                inter_map = {
                    'nearest': cv2.INTER_NEAREST,
                    'bilinear': cv2.INTER_LINEAR,
                    'bicubic': cv2.INTER_CUBIC
                }
                inter = inter_map.get(args.interpolation, cv2.INTER_LINEAR)
                resized = cv2.resize(image, (width, height), interpolation=inter)
                cv2.imwrite(output_path, resized)
            else:
                from PIL import Image
                inter_map = {
                    'nearest': Image.NEAREST,
                    'bilinear': Image.BILINEAR,
                    'bicubic': Image.BICUBIC
                }
                inter = inter_map.get(args.interpolation, Image.BILINEAR)
                resized = image.resize((width, height), inter)
                resized.save(output_path)

            print(f"  [OK] 生成：{output_path}")
            generated += 1

    print(f"\n图像增强完成！")
    print(f"  生成图像数：{generated}")
    print(f"  输出目录：{output_dir}")

    return 0
