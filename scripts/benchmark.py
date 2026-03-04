#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能评测脚本

用于测试不同推理模式的性能
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import InferenceAPI
from utils.profiler import Profiler


def benchmark_single_image(inference_type, image_path, model_path, device_id, resolution):
    """测试单张图片推理性能
    
    Args:
        inference_type: 推理类型
        image_path: 图片路径
        model_path: 模型路径
        device_id: 设备ID
        resolution: 分辨率
    """
    print(f"\n===== 测试单张图片推理 ({inference_type}) =====")
    print(f"图片路径: {image_path}")
    print(f"模型路径: {model_path}")
    print(f"设备ID: {device_id}")
    print(f"分辨率: {resolution}")
    
    # 运行多次以获得更准确的结果
    profiler = Profiler(enable=True)
    for i in range(5):
        print(f"\n运行第 {i+1} 次...")
        profiler.start()
        try:
            result = InferenceAPI.inference_image(
                inference_type=inference_type,
                image_path=image_path,
                model_path=model_path,
                device_id=device_id,
                resolution=resolution
            )
            print(f"检测到 {len(result) if result else 0} 个目标")
        except Exception as e:
            print(f"推理失败: {e}")
        profiler.stop()
    
    profiler.print_stats(f"单张图片推理 ({inference_type}) 平均性能")


def benchmark_batch_images(inference_type, image_paths, model_path, device_id, resolution):
    """测试批量图片推理性能
    
    Args:
        inference_type: 推理类型
        image_paths: 图片路径列表
        model_path: 模型路径
        device_id: 设备ID
        resolution: 分辨率
    """
    print(f"\n===== 测试批量图片推理 ({inference_type}) =====")
    print(f"图片数量: {len(image_paths)}")
    print(f"模型路径: {model_path}")
    print(f"设备ID: {device_id}")
    print(f"分辨率: {resolution}")
    
    # 运行多次以获得更准确的结果
    profiler = Profiler(enable=True)
    for i in range(3):
        print(f"\n运行第 {i+1} 次...")
        profiler.start()
        try:
            results = InferenceAPI.inference_batch(
                inference_type=inference_type,
                image_paths=image_paths,
                model_path=model_path,
                device_id=device_id,
                resolution=resolution
            )
            total_detections = sum(len(result) if result else 0 for result in results)
            print(f"平均每张图片检测到 {total_detections / len(results):.1f} 个目标")
        except Exception as e:
            print(f"推理失败: {e}")
        profiler.stop()
    
    profiler.print_stats(f"批量图片推理 ({inference_type}) 平均性能")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='性能评测脚本')
    parser.add_argument('--image', required=True, help='测试图片路径')
    parser.add_argument('--model', default='models/yolov8s.om', help='OM模型文件路径')
    parser.add_argument('--device', type=int, default=0, help='设备ID')
    parser.add_argument('--resolution', default='640x640', help='输入分辨率')
    parser.add_argument('--batch', action='store_true', help='测试批量推理')
    parser.add_argument('--batch-size', type=int, default=5, help='批量大小')
    parser.add_argument('--modes', nargs='+', default=['base', 'fast', 'multithread'],
                        choices=['base', 'fast', 'multithread', 'high_res'],
                        help='测试的推理模式')
    
    args = parser.parse_args()
    
    # 检查图片文件是否存在
    if not os.path.exists(args.image):
        print(f"图片文件不存在: {args.image}")
        return
    
    # 为批量测试准备多张图片
    image_paths = [args.image] * args.batch_size if args.batch else [args.image]
    
    # 测试不同的推理模式
    for mode in args.modes:
        if args.batch:
            benchmark_batch_images(mode, image_paths, args.model, args.device, args.resolution)
        else:
            benchmark_single_image(mode, args.image, args.model, args.device, args.resolution)


if __name__ == "__main__":
    main()
