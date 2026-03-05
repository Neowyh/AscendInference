#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例

展示如何使用昇腾推理 API
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.api import InferenceAPI
from src.inference import Inference, MultithreadInference, HighResInference


def example_basic_inference():
    """基础推理示例"""
    print("=" * 50)
    print("示例 1: 基础推理")
    print("=" * 50)
    
    config = Config(
        model_path="models/yolov8s.om",
        device_id=0,
        resolution="640x640"
    )
    
    inference = Inference(config)
    
    try:
        with inference:
            result = inference.run_inference("test.jpg", backend="pil")
            if result is not None:
                print(f"推理成功！结果形状：{result.shape}")
            else:
                print("推理失败")
    except Exception as e:
        print(f"推理异常：{e}")


def example_api_usage():
    """使用统一 API 示例"""
    print("\n" + "=" * 50)
    print("示例 2: 使用统一 API")
    print("=" * 50)
    
    config = Config(
        model_path="models/yolov8s.om",
        device_id=0,
        resolution="640x640"
    )
    
    result = InferenceAPI.inference_image(
        mode="base",
        image_path="test.jpg",
        config=config
    )
    
    if result is not None:
        print(f"推理成功！结果形状：{result.shape}")


def example_multithread():
    """多线程推理示例"""
    print("\n" + "=" * 50)
    print("示例 3: 多线程推理")
    print("=" * 50)
    
    config = Config(
        model_path="models/yolov8s.om",
        device_id=0,
        resolution="640x640",
        num_threads=4
    )
    
    image_paths = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"]
    
    results = InferenceAPI.inference_batch(
        mode="multithread",
        image_paths=image_paths,
        config=config
    )
    
    print(f"处理了 {len(results)} 张图片")


def example_high_res():
    """高分辨率推理示例"""
    print("\n" + "=" * 50)
    print("示例 4: 高分辨率推理")
    print("=" * 50)
    
    config = Config(
        model_path="models/yolov8s.om",
        device_id=0,
        tile_size=640,
        overlap=100
    )
    
    result = InferenceAPI.inference_image(
        mode="high_res",
        image_path="high_res_test.jpg",
        config=config
    )
    
    if result:
        print(f"推理成功！子块数量：{result['num_tiles']}")


def example_cli():
    """使用命令行工具示例"""
    print("\n" + "=" * 50)
    print("示例 5: 使用命令行工具")
    print("=" * 50)
    print("""
命令行用法:

1. 推理单张图片:
   python main.py single test.jpg --model models/yolov8s.om --device 0 --resolution 640x640

2. 批量推理:
   python main.py batch ./images --output ./results --model models/yolov8s.om --threads 4

3. 性能测试:
   python main.py benchmark test.jpg --iterations 10 --model models/yolov8s.om

4. 查看帮助:
   python main.py --help
   python main.py single --help
""")


def main():
    """运行所有示例"""
    print("昇腾推理 - 使用示例\n")
    
    print("注意：以下示例需要实际的模型文件和图像文件才能运行\n")
    
    example_basic_inference()
    example_api_usage()
    example_multithread()
    example_high_res()
    example_cli()
    
    print("\n" + "=" * 50)
    print("示例结束")
    print("=" * 50)


if __name__ == "__main__":
    main()
