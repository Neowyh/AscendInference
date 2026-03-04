#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能测试脚本

功能：
- 测试不同模型在不同分辨率下的性能
- 生成性能报告
- 支持多线程测试
"""

import os
import sys
import argparse
import json
import time

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.yolo_inference_multithread import MultithreadInference

def benchmark_model(model_path, resolution, threads, backend, test_images):
    """测试单个模型的性能"""
    print(f"\n测试模型: {model_path}")
    print(f"分辨率: {resolution}")
    print(f"线程数: {threads}")
    
    # 创建推理实例
    inference = MultithreadInference(model_path, threads, resolution)
    
    try:
        # 初始化
        if not inference.init():
            print("初始化失败")
            return None
        
        # 预热
        print("预热中...")
        for i in range(3):
            if not inference.preprocess(test_images[0], backend):
                print("预热失败")
                return None
            if not inference.inference():
                print("预热失败")
                return None
            inference.get_result()
        
        # 正式测试
        print("开始测试...")
        start_time = time.time()
        
        # 创建多线程推理实例进行正式测试
        bench_inference = MultithreadInference(model_path, threads, resolution)
        if not bench_inference.start():
            print("无法启动测试")
            return None
        
        # 添加任务
        for image_path in test_images:
            bench_inference.add_task(image_path, backend)
        
        # 等待完成
        bench_inference.wait_completion()
        
        # 获取结果
        results = bench_inference.get_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 计算性能
        if len(results) > 0:
            avg_time = total_time / len(results)
            fps = len(results) / total_time
        else:
            avg_time = 0
            fps = 0
        
        # 停止推理
        bench_inference.stop()
        
        print(f"测试完成: {len(results)} 张图像")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"平均时间: {avg_time:.4f} 秒/张")
        print(f"吞吐率: {fps:.2f} 张/秒")
        
        # 返回性能数据
        return {
            "model": os.path.basename(model_path),
            "resolution": resolution,
            "threads": threads,
            "images": len(results),
            "total_time": total_time,
            "avg_time": avg_time,
            "fps": fps
        }
        
    finally:
        # 释放资源
        inference.destroy()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型性能测试脚本')
    parser.add_argument('test_dir', help='测试图像目录')
    parser.add_argument('--models', nargs='+', default=['yolov5s.om'], help='测试模型列表')
    parser.add_argument('--resolutions', nargs='+', default=['640x640', '1k'], help='测试分辨率列表')
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4], help='测试线程数列表')
    parser.add_argument('--backend', default='pil', choices=['pil', 'opencv'],
                        help='图像读取后端')
    parser.add_argument('--output', default='benchmark_results.json', help='性能报告输出文件')
    
    args = parser.parse_args()
    
    # 收集测试图像
    test_images = []
    for file in os.listdir(args.test_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_images.append(os.path.join(args.test_dir, file))
    
    if not test_images:
        print(f"测试目录中没有图像文件: {args.test_dir}")
        sys.exit(1)
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 运行所有测试
    results = []
    
    for model in args.models:
        if not os.path.exists(model):
            print(f"模型文件不存在: {model}")
            continue
        
        for resolution in args.resolutions:
            for threads in args.threads:
                result = benchmark_model(model, resolution, threads, args.backend, test_images)
                if result:
                    results.append(result)
    
    # 保存性能报告
    if results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n性能测试完成")
        print(f"测试结果保存在: {args.output}")
        
        # 打印汇总
        print("\n性能汇总:")
        print("-" * 80)
        print(f"{'模型':<15} {'分辨率':<10} {'线程':<6} {'FPS':<10} {'平均时间(ms)':<15}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['model']:<15} {result['resolution']:<10} {result['threads']:<6} {result['fps']:<10.2f} {result['avg_time']*1000:<15.2f}")
        
        print("-" * 80)
    else:
        print("没有有效的测试结果")


if __name__ == "__main__":
    main()
