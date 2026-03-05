#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行工具

统一的 CLI 入口，支持多种推理模式
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

from config import Config
from src.api import InferenceAPI
from src.inference import Inference, MultithreadInference, HighResInference


def load_config(args):
    """加载配置（JSON 文件 + 命令行参数覆盖）"""
    # 1. 从 JSON 文件加载基础配置
    if args.config:
        config = Config.from_json(args.config)
        print(f"已加载配置文件：{args.config}")
    else:
        config = Config()
        print("使用默认配置")
    
    # 2. 应用命令行参数覆盖（优先级更高）
    overrides = {}
    if args.model:
        overrides['model_path'] = args.model
    if hasattr(args, 'device') and args.device:
        overrides['device_id'] = args.device
    if hasattr(args, 'resolution') and args.resolution:
        overrides['resolution'] = args.resolution
    if hasattr(args, 'backend') and args.backend:
        overrides['backend'] = args.backend
    if hasattr(args, 'threads') and args.threads:
        overrides['num_threads'] = args.threads
    
    if overrides:
        config.apply_overrides(**overrides)
        print(f"命令行参数覆盖：{overrides}")
    
    return config


def cmd_infer(args):
    """统一推理接口（支持单张/批量/性能测试）"""
    config = load_config(args)
    
    # 检测输入是单张图片还是目录
    if os.path.isfile(args.input):
        image_paths = [args.input]
        is_batch = False
    elif os.path.isdir(args.input):
        image_paths = []
        for file in os.listdir(args.input):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(args.input, file))
        if not image_paths:
            print(f"目录中没有图像文件：{args.input}")
            return 1
        is_batch = len(image_paths) > 1
    else:
        print(f"输入路径不存在：{args.input}")
        return 1
    
    # 检测 AI 核心数并设置线程数
    ai_cores = Config.MAX_AI_CORES
    threads_per_core = getattr(args, 'threads_per_core', 1)
    if args.mode == 'multithread':
        config.num_threads = ai_cores * threads_per_core
        print(f"\n检测到 AI 核心数：{ai_cores}")
        print(f"每个核心线程数：{threads_per_core}")
        print(f"总线程数：{config.num_threads}")
    
    iterations = getattr(args, 'iterations', 1)
    is_benchmark = iterations > 1
    
    # 显示配置信息
    if is_benchmark:
        print(f"\n性能测试配置:")
        print(f"  图像：{args.input}")
        print(f"  模型：{config.model_path}")
        print(f"  设备：{config.device_id}")
        print(f"  分辨率：{config.resolution}")
        print(f"  次数：{iterations}")
    elif is_batch:
        print(f"\n批量推理配置:")
        print(f"  图像数量：{len(image_paths)}")
        print(f"  模型：{config.model_path}")
        print(f"  设备：{config.device_id}")
        print(f"  分辨率：{config.resolution}")
        print(f"  模式：{args.mode}")
    else:
        print(f"\n推理配置:")
        print(f"  图像：{args.input}")
        print(f"  模型：{config.model_path}")
        print(f"  设备：{config.device_id}")
        print(f"  分辨率：{config.resolution}")
        print(f"  后端：{config.backend}")
        print(f"  模式：{args.mode}")
    
    # 根据模式创建推理实例
    if args.mode == 'high_res':
        inference = HighResInference(config)
    elif args.mode == 'multithread':
        inference = MultithreadInference(config)
        if not inference.start():
            print("无法启动推理")
            return 1
    else:
        inference = Inference(config)
        if not inference.init():
            print("初始化失败")
            return 1
    
    all_inference_times = []
    all_total_times = []
    all_preprocess_times = []
    all_get_result_times = []
    results = []
    
    try:
        for img_idx, image_path in enumerate(image_paths):
            if is_batch and not is_benchmark:
                print(f"\n处理第 {img_idx+1}/{len(image_paths)} 张图像：{os.path.basename(image_path)}")
            
            inference_times = []
            total_times = []
            preprocess_times = []
            get_result_times = []
            
            for i in range(iterations):
                if args.mode == 'high_res':
                    total_start = time.time()
                    result = inference.process_image(image_path, config.backend)
                    total_elapsed = time.time() - total_start
                    inference_times.append(total_elapsed)
                    total_times.append(total_elapsed)
                    
                elif args.mode == 'multithread':
                    total_start = time.time()
                    inference.add_task(image_path, config.backend)
                    inference.wait_completion()
                    batch_results = inference.get_results()
                    result = batch_results[0][1] if batch_results else None
                    total_elapsed = time.time() - total_start
                    inference_times.append(total_elapsed)
                    total_times.append(total_elapsed)
                    
                else:
                    # Base 模式：分别统计预处理、推理执行、获取结果的时间
                    preprocess_start = time.time()
                    if not inference.preprocess(image_path, config.backend):
                        print(f"第 {i+1} 次预处理失败")
                        continue
                    preprocess_time = time.time() - preprocess_start
                    
                    execute_start = time.time()
                    if not inference.execute():
                        print(f"第 {i+1} 次推理失败")
                        continue
                    execute_time = time.time() - execute_start
                    
                    get_result_start = time.time()
                    result = inference.get_result()
                    get_result_time = time.time() - get_result_start
                    
                    total_time = preprocess_time + execute_time + get_result_time
                    inference_times.append(execute_time)
                    total_times.append(total_time)
                    preprocess_times.append(preprocess_time)
                    get_result_times.append(get_result_time)
            
            # 保存结果
            if inference_times:
                results.append({
                    'image': image_path,
                    'result': result,
                    'inference_times': inference_times,
                    'total_times': total_times,
                    'preprocess_times': preprocess_times,
                    'get_result_times': get_result_times
                })
                all_inference_times.extend(inference_times)
                all_total_times.extend(total_times)
                if args.mode == 'base':
                    all_preprocess_times.extend(preprocess_times)
                    all_get_result_times.extend(get_result_times)
            
            # 显示每次的结果
            if (is_benchmark or is_batch) and inference_times:
                avg_time = sum(inference_times) / len(inference_times)
                if img_idx < 2 or img_idx == len(image_paths) - 1:
                    if is_benchmark:
                        print(f"  图像 {img_idx+1}: 平均推理={avg_time:.4f} 秒")
                    else:
                        print(f"  平均耗时：{avg_time:.4f} 秒")
        
        # 显示统计信息
        if all_inference_times:
            if is_benchmark or is_batch:
                avg_inference = sum(all_inference_times) / len(all_inference_times)
                min_inference = min(all_inference_times)
                max_inference = max(all_inference_times)
                
                print(f"\n性能统计:")
                print(f"  平均推理时间：{avg_inference:.4f} 秒")
                print(f"  最小推理时间：{min_inference:.4f} 秒")
                print(f"  最大推理时间：{max_inference:.4f} 秒")
                print(f"  吞吐率：{1.0/avg_inference:.2f} FPS")
                
                if all_total_times:
                    avg_total = sum(all_total_times) / len(all_total_times)
                    print(f"\n总时间统计:")
                    print(f"  平均总时间：{avg_total:.4f} 秒")
                    print(f"  总吞吐率：{len(image_paths)*iterations/sum(all_total_times):.2f} 张/秒")
            
            elif not is_benchmark and not is_batch and all_total_times:
                if args.mode == 'base' and all_preprocess_times:
                    avg_preprocess = sum(all_preprocess_times) / len(all_preprocess_times)
                    avg_inference = sum(all_inference_times) / len(all_inference_times)
                    avg_get_result = sum(all_get_result_times) / len(all_get_result_times)
                    avg_total = sum(all_total_times) / len(all_total_times)
                    
                    print(f"\n时间统计:")
                    print(f"  预处理：{avg_preprocess:.4f} 秒")
                    print(f"  模型推理：{avg_inference:.4f} 秒")
                    print(f"  后处理：{avg_get_result:.4f} 秒")
                    print(f"  总时间：{avg_total:.4f} 秒")
                else:
                    print(f"\n时间统计:")
                    print(f"  推理执行：{all_inference_times[0]:.4f} 秒")
                    print(f"  总时间：{all_total_times[0]:.4f} 秒")
    
    finally:
        # 销毁资源
        if args.mode == 'multithread':
            inference.stop()
        else:
            inference.destroy()
    
    # 保存结果到文件
    if args.output and results:
        os.makedirs(args.output, exist_ok=True)
        results_file = os.path.join(args.output, "results.json")
        
        results_dict = []
        for r in results:
            filename = os.path.basename(r['image'])
            result_data = r['result']
            results_dict.append({
                "image": filename,
                "status": "success" if result_data is not None else "failed",
                "shape": result_data.shape if result_data is not None and hasattr(result_data, 'shape') else None
            })
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到：{results_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='昇腾推理工具', prog='ascend-inference')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 统一的推理命令
    infer_parser = subparsers.add_parser('infer', help='推理（支持单张/批量/性能测试）')
    infer_parser.add_argument('input', help='输入图像路径或目录')
    infer_parser.add_argument('--config', help='JSON 配置文件路径（优先级：命令行 > JSON > 默认值）')
    infer_parser.add_argument('--model', help='模型路径')
    infer_parser.add_argument('--device', type=int, help='设备 ID')
    infer_parser.add_argument('--resolution', help='分辨率')
    infer_parser.add_argument('--backend', choices=['pil', 'opencv'], help='图像读取后端')
    infer_parser.add_argument('--mode', default='base', choices=['base', 'multithread', 'high_res'], help='推理模式')
    infer_parser.add_argument('--iterations', type=int, default=1, help='推理次数（>1 时进行性能测试）')
    infer_parser.add_argument('--threads-per-core', type=int, default=1, help='每个 AI 核心的线程数（仅 multithread 模式）')
    infer_parser.add_argument('--output', help='输出结果目录')
    infer_parser.set_defaults(func=cmd_infer)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
