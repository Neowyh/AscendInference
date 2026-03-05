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


def cmd_single(args):
    """推理单张图片"""
    config = load_config(args)
    
    if not os.path.exists(args.image):
        print(f"图像文件不存在：{args.image}")
        return 1
    
    print(f"\n推理配置:")
    print(f"  模式：{args.mode}")
    print(f"  图像：{args.image}")
    print(f"  模型：{config.model_path}")
    print(f"  设备：{config.device_id}")
    print(f"  分辨率：{config.resolution}")
    print(f"  后端：{config.backend}")
    
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
    
    try:
        if args.mode == 'high_res':
            # 高分辨率模式
            start = time.time()
            result = inference.process_image(args.image, config.backend)
            elapsed = time.time() - start
            print(f"\n时间统计:")
            print(f"  总时间：{elapsed:.4f} 秒")
            
        elif args.mode == 'multithread':
            # 多线程模式
            inference.add_task(args.image, config.backend)
            inference.wait_completion()
            results = inference.get_results()
            result = results[0][1] if results else None
            print(f"\n时间统计:")
            print(f"  多线程处理完成")
            
        else:
            # Base 模式：统计各阶段时间
            preprocess_start = time.time()
            if not inference.preprocess(args.image, config.backend):
                print("预处理失败")
                return 1
            preprocess_time = time.time() - preprocess_start
            
            execute_start = time.time()
            if not inference.execute():
                print("推理执行失败")
                return 1
            execute_time = time.time() - execute_start
            
            get_result_start = time.time()
            result = inference.get_result()
            get_result_time = time.time() - get_result_start
            
            total_time = preprocess_time + execute_time + get_result_time
            
            print(f"\n时间统计:")
            print(f"  预处理：{preprocess_time:.4f} 秒")
            print(f"  模型推理：{execute_time:.4f} 秒")
            print(f"  后处理：{get_result_time:.4f} 秒")
            print(f"  总时间：{total_time:.4f} 秒")
    
    finally:
        # 销毁资源
        if args.mode == 'multithread':
            inference.stop()
        else:
            inference.destroy()
    
    if result is not None:
        print(f"\n推理成功！")
        print(f"结果形状：{result.shape if hasattr(result, 'shape') else 'N/A'}")
    else:
        print(f"\n推理失败")
        return 1
    
    return 0


def cmd_batch(args):
    """批量推理"""
    config = load_config(args)
    
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在：{args.input_dir}")
        return 1
    
    image_paths = []
    for file in os.listdir(args.input_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(args.input_dir, file))
    
    if not image_paths:
        print(f"目录中没有图像文件：{args.input_dir}")
        return 1
    
    print(f"\n批量推理配置:")
    print(f"  模式：{args.mode}")
    print(f"  图像数量：{len(image_paths)}")
    print(f"  模型：{config.model_path}")
    print(f"  设备：{config.device_id}")
    print(f"  分辨率：{config.resolution}")
    print(f"  线程数：{config.num_threads}")
    
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
    
    results = []
    preprocess_times = []
    execute_times = []
    get_result_times = []
    total_times = []
    
    try:
        for i, image_path in enumerate(image_paths):
            total_start = time.time()
            
            if args.mode == 'high_res':
                result = inference.process_image(image_path, config.backend)
                total_time = time.time() - total_start
                results.append(result)
                total_times.append(total_time)
                
            elif args.mode == 'multithread':
                inference.add_task(image_path, config.backend)
                inference.wait_completion()
                batch_results = inference.get_results()
                result = batch_results[0][1] if batch_results else None
                results.append(result)
                total_time = time.time() - total_start
                total_times.append(total_time)
                
            else:
                # Base 模式：统计各阶段时间
                preprocess_start = time.time()
                if not inference.preprocess(image_path, config.backend):
                    print(f"第 {i+1} 张图片预处理失败")
                    results.append(None)
                    continue
                preprocess_time = time.time() - preprocess_start
                
                execute_start = time.time()
                if not inference.execute():
                    print(f"第 {i+1} 张图片推理失败")
                    results.append(None)
                    continue
                execute_time = time.time() - execute_start
                
                get_result_start = time.time()
                result = inference.get_result()
                get_result_time = time.time() - get_result_start
                
                results.append(result)
                preprocess_times.append(preprocess_time)
                execute_times.append(execute_time)
                get_result_times.append(get_result_time)
                total_times.append(preprocess_time + execute_time + get_result_time)
        
        # 打印时间统计
        if args.mode == 'base' and preprocess_times:
            avg_preprocess = sum(preprocess_times) / len(preprocess_times)
            avg_execute = sum(execute_times) / len(execute_times)
            avg_get_result = sum(get_result_times) / len(get_result_times)
            avg_total = sum(total_times) / len(total_times)
            
            print(f"\n时间统计 (平均):")
            print(f"  预处理：{avg_preprocess:.4f} 秒")
            print(f"  模型推理：{avg_execute:.4f} 秒")
            print(f"  后处理：{avg_get_result:.4f} 秒")
            print(f"  总时间：{avg_total:.4f} 秒")
        else:
            avg_total = sum(total_times) / len(total_times) if total_times else 0
            print(f"\n时间统计:")
            print(f"  平均总时间：{avg_total:.4f} 秒")
    
    finally:
        if args.mode == 'multithread':
            inference.stop()
        else:
            inference.destroy()
    
    # 保存结果
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        results_file = os.path.join(args.output, "results.json")
        
        results_dict = {}
        for path, result in zip(image_paths, results):
            filename = os.path.basename(path)
            if result is not None:
                results_dict[filename] = {
                    "status": "success",
                    "shape": result.shape if hasattr(result, 'shape') else None,
                    "sample": result[:10].tolist() if hasattr(result, 'tolist') else None
                }
            else:
                results_dict[filename] = {"status": "failed"}
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到：{results_file}")
    
    # 批量统计
    success_count = sum(1 for r in results if r is not None)
    total_elapsed = sum(total_times) if total_times else 0
    
    print(f"\n批量推理完成！")
    print(f"成功：{success_count}/{len(image_paths)}")
    print(f"总耗时：{total_elapsed:.2f} 秒")
    print(f"平均耗时：{total_elapsed/len(image_paths):.4f} 秒/张")
    print(f"吞吐率：{len(image_paths)/total_elapsed:.2f} 张/秒")
    
    return 0


def cmd_benchmark(args):
    """性能测试"""
    config = load_config(args)
    
    if not os.path.exists(args.image):
        print(f"图像文件不存在：{args.image}")
        return 1
    
    print(f"\n性能测试配置:")
    print(f"  图像：{args.image}")
    print(f"  模型：{config.model_path}")
    print(f"  设备：{config.device_id}")
    print(f"  分辨率：{config.resolution}")
    print(f"  次数：{args.iterations}")
    
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
    
    inference_times = []
    total_times = []
    
    try:
        for i in range(args.iterations):
            # 总时间（包括预处理、推理、后处理）
            total_start = time.time()
            
            if args.mode == 'high_res':
                result = inference.process_image(args.image, config.backend)
                total_elapsed = time.time() - total_start
                inference_elapsed = total_elapsed  # high_res 模式不单独统计推理时间
                inference_times.append(inference_elapsed)
                total_times.append(total_elapsed)
                
            elif args.mode == 'multithread':
                inference.add_task(args.image, config.backend)
                inference.wait_completion()
                results = inference.get_results()
                result = results[0][1] if results else None
                total_elapsed = time.time() - total_start
                inference_elapsed = total_elapsed  # multithread 模式不单独统计推理时间
                inference_times.append(inference_elapsed)
                total_times.append(total_elapsed)
                
            else:
                # base 模式：分别统计预处理、推理执行、获取结果的时间
                preprocess_start = time.time()
                if not inference.preprocess(args.image, config.backend):
                    print(f"第 {i+1} 次预处理失败")
                    continue
                preprocess_elapsed = time.time() - preprocess_start
                
                # 只计算推理执行时间
                execute_start = time.time()
                if not inference.execute():
                    print(f"第 {i+1} 次推理失败")
                    continue
                execute_elapsed = time.time() - execute_start
                
                get_result_start = time.time()
                result = inference.get_result()
                get_result_elapsed = time.time() - get_result_start
                
                inference_times.append(execute_elapsed)
                total_times.append(preprocess_elapsed + execute_elapsed + get_result_elapsed)
            
            if i < 3 or i == args.iterations - 1:
                if args.mode in ['high_res', 'multithread']:
                    print(f"第 {i+1} 次：{inference_times[-1]:.4f} 秒")
                else:
                    print(f"第 {i+1} 次：推理={inference_times[-1]:.4f} 秒，总时间={total_times[-1]:.4f} 秒")
    finally:
        if args.mode == 'multithread':
            inference.stop()
        else:
            inference.destroy()
    
    # 计算统计信息
    if inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        min_inference = min(inference_times)
        max_inference = max(inference_times)
        
        print(f"\n性能统计:")
        if args.mode in ['high_res', 'multithread']:
            print(f"平均耗时：{avg_inference:.4f} 秒")
            print(f"最小耗时：{min_inference:.4f} 秒")
            print(f"最大耗时：{max_inference:.4f} 秒")
            print(f"吞吐率：{1.0/avg_inference:.2f} FPS")
        else:
            print(f"推理执行时间:")
            print(f"  平均：{avg_inference:.4f} 秒")
            print(f"  最小：{min_inference:.4f} 秒")
            print(f"  最大：{max_inference:.4f} 秒")
            print(f"  吞吐率：{1.0/avg_inference:.2f} FPS")
            
            if total_times:
                avg_total = sum(total_times) / len(total_times)
                print(f"\n总时间（包括预处理和后处理）:")
                print(f"  平均：{avg_total:.4f} 秒")
                print(f"  吞吐率：{1.0/avg_total:.2f} FPS")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description='昇腾推理工具', prog='ascend-inference')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    single_parser = subparsers.add_parser('single', help='推理单张图片')
    single_parser.add_argument('image', help='图像路径')
    single_parser.add_argument('--config', help='JSON 配置文件路径（优先级：命令行 > JSON > 默认值）')
    single_parser.add_argument('--model', help='模型路径')
    single_parser.add_argument('--device', type=int, help='设备 ID')
    single_parser.add_argument('--resolution', help='分辨率')
    single_parser.add_argument('--backend', choices=['pil', 'opencv'], help='图像读取后端')
    single_parser.add_argument('--mode', default='base', choices=['base', 'multithread', 'high_res'], help='推理模式')
    single_parser.set_defaults(func=cmd_single)
    
    batch_parser = subparsers.add_parser('batch', help='批量推理')
    batch_parser.add_argument('input_dir', help='输入图像目录')
    batch_parser.add_argument('--output', help='输出结果目录')
    batch_parser.add_argument('--config', help='JSON 配置文件路径')
    batch_parser.add_argument('--model', help='模型路径')
    batch_parser.add_argument('--device', type=int, help='设备 ID')
    batch_parser.add_argument('--resolution', help='分辨率')
    batch_parser.add_argument('--backend', choices=['pil', 'opencv'], help='图像读取后端')
    batch_parser.add_argument('--mode', default='multithread', choices=['base', 'multithread', 'high_res'], help='推理模式')
    batch_parser.add_argument('--threads', type=int, help='线程数')
    batch_parser.set_defaults(func=cmd_batch)
    
    bench_parser = subparsers.add_parser('benchmark', help='性能测试')
    bench_parser.add_argument('image', help='图像路径')
    bench_parser.add_argument('--config', help='JSON 配置文件路径')
    bench_parser.add_argument('--model', help='模型路径')
    bench_parser.add_argument('--device', type=int, help='设备 ID')
    bench_parser.add_argument('--resolution', help='分辨率')
    bench_parser.add_argument('--backend', choices=['pil', 'opencv'], help='图像读取后端')
    bench_parser.add_argument('--mode', default='base', choices=['base', 'multithread', 'high_res'], help='推理模式')
    bench_parser.add_argument('--iterations', type=int, default=10, help='推理次数')
    bench_parser.set_defaults(func=cmd_benchmark)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
