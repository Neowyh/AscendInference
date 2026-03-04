#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理脚本

功能：
- 批量处理目录中的所有图像
- 支持多种推理模式
- 输出推理结果到指定目录
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.yolo_inference_multithread import MultithreadInference

def batch_process(input_dir, output_dir, model_path, resolution, threads, backend):
    """批量处理图像"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 收集所有图像文件
    image_paths = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(input_dir, file))
    
    if not image_paths:
        print(f"目录中没有图像文件: {input_dir}")
        return False
    
    print(f"找到 {len(image_paths)} 张图像")
    
    # 创建多线程推理实例
    inference = MultithreadInference(model_path, threads, resolution)
    
    try:
        # 启动推理
        if not inference.start():
            print("无法启动推理")
            return False
        
        print(f"启动 {len(inference.workers)} 个推理线程")
        
        # 添加任务
        start_time = datetime.now()
        for image_path in image_paths:
            inference.add_task(image_path, backend)
        
        # 等待完成
        inference.wait_completion()
        
        # 获取结果
        results = inference.get_results()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 保存结果
        results_dict = {}
        for image_path, result in results:
            filename = os.path.basename(image_path)
            if result is not None:
                # 保存结果为JSON
                results_dict[filename] = {
                    "status": "success",
                    "shape": result.shape,
                    "sample_values": result[:10].tolist()
                }
            else:
                results_dict[filename] = {
                    "status": "failed"
                }
        
        # 保存结果文件
        results_file = os.path.join(output_dir, "inference_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量推理完成")
        print(f"处理图像: {len(results)} 张")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"平均时间: {total_time / len(results):.2f} 秒/张")
        print(f"吞吐率: {len(results) / total_time:.2f} 张/秒")
        print(f"结果保存在: {results_file}")
        
        return True
        
    finally:
        # 停止推理
        inference.stop()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量推理脚本')
    parser.add_argument('input_dir', help='输入图像目录')
    parser.add_argument('--output', default='batch_results', help='输出结果目录')
    parser.add_argument('--model', default='yolov5s.om', help='OM模型文件路径')
    parser.add_argument('--resolution', default='640x640', help='输入分辨率')
    parser.add_argument('--threads', type=int, default=4, help='线程数量')
    parser.add_argument('--backend', default='pil', choices=['pil', 'opencv'],
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 执行批量处理
    if batch_process(args.input_dir, args.output, args.model, args.resolution, args.threads, args.backend):
        print("批量推理成功完成")
    else:
        print("批量推理失败")


if __name__ == "__main__":
    main()
