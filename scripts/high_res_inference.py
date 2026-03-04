#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率图像处理示例脚本

功能：
- 处理高分辨率图像（如4k、6k等）
- 自动分块并并行处理
- 支持不同的子块大小和重叠比例
- 生成处理报告
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from yolo_inference_high_res import HighResInference

def process_high_res_image(image_path, output_dir, model_path, tile_size, overlap, threads, backend):
    """处理高分辨率图像"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建高分辨率推理实例
    inference = HighResInference(model_path, threads, tile_size, overlap)
    
    try:
        # 启动推理
        if not inference.start():
            print("无法启动推理")
            return False
        
        print(f"启动 {len(inference.workers)} 个推理线程")
        
        # 处理图像
        start_time = datetime.now()
        result = inference.process_image(image_path, backend)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        
        if result:
            # 保存结果
            result_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n高分辨率图像处理完成")
            print(f"总处理时间: {total_time:.2f} 秒")
            print(f"结果保存在: {result_file}")
            
            # 生成处理报告
            report = {
                "image": image_path,
                "image_shape": result["image_shape"],
                "tile_size": tile_size,
                "overlap": overlap,
                "threads": threads,
                "num_tiles": result["num_tiles"],
                "success_tiles": len(result["sub_results"]),
                "total_time": total_time,
                "timestamp": end_time.isoformat()
            }
            
            report_file = os.path.join(output_dir, "high_res_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"报告保存在: {report_file}")
            
        return True
        
    finally:
        # 停止推理
        inference.stop()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高分辨率图像处理示例')
    parser.add_argument('image_path', help='高分辨率图像路径')
    parser.add_argument('--output', default='high_res_results', help='输出结果目录')
    parser.add_argument('--model', default='yolov5s.om', help='OM模型文件路径')
    parser.add_argument('--tile-size', type=int, nargs=2, default=(640, 640),
                        help='子块大小 (高度 宽度)')
    parser.add_argument('--overlap', type=float, default=0.2, help='重叠比例')
    parser.add_argument('--threads', type=int, default=4, help='线程数量')
    parser.add_argument('--backend', default='pil', choices=['pil', 'opencv'],
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    # 检查输入图像
    if not os.path.exists(args.image_path):
        print(f"图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    # 执行高分辨率图像处理
    if process_high_res_image(args.image_path, args.output, args.model, 
                             tuple(args.tile_size), args.overlap, args.threads, args.backend):
        print("高分辨率图像处理成功完成")
    else:
        print("高分辨率图像处理失败")


if __name__ == "__main__":
    main()
