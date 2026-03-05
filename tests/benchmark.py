#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试

测试推理性能，包括：
- 单次推理时间
- 批量推理吞吐率
- 不同分辨率性能对比
- 内存池性能对比
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import List, Dict, Optional

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.inference import Inference


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, config: Config):
        """初始化基准测试
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.results: Dict[str, List[float]] = {}
    
    def test_single_inference(
        self, 
        image_path: str, 
        iterations: int = 100,
        warmup: bool = True
    ) -> Dict[str, float]:
        """测试单次推理性能
        
        Args:
            image_path: 图像路径
            iterations: 测试迭代次数
            warmup: 是否预热
            
        Returns:
            性能统计字典
        """
        print(f"\n单次推理性能测试")
        print(f"  图像：{image_path}")
        print(f"  迭代次数：{iterations}")
        print(f"  分辨率：{self.config.resolution}")
        
        inference = Inference(self.config)
        
        try:
            # 初始化（包括预热）
            if not inference.init(warmup=warmup, warmup_iterations=3):
                print("初始化失败")
                return {}
            
            times = []
            
            # 执行测试
            for i in range(iterations):
                start = time.time()
                result = inference.run_inference(image_path, self.config.backend)
                elapsed = time.time() - start
                
                if result is not None:
                    times.append(elapsed)
                
                if (i + 1) % 10 == 0 and self.config.enable_logging:
                    print(f"  进度：{i+1}/{iterations}")
            
            # 统计结果
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                fps = 1.0 / avg_time
                
                results = {
                    'avg_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'fps': fps,
                    'total_iterations': len(times)
                }
                
                print(f"\n性能统计:")
                print(f"  平均时间：{avg_time:.4f} 秒")
                print(f"  最小时间：{min_time:.4f} 秒")
                print(f"  最大时间：{max_time:.4f} 秒")
                print(f"  吞吐率：{fps:.2f} FPS")
                
                return results
            else:
                print("没有成功的推理结果")
                return {}
        
        finally:
            inference.destroy()
    
    def test_batch_inference(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> Dict[str, float]:
        """测试批量推理性能
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批次大小
            
        Returns:
            性能统计字典
        """
        print(f"\n批量推理性能测试")
        print(f"  图像数量：{len(image_paths)}")
        print(f"  批次大小：{batch_size}")
        
        inference = Inference(self.config)
        
        try:
            if not inference.init(warmup=True):
                print("初始化失败")
                return {}
            
            total_images = 0
            total_time = 0.0
            
            # 按批次处理
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                
                start = time.time()
                for path in batch_paths:
                    result = inference.run_inference(path, self.config.backend)
                    if result is not None:
                        total_images += 1
                elapsed = time.time() - start
                total_time += elapsed
            
            # 统计结果
            if total_images > 0:
                avg_time = total_time / len(image_paths)
                fps = total_images / total_time
                
                results = {
                    'avg_time': avg_time,
                    'total_time': total_time,
                    'fps': fps,
                    'total_images': total_images
                }
                
                print(f"\n性能统计:")
                print(f"  平均时间：{avg_time:.4f} 秒/张")
                print(f"  总时间：{total_time:.2f} 秒")
                print(f"  吞吐率：{fps:.2f} FPS")
                
                return results
            else:
                return {}
        
        finally:
            inference.destroy()
    
    def compare_resolutions(
        self,
        image_path: str,
        resolutions: List[str] = None,
        iterations: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """对比不同分辨率的性能
        
        Args:
            image_path: 图像路径
            resolutions: 分辨率列表
            iterations: 每个分辨率的迭代次数
            
        Returns:
            各分辨率的性能统计
        """
        if resolutions is None:
            resolutions = ["640x640", "1k", "2k"]
        
        print(f"\n分辨率性能对比测试")
        print(f"  图像：{image_path}")
        print(f"  分辨率：{', '.join(resolutions)}")
        
        all_results = {}
        
        for resolution in resolutions:
            self.config.resolution = resolution
            
            results = self.test_single_inference(
                image_path,
                iterations=iterations,
                warmup=True
            )
            
            if results:
                all_results[resolution] = results
        
        # 打印对比结果
        print(f"\n分辨率性能对比:")
        print(f"{'分辨率':<15} {'平均时间 (s)':<15} {'FPS':<10}")
        print("-" * 40)
        for res, stats in sorted(all_results.items(), key=lambda x: x[1]['fps'], reverse=True):
            print(f"{res:<15} {stats['avg_time']:<15.4f} {stats['fps']:<10.2f}")
        
        return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='性能基准测试工具')
    parser.add_argument('--mode', default='single', 
                       choices=['single', 'batch', 'compare'],
                       help='测试模式')
    parser.add_argument('--image', default='test.jpg',
                       help='测试图像路径')
    parser.add_argument('--image-dir', help='测试图像目录（批量测试）')
    parser.add_argument('--iterations', type=int, default=100,
                       help='迭代次数')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--model', default='models/yolov8s.om',
                       help='模型路径')
    parser.add_argument('--device', type=int, default=0,
                       help='设备 ID')
    parser.add_argument('--resolution', default='640x640',
                       help='分辨率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config(
        model_path=args.model,
        device_id=args.device,
        resolution=args.resolution,
        enable_logging=True,
        log_level='info'
    )
    
    # 创建基准测试
    benchmark = PerformanceBenchmark(config)
    
    # 执行测试
    if args.mode == 'single':
        benchmark.test_single_inference(
            args.image,
            iterations=args.iterations
        )
    
    elif args.mode == 'batch':
        if not args.image_dir or not os.path.isdir(args.image_dir):
            print("错误：批量测试需要指定 --image-dir")
            sys.exit(1)
        
        image_paths = []
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(args.image_dir, file))
        
        if not image_paths:
            print("目录中没有图像文件")
            sys.exit(1)
        
        benchmark.test_batch_inference(
            image_paths,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'compare':
        benchmark.compare_resolutions(
            args.image,
            iterations=args.iterations
        )


if __name__ == "__main__":
    main()
