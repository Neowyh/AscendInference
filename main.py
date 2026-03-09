#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
昇腾推理工具 - 统一命令行入口

所有功能通过此入口调用：
- infer: 推理（单张/批量/性能测试）
- check: 环境检查
- enhance: 图像增强
- package: 项目打包
- config: 配置管理
"""

import os
import sys
import argparse
import json
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES
from src.api import InferenceAPI
from src.inference import Inference, MultithreadInference, HighResInference
from utils.logger import LoggerConfig
from utils.profiler import profile_context

import numpy as np

__version__ = "1.0.0"

logger = LoggerConfig.setup_logger('ascend_inference.main')
perf_logger = LoggerConfig.setup_logger('ascend_inference.performance', level='info')


def load_config(args) -> Config:
    """加载配置（JSON 文件 + 命令行参数覆盖）"""
    if args.config:
        config = Config.from_json(args.config)
        logger.info(f"已加载配置文件：{args.config}")
    else:
        config = Config()
        logger.info("使用默认配置")
    
    overrides = {}
    if hasattr(args, 'model') and args.model:
        overrides['model_path'] = args.model
    if hasattr(args, 'device') and args.device is not None:
        overrides['device_id'] = args.device
    if hasattr(args, 'resolution') and args.resolution:
        overrides['resolution'] = args.resolution
    if hasattr(args, 'backend') and args.backend:
        overrides['backend'] = args.backend
    if hasattr(args, 'threads') and args.threads:
        overrides['num_threads'] = args.threads
    
    if overrides:
        config.apply_overrides(**overrides)
        logger.info(f"命令行参数覆盖：{overrides}")
    
    return config


class PerformanceTester:
    """性能测试器（整合所有性能测试功能）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results: Dict[str, Any] = {}
    
    def test_single(self, image_path: str, iterations: int = 100, warmup: bool = True) -> Dict:
        """单次推理性能测试"""
        perf_logger.info(f"单次推理性能测试")
        perf_logger.info(f"  图像：{image_path}")
        perf_logger.info(f"  迭代次数：{iterations}")
        perf_logger.info(f"  分辨率：{self.config.resolution}")
        
        inference = Inference(self.config)
        
        try:
            if not inference.init(warmup=warmup, warmup_iterations=3):
                perf_logger.error("初始化失败")
                return {}
            
            times = []
            for i in range(iterations):
                start = time.time()
                result = inference.run_inference(image_path, self.config.backend)
                elapsed = time.time() - start
                
                if result is not None:
                    times.append(elapsed)
                
                if (i + 1) % 10 == 0 and self.config.enable_logging:
                    perf_logger.info(f"  进度：{i+1}/{iterations}")
            
            if times:
                results = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'fps': 1.0 / (sum(times) / len(times)),
                    'total_iterations': len(times)
                }
                
                perf_logger.info(f"性能统计:")
                perf_logger.info(f"  平均时间：{results['avg_time']:.4f} 秒")
                perf_logger.info(f"  最小时间：{results['min_time']:.4f} 秒")
                perf_logger.info(f"  最大时间：{results['max_time']:.4f} 秒")
                perf_logger.info(f"  吞吐率：{results['fps']:.2f} FPS")
                
                return results
            else:
                perf_logger.warning("没有成功的推理结果")
                return {}
        
        finally:
            inference.destroy()
    
    def test_threads(self, image_path: str, thread_counts: List[int] = None) -> Dict:
        """多线程性能测试"""
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8]
        
        perf_logger.info("\n" + "="*60)
        perf_logger.info("多线程并发性能测试")
        perf_logger.info("="*60)
        
        results = {}
        
        for num_threads in thread_counts:
            perf_logger.info(f"\n测试线程数：{num_threads}")
            perf_logger.info("-" * 40)
            
            config = Config(
                model_path=self.config.model_path,
                device_id=self.config.device_id,
                resolution=self.config.resolution,
                num_threads=num_threads,
                backend=self.config.backend
            )
            
            mt_inference = MultithreadInference(config)
            
            if not mt_inference.start():
                perf_logger.error("启动失败")
                continue
            
            for _ in range(3):
                mt_inference.add_task(image_path, config.backend)
            mt_inference.wait_completion()
            mt_inference.get_results()
            
            iterations = 20
            times = []
            
            for i in range(iterations):
                start = time.time()
                mt_inference.add_task(image_path, config.backend)
                mt_inference.wait_completion()
                results_batch = mt_inference.get_results()
                elapsed = time.time() - start
                
                if results_batch:
                    times.append(elapsed)
                
                if (i + 1) % 5 == 0:
                    perf_logger.info(f"  进度：{i+1}/{iterations}, 耗时：{elapsed:.4f}秒")
            
            if times:
                avg_time = sum(times) / len(times)
                fps = iterations / sum(times)
                
                results[num_threads] = {
                    'avg_time': avg_time,
                    'fps': fps,
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times)
                }
                
                perf_logger.info(f"\n结果:")
                perf_logger.info(f"  平均耗时：{avg_time:.4f} 秒")
                perf_logger.info(f"  FPS: {fps:.2f}")
            
            mt_inference.stop()
        
        self._print_thread_comparison(results)
        return results
    
    def test_resolutions(self, image_path: str, resolutions: List[str] = None) -> Dict:
        """分辨率性能测试"""
        if resolutions is None:
            resolutions = ["640x640", "1k", "2k"]
        
        perf_logger.info("\n" + "="*60)
        perf_logger.info("分辨率性能测试")
        perf_logger.info("="*60)
        
        results = {}
        
        for resolution in resolutions:
            if resolution not in SUPPORTED_RESOLUTIONS:
                perf_logger.warning(f"不支持的分辨率：{resolution}")
                continue
            
            perf_logger.info(f"\n测试分辨率：{resolution}")
            perf_logger.info("-" * 40)
            
            config = Config(
                model_path=self.config.model_path,
                device_id=self.config.device_id,
                resolution=resolution,
                backend=self.config.backend
            )
            
            width, height = SUPPORTED_RESOLUTIONS[resolution]
            total_pixels = width * height
            
            inference = Inference(config)
            
            if not inference.init(warmup=True):
                perf_logger.error("初始化失败")
                continue
            
            times = []
            iterations = 20
            
            for i in range(iterations):
                start = time.time()
                result = inference.run_inference(image_path, config.backend)
                elapsed = time.time() - start
                
                if result is not None:
                    times.append(elapsed)
                
                if (i + 1) % 5 == 0:
                    perf_logger.info(f"  进度：{i+1}/{iterations}, 耗时：{elapsed:.4f}秒")
            
            inference.destroy()
            
            if times:
                avg_time = sum(times) / len(times)
                fps = iterations / sum(times)
                
                results[resolution] = {
                    'avg_time': avg_time,
                    'fps': fps,
                    'min_time': min(times),
                    'max_time': max(times),
                    'pixels': total_pixels,
                    'fps_per_million_pixels': fps / (total_pixels / 1e6)
                }
                
                perf_logger.info(f"\n结果:")
                perf_logger.info(f"  平均耗时：{avg_time:.4f} 秒")
                perf_logger.info(f"  FPS: {fps:.2f}")
        
        self._print_resolution_comparison(results)
        return results
    
    def _print_thread_comparison(self, results: Dict):
        """打印线程性能对比"""
        if not results:
            return
        
        perf_logger.info("\n" + "="*60)
        perf_logger.info("多线程性能对比")
        perf_logger.info("="*60)
        perf_logger.info(f"{'线程数':<10} {'平均时间 (s)':<15} {'FPS':<10} {'提升比':<10}")
        perf_logger.info("-" * 60)
        
        base_fps = None
        for num_threads in sorted(results.keys()):
            stats = results[num_threads]
            if base_fps is None:
                base_fps = stats['fps']
                improvement = 1.0
            else:
                improvement = stats['fps'] / base_fps if base_fps > 0 else 0
            
            perf_logger.info(f"{num_threads:<10} {stats['avg_time']:<15.4f} {stats['fps']:<10.2f} {improvement:<10.2f}x")
    
    def _print_resolution_comparison(self, results: Dict):
        """打印分辨率性能对比"""
        if not results:
            return
        
        perf_logger.info("\n" + "="*60)
        perf_logger.info("分辨率性能对比")
        perf_logger.info("="*60)
        perf_logger.info(f"{'分辨率':<12} {'像素数':<12} {'平均时间 (s)':<15} {'FPS':<10}")
        perf_logger.info("-" * 60)
        
        for resolution in sorted(results.keys(), key=lambda x: results[x]['fps'], reverse=True):
            stats = results[resolution]
            perf_logger.info(f"{resolution:<12} {stats['pixels']:<12,} {stats['avg_time']:<15.4f} {stats['fps']:<10.2f}")


def cmd_infer(args):
    """推理命令"""
    config = load_config(args)
    
    if os.path.isfile(args.input):
        image_paths = [args.input]
        is_batch = False
    elif os.path.isdir(args.input):
        image_paths = []
        for file in os.listdir(args.input):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(args.input, file))
        if not image_paths:
            logger.warning(f"目录中没有图像文件：{args.input}")
            return 1
        is_batch = len(image_paths) > 1
    else:
        logger.error(f"输入路径不存在：{args.input}")
        return 1
    
    if args.mode == 'multithread':
        ai_cores = MAX_AI_CORES
        threads_per_core = getattr(args, 'threads_per_core', 1)
        config.num_threads = ai_cores * threads_per_core
        logger.info(f"检测到 AI 核心数：{ai_cores}")
        logger.info(f"每个核心线程数：{threads_per_core}")
        logger.info(f"总线程数：{config.num_threads}")
    
    if args.benchmark or args.test_threads or args.test_resolutions:
        perf_logger.info("\n" + "="*60)
        perf_logger.info("性能测试模式")
        perf_logger.info("="*60)
        
        tester = PerformanceTester(config)
        
        if args.benchmark:
            tester.test_single(image_paths[0], iterations=args.iterations or 100)
        
        if args.test_threads:
            tester.test_threads(image_paths[0], args.thread_counts)
        
        if args.test_resolutions:
            tester.test_resolutions(image_paths[0])
        
        return 0
    
    iterations = getattr(args, 'iterations', 1)
    is_benchmark = iterations > 1
    
    if is_benchmark:
        logger.info(f"性能测试配置:")
        logger.info(f"  图像：{args.input}")
        logger.info(f"  模型：{config.model_path}")
        logger.info(f"  设备：{config.device_id}")
        logger.info(f"  分辨率：{config.resolution}")
        logger.info(f"  次数：{iterations}")
    elif is_batch:
        logger.info(f"批量推理配置:")
        logger.info(f"  图像数量：{len(image_paths)}")
        logger.info(f"  模型：{config.model_path}")
        logger.info(f"  模式：{args.mode}")
    else:
        logger.info(f"推理配置:")
        logger.info(f"  图像：{args.input}")
        logger.info(f"  模型：{config.model_path}")
        logger.info(f"  设备：{config.device_id}")
        logger.info(f"  分辨率：{config.resolution}")
        logger.info(f"  后端：{config.backend}")
        logger.info(f"  模式：{args.mode}")
    
    if args.mode == 'high_res':
        inference = HighResInference(config)
    elif args.mode == 'multithread':
        inference = MultithreadInference(config)
        if not inference.start():
            logger.error("无法启动推理")
            return 1
    else:
        inference = Inference(config)
        if not inference.init():
            logger.error("初始化失败")
            return 1
    
    all_inference_times = []
    all_total_times = []
    all_preprocess_times = []
    all_get_result_times = []
    results = []
    
    try:
        for img_idx, image_path in enumerate(image_paths):
            if is_batch and not is_benchmark:
                logger.info(f"处理第 {img_idx+1}/{len(image_paths)} 张图像：{os.path.basename(image_path)}")
            
            inference_times = []
            total_times = []
            preprocess_times = []
            get_result_times = []
            
            for i in range(iterations):
                preprocess_start = time.time()
                if not inference.preprocess(image_path, config.backend):
                    logger.warning(f"第 {i+1} 次预处理失败")
                    continue
                preprocess_time = time.time() - preprocess_start
                
                execute_start = time.time()
                if not inference.execute():
                    logger.warning(f"第 {i+1} 次推理失败")
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
            
            if (is_benchmark or is_batch) and inference_times:
                avg_time = sum(inference_times) / len(inference_times)
                if img_idx < 2 or img_idx == len(image_paths) - 1:
                    if is_benchmark:
                        logger.info(f"  图像 {img_idx+1}: 平均推理={avg_time:.4f} 秒")
                    else:
                        logger.info(f"  平均耗时：{avg_time:.4f} 秒")
        
        if all_inference_times:
            if is_benchmark or is_batch:
                avg_inference = sum(all_inference_times) / len(all_inference_times)
                min_inference = min(all_inference_times)
                max_inference = max(all_inference_times)
                
                logger.info(f"性能统计:")
                logger.info(f"  平均推理时间：{avg_inference:.4f} 秒")
                logger.info(f"  最小推理时间：{min_inference:.4f} 秒")
                logger.info(f"  最大推理时间：{max_inference:.4f} 秒")
                logger.info(f"  吞吐率：{1.0/avg_inference:.2f} FPS")
                
                if all_total_times:
                    avg_total = sum(all_total_times) / len(all_total_times)
                    logger.info(f"总时间统计:")
                    logger.info(f"  平均总时间：{avg_total:.4f} 秒")
                    logger.info(f"  总吞吐率：{len(image_paths)*iterations/sum(all_total_times):.2f} 张/秒")
            
            elif not is_benchmark and not is_batch and all_total_times:
                if all_preprocess_times:
                    avg_preprocess = sum(all_preprocess_times) / len(all_preprocess_times)
                    avg_inference = sum(all_inference_times) / len(all_inference_times)
                    avg_get_result = sum(all_get_result_times) / len(all_get_result_times)
                    avg_total = sum(all_total_times) / len(all_total_times)
                    
                    preprocess_ratio = (sum(all_preprocess_times) / sum(all_total_times)) * 100
                    execute_ratio = (sum(all_inference_times) / sum(all_total_times)) * 100
                    get_result_ratio = (sum(all_get_result_times) / sum(all_total_times)) * 100
                    
                    logger.info(f"时间统计:")
                    logger.info(f"  预处理：{avg_preprocess:.4f} 秒 ({preprocess_ratio:.1f}%)")
                    logger.info(f"  模型推理：{avg_inference:.4f} 秒 ({execute_ratio:.1f}%)")
                    logger.info(f"  后处理：{avg_get_result:.4f} 秒 ({get_result_ratio:.1f}%)")
                    logger.info(f"  总时间：{avg_total:.4f} 秒")
                    logger.info(f"性能统计:")
                    logger.info(f"  平均推理时间：{avg_inference:.4f} 秒")
                    logger.info(f"  最小推理时间：{min(all_inference_times):.4f} 秒")
                    logger.info(f"  最大推理时间：{max(all_inference_times):.4f} 秒")
                    logger.info(f"  吞吐率：{1.0/avg_inference:.2f} FPS")
    
    finally:
        if args.mode == 'multithread':
            inference.stop()
        else:
            inference.destroy()
    
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
        logger.info(f"结果已保存到：{results_file}")
    
    return 0


def cmd_check(args):
    """环境检查命令"""
    print("=" * 60)
    print("昇腾推理项目 - 环境检查")
    print("=" * 60)
    
    errors = []
    warnings = []
    passed = 0
    has_acl = False
    
    print("\n[1] Python 环境")
    version = sys.version_info
    print(f"  Python 版本：{version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 6:
        print("  [OK] 版本兼容")
        passed += 1
    else:
        print("  [ERROR] 需要 Python 3.6+")
        errors.append("Python 版本过低")
    
    print("\n[2] 依赖库")
    required = ['numpy', 'PIL']
    optional = ['cv2']
    
    for lib in required:
        try:
            __import__(lib)
            print(f"  [OK] {lib} 已安装")
            passed += 1
        except ImportError:
            print(f"  [ERROR] {lib} 未安装")
            errors.append(f"缺少库：{lib}")
    
    for lib in optional:
        try:
            __import__(lib)
            print(f"  [OK] {lib} 已安装 (可选)")
        except ImportError:
            print(f"  [INFO] {lib} 未安装 (可选)")
    
    print("\n[3] ACL 库")
    try:
        import acl
        print("  [OK] ACL 库可导入")
        has_acl = True
        passed += 1
    except ImportError:
        print("  [WARNING] ACL 库未找到 (仅在昇腾设备上可用)")
        warnings.append("ACL 库未安装 - 如在非昇腾设备测试可忽略")
    
    print("\n[4] 配置模块")
    try:
        config = Config()
        print(f"  模型路径：{config.model_path}")
        print(f"  设备 ID: {config.device_id}")
        print(f"  分辨率：{config.resolution}")
        print(f"  AI 核心数：{MAX_AI_CORES}")
        print("  [OK] 配置模块正常")
        passed += 1
    except Exception as e:
        print(f"  [ERROR] 配置模块异常：{e}")
        errors.append(f"配置模块：{e}")
    
    print("\n[5] 模型文件")
    try:
        config = Config()
        if os.path.exists(config.model_path):
            file_size = os.path.getsize(config.model_path)
            print(f"  [OK] 模型文件存在：{config.model_path}")
            print(f"       文件大小：{file_size / 1024 / 1024:.2f} MB")
            passed += 1
        else:
            print(f"  [WARNING] 模型文件不存在：{config.model_path}")
            warnings.append(f"模型文件缺失：{config.model_path}")
    except Exception as e:
        print(f"  [ERROR] 检查失败：{e}")
        errors.append(f"模型检查：{e}")
    
    print("\n[6] 推理模块")
    try:
        from src.inference import Inference, MultithreadInference, HighResInference
        print("  [OK] 推理类导入成功")
        print("    - Inference (基础推理)")
        print("    - MultithreadInference (多线程推理)")
        print("    - HighResInference (高分辨率推理)")
        passed += 1
    except Exception as e:
        print(f"  [ERROR] 推理模块导入失败：{e}")
        errors.append(f"推理模块：{e}")
    
    print("\n[7] API 模块")
    try:
        from src.api import InferenceAPI
        print("  [OK] InferenceAPI 导入成功")
        passed += 1
    except Exception as e:
        print(f"  [ERROR] API 模块导入失败：{e}")
        errors.append(f"API 模块：{e}")
    
    print("\n[8] 支持的分辨率")
    for res, (w, h) in SUPPORTED_RESOLUTIONS.items():
        print(f"  {res}: {w}x{h}")
    passed += 1
    
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    print(f"通过：{passed} 项")
    
    if errors:
        print(f"\n❌ 错误 ({len(errors)} 项):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    
    if warnings:
        print(f"\n⚠️  警告 ({len(warnings)} 项):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    print("\n" + "=" * 60)
    
    if not errors:
        print("[SUCCESS] ✅ 所有必需检查通过！")
        return 0
    else:
        print(f"[FAILED] ❌ {len(errors)} 项检查失败，请修复错误")
        return 1


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


def cmd_package(args):
    """项目打包命令"""
    script_dir = Path(__file__).parent.resolve()
    output_zip = Path(args.output) if args.output else script_dir / f"{script_dir.name}_packaged.zip"
    
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.git',
        '.gitignore',
        '*.zip',
        'data',
        '*.log',
        '.DS_Store',
        'Thumbs.db',
        '.pytest_cache',
        '.mypy_cache',
        '.tox',
        '.venv',
        'venv',
        'env',
        'build',
        'dist',
        '*.egg-info'
    ]
    
    exclude_dirs = {p for p in exclude_patterns if '*' not in p and not p.startswith('.')}
    
    print(f"正在打包项目：{script_dir}")
    print(f"输出文件：{output_zip}")
    print()
    
    file_count = 0
    total_size = 0
    
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    if output_zip.exists():
        output_zip.unlink()
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(script_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                file_path = Path(root) / file
                
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern.startswith('*') and file_path.suffix == pattern[1:]:
                        should_exclude = True
                        break
                    elif pattern.startswith('.') and file_path.name.startswith(pattern):
                        should_exclude = True
                        break
                
                if should_exclude:
                    continue
                
                rel_path = file_path.relative_to(script_dir.parent)
                zipf.write(file_path, rel_path)
                
                file_count += 1
                total_size += file_path.stat().st_size
                
                if file_count <= 20 or file_count % 10 == 0:
                    print(f"  添加：{rel_path}")
    
    zip_size = output_zip.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print()
    print(f"打包完成!")
    print(f"  文件数量：{file_count}")
    print(f"  原始大小：{total_size / 1024:.2f} KB")
    print(f"  压缩后大小：{zip_size / 1024:.2f} KB")
    print(f"  压缩率：{compression_ratio:.1f}%")
    print(f"  输出文件：{output_zip}")
    
    return 0


def cmd_config(args):
    """配置管理命令"""
    if args.show:
        print("=" * 60)
        print("当前配置")
        print("=" * 60)
        
        config = Config()
        if args.config:
            config = Config.from_json(args.config)
        
        print(f"\n模型配置:")
        print(f"  model_path: {config.model_path}")
        print(f"  device_id: {config.device_id}")
        print(f"  resolution: {config.resolution}")
        
        print(f"\n推理配置:")
        print(f"  tile_size: {config.tile_size}")
        print(f"  overlap: {config.overlap}")
        print(f"  num_threads: {config.num_threads}")
        print(f"  backend: {config.backend}")
        
        print(f"\n检测配置:")
        print(f"  conf_threshold: {config.conf_threshold}")
        print(f"  iou_threshold: {config.iou_threshold}")
        print(f"  max_detections: {config.max_detections}")
        
        print(f"\n日志配置:")
        print(f"  enable_logging: {config.enable_logging}")
        print(f"  log_level: {config.log_level}")
        print(f"  enable_profiling: {config.enable_profiling}")
        
        print(f"\n系统配置:")
        print(f"  MAX_AI_CORES: {MAX_AI_CORES}")
        
        print(f"\n支持的分辨率:")
        for res, (w, h) in SUPPORTED_RESOLUTIONS.items():
            print(f"  {res}: {w}x{h}")
        
        return 0
    
    if args.validate:
        print("=" * 60)
        print("配置验证")
        print("=" * 60)
        
        errors = []
        
        config = Config()
        if args.config:
            config = Config.from_json(args.config)
        
        if not os.path.exists(config.model_path):
            errors.append(f"模型文件不存在：{config.model_path}")
        
        if config.resolution not in SUPPORTED_RESOLUTIONS:
            errors.append(f"不支持的分辨率：{config.resolution}")
        
        if config.device_id < 0:
            errors.append(f"无效的设备 ID：{config.device_id}")
        
        if config.num_threads < 1 or config.num_threads > MAX_AI_CORES:
            errors.append(f"线程数应在 1-{MAX_AI_CORES} 之间：{config.num_threads}")
        
        if config.backend not in ['pil', 'opencv']:
            errors.append(f"不支持的后端：{config.backend}")
        
        if config.conf_threshold < 0 or config.conf_threshold > 1:
            errors.append(f"置信度阈值应在 0-1 之间：{config.conf_threshold}")
        
        if config.iou_threshold < 0 or config.iou_threshold > 1:
            errors.append(f"IOU 阈值应在 0-1 之间：{config.iou_threshold}")
        
        if errors:
            print("\n❌ 配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return 1
        else:
            print("\n✅ 配置验证通过")
            return 0
    
    if args.generate:
        output_path = Path(args.generate)
        config = Config()
        
        config_dict = {
            "model_path": config.model_path,
            "device_id": config.device_id,
            "resolution": config.resolution,
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "num_threads": config.num_threads,
            "backend": config.backend,
            "conf_threshold": config.conf_threshold,
            "iou_threshold": config.iou_threshold,
            "max_detections": config.max_detections,
            "enable_logging": config.enable_logging,
            "log_level": config.log_level,
            "enable_profiling": config.enable_profiling
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"配置文件已生成：{output_path}")
        return 0
    
    print("请指定操作：--show, --validate 或 --generate")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description='昇腾推理工具 - 统一命令行入口',
        prog='ascend-inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''版本：{__version__}

示例：
  # 推理单张图片
  python main.py infer test.jpg --model models/yolov8s.om

  # 使用配置文件
  python main.py infer test.jpg --config config/default.json

  # 性能测试
  python main.py infer test.jpg --benchmark --iterations 100

  # 环境检查
  python main.py check

  # 图像增强
  python main.py enhance test.jpg --output ./enhanced

  # 项目打包
  python main.py package

  # 查看配置
  python main.py config --show
'''
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # infer 命令
    infer_parser = subparsers.add_parser('infer', help='推理（支持单张/批量/性能测试）')
    infer_parser.add_argument('input', help='输入图像路径或目录')
    infer_parser.add_argument('--config', help='JSON 配置文件路径')
    infer_parser.add_argument('--model', help='模型路径')
    infer_parser.add_argument('--device', type=int, help='设备 ID')
    infer_parser.add_argument('--resolution', help='分辨率')
    infer_parser.add_argument('--backend', choices=['pil', 'opencv'], help='图像读取后端')
    infer_parser.add_argument('--mode', default='base', choices=['base', 'multithread', 'high_res'], help='推理模式')
    infer_parser.add_argument('--iterations', type=int, default=1, help='推理次数（>1 时进行性能测试）')
    infer_parser.add_argument('--threads-per-core', type=int, default=1, help='每个 AI 核心的线程数')
    infer_parser.add_argument('--output', help='输出结果目录')
    infer_parser.add_argument('--benchmark', action='store_true', help='运行性能基准测试')
    infer_parser.add_argument('--test-threads', action='store_true', help='测试多线程性能')
    infer_parser.add_argument('--test-resolutions', action='store_true', help='测试分辨率影响')
    infer_parser.add_argument('--thread-counts', nargs='+', type=int, default=[1, 2, 4, 8], help='测试的线程数列表')
    infer_parser.set_defaults(func=cmd_infer)
    
    # check 命令
    check_parser = subparsers.add_parser('check', help='环境检查')
    check_parser.set_defaults(func=cmd_check)
    
    # enhance 命令
    enhance_parser = subparsers.add_parser('enhance', help='图像增强')
    enhance_parser.add_argument('image_path', help='输入图像路径')
    enhance_parser.add_argument('--output', help='输出目录')
    enhance_parser.add_argument('--count', type=int, default=1, help='扩增数量')
    enhance_parser.add_argument('--resolutions', nargs='+', choices=list(SUPPORTED_RESOLUTIONS.keys()), help='分辨率列表')
    enhance_parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='处理后端')
    enhance_parser.add_argument('--interpolation', choices=['nearest', 'bilinear', 'bicubic'], default='bilinear', help='插值方法')
    enhance_parser.set_defaults(func=cmd_enhance)
    
    # package 命令
    package_parser = subparsers.add_parser('package', help='项目打包')
    package_parser.add_argument('--output', help='输出 zip 文件路径')
    package_parser.set_defaults(func=cmd_package)
    
    # config 命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_parser.add_argument('--config', help='JSON 配置文件路径')
    config_parser.add_argument('--show', action='store_true', help='显示当前配置')
    config_parser.add_argument('--validate', action='store_true', help='验证配置')
    config_parser.add_argument('--generate', metavar='PATH', help='生成默认配置文件')
    config_parser.set_defaults(func=cmd_config)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
