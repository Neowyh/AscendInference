#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理命令实现
"""
import os
import time
from typing import Dict, List, Any, Optional
import numpy as np

from config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES
from src.inference import Inference, MultithreadInference, HighResInference
from utils.logger import LoggerConfig

logger = LoggerConfig.setup_logger('ascend_inference.main')
perf_logger = LoggerConfig.setup_logger('ascend_inference.performance', level='info')

# 性能测试常量
DEFAULT_BENCHMARK_ITERATIONS = 100
DEFAULT_THREAD_TEST_ITERATIONS = 20
DEFAULT_RESOLUTION_TEST_ITERATIONS = 20
DEFAULT_THREAD_COUNTS = [1, 2, 4, 8]
DEFAULT_TEST_RESOLUTIONS = ["640x640", "1k", "2k"]


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
    if hasattr(args, 'warmup') and args.warmup is not None:
        overrides['warmup'] = args.warmup
    if hasattr(args, 'warmup_iterations') and args.warmup_iterations is not None:
        overrides['warmup_iterations'] = args.warmup_iterations

    if overrides:
        config.apply_overrides(**overrides)
        logger.info(f"命令行参数覆盖：{overrides}")

    return config


class PerformanceTester:
    """性能测试器（整合所有性能测试功能）"""

    def __init__(self, config: Config):
        self.config = config
        self.results: Dict[str, Any] = {}

    def test_single(self, image_path: str, iterations: int = DEFAULT_BENCHMARK_ITERATIONS) -> Dict:
        """单次推理性能测试"""
        perf_logger.info(f"单次推理性能测试")
        perf_logger.info(f"  图像：{image_path}")
        perf_logger.info(f"  迭代次数：{iterations}")
        perf_logger.info(f"  分辨率：{self.config.resolution}")

        inference = Inference(self.config)

        try:
            if not inference.init():
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
            thread_counts = DEFAULT_THREAD_COUNTS

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
            resolutions = DEFAULT_TEST_RESOLUTIONS

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
    import json
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
            tester.test_single(image_paths[0], iterations=args.iterations or DEFAULT_BENCHMARK_ITERATIONS)

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
