#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极限性能评测命令

用于追求极限吞吐量，支持：
- 策略组合配置
- 资源利用率监控
- 极限性能报告生成
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import Config
from benchmark import ExtremePerformanceScenario, BenchmarkResult
from benchmark.reporters import render_report
from reporting.archive import archive_result
from utils.logger import LoggerConfig


logger = LoggerConfig.setup_logger('ascend_inference.extreme_bench')


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器
    
    Returns:
        argparse.ArgumentParser: 解析器
    """
    parser = argparse.ArgumentParser(
        description='极限性能评测 - 追求极限吞吐量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用配置文件
  python main.py extreme-bench --model models/yolov8n.om --images test_images/ \\
      --config config/extreme.json
  
  # 快速测试
  python main.py extreme-bench --model models/yolov8n.om --images test.jpg \\
      --threads 4 --duration 10
  
  # 启用多策略组合
  python main.py extreme-bench --model models/yolov8n.om --images test.jpg \\
      --enable-multithread --threads 8 --enable-memory-pool --output report.txt

配置文件示例 (config/extreme.json):
  {
    "strategies": {
      "multithread": {"enabled": true, "num_threads": 4},
      "batch": {"enabled": true, "batch_size": 8},
      "memory_pool": {"enabled": true, "pool_size": 15}
    }
  }
'''
    )
    
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--images', nargs='+', required=True, help='测试图像路径或目录')
    parser.add_argument('--config', help='策略配置文件 (JSON)')
    parser.add_argument('--iterations', type=int, default=100, help='测试迭代次数 (默认: 100)')
    parser.add_argument('--warmup', type=int, default=5, help='预热次数 (默认: 5)')
    parser.add_argument('--duration', type=int, default=10, help='测试时长/秒 (默认: 10)')
    parser.add_argument('--output', '-o', help='报告输出路径')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--device', type=int, default=0, help='设备ID')
    parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='图像处理后端')
    
    strategy_group = parser.add_argument_group('策略开关')
    strategy_group.add_argument('--enable-multithread', action='store_true', help='启用多线程策略')
    strategy_group.add_argument('--threads', type=int, default=4, help='多线程数')
    strategy_group.add_argument('--enable-batch', action='store_true', help='启用批处理策略')
    strategy_group.add_argument('--batch-size', type=int, default=4, help='批大小')
    strategy_group.add_argument('--enable-pipeline', action='store_true', help='启用流水线策略')
    strategy_group.add_argument('--enable-memory-pool', action='store_true', help='启用内存池策略')
    strategy_group.add_argument('--enable-high-res', action='store_true', help='启用高分辨率策略')
    
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """验证参数
    
    Args:
        args: 命令行参数
        
    Returns:
        bool: 是否有效
    """
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return False
    
    for image_path in args.images:
        if os.path.isdir(image_path):
            continue
        if not os.path.exists(image_path):
            logger.error(f"图像路径不存在: {image_path}")
            return False
    
    if args.config and not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return False
    
    return True


def collect_images(image_paths: List[str]) -> List[str]:
    """收集图像文件
    
    Args:
        image_paths: 输入路径列表
        
    Returns:
        list: 图像文件列表
    """
    images = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for path in image_paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.lower().endswith(supported_extensions):
                    images.append(os.path.join(path, file))
        elif os.path.isfile(path):
            images.append(path)
    
    return images


def build_strategy_config(args: argparse.Namespace) -> Dict[str, Any]:
    """构建策略配置
    
    Args:
        args: 命令行参数
        
    Returns:
        Dict: 策略配置
    """
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('strategies', {})
    
    strategy_config = {}
    
    if args.enable_multithread:
        strategy_config['multithread'] = {
            'enabled': True,
            'num_threads': args.threads
        }
    
    if args.enable_batch:
        strategy_config['batch'] = {
            'enabled': True,
            'batch_size': args.batch_size
        }
    
    if args.enable_pipeline:
        strategy_config['pipeline'] = {
            'enabled': True
        }
    
    if args.enable_memory_pool:
        strategy_config['memory_pool'] = {
            'enabled': True
        }
    
    if args.enable_high_res:
        strategy_config['high_res'] = {
            'enabled': True
        }
    
    return strategy_config


def run_benchmark(args: argparse.Namespace) -> int:
    """运行极限性能评测
    
    Args:
        args: 命令行参数
        
    Returns:
        int: 退出码
    """
    logger.info("=" * 60)
    logger.info("极限性能评测")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model}")
    logger.info(f"测试时长: {args.duration} 秒")
    
    images = collect_images(args.images)
    if not images:
        logger.error("没有找到有效的图像文件")
        return 1
    
    logger.info(f"图像数量: {len(images)}")
    
    strategy_config = build_strategy_config(args)
    
    if strategy_config:
        logger.info("启用的策略:")
        for name, config in strategy_config.items():
            if config.get('enabled'):
                logger.info(f"  - {name}: {config}")
    else:
        logger.info("未启用任何策略（基准测试）")
    
    scenario = ExtremePerformanceScenario({
        'strategy_config': strategy_config,
        'iterations': args.iterations,
        'warmup': args.warmup,
        'duration_seconds': args.duration
    })
    
    results = scenario.run([args.model], images)
    
    if not results:
        logger.error("评测失败，没有产生有效结果")
        return 1
    
    report, report_model, report_extension = render_report(
        results,
        task_name=scenario.name,
        output_format=args.format,
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"报告已保存到: {output_path}")

        route_items = report_model.get("route_comparison", [])
        route_type = route_items[0]["route"] if len(route_items) == 1 else "mixed"
        archived = archive_result(
            output_path.parent / "archives",
            {"task_name": scenario.name, "route_type": route_type},
            report,
            report_model,
            report_extension=report_extension,
        )
        logger.info(f"归档已保存到: {archived['archive_dir']}")
    else:
        print(report)
    
    return 0


def cmd_extreme_bench(args: argparse.Namespace) -> int:
    """极限性能评测命令入口
    
    Args:
        args: 命令行参数
        
    Returns:
        int: 退出码
    """
    if not validate_args(args):
        return 1
    
    return run_benchmark(args)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    exit(cmd_extreme_bench(args))
