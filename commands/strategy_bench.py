#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略验证评测命令

用于验证各种加速策略的效果，支持：
- 多策略对比测试
- 加速比和并行效率计算
- 策略验证报告生成
"""

import os
import argparse
from typing import List, Optional

from config import Config
from benchmark import StrategyValidationScenario, BenchmarkResult
from evaluations.routes import REMOTE_SENSING_ROUTES
from utils.logger import LoggerConfig


logger = LoggerConfig.setup_logger('ascend_inference.strategy_bench')


AVAILABLE_STRATEGIES = ['multithread', 'batch', 'pipeline', 'memory_pool', 'high_res']


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器
    
    Returns:
        argparse.ArgumentParser: 解析器
    """
    parser = argparse.ArgumentParser(
        description='策略验证评测 - 验证加速策略效果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 测试所有策略
  python main.py strategy-bench --model models/yolov8n.om --image test.jpg
  
  # 测试特定策略
  python main.py strategy-bench --model models/yolov8n.om --image test.jpg \\
      --strategies multithread batch pipeline
  
  # 自定义参数
  python main.py strategy-bench --model models/yolov8n.om --image test.jpg \\
      --iterations 50 --threads 8 --output report.txt
'''
    )
    
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--image', required=True, help='测试图像路径')
    parser.add_argument('--strategies', nargs='+', choices=AVAILABLE_STRATEGIES,
                        default=AVAILABLE_STRATEGIES, help='要测试的策略')
    parser.add_argument('--iterations', type=int, default=50, help='测试迭代次数 (默认: 50)')
    parser.add_argument('--warmup', type=int, default=3, help='预热次数 (默认: 3)')
    parser.add_argument('--threads', type=int, default=4, help='多线程策略的线程数')
    parser.add_argument('--batch-size', type=int, default=4, help='批处理策略的批大小')
    parser.add_argument('--output', '-o', help='报告输出路径')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='输出格式')
    parser.add_argument('--device', type=int, default=0, help='设备ID')
    parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='图像处理后端')
    parser.add_argument('--routes', nargs='+', choices=list(REMOTE_SENSING_ROUTES), help='遥感路线类型')
    parser.add_argument('--image-size-tiers', nargs='+', help='遥感大图分档，例如 6K')
    
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
    
    if not os.path.exists(args.image):
        logger.error(f"图像文件不存在: {args.image}")
        return False
    
    if args.iterations < 1:
        logger.error("迭代次数必须大于0")
        return False
    
    if args.threads < 1:
        logger.error("线程数必须大于0")
        return False
    
    return True


def run_benchmark(args: argparse.Namespace) -> int:
    """运行策略验证评测
    
    Args:
        args: 命令行参数
        
    Returns:
        int: 退出码
    """
    logger.info("=" * 60)
    logger.info("策略验证评测")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model}")
    logger.info(f"测试策略: {', '.join(args.strategies)}")
    logger.info(f"迭代次数: {args.iterations}")
    logger.info(f"预热次数: {args.warmup}")
    
    routes = getattr(args, 'routes', None)
    image_size_tiers = getattr(args, 'image_size_tiers', None)
    if not isinstance(routes, (list, tuple)):
        routes = None
    if not isinstance(image_size_tiers, (list, tuple)):
        image_size_tiers = None

    scenario = StrategyValidationScenario({
        'strategies': args.strategies,
        'iterations': args.iterations,
        'warmup': args.warmup,
        'threads': args.threads,
        'batch_size': args.batch_size,
        'device_id': args.device,
        'backend': args.backend,
        'routes': list(routes or []),
        'image_size_tiers': list(image_size_tiers or []),
    })
    
    results = scenario.run([args.model], [args.image])
    
    if not results:
        logger.error("评测失败，没有产生有效结果")
        return 1
    
    report = scenario.generate_report(results)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"报告已保存到: {args.output}")
    else:
        print(report)
    
    return 0


def cmd_strategy_bench(args: argparse.Namespace) -> int:
    """策略验证评测命令入口
    
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
    exit(cmd_strategy_bench(args))
