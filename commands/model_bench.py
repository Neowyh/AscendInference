#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型选型评测命令

用于对比不同模型的性能，支持：
- 多模型同时测试
- 完整的延迟分布统计
- 模型信息输出
- 对比报告生成
"""

import os
import argparse
from typing import List, Optional

from config import Config
from benchmark import ModelSelectionScenario, RouteExperimentScenario, BenchmarkResult
from evaluations.routes import REMOTE_SENSING_ROUTES
from evaluations.tiers import STANDARD_INPUT_TIERS
from utils.logger import LoggerConfig


logger = LoggerConfig.setup_logger('ascend_inference.model_bench')


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器
    
    Returns:
        argparse.ArgumentParser: 解析器
    """
    parser = argparse.ArgumentParser(
        description='模型选型评测 - 对比不同模型的性能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 测试单个模型
  python main.py model-bench models/yolov8s.om --images test.jpg
  
  # 对比多个模型
  python main.py model-bench models/yolov5s.om models/yolov8n.om models/yolov10n.om \\
      --images test1.jpg test2.jpg --iterations 100 --warmup 5
  
  # 输出到文件
  python main.py model-bench models/*.om --images test.jpg --output report.txt
'''
    )
    
    parser.add_argument('models', nargs='+', help='模型路径列表')
    parser.add_argument('--images', nargs='+', required=True, help='测试图像路径')
    parser.add_argument('--iterations', type=int, default=100, help='测试迭代次数 (默认: 100)')
    parser.add_argument('--warmup', type=int, default=5, help='预热次数 (默认: 5)')
    parser.add_argument('--output', '-o', help='报告输出路径')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='输出格式 (默认: text)')
    parser.add_argument('--device', type=int, default=0, help='设备ID (默认: 0)')
    parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='图像处理后端')
    parser.add_argument('--enable-monitoring', action='store_true', help='启用资源监控')
    parser.add_argument(
        '--input-tiers',
        nargs='+',
        choices=list(STANDARD_INPUT_TIERS),
        default=list(STANDARD_INPUT_TIERS),
        help='标准评测输入分档',
    )
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
    for model_path in args.models:
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
    
    for image_path in args.images:
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return False
    
    if args.iterations < 1:
        logger.error("迭代次数必须大于0")
        return False
    
    if args.warmup < 0:
        logger.error("预热次数不能为负数")
        return False
    
    return True


def run_benchmark(args: argparse.Namespace) -> int:
    """运行模型选型评测
    
    Args:
        args: 命令行参数
        
    Returns:
        int: 退出码
    """
    logger.info("=" * 60)
    logger.info("模型选型评测")
    logger.info("=" * 60)
    logger.info(f"模型数量: {len(args.models)}")
    logger.info(f"测试图像: {len(args.images)}")
    logger.info(f"迭代次数: {args.iterations}")
    logger.info(f"预热次数: {args.warmup}")
    
    scenario_config = {
        'iterations': args.iterations,
        'warmup': args.warmup,
        'enable_monitoring': args.enable_monitoring,
        'device_id': args.device,
        'backend': args.backend,
        'input_tiers': list(args.input_tiers),
    }
    scenario_cls = ModelSelectionScenario
    routes = getattr(args, 'routes', None)
    image_size_tiers = getattr(args, 'image_size_tiers', None)
    if not isinstance(routes, (list, tuple)):
        routes = None
    if not isinstance(image_size_tiers, (list, tuple)):
        image_size_tiers = None
    if routes or image_size_tiers:
        scenario_cls = RouteExperimentScenario
        scenario_config['routes'] = list(routes or REMOTE_SENSING_ROUTES)
        scenario_config['image_size_tiers'] = list(image_size_tiers or ['6K'])

    scenario = scenario_cls(scenario_config)
    
    results = scenario.run(args.models, args.images)
    
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


def cmd_model_bench(args: argparse.Namespace) -> int:
    """模型选型评测命令入口
    
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
    exit(cmd_model_bench(args))
