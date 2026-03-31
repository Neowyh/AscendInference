#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
昇腾推理工具 - 统一命令行入口

所有功能通过此入口调用：
- infer: 推理（单张/批量/性能测试）
- model-bench: 模型选型评测
- strategy-bench: 策略验证评测
- extreme-bench: 极限性能评测
- check: 环境检查
- enhance: 图像增强
- package: 项目打包
- config: 配置管理
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES
from commands import cmd_infer, cmd_check, cmd_enhance, cmd_package, cmd_config
from evaluations.routes import REMOTE_SENSING_ROUTES
from evaluations.tiers import STANDARD_INPUT_TIERS

__version__ = "1.1.0"


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
    infer_parser.add_argument('--config', default='config/default.json', help='JSON 配置文件路径')
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
    
    # ========== 模型选型评测命令 ==========
    model_bench_parser = subparsers.add_parser('model-bench', help='模型选型评测 - 对比不同模型性能')
    model_bench_parser.add_argument('models', nargs='+', help='模型路径列表')
    model_bench_parser.add_argument('--images', nargs='+', required=True, help='测试图像路径')
    model_bench_parser.add_argument('--iterations', type=int, default=100, help='测试迭代次数 (默认: 100)')
    model_bench_parser.add_argument('--warmup', type=int, default=5, help='预热次数 (默认: 5)')
    model_bench_parser.add_argument('--output', '-o', help='报告输出路径')
    model_bench_parser.add_argument('--format', choices=['text', 'json'], default='text', help='输出格式')
    model_bench_parser.add_argument('--device', type=int, default=0, help='设备ID')
    model_bench_parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='图像处理后端')
    model_bench_parser.add_argument('--enable-monitoring', action='store_true', help='启用资源监控')
    model_bench_parser.add_argument(
        '--input-tiers',
        nargs='+',
        choices=list(STANDARD_INPUT_TIERS),
        default=list(STANDARD_INPUT_TIERS),
        help='标准评测输入分档',
    )
    model_bench_parser.add_argument('--routes', nargs='+', choices=list(REMOTE_SENSING_ROUTES), help='遥感路线类型')
    model_bench_parser.add_argument('--image-size-tiers', nargs='+', help='遥感大图分档，例如 6K')
    model_bench_parser.set_defaults(func=_cmd_model_bench)
    
    # ========== 策略验证评测命令 ==========
    strategy_bench_parser = subparsers.add_parser('strategy-bench', help='策略验证评测 - 验证加速策略效果')
    strategy_bench_parser.add_argument('--model', required=True, help='模型路径')
    strategy_bench_parser.add_argument('--image', required=True, help='测试图像路径')
    strategy_bench_parser.add_argument('--strategies', nargs='+', 
                                        choices=['multithread', 'batch', 'pipeline', 'memory_pool', 'high_res'],
                                        default=['multithread', 'batch', 'pipeline', 'memory_pool'],
                                        help='要测试的策略')
    strategy_bench_parser.add_argument('--iterations', type=int, default=50, help='测试迭代次数')
    strategy_bench_parser.add_argument('--warmup', type=int, default=3, help='预热次数')
    strategy_bench_parser.add_argument('--threads', type=int, default=4, help='多线程策略的线程数')
    strategy_bench_parser.add_argument('--batch-size', type=int, default=4, help='批处理策略的批大小')
    strategy_bench_parser.add_argument('--output', '-o', help='报告输出路径')
    strategy_bench_parser.add_argument('--device', type=int, default=0, help='设备ID')
    strategy_bench_parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='图像处理后端')
    strategy_bench_parser.add_argument('--routes', nargs='+', choices=list(REMOTE_SENSING_ROUTES), help='遥感路线类型')
    strategy_bench_parser.add_argument('--image-size-tiers', nargs='+', help='遥感大图分档，例如 6K')
    strategy_bench_parser.set_defaults(func=_cmd_strategy_bench)
    
    # ========== 极限性能评测命令 ==========
    extreme_bench_parser = subparsers.add_parser('extreme-bench', help='极限性能评测 - 追求极限吞吐量')
    extreme_bench_parser.add_argument('--model', required=True, help='模型路径')
    extreme_bench_parser.add_argument('--images', nargs='+', required=True, help='测试图像路径或目录')
    extreme_bench_parser.add_argument('--config', help='策略配置文件 (JSON)')
    extreme_bench_parser.add_argument('--iterations', type=int, default=100, help='测试迭代次数')
    extreme_bench_parser.add_argument('--warmup', type=int, default=5, help='预热次数')
    extreme_bench_parser.add_argument('--duration', type=int, default=10, help='测试时长/秒')
    extreme_bench_parser.add_argument('--output', '-o', help='报告输出路径')
    extreme_bench_parser.add_argument('--device', type=int, default=0, help='设备ID')
    extreme_bench_parser.add_argument('--backend', choices=['pil', 'opencv'], default='pil', help='图像处理后端')
    extreme_bench_parser.add_argument('--enable-multithread', action='store_true', help='启用多线程策略')
    extreme_bench_parser.add_argument('--threads', type=int, default=4, help='多线程数')
    extreme_bench_parser.add_argument('--enable-batch', action='store_true', help='启用批处理策略')
    extreme_bench_parser.add_argument('--batch-size', type=int, default=4, help='批大小')
    extreme_bench_parser.add_argument('--enable-pipeline', action='store_true', help='启用流水线策略')
    extreme_bench_parser.add_argument('--enable-memory-pool', action='store_true', help='启用内存池策略')
    extreme_bench_parser.set_defaults(func=_cmd_extreme_bench)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


def _cmd_model_bench(args: argparse.Namespace) -> int:
    """模型选型评测命令处理"""
    from commands.model_bench import cmd_model_bench
    return cmd_model_bench(args)


def _cmd_strategy_bench(args: argparse.Namespace) -> int:
    """策略验证评测命令处理"""
    from commands.strategy_bench import cmd_strategy_bench
    return cmd_strategy_bench(args)


def _cmd_extreme_bench(args: argparse.Namespace) -> int:
    """极限性能评测命令处理"""
    from commands.extreme_bench import cmd_extreme_bench
    return cmd_extreme_bench(args)


if __name__ == "__main__":
    sys.exit(main())
