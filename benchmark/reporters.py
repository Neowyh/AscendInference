#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器模块

提供多种格式的评测报告生成功能：
- TextReporter: 文本格式报告
- JsonReporter: JSON格式报告
- HtmlReporter: HTML格式报告
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional

from benchmark.scenarios import BenchmarkResult
from reporting.renderers import JsonReportRenderer, MarkdownReportRenderer


class Reporter(ABC):
    """报告生成器基类"""
    
    @abstractmethod
    def generate(self, results: List[BenchmarkResult]) -> str:
        """生成报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: 报告内容
        """
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """获取文件扩展名
        
        Returns:
            str: 文件扩展名
        """
        pass


class TextReporter(Reporter):
    """文本格式报告生成器"""
    
    def __init__(self, title: str = "性能评测报告"):
        """初始化文本报告生成器
        
        Args:
            title: 报告标题
        """
        self.title = title
    
    def generate(self, results: List[BenchmarkResult]) -> str:
        """生成文本格式报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: 报告内容
        """
        lines = [
            "=" * 80,
            self.title,
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"结果数量: {len(results)}",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            lines.extend(self._format_result(result, i))
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_result(self, result: BenchmarkResult, index: int) -> List[str]:
        """格式化单个结果
        
        Args:
            result: 评测结果
            index: 结果索引
            
        Returns:
            List[str]: 格式化后的行列表
        """
        lines = [
            f"[{index}] {result.scenario_name}",
            "-" * 60,
            f"模型: {result.model_info.name}",
            f"策略: {', '.join(result.strategies) if result.strategies else '无'}",
            ""
        ]
        
        if result.metrics:
            lines.extend(self._format_metrics(result.metrics))
        
        if result.resource_stats:
            lines.extend(self._format_resource_stats(result.resource_stats))
        
        lines.append("")
        
        return lines
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> List[str]:
        """格式化指标
        
        Args:
            metrics: 指标字典
            
        Returns:
            List[str]: 格式化后的行列表
        """
        lines = ["性能指标:"]
        
        if 'fps' in metrics:
            fps = metrics['fps']
            lines.append(f"  纯推理FPS: {fps.get('pure', 0):.2f}")
            lines.append(f"  端到端FPS: {fps.get('e2e', 0):.2f}")
        
        if 'preprocess' in metrics:
            prep = metrics['preprocess']
            lines.append(f"  预处理时间: avg={prep.get('avg', 0):.2f}ms, "
                        f"p50={prep.get('p50', 0):.2f}ms, p95={prep.get('p95', 0):.2f}ms")
        
        if 'execute' in metrics:
            exec = metrics['execute']
            lines.append(f"  推理时间:   avg={exec.get('avg', 0):.2f}ms, "
                        f"p50={exec.get('p50', 0):.2f}ms, p95={exec.get('p95', 0):.2f}ms")
        
        if 'postprocess' in metrics:
            post = metrics['postprocess']
            lines.append(f"  后处理时间: avg={post.get('avg', 0):.2f}ms, "
                        f"p50={post.get('p50', 0):.2f}ms, p95={post.get('p95', 0):.2f}ms")
        
        if 'total' in metrics:
            total = metrics['total']
            lines.append(f"  总时间:     avg={total.get('avg', 0):.2f}ms, "
                        f"p50={total.get('p50', 0):.2f}ms, p95={total.get('p95', 0):.2f}ms, "
                        f"p99={total.get('p99', 0):.2f}ms")
        
        if 'strategy' in metrics:
            strategy = metrics['strategy']
            lines.append(f"  加速比: {strategy.get('speedup', 1.0):.2f}x")
            lines.append(f"  并行效率: {strategy.get('parallel_efficiency', 0):.1f}%")
        
        if 'throughput_fps' in metrics:
            lines.append(f"  吞吐FPS: {metrics['throughput_fps']:.2f}")
        
        return lines
    
    def _format_resource_stats(self, stats: Dict[str, Any]) -> List[str]:
        """格式化资源统计
        
        Args:
            stats: 资源统计字典
            
        Returns:
            List[str]: 格式化后的行列表
        """
        lines = ["资源利用:"]
        
        if 'cpu' in stats:
            cpu = stats['cpu']
            lines.append(f"  CPU: avg={cpu.get('avg', 0):.1f}%, "
                        f"max={cpu.get('max', 0):.1f}%")
        
        if 'memory' in stats:
            mem = stats['memory']
            lines.append(f"  内存: {mem.get('current_mb', 0):.1f}MB / "
                        f"{mem.get('total_mb', 0):.1f}MB ({mem.get('avg_percent', 0):.1f}%)")
        
        if 'npu' in stats:
            npu = stats['npu']
            lines.append(f"  NPU利用率: avg={npu.get('avg_utilization', 0):.1f}%, "
                        f"max={npu.get('max_utilization', 0):.1f}%")
        
        return lines
    
    def get_file_extension(self) -> str:
        """获取文件扩展名
        
        Returns:
            str: .txt
        """
        return ".txt"


class JsonReporter(Reporter):
    """JSON格式报告生成器"""
    
    def __init__(self, indent: int = 2):
        """初始化JSON报告生成器
        
        Args:
            indent: 缩进空格数
        """
        self.indent = indent
    
    def generate(self, results: List[BenchmarkResult]) -> str:
        """生成JSON格式报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: JSON格式报告
        """
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'result_count': len(results),
                'version': '1.0'
            },
            'results': [self._result_to_dict(r) for r in results]
        }
        
        return json.dumps(report, indent=self.indent, ensure_ascii=False, default=str)
    
    def _result_to_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """将结果转换为字典
        
        Args:
            result: 评测结果
            
        Returns:
            Dict: 结果字典
        """
        return {
            'scenario_name': result.scenario_name,
            'model_info': {
                'path': result.model_info.path,
                'name': result.model_info.name,
                'input_size': result.model_info.input_size,
                'output_size': result.model_info.output_size,
                'resolution': result.model_info.resolution
            },
            'metrics': result.metrics,
            'strategies': result.strategies,
            'config': result.config,
            'resource_stats': result.resource_stats,
            'timestamp': result.timestamp
        }
    
    def get_file_extension(self) -> str:
        """获取文件扩展名
        
        Returns:
            str: .json
        """
        return ".json"


class HtmlReporter(Reporter):
    """HTML格式报告生成器"""
    
    def __init__(self, title: str = "性能评测报告", style: str = "default"):
        """初始化HTML报告生成器
        
        Args:
            title: 报告标题
            style: 样式名称
        """
        self.title = title
        self.style = style
    
    def generate(self, results: List[BenchmarkResult]) -> str:
        """生成HTML格式报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: HTML格式报告
        """
        html = [
            '<!DOCTYPE html>',
            '<html lang="zh-CN">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'    <title>{self.title}</title>',
            self._get_style(),
            '</head>',
            '<body>',
            '    <div class="container">',
            f'        <h1>{self.title}</h1>',
            f'        <p class="timestamp">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
            self._generate_summary(results),
            self._generate_results(results),
            '    </div>',
            '</body>',
            '</html>'
        ]
        
        return '\n'.join(html)
    
    def _get_style(self) -> str:
        """获取CSS样式
        
        Returns:
            str: CSS样式
        """
        return '''    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        h3 {
            color: #3498db;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .summary {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .result-card {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
            margin: 15px 0;
            background-color: #fafafa;
        }
        .result-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .metric-item {
            background-color: #fff;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .tag {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.85em;
            margin-right: 5px;
        }
        .progress-bar {
            background-color: #ecf0f1;
            border-radius: 5px;
            height: 20px;
            overflow: hidden;
        }
        .progress-fill {
            background-color: #3498db;
            height: 100%;
            transition: width 0.3s ease;
        }
    </style>'''
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> str:
        """生成摘要部分
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: HTML内容
        """
        scenarios = set(r.scenario_name for r in results)
        models = set(r.model_info.name for r in results)
        
        total_fps = sum(r.metrics.get('fps', {}).get('pure', 0) for r in results if r.metrics)
        avg_fps = total_fps / len(results) if results else 0
        
        return f'''
        <div class="summary">
            <h2>评测摘要</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">测试结果数</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{len(scenarios)}</div>
                    <div class="metric-label">评测场景数</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{len(models)}</div>
                    <div class="metric-label">测试模型数</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{avg_fps:.1f}</div>
                    <div class="metric-label">平均FPS</div>
                </div>
            </div>
        </div>'''
    
    def _generate_results(self, results: List[BenchmarkResult]) -> str:
        """生成结果部分
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: HTML内容
        """
        cards = []
        
        for i, result in enumerate(results, 1):
            cards.append(self._generate_result_card(result, i))
        
        return f'''
        <div class="results">
            <h2>详细结果</h2>
            {''.join(cards)}
        </div>'''
    
    def _generate_result_card(self, result: BenchmarkResult, index: int) -> str:
        """生成结果卡片
        
        Args:
            result: 评测结果
            index: 索引
            
        Returns:
            str: HTML内容
        """
        fps = result.metrics.get('fps', {}) if result.metrics else {}
        pure_fps = fps.get('pure', 0)
        e2e_fps = fps.get('e2e', 0)
        
        strategy_tags = ''.join(f'<span class="tag">{s}</span>' for s in result.strategies) \
            if result.strategies else '<span class="tag">无策略</span>'
        
        return f'''
        <div class="result-card">
            <h3>#{index} {result.scenario_name}</h3>
            <p><strong>模型:</strong> {result.model_info.name}</p>
            <p><strong>策略:</strong> {strategy_tags}</p>
            
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value">{pure_fps:.1f}</div>
                    <div class="metric-label">纯推理FPS</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{e2e_fps:.1f}</div>
                    <div class="metric-label">端到端FPS</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{result.metrics.get('total', {}).get('avg', 0):.1f}ms</div>
                    <div class="metric-label">平均延迟</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{result.metrics.get('total', {}).get('p95', 0):.1f}ms</div>
                    <div class="metric-label">P95延迟</div>
                </div>
            </div>
            
            {self._generate_metrics_table(result)}
        </div>'''
    
    def _generate_metrics_table(self, result: BenchmarkResult) -> str:
        """生成指标表格
        
        Args:
            result: 评测结果
            
        Returns:
            str: HTML内容
        """
        if not result.metrics:
            return ''
        
        rows = []
        for key in ['preprocess', 'execute', 'postprocess', 'total']:
            if key in result.metrics:
                m = result.metrics[key]
                rows.append(f'''
                <tr>
                    <td>{key}</td>
                    <td>{m.get('avg', 0):.2f}</td>
                    <td>{m.get('min', 0):.2f}</td>
                    <td>{m.get('max', 0):.2f}</td>
                    <td>{m.get('p50', 0):.2f}</td>
                    <td>{m.get('p95', 0):.2f}</td>
                    <td>{m.get('p99', 0):.2f}</td>
                </tr>''')
        
        if not rows:
            return ''
        
        return f'''
        <table>
            <thead>
                <tr>
                    <th>阶段</th>
                    <th>平均(ms)</th>
                    <th>最小(ms)</th>
                    <th>最大(ms)</th>
                    <th>P50(ms)</th>
                    <th>P95(ms)</th>
                    <th>P99(ms)</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>'''
    
    def get_file_extension(self) -> str:
        """获取文件扩展名
        
        Returns:
            str: .html
        """
        return ".html"


def create_reporter(format: str = 'text', **kwargs) -> Reporter:
    """创建报告生成器
    
    Args:
        format: 格式类型 ('text', 'json', 'html')
        **kwargs: 额外参数
        
    Returns:
        Reporter: 报告生成器实例
    """
    reporters = {
        'text': TextReporter,
        'json': JsonReporter,
        'html': HtmlReporter
    }
    
    reporter_class = reporters.get(format, TextReporter)
    return reporter_class(**kwargs)


def save_report(report: str, output_path: str, reporter: Reporter) -> None:
    """保存报告到文件
    
    Args:
        report: 报告内容
        output_path: 输出路径
        reporter: 报告生成器
    """
    if not output_path.endswith(reporter.get_file_extension()):
        output_path += reporter.get_file_extension()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


def result_to_report_dict(result: BenchmarkResult) -> Dict[str, Any]:
    """把评测结果转为统一报告模型中的条目。"""
    strategies = getattr(result, "strategies", [])
    if not isinstance(strategies, (list, tuple)):
        strategies = []
    metrics = getattr(result, "metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    resource_stats = getattr(result, "resource_stats", {})
    if not isinstance(resource_stats, dict):
        resource_stats = {}
    config = getattr(result, "config", {})
    if not isinstance(config, dict):
        config = {}
    route_type = getattr(result, "route_type", "") or config.get("route_type") or "standard"
    route_type = route_type if isinstance(route_type, str) else str(route_type)
    model_info = getattr(result, "model_info", None)
    model_name = getattr(model_info, "name", "unknown") if model_info is not None else "unknown"
    model_name = model_name if isinstance(model_name, str) else str(model_name)
    return {
        "scenario_name": result.scenario_name,
        "model_name": model_name,
        "route_type": route_type,
        "strategies": [item if isinstance(item, str) else str(item) for item in strategies],
        "metrics": metrics,
        "resource_stats": resource_stats,
        "config": config,
        "timestamp": result.timestamp,
    }


def build_report_model(results: List[BenchmarkResult], task_name: str) -> Dict[str, Any]:
    """构建统一报告模型。"""
    report_results = [result_to_report_dict(result) for result in results]
    route_groups: Dict[str, Dict[str, Any]] = {}

    for item in report_results:
        route = item["route_type"] or "standard"
        group = route_groups.setdefault(
            route,
            {
                "route": route,
                "result_count": 0,
                "models": set(),
                "strategies": set(),
            },
        )
        group["result_count"] += 1
        group["models"].add(item["model_name"])
        group["strategies"].update(item["strategies"])

    route_comparison = [
        {
            "route": route,
            "result_count": data["result_count"],
            "models": sorted(data["models"]),
            "strategies": sorted(data["strategies"]),
        }
        for route, data in sorted(route_groups.items())
    ]

    return {
        "title": "Evaluation Report",
        "task_name": task_name,
        "generated_at": datetime.now().isoformat(),
        "result_count": len(results),
        "route_comparison": route_comparison,
        "results": report_results,
    }


def render_report(
    results: List[BenchmarkResult],
    task_name: str,
    output_format: str = "text",
) -> tuple[str, Dict[str, Any], str]:
    """渲染统一报告并返回报告体、报告模型和扩展名。"""
    report_model = build_report_model(results, task_name=task_name)
    if output_format == "json":
        return JsonReportRenderer().render(report_model), report_model, ".json"
    return MarkdownReportRenderer().render(report_model), report_model, ".md"
