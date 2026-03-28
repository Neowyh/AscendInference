#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的统计指标收集器

提供性能评测的统一指标收集和统计功能，支持：
- 分阶段计时（预处理/推理/后处理/排队等待）
- 预热和正式测试分离
- 完整的延迟分布统计（P50/P95/P99）
- FPS计算（纯推理FPS/端到端FPS）
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class TimingRecord:
    """单次计时记录
    
    记录一次推理过程中各阶段的时间消耗
    """
    preprocess_time: float = 0.0
    execute_time: float = 0.0
    postprocess_time: float = 0.0
    queue_wait_time: float = 0.0
    total_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def calculate_total(self) -> float:
        """计算总时间（如果未设置）"""
        if self.total_time == 0.0:
            self.total_time = (
                self.preprocess_time + 
                self.execute_time + 
                self.postprocess_time + 
                self.queue_wait_time
            )
        return self.total_time


@dataclass
class StageStatistics:
    """单阶段统计结果"""
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    sum: float = 0.0
    count: int = 0


def _calc_stats(values: List[float]) -> StageStatistics:
    """计算统计指标
    
    Args:
        values: 时间值列表（秒）
        
    Returns:
        StageStatistics: 统计结果（毫秒）
    """
    if not values:
        return StageStatistics()
    
    arr = np.array(values) * 1000
    
    return StageStatistics(
        avg=float(np.mean(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        std=float(np.std(arr)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        sum=float(np.sum(arr)),
        count=len(values)
    )


class MetricsCollector:
    """统一的统计指标收集器
    
    支持预热和正式测试分离，提供完整的性能统计
    
    Example:
        collector = MetricsCollector()
        
        # 预热阶段
        for _ in range(5):
            record = TimingRecord()
            record.preprocess_time = 0.01
            record.execute_time = 0.02
            record.postprocess_time = 0.005
            collector.record(record)
        
        # 结束预热
        collector.finish_warmup()
        
        # 正式测试
        for _ in range(100):
            record = TimingRecord()
            # ... 记录时间
            collector.record(record)
        
        # 获取统计结果
        stats = collector.get_statistics()
    """
    
    def __init__(self, auto_warmup: bool = True, warmup_iterations: int = 5):
        """初始化指标收集器
        
        Args:
            auto_warmup: 是否自动判断预热结束
            warmup_iterations: 预热迭代次数（auto_warmup=True时有效）
        """
        self.records: List[TimingRecord] = []
        self.warmup_records: List[TimingRecord] = []
        self.is_warmup: bool = True
        self.auto_warmup: bool = auto_warmup
        self.warmup_iterations: int = warmup_iterations
        
        self._start_time: Optional[float] = None
        self._current_record: Optional[TimingRecord] = None
    
    def start_iteration(self) -> TimingRecord:
        """开始一次新的迭代，返回计时记录对象
        
        Returns:
            TimingRecord: 计时记录对象
        """
        self._current_record = TimingRecord()
        self._start_time = time.time()
        return self._current_record
    
    def record_preprocess(self, elapsed: float) -> None:
        """记录预处理时间
        
        Args:
            elapsed: 预处理耗时（秒）
        """
        if self._current_record:
            self._current_record.preprocess_time = elapsed
    
    def record_execute(self, elapsed: float) -> None:
        """记录推理执行时间
        
        Args:
            elapsed: 执行耗时（秒）
        """
        if self._current_record:
            self._current_record.execute_time = elapsed
    
    def record_postprocess(self, elapsed: float) -> None:
        """记录后处理时间
        
        Args:
            elapsed: 后处理耗时（秒）
        """
        if self._current_record:
            self._current_record.postprocess_time = elapsed
    
    def record_queue_wait(self, elapsed: float) -> None:
        """记录排队等待时间
        
        Args:
            elapsed: 等待耗时（秒）
        """
        if self._current_record:
            self._current_record.queue_wait_time = elapsed
    
    def finish_iteration(self) -> None:
        """结束当前迭代，记录结果"""
        if self._current_record:
            self._current_record.calculate_total()
            self.record(self._current_record)
            self._current_record = None
    
    def record(self, record: TimingRecord) -> None:
        """记录一次计时
        
        Args:
            record: 计时记录
        """
        if self.is_warmup:
            self.warmup_records.append(record)
            if self.auto_warmup and len(self.warmup_records) >= self.warmup_iterations:
                self.finish_warmup()
        else:
            self.records.append(record)
    
    def finish_warmup(self) -> None:
        """结束预热阶段"""
        self.is_warmup = False
    
    def reset(self) -> None:
        """重置收集器"""
        self.records.clear()
        self.warmup_records.clear()
        self.is_warmup = True
        self._current_record = None
        self._start_time = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """计算统计结果
        
        Returns:
            Dict: 统计结果，包含：
                - preprocess: 预处理统计
                - execute: 推理执行统计
                - postprocess: 后处理统计
                - queue_wait: 排队等待统计
                - total: 总时间统计
                - ratios: 各阶段时间占比
                - fps: FPS指标
                - iterations: 迭代次数信息
        """
        if not self.records:
            return self._empty_statistics()
        
        preprocess_times = [r.preprocess_time for r in self.records]
        execute_times = [r.execute_time for r in self.records]
        postprocess_times = [r.postprocess_time for r in self.records]
        total_times = [r.total_time for r in self.records]
        queue_wait_times = [r.queue_wait_time for r in self.records]
        
        total_sum = sum(total_times)
        
        preprocess_stats = _calc_stats(preprocess_times)
        execute_stats = _calc_stats(execute_times)
        postprocess_stats = _calc_stats(postprocess_times)
        total_stats = _calc_stats(total_times)
        queue_wait_stats = _calc_stats(queue_wait_times)
        
        return {
            'preprocess': preprocess_stats.__dict__,
            'execute': execute_stats.__dict__,
            'postprocess': postprocess_stats.__dict__,
            'total': total_stats.__dict__,
            'queue_wait': queue_wait_stats.__dict__,
            'ratios': {
                'preprocess': (preprocess_stats.sum / total_stats.sum * 100) if total_stats.sum > 0 else 0,
                'execute': (execute_stats.sum / total_stats.sum * 100) if total_stats.sum > 0 else 0,
                'postprocess': (postprocess_stats.sum / total_stats.sum * 100) if total_stats.sum > 0 else 0,
                'queue_wait': (queue_wait_stats.sum / total_stats.sum * 100) if total_stats.sum > 0 else 0
            },
            'fps': {
                'pure': 1000.0 / execute_stats.avg if execute_stats.avg > 0 else 0,
                'e2e': 1000.0 / total_stats.avg if total_stats.avg > 0 else 0
            },
            'iterations': {
                'warmup': len(self.warmup_records),
                'test': len(self.records),
                'total': len(self.warmup_records) + len(self.records)
            },
            'duration': {
                'test_time_ms': total_stats.sum,
                'avg_iteration_ms': total_stats.avg
            }
        }
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """返回空统计结果"""
        empty_stats = StageStatistics().__dict__
        return {
            'preprocess': empty_stats.copy(),
            'execute': empty_stats.copy(),
            'postprocess': empty_stats.copy(),
            'total': empty_stats.copy(),
            'queue_wait': empty_stats.copy(),
            'ratios': {
                'preprocess': 0.0,
                'execute': 0.0,
                'postprocess': 0.0,
                'queue_wait': 0.0
            },
            'fps': {
                'pure': 0.0,
                'e2e': 0.0
            },
            'iterations': {
                'warmup': len(self.warmup_records),
                'test': 0,
                'total': len(self.warmup_records)
            },
            'duration': {
                'test_time_ms': 0.0,
                'avg_iteration_ms': 0.0
            }
        }
    
    def get_summary(self) -> str:
        """获取统计摘要字符串
        
        Returns:
            str: 格式化的统计摘要
        """
        stats = self.get_statistics()
        
        lines = [
            "=" * 60,
            "性能统计摘要",
            "=" * 60,
            f"迭代次数: 预热={stats['iterations']['warmup']}, 测试={stats['iterations']['test']}",
            "",
            "时间统计 (毫秒):",
            f"  预处理:   avg={stats['preprocess']['avg']:.2f}, "
            f"p50={stats['preprocess']['p50']:.2f}, p95={stats['preprocess']['p95']:.2f}",
            f"  推理执行: avg={stats['execute']['avg']:.2f}, "
            f"p50={stats['execute']['p50']:.2f}, p95={stats['execute']['p95']:.2f}",
            f"  后处理:   avg={stats['postprocess']['avg']:.2f}, "
            f"p50={stats['postprocess']['p50']:.2f}, p95={stats['postprocess']['p95']:.2f}",
            f"  总时间:   avg={stats['total']['avg']:.2f}, "
            f"p50={stats['total']['p50']:.2f}, p95={stats['total']['p95']:.2f}, p99={stats['total']['p99']:.2f}",
            "",
            "时间占比:",
            f"  预处理:   {stats['ratios']['preprocess']:.1f}%",
            f"  推理执行: {stats['ratios']['execute']:.1f}%",
            f"  后处理:   {stats['ratios']['postprocess']:.1f}%",
            f"  排队等待: {stats['ratios']['queue_wait']:.1f}%",
            "",
            "性能指标:",
            f"  纯推理FPS: {stats['fps']['pure']:.2f}",
            f"  端到端FPS: {stats['fps']['e2e']:.2f}",
            "=" * 60
        ]
        
        return "\n".join(lines)


class MultiThreadMetricsCollector:
    """多线程指标收集器
    
    用于收集多线程/流水线场景下的吞吐量指标
    """
    
    def __init__(self):
        """初始化多线程指标收集器"""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_tasks: int = 0
        self.completed_tasks: int = 0
        self.latencies: List[float] = []
        self.throughput_records: List[Dict[str, Any]] = []
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
    
    def stop(self) -> None:
        """停止计时"""
        self.end_time = time.time()
    
    def record_task_complete(self, latency: float) -> None:
        """记录任务完成
        
        Args:
            latency: 任务延迟（秒）
        """
        self.completed_tasks += 1
        self.latencies.append(latency)
    
    def add_throughput_sample(self, timestamp: float, completed: int) -> None:
        """添加吞吐量采样点
        
        Args:
            timestamp: 时间戳
            completed: 已完成任务数
        """
        self.throughput_records.append({
            'timestamp': timestamp,
            'completed': completed
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """计算统计结果
        
        Returns:
            Dict: 统计结果
        """
        if not self.start_time or not self.end_time:
            return {}
        
        total_time = self.end_time - self.start_time
        
        if not self.latencies:
            return {
                'total_time': total_time,
                'completed_tasks': self.completed_tasks,
                'throughput_fps': self.completed_tasks / total_time if total_time > 0 else 0
            }
        
        latency_stats = _calc_stats(self.latencies)
        
        return {
            'total_time': total_time,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'throughput_fps': self.completed_tasks / total_time if total_time > 0 else 0,
            'latency': latency_stats.__dict__,
            'throughput_samples': self.throughput_records
        }
    
    def reset(self) -> None:
        """重置收集器"""
        self.start_time = None
        self.end_time = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.latencies.clear()
        self.throughput_records.clear()


class StrategyMetrics:
    """策略相关指标
    
    用于收集策略特定的性能指标，如加速比、并行效率等
    """
    
    def __init__(self, baseline_fps: float = 0.0):
        """初始化策略指标
        
        Args:
            baseline_fps: 基准FPS（无策略时的FPS）
        """
        self.baseline_fps = baseline_fps
        self.strategy_fps: float = 0.0
        self.theoretical_speedup: float = 1.0
        self.collector = MetricsCollector(auto_warmup=False)
    
    def set_baseline(self, fps: float) -> None:
        """设置基准FPS
        
        Args:
            fps: 基准FPS
        """
        self.baseline_fps = fps
    
    def set_strategy_fps(self, fps: float) -> None:
        """设置策略FPS
        
        Args:
            fps: 策略FPS
        """
        self.strategy_fps = fps
    
    def set_theoretical_speedup(self, speedup: float) -> None:
        """设置理论加速比
        
        Args:
            speedup: 理论加速比
        """
        self.theoretical_speedup = speedup
    
    def get_speedup(self) -> float:
        """计算加速比
        
        Returns:
            float: 加速比
        """
        if self.baseline_fps <= 0:
            return 1.0
        return self.strategy_fps / self.baseline_fps
    
    def get_parallel_efficiency(self) -> float:
        """计算并行效率
        
        Returns:
            float: 并行效率（百分比）
        """
        if self.theoretical_speedup <= 0:
            return 0.0
        return (self.get_speedup() / self.theoretical_speedup) * 100
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取策略指标
        
        Returns:
            Dict: 策略指标
        """
        return {
            'baseline_fps': self.baseline_fps,
            'strategy_fps': self.strategy_fps,
            'speedup': self.get_speedup(),
            'theoretical_speedup': self.theoretical_speedup,
            'parallel_efficiency': self.get_parallel_efficiency(),
            'base_metrics': self.collector.get_statistics()
        }
