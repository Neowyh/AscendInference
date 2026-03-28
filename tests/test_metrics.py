#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标收集器模块测试

测试 utils/metrics.py 中的核心功能
"""

import pytest
import time

from utils.metrics import (
    TimingRecord,
    MetricsCollector,
    StrategyMetrics,
    StageStatistics,
    _calc_stats
)


class TestTimingRecord:
    """TimingRecord 数据类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        record = TimingRecord()
        assert record.preprocess_time == 0.0
        assert record.execute_time == 0.0
        assert record.postprocess_time == 0.0
        assert record.queue_wait_time == 0.0
        assert record.total_time == 0.0
        assert record.timestamp > 0
    
    def test_custom_values(self):
        """测试自定义值"""
        record = TimingRecord(
            preprocess_time=0.01,
            execute_time=0.02,
            postprocess_time=0.005,
            queue_wait_time=0.001,
            total_time=0.036
        )
        assert record.preprocess_time == 0.01
        assert record.execute_time == 0.02
        assert record.postprocess_time == 0.005
        assert record.queue_wait_time == 0.001
        assert record.total_time == 0.036
    
    def test_calculate_total(self):
        """测试计算总时间"""
        record = TimingRecord(
            preprocess_time=0.01,
            execute_time=0.02,
            postprocess_time=0.005,
            queue_wait_time=0.001
        )
        total = record.calculate_total()
        assert total == pytest.approx(0.036, rel=1e-6)
        assert record.total_time == pytest.approx(0.036, rel=1e-6)


class TestStageStatistics:
    """StageStatistics 数据类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        stats = StageStatistics()
        assert stats.avg == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.std == 0.0
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0
        assert stats.sum == 0.0
        assert stats.count == 0


class TestCalcStats:
    """_calc_stats 函数测试"""
    
    def test_empty_values(self):
        """测试空列表"""
        stats = _calc_stats([])
        assert stats.avg == 0.0
        assert stats.count == 0
    
    def test_single_value(self):
        """测试单个值"""
        stats = _calc_stats([0.01])
        assert stats.avg == pytest.approx(10.0, rel=1e-6)
        assert stats.min == pytest.approx(10.0, rel=1e-6)
        assert stats.max == pytest.approx(10.0, rel=1e-6)
        assert stats.count == 1
    
    def test_multiple_values(self):
        """测试多个值"""
        values = [0.01, 0.02, 0.03, 0.04, 0.05]
        stats = _calc_stats(values)
        
        assert stats.avg == pytest.approx(30.0, rel=1e-6)
        assert stats.min == pytest.approx(10.0, rel=1e-6)
        assert stats.max == pytest.approx(50.0, rel=1e-6)
        assert stats.count == 5


class TestMetricsCollector:
    """MetricsCollector 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        collector = MetricsCollector()
        assert collector.is_warmup is True
        assert collector.auto_warmup is True
        assert collector.warmup_iterations == 5
        assert len(collector.records) == 0
        assert len(collector.warmup_records) == 0
    
    def test_init_custom(self):
        """测试自定义初始化"""
        collector = MetricsCollector(auto_warmup=False, warmup_iterations=10)
        assert collector.auto_warmup is False
        assert collector.warmup_iterations == 10
    
    def test_record_warmup(self):
        """测试预热阶段记录"""
        collector = MetricsCollector(auto_warmup=False, warmup_iterations=3)
        
        record = TimingRecord(execute_time=0.01)
        collector.record(record)
        
        assert len(collector.warmup_records) == 1
        assert len(collector.records) == 0
        assert collector.is_warmup is True
    
    def test_finish_warmup(self):
        """测试结束预热"""
        collector = MetricsCollector(auto_warmup=False)
        
        collector.record(TimingRecord(execute_time=0.01))
        collector.finish_warmup()
        
        assert collector.is_warmup is False
        
        collector.record(TimingRecord(execute_time=0.02))
        assert len(collector.warmup_records) == 1
        assert len(collector.records) == 1
    
    def test_auto_warmup(self):
        """测试自动预热结束"""
        collector = MetricsCollector(auto_warmup=True, warmup_iterations=3)
        
        for _ in range(3):
            collector.record(TimingRecord(execute_time=0.01))
        
        assert collector.is_warmup is False
        assert len(collector.warmup_records) == 3
        
        collector.record(TimingRecord(execute_time=0.02))
        assert len(collector.records) == 1
    
    def test_start_iteration(self):
        """测试开始迭代"""
        collector = MetricsCollector()
        
        record = collector.start_iteration()
        assert isinstance(record, TimingRecord)
        assert collector._current_record is record
    
    def test_record_methods(self):
        """测试分阶段记录方法"""
        collector = MetricsCollector()
        collector.start_iteration()
        
        collector.record_preprocess(0.01)
        collector.record_execute(0.02)
        collector.record_postprocess(0.005)
        collector.record_queue_wait(0.001)
        
        assert collector._current_record.preprocess_time == 0.01
        assert collector._current_record.execute_time == 0.02
        assert collector._current_record.postprocess_time == 0.005
        assert collector._current_record.queue_wait_time == 0.001
    
    def test_finish_iteration(self):
        """测试结束迭代"""
        collector = MetricsCollector(auto_warmup=False)
        collector.finish_warmup()
        
        collector.start_iteration()
        collector.record_execute(0.02)
        collector.finish_iteration()
        
        assert collector._current_record is None
        assert len(collector.records) == 1
    
    def test_get_statistics_empty(self):
        """测试空统计结果"""
        collector = MetricsCollector()
        stats = collector.get_statistics()
        
        assert stats['iterations']['test'] == 0
        assert stats['fps']['pure'] == 0.0
        assert stats['fps']['e2e'] == 0.0
    
    def test_get_statistics(self):
        """测试统计结果"""
        collector = MetricsCollector(auto_warmup=False)
        collector.finish_warmup()
        
        for i in range(10):
            record = TimingRecord(
                preprocess_time=0.01,
                execute_time=0.02 + i * 0.001,
                postprocess_time=0.005,
                total_time=0.035 + i * 0.001
            )
            collector.record(record)
        
        stats = collector.get_statistics()
        
        assert stats['iterations']['test'] == 10
        assert stats['fps']['pure'] > 0
        assert stats['fps']['e2e'] > 0
        assert 'preprocess' in stats
        assert 'execute' in stats
        assert 'postprocess' in stats
        assert 'total' in stats
        assert 'ratios' in stats
    
    def test_reset(self):
        """测试重置"""
        collector = MetricsCollector(auto_warmup=False)
        collector.finish_warmup()
        
        collector.record(TimingRecord(execute_time=0.01))
        collector.reset()
        
        assert len(collector.records) == 0
        assert len(collector.warmup_records) == 0
        assert collector.is_warmup is True
    
    def test_get_summary(self):
        """测试统计摘要"""
        collector = MetricsCollector(auto_warmup=False)
        collector.finish_warmup()
        
        for _ in range(5):
            collector.record(TimingRecord(
                preprocess_time=0.01,
                execute_time=0.02,
                postprocess_time=0.005,
                total_time=0.035
            ))
        
        summary = collector.get_summary()
        
        assert "性能统计摘要" in summary
        assert "迭代次数" in summary
        assert "时间统计" in summary


class TestStrategyMetrics:
    """StrategyMetrics 类测试"""
    
    def test_init(self):
        """测试初始化"""
        metrics = StrategyMetrics()
        assert metrics.baseline_fps == 0.0
        assert metrics.strategy_fps == 0.0
        assert metrics.theoretical_speedup == 1.0
    
    def test_set_baseline(self):
        """测试设置基准FPS"""
        metrics = StrategyMetrics()
        metrics.set_baseline(100.0)
        assert metrics.baseline_fps == 100.0
    
    def test_set_strategy_fps(self):
        """测试设置策略FPS"""
        metrics = StrategyMetrics()
        metrics.set_strategy_fps(400.0)
        assert metrics.strategy_fps == 400.0
    
    def test_get_speedup(self):
        """测试计算加速比"""
        metrics = StrategyMetrics()
        metrics.set_baseline(100.0)
        metrics.set_strategy_fps(400.0)
        
        assert metrics.get_speedup() == 4.0
    
    def test_get_speedup_zero_baseline(self):
        """测试基准为0时的加速比"""
        metrics = StrategyMetrics()
        metrics.set_strategy_fps(400.0)
        
        assert metrics.get_speedup() == 1.0
    
    def test_get_parallel_efficiency(self):
        """测试计算并行效率"""
        metrics = StrategyMetrics()
        metrics.set_baseline(100.0)
        metrics.set_strategy_fps(400.0)
        metrics.set_theoretical_speedup(4.0)
        
        assert metrics.get_parallel_efficiency() == 100.0
    
    def test_get_parallel_efficiency_partial(self):
        """测试部分并行效率"""
        metrics = StrategyMetrics()
        metrics.set_baseline(100.0)
        metrics.set_strategy_fps(300.0)
        metrics.set_theoretical_speedup(4.0)
        
        assert metrics.get_parallel_efficiency() == 75.0
    
    def test_get_metrics(self):
        """测试获取策略指标"""
        metrics = StrategyMetrics()
        metrics.set_baseline(100.0)
        metrics.set_strategy_fps(400.0)
        metrics.set_theoretical_speedup(4.0)
        
        result = metrics.get_metrics()
        
        assert result['baseline_fps'] == 100.0
        assert result['strategy_fps'] == 400.0
        assert result['speedup'] == 4.0
        assert result['theoretical_speedup'] == 4.0
        assert result['parallel_efficiency'] == 100.0


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
