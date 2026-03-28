#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测场景模块测试

测试 benchmark/scenarios.py 中的核心功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from benchmark.scenarios import (
    BenchmarkScenario,
    BenchmarkResult,
    ModelInfo,
    ModelSelectionScenario,
    StrategyValidationScenario,
    ExtremePerformanceScenario
)


class TestModelInfo:
    """ModelInfo 数据类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        info = ModelInfo()
        assert info.path == ""
        assert info.name == ""
        assert info.input_size == 0
        assert info.output_size == 0
        assert info.input_width == 0
        assert info.input_height == 0
        assert info.resolution == ""
    
    def test_custom_values(self):
        """测试自定义值"""
        info = ModelInfo(
            path="/path/to/model.om",
            name="yolov8n",
            input_size=640*640*3,
            output_size=8400,
            input_width=640,
            input_height=640,
            resolution="640x640"
        )
        
        assert info.path == "/path/to/model.om"
        assert info.name == "yolov8n"
        assert info.input_size == 640*640*3
        assert info.output_size == 8400
        assert info.input_width == 640
        assert info.input_height == 640
        assert info.resolution == "640x640"


class TestBenchmarkResult:
    """BenchmarkResult 数据类测试"""
    
    def test_default_values(self):
        """测试默认值"""
        result = BenchmarkResult()
        assert result.scenario_name == ""
        assert result.strategies == []
        assert result.metrics == {}
        assert result.config == {}
        assert result.resource_stats == {}
        assert result.timestamp > 0
    
    def test_custom_values(self):
        """测试自定义值"""
        result = BenchmarkResult(
            scenario_name="test_scenario",
            model_info=ModelInfo(name="test_model"),
            strategies=["strategy1", "strategy2"],
            metrics={'fps': {'pure': 100.0, 'e2e': 80.0}},
            config={'iterations': 100},
            resource_stats={'cpu': {'avg': 50.0}}
        )
        
        assert result.scenario_name == "test_scenario"
        assert result.model_info.name == "test_model"
        assert len(result.strategies) == 2
        assert result.metrics['fps']['pure'] == 100.0
        assert result.config['iterations'] == 100
        assert result.resource_stats['cpu']['avg'] == 50.0

    def test_split_metrics_preserve_legacy_metrics_view(self):
        result = BenchmarkResult(
            scenario_name="model_selection",
            model_info=ModelInfo(name="test_model"),
            model_metrics={
                'preprocess': {'avg': 10.0},
                'execute': {'avg': 12.0},
                'fps': {'pure': 83.3}
            },
            system_metrics={
                'fps': {'e2e': 55.5},
                'iterations': {'test': 100},
                'duration': {'test_time_ms': 1200.0}
            },
            resource_stats={'cpu': {'avg': 42.0}}
        )

        assert result.model_metrics['execute']['avg'] == 12.0
        assert result.system_metrics['fps']['e2e'] == 55.5
        assert result.metrics['fps']['pure'] == 83.3
        assert result.metrics['fps']['e2e'] == 55.5
        assert result.metrics['iterations']['test'] == 100

    def test_reassigning_fields_updates_execution_record(self):
        result = BenchmarkResult(
            scenario_name="initial",
            model_info=ModelInfo(name="test_model"),
            metrics={"fps": {"pure": 100.0}},
        )

        result.scenario_name = "updated"
        result.strategies = ["baseline"]

        assert result.scenario_name == "updated"
        assert result.execution_record.task_name == "updated"
        assert result.strategies == ["baseline"]
        assert result.execution_record.strategies == ["baseline"]

    def test_execution_record_mutations_are_visible_through_compat_views(self):
        result = BenchmarkResult(
            scenario_name="initial",
            model_info=ModelInfo(name="test_model", resolution="640x640"),
            metrics={"execute": {"avg": 12.0}, "fps": {"pure": 100.0, "e2e": 80.0}},
            config={"iterations": 10},
            resource_stats={"cpu": {"avg": 20.0}},
            timestamp=111.0,
        )

        result.execution_record.task_name = "updated"
        result.execution_record.model_info.name = "updated_model"
        result.execution_record.config["iterations"] = 20
        result.execution_record.resource_stats["cpu"]["avg"] = 30.0
        result.execution_record.timestamp = 222.0
        result.execution_record.model_metrics["execute"]["avg"] = 15.0

        assert result.scenario_name == "updated"
        assert result.model_info.name == "updated_model"
        assert result.config["iterations"] == 20
        assert result.resource_stats["cpu"]["avg"] == 30.0
        assert result.timestamp == 222.0
        assert result.metrics["execute"]["avg"] == 15.0


class TestModelSelectionScenario:
    """ModelSelectionScenario 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        scenario = ModelSelectionScenario()
        assert scenario.name == "model_selection"
        assert scenario.iterations == 100
        assert scenario.warmup == 5
        assert scenario.enable_monitoring is True
    
    def test_init_custom(self):
        """测试自定义初始化"""
        scenario = ModelSelectionScenario({
            'iterations': 50,
            'warmup': 3,
            'enable_monitoring': False
        })
        assert scenario.iterations == 50
        assert scenario.warmup == 3
        assert scenario.enable_monitoring is False
    
    def test_generate_report_empty(self):
        """测试空结果报告生成"""
        scenario = ModelSelectionScenario()
        report = scenario.generate_report([])
        
        assert "模型选型评测报告" in report
    
    def test_generate_report_with_results(self):
        """测试有结果报告生成"""
        scenario = ModelSelectionScenario()
        
        results = [
            BenchmarkResult(
                scenario_name="model_selection",
                model_info=ModelInfo(name="yolov8n.om", resolution="640x640"),
                metrics={
                    'preprocess': {'avg': 10.0, 'p50': 9.5, 'p95': 12.0},
                    'execute': {'avg': 20.0, 'p50': 19.0, 'p95': 22.0},
                    'postprocess': {'avg': 5.0, 'p50': 4.5, 'p95': 6.0},
                    'total': {'avg': 35.0, 'p50': 33.0, 'p95': 40.0, 'p99': 45.0},
                    'fps': {'pure': 50.0, 'e2e': 28.5},
                    'ratios': {'preprocess': 28.5, 'execute': 57.1, 'postprocess': 14.3}
                }
            )
        ]
        
        report = scenario.generate_report(results)
        
        assert "模型选型评测报告" in report
        assert "yolov8n.om" in report
        assert "模型对比表格" in report
    
    def test_get_results(self):
        """测试获取结果"""
        scenario = ModelSelectionScenario()
        assert scenario.get_results() == []
        
        scenario._results = [BenchmarkResult(scenario_name="test")]
        assert len(scenario.get_results()) == 1


class TestStrategyValidationScenario:
    """StrategyValidationScenario 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        scenario = StrategyValidationScenario()
        assert scenario.name == "strategy_validation"
        assert 'multithread' in scenario.strategies_to_test
        assert 'batch' in scenario.strategies_to_test
        assert 'pipeline' in scenario.strategies_to_test
    
    def test_init_custom(self):
        """测试自定义初始化"""
        scenario = StrategyValidationScenario({
            'strategies': ['multithread', 'batch'],
            'iterations': 30,
            'warmup': 2
        })
        assert scenario.strategies_to_test == ['multithread', 'batch']
        assert scenario.iterations == 30
        assert scenario.warmup == 2
    
    def test_generate_report(self):
        """测试生成报告"""
        scenario = StrategyValidationScenario()
        
        results = [
            BenchmarkResult(
                scenario_name="baseline",
                strategies=["baseline"],
                metrics={'fps': {'pure': 100.0}}
            ),
            BenchmarkResult(
                scenario_name="strategy_validation",
                strategies=["multithread"],
                metrics={
                    'fps': {'pure': 400.0},
                    'strategy': {'speedup': 4.0, 'parallel_efficiency': 100.0}
                }
            )
        ]
        
        report = scenario.generate_report(results)
        
        assert "策略验证评测报告" in report
        assert "基准性能" in report
        assert "策略对比" in report


    def test_run_baseline_keeps_execution_record_in_sync(self):
        scenario = StrategyValidationScenario()
        baseline = BenchmarkResult(
            scenario_name="model_selection",
            model_info=ModelInfo(name="baseline_model"),
            metrics={"fps": {"pure": 100.0, "e2e": 80.0}},
            strategies=[],
        )

        with patch.object(ModelSelectionScenario, "run", return_value=[baseline]):
            result = scenario._run_baseline("model.om", "image.jpg")

        assert result is baseline
        assert result.scenario_name == "baseline"
        assert result.execution_record.task_name == "baseline"
        assert result.strategies == ["baseline"]
        assert result.execution_record.strategies == ["baseline"]


class TestExtremePerformanceScenario:
    """ExtremePerformanceScenario 类测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        scenario = ExtremePerformanceScenario()
        assert scenario.name == "extreme_performance"
        assert scenario.iterations == 100
        assert scenario.warmup == 5
        assert scenario.duration_seconds == 10
    
    def test_init_custom(self):
        """测试自定义初始化"""
        scenario = ExtremePerformanceScenario({
            'iterations': 200,
            'warmup': 10,
            'duration_seconds': 30,
            'strategy_config': {'multithread': {'enabled': True}}
        })
        assert scenario.iterations == 200
        assert scenario.warmup == 10
        assert scenario.duration_seconds == 30
        assert 'multithread' in scenario.strategy_config
    
    def test_generate_report(self):
        """测试生成报告"""
        scenario = ExtremePerformanceScenario()
        
        results = [
            BenchmarkResult(
                scenario_name="extreme_performance",
                strategies=["multithread", "memory_pool"],
                metrics={
                    'throughput_fps': 500.0,
                    'completed_tasks': 1000,
                    'test_duration': 10.0,
                    'fps': {'pure': 500.0, 'e2e': 500.0}
                },
                resource_stats={
                    'cpu': {'avg': 80.0, 'max': 95.0},
                    'memory': {'current_mb': 1024.0, 'total_mb': 8192.0}
                },
                config={'multithread': {'enabled': True, 'num_threads': 4}}
            )
        ]
        
        report = scenario.generate_report(results)
        
        assert "极限性能评测报告" in report
        assert "启用的策略" in report
        assert "性能指标" in report


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
