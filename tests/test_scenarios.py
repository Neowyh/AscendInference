#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测场景模块测试

测试 benchmark/scenarios.py 中的核心功能
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import inspect

from evaluations.tiers import InputTier
from benchmark.scenarios import (
    BenchmarkScenario,
    BenchmarkResult,
    ModelInfo,
    ModelSelectionScenario,
    RouteExperimentScenario,
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

    def test_constructor_and_setter_snapshot_nested_inputs(self):
        metrics = {
            "execute": {"avg": 12.0},
            "fps": {"pure": 100.0, "e2e": 80.0},
        }
        model_info = ModelInfo(name="test_model", resolution="640x640")
        config = {"nested": {"enabled": True}}
        resource_stats = {"cpu": {"avg": 20.0}}
        strategies = ["baseline", "batch"]

        result = BenchmarkResult(
            scenario_name="initial",
            model_info=model_info,
            metrics=metrics,
            strategies=strategies,
            config=config,
            resource_stats=resource_stats,
        )

        model_info.name = "mutated_model"
        metrics["execute"]["avg"] = 99.0
        config["nested"]["enabled"] = False
        resource_stats["cpu"]["avg"] = 42.0
        strategies.append("pipeline")

        assert result.model_info.name == "test_model"
        assert result.execution_record.model_name == "test_model"
        assert result.metrics["execute"]["avg"] == 12.0
        assert result.config["nested"]["enabled"] is True
        assert result.resource_stats["cpu"]["avg"] == 20.0
        assert result.strategies == ["baseline", "batch"]

        setter_metrics = {
            "execute": {"avg": 15.0},
            "fps": {"pure": 88.0, "e2e": 77.0},
        }
        result.metrics = setter_metrics
        setter_metrics["execute"]["avg"] = 123.0

        assert result.metrics["execute"]["avg"] == 15.0

    def test_module_exposes_single_benchmark_result_definition(self):
        import benchmark.scenarios as scenarios_module

        source = inspect.getsource(scenarios_module)
        assert source.count("class BenchmarkResult") == 1

    def test_route_type_is_preserved_on_execution_record(self):
        result = BenchmarkResult(
            scenario_name="route_experiment",
            model_info=ModelInfo(name="test_model"),
            metrics={"fps": {"pure": 100.0}},
            route_type="tiled_route",
        )

        assert result.execution_record.route_type == "tiled_route"


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

    def test_model_selection_scenario_expands_across_standard_input_tiers(self):
        scenario = ModelSelectionScenario()

        matrix = scenario.build_matrix(["model.om"], ["image.jpg"])

        assert len(matrix) == 3
        assert [item["input_tier"] for item in matrix] == ["720p", "1080p", "4K"]

    def test_model_selection_scenario_build_matrix_maps_runtime_resolution(self):
        scenario = ModelSelectionScenario({"input_tiers": ["720p", "4K"]})

        matrix = scenario.build_matrix(["model.om"], ["image.jpg"])

        assert matrix == [
            {
                "model_path": "model.om",
                "image_path": "image.jpg",
                "input_tier": "720p",
                "runtime_resolution": InputTier.TIER_720P.runtime_resolution,
            },
            {
                "model_path": "model.om",
                "image_path": "image.jpg",
                "input_tier": "4K",
                "runtime_resolution": InputTier.TIER_4K.runtime_resolution,
            },
        ]

    def test_model_selection_scenario_run_calls_single_model_for_each_input_tier(self):
        scenario = ModelSelectionScenario({"input_tiers": ["720p", "1080p"]})
        observed_calls = []

        def fake_run_single_model(model_path, image_path, input_tier=None, runtime_resolution=None):
            observed_calls.append(
                {
                    "model_path": model_path,
                    "image_path": image_path,
                    "input_tier": input_tier,
                    "runtime_resolution": runtime_resolution,
                }
            )
            return BenchmarkResult(
                scenario_name=scenario.name,
                model_info=ModelInfo(name=model_path),
                config={
                    "input_tier": input_tier,
                    "runtime_resolution": runtime_resolution,
                },
            )

        with patch.object(scenario, "_run_single_model", side_effect=fake_run_single_model):
            results = scenario.run(["model.om"], ["image.jpg"])

        assert observed_calls == [
            {
                "model_path": "model.om",
                "image_path": "image.jpg",
                "input_tier": "720p",
                "runtime_resolution": InputTier.TIER_720P.runtime_resolution,
            },
            {
                "model_path": "model.om",
                "image_path": "image.jpg",
                "input_tier": "1080p",
                "runtime_resolution": InputTier.TIER_1080P.runtime_resolution,
            },
        ]
        assert [result.config["input_tier"] for result in results] == ["720p", "1080p"]

    def test_model_selection_scenario_does_not_override_model_resolution_for_tier_metadata(self, monkeypatch):
        scenario = ModelSelectionScenario()
        captured = {}

        class FakeInference:
            def __init__(self, config):
                captured["resolution"] = config.resolution
                captured["input_tier"] = config.evaluation.input_tier
                self.input_size = 640 * 640 * 3
                self.output_size = 8400
                self.input_width = 640
                self.input_height = 640
                self.resolution = config.resolution

            def init(self):
                return None

            def run_inference(self, image_path, backend):
                return None

            def preprocess(self, image_path, backend):
                return None

            def execute(self):
                return None

            def get_result(self):
                return None

            def destroy(self):
                return None

        monkeypatch.setattr("src.inference.Inference", FakeInference)
        monkeypatch.setattr("benchmark.scenarios.MetricsCollector", Mock(return_value=Mock(
            finish_warmup=Mock(),
            record=Mock(),
            get_statistics=Mock(return_value={"fps": {"pure": 1.0, "e2e": 1.0}}),
        )))
        monkeypatch.setattr("benchmark.scenarios.SimpleResourceMonitor", Mock(return_value=Mock(
            sample=Mock(),
            get_stats=Mock(return_value={}),
        )))

        result = scenario._run_single_model(
            "model.om",
            "image.jpg",
            input_tier="1080p",
            runtime_resolution=InputTier.TIER_1080P.runtime_resolution,
        )

        assert captured["resolution"] == "640x640"
        assert captured["input_tier"] == "1080p"
        assert result.config["runtime_resolution"] == "640x640"
        assert result.config["input_tier_runtime_resolution"] == InputTier.TIER_1080P.runtime_resolution

    def test_model_selection_scenario_uses_configured_device_and_backend(self, monkeypatch):
        scenario = ModelSelectionScenario({"device_id": 3, "backend": "opencv"})
        captured = {}

        class FakeInference:
            def __init__(self, config):
                captured["device_id"] = config.device_id
                captured["backend"] = config.backend
                self.input_size = 640 * 640 * 3
                self.output_size = 8400
                self.input_width = 640
                self.input_height = 640
                self.resolution = config.resolution

            def init(self):
                return None

            def run_inference(self, image_path, backend):
                return None

            def preprocess(self, image_path, backend):
                return None

            def execute(self):
                return None

            def get_result(self):
                return None

            def destroy(self):
                return None

        monkeypatch.setattr("src.inference.Inference", FakeInference)
        monkeypatch.setattr("benchmark.scenarios.MetricsCollector", Mock(return_value=Mock(
            finish_warmup=Mock(),
            record=Mock(),
            get_statistics=Mock(return_value={"fps": {"pure": 1.0, "e2e": 1.0}}),
        )))
        monkeypatch.setattr("benchmark.scenarios.SimpleResourceMonitor", Mock(return_value=Mock(
            sample=Mock(),
            get_stats=Mock(return_value={}),
        )))

        scenario._run_single_model("model.om", "image.jpg")

        assert captured["device_id"] == 3
        assert captured["backend"] == "opencv"


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

    def test_strategy_validation_scenario_run_expands_across_routes(self):
        scenario = StrategyValidationScenario(
            {
                "strategies": ["multithread"],
                "routes": ["tiled_route", "large_input_route"],
                "image_size_tiers": ["6K"],
            }
        )
        baseline_calls = []
        strategy_calls = []

        def fake_baseline(model_path, image_path, route_type=None, image_size_tier=None):
            baseline_calls.append(
                {
                    "model_path": model_path,
                    "image_path": image_path,
                    "route_type": route_type,
                    "image_size_tier": image_size_tier,
                }
            )
            return BenchmarkResult(
                scenario_name="baseline",
                model_info=ModelInfo(name=model_path),
                metrics={"fps": {"pure": 100.0}},
                route_type=route_type,
            )

        def fake_strategy(strategy_name, model_path, image_path, baseline_fps, route_type=None, image_size_tier=None):
            strategy_calls.append(
                {
                    "strategy_name": strategy_name,
                    "model_path": model_path,
                    "image_path": image_path,
                    "baseline_fps": baseline_fps,
                    "route_type": route_type,
                    "image_size_tier": image_size_tier,
                }
            )
            return BenchmarkResult(
                scenario_name="strategy_validation",
                model_info=ModelInfo(name=model_path),
                metrics={"fps": {"pure": baseline_fps * 2}},
                strategies=[strategy_name],
                route_type=route_type,
            )

        with patch.object(scenario, "_run_baseline", side_effect=fake_baseline), patch.object(
            scenario, "_run_strategy", side_effect=fake_strategy
        ):
            results = scenario.run(["model.om"], ["image.jpg"])

        assert baseline_calls == [
            {
                "model_path": "model.om",
                "image_path": "image.jpg",
                "route_type": "tiled_route",
                "image_size_tier": "6K",
            },
            {
                "model_path": "model.om",
                "image_path": "image.jpg",
                "route_type": "large_input_route",
                "image_size_tier": "6K",
            },
        ]
        assert [item["route_type"] for item in strategy_calls] == ["tiled_route", "large_input_route"]
        assert [result.execution_record.route_type for result in results] == [
            "tiled_route",
            "tiled_route",
            "large_input_route",
            "large_input_route",
        ]


class TestRouteExperimentScenario:
    def test_remote_sensing_route_matrix_includes_tiled_and_large_input_routes(self):
        scenario = RouteExperimentScenario({"image_size_tiers": ["6K"]})

        matrix = scenario.build_route_matrix(["small.om", "6k.om"], ["image_6k.jpg"])

        route_types = {item["route_type"] for item in matrix}
        assert route_types == {"tiled_route", "large_input_route"}

    def test_route_experiment_scenario_run_expands_across_routes(self):
        scenario = RouteExperimentScenario({"image_size_tiers": ["6K"]})
        observed_calls = []

        def fake_run_single_model(
            model_path,
            image_path,
            input_tier=None,
            runtime_resolution=None,
            route_type=None,
            image_size_tier=None,
        ):
            observed_calls.append(
                {
                    "model_path": model_path,
                    "image_path": image_path,
                    "route_type": route_type,
                    "runtime_resolution": runtime_resolution,
                    "image_size_tier": image_size_tier,
                }
            )
            return BenchmarkResult(
                scenario_name=scenario.name,
                model_info=ModelInfo(name=model_path),
                config={
                    "route_type": route_type,
                    "image_size_tier": image_size_tier,
                    "runtime_resolution": runtime_resolution,
                },
            )

        with patch.object(scenario, "_run_single_model", side_effect=fake_run_single_model):
            results = scenario.run(["small.om"], ["image_6k.jpg"])

        assert observed_calls == [
            {
                "model_path": "small.om",
                "image_path": "image_6k.jpg",
                "route_type": "tiled_route",
                "runtime_resolution": None,
                "image_size_tier": "6K",
            },
            {
                "model_path": "small.om",
                "image_path": "image_6k.jpg",
                "route_type": "large_input_route",
                "runtime_resolution": "6k",
                "image_size_tier": "6K",
            },
        ]
        assert [result.config["route_type"] for result in results] == ["tiled_route", "large_input_route"]


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
