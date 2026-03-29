#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
评测场景模块

提供三层评测场景：
1. 模型选型评测 - 对比不同模型性能
2. 策略验证评测 - 验证加速策略效果
3. 极限性能评测 - 追求极限吞吐量
"""

import os
import time
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

from config import Config
from evaluations.routes import REMOTE_SENSING_ROUTES, RouteType
from evaluations.tiers import InputTier, STANDARD_INPUT_TIERS
from reporting.models import ExecutionRecord
from utils.metrics import MetricsCollector, TimingRecord
from utils.monitor import ResourceMonitor, SimpleResourceMonitor
from utils.exceptions import BenchmarkError, InferenceError
from utils.logger import LoggerConfig, get_logger

logger = LoggerConfig.setup_logger('ascend_inference.scenarios', format_type='text')


@dataclass
class ModelInfo:
    """模型信息"""
    path: str = ""
    name: str = ""
    input_size: int = 0
    output_size: int = 0
    input_width: int = 0
    input_height: int = 0
    resolution: str = ""
class BenchmarkScenario(ABC):
    """评测场景基类"""
    
    name: str = "base"
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化评测场景
        
        Args:
            config: 场景配置
        """
        self.config = config or {}
        self._results: List[BenchmarkResult] = []
    
    @abstractmethod
    def run(self, models: List[str], images: List[str], **kwargs) -> List[BenchmarkResult]:
        """运行评测
        
        Args:
            models: 模型路径列表
            images: 图像路径列表
            **kwargs: 其他参数
            
        Returns:
            List[BenchmarkResult]: 评测结果列表
        """
        pass
    
    @abstractmethod
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """生成报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: 报告内容
        """
        pass
    
    def get_results(self) -> List[BenchmarkResult]:
        """获取评测结果
        
        Returns:
            List[BenchmarkResult]: 评测结果列表
        """
        return self._results


class ModelSelectionScenario(BenchmarkScenario):
    """模型选型评测场景
    
    特点：
    - 不启用任何优化策略
    - 全面细致的统计指标
    - 支持多模型对比
    - 输出模型信息（输入/输出大小等）
    - 完整延迟分布（P50/P95/P99）
    """
    
    name = "model_selection"
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化模型选型评测场景
        
        Args:
            config: 场景配置，支持：
                - iterations: 测试迭代次数
                - warmup: 预热次数
                - enable_monitoring: 启用资源监控
        """
        super().__init__(config)
        self.iterations = self.config.get('iterations', 100)
        self.warmup = self.config.get('warmup', 5)
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        self.device_id = self.config.get('device_id', 0)
        self.backend = self.config.get('backend', 'pil')
        self.input_tiers = [
            InputTier.from_value(input_tier).value
            for input_tier in self.config.get('input_tiers', STANDARD_INPUT_TIERS)
        ]

    def build_matrix(self, models: List[str], images: List[str]) -> List[Dict[str, str]]:
        """按标准输入分档展开评测矩阵。"""
        matrix: List[Dict[str, str]] = []
        for model_path in models:
            for image_path in images:
                for input_tier in self.input_tiers:
                    tier = InputTier.from_value(input_tier)
                    matrix.append(
                        {
                            "model_path": model_path,
                            "image_path": image_path,
                            "input_tier": tier.value,
                            "runtime_resolution": tier.runtime_resolution,
                        }
                    )
        return matrix
    
    def run(self, models: List[str], images: List[str], **kwargs) -> List[BenchmarkResult]:
        """运行模型选型评测
        
        Args:
            models: 模型路径列表
            images: 图像路径列表
            **kwargs: 其他参数
            
        Returns:
            List[BenchmarkResult]: 评测结果列表
        """
        self._results = []
        
        for entry in self.build_matrix(models, images):
            result = self._run_single_model(
                entry["model_path"],
                entry["image_path"],
                input_tier=entry["input_tier"],
                runtime_resolution=entry["runtime_resolution"],
            )
            if result:
                self._results.append(result)
        
        return self._results
    
    def _run_single_model(
        self,
        model_path: str,
        image_path: str,
        input_tier: Optional[str] = None,
        runtime_resolution: Optional[str] = None,
        route_type: Optional[str] = None,
        image_size_tier: Optional[str] = None,
    ) -> Optional[BenchmarkResult]:
        """运行单个模型的评测
        
        Args:
            model_path: 模型路径
            image_path: 图像路径
            
        Returns:
            BenchmarkResult: 评测结果
        """
        from src.inference import Inference
        
        config = Config(model_path=model_path, device_id=self.device_id, backend=self.backend)
        if input_tier:
            config.evaluation.input_tier = input_tier
        if route_type:
            config.evaluation.route_type = route_type
        if route_type == RouteType.LARGE_INPUT_ROUTE.value and runtime_resolution:
            config.resolution = runtime_resolution
        config.strategies.multithread.enabled = False
        config.strategies.batch.enabled = False
        config.strategies.pipeline.enabled = False
        
        inference = Inference(config)
        
        try:
            inference.init()
            
            model_info = self._collect_model_info(inference, model_path)
            
            collector = MetricsCollector(auto_warmup=False)
            
            for _ in range(self.warmup):
                inference.run_inference(image_path, config.backend)
            collector.finish_warmup()
            
            monitor = None
            if self.enable_monitoring:
                monitor = SimpleResourceMonitor()
            
            for _ in range(self.iterations):
                if monitor:
                    monitor.sample()
                
                record = TimingRecord()
                
                start = time.time()
                inference.preprocess(image_path, config.backend)
                record.preprocess_time = time.time() - start
                
                start = time.time()
                inference.execute()
                record.execute_time = time.time() - start
                
                start = time.time()
                inference.get_result()
                record.postprocess_time = time.time() - start
                
                record.calculate_total()
                collector.record(record)
            
            metrics = collector.get_statistics()
            
            resource_stats = {}
            if monitor:
                resource_stats = monitor.get_stats()
            
            return BenchmarkResult(
                scenario_name=self.name,
                model_info=model_info,
                metrics=metrics,
                route_type=route_type or "",
                strategies=[],
                config={
                    'iterations': self.iterations,
                    'warmup': self.warmup,
                    'image': image_path,
                    'input_tier': input_tier,
                    'route_type': route_type,
                    'image_size_tier': image_size_tier,
                    'runtime_resolution': config.resolution,
                    'input_tier_runtime_resolution': runtime_resolution,
                    'device_id': config.device_id,
                    'backend': config.backend,
                },
                resource_stats=resource_stats
            )
            
        except InferenceError as e:
            logger.error(f"评测失败: {model_path}, 错误: {e}")
            raise BenchmarkError(
                f"模型评测失败: {model_path}",
                error_code=3001,
                original_error=e,
                details={"model_path": model_path, "image_path": image_path}
            ) from e
        except Exception as e:
            logger.error(f"评测失败: {model_path}, 错误: {e}")
            raise BenchmarkError(
                f"模型评测异常: {model_path}",
                error_code=3002,
                original_error=e,
                details={"model_path": model_path, "image_path": image_path}
            ) from e
        finally:
            inference.destroy()
    
    def _collect_model_info(self, inference: Any, model_path: str) -> ModelInfo:
        """收集模型信息
        
        Args:
            inference: 推理实例
            model_path: 模型路径
            
        Returns:
            ModelInfo: 模型信息
        """
        return ModelInfo(
            path=model_path,
            name=os.path.basename(model_path),
            input_size=inference.input_size,
            output_size=inference.output_size,
            input_width=inference.input_width,
            input_height=inference.input_height,
            resolution=inference.resolution
        )
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """生成模型选型报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: 报告内容
        """
        lines = [
            "=" * 80,
            "模型选型评测报告",
            "=" * 80,
            ""
        ]
        
        for result in results:
            lines.extend([
                f"模型: {result.model_info.name}",
                f"  路径: {result.model_info.path}",
                f"  输入大小: {result.model_info.input_size / 1024:.2f} KB",
                f"  输出大小: {result.model_info.output_size / 1024:.2f} KB",
                f"  分辨率: {result.model_info.resolution}",
                ""
            ])
            
            m = result.metrics
            if m:
                lines.extend([
                "  时间统计:",
                f"    预处理:   avg={m['preprocess']['avg']:.2f} ms, "
                f"p50={m['preprocess']['p50']:.2f}, p95={m['preprocess']['p95']:.2f}",
                f"    推理执行: avg={m['execute']['avg']:.2f} ms, "
                f"p50={m['execute']['p50']:.2f}, p95={m['execute']['p95']:.2f}",
                f"    后处理:   avg={m['postprocess']['avg']:.2f} ms, "
                f"p50={m['postprocess']['p50']:.2f}, p95={m['postprocess']['p95']:.2f}",
                f"    总时间:   avg={m['total']['avg']:.2f} ms, "
                f"p50={m['total']['p50']:.2f}, p95={m['total']['p95']:.2f}, p99={m['total']['p99']:.2f}",
                "",
                "  时间占比:",
                f"    预处理:   {m['ratios']['preprocess']:.1f}%",
                f"    推理执行: {m['ratios']['execute']:.1f}%",
                f"    后处理:   {m['ratios']['postprocess']:.1f}%",
                "",
                "  性能指标:",
                f"    纯推理FPS: {m['fps']['pure']:.2f}",
                f"    端到端FPS: {m['fps']['e2e']:.2f}",
                ""
            ])
        
        lines.extend([
            "=" * 80,
            "模型对比表格",
            "=" * 80,
            f"{'模型':<30} {'推理时间(ms)':<15} {'纯推理FPS':<15} {'端到端FPS':<15}",
            "-" * 80
        ])
        
        for result in results:
            model_name = result.model_info.name
            exec_time = result.metrics.get('execute', {}).get('avg', 0)
            pure_fps = result.metrics.get('fps', {}).get('pure', 0)
            e2e_fps = result.metrics.get('fps', {}).get('e2e', 0)
            lines.append(f"{model_name:<30} {exec_time:<15.2f} {pure_fps:<15.2f} {e2e_fps:<15.2f}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


class RouteExperimentScenario(ModelSelectionScenario):
    """遥感双路线对照评测场景。"""

    name = "route_experiment"

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.routes = [
            RouteType.from_value(route).value
            for route in self.config.get('routes', REMOTE_SENSING_ROUTES)
        ]
        self.image_size_tiers = list(self.config.get('image_size_tiers', ['6K']))

    def build_route_matrix(self, models: List[str], images: List[str]) -> List[Dict[str, Optional[str]]]:
        matrix: List[Dict[str, Optional[str]]] = []
        for model_path in models:
            for image_path in images:
                for image_size_tier in self.image_size_tiers:
                    normalized_resolution = str(image_size_tier).lower()
                    for route_type in self.routes:
                        matrix.append(
                            {
                                "model_path": model_path,
                                "image_path": image_path,
                                "route_type": route_type,
                                "image_size_tier": image_size_tier,
                                "runtime_resolution": (
                                    normalized_resolution
                                    if route_type == RouteType.LARGE_INPUT_ROUTE.value
                                    else None
                                ),
                            }
                        )
        return matrix

    def run(self, models: List[str], images: List[str], **kwargs) -> List[BenchmarkResult]:
        self._results = []
        for entry in self.build_route_matrix(models, images):
            result = self._run_single_model(
                entry["model_path"],
                entry["image_path"],
                runtime_resolution=entry["runtime_resolution"],
                route_type=entry["route_type"],
                image_size_tier=entry["image_size_tier"],
            )
            if result:
                self._results.append(result)
        return self._results


@dataclass(init=False)
class BenchmarkResult:
    """统一执行结果。

    通过 execution_record 承载拆分后的指标，同时保留旧的 metrics 兼容视图。
    """

    scenario_name: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    execution_record: ExecutionRecord = field(default_factory=ExecutionRecord)
    strategies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    resource_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __init__(
        self,
        scenario_name: str = "",
        model_info: Optional[ModelInfo] = None,
        *,
        route_type: str = "",
        execution_record: Optional[ExecutionRecord] = None,
        model_metrics: Optional[Dict[str, Any]] = None,
        system_metrics: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        resource_stats: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        model_info_obj = model_info

        if execution_record is None:
            model_info_obj = deepcopy(model_info_obj) if model_info_obj is not None else ModelInfo()
            if model_metrics is not None or system_metrics is not None:
                execution_record = ExecutionRecord(
                    task_name=scenario_name,
                    route_type=route_type,
                    model_name=model_info_obj.name,
                    model_info=model_info_obj,
                    model_metrics=dict(model_metrics or {}),
                    system_metrics=dict(system_metrics or {}),
                    resource_stats=dict(resource_stats or {}),
                    config=dict(config or {}),
                    strategies=list(strategies or []),
                    timestamp=time.time() if timestamp is None else timestamp,
                )
            else:
                execution_record = ExecutionRecord.from_legacy_metrics(
                    metrics,
                    task_name=scenario_name,
                    route_type=route_type,
                    model_name=model_info_obj.name,
                    model_info=model_info_obj,
                    resource_stats=resource_stats,
                    config=config,
                    strategies=strategies,
                    timestamp=timestamp,
                )
        else:
            if model_info_obj is None:
                model_info_obj = execution_record.model_info or ModelInfo(name=execution_record.model_name)
            else:
                model_info_obj = deepcopy(model_info_obj)
                execution_record.model_info = model_info_obj
            if execution_record.model_info is None:
                execution_record.model_info = deepcopy(model_info_obj)
            if not execution_record.model_name:
                execution_record.model_name = getattr(execution_record.model_info, "name", model_info_obj.name)
            if route_type and not execution_record.route_type:
                execution_record.route_type = route_type

        self.execution_record = execution_record
        self.scenario_name = scenario_name or self.execution_record.task_name
        self.model_info = model_info_obj or self.execution_record.model_info or ModelInfo()
        self.route_type = route_type or self.execution_record.route_type
        self.strategies = list(strategies if strategies is not None else self.execution_record.strategies)
        self.config = dict(config if config is not None else self.execution_record.config)
        self.resource_stats = dict(resource_stats if resource_stats is not None else self.execution_record.resource_stats)
        self.timestamp = self.execution_record.timestamp if timestamp is None else timestamp

    @property
    def scenario_name(self) -> str:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.task_name if execution_record else self.__dict__.get("_scenario_name", "")

    @scenario_name.setter
    def scenario_name(self, value: str) -> None:
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.task_name = value
        object.__setattr__(self, "_scenario_name", value)

    @property
    def model_info(self) -> ModelInfo:
        execution_record = self.__dict__.get("execution_record")
        if execution_record and execution_record.model_info is not None:
            return execution_record.model_info
        return self.__dict__.get("_model_info", ModelInfo())

    @model_info.setter
    def model_info(self, value: Optional[ModelInfo]) -> None:
        model_info = deepcopy(value) if value is not None else ModelInfo()
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.model_info = model_info
            execution_record.model_name = getattr(model_info, "name", "")
        object.__setattr__(self, "_model_info", model_info)

    @property
    def route_type(self) -> str:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.route_type if execution_record else self.__dict__.get("_route_type", "")

    @route_type.setter
    def route_type(self, value: str) -> None:
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.route_type = value
        object.__setattr__(self, "_route_type", value)

    @property
    def model_metrics(self) -> Dict[str, Any]:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.model_metrics if execution_record else {}

    @property
    def system_metrics(self) -> Dict[str, Any]:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.system_metrics if execution_record else {}

    @property
    def metrics(self) -> Dict[str, Any]:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.to_legacy_metrics() if execution_record else {}

    @metrics.setter
    def metrics(self, value: Dict[str, Any]) -> None:
        value = dict(value or {})
        execution_record = self.__dict__.get("execution_record")
        if execution_record is None:
            object.__setattr__(self, "_legacy_metrics", value)
            return

        synced_record = ExecutionRecord.from_legacy_metrics(
            value,
            task_name=self.scenario_name,
            route_type=execution_record.route_type,
            model_name=self.model_info.name,
            model_info=self.model_info,
            resource_stats=self.resource_stats,
            config=self.config,
            strategies=self.strategies,
            timestamp=self.timestamp,
        )
        execution_record.model_metrics = synced_record.model_metrics
        execution_record.system_metrics = synced_record.system_metrics
        execution_record.model_info = synced_record.model_info
        execution_record.model_name = getattr(synced_record.model_info, "name", self.model_info.name)
        execution_record.resource_stats = synced_record.resource_stats
        execution_record.config = synced_record.config
        execution_record.strategies = synced_record.strategies
        execution_record.task_name = synced_record.task_name
        execution_record.timestamp = synced_record.timestamp

    @property
    def strategies(self) -> List[str]:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.strategies if execution_record else self.__dict__.get("_strategies", [])

    @strategies.setter
    def strategies(self, value: Optional[List[str]]) -> None:
        strategies = deepcopy(value or [])
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.strategies = strategies
        object.__setattr__(self, "_strategies", strategies)

    @property
    def config(self) -> Dict[str, Any]:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.config if execution_record else self.__dict__.get("_config", {})

    @config.setter
    def config(self, value: Optional[Dict[str, Any]]) -> None:
        config = deepcopy(value or {})
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.config = config
        object.__setattr__(self, "_config", config)

    @property
    def resource_stats(self) -> Dict[str, Any]:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.resource_stats if execution_record else self.__dict__.get("_resource_stats", {})

    @resource_stats.setter
    def resource_stats(self, value: Optional[Dict[str, Any]]) -> None:
        resource_stats = deepcopy(value or {})
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.resource_stats = resource_stats
        object.__setattr__(self, "_resource_stats", resource_stats)

    @property
    def timestamp(self) -> float:
        execution_record = self.__dict__.get("execution_record")
        return execution_record.timestamp if execution_record else self.__dict__.get("_timestamp", 0.0)

    @timestamp.setter
    def timestamp(self, value: float) -> None:
        execution_record = self.__dict__.get("execution_record")
        if execution_record is not None:
            execution_record.timestamp = value
        object.__setattr__(self, "_timestamp", value)


class StrategyValidationScenario(BenchmarkScenario):
    """策略验证评测场景
    
    特点：
    - 首先获取基准性能（无策略）
    - 分别测试每种策略
    - 计算加速比、并行效率
    - 支持策略对比
    """
    
    name = "strategy_validation"
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化策略验证评测场景
        
        Args:
            config: 场景配置，支持：
                - strategies: 要测试的策略列表
                - iterations: 测试迭代次数
                - warmup: 预热次数
        """
        super().__init__(config)
        self.strategies_to_test = self.config.get('strategies', [
            'multithread', 'batch', 'pipeline', 'memory_pool'
        ])
        self.iterations = self.config.get('iterations', 50)
        self.warmup = self.config.get('warmup', 3)
        self.device_id = self.config.get('device_id', 0)
        self.backend = self.config.get('backend', 'pil')
        self.routes = list(self.config.get('routes', []))
        self.image_size_tiers = list(self.config.get('image_size_tiers', []))
        if self.image_size_tiers and not self.routes:
            self.routes = list(REMOTE_SENSING_ROUTES)
        if (
            RouteType.LARGE_INPUT_ROUTE.value in self.routes
            and not self.image_size_tiers
        ):
            self.image_size_tiers = ['6K']
    
    def run(self, models: List[str], images: List[str], **kwargs) -> List[BenchmarkResult]:
        """运行策略验证评测
        
        Args:
            models: 模型路径列表
            images: 图像路径列表
            **kwargs: 其他参数
            
        Returns:
            List[BenchmarkResult]: 评测结果列表
        """
        self._results = []
        
        model_path = models[0] if models else ""
        image_path = images[0] if images else ""
        
        if not model_path or not image_path:
            return self._results

        route_types = self.routes or [None]
        image_size_tiers = self.image_size_tiers or [None]

        for image_size_tier in image_size_tiers:
            for route_type in route_types:
                baseline_result = self._run_baseline(
                    model_path,
                    image_path,
                    route_type=route_type,
                    image_size_tier=image_size_tier,
                )
                if baseline_result:
                    self._results.append(baseline_result)

                baseline_fps = baseline_result.metrics.get('fps', {}).get('pure', 0) if baseline_result else 0

                for strategy_name in self.strategies_to_test:
                    strategy_result = self._run_strategy(
                        strategy_name,
                        model_path,
                        image_path,
                        baseline_fps,
                        route_type=route_type,
                        image_size_tier=image_size_tier,
                    )
                    if strategy_result:
                        self._results.append(strategy_result)
        
        return self._results
    
    def _run_baseline(
        self,
        model_path: str,
        image_path: str,
        route_type: Optional[str] = None,
        image_size_tier: Optional[str] = None,
    ) -> Optional[BenchmarkResult]:
        """运行基准测试（无策略）
        
        Args:
            model_path: 模型路径
            image_path: 图像路径
            
        Returns:
            BenchmarkResult: 基准结果
        """
        scenario_cls = RouteExperimentScenario if route_type else ModelSelectionScenario
        scenario_config = {
            'iterations': self.iterations,
            'warmup': self.warmup,
            'enable_monitoring': False,
            'device_id': self.device_id,
            'backend': self.backend,
        }
        if route_type:
            scenario_config['routes'] = [route_type]
            scenario_config['image_size_tiers'] = [image_size_tier] if image_size_tier else ['6K']

        scenario = scenario_cls(scenario_config)
        
        results = scenario.run([model_path], [image_path])
        
        if results:
            result = results[0]
            result.scenario_name = "baseline"
            result.strategies = ["baseline"]
            return result
        
        return None
    
    def _run_strategy(
        self,
        strategy_name: str,
        model_path: str,
        image_path: str,
        baseline_fps: float,
        route_type: Optional[str] = None,
        image_size_tier: Optional[str] = None,
    ) -> Optional[BenchmarkResult]:
        """运行单个策略测试
        
        Args:
            strategy_name: 策略名称
            model_path: 模型路径
            image_path: 图像路径
            baseline_fps: 基准FPS
            
        Returns:
            BenchmarkResult: 策略结果
        """
        config = Config(model_path=model_path, device_id=self.device_id, backend=self.backend)
        if route_type:
            config.evaluation.route_type = route_type
        if route_type == RouteType.LARGE_INPUT_ROUTE.value and image_size_tier:
            config.resolution = str(image_size_tier).lower()
        
        if strategy_name == 'multithread':
            config.strategies.multithread.enabled = True
            config.strategies.multithread.num_threads = 4
            theoretical_speedup = 4.0
        elif strategy_name == 'batch':
            config.strategies.batch.enabled = True
            config.strategies.batch.batch_size = 4
            theoretical_speedup = 4.0
        elif strategy_name == 'pipeline':
            config.strategies.pipeline.enabled = True
            theoretical_speedup = 3.0
        elif strategy_name == 'memory_pool':
            config.strategies.memory_pool.enabled = True
            theoretical_speedup = 1.1
        else:
            return None
        
        try:
            if strategy_name == 'multithread':
                return self._run_multithread_strategy(
                    config,
                    image_path,
                    baseline_fps,
                    theoretical_speedup,
                    route_type=route_type,
                    image_size_tier=image_size_tier,
                )
            else:
                return self._run_simple_strategy(
                    config,
                    image_path,
                    strategy_name,
                    baseline_fps,
                    theoretical_speedup,
                    route_type=route_type,
                    image_size_tier=image_size_tier,
                )
        except InferenceError as e:
            logger.error(f"策略 {strategy_name} 测试失败: {e}")
            raise BenchmarkError(
                f"策略测试失败: {strategy_name}",
                error_code=3003,
                original_error=e,
                details={"strategy": strategy_name, "model_path": model_path}
            ) from e
        except Exception as e:
            logger.error(f"策略 {strategy_name} 测试失败: {e}")
            raise BenchmarkError(
                f"策略测试异常: {strategy_name}",
                error_code=3004,
                original_error=e,
                details={"strategy": strategy_name, "model_path": model_path}
            ) from e
    
    def _run_simple_strategy(
        self,
        config: Config,
        image_path: str,
        strategy_name: str,
        baseline_fps: float,
        theoretical_speedup: float,
        route_type: Optional[str] = None,
        image_size_tier: Optional[str] = None,
    ) -> Optional[BenchmarkResult]:
        """运行简单策略测试
        
        Args:
            config: 配置
            image_path: 图像路径
            strategy_name: 策略名称
            baseline_fps: 基准FPS
            theoretical_speedup: 理论加速比
            
        Returns:
            BenchmarkResult: 结果
        """
        from src.inference import Inference
        
        inference = Inference(config)
        
        try:
            inference.init()
            
            collector = MetricsCollector(auto_warmup=False)
            
            for _ in range(self.warmup):
                inference.run_inference(image_path, config.backend)
            collector.finish_warmup()
            
            for _ in range(self.iterations):
                record = TimingRecord()
                
                start = time.time()
                inference.preprocess(image_path, config.backend)
                record.preprocess_time = time.time() - start
                
                start = time.time()
                inference.execute()
                record.execute_time = time.time() - start
                
                start = time.time()
                inference.get_result()
                record.postprocess_time = time.time() - start
                
                record.calculate_total()
                collector.record(record)
            
            metrics = collector.get_statistics()
            strategy_fps = metrics.get('fps', {}).get('pure', 0)
            
            speedup = strategy_fps / baseline_fps if baseline_fps > 0 else 1.0
            parallel_efficiency = (speedup / theoretical_speedup) * 100 if theoretical_speedup > 0 else 0
            
            metrics['strategy'] = {
                'speedup': speedup,
                'parallel_efficiency': parallel_efficiency,
                'theoretical_speedup': theoretical_speedup,
                'baseline_fps': baseline_fps,
                'strategy_fps': strategy_fps
            }
            
            return BenchmarkResult(
                scenario_name=self.name,
                model_info=ModelInfo(name=os.path.basename(config.model_path)),
                metrics=metrics,
                strategies=[strategy_name],
                route_type=route_type or "",
                config={
                    'iterations': self.iterations,
                    'warmup': self.warmup,
                    'route_type': route_type,
                    'image_size_tier': image_size_tier,
                    'runtime_resolution': config.resolution,
                    'device_id': config.device_id,
                    'backend': config.backend,
                }
            )
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"简单策略测试异常: {strategy_name}, {e}")
            raise BenchmarkError(
                f"简单策略测试异常: {strategy_name}",
                error_code=3005,
                original_error=e,
                details={"strategy": strategy_name}
            ) from e
        finally:
            inference.destroy()
    
    def _run_multithread_strategy(
        self,
        config: Config,
        image_path: str,
        baseline_fps: float,
        theoretical_speedup: float,
        route_type: Optional[str] = None,
        image_size_tier: Optional[str] = None,
    ) -> Optional[BenchmarkResult]:
        """运行多线程策略测试
        
        Args:
            config: 配置
            image_path: 图像路径
            baseline_fps: 基准FPS
            theoretical_speedup: 理论加速比
            
        Returns:
            BenchmarkResult: 结果
        """
        from src.inference import MultithreadInference
        
        mt_inference = MultithreadInference(config)
        
        try:
            mt_inference.start()
            
            for _ in range(self.warmup):
                mt_inference.add_task(image_path, config.backend)
            mt_inference.wait_completion()
            mt_inference.get_results()
            
            start_time = time.time()
            for _ in range(self.iterations):
                mt_inference.add_task(image_path, config.backend)
            mt_inference.wait_completion()
            total_time = time.time() - start_time
            
            results = mt_inference.get_results()
            completed = len(results)
            
            strategy_fps = completed / total_time if total_time > 0 else 0
            speedup = strategy_fps / baseline_fps if baseline_fps > 0 else 1.0
            parallel_efficiency = (speedup / theoretical_speedup) * 100 if theoretical_speedup > 0 else 0
            
            metrics = {
                'fps': {'pure': strategy_fps, 'e2e': strategy_fps},
                'strategy': {
                    'speedup': speedup,
                    'parallel_efficiency': parallel_efficiency,
                    'theoretical_speedup': theoretical_speedup,
                    'baseline_fps': baseline_fps,
                    'strategy_fps': strategy_fps,
                    'throughput_fps': strategy_fps
                },
                'iterations': {'test': completed}
            }
            
            return BenchmarkResult(
                scenario_name=self.name,
                model_info=ModelInfo(name=os.path.basename(config.model_path)),
                metrics=metrics,
                strategies=['multithread'],
                route_type=route_type or "",
                config={
                    'iterations': self.iterations,
                    'warmup': self.warmup,
                    'num_threads': config.num_threads,
                    'route_type': route_type,
                    'image_size_tier': image_size_tier,
                    'runtime_resolution': config.resolution,
                    'device_id': config.device_id,
                    'backend': config.backend,
                }
            )
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"多线程策略测试异常: {e}")
            raise BenchmarkError(
                "多线程策略测试异常",
                error_code=3006,
                original_error=e,
                details={"strategy": "multithread"}
            ) from e
        finally:
            mt_inference.stop()
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """生成策略验证报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: 报告内容
        """
        lines = [
            "=" * 80,
            "策略验证评测报告",
            "=" * 80,
            ""
        ]
        
        baseline = None
        for result in results:
            if 'baseline' in result.strategies:
                baseline = result
                break
        
        if baseline:
            lines.extend([
                "基准性能（无策略）:",
                f"  纯推理FPS: {baseline.metrics.get('fps', {}).get('pure', 0):.2f}",
                ""
            ])
        
        lines.extend([
            "策略对比:",
            f"{'策略':<20} {'吞吐FPS':<15} {'加速比':<15} {'并行效率':<15}",
            "-" * 80
        ])
        
        for result in results:
            if 'baseline' in result.strategies:
                strategy_name = "baseline"
                throughput = result.metrics.get('fps', {}).get('pure', 0)
                speedup = 1.0
                efficiency = 100.0
            else:
                strategy_name = result.strategies[0] if result.strategies else "unknown"
                strategy_metrics = result.metrics.get('strategy', {})
                throughput = strategy_metrics.get('strategy_fps', result.metrics.get('fps', {}).get('pure', 0))
                speedup = strategy_metrics.get('speedup', 1.0)
                efficiency = strategy_metrics.get('parallel_efficiency', 0)
            
            lines.append(f"{strategy_name:<20} {throughput:<15.2f} {speedup:<15.2f}x {efficiency:<15.1f}%")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


class ExtremePerformanceScenario(BenchmarkScenario):
    """极限性能评测场景
    
    特点：
    - 根据配置组合多种策略
    - 追求极限吞吐量
    - 监控资源利用率
    """
    
    name = "extreme_performance"
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化极限性能评测场景
        
        Args:
            config: 场景配置，支持：
                - strategy_config: 策略配置
                - iterations: 测试迭代次数
                - warmup: 预热次数
                - duration_seconds: 测试时长（秒）
        """
        super().__init__(config)
        self.strategy_config = self.config.get('strategy_config', {})
        self.iterations = self.config.get('iterations', 100)
        self.warmup = self.config.get('warmup', 5)
        self.duration_seconds = self.config.get('duration_seconds', 10)
    
    def run(self, models: List[str], images: List[str], **kwargs) -> List[BenchmarkResult]:
        """运行极限性能评测
        
        Args:
            models: 模型路径列表
            images: 图像路径列表
            **kwargs: 其他参数
            
        Returns:
            List[BenchmarkResult]: 评测结果列表
        """
        self._results = []
        
        model_path = models[0] if models else ""
        if not model_path:
            return self._results
        
        result = self._run_extreme_test(model_path, images)
        if result:
            self._results.append(result)
        
        return self._results
    
    def _run_extreme_test(self, model_path: str, images: List[str]) -> Optional[BenchmarkResult]:
        """运行极限性能测试
        
        Args:
            model_path: 模型路径
            images: 图像路径列表
            
        Returns:
            BenchmarkResult: 结果
        """
        config = Config(model_path=model_path)
        
        if self.strategy_config:
            for key, value in self.strategy_config.items():
                if hasattr(config.strategies, key):
                    strategy_config = getattr(config.strategies, key)
                    if hasattr(strategy_config, 'enabled'):
                        strategy_config.enabled = value.get('enabled', False)
                    for k, v in value.items():
                        if hasattr(strategy_config, k):
                            setattr(strategy_config, k, v)
        
        enabled_strategies = config.get_enabled_strategies()
        
        if 'multithread' in enabled_strategies:
            return self._run_multithread_extreme(config, images, enabled_strategies)
        else:
            return self._run_simple_extreme(config, images, enabled_strategies)
    
    def _run_simple_extreme(self, config: Config, images: List[str], 
                            enabled_strategies: List[str]) -> Optional[BenchmarkResult]:
        """运行简单极限性能测试
        
        Args:
            config: 配置
            images: 图像列表
            enabled_strategies: 启用的策略列表
            
        Returns:
            BenchmarkResult: 结果
        """
        from src.inference import Inference
        
        inference = Inference(config)
        
        try:
            inference.init()
            
            image_path = images[0] if images else ""
            
            collector = MetricsCollector(auto_warmup=False)
            monitor = ResourceMonitor()
            monitor.start()
            
            for _ in range(self.warmup):
                inference.run_inference(image_path, config.backend)
            collector.finish_warmup()
            
            for _ in range(self.iterations):
                record = TimingRecord()
                
                start = time.time()
                inference.preprocess(image_path, config.backend)
                record.preprocess_time = time.time() - start
                
                start = time.time()
                inference.execute()
                record.execute_time = time.time() - start
                
                start = time.time()
                inference.get_result()
                record.postprocess_time = time.time() - start
                
                record.calculate_total()
                collector.record(record)
            
            monitor.stop()
            
            metrics = collector.get_statistics()
            resource_stats = monitor.get_stats()
            
            return BenchmarkResult(
                scenario_name=self.name,
                model_info=ModelInfo(name=os.path.basename(config.model_path)),
                metrics=metrics,
                strategies=enabled_strategies,
                config=config.strategies.to_dict(),
                resource_stats=resource_stats
            )
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"极限性能测试异常: {e}")
            raise BenchmarkError(
                "极限性能测试异常",
                error_code=3007,
                original_error=e,
                details={"strategies": enabled_strategies}
            ) from e
        finally:
            inference.destroy()
    
    def _run_multithread_extreme(self, config: Config, images: List[str],
                                  enabled_strategies: List[str]) -> Optional[BenchmarkResult]:
        """运行多线程极限性能测试
        
        Args:
            config: 配置
            images: 图像列表
            enabled_strategies: 启用的策略列表
            
        Returns:
            BenchmarkResult: 结果
        """
        from src.inference import MultithreadInference
        
        mt_inference = MultithreadInference(config)
        
        try:
            mt_inference.start()
            
            image_path = images[0] if images else ""
            
            monitor = ResourceMonitor()
            monitor.start()
            
            for _ in range(self.warmup):
                mt_inference.add_task(image_path, config.backend)
            mt_inference.wait_completion()
            mt_inference.get_results()
            
            start_time = time.time()
            total_tasks = 0
            
            while time.time() - start_time < self.duration_seconds:
                for _ in range(10):
                    mt_inference.add_task(image_path, config.backend)
                    total_tasks += 1
                mt_inference.wait_completion()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            results = mt_inference.get_results()
            completed = len(results)
            
            monitor.stop()
            
            throughput_fps = completed / total_time if total_time > 0 else 0
            
            metrics = {
                'fps': {'pure': throughput_fps, 'e2e': throughput_fps},
                'throughput_fps': throughput_fps,
                'total_tasks': total_tasks,
                'completed_tasks': completed,
                'test_duration': total_time,
                'iterations': {'test': completed}
            }
            
            resource_stats = monitor.get_stats()
            
            return BenchmarkResult(
                scenario_name=self.name,
                model_info=ModelInfo(name=os.path.basename(config.model_path)),
                metrics=metrics,
                strategies=enabled_strategies,
                config=config.strategies.to_dict(),
                resource_stats=resource_stats
            )
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"多线程极限性能测试异常: {e}")
            raise BenchmarkError(
                "多线程极限性能测试异常",
                error_code=3008,
                original_error=e,
                details={"strategies": enabled_strategies}
            ) from e
        finally:
            mt_inference.stop()
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        """生成极限性能报告
        
        Args:
            results: 评测结果列表
            
        Returns:
            str: 报告内容
        """
        lines = [
            "=" * 80,
            "极限性能评测报告",
            "=" * 80,
            ""
        ]
        
        for result in results:
            lines.extend([
                f"启用的策略: {', '.join(result.strategies)}",
                "",
                "配置:"
            ])
            
            for key, value in result.config.items():
                if isinstance(value, dict) and value.get('enabled'):
                    lines.append(f"  {key}: {value}")
            
            lines.append("")
            
            m = result.metrics
            if 'throughput_fps' in m:
                lines.extend([
                    "性能指标:",
                    f"  吞吐FPS: {m['throughput_fps']:.2f}",
                    f"  完成任务数: {m.get('completed_tasks', 0)}",
                    f"  测试时长: {m.get('test_duration', 0):.2f} 秒",
                    ""
                ])
            elif m.get('fps'):
                lines.extend([
                    "性能指标:",
                    f"  纯推理FPS: {m['fps'].get('pure', 0):.2f}",
                    f"  端到端FPS: {m['fps'].get('e2e', 0):.2f}",
                    ""
                ])
            
            if result.resource_stats:
                lines.extend([
                    "资源利用率:",
                    f"  CPU平均: {result.resource_stats.get('cpu', {}).get('avg', 0):.1f}%",
                    f"  内存使用: {result.resource_stats.get('memory', {}).get('current_mb', 0):.1f} MB",
                    ""
                ])
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
