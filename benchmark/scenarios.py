#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测场景模块

提供三层评测场景：
1. 模型选型评测 - 对比不同模型性能
2. 策略验证评测 - 验证加速策略效果
3. 极限性能评测 - 追求极限吞吐量
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

from config import Config
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


@dataclass
class BenchmarkResult:
    """评测结果"""
    scenario_name: str = ""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    metrics: Dict[str, Any] = field(default_factory=dict)
    strategies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    resource_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


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
        
        for model_path in models:
            for image_path in images:
                result = self._run_single_model(model_path, image_path)
                if result:
                    self._results.append(result)
        
        return self._results
    
    def _run_single_model(self, model_path: str, image_path: str) -> Optional[BenchmarkResult]:
        """运行单个模型的评测
        
        Args:
            model_path: 模型路径
            image_path: 图像路径
            
        Returns:
            BenchmarkResult: 评测结果
        """
        from src.inference import Inference
        
        config = Config(model_path=model_path)
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
                strategies=[],
                config={
                    'iterations': self.iterations,
                    'warmup': self.warmup,
                    'image': image_path
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
        execution_record: Optional[ExecutionRecord] = None,
        model_metrics: Optional[Dict[str, Any]] = None,
        system_metrics: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        strategies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        resource_stats: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self.scenario_name = scenario_name
        self.model_info = model_info or ModelInfo()
        self.strategies = list(strategies or [])
        self.config = dict(config or {})
        self.resource_stats = dict(resource_stats or {})
        self.timestamp = time.time() if timestamp is None else timestamp

        if execution_record is None:
            if model_metrics is not None or system_metrics is not None:
                execution_record = ExecutionRecord(
                    task_name=scenario_name,
                    route_type="",
                    model_name=self.model_info.name,
                    model_metrics=dict(model_metrics or {}),
                    system_metrics=dict(system_metrics or {}),
                    resource_stats=dict(self.resource_stats),
                    config=dict(self.config),
                    strategies=list(self.strategies),
                    timestamp=self.timestamp,
                )
            else:
                execution_record = ExecutionRecord.from_legacy_metrics(
                    metrics,
                    task_name=scenario_name,
                    route_type="",
                    model_name=self.model_info.name,
                    resource_stats=self.resource_stats,
                    config=self.config,
                    strategies=self.strategies,
                    timestamp=self.timestamp,
                )
        else:
            if not execution_record.resource_stats and self.resource_stats:
                execution_record.resource_stats = dict(self.resource_stats)
            if not execution_record.config and self.config:
                execution_record.config = dict(self.config)
            if not execution_record.strategies and self.strategies:
                execution_record.strategies = list(self.strategies)
            if not execution_record.task_name:
                execution_record.task_name = scenario_name
            if not execution_record.model_name:
                execution_record.model_name = self.model_info.name

        self.execution_record = execution_record
        self._legacy_metrics = self.execution_record.to_legacy_metrics()

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

        if name.startswith("_"):
            return

        execution_record = self.__dict__.get("execution_record")
        if execution_record is None:
            return

        if name == "scenario_name":
            execution_record.task_name = value
        elif name == "strategies":
            execution_record.strategies = list(value or [])
        elif name == "config":
            execution_record.config = dict(value or {})
        elif name == "resource_stats":
            execution_record.resource_stats = dict(value or {})
        elif name == "timestamp":
            execution_record.timestamp = value
        elif name == "model_info":
            execution_record.model_name = getattr(value, "name", "")

    @property
    def model_metrics(self) -> Dict[str, Any]:
        return self.execution_record.model_metrics

    @property
    def system_metrics(self) -> Dict[str, Any]:
        return self.execution_record.system_metrics

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._legacy_metrics

    @metrics.setter
    def metrics(self, value: Dict[str, Any]) -> None:
        value = dict(value or {})
        self._legacy_metrics = value

        execution_record = self.__dict__.get("execution_record")
        if execution_record is None:
            return

        synced_record = ExecutionRecord.from_legacy_metrics(
            value,
            task_name=self.scenario_name,
            route_type=getattr(execution_record, "route_type", ""),
            model_name=self.model_info.name,
            resource_stats=self.resource_stats,
            config=self.config,
            strategies=self.strategies,
            timestamp=self.timestamp,
        )
        execution_record.model_metrics = synced_record.model_metrics
        execution_record.system_metrics = synced_record.system_metrics
        execution_record.resource_stats = dict(self.resource_stats)
        execution_record.config = dict(self.config)
        execution_record.strategies = list(self.strategies)
        execution_record.task_name = self.scenario_name
        execution_record.model_name = self.model_info.name
        execution_record.timestamp = self.timestamp


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
        
        baseline_result = self._run_baseline(model_path, image_path)
        if baseline_result:
            self._results.append(baseline_result)
        
        baseline_fps = baseline_result.metrics.get('fps', {}).get('pure', 0) if baseline_result else 0
        
        for strategy_name in self.strategies_to_test:
            strategy_result = self._run_strategy(strategy_name, model_path, image_path, baseline_fps)
            if strategy_result:
                self._results.append(strategy_result)
        
        return self._results
    
    def _run_baseline(self, model_path: str, image_path: str) -> Optional[BenchmarkResult]:
        """运行基准测试（无策略）
        
        Args:
            model_path: 模型路径
            image_path: 图像路径
            
        Returns:
            BenchmarkResult: 基准结果
        """
        scenario = ModelSelectionScenario({
            'iterations': self.iterations,
            'warmup': self.warmup,
            'enable_monitoring': False
        })
        
        results = scenario.run([model_path], [image_path])
        
        if results:
            result = results[0]
            result.scenario_name = "baseline"
            result.strategies = ["baseline"]
            return result
        
        return None
    
    def _run_strategy(self, strategy_name: str, model_path: str, image_path: str, 
                      baseline_fps: float) -> Optional[BenchmarkResult]:
        """运行单个策略测试
        
        Args:
            strategy_name: 策略名称
            model_path: 模型路径
            image_path: 图像路径
            baseline_fps: 基准FPS
            
        Returns:
            BenchmarkResult: 策略结果
        """
        config = Config(model_path=model_path)
        
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
                return self._run_multithread_strategy(config, image_path, baseline_fps, theoretical_speedup)
            else:
                return self._run_simple_strategy(config, image_path, strategy_name, baseline_fps, theoretical_speedup)
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
    
    def _run_simple_strategy(self, config: Config, image_path: str, strategy_name: str,
                             baseline_fps: float, theoretical_speedup: float) -> Optional[BenchmarkResult]:
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
                config={'iterations': self.iterations, 'warmup': self.warmup}
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
    
    def _run_multithread_strategy(self, config: Config, image_path: str,
                                   baseline_fps: float, theoretical_speedup: float) -> Optional[BenchmarkResult]:
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
                config={'iterations': self.iterations, 'warmup': self.warmup, 'num_threads': config.num_threads}
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
