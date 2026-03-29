# 昇腾端侧 YOLO 评测系统实现说明文档

**版本**: 1.1.0  
**日期**: 2026-03-28  
**适用对象**: 开发者

---

## 一、系统架构

## 评测主线

### 标准评测主线

- 入口：`model-bench --input-tiers 720p 1080p 4K`
- 目标：在统一模型输入规格下，对不同原图输入分档的端到端成本进行公平比较
- 核心矩阵：`模型 x 输入分档 x 策略组合`
- 输出：模型执行指标、系统端到端指标、统一归档报告

### 高分辨率遥感主线

- 入口：`model-bench --routes tiled_route large_input_route --image-size-tiers 6K`
- `tiled_route`：滑窗切片、tile 推理、全图回拼
- `large_input_route`：固定大输入尺寸 `.om` 模型整图直检
- 核心矩阵：`路线类型 x 模型 x 图幅档位 x 策略组合`
- 第一阶段重点：性能与资源，不把检测效果纳入正式验收

### 策略验证主线

- 入口：`strategy-bench --strategies ... --threads ... --batch-size ...`
- 通过 `StrategyCompositionEngine` 做策略单元规范化、路线兼容性校验和执行器映射
- 当前真实执行器：
  - `multithread -> MultithreadInference`
  - `batch -> Inference(batch_size=N)`
  - `pipeline -> PipelineInference`
  - `memory_pool -> Inference + configurable pool`
  - `high_res -> high_res_tiling -> HighResInference`

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI 入口层                                │
│  main.py → commands/ (infer, model-bench, strategy-bench, ...)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        业务逻辑层                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  评测场景   │  │  策略单元   │  │  报告归档   │              │
│  │ benchmark/  │  │ strategies/ │  │ reporting/  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        核心推理层                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Inference 核心类                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │ 预处理   │ │ 执行器   │ │ 后处理   │ │ 资源管理 │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Multithread  │ │  Pipeline    │ │  HighRes     │             │
│  │  Inference   │ │  Inference   │ │  Inference   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        基础设施层                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ 配置管理 │ │ 日志系统 │ │ 异常处理 │ │ 参数验证 │           │
│  │ config/  │ │ logger   │ │exceptions│ │validators│           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ 指标收集 │ │ 资源监控 │ │ 内存池   │ │ 性能分析 │           │
│  │ metrics  │ │ monitor  │ │memory_pool│ │ profiler │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        硬件抽象层                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ACL 工具库 (acl_utils)                 │   │
│  │  init_acl | load_model | malloc_device | execute | ...   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 模块职责

| 模块 | 职责 | 关键文件 |
|------|------|---------|
| CLI入口 | 命令行解析和路由 | main.py, commands/ |
| 评测场景 | 标准评测、遥感路线、策略验证 | benchmark/scenarios.py |
| 策略组件 | 策略单元、组合规则、执行器映射 | src/strategies/ |
| 报告层 | 报告模型、渲染器、归档目录 | benchmark/reporters.py, reporting/ |
| 核心推理 | 推理执行 | src/inference.py |
| 配置管理 | 配置加载和验证 | config/ |
| 指标收集 | 性能统计 | utils/metrics.py |
| 资源监控 | 系统资源监控 | utils/monitor.py |

---

## 二、核心组件实现

### 2.0 统一报告与归档

- `benchmark/reporters.py`：把 `BenchmarkResult` 归一化为统一报告模型
- `reporting/renderers.py`：输出 Markdown 或 JSON 报告
- `reporting/archive.py`：按 `archives/<task>/<route>/` 写入报告、原始结果和元数据
- `scripts/run_smoke_eval.py`：根据 smoke 样例配置构建或执行真实硬件 smoke 命令

归档目录示例：

```text
reports/
└── archives/
    └── strategy_validation/
        └── mixed/
            ├── report.md
            ├── raw_results.json
            └── metadata.json
```

### 2.1 Inference 核心类

**文件**: `src/inference.py`

**类图**:
```
┌─────────────────────────────────────────┐
│              Inference                   │
├─────────────────────────────────────────┤
│ - config: Config                        │
│ - context: acl.rt_context               │
│ - model_id: int                         │
│ - input_buffer: int                     │
│ - output_buffer: int                    │
├─────────────────────────────────────────┤
│ + init() -> bool                        │
│ + preprocess(image_data) -> None        │
│ + execute() -> None                     │
│ + get_result() -> np.ndarray            │
│ + run_inference(image_data) -> ndarray  │
│ + destroy() -> None                     │
└─────────────────────────────────────────┘
```

**核心流程**:
```python
# 标准推理流程
inference = Inference(config)
inference.init()                    # 初始化ACL和加载模型
inference.preprocess(image_path)    # 预处理图像到设备内存
inference.execute()                 # 执行推理
result = inference.get_result()     # 获取结果
inference.destroy()                 # 释放资源
```

**内存管理**:
```
┌──────────────┐    memcpy    ┌──────────────┐    memcpy    ┌──────────────┐
│  Host Memory │ ──────────▶ │ Device Memory│ ──────────▶ │ Device Memory│
│  (input_host)│  H2D        │ (input_buffer)│  Execute    │(output_buffer)│
└──────────────┘             └──────────────┘             └──────────────┘
                                                              │
                                                              │ memcpy D2H
                                                              ▼
                                                          ┌──────────────┐
                                                          │  Host Memory │
                                                          │ (output_host)│
                                                          └──────────────┘
```

### 2.2 策略组件体系

**文件**: `src/strategies/`

**类继承关系**:
```
                    ┌──────────────┐
                    │   Strategy   │ (抽象基类)
                    ├──────────────┤
                    │ + name: str  │
                    │ + apply()    │
                    │ + get_metrics()│
                    └──────────────┘
                           ▲
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│MultithreadStrategy│ │ PipelineStrategy│ │  BatchStrategy  │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ - num_threads   │ │ - queue_size    │ │ - batch_size    │
│ - work_stealing │ │ - num_workers   │ │ - timeout_ms    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

**策略组合器**:
```python
composer = StrategyComposer()
composer.add_strategy(MultithreadStrategy(num_threads=4))
composer.add_strategy(MemoryPoolStrategy(pool_size=10))

context = InferenceContext(inference_instance)
context = composer.apply_all(context)  # 按顺序应用所有策略
metrics = composer.get_all_metrics()    # 收集所有指标
```

### 2.3 三层评测体系

**文件**: `benchmark/scenarios.py`

**评测场景对比**:

| 特性 | ModelSelectionScenario | StrategyValidationScenario | ExtremePerformanceScenario |
|------|------------------------|---------------------------|---------------------------|
| 目标 | 对比模型性能 | 验证策略效果 | 追求极限吞吐 |
| 策略 | 无 | 单一策略 | 策略组合 |
| 指标 | 延迟分布、FPS | 加速比、并行效率 | 吞吐量、资源利用率 |
| 预热 | 支持 | 支持 | 支持 |
| 监控 | 可选 | 可选 | 必选 |

**评测流程**:
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   预热阶段   │ ──▶ │  正式测试   │ ──▶ │  生成报告   │
│  (warmup)   │     │ (iterations)│     │  (report)   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
  不记录指标          记录完整指标          输出格式化报告
```

---

## 三、关键算法实现

### 3.1 工作窃取负载均衡

**位置**: `src/inference.py` MultithreadInference._worker_thread

**算法描述**:
```
1. 每个 worker 有独立的任务队列
2. Worker 先从自己的队列取任务
3. 自己队列为空时，尝试从其他 worker 队列偷取
4. 所有队列都为空时，进入休眠等待
```

**代码实现**:
```python
def _worker_thread(self, worker_id: int, worker: Inference) -> None:
    while self.running:
        try:
            # 先尝试从自己的队列取任务
            task = self.task_queues[worker_id].get(block=False)
            # 处理任务...
        except queue.Empty:
            # 自己队列为空，尝试窃取
            for other_id in range(len(self.task_queues)):
                if other_id == worker_id:
                    continue
                try:
                    task = self.task_queues[other_id].get(block=False)
                    # 处理偷取的任务...
                    break
                except queue.Empty:
                    continue
```

### 3.2 高分辨率图像分块推理

**位置**: `src/inference.py` split_image, HighResInference

**算法流程**:
```
┌──────────────┐
│  原始图像    │
│ (H x W)      │
└──────────────┘
       │
       ▼ split_image()
┌──────────────┐
│  分割子块    │
│ tile_size +  │
│ overlap      │
└──────────────┘
       │
       ▼ 并行推理
┌──────────────┐
│  子块结果    │
│ [N个检测结果]│
└──────────────┘
       │
       ▼ 权重融合
┌──────────────┐
│  合并结果    │
│ (去重叠)     │
└──────────────┘
```

**权重矩阵计算**:
```python
# 使用汉宁窗消除边缘硬拼接效应
hann_2d = np.outer(np.hanning(tile_h), np.hanning(tile_w))

# 累积权重
weight_map[y1:y2, x1:x2] += hann_2d[:y2-y1, :x2-x1]

# 归一化
weight_map[weight_map < 1e-6] = 1.0
```

### 3.3 延迟分布统计

**位置**: `utils/metrics.py` _calc_stats

**统计指标**:
```python
def _calc_stats(values: List[float]) -> StageStatistics:
    arr = np.array(values) * 1000  # 转换为毫秒
    
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
```

---

## 四、接口设计

### 4.1 命令行接口

**命令注册**:
```python
# main.py
subparsers = parser.add_subparsers(dest='command')

# 注册各命令
infer_parser = subparsers.add_parser('infer', help='推理')
model_bench_parser = subparsers.add_parser('model-bench', help='模型选型评测')
strategy_bench_parser = subparsers.add_parser('strategy-bench', help='策略验证评测')
extreme_bench_parser = subparsers.add_parser('extreme-bench', help='极限性能评测')
```

**命令处理函数**:
```python
def cmd_infer(args: argparse.Namespace) -> int:
    # 1. 验证参数
    # 2. 加载配置
    # 3. 执行推理
    # 4. 输出结果
    return 0  # 成功返回0
```

### 4.2 Python API 接口

**单张推理**:
```python
from src.api import InferenceAPI

result = InferenceAPI.inference_image(
    mode='base',           # 'base' | 'multithread' | 'high_res'
    image_path='test.jpg',
    config=config
)
```

**批量推理**:
```python
results = InferenceAPI.inference_batch(
    mode='multithread',
    image_paths=['test1.jpg', 'test2.jpg'],
    config=config
)
```

### 4.3 配置文件接口

**JSON Schema**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "model_path": {"type": "string"},
    "device_id": {"type": "integer", "minimum": 0},
    "resolution": {"type": "string", "pattern": "^\\d+x\\d+$"},
    "strategies": {
      "type": "object",
      "properties": {
        "multithread": {"$ref": "#/definitions/strategy_config"},
        "batch": {"$ref": "#/definitions/strategy_config"}
      }
    }
  }
}
```

---

## 五、扩展开发指南

### 5.1 添加新的推理策略

**步骤**:

1. 创建策略类:
```python
# src/strategies/my_strategy.py
from .base import Strategy, InferenceContext

class MyStrategy(Strategy):
    name = "my_strategy"
    
    def __init__(self, config: Optional[MyStrategyConfig] = None):
        super().__init__(config or MyStrategyConfig())
    
    def apply(self, context: InferenceContext) -> InferenceContext:
        # 实现策略逻辑
        return context
    
    def get_metrics(self) -> Dict[str, Any]:
        return {'custom_metric': self._custom_value}
```

2. 创建配置类:
```python
# config/strategy_config.py
@dataclass
class MyStrategyConfig:
    enabled: bool = False
    custom_param: int = 10
```

3. 注册策略:
```python
# src/strategies/composer.py
from .my_strategy import MyStrategy
StrategyComposer.register_strategy('my_strategy', MyStrategy)
```

### 5.2 添加新的评测场景

**步骤**:

1. 创建场景类:
```python
# benchmark/scenarios.py
class MyScenario(BenchmarkScenario):
    name = "my_scenario"
    
    def run(self, models: List[str], images: List[str], **kwargs) -> List[BenchmarkResult]:
        # 实现评测逻辑
        return self._results
    
    def generate_report(self, results: List[BenchmarkResult]) -> str:
        # 生成报告
        return report_content
```

2. 添加命令行接口:
```python
# commands/my_bench.py
def cmd_my_bench(args: argparse.Namespace) -> int:
    scenario = MyScenario(config)
    results = scenario.run(models, images)
    report = scenario.generate_report(results)
    print(report)
    return 0
```

### 5.3 添加新的报告格式

**步骤**:

1. 创建报告生成器:
```python
# benchmark/reporters.py
class MyReporter(Reporter):
    def generate(self, results: List[BenchmarkResult]) -> str:
        # 生成自定义格式报告
        return content
    
    def get_file_extension(self) -> str:
        return ".my_format"
```

2. 注册报告生成器:
```python
# benchmark/reporters.py
def create_reporter(format: str, **kwargs) -> Reporter:
    reporters = {
        'text': TextReporter,
        'json': JsonReporter,
        'html': HtmlReporter,
        'my_format': MyReporter  # 添加新格式
    }
    return reporters.get(format, TextReporter)(**kwargs)
```

---

## 六、调试与测试

### 6.1 日志配置

**日志级别**:
```python
from utils.logger import LoggerConfig

# 配置日志级别
LoggerConfig.set_level('DEBUG')  # DEBUG | INFO | WARNING | ERROR

# 获取日志器
logger = LoggerConfig.setup_logger('ascend_inference.inference')
```

**日志输出示例**:
```
[DEBUG] 2026-03-28 10:00:00 - inference - 预处理完成
[INFO]  2026-03-28 10:00:01 - inference - 模型加载成功：models/yolov8s.om
[WARN]  2026-03-28 10:00:02 - inference - 内存使用率较高：85%
[ERROR] 2026-03-28 10:00:03 - inference - 推理执行失败：错误码 2402
```

### 6.2 性能分析

**使用性能分析器**:
```python
from utils.profiler import profile_context, profile_decorator

# 上下文管理器方式
with profile_context("推理操作"):
    inference.run_inference(image_path)

# 装饰器方式
@profile_decorator
def my_function():
    pass
```

### 6.3 单元测试

**测试示例**:
```python
# tests/test_inference.py
import pytest
from unittest.mock import Mock, patch

class TestInference:
    @pytest.fixture
    def mock_acl(self):
        with patch('src.inference.HAS_ACL', True):
            with patch('src.inference.init_acl') as mock_init:
                mock_init.return_value = (Mock(), Mock())
                yield mock_init
    
    def test_init_success(self, mock_acl):
        inference = Inference(Config())
        assert inference.init() is True
    
    def test_preprocess_image_not_found(self):
        inference = Inference(Config())
        with pytest.raises(PreprocessError):
            inference.preprocess("nonexistent.jpg")
```

---

## 七、性能优化建议

### 7.1 内存优化

- 使用内存池复用缓冲区
- 及时释放不需要的内存
- 批处理时预分配内存

### 7.2 并发优化

- 合理设置线程数（不超过 AI Core 数量）
- 使用工作窃取平衡负载
- 避免锁竞争

### 7.3 I/O 优化

- 使用异步 I/O 读取图像
- 预取下一批图像
- 使用内存映射读取大文件

---

*文档版本: 1.1.0*
*最后更新: 2026-03-28*
