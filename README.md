# 昇腾端侧 YOLO 选型与优化系统

面向昇腾端侧设备的 YOLO 标准化评测与优化工程，支持常规输入分档评测、遥感高分辨率双路线对照、策略单元化验证，以及 Markdown/JSON 报告归档。

## 项目特点

- **三层评测体系**：
  - 模型选型评测：快速评估模型性能
  - 策略验证评测：验证优化策略效果
  - 极限性能评测：压力测试和资源监控
- **模块化架构**：推理模块拆分为预处理器、执行器、后处理器等独立组件
- **策略组件化**：支持策略组合和动态配置
- **高性能优化**：
  - 多线程推理（工作窃取负载均衡）
  - 流水线并行推理
  - 高分辨率图像分块推理（权重融合）
  - 推理池（实例复用）
  - 自适应批处理
  - 并行预处理
- **统一指标收集**：延迟分布统计（P50/P95/P99）、资源监控
- **完善的文档体系**：用户手册、需求规格、实现说明、运维手册

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

### 命令行使用

```bash
# 查看帮助
python main.py --help

# 推理命令
python main.py infer test.jpg --model models/yolov8s.om

# 环境检查
python main.py check

# 图像增强
python main.py enhance test.jpg --output ./enhanced

# 配置管理
python main.py config --show
```

## 评测主线

### 标准评测主线

- `720p / 1080p / 4K` 输入分档对比
- 多模型公平评测，区分模型执行指标与端到端系统指标
- 适合常规端侧检测场景下的模型选型

```bash
python main.py model-bench --models models/yolov8n.om models/yolov8s.om \
  --images test.jpg --input-tiers 720p 1080p 4K --output reports/standard.md
```

### 高分辨率遥感主线

- `tiled_route`：滑窗切片 + tile 推理 + 全图回拼
- `large_input_route`：固定大输入尺寸 `.om` 模型整图直检
- 适合 `6K` 等高分辨率遥感图的大图路线对照

```bash
python main.py model-bench --models models/small.om models/6k.om \
  --images image_6k.jpg --routes tiled_route large_input_route \
  --image-size-tiers 6K --output reports/remote.md
```

### 策略验证主线

- 支持 `multithread / batch / pipeline / memory_pool / high_res`
- 策略统一映射到可校验的策略单元与真实执行器
- 报告会按任务和路线自动归档

```bash
python main.py strategy-bench --model models/yolov8n.om --image image_6k.jpg \
  --strategies multithread batch pipeline high_res \
  --routes tiled_route large_input_route --image-size-tiers 6K \
  --threads 8 --batch-size 8 --output reports/strategy.md
```

## 三层评测体系

### 1. 模型选型评测

快速评估不同模型的性能指标：

```bash
python main.py model-bench --models models/yolov8n.om models/yolov8s.om --iterations 100
```

输出指标：
- 纯推理 FPS / 端到端 FPS
- 预处理/推理/后处理延迟
- P50/P95/P99 延迟分布
- 输入分档与路线维度的标准对照

### 2. 策略验证评测

验证优化策略的加速效果：

```bash
python main.py strategy-bench --strategies multithread batch pipeline --baseline
```

输出指标：
- 加速比
- 并行效率
- 吞吐量提升
- 路线兼容性校验和策略组合约束

### 3. 极限性能评测

压力测试和资源监控：

```bash
python main.py extreme-bench --duration 60 --monitor
```

输出指标：
- 持续吞吐量
- CPU/NPU 利用率
- 内存使用统计

## 报告与归档

- `text` 格式现在默认输出 Markdown 报告
- `json` 格式输出统一报告模型
- 指定 `--output` 时，会在输出目录下自动创建 `archives/<task>/<route>/`
- 归档内容包含：
  - `report.md` 或 `report.json`
  - `raw_results.json`
  - `metadata.json`

## Python API 使用

### 基础推理

```python
from config import Config
from src.inference import Inference

config = Config(model_path="models/yolov8s.om")

with Inference(config) as infer:
    result = infer.run_inference("test.jpg")
    print(f"推理结果形状: {result.shape}")
```

### 批量推理

```python
from src.inference import Inference

infer = Inference(config, batch_size=4)
infer.init()

results = infer.run_inference_batch([
    "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"
])

for i, result in enumerate(results):
    print(f"图像{i}: {result.shape}")

infer.destroy()
```

### 多线程推理

```python
from src.inference import MultithreadInference

mt = MultithreadInference(config)
mt.start()

for i in range(100):
    mt.add_task(f"image_{i}.jpg")

mt.wait_completion()
results = mt.get_results()
mt.stop()
```

### 流水线推理

```python
from src.inference import PipelineInference

pipeline = PipelineInference(config, batch_size=4)
pipeline.start(num_preprocess_threads=2, num_infer_threads=1)

def callback(batch_id, sub_batch_id, results):
    print(f"批次{batch_id}完成")

pipeline.submit(["img1.jpg", "img2.jpg"], callback=callback)
pipeline.wait_for_completion()
pipeline.stop()
```

### 推理池（实例复用）

```python
from src.inference.pool import InferencePool

with InferencePool(config, pool_size=4) as pool:
    # 单次推理
    result = pool.infer("test.jpg")
    
    # 批量推理
    results = pool.infer_batch(["img1.jpg", "img2.jpg"])
    
    # 异步提交
    future = pool.submit("test.jpg")
    result = future.result()
```

### 策略组件

```python
from src.strategies import StrategyComposer, MultithreadStrategy, BatchStrategy

composer = StrategyComposer()
composer.add_strategy(MultithreadStrategy(num_threads=4))
composer.add_strategy(BatchStrategy(batch_size=8))

context = composer.apply_all(context)
```

### 指标收集

```python
from utils.metrics import MetricsCollector, TimingRecord

collector = MetricsCollector(auto_warmup=True)

for _ in range(100):
    record = TimingRecord()
    record.preprocess_time = 0.01
    record.execute_time = 0.05
    record.postprocess_time = 0.02
    record.calculate_total()
    collector.record(record)

stats = collector.get_statistics()
print(f"FPS: {stats['fps']}")
print(f"P95延迟: {stats['latency']['p95']}")
```

### 资源监控

```python
from utils.monitor import ResourceMonitor

monitor = ResourceMonitor()
monitor.start()

# ... 执行推理 ...

monitor.stop()
stats = monitor.get_stats()
print(f"CPU利用率: {stats['cpu_avg']}")
print(f"内存使用: {stats['memory_avg']}")
```

## 推理模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `base` | 标准推理 | 单张图片，需要详细时间统计 |
| `multithread` | 多线程推理 | 批量处理，高性能吞吐 |
| `pipeline` | 流水线推理 | 高并发服务场景 |
| `high_res` | 高分辨率分块推理 | 大图像，超出模型输入尺寸 |

## 项目结构

```
AscendInference/
├── benchmark/              # 评测模块
│   ├── scenarios.py        # 三层评测场景
│   └── reporters.py        # 报告生成器
├── commands/               # CLI命令
│   ├── infer.py            # 推理命令
│   ├── model_bench.py      # 模型选型评测
│   ├── strategy_bench.py   # 策略验证评测
│   └── extreme_bench.py    # 极限性能评测
├── config/                 # 配置管理
│   ├── config.py           # 配置类
│   ├── strategy_config.py  # 策略配置
│   └── validator.py        # 配置验证器
├── docs/                   # 文档目录
│   ├── user-manual.md      # 用户手册
│   ├── requirements-specification.md  # 需求规格
│   ├── implementation-guide.md        # 实现说明
│   └── operations-manual.md           # 运维手册
├── src/
│   ├── inference/          # 推理模块
│   │   ├── base.py         # 基础推理类
│   │   ├── preprocessor.py # 预处理器
│   │   ├── executor.py     # 执行器
│   │   ├── postprocessor.py# 后处理器
│   │   ├── multithread.py  # 多线程推理
│   │   ├── pipeline.py     # 流水线推理
│   │   ├── high_res.py     # 高分辨率推理
│   │   └── pool.py         # 推理池
│   ├── strategies/         # 策略组件
│   │   ├── base.py         # 策略基类
│   │   ├── composer.py     # 策略组合器
│   │   └── adaptive_batch.py # 自适应批处理
│   ├── preprocessing/      # 预处理模块
│   │   └── parallel_preprocessor.py # 并行预处理器
│   └── api.py              # 统一API
├── tests/                  # 测试目录
│   ├── test_inference_core.py  # 核心推理测试
│   ├── test_strategies.py  # 策略测试
│   └── test_scenarios.py   # 场景测试
└── utils/                  # 工具模块
    ├── metrics.py          # 指标收集
    ├── monitor.py          # 资源监控
    ├── exceptions.py       # 异常定义
    └── validators.py       # 参数验证
```

## 配置系统

### 完整配置项

```json
{
  "model_path": "models/yolov8s.om",
  "device_id": 0,
  "resolution": "640x640",
  "num_threads": 4,
  "backend": "opencv",
  "strategies": {
    "multithread": {
      "enabled": true,
      "num_threads": 4
    },
    "batch": {
      "enabled": true,
      "batch_size": 4
    },
    "pipeline": {
      "enabled": false,
      "queue_size": 10
    },
    "high_res": {
      "enabled": false,
      "tile_size": 640,
      "overlap": 100
    }
  }
}
```

### 配置验证

```python
from config import Config
from config.validator import validate_config

config = Config(model_path="models/yolov8s.om")
result = validate_config(config)

if result.is_valid:
    print("配置有效")
else:
    print(f"配置错误: {result.errors}")
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行核心测试
pytest tests/test_inference_core.py tests/test_strategies.py -v

# 运行重构验证测试
pytest tests/test_refactor_validation.py -v
```

## 环境要求

- Python 3.6+
- Huawei Ascend AI 处理器
- AscendCL (ACL) 库
- NumPy
- PIL (Pillow)
- 可选：OpenCV

## 性能优化建议

| 优化项 | 性能提升 | 适用场景 |
|--------|----------|----------|
| 推理池复用 | +15% | 频繁推理场景 |
| 批处理推理 | +200% | 大吞吐量场景 |
| 流水线并行 | +30% | 高并发服务 |
| 自适应批处理 | +10% | 动态负载场景 |
| 并行预处理 | +30% | CPU密集预处理 |

## 文档

详细文档请参阅 [docs/README.md](docs/README.md)：

- [用户手册](docs/user-manual.md) - 快速入门和命令详解
- [需求规格说明书](docs/requirements-specification.md) - 功能需求定义
- [实现说明文档](docs/implementation-guide.md) - 系统架构和扩展开发
- [运维手册](docs/operations-manual.md) - 部署和故障排查

## 许可证

MIT License
