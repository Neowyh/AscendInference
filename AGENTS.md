# AGENTS.md

本文件为在 `AscendInference` 仓库内协作的编码代理提供项目级说明，重点帮助代理快速理解模块边界、验证方式、文档约定与常见风险点。

## 项目概览

AscendInference 是一个面向华为昇腾设备的推理与评测工具集，覆盖两类核心能力：

- 推理执行与策略优化
- 模型选型、策略验证、极限压测三层评测与报告归档

当前仓库不再只是单一推理入口，而是由推理运行时、评测编排、注册表契约、报告输出和运维支撑共同组成。

## 常用命令

### 基础验证

```bash
pytest tests -q
python main.py --help
python main.py check
```

### 推理命令

```bash
python main.py infer test.jpg --model models/yolov8s.om
python main.py infer test.jpg --config config/default.json
python main.py infer ./images --output ./results
python main.py infer test.jpg --mode multithread --threads-per-core 2
python main.py infer large.jpg --mode high_res
python main.py infer test.jpg --benchmark --iterations 100
python main.py infer test.jpg --test-threads --thread-counts 1 2 4 8
```

### 评测命令

```bash
python main.py model-bench models/yolov8n.om models/yolov8s.om --images test.jpg --input-tiers 720p 1080p
python main.py strategy-bench --model models/yolov8s.om --image test.jpg --strategies multithread batch pipeline
python main.py extreme-bench --model models/yolov8s.om --images test.jpg --duration 60
```

### Smoke 命令构建与执行

```bash
python scripts/run_smoke_eval.py --mode standard
python scripts/run_smoke_eval.py --mode remote
python scripts/run_smoke_eval.py --mode strategy
python scripts/run_smoke_eval.py --mode standard --run
```

### 配置管理

```bash
python main.py config --show
python main.py config --validate
python main.py config --generate config/my_config.json
```

## 当前架构

### 模块职责

- `main.py`
  - 统一 CLI 入口，负责推理、评测、环境检查、增强、打包和配置管理
- `commands/`
  - 各命令处理器，例如 `infer`、`model_bench`、`strategy_bench`、`extreme_bench`、`config`、`check`
- `src/inference/`
  - 推理运行时核心，实现位于 `base.py`、`multithread.py`、`pipeline.py`、`high_res.py`、`preprocessor.py`、`executor.py`、`postprocessor.py`、`pool.py`
- `src/strategies/`
  - 策略抽象、策略组合与执行约束
- `benchmark/`
  - 评测场景装配与旧有 benchmark 兼容层
- `evaluations/`
  - 评测路线、输入分层、任务编排与评测主线定义
- `registry/`
  - 设备、模型、场景等注册式数据契约
- `reporting/`
  - 报告标准模型、渲染器与归档布局
- `config/`
  - 运行配置、策略配置、评测配置和校验逻辑
- `utils/`
  - ACL 帮助函数、参数校验、异常体系、日志、指标、监控和内存池
- `tests/`
  - 推理、策略、评测、注册表、报告、配置等测试

### 推理模式

| 模式 | 运行时类 | 典型用途 |
|------|----------|----------|
| `base` | `Inference` | 单图或简单批量推理 |
| `multithread` | `MultithreadInference` | 多线程提升吞吐 |
| `high_res` | `HighResInference` | 大图分块推理 |
| `pipeline` | `PipelineInference` | 预处理、执行、后处理流水并行 |

### 三层评测体系

| 评测层级 | 命令入口 | 目标 |
|------|----------|------|
| 模型选型评测 | `model-bench` | 比较不同模型在统一输入分层下的表现 |
| 策略验证评测 | `strategy-bench` | 对比优化策略与组合效果 |
| 极限性能评测 | `extreme-bench` | 持续压测、吞吐与资源监控 |

## 配置与校验规则

### 优先级

```text
命令行参数 > JSON 配置文件 > dataclass 默认值
```

### 新增配置项时

1. 在 `config/config.py` 或 `config/strategy_config.py` 中更新 dataclass。
2. 在 `config/` 下补充或调整 JSON 模板。
3. 在 `config/validator.py` 或 `utils/validators.py` 中同步校验规则。
4. 如果影响用户行为，需要同步更新 `README.md`、`docs/00-文档导航.md`、主题文档和 `AGENTS.md`。

### 分辨率词汇一致性

仓库中同时存在旧版和新版分辨率表达方式。修改分辨率相关行为时，需要一起核对：

- `config/config.py`
- `config/validator.py`
- `evaluations/tiers.py`
- CLI 帮助文本
- `README.md` 与 `docs/`

不要引入新的分辨率词汇而不完成全链路对齐。

## ACL 与资源管理

### 关键约束

1. 任何昇腾执行假设都要先检查 `HAS_ACL`。
2. 优先使用明确异常类型，例如 `ACLError`、`ModelLoadError`、`DeviceError`、`InputValidationError`。
3. 工作线程执行 ACL 操作前要正确设置 Ascend 上下文。
4. `destroy()` 路径必须释放资源。
5. 优先使用上下文管理器封装生命周期。

### 推荐写法

```python
with inference:
    result = inference.run_inference(image_path)
```

或

```python
infer = Inference(config)
infer.init()
try:
    result = infer.run_inference(image_path)
finally:
    infer.destroy()
```

避免实例初始化后长期悬挂不释放。

## 测试注意事项

### 当前环境特征

- Windows 临时目录权限可能导致 `PermissionError`。
- `.pytest_cache` 在权限受限场景下可能出现写入告警。
- ACL 缺失环境与真实代码回归要分开判断，不能混为一类失败。

### 测试编写约定

- 涉及导入缺失或 ACL 不可用的测试，应显式 mock 对应条件。
- 非路径校验测试不要依赖本地不存在的图片文件。
- 日志测试要考虑全局 logger 复用和 handler 状态残留。

### 完成前验证

优先执行：

```bash
pytest tests -q
```

然后把失败按以下三类拆开判断：

- 环境噪音
- 基线缺陷
- 新改动缺陷

## 分支审查与合并规范

### 已观察到的分支角色

- `master`
  - 最终主线
- `codex/worktree-bootstrap`
  - `.worktrees/` 忽略规则的整理分支
- `codex/ascend-yolo-system`
  - 评测、注册表与报告体系扩展分支

### 合并策略

1. 保留 `.worktrees/` 忽略规则，但不要长期保留仅做整理用途的分支。
2. 先稳定 `master` 基线，再合并大功能分支。
3. 合并评测系统前，必须确认：
   - 基线测试已分类或稳定
   - CLI 与文档已同步
   - 新评测链路存在明确验收矩阵

### 审查重点

- 正确性与回归风险
- 接口一致性
- 测试可信度
- 文档漂移
- 数据模型边界是否清晰

## 日志规范提醒

仓库内统一使用 `LoggerConfig` 建立日志器：

```python
from utils.logger import LoggerConfig

logger = LoggerConfig.setup_logger(
    name="my_app",
    level="info",
    log_file="app.log",
    format_type="json",
    sample_rate=0.1,
)

LoggerConfig.log_with_context(
    logger,
    "info",
    "Inference completed",
    image_path="test.jpg",
    inference_time=0.012,
    status="success",
)
```

日志器复用与 `propagate` 行为仍是维护热点；如果修改日志初始化语义，需要同步更新测试。

## 文档同步规则

当你新增或修改以下内容时，必须同时更新文档：

- CLI 命令
- 配置字段
- 评测分层或路线
- 报告与归档行为
- 主要模块边界

最少需要同步：

1. `README.md`
2. `docs/00-文档导航.md`
3. 对应主题文档
4. `AGENTS.md`

历史过程产物统一归档到 `docs/99-历史记录/`，不进入主导航。

## 仓库约束

- 真实设备执行依赖昇腾硬件与 ACL 库。
- 模型格式以 `.om` 为主。
- 非昇腾环境下，部分评测能力更适合作为契约和文档检查对象，而非真实硬件验收。
- 路径校验是刻意保持严格的；调整导入、ACL 与路径校验的先后顺序时，要同步考虑错误优先级语义。
