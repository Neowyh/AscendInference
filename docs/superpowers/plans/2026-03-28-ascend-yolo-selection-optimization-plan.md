# AscendInference 昇腾端侧 YOLO 选型与优化系统实施计划

> **面向代理执行者：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 子技能按任务逐项实施。步骤使用复选框 `- [ ]` 语法进行跟踪。

**目标：** 将 `AscendInference` 演进为一个以报告为核心的昇腾端侧 YOLO 选型与优化系统，覆盖标准评测主线和高分辨率遥感双路线对照主线。

**架构：** 保留现有 `commands/ + benchmark/ + src/ + config/ + tests/` 骨架，在其上补充资产定义、评测任务编排、路线化执行记录、报告渲染与归档模块。执行逻辑继续主要落在 `benchmark/` 与 `src/`，任务定义、报告模型和归档逻辑则拆入新的聚焦模块。

**技术栈：** Python 3、argparse CLI、dataclasses、现有 `benchmark` 场景体系、现有配置对象、pytest、JSON/Markdown 报告

---

## 计划中的文件结构

### 需要修改的现有文件

- `main.py`
  新增与计划一致的 CLI 入口以及路线化参数，同时保持现有命令可继续使用。
- `commands/model_bench.py`
  从直接包裹场景类，重构为标准模型评测任务入口。
- `commands/strategy_bench.py`
  从直接包裹场景类，重构为策略组合评测任务入口。
- `commands/extreme_bench.py`
  与统一执行记录结构对齐输出格式。
- `benchmark/scenarios.py`
  拆分共享 benchmark 记录能力，并补充标准评测与遥感路线实验编排。
- `benchmark/reporters.py`
  如可复用则保留，否则改造为最终报告渲染的底层支持。
- `config/config.py`
  扩展输入分档、路线类型、路线参数、报告输出等配置项。
- `config/strategy_config.py`
  扩展策略定义，使其支持单元化策略和路线范围校验。
- `config/validator.py`
  校验新的评测任务输入和路线/任务兼容性。
- `tests/test_scenarios.py`
  补充新场景任务编排与路线对照行为测试。
- `tests/test_strategy_config.py`
  补充新策略单元和路线校验测试。
- `README.md`
  更新系统定位和新的评测/报告工作流说明。

### 需要新增的文件

- `registry/__init__.py`
  资产注册包入口。
- `registry/models.py`
  模型资产定义与加载辅助函数。
- `registry/devices.py`
  设备资产定义与加载辅助函数。
- `registry/scenarios.py`
  标准场景与遥感路线场景定义。
- `registry/loader.py`
  YAML/JSON 注册表加载通用工具。
- `evaluations/__init__.py`
  评测任务编排包入口。
- `evaluations/tasks.py`
  标准评测任务、路线对照任务、归档元数据的数据结构。
- `evaluations/tiers.py`
  `720p / 1080p / 4K` 输入分档帮助模块。
- `evaluations/routes.py`
  `tiled_route` 与 `large_input_route` 的路线枚举和参数模型。
- `reporting/__init__.py`
  报告包入口。
- `reporting/models.py`
  标准报告数据模型和执行记录视图。
- `reporting/renderers.py`
  Markdown/JSON 渲染逻辑。
- `reporting/archive.py`
  结果目录布局和归档辅助函数。
- `tests/test_registry.py`
  资产加载与校验测试。
- `tests/test_evaluation_tasks.py`
  评测任务构建与路线矩阵展开测试。
- `tests/test_reporting.py`
  报告渲染与归档布局测试。
- `config/evaluation/default_standard_eval.json`
  标准评测任务模板。
- `config/evaluation/default_remote_sensing_eval.json`
  遥感路线对照任务模板。

## 里程碑概览

1. 建立资产、分档、路线、任务、报告的数据契约。
2. 围绕这些契约重构 benchmark 执行层。
3. 通过 CLI 和配置暴露新工作流。
4. 输出并归档标准报告。
5. 回填测试和文档。

## 任务 1：建立资产注册与任务契约

**文件：**
- Create: `registry/__init__.py`
- Create: `registry/models.py`
- Create: `registry/devices.py`
- Create: `registry/scenarios.py`
- Create: `registry/loader.py`
- Create: `evaluations/__init__.py`
- Create: `evaluations/tasks.py`
- Create: `evaluations/tiers.py`
- Create: `evaluations/routes.py`
- Test: `tests/test_registry.py`
- Test: `tests/test_evaluation_tasks.py`

- [ ] **步骤 1：先写失败测试**

```python
from evaluations.routes import RouteType
from evaluations.tiers import STANDARD_INPUT_TIERS
from registry.models import ModelAsset


def test_standard_input_tiers_are_fixed():
    assert [tier.name for tier in STANDARD_INPUT_TIERS] == ["720p", "1080p", "4K"]


def test_remote_sensing_route_types_include_tiled_and_large_input():
    assert {route.value for route in RouteType} >= {"tiled_route", "large_input_route"}


def test_model_asset_preserves_input_spec_and_route_capabilities():
    asset = ModelAsset(
        name="yolov8n_6k",
        path="models/yolov8n_6k.om",
        input_resolution="6k",
        supported_routes=["large_input_route"],
    )
    assert asset.input_resolution == "6k"
    assert asset.supported_routes == ["large_input_route"]
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_registry.py tests/test_evaluation_tasks.py -v`  
预期：FAIL，因为新的注册表/任务模块尚不存在。

- [ ] **步骤 3：实现最小可用的数据类**

```python
from dataclasses import dataclass, field
from enum import Enum


class RouteType(str, Enum):
    TILED = "tiled_route"
    LARGE_INPUT = "large_input_route"


@dataclass
class ModelAsset:
    name: str
    path: str
    input_resolution: str
    supported_routes: list[str] = field(default_factory=list)
```

- [ ] **步骤 4：补充分档、路线、任务辅助结构**

```python
STANDARD_INPUT_TIERS = (
    InputTier(name="720p", width=1280, height=720),
    InputTier(name="1080p", width=1920, height=1080),
    InputTier(name="4K", width=3840, height=2160),
)
```

- [ ] **步骤 5：再次运行测试，确认通过**

运行：`pytest tests/test_registry.py tests/test_evaluation_tasks.py -v`  
预期：PASS

- [ ] **步骤 6：提交**

```bash
git add registry evaluations tests/test_registry.py tests/test_evaluation_tasks.py
git commit -m "feat: add evaluation asset and task contracts"
```

## 任务 2：扩展配置，支持输入分档、路线与报告任务

**文件：**
- Modify: `config/config.py`
- Modify: `config/strategy_config.py`
- Modify: `config/validator.py`
- Create: `config/evaluation/default_standard_eval.json`
- Create: `config/evaluation/default_remote_sensing_eval.json`
- Test: `tests/test_strategy_config.py`
- Test: `tests/test_evaluation_tasks.py`

- [ ] **步骤 1：先写失败测试**

```python
from config import Config


def test_config_supports_standard_input_tiers():
    config = Config()
    assert "6k" in config.SUPPORTED_RESOLUTIONS


def test_config_accepts_route_and_report_settings():
    config = Config()
    config.apply_overrides(route_type="tiled_route", report_format="markdown")
    assert config.route_type == "tiled_route"
    assert config.report_format == "markdown"
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_strategy_config.py tests/test_evaluation_tasks.py -v`  
预期：FAIL，因为路线/报告配置字段和校验尚不存在。

- [ ] **步骤 3：新增路线相关配置字段**

```python
@dataclass
class Config:
    route_type: str = "standard"
    input_tier: str = "1080p"
    report_format: str = "markdown"
    archive_results: bool = True
```

- [ ] **步骤 4：补充策略与校验规则**

```python
if config.route_type == "large_input_route" and not config.resolution.endswith("6k"):
    errors.append("large_input_route requires a large-input model resolution")
```

- [ ] **步骤 5：新增评测配置模板**

```json
{
  "task_name": "default-standard-eval",
  "input_tiers": ["720p", "1080p", "4K"],
  "strategies": ["single_thread", "multithread"]
}
```

- [ ] **步骤 6：再次运行测试，确认通过**

运行：`pytest tests/test_strategy_config.py tests/test_evaluation_tasks.py -v`  
预期：PASS

- [ ] **步骤 7：提交**

```bash
git add config tests/test_strategy_config.py tests/test_evaluation_tasks.py
git commit -m "feat: extend config for route-aware evaluation tasks"
```

## 任务 3：统一 Benchmark 结果模型与执行记录

**文件：**
- Modify: `benchmark/scenarios.py`
- Create: `reporting/models.py`
- Test: `tests/test_scenarios.py`
- Test: `tests/test_reporting.py`

- [ ] **步骤 1：先写失败测试**

```python
from reporting.models import ExecutionRecord


def test_execution_record_distinguishes_model_and_system_metrics():
    record = ExecutionRecord(
        route_type="standard",
        model_metrics={"execute_ms": 12.0},
        system_metrics={"e2e_ms": 18.0},
    )
    assert record.model_metrics["execute_ms"] == 12.0
    assert record.system_metrics["e2e_ms"] == 18.0
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_scenarios.py tests/test_reporting.py -v`  
预期：FAIL，因为 `ExecutionRecord` 和路线化输出尚不存在。

- [ ] **步骤 3：引入统一执行记录模型**

```python
@dataclass
class ExecutionRecord:
    task_name: str
    route_type: str
    model_name: str
    model_metrics: dict
    system_metrics: dict
    resource_stats: dict
```

- [ ] **步骤 4：重构 benchmark 结果输出**

```python
return BenchmarkResult(
    scenario_name=self.name,
    metrics={"model": model_metrics, "system": system_metrics},
    config={"route_type": route_type, "input_tier": input_tier},
)
```

- [ ] **步骤 5：再次运行测试，确认通过**

运行：`pytest tests/test_scenarios.py tests/test_reporting.py -v`  
预期：PASS

- [ ] **步骤 6：提交**

```bash
git add benchmark/scenarios.py reporting/models.py tests/test_scenarios.py tests/test_reporting.py
git commit -m "refactor: normalize execution records for evaluation reports"
```

## 任务 4：实现标准评测主线编排

**文件：**
- Modify: `benchmark/scenarios.py`
- Modify: `commands/model_bench.py`
- Modify: `main.py`
- Test: `tests/test_scenarios.py`
- Test: `tests/test_evaluation_tasks.py`

- [ ] **步骤 1：先写失败测试**

```python
def test_model_selection_scenario_expands_across_standard_input_tiers():
    scenario = ModelSelectionScenario({"input_tiers": ["720p", "1080p", "4K"]})
    matrix = scenario.build_matrix(["a.om"], ["img.jpg"])
    assert len(matrix) == 3
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_scenarios.py tests/test_evaluation_tasks.py -v`  
预期：FAIL，因为尚未实现分档矩阵展开。

- [ ] **步骤 3：在 benchmark 场景中加入分档编排能力**

```python
def build_matrix(self, models, images):
    return [
        {"model": model, "image": image, "input_tier": tier}
        for model in models
        for image in images
        for tier in self.input_tiers
    ]
```

- [ ] **步骤 4：更新 CLI，支持分档评测**

```python
parser.add_argument("--input-tiers", nargs="+", default=["720p", "1080p", "4K"])
```

- [ ] **步骤 5：再次运行测试，确认通过**

运行：`pytest tests/test_scenarios.py tests/test_evaluation_tasks.py -v`  
预期：PASS

- [ ] **步骤 6：提交**

```bash
git add benchmark/scenarios.py commands/model_bench.py main.py tests/test_scenarios.py tests/test_evaluation_tasks.py
git commit -m "feat: add tier-based standard evaluation orchestration"
```

## 任务 5：实现遥感双路线对照编排

**文件：**
- Modify: `benchmark/scenarios.py`
- Modify: `commands/model_bench.py`
- Modify: `commands/strategy_bench.py`
- Modify: `main.py`
- Test: `tests/test_scenarios.py`
- Test: `tests/test_evaluation_tasks.py`

- [ ] **步骤 1：先写失败测试**

```python
def test_remote_sensing_route_matrix_includes_tiled_and_large_input_routes():
    scenario = RouteExperimentScenario({"image_size_tiers": ["6K"]})
    matrix = scenario.build_route_matrix(["small.om", "6k.om"], ["image_6k.jpg"])
    route_types = {item["route_type"] for item in matrix}
    assert route_types == {"tiled_route", "large_input_route"}
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_scenarios.py tests/test_evaluation_tasks.py -v`  
预期：FAIL，因为路线对照编排尚不存在。

- [ ] **步骤 3：增加路线化场景编排**

```python
def build_route_matrix(self, models, images):
    return [
        {"model": model, "image": image, "route_type": route}
        for model in models
        for image in images
        for route in ["tiled_route", "large_input_route"]
    ]
```

- [ ] **步骤 4：增加路线相关 CLI 参数**

```python
parser.add_argument("--routes", nargs="+", default=["tiled_route", "large_input_route"])
parser.add_argument("--image-size-tiers", nargs="+", default=["6K"])
```

- [ ] **步骤 5：再次运行测试，确认通过**

运行：`pytest tests/test_scenarios.py tests/test_evaluation_tasks.py -v`  
预期：PASS

- [ ] **步骤 6：提交**

```bash
git add benchmark/scenarios.py commands/model_bench.py commands/strategy_bench.py main.py tests/test_scenarios.py tests/test_evaluation_tasks.py
git commit -m "feat: add remote sensing route comparison orchestration"
```

## 任务 6：抽离策略单元与组合规则

**文件：**
- Modify: `config/strategy_config.py`
- Create: `src/strategies/base_unit.py`
- Create: `src/strategies/composition.py`
- Modify: `src/strategies/__init__.py`
- Modify: `commands/strategy_bench.py`
- Test: `tests/test_strategies.py`
- Test: `tests/test_strategy_config.py`

- [ ] **步骤 1：先写失败测试**

```python
from src.strategies.composition import StrategyCompositionEngine


def test_strategy_composition_rejects_invalid_route_combination():
    engine = StrategyCompositionEngine()
    result = engine.validate(["high_res_tiling"], route_type="large_input_route")
    assert result.is_valid is False
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_strategies.py tests/test_strategy_config.py -v`  
预期：FAIL，因为策略单元组合和路线约束尚不存在。

- [ ] **步骤 3：增加最小策略单元接口**

```python
@dataclass
class StrategyUnit:
    name: str
    supported_routes: tuple[str, ...]
```

- [ ] **步骤 4：增加组合和冲突校验**

```python
if route_type == "large_input_route" and "high_res_tiling" in strategies:
    return ValidationResult(is_valid=False, errors=["high_res_tiling is incompatible with large_input_route"])
```

- [ ] **步骤 5：再次运行测试，确认通过**

运行：`pytest tests/test_strategies.py tests/test_strategy_config.py -v`  
预期：PASS

- [ ] **步骤 6：提交**

```bash
git add src/strategies config/strategy_config.py commands/strategy_bench.py tests/test_strategies.py tests/test_strategy_config.py
git commit -m "feat: add modular strategy composition rules"
```

## 任务 7：实现 Markdown/JSON 报告渲染与归档

**文件：**
- Create: `reporting/renderers.py`
- Create: `reporting/archive.py`
- Modify: `benchmark/reporters.py`
- Modify: `commands/model_bench.py`
- Modify: `commands/strategy_bench.py`
- Modify: `commands/extreme_bench.py`
- Test: `tests/test_reporting.py`

- [ ] **步骤 1：先写失败测试**

```python
from reporting.renderers import MarkdownReportRenderer


def test_markdown_report_contains_route_comparison_section():
    report = MarkdownReportRenderer().render({"route_comparison": [{"route": "tiled_route"}]})
    assert "Route Comparison" in report
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_reporting.py -v`  
预期：FAIL，因为新报告渲染器和归档模块尚不存在。

- [ ] **步骤 3：实现报告渲染器**

```python
class MarkdownReportRenderer:
    def render(self, report_model):
        return "# Evaluation Report\n\n## Route Comparison\n"
```

- [ ] **步骤 4：实现归档目录布局辅助逻辑**

```python
def build_archive_path(root, task_name, route_type):
    return root / task_name / route_type
```

- [ ] **步骤 5：将命令接入报告渲染与保存**

```python
report_body = renderer.render(report_model)
archive_result(archive_root, task_metadata, report_body, raw_results)
```

- [ ] **步骤 6：再次运行测试，确认通过**

运行：`pytest tests/test_reporting.py -v`  
预期：PASS

- [ ] **步骤 7：提交**

```bash
git add reporting benchmark/reporters.py commands/model_bench.py commands/strategy_bench.py commands/extreme_bench.py tests/test_reporting.py
git commit -m "feat: add report rendering and archive layout"
```

## 任务 8：回填文档与端到端 Mock 验证

**文件：**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/implementation-guide.md`
- Modify: `tests/test_all.py`
- Modify: `tests/test_refactor_validation.py`

- [ ] **步骤 1：先写失败的端到端 Mock 测试**

```python
def test_end_to_end_standard_evaluation_produces_archive_metadata():
    result = run_mock_standard_evaluation()
    assert result["report_path"].endswith(".md")
```

- [ ] **步骤 2：运行测试，确认失败**

运行：`pytest tests/test_all.py tests/test_refactor_validation.py -v`  
预期：FAIL，因为新评测/报告链路还未被覆盖。

- [ ] **步骤 3：补充 Mock 端到端验证**

```python
def run_mock_standard_evaluation():
    return {"report_path": "reports/mock/report.md", "raw_results": 3}
```

- [ ] **步骤 4：更新顶层文档**

```markdown
## 评测主线

- 标准评测：720p / 1080p / 4K 分档对比
- 遥感评测：tiled route 与 large-input route 对照
```

- [ ] **步骤 5：再次运行测试，确认通过**

运行：`pytest tests/test_all.py tests/test_refactor_validation.py -v`  
预期：PASS

- [ ] **步骤 6：提交**

```bash
git add README.md docs/README.md docs/implementation-guide.md tests/test_all.py tests/test_refactor_validation.py
git commit -m "docs: document evaluation system workflows"
```

## 最终验证

全部任务完成后，运行以下聚焦测试集：

- `pytest tests/test_registry.py tests/test_evaluation_tasks.py tests/test_reporting.py tests/test_scenarios.py tests/test_strategies.py tests/test_strategy_config.py -v`
- `pytest tests/test_all.py tests/test_refactor_validation.py -v`

如果有 Ascend 硬件环境，再补充执行以下 smoke check：

- `python main.py model-bench --help`
- `python main.py strategy-bench --help`

## 风险与约束

- 在引入路线与分档概念时，保持现有 CLI 参数兼容性。
- 不要把报告渲染逻辑重新塞回 `benchmark/scenarios.py`，渲染逻辑应保持在 `reporting/`。
- 不要把路线选择逻辑埋进策略定义里，路线选择属于任务编排，不属于策略组合。
- `large_input_route` 只能用于显式声明支持大输入的模型。
- 遥感检测效果指标在第一阶段只是预留字段，不应变成必须产出的正式结果。

## 执行说明

- 优先按顺序完成任务 1 到任务 3，再开始改 CLI。
- 每完成一个任务就保持测试通过，不要等最后一起修。
- 如果在任务 3 到任务 5 期间 `benchmark/scenarios.py` 继续膨胀，可在同一任务内顺手拆出 `benchmark/standard_eval.py` 和 `benchmark/remote_sensing_eval.py`，不要放任单文件继续失控。

