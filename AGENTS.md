# AGENTS.md

This file provides repository-specific guidance to coding agents working in `AscendInference`.

## Project Overview

AscendInference is an AscendCL-based inference and evaluation toolkit for Huawei Ascend devices. The repository now covers two closely related concerns:

- model inference and optimization strategies
- a three-track evaluation/reporting system for model selection, strategy validation, and extreme-performance testing

The codebase is no longer just a single inference entrypoint. It now includes route/tier-aware evaluation flows, registry-style contracts, report normalization, and archive generation.

## Common Commands

### Core Verification

```bash
pytest tests -v
python main.py --help
python main.py check
```

### Inference

```bash
python main.py infer test.jpg --model models/yolov8s.om
python main.py infer test.jpg --config config/default.json
python main.py infer ./images --output ./results
python main.py infer test.jpg --mode multithread --threads-per-core 2
python main.py infer large.jpg --mode high_res
python main.py infer test.jpg --benchmark --iterations 100
python main.py infer test.jpg --test-threads --thread-counts 1 2 4 8
```

### Evaluation Commands

```bash
# Model selection / standard evaluation
python main.py model-bench models/yolov8n.om models/yolov8s.om --images test.jpg --input-tiers 720p 1080p

# Strategy validation
python main.py strategy-bench --model models/yolov8s.om --image test.jpg --strategies multithread batch pipeline

# Extreme performance evaluation
python main.py extreme-bench --model models/yolov8s.om --images test.jpg --duration 60
```

### Smoke Command Builder

```bash
python scripts/run_smoke_eval.py --mode standard
python scripts/run_smoke_eval.py --mode remote
python scripts/run_smoke_eval.py --mode strategy
python scripts/run_smoke_eval.py --mode standard --run
```

### Configuration Management

```bash
python main.py config --show
python main.py config --validate
python main.py config --generate config/my_config.json
```

### Image Enhancement

```bash
python main.py enhance test.jpg --output ./enhanced --resolutions 640x640 1k 2k
```

## Current Architecture

### Module Responsibilities

- `main.py`
  - unified CLI entry for inference, evaluation, environment check, enhancement, packaging, and config management
- `commands/`
  - CLI command handlers such as `infer`, `model_bench`, `strategy_bench`, `extreme_bench`, `config`, `check`
- `src/inference/`
  - core inference runtime split into focused modules such as `base.py`, `multithread.py`, `pipeline.py`, `high_res.py`, `preprocessor.py`, `executor.py`, `postprocessor.py`, `pool.py`
- `src/strategies/`
  - strategy abstractions, composition rules, and strategy-specific optimization units
- `benchmark/`
  - benchmark scenarios and report-building glue
- `evaluations/`
  - route definitions and standard input tiers used by the newer evaluation flows
- `registry/`
  - registry-style data contracts for devices, models, and scenarios
- `reporting/`
  - normalized report models, renderers, and archive helpers
- `config/`
  - runtime config dataclasses, strategy config, evaluation config, and validation helpers
- `utils/`
  - ACL helpers, validators, exceptions, logger, metrics, monitor, profiler, memory pool
- `tests/`
  - unit and integration-style tests for inference, strategy, registry, reporting, config, and scenario layers

### Inference Modes

| Mode | Runtime | Use Case |
|------|---------|----------|
| `base` | `Inference` | single-image or simple batch inference |
| `multithread` | `MultithreadInference` | higher throughput using multiple worker threads |
| `high_res` | `HighResInference` | tiled handling for oversized images |
| `pipeline` | `PipelineInference` | overlap preprocess / execute / postprocess for throughput |

### Evaluation Tracks

| Track | Entry Command | Focus |
|------|---------------|-------|
| Standard model selection | `model-bench` | compare models under normalized input tiers |
| Strategy validation | `strategy-bench` | compare optimization strategies and compositions |
| Extreme performance | `extreme-bench` | throughput, resource pressure, and stress-style runs |

## Configuration and Validation Rules

### Priority

```text
Command line arguments > JSON config file > dataclass defaults
```

### When Adding Config

1. Add or update the dataclass in `config/config.py` or `config/strategy_config.py`.
2. Add or update the corresponding JSON template in `config/`.
3. Update validation behavior in `config/validator.py` or `utils/validators.py`.
4. Update command help, README, and `AGENTS.md` if the setting changes user-facing behavior.

### Important Constraint

The repository currently uses multiple resolution naming systems across older and newer code paths. When changing resolution-related behavior, confirm consistency across:

- `config/config.py`
- `config/validator.py`
- `evaluations/tiers.py`
- CLI help text
- README / docs

Do not introduce a new resolution vocabulary without reconciling all of the above.

## ACL and Resource Management

### Critical Rules

1. Check `HAS_ACL` before assuming Ascend runtime availability.
2. Use specific exception types such as `ACLError`, `ModelLoadError`, `DeviceError`, `InputValidationError`.
3. In worker threads, set Ascend context correctly before ACL work.
4. Always release ACL resources in `destroy()` paths.
5. Prefer context-manager patterns when the API supports them.

### Safe Usage Pattern

```python
with inference:
    result = inference.run_inference(image_path)
```

or

```python
infer = Inference(config)
infer.init()
try:
    result = infer.run_inference(image_path)
finally:
    infer.destroy()
```

Avoid leaving inference objects initialized without cleanup.

## Testing Notes

### Current Environment Caveats

- Some tests are sensitive to Windows temp-directory permissions.
- In this environment, `pytest tests -q` has produced `PermissionError` failures under the system temp directory.
- `.pytest_cache` writes may also warn if permissions are constrained.

Do not classify every test failure as a code regression without separating:

- environment noise
- baseline defects
- new feature defects

### Mocking Expectations

- Tests that target import or ACL-unavailable behavior should mock those conditions directly.
- Tests should not rely on missing local image files unless the test is explicitly about path validation.
- Logger tests should account for global logger reuse and handler state.

### Verification Bias

Before calling work complete, prefer:

```bash
pytest tests -q
```

and then classify failures carefully rather than summarizing them as a single bucket.

## Branch Review and Merge Guidance

### Branch Roles Observed Locally

- `master`
  - intended final merge target
- `codex/worktree-bootstrap`
  - minimal housekeeping branch for `.worktrees/` ignore behavior
- `codex/ascend-yolo-system`
  - major feature branch for evaluation, registry, and reporting expansion

### Merge Strategy

1. Preserve the `.worktrees/` ignore rule, but do not keep `codex/worktree-bootstrap` as a long-lived branch.
2. Stabilize baseline tests on `master` before merging the major feature branch.
3. Merge `codex/ascend-yolo-system` only after:
   - baseline stabilization
   - CLI/doc synchronization
   - acceptance coverage for the new evaluation/reporting paths

### During Review

Prioritize:

- correctness and regression risk
- interface consistency
- test trustworthiness
- documentation drift
- maintainability of new data-model boundaries

## Logging Guidance

Use `LoggerConfig` for repository-standard logging.

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

Be aware that logger reuse is currently a maintenance hotspot. If you change logger setup semantics, update tests accordingly.

## Documentation Sync Rules

Whenever you add or change:

- a CLI command
- a config field
- an evaluation tier or route
- a report/archive behavior
- a major module boundary

you must update the relevant documentation set together:

1. `README.md` for user-facing entrypoints
2. `docs/implementation-guide.md` for architecture details
3. `docs/README.md` if navigation changes
4. `AGENTS.md` for repository-specific contributor guidance

## Constraints

- Ascend hardware and ACL libraries are required for real device execution.
- `.om` is the expected model format.
- Some evaluation features are best-effort in non-Ascend environments and should be treated as contract/documentation review targets unless hardware is available.
- Path validation is intentionally strict; be careful when changing error-precedence behavior around validation versus ACL/import checks.
