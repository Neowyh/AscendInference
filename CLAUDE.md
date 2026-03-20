# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AscendCL model inference tool for Huawei Ascend AI processors, providing high-performance inference with single-thread, multi-thread, high-resolution, batch processing, and pipeline parallel modes.

## Common Commands

### Running Tests
```bash
pytest tests/ -v
```

### Environment Check
```bash
python main.py check
```

### Inference
```bash
# Single image inference
python main.py infer test.jpg --model models/yolov8s.om

# Use config file

python main.py infer test.jpg --config config/default.json

# Batch inference (input is directory)
python main.py infer ./images --output ./results

# Multithread inference
python main.py infer test.jpg --mode multithread --threads-per-core 2

# High-resolution tile-based inference
python main.py infer large.jpg --mode high_res

# Performance benchmark
python main.py infer test.jpg --benchmark --iterations 100

# Multi-thread performance test
python main.py infer test.jpg --test-threads --thread-counts 1 2 4 8
```

### Configuration Management
```bash
# Show current config
python main.py config --show

# Validate config
python main.py config --validate

# Generate default config
python main.py config --generate config/my_config.json
```

### Image Enhancement
```bash
python main.py enhance test.jpg --output ./enhanced --resolutions 640x640 1k 2k
```

## Core Architecture

### Module Responsibilities
- **commands/**: CLI command implementations (infer/check/enhance/package/config)
- **src/inference.py**: Core inference classes (Inference/MultithreadInference/PipelineInference/HighResInference)
- **src/api.py**: Unified API layer (InferenceAPI)
- **config/config.py**: Configuration management
- **utils/acl_utils.py**: ACL initialization, model loading, memory management
- **utils/validators.py**: Parameter validation with security checks
- **utils/exceptions.py**: Hierarchical exception system
- **utils/logger.py**: Structured logging with text/JSON support
- **utils/memory_pool.py**: Memory pool for buffer reuse
- **utils/profiler.py**: Performance profiling

### Inference Modes
| Mode | Class | Use Case |
|------|-------|----------|
| base | Inference | Single image with detailed timing |
| multithread | MultithreadInference | Batch processing with work-stealing load balancing |
| high_res | HighResInference | Large images split into tiles with weighted blending |
| pipeline | PipelineInference | High throughput with preprocess/infer/postprocess overlap |

### Performance Optimizations
1. **Memory Pool**: Pre-allocated buffers reused across inference runs (15%+ improvement)
2. **OpenCV Backend**: 30-50% faster image preprocessing than PIL
3. **Work-Stealing Multithreading**: Dynamic load balancing (20%+ improvement)
4. **Batch Processing**: Batch size 2-8 for maximum NPU utilization (200-300% improvement)
5. **Pipeline Parallelism**: CPU/NPU overlap (30%+ improvement)

## Exception System

All exceptions inherit from `InferenceError` base class with:
- `error_code`: Integer error code
- `original_error`: Original exception if wrapped
- `details`: Dictionary with contextual information

### Error Code Ranges
- **2000-2099**: ACL/device errors (initialization, model load, memory)
- **2100-2199**: Model-related errors
- **2200-2299**: Preprocessing errors
- **2300-2399**: ACL execution errors
- **2400-2499**: Inference execution errors
- **2500-2599**: Postprocessing errors
- **3000-3099**: Input validation errors
- **4000+**: Other errors

### Exception Types
- `ACLError`: ACL operation failures (includes `acl_ret` in details)
- `ModelLoadError`: Model loading failures
- `DeviceError`: Device initialization failures
- `PreprocessError`: Image preprocessing failures
- `PostprocessError`: Result postprocessing failures
- `MemoryError`: Memory allocation/free failures
- `InputValidationError`: Parameter validation failures
- `ThreadError`: Multithreading operation failures

### Best Practices for Exceptions
- Always use specific exception types (e.g., `ModelLoadError` not `InferenceError`)
- Provide `error_code` from appropriate range
- Include `details` with relevant context
- Preserve `original_error` when wrapping exceptions

## Parameter Validation

All public APIs use `utils/validators.py` for validation:
- Path security validation (prevents directory traversal)
- Numeric range validation
- Enum validation (backends, modes, resolutions)
- Business parameter validation (resolution, device_id, batch_size)

### Security
Paths are validated against current working directory to prevent traversal attacks. Use `validate_file_path()` with `must_exist=True` and `allowed_extensions` for model files.

## Configuration System

Three-tier priority:
```
Command line arguments > JSON config file > Code defaults
```

### Adding New Configuration Items
1. Add field to `Config` dataclass in [config/config.py](config/config.py)
2. Add default value in [config/default.json](config/default.json)
3. Add validation in [utils/validators.py](utils/validators.py) if needed

### AI Core Configuration
Modify `MAX_AI_CORES` in [config/config.py](config/config.py):
- Ascend 310P: 4
- Ascend 310 (dual-core): 8
- Ascend 910: 32

## ACL Interaction

### Important Guidelines
1. Check `HAS_ACL` flag before any ACL operations
2. Use `ACLError` for ACL failures with `acl_ret` parameter
3. Call `acl.rt.set_context(context)` in worker threads
4. Always clean up resources in `destroy()` methods
5. Use context managers (`with inference:`) for automatic cleanup

### Memory Management
- Use `malloc_device()`/`malloc_host()` from [utils/acl_utils.py](utils/acl_utils.py)
- Prefer `MemoryPool` class for frequently allocated buffers
- Always pair allocation with corresponding `free_device()`/`free_host()`

## Logging

Use structured logging for production environments:

```python
from utils.logger import LoggerConfig

# Setup logger (JSON format recommended for production)
logger = LoggerConfig.setup_logger(
    name="my_app",
    level="info",
    log_file="app.log",
    format_type="json",  # or "text"
    sample_rate=0.1      # Sample rate, ERROR always outputs
)

# Log with context
LoggerConfig.log_with_context(logger, "info", "Inference completed",
    image_path="test.jpg",
    inference_time=0.012,
    status="success"
)
```

## Adding New Inference Modes

1. Create new inference class in [src/inference.py](src/inference.py)
2. Add mode option to argparse in [main.py](main.py)
3. Add mode branch in [src/api.py](src/api.py) `InferenceAPI.inference_image()`
4. Add mode validation in [utils/validators.py](utils/validators.py)
5. Update README.md

## Resource Management

### Critical Pattern
Always use context managers or explicit cleanup:

```python
# Good: Context manager
with inference:
    result = inference.run_inference(image_path)

# Good: Explicit cleanup
infer = Inference(config)
infer.init()
try:
    result = infer.run_inference(image_path)
finally:
    infer.destroy()

# Bad: May leak resources
infer = Inference(config)
infer.init()
result = infer.run_inference(image_path)
```

### Leak Detection
The `__del__` methods in inference classes log warnings when resources are not properly released.

## Design Decisions

### Why OpenCV as default backend
OpenCV image processing is 30%+ faster than PIL, especially for resize and color conversion operations.

### Why memory pool
Each preprocessing step allocates host memory. Frequent allocation/deallocation hurts performance. Memory pool pre-allocates and reuses buffers.

### Why work-stealing algorithm
Simple round-robin allocation causes imbalance when image processing times vary. Work-stealing lets idle workers steal tasks from other queues.

### Why pipeline parallelism
Traditional: preprocess → infer → postprocess
Pipeline: Three stages process different batches simultaneously, CPU preprocess and NPU infer overlap completely.

## Constraints
- Only works on Ascend devices, not other GPUs
- Requires ACL library installation
- Model format must be .om (Ascend model format)
- Inference throughput limited by batch_size and available NPU memory
- Path validation prevents access outside working directory
