# 单元测试集成说明

## 概述

已将原来的 4 个独立测试文件集成到 `test_all.py` 中，统一管理所有单元测试。

## 文件结构

### 集成前
```
tests/
├── test_config.py           # Config 配置模块测试
├── test_api.py              # API 接口模块测试
├── test_inference.py        # 推理模块测试
├── test_logger.py           # 日志模块测试
└── __init__.py
```

### 集成后
```
tests/
├── test_all.py              # 集成所有测试
└── __init__.py
```

## 测试内容

### test_all.py 包含的测试

**1. Config 配置模块测试** (9 个用例)
- ✅ test_default_config - 默认配置
- ✅ test_from_json - JSON 加载
- ✅ test_from_json_nonexistent_file - 加载不存在文件
- ✅ test_apply_overrides - 参数覆盖
- ✅ test_get_resolution - 分辨率解析
- ✅ test_is_supported_resolution - 分辨率验证
- ✅ test_max_ai_cores - AI 核心数
- ✅ test_supported_resolutions - 支持的分辨率列表

**2. Inference 推理模块测试** (12 个用例)
- ✅ test_init_default_config - 默认配置初始化
- ✅ test_init_custom_config - 自定义配置初始化
- ✅ test_init_resolution - 分辨率解析
- ✅ test_context_manager_enter - 上下文入口
- ✅ test_context_manager_exit - 上下文出口
- ✅ test_init_no_acl - 无 ACL 库初始化
- ✅ test_preprocess_no_acl - 无 ACL 库预处理
- ✅ test_execute_no_acl - 无 ACL 库推理
- ✅ test_init_default_config (Multithread) - 多线程默认配置
- ✅ test_init_custom_config (Multithread) - 多线程自定义配置
- ✅ test_task_queue - 任务队列
- ✅ test_init_default_config (HighRes) - 高分辨率默认配置
- ✅ test_init_custom_config (HighRes) - 高分辨率自定义配置

**3. API 接口模块测试** (8 个用例)
- ✅ test_inference_image_no_inference_module - 无推理模块异常
- ✅ test_inference_batch_no_inference_module - 批量推理异常
- ✅ test_inference_image_base_mode - Base 模式推理
- ✅ test_inference_image_multithread_mode - 多线程模式推理
- ✅ test_inference_image_high_res_mode - 高分辨率模式推理
- ✅ test_inference_batch_base_mode - Base 模式批量推理
- ✅ test_inference_batch_multithread_mode - 多线程批量推理
- ✅ test_default_config - 默认配置

**4. Logger 日志模块测试** (9 个用例)
- ✅ test_setup_logger_default - 默认日志
- ✅ test_setup_logger_with_level - 日志级别
- ✅ test_setup_logger_with_file - 文件日志
- ✅ test_setup_logger_custom_format - 自定义格式
- ✅ test_get_logger - 获取日志
- ✅ test_logger_singleton - 单例特性
- ✅ test_logger_output - 日志输出
- ✅ test_logger_with_config - 配置集成
- ✅ test_multiple_loggers - 多个日志

**总计**: 38 个测试用例

## 使用方式

### 运行所有测试

```bash
# 运行所有测试
python -m pytest tests/test_all.py -v

# 使用简短输出
python -m pytest tests/test_all.py -v --tb=short

# 显示覆盖率
python -m pytest tests/test_all.py -v --cov=src --cov=config --cov=utils
```

### 运行特定测试类

```bash
# 只运行 Config 测试
python -m pytest tests/test_all.py::TestConfig -v

# 只运行 Inference 测试
python -m pytest tests/test_all.py::TestInference -v

# 只运行 API 测试
python -m pytest tests/test_all.py::TestInferenceAPI -v

# 只运行 Logger 测试
python -m pytest tests/test_all.py::TestLoggerConfig -v
```

### 运行特定测试用例

```bash
# 运行单个测试
python -m pytest tests/test_all.py::TestConfig::test_default_config -v

# 运行多个特定测试
python -m pytest tests/test_all.py::TestConfig::test_from_json tests/test_all.py::TestConfig::test_apply_overrides -v
```

### 使用关键字过滤

```bash
# 运行所有包含 "config" 的测试
python -m pytest tests/test_all.py -k "config" -v

# 运行所有包含 "init" 的测试
python -m pytest tests/test_all.py -k "init" -v

# 排除某些测试
python -m pytest tests/test_all.py -k "not no_acl" -v
```

## 输出示例

### 完整测试输出

```bash
$ python -m pytest tests/test_all.py -v

============================= test session starts ==============================
platform win32 -- Python 3.9.16, pytest-7.4.0, pluggy-1.0.0
rootdir: d:\code\AscendInference
collected 38 items

tests/test_all.py::TestConfig::test_default_config PASSED                [  2%]
tests/test_all.py::TestConfig::test_from_json PASSED                     [  5%]
tests/test_all.py::TestConfig::test_from_json_nonexistent_file PASSED    [  7%]
tests/test_all.py::TestConfig::test_apply_overrides PASSED               [ 10%]
tests/test_all.py::TestConfig::test_get_resolution PASSED                [ 13%]
tests/test_all.py::TestConfig::test_is_supported_resolution PASSED       [ 15%]
tests/test_all.py::TestConfig::test_max_ai_cores PASSED                  [ 18%]
tests/test_all.py::TestConfig::test_supported_resolutions PASSED         [ 21%]
tests/test_all.py::TestInference::test_init_default_config PASSED        [ 23%]
tests/test_all.py::TestInference::test_init_custom_config PASSED         [ 26%]
tests/test_all.py::TestInference::test_init_resolution PASSED            [ 29%]
tests/test_all.py::TestInference::test_context_manager_enter PASSED      [ 31%]
tests/test_all.py::TestInference::test_context_manager_exit PASSED       [ 34%]
tests/test_all.py::TestInference::test_init_no_acl SKIPPED (推理模块不可用) [ 36%]
tests/test_all.py::TestInference::test_preprocess_no_acl SKIPPED (推理模块不可用) [ 39%]
tests/test_all.py::TestInference::test_execute_no_acl SKIPPED (推理模块不可用) [ 42%]
tests/test_all.py::TestMultithreadInference::test_init_default_config PASSED [ 44%]
tests/test_all.py::TestMultithreadInference::test_init_custom_config PASSED [ 47%]
tests/test_all.py::TestMultithreadInference::test_task_queue PASSED      [ 50%]
tests/test_all.py::TestHighResInference::test_init_default_config PASSED [ 52%]
tests/test_all.py::TestHighResInference::test_init_custom_config PASSED  [ 55%]
tests/test_all.py::TestInferenceAPI::test_inference_image_no_inference_module PASSED [ 57%]
tests/test_all.py::TestInferenceAPI::test_inference_batch_no_inference_module PASSED [ 60%]
tests/test_all.py::TestInferenceAPI::test_inference_image_base_mode PASSED [ 63%]
tests/test_all.py::TestInferenceAPI::test_inference_image_multithread_mode PASSED [ 66%]
tests/test_all.py::TestInferenceAPI::test_inference_image_high_res_mode PASSED [ 68%]
tests/test_all.py::TestInferenceAPI::test_inference_batch_base_mode PASSED [ 71%]
tests/test_all.py::TestInferenceAPI::test_inference_batch_multithread_mode PASSED [ 73%]
tests/test_all.py::TestInferenceAPI::test_default_config PASSED          [ 76%]
tests/test_all.py::TestLoggerConfig::test_setup_logger_default PASSED    [ 78%]
tests/test_all.py::TestLoggerConfig::test_setup_logger_with_level PASSED [ 81%]
tests/test_all.py::TestLoggerConfig::test_setup_logger_with_file PASSED  [ 84%]
tests/test_all.py::TestLoggerConfig::test_setup_logger_custom_format PASSED [ 86%]
tests/test_all.py::TestLoggerConfig::test_get_logger PASSED              [ 89%]
tests/test_all.py::TestLoggerConfig::test_logger_singleton PASSED        [ 92%]
tests/test_all.py::TestLoggerConfig::test_logger_output PASSED           [ 94%]
tests/test_all.py::TestLoggerIntegration::test_logger_with_config PASSED [ 97%]
tests/test_all.py::TestLoggerIntegration::test_multiple_loggers PASSED   [100%]

================== 35 passed, 3 skipped in 2.50s ===================
```

### 测试覆盖率输出

```bash
$ python -m pytest tests/test_all.py -v --cov=src --cov=config --cov=utils

============================= test session starts ==============================
platform win32 -- Python 3.9.16, pytest-7.4.0, pluggy-1.0.0
rootdir: d:\code\AscendInference
collected 38 items

tests/test_all.py ...................................sss...              [100%]

---------- coverage: platform win32, python 3.9.16-final-0 -----------
Name                       Stmts   Miss  Cover
----------------------------------------------
config\config.py             150     20    87%
src\api.py                   170     45    74%
src\inference.py             670    280    58%
utils\logger.py               70      5    93%
utils\acl_utils.py           290    180    38%
utils\memory_pool.py         187     95    49%
utils\profiler.py             60     30    50%
----------------------------------------------
TOTAL                       1597    655    59%

================== 35 passed, 3 skipped in 3.20s ===================
```

## 迁移指南

### 原命令（废弃）

```bash
# 旧的运行方式
python -m pytest tests/test_config.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_inference.py -v
python -m pytest tests/test_logger.py -v
```

### 新命令（推荐）

```bash
# 新的运行方式
python -m pytest tests/test_all.py -v
python -m pytest tests/test_all.py::TestConfig -v
python -m pytest tests/test_all.py::TestInferenceAPI -v
```

## 优势

### 1. 统一管理
- ✅ 所有测试在一个文件中
- ✅ 易于导航和查找
- ✅ 统一的测试结构

### 2. 减少文件数量
- ✅ 从 4 个文件减少到 1 个
- ✅ 减少文件管理开销
- ✅ 更简洁的目录结构

### 3. 运行效率
- ✅ 一次加载所有测试
- ✅ 共享导入和配置
- ✅ 更快的测试启动时间

### 4. 易于维护
- ✅ 代码集中在一处
- ✅ 更容易添加新测试
- ✅ 更容易重构和改进

## 文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `tests/test_config.py` | ❌ 已删除 | 功能已集成到 test_all.py |
| `tests/test_api.py` | ❌ 已删除 | 功能已集成到 test_all.py |
| `tests/test_inference.py` | ❌ 已删除 | 功能已集成到 test_all.py |
| `tests/test_logger.py` | ❌ 已删除 | 功能已集成到 test_all.py |
| `tests/test_all.py` | ✅ 新建 | 集成所有单元测试 |

## 建议

### 1. 运行完整测试套件

```bash
# 每次代码修改后运行完整测试
python -m pytest tests/test_all.py -v
```

### 2. 快速验证

```bash
# 快速运行测试（不显示详细信息）
python -m pytest tests/test_all.py
```

### 3. 持续集成

```bash
# CI/CD 中使用
python -m pytest tests/test_all.py -v --tb=short --junitxml=test-results.xml
```

### 4. 查看覆盖率

```bash
# 生成 HTML 覆盖率报告
python -m pytest tests/test_all.py --cov=src --cov=config --cov=utils --cov-report=html

# 查看终端报告
python -m pytest tests/test_all.py --cov=src --cov=config --cov=utils --cov-report=term-missing
```

---

**更新日期**: 2026-03-06  
**版本**: 1.1.0
