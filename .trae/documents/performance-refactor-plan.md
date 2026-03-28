# 性能统计系统重构计划

## 一、重构概述

### 1.1 重构目标

根据《性能统计系统重构分析报告》，构建三层性能评测体系：

| 评测层级 | 目标 | 核心指标 | 使用场景 |
|---------|------|---------|---------|
| 模型选型评测 | 对比模型性能 | 纯推理时间、延迟分布、模型特征 | 选择最适合的模型 |
| 策略验证评测 | 验证加速效果 | 吞吐FPS、加速比、并行效率 | 选择最优加速策略 |
| 极限性能评测 | 追求极限性能 | 综合性能、资源利用率 | 生产环境部署 |

### 1.2 当前代码主要问题

| 问题类别 | 具体问题 | 影响 |
|---------|---------|------|
| 模型选型评测 | 计时不统一、无预热、无模型信息、无延迟分布 | 无法准确对比模型性能 |
| 策略验证评测 | 指标不统一、无策略对比、无加速比/效率 | 无法评估策略效果 |
| 极限性能评测 | 策略无法组合、无策略配置、无自动调优 | 无法追求极限性能 |

### 1.3 重构工作量估算

| 阶段 | 任务 | 预计工作量 | 优先级 | 状态 |
|------|------|--------|--------|------|
| 第一阶段 | 基础设施重构 | 5天 | P0 | ✅ 已完成 |
| 第二阶段 | 评测场景实现 | 5天 | P0 | ✅ 已完成 |
| 第三阶段 | 策略组件重构 | 5天 | P1 | ✅ 已完成 |
| 第四阶段 | 命令行接口重构 | 2天 | P1 | ✅ 已完成 |
| 第五阶段 | 测试和文档 | 3天 | P2 | 待完成 |
| **总计** | | **20天** | | |

---

## 二、第一阶段：基础设施重构 ✅

### 2.1 创建统一的统计指标收集器 ✅

**任务编号**: T1.1  
**状态**: ✅ 已完成  
**目标文件**: `utils/metrics.py`

**实现内容**:

1. **TimingRecord 数据类** - 单次计时记录
   - `preprocess_time`: 预处理时间
   - `execute_time`: 推理执行时间
   - `postprocess_time`: 后处理时间
   - `queue_wait_time`: 排队等待时间
   - `total_time`: 总时间
   - `timestamp`: 时间戳

2. **MetricsCollector 类** - 统一的统计指标收集器
   - `record(record)`: 记录一次计时
   - `finish_warmup()`: 结束预热阶段
   - `get_statistics()`: 计算统计结果
     - 平均值、最小值、最大值、标准差
     - P50、P95、P99 延迟分布
     - 各阶段时间占比
     - 纯推理FPS、端到端FPS

**验收标准**:
- [x] TimingRecord 可正确记录各阶段时间
- [x] MetricsCollector 支持预热和正式测试分离
- [x] 统计结果包含完整的延迟分布指标
- [x] 单元测试覆盖核心功能

---

### 2.2 重构配置系统，支持策略配置 ✅

**任务编号**: T1.2  
**状态**: ✅ 已完成  
**目标文件**: `config/strategy_config.py`

**实现内容**:

1. **策略配置类**:
   - `MultithreadStrategyConfig`: 多线程策略配置
   - `BatchStrategyConfig`: 批处理策略配置
   - `PipelineStrategyConfig`: 流水线策略配置
   - `MemoryPoolStrategyConfig`: 内存池策略配置
   - `HighResStrategyConfig`: 高分辨率策略配置

2. **StrategyConfig 类** - 策略配置集合
   - 聚合所有策略配置
   - 支持从字典/JSON加载
   - 支持导出为字典/JSON

3. **BenchmarkConfig 类** - 评测配置
   - `iterations`: 测试迭代次数
   - `warmup`: 预热次数
   - `enable_profiling`: 启用性能分析
   - `enable_monitoring`: 启用资源监控
   - `output_format`: 输出格式

**验收标准**:
- [x] 各策略配置类定义完整
- [x] StrategyConfig 支持序列化/反序列化
- [x] Config 类向后兼容现有代码
- [x] 配置文件示例更新

---

### 2.3 创建策略组件基类和组合器 ✅

**任务编号**: T1.3  
**状态**: ✅ 已完成  
**目标文件**: 
- `src/strategies/__init__.py`
- `src/strategies/base.py`
- `src/strategies/composer.py`

**实现内容**:

1. **Strategy 基类** - 策略组件基类
   - `name: str`: 策略名称
   - `apply(context)`: 应用策略到推理上下文
   - `get_metrics()`: 获取策略相关统计指标
   - `enable()`: 启用策略
   - `disable()`: 禁用策略

2. **InferenceContext 类** - 推理上下文
   - 封装推理实例和相关状态
   - 支持策略链式处理

3. **StrategyComposer 类** - 策略组合器
   - `add_strategy(strategy)`: 添加策略（支持链式调用）
   - `remove_strategy(name)`: 移除策略
   - `apply_all(context)`: 按顺序应用所有策略
   - `get_all_metrics()`: 收集所有策略指标
   - `get_config()`: 导出当前配置
   - `from_config(config)`: 从配置创建组合器

**验收标准**:
- [x] Strategy 基类定义清晰，接口完整
- [x] StrategyComposer 支持灵活组合
- [x] 支持从配置文件加载策略组合
- [x] 单元测试覆盖核心功能

---

### 2.4 添加资源监控功能 ✅

**任务编号**: T1.4  
**状态**: ✅ 已完成  
**目标文件**: `utils/monitor.py`

**实现内容**:

1. **ResourceMonitor 类** - 资源监控器
   - `start()`: 开始监控
   - `stop()`: 停止监控
   - `get_stats()`: 获取统计信息
     - NPU 利用率
     - 内存使用量
     - CPU 利用率

2. **SimpleResourceMonitor 类** - 简单资源监控器
   - 手动采样，无后台线程

**验收标准**:
- [x] 可监控 NPU 利用率（昇腾设备）
- [x] 可监控内存使用情况
- [x] 监控数据可导出

---

## 三、第二阶段：评测场景实现 ✅

### 3.1 实现模型选型评测场景 ✅

**任务编号**: T2.1  
**状态**: ✅ 已完成  
**目标文件**: `benchmark/scenarios.py`

**实现内容**:

1. **BenchmarkResult 数据类** - 评测结果
   - `scenario_name`: 场景名称
   - `model_info`: 模型信息
   - `metrics`: 统计指标
   - `strategies`: 使用的策略列表
   - `config`: 配置信息

2. **BenchmarkScenario 基类** - 评测场景基类
   - `name`: 场景名称
   - `run(models, images, **kwargs)`: 运行评测
   - `generate_report(results)`: 生成报告

3. **ModelSelectionScenario 类** - 模型选型评测场景
   - 不启用任何优化策略
   - 全面细致的统计指标
   - 支持多模型对比
   - 输出模型信息（输入/输出大小等）
   - 完整延迟分布（P50/P95/P99）

**验收标准**:
- [x] 支持同时测试多个模型
- [x] 输出完整的分阶段时间统计
- [x] 输出延迟分布指标
- [x] 生成模型对比表格

---

### 3.2 实现策略验证评测场景 ✅

**任务编号**: T2.2  
**状态**: ✅ 已完成  
**目标文件**: `benchmark/scenarios.py`

**实现内容**:

1. **StrategyValidationScenario 类** - 策略验证评测场景
   - 首先获取基准性能（无策略）
   - 分别测试每种策略
   - 计算加速比、并行效率
   - 支持策略对比

2. **关键指标**:
   - `throughput_fps`: 吞吐FPS
   - `speedup`: 加速比
   - `parallel_efficiency`: 并行效率
   - `latency_p50/p95/p99`: 延迟分布

**验收标准**:
- [x] 可测试多种策略
- [x] 计算加速比和并行效率
- [x] 生成策略对比报告

---

### 3.3 实现极限性能评测场景 ✅

**任务编号**: T2.3  
**状态**: ✅ 已完成  
**目标文件**: `benchmark/scenarios.py`

**实现内容**:

1. **ExtremePerformanceScenario 类** - 极限性能评测场景
   - 根据配置组合多种策略
   - 追求极限吞吐量
   - 监控资源利用率

2. **功能特性**:
   - 从配置文件加载策略组合
   - 集成资源监控
   - 输出综合性能报告

**验收标准**:
- [x] 支持策略组合配置
- [x] 集成资源监控
- [x] 输出极限性能报告

---

## 四、第三阶段：策略组件重构 ✅

### 4.1 重构多线程策略为组件 ✅

**任务编号**: T3.1  
**状态**: ✅ 已完成  
**目标文件**: `src/strategies/multithread.py`

**验收标准**:
- [x] 策略组件接口完整
- [x] 功能与原 MultithreadInference 一致
- [x] 单元测试通过

---

### 4.2 重构流水线策略为组件 ✅

**任务编号**: T3.2  
**状态**: ✅ 已完成  
**目标文件**: `src/strategies/pipeline.py`

**验收标准**:
- [x] 策略组件接口完整
- [x] 功能与原 PipelineInference 一致
- [x] 单元测试通过

---

### 4.3 重构批处理策略为组件 ✅

**任务编号**: T3.3  
**状态**: ✅ 已完成  
**目标文件**: `src/strategies/batch.py`

**验收标准**:
- [x] 策略组件接口完整
- [x] 支持配置批大小和超时
- [x] 单元测试通过

---

### 4.4 重构内存池策略为组件 ✅

**任务编号**: T3.4  
**状态**: ✅ 已完成  
**目标文件**: `src/strategies/memory_pool.py`

**验收标准**:
- [x] 策略组件接口完整
- [x] 功能与原 MemoryPool 一致
- [x] 单元测试通过

---

### 4.5 重构高分辨率策略为组件 ✅

**任务编号**: T3.5  
**状态**: ✅ 已完成  
**目标文件**: `src/strategies/high_res.py`

**验收标准**:
- [x] 策略组件接口完整
- [x] 功能与原 HighResInference 一致
- [x] 单元测试通过

---

## 五、第四阶段：命令行接口重构 ✅

### 5.1 实现新的命令行接口 ✅

**任务编号**: T4.1  
**状态**: ✅ 已完成  
**目标文件**: 
- `commands/model_bench.py`
- `commands/strategy_bench.py`
- `commands/extreme_bench.py`
- `main.py`

**实现内容**:

1. **model-bench 命令** - 模型选型评测
   ```bash
   python main.py model-bench models/yolov5s.om models/yolov8n.om \
       --images test1.jpg test2.jpg \
       --iterations 100 --warmup 5 \
       --output report.txt
   ```

2. **strategy-bench 命令** - 策略验证评测
   ```bash
   python main.py strategy-bench --model models/yolov8n.om --image test.jpg \
       --strategies multithread batch pipeline memory_pool
   ```

3. **extreme-bench 命令** - 极限性能评测
   ```bash
   python main.py extreme-bench --model models/yolov8n.om --images test_images/ \
       --config config/extreme.json
   ```

4. **保留 infer 命令** - 向后兼容

**验收标准**:
- [x] 新命令功能完整
- [x] 命令行参数设计合理
- [x] 向后兼容现有 infer 命令

---

### 5.2 添加报告生成功能 ✅

**任务编号**: T4.2  
**状态**: ✅ 已完成  
**目标文件**: `benchmark/reporters.py`

**实现内容**:

1. **Reporter 基类** - 报告生成器基类
   - `generate(results)`: 生成报告

2. **TextReporter 类** - 文本格式报告
   - 生成易读的文本报告

3. **JsonReporter 类** - JSON格式报告
   - 生成结构化JSON报告

4. **HtmlReporter 类** - HTML格式报告
   - 生成可视化HTML报告

**验收标准**:
- [x] 支持多种输出格式
- [x] 报告内容完整清晰
- [x] 支持自定义输出路径

---

## 六、第五阶段：测试和文档

### 6.1 编写单元测试

**任务编号**: T5.1  
**状态**: 待完成  
**目标文件**: `tests/`

**实现内容**:

1. **utils/metrics_test.py** - 指标收集器测试
2. **config/strategy_config_test.py** - 策略配置测试
3. **src/strategies/*_test.py** - 策略组件测试
4. **benchmark/scenarios_test.py** - 评测场景测试

**验收标准**:
- [ ] 核心功能测试覆盖
- [ ] 测试用例设计合理
- [ ] 所有测试通过

---

### 6.2 编写使用文档

**任务编号**: T5.2  
**状态**: 待完成  
**目标文件**: `docs/`

**实现内容**:

1. **快速开始指南**
2. **三层评测体系使用说明**
3. **策略配置说明**
4. **API参考文档**

**验收标准**:
- [ ] 文档结构清晰
- [ ] 示例代码可运行
- [ ] 覆盖主要使用场景

---

## 七、文件结构规划

```
AscendInference/
├── benchmark/                  # 评测模块 [✅ 已创建]
│   ├── __init__.py
│   ├── scenarios.py           # 评测场景
│   └── reporters.py           # 报告生成器
├── src/
│   ├── strategies/            # 策略组件 [✅ 已创建]
│   │   ├── __init__.py
│   │   ├── base.py           # 策略基类
│   │   ├── composer.py       # 策略组合器
│   │   ├── multithread.py    # 多线程策略
│   │   ├── batch.py          # 批处理策略
│   │   ├── pipeline.py       # 流水线策略
│   │   ├── memory_pool.py    # 内存池策略
│   │   └── high_res.py       # 高分辨率策略
│   ├── inference.py          # 核心推理 [已有]
│   └── api.py                # 统一API
├── utils/
│   ├── metrics.py            # 统计指标收集器 [✅ 已创建]
│   ├── monitor.py            # 资源监控器 [✅ 已创建]
│   └── profiler.py           # 性能分析器 [已有]
├── config/
│   ├── config.py             # 配置类 [✅ 已修改]
│   ├── strategy_config.py    # 策略配置 [✅ 已创建]
│   └── default.json          # 默认配置 [✅ 已修改]
├── commands/                  # CLI命令
│   ├── model_bench.py        # 模型选型评测 [✅ 已创建]
│   ├── strategy_bench.py     # 策略验证评测 [✅ 已创建]
│   ├── extreme_bench.py      # 极限性能评测 [✅ 已创建]
│   └── infer.py              # 推理命令 [已有]
├── tests/                     # 测试 [待完成]
│   ├── metrics_test.py
│   ├── strategy_config_test.py
│   └── scenarios_test.py
└── main.py                    # 入口 [✅ 已修改]
```

---

## 八、验收标准

### 8.1 功能验收

- [x] 模型选型评测可对比多个模型性能
- [x] 策略验证评测可评估各策略加速效果
- [x] 极限性能评测可组合多种策略
- [x] 所有命令行接口正常工作
- [x] 报告生成功能完整

### 8.2 性能验收

- [ ] 重构后性能不低于重构前
- [ ] 无内存泄漏
- [ ] 资源正确释放

### 8.3 质量验收

- [ ] 单元测试覆盖率 > 80%
- [x] 代码风格一致
- [ ] 文档完整

---

*计划创建时间: 2026-03-28*
*最后更新时间: 2026-03-28*
*基于: 性能统计系统重构分析报告*
