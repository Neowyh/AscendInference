# 昇腾推理工具文档中心

**版本**: 1.1.0  
**更新日期**: 2026-03-28

---

## 文档导航

### 用户文档

| 文档 | 说明 | 适用对象 |
|------|------|---------|
| [用户手册](user-manual.md) | 快速入门、命令详解、配置说明 | 所有用户 |
| [需求规格说明书](requirements-specification.md) | 功能需求、非功能需求、接口定义 | 产品经理、开发者 |

### 开发文档

| 文档 | 说明 | 适用对象 |
|------|------|---------|
| [实现说明文档](implementation-guide.md) | 系统架构、核心组件、扩展开发 | 开发工程师 |
| [API参考文档](api_reference.py) | API接口定义和使用示例 | 开发工程师 |
| [重构分析报告](refactoring-analysis.md) | 项目重构优化分析 | 架构师、开发者 |
| [优化实施计划](optimization-plan.md) | 优化任务清单和实施步骤 | 开发团队 |

### 运维文档

| 文档 | 说明 | 适用对象 |
|------|------|---------|
| [运维手册](operations-manual.md) | 部署指南、监控告警、故障排查 | 运维工程师 |
| [日志使用规范](logging-guidelines.md) | 日志级别使用指南 | 开发者、运维者 |

---

## 快速链接

### 新手入门

1. 阅读 [用户手册](user-manual.md) 了解基本使用方法
2. 查看 [需求规格说明书](requirements-specification.md) 了解系统功能
3. 参考 [运维手册](operations-manual.md) 进行环境部署

### 开发指南

1. 阅读 [实现说明文档](implementation-guide.md) 了解系统架构
2. 参考 [API参考文档](api_reference.py) 进行接口调用
3. 查看 [日志使用规范](logging-guidelines.md) 规范日志输出

### 项目维护

1. 参考 [重构分析报告](refactoring-analysis.md) 了解优化方向
2. 按照 [优化实施计划](optimization-plan.md) 执行优化任务

---

## 项目结构

```
AscendInference/
├── benchmark/              # 评测模块
│   ├── scenarios.py        # 三层评测场景
│   └── reporters.py        # 报告生成器
├── commands/               # CLI命令
│   ├── model_bench.py      # 模型选型评测命令
│   ├── strategy_bench.py   # 策略验证评测命令
│   └── extreme_bench.py    # 极限性能评测命令
├── config/                 # 配置管理
│   ├── __init__.py         # 配置类
│   ├── strategy_config.py  # 策略配置
│   └── validator.py        # 配置验证器
├── docs/                   # 文档目录
│   ├── user-manual.md      # 用户手册
│   ├── requirements-specification.md  # 需求规格
│   ├── implementation-guide.md        # 实现说明
│   ├── operations-manual.md           # 运维手册
│   ├── refactoring-analysis.md        # 重构分析
│   ├── optimization-plan.md           # 优化计划
│   └── logging-guidelines.md          # 日志规范
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
│   └── api.py              # API接口
├── tests/                  # 测试目录
│   ├── test_inference_core.py      # 核心推理测试
│   ├── test_refactor_validation.py # 重构验证测试
│   ├── test_metrics.py     # 指标测试
│   ├── test_strategies.py  # 策略测试
│   └── test_scenarios.py   # 场景测试
└── utils/                  # 工具模块
    ├── metrics.py          # 指标收集
    ├── monitor.py          # 资源监控
    ├── exceptions.py       # 异常定义
    ├── validators.py       # 参数验证
    ├── logger.py           # 日志系统
    └── memory_pool.py      # 内存池
```

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| 1.1.0 | 2026-03-28 | 三层评测体系、策略组件化、模块化重构 |
| 1.0.0 | - | 基础推理功能 |

---

*文档维护: AscendInference 开发团队*
