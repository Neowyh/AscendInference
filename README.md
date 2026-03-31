# AscendInference 项目说明

AscendInference 是一个面向华为昇腾设备的推理与评测工具集，围绕 `.om` 模型提供推理执行、策略优化、三层评测、报告归档与运维支撑能力。仓库既支持常规单图推理、多线程推理、高分辨率分块推理，也支持 `model-bench`、`strategy-bench`、`extreme-bench` 三类评测命令。

## 快速入口

- 文档导航：[`docs/00-文档导航.md`](docs/00-%E6%96%87%E6%A1%A3%E5%AF%BC%E8%88%AA.md)
- 协作说明：[`AGENTS.md`](AGENTS.md)
- 常用验证：`pytest tests -q`
- 环境检查：`python main.py check`

## 常用命令

### 推理

```bash
python main.py infer test.jpg --model models/yolov8s.om
python main.py infer test.jpg --config config/default.json
python main.py infer ./images --output ./results
python main.py infer test.jpg --mode multithread --threads-per-core 2
python main.py infer large.jpg --mode high_res
```

### 评测

```bash
python main.py model-bench models/yolov8n.om models/yolov8s.om --images test.jpg --input-tiers 720p 1080p
python main.py strategy-bench --model models/yolov8s.om --image test.jpg --strategies multithread batch pipeline
python main.py extreme-bench --model models/yolov8s.om --images test.jpg --duration 60
```

### Smoke 预检

```bash
python scripts/run_smoke_eval.py --mode standard
python scripts/run_smoke_eval.py --mode remote
python scripts/run_smoke_eval.py --mode strategy
```

### 配置管理

```bash
python main.py config --show
python main.py config --validate
python main.py config --generate config/my_config.json
```

## 项目结构

```text
AscendInference/
├── benchmark/            # 评测场景与报告拼装
├── commands/             # CLI 命令处理
├── config/               # 运行配置、评测配置、校验逻辑
├── docs/                 # 中文文档体系
├── evaluations/          # 评测路线、输入分层与任务编排
├── registry/             # 设备、模型、场景注册契约
├── reporting/            # 报告模型、渲染与归档
├── scripts/              # smoke 与辅助脚本
├── src/                  # 推理运行时与策略模块
├── tests/                # 单元测试与集成测试
└── utils/                # ACL、日志、异常、指标与监控工具
```

## 文档分区

- `docs/01-项目总览/`：需求、范围与背景说明
- `docs/02-使用指南/`：用户手册、接口示例与命令说明
- `docs/03-设计与实现/`：实现说明、重构分析、性能分析
- `docs/04-运维与维护/`：运维手册、日志规范、长期优化计划
- `docs/05-评审与计划/`：近期分支审查、合并方案与优化计划
- `docs/99-历史记录/`：历史设计稿、实施计划、PR 说明等归档材料

## 当前维护约定

- 所有正式文档统一放在 `docs/` 下维护，根目录仅保留 `README.md` 与 `AGENTS.md` 两个入口。
- 新增命令、配置字段、评测路线或报告行为时，需要同步更新 `README.md`、`docs/00-文档导航.md` 与相关主题文档。
- 历史过程产物不进入主导航，统一归档到 `docs/99-历史记录/`。
