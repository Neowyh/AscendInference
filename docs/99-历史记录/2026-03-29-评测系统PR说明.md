# PR 标题

将 AscendInference 改造为面向昇腾端侧设备的 YOLO 选型与优化评测系统

## 背景

本次改造把仓库从“昇腾推理与 benchmark 工具”升级为“面向昇腾端侧设备的 YOLO 标准化评测与优化系统工程”。第一阶段重点是形成标准评测、遥感大图路线对照、策略单元化验证、报告渲染与归档的完整闭环。

## 本次改动

### 1. 标准评测主线

- 新增 `720p / 1080p / 4K` 输入分档
- 标准评测矩阵支持 `模型 x 输入分档 x 策略组合`
- 将设备、后端、分档元数据打通到执行记录与报告

### 2. 遥感高分辨率主线

- 新增 `tiled_route` 与 `large_input_route` 双路线评测
- 支持 `6K` 大图档位与路线对照
- 将路线类型统一透传到 `ExecutionRecord`

### 3. 策略单元化与组合规则

- 新增 `StrategyUnit`、`ValidationResult`、`StrategyCompositionEngine`
- 高分辨率切片策略统一规范化为 `high_res_tiling`
- 增加路线兼容性校验，阻止 `large_input_route + high_res_tiling` 这类无效组合
- `multithread / batch / pipeline / high_res` 都切到了真实执行器
- `threads` 和 `batch_size` 现在真正影响执行计划

### 4. 报告与归档

- 新增统一报告模型构建
- 新增 Markdown / JSON 渲染器
- 新增 `archives/<task>/<route>/` 归档布局
- `model-bench / strategy-bench / extreme-bench` 都已接入统一报告输出

### 5. 文档与 smoke

- 更新 README、文档中心和实现说明
- 新增 Ascend 硬件 smoke 评测脚本
- 新增标准评测、遥感评测、策略评测三份 smoke 样例配置

## 验证

已通过：

- `python -B -m pytest tests/test_registry.py tests/test_evaluation_tasks.py tests/test_reporting.py tests/test_scenarios.py tests/test_strategies.py tests/test_strategy_config.py -q`
- `python -B -m pytest tests/test_all.py tests/test_refactor_validation.py -q`
- `python main.py model-bench --help`
- `python main.py strategy-bench --help`

## 风险与后续

- 当前验证以 mock 和静态 CLI 检查为主，还没有在真实 Ascend 硬件上完成 smoke benchmark
- 检测效果指标（recall / mAP）仍在后续阶段，不在本次验收范围内
- 环境残留的 `__pycache__` 和不可访问的临时测试目录未纳入本次提交
