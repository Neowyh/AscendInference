# 昇腾推理工具 用户手册

## 概述

昇腾推理工具是一个面向华为昇腾AI处理器的模型推理性能评测和优化工具。提供三层性能评测体系，帮助用户选择最适合的模型和优化策略。

## 快速开始

### 环境要求

- Python 3.8+
- 华为昇腾AI处理器（可选，支持模拟模式）
- CANN 软件栈（可选）

### 安装

```bash
# 克隆项目
git clone <repository_url>
cd AscendInference

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

#### 1. 模型推理

```bash
# 单张图像推理
python main.py infer --model models/yolov8s.om --image test.jpg

# 批量推理
python main.py infer --model models/yolov8s.om --images test1.jpg test2.jpg

# 性能测试
python main.py infer --model models/yolov8s.om --image test.jpg --perf-test
```

#### 2. 模型选型评测

对比不同模型的性能，选择最适合的模型：

```bash
# 测试单个模型
python main.py model-bench models/yolov8s.om --images test.jpg

# 对比多个模型
python main.py model-bench models/yolov5s.om models/yolov8n.om models/yolov10n.om \
    --images test1.jpg test2.jpg \
    --iterations 100 --warmup 5 \
    --output report.txt
```

**输出示例**：

```
================================================================================
模型选型评测报告
================================================================================

模型: yolov8n.om
  路径: models/yolov8n.om
  输入大小: 1228.80 KB
  输出大小: 32.77 KB
  分辨率: 640x640

  时间统计:
    预处理:   avg=10.25 ms, p50=9.80, p95=12.50
    推理执行: avg=20.15 ms, p50=19.50, p95=22.80
    后处理:   avg=5.10 ms, p50=4.90, p95=6.20
    总时间:   avg=35.50 ms, p50=34.20, p95=41.50, p99=45.80

  性能指标:
    纯推理FPS: 49.63
    端到端FPS: 28.17

================================================================================
模型对比表格
================================================================================
模型                           推理时间(ms)    纯推理FPS       端到端FPS       
--------------------------------------------------------------------------------
yolov5s.om                     18.20          54.95           30.25
yolov8n.om                     20.15          49.63           28.17
yolov10n.om                    22.50          44.44           25.80
================================================================================
```

#### 3. 策略验证评测

验证各种加速策略的效果：

```bash
# 测试所有策略
python main.py strategy-bench --model models/yolov8n.om --image test.jpg

# 测试特定策略
python main.py strategy-bench --model models/yolov8n.om --image test.jpg \
    --strategies multithread batch pipeline

# 自定义参数
python main.py strategy-bench --model models/yolov8n.om --image test.jpg \
    --iterations 50 --threads 8 --output report.txt
```

**输出示例**：

```
================================================================================
策略验证评测报告
================================================================================

基准性能（无策略）:
  纯推理FPS: 50.00

策略对比:
策略                吞吐FPS         加速比          并行效率       
--------------------------------------------------------------------------------
baseline            50.00           1.00x           100.0%
multithread         180.00          3.60x           90.0%
batch               150.00          3.00x           75.0%
pipeline            120.00          2.40x           80.0%
================================================================================
```

#### 4. 极限性能评测

追求极限吞吐量：

```bash
# 使用配置文件
python main.py extreme-bench --model models/yolov8n.om --images test_images/ \
    --config config/extreme.json

# 快速测试
python main.py extreme-bench --model models/yolov8n.om --images test.jpg \
    --enable-multithread --threads 8 --duration 10

# 启用多策略组合
python main.py extreme-bench --model models/yolov8n.om --images test.jpg \
    --enable-multithread --threads 8 \
    --enable-memory-pool \
    --output report.txt
```

## 命令详解

### infer 命令

基础推理命令，支持单张/批量推理和性能测试。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型路径 | 必填 |
| --image/--images | 图像路径 | 必填 |
| --config | 配置文件路径 | config/default.json |
| --device | 设备ID | 0 |
| --backend | 图像处理后端 | pil |
| --perf-test | 性能测试模式 | False |
| --iterations | 性能测试迭代次数 | 100 |
| --warmup | 预热次数 | 5 |

### model-bench 命令

模型选型评测，对比不同模型性能。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| models | 模型路径列表（位置参数） | 必填 |
| --images | 测试图像路径 | 必填 |
| --iterations | 测试迭代次数 | 100 |
| --warmup | 预热次数 | 5 |
| --output | 报告输出路径 | 控制台输出 |
| --format | 输出格式 | text |
| --enable-monitoring | 启用资源监控 | False |

### strategy-bench 命令

策略验证评测，验证加速策略效果。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型路径 | 必填 |
| --image | 测试图像路径 | 必填 |
| --strategies | 要测试的策略 | 全部策略 |
| --iterations | 测试迭代次数 | 50 |
| --warmup | 预热次数 | 3 |
| --threads | 多线程数 | 4 |
| --batch-size | 批大小 | 4 |

### extreme-bench 命令

极限性能评测，追求极限吞吐量。

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 模型路径 | 必填 |
| --images | 测试图像路径或目录 | 必填 |
| --config | 策略配置文件 | - |
| --duration | 测试时长（秒） | 10 |
| --enable-multithread | 启用多线程策略 | False |
| --threads | 多线程数 | 4 |
| --enable-batch | 启用批处理策略 | False |
| --enable-pipeline | 启用流水线策略 | False |
| --enable-memory-pool | 启用内存池策略 | False |

## 配置文件

### 默认配置 (config/default.json)

```json
{
  "model_path": "models/yolov8s.om",
  "device_id": 0,
  "resolution": "640x640",
  "strategies": {
    "multithread": {
      "enabled": false,
      "num_threads": 4,
      "work_stealing": true
    },
    "batch": {
      "enabled": false,
      "batch_size": 4,
      "timeout_ms": 10.0
    },
    "pipeline": {
      "enabled": false,
      "queue_size": 10
    },
    "memory_pool": {
      "enabled": false,
      "pool_size": 10
    }
  },
  "benchmark": {
    "iterations": 100,
    "warmup": 5,
    "enable_profiling": true,
    "enable_monitoring": true
  }
}
```

### 策略配置说明

#### 多线程策略 (multithread)

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| enabled | bool | 是否启用 | false |
| num_threads | int | 线程数 | 4 |
| work_stealing | bool | 工作窃取负载均衡 | true |
| dynamic_scaling | bool | 动态算力调整 | false |

#### 批处理策略 (batch)

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| enabled | bool | 是否启用 | false |
| batch_size | int | 批大小 | 4 |
| timeout_ms | float | 超时时间（毫秒） | 10.0 |
| dynamic_batch | bool | 动态批处理 | false |

#### 流水线策略 (pipeline)

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| enabled | bool | 是否启用 | false |
| queue_size | int | 队列大小 | 10 |
| num_preprocess_threads | int | 预处理线程数 | 2 |
| num_infer_threads | int | 推理线程数 | 1 |
| num_postprocess_threads | int | 后处理线程数 | 1 |

#### 内存池策略 (memory_pool)

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| enabled | bool | 是否启用 | false |
| pool_size | int | 池大小 | 10 |
| growth_factor | float | 增长因子 | 1.5 |
| max_buffers | int | 最大缓冲区数 | 20 |

## 输出格式

### 文本格式 (text)

默认输出格式，易读的文本报告。

### JSON格式 (json)

结构化JSON报告，便于程序处理：

```bash
python main.py model-bench models/yolov8n.om --images test.jpg --format json --output report.json
```

### HTML格式 (html)

可视化HTML报告：

```bash
python main.py model-bench models/yolov8n.om --images test.jpg --format html --output report.html
```

## 常见问题

### Q: 如何选择合适的评测场景？

- **模型选型评测**：需要对比多个模型，选择最适合的模型时使用
- **策略验证评测**：已确定模型，需要选择最优加速策略时使用
- **极限性能评测**：生产环境部署前，追求极限吞吐量时使用

### Q: 预热次数如何设置？

建议设置 3-10 次预热，让系统达到稳定状态后再进行正式测试。

### Q: 如何解读评测报告？

重点关注：
- **纯推理FPS**：模型本身的推理性能
- **端到端FPS**：包含预处理和后处理的完整性能
- **延迟分布**：P95/P99 延迟反映尾部延迟情况
- **加速比**：策略优化效果
- **并行效率**：资源利用效率

## 版本历史

- v1.1.0：新增三层评测体系、策略组件化、报告生成功能
- v1.0.0：基础推理功能

---

*文档版本: 1.1.0*
*最后更新: 2026-03-28*
