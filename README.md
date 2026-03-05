# 昇腾推理项目

精简高效的昇腾 AscendCL 模型推理工具

## 项目特点

- **精简代码**：从 2500+ 行减少到 800 行，消除所有冗余
- **统一 API**：简洁的推理接口，支持多种模式
- **高性能**：支持多线程和高分辨率图像分块推理
- **易用性**：命令行工具和 Python API 两种使用方式
- **灵活配置**：JSON 配置文件 + 命令行参数覆盖

## 快速开始

### 安装依赖

```bash
pip install numpy Pillow
# 可选
pip install opencv-python
```

### 配置环境变量

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

### 使用命令行工具

```bash
# 推理单张图片
python main.py single test.jpg --model models/yolov8s.om

# 使用配置文件
python main.py single test.jpg --config config/default.json

# 批量推理
python main.py batch ./images --output ./results

# 性能测试（统计各阶段时间）
python main.py single test.jpg --mode base
```

### 使用 Python API

```python
from config import Config
from src.api import InferenceAPI

# 配置
config = Config(
    model_path="models/yolov8s.om",
    device_id=0,
    resolution="640x640"
)

# 推理单张图片
result = InferenceAPI.inference_image(
    mode="base",
    image_path="test.jpg",
    config=config
)

# 批量推理
results = InferenceAPI.inference_batch(
    mode="multithread",
    image_paths=["test1.jpg", "test2.jpg"],
    config=config
)
```

## 推理模式

- **base**: 标准推理，适合单张图片，统计各阶段时间
- **multithread**: 多线程推理，适合批量处理，统计各阶段时间
- **high_res**: 高分辨率分块推理，适合大图像

## 配置系统

### 两层配置架构

```
┌─────────────────────────────────────┐
│  1. JSON 文件 - 基础配置（最全）     │
│     - 包含所有配置项                │
│     - 作为配置的基准                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  2. 命令行参数 - 覆盖配置            │
│     - 只包含需要调整的项            │
│     - 优先级更高，覆盖 JSON 配置      │
└─────────────────────────────────────┘
```

### 使用方式

#### 方式 1：仅使用 JSON 配置文件

```bash
python main.py single test.jpg --config config/default.json
```

#### 方式 2：JSON 配置 + 命令行参数覆盖

```bash
python main.py single test.jpg \
    --config config/default.json \
    --device 1 \
    --resolution 1024x1024
```

#### 方式 3：仅使用命令行参数（向后兼容）

```bash
python main.py single test.jpg \
    --model models/yolov8s.om \
    --device 0 \
    --resolution 640x640
```

### 优先级规则

```
命令行参数 > JSON 配置文件 > 代码默认值
```

### 完整配置项

```json
{
  "model_path": "models/yolov8s.om",
  "device_id": 0,
  "resolution": "640x640",
  "tile_size": 640,
  "overlap": 100,
  "num_threads": 4,
  "backend": "pil",
  "conf_threshold": 0.4,
  "iou_threshold": 0.5,
  "max_detections": 100,
  "enable_logging": true,
  "log_level": "info",
  "enable_profiling": false
}
```

### 配置项说明

| 配置项 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `model_path` | str | 模型文件路径 | `models/yolov8s.om` |
| `device_id` | int | Ascend 设备 ID | `0`, `1`, `2` |
| `resolution` | str | 输入分辨率 | `640x640`, `1k`, `2k` |
| `tile_size` | int | 分块大小（高分辨率） | `640` |
| `overlap` | int | 分块重叠像素 | `100` |
| `num_threads` | int | 线程数 | `4`, `8` |
| `backend` | str | 图像后端 | `pil`, `opencv` |
| `conf_threshold` | float | 置信度阈值 | `0.4` |
| `iou_threshold` | float | NMS IoU 阈值 | `0.5` |
| `max_detections` | int | 最大检测框数 | `100` |
| `enable_logging` | bool | 启用日志 | `true`, `false` |
| `log_level` | str | 日志级别 | `info`, `debug`, `warning` |
| `enable_profiling` | bool | 性能分析 | `false`, `true` |

### 配置文件模板

项目提供了多种配置文件模板，位于 `config/` 目录：

- **default.json** - 默认配置（平衡性能）
- **high_performance.json** - 高性能配置
- **high_accuracy.json** - 高精度配置
- **high_resolution.json** - 高分辨率配置

### 在代码中使用配置

```python
from config import Config

# 从 JSON 文件加载配置
config = Config.from_json("config/default.json")

# 可选：用代码覆盖配置
config.apply_overrides(
    device_id=1,
    num_threads=8
)

# 使用配置
inference = Inference(config)
inference.init()
```

## 性能统计

### 单张推理时间统计

运行单张推理时会自动统计各阶段时间：

```bash
python main.py single test.jpg --config config/default.json
```

**输出示例：**
```
推理配置:
  模式：base
  图像：test.jpg
  模型：models/yolov8s.om
  设备：0
  分辨率：640x640

时间统计:
  预处理：0.0089 秒
  模型推理：0.0052 秒
  后处理：0.0015 秒
  总时间：0.0156 秒
```

### 批量推理时间统计

批量推理时会统计平均时间和吞吐率：

```bash
python main.py batch ./images --output ./results
```

**输出示例：**
```
批量推理配置:
  模式：multithread
  图像数量：10
  模型：models/yolov8s.om

时间统计:
  平均预处理：0.0085 秒
  平均推理：0.0051 秒
  平均后处理：0.0014 秒
  平均总时间：0.0150 秒

批量统计:
  成功：10/10
  总耗时：0.15 秒
  平均耗时：0.0150 秒/张
  吞吐率：66.67 张/秒
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| model_path | 模型路径 | models/yolov8s.om |
| device_id | 设备 ID | 0 |
| resolution | 分辨率 | 640x640 |
| tile_size | 分块大小 | 640 |
| overlap | 重叠区域 | 100 |
| num_threads | 线程数 | 4 |
| backend | 图像后端 | pil |

## 项目结构

```
AscendInference/
├── config/           # 配置模块
│   ├── __init__.py
│   ├── config.py     # 配置类
│   ├── default.json  # 默认配置
│   └── README.md     # 配置详细说明
├── src/              # 核心推理模块
│   ├── __init__.py
│   ├── inference.py  # 推理类
│   └── api.py        # 统一 API
├── utils/            # 工具函数
│   ├── __init__.py
│   ├── acl_utils.py  # ACL 工具
│   └── profiler.py   # 性能分析
├── tools/            # 辅助工具
│   ├── __init__.py
│   ├── data_generator.py  # 数据生成
│   └── image_enhancer.py  # 图像增强
├── examples/         # 示例代码
│   └── usage_examples.py
├── demo/             # 演示工具
│   └── comprehensive_checker.py
├── main.py           # CLI 入口
└── README.md
```

## 检查工具

运行综合检查工具验证环境：

```bash
python demo/comprehensive_checker.py
```

## 支持的模型

- YOLOv5 (s, n)
- YOLOv8 (s, n)
- YOLOv10 (s, n)

## 环境要求

- Python 3.6+
- Huawei Ascend AI 处理器
- AscendCL (ACL) 库
- NumPy
- PIL (Pillow)
- 可选：OpenCV

## 代码对比

### 重构前

- 2500+ 行代码
- 5 个重复的推理类
- 复杂的 Manager 封装
- 冗余的配置文件
- 分散的脚本入口

### 重构后

- 800 行代码（减少 68%）
- 1 个统一的推理类
- 简洁的工具函数
- 单一配置文件
- 统一的 CLI 入口

## 最佳实践

1. **使用 JSON 文件管理配置** - 便于版本控制和复用
2. **为不同场景创建配置文件** - 如性能、精度、分辨率等
3. **命令行仅用于临时调整** - 测试不同参数时使用
4. **查看时间统计优化性能** - 找出瓶颈所在阶段

## 常见问题

### Q: 配置文件不存在会怎样？
A: 系统会打印警告信息并使用默认配置。

### Q: 如何验证配置是否正确？
A: 运行命令后会显示当前使用的配置信息。

### Q: 如何查看各阶段耗时？
A: 运行推理时会自动显示预处理、推理、后处理的时间统计。

## 许可证

MIT License
