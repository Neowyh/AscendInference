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
pip install -r requirements.txt
```

### 配置环境变量

```bash
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

### 使用命令行工具

```bash
# 推理单张图片
python main.py infer test.jpg --model models/yolov8s.om

# 使用配置文件
python main.py infer test.jpg --config config/default.json

# 性能测试（推理 10 次，统计平均时间和 FPS）
python main.py infer test.jpg --iterations 10

# 批量推理（输入为目录）
python main.py infer ./images --output ./results

# 多线程推理（每个 AI 核心 2 个线程）
python main.py infer test.jpg --mode multithread --threads-per-core 2
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
命令行参数 > JSON 配置文件 > 代码默认值
```

### 使用方式

#### 方式 1：仅使用 JSON 配置文件

```bash
python main.py infer test.jpg --config config/default.json
```

#### 方式 2：JSON 配置 + 命令行参数覆盖

```bash
python main.py infer test.jpg \
    --config config/default.json \
    --device 1 \
    --resolution 1024x1024
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

## 性能统计

运行推理时会自动统计各阶段时间：

```bash
python main.py infer test.jpg --config config/default.json
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

## 项目结构

```
AscendInference/
├── config/           # 配置模块
│   ├── __init__.py
│   ├── config.py     # 配置类
│   └── default.json  # 默认配置
├── src/              # 核心推理模块
│   ├── __init__.py
│   ├── inference.py  # 推理类
│   └── api.py        # 统一 API
├── utils/            # 工具函数
│   ├── __init__.py
│   ├── acl_utils.py  # ACL 工具
│   ├── profiler.py   # 性能分析
│   └── logger.py     # 日志系统
│   └── memory_pool.py # 内存池
├── tools/            # 辅助工具
│   ├── __init__.py
│   ├── data_generator.py  # 数据生成
│   └── image_enhancer.py  # 图像增强
├── tests/            # 单元测试
│   ├── __init__.py
│   ├── test_config.py
│   └── test_logger.py
├── main.py           # CLI 入口
├── requirements.txt  # 依赖
├── pyproject.toml    # 项目配置
└── README.md
```

## 测试

运行单元测试：

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

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
