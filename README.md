# 昇腾推理项目

精简高效的昇腾 AscendCL 模型推理工具

## 项目特点

- **统一入口**：所有功能通过 `main.py` 统一调用
- **简洁命令**：`infer`、`check`、`enhance`、`package`、`config` 五大命令
- **灵活配置**：JSON 配置文件 + 命令行参数覆盖
- **高性能**：支持多线程和高分辨率图像分块推理
- **易用性**：命令行工具和 Python API 两种使用方式

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

### 统一命令行入口

```bash
# 查看帮助
python main.py --help

# 查看子命令帮助
python main.py infer --help
python main.py check --help
python main.py enhance --help
python main.py package --help
python main.py config --help
```

## 命令详解

### 1. infer - 推理命令

```bash
# 推理单张图片
python main.py infer test.jpg --model models/yolov8s.om

# 使用配置文件
python main.py infer test.jpg --config config/default.json

# 批量推理（输入为目录）
python main.py infer ./images --output ./results

# 多线程推理
python main.py infer test.jpg --mode multithread --threads-per-core 2

# 高分辨率分块推理
python main.py infer large.jpg --mode high_res

# 性能基准测试
python main.py infer test.jpg --benchmark --iterations 100

# 多线程性能测试
python main.py infer test.jpg --test-threads --thread-counts 1 2 4 8

# 分辨率性能测试
python main.py infer test.jpg --test-resolutions
```

### 2. check - 环境检查

```bash
# 检查运行环境
python main.py check
```

检查项目包括：
- Python 版本
- 依赖库
- ACL 库
- 配置模块
- 模型文件
- 推理模块
- API 模块
- 支持的分辨率

### 3. enhance - 图像增强

```bash
# 增强单张图片到所有支持的分辨率
python main.py enhance test.jpg --output ./enhanced

# 指定分辨率
python main.py enhance test.jpg --resolutions 640x640 1k 2k

# 扩增多份
python main.py enhance test.jpg --count 5

# 使用 OpenCV 后端
python main.py enhance test.jpg --backend opencv --interpolation bicubic
```

### 4. package - 项目打包

```bash
# 打包项目
python main.py package

# 指定输出路径
python main.py package --output ./release.zip
```

### 5. config - 配置管理

```bash
# 显示当前配置
python main.py config --show

# 验证配置
python main.py config --validate

# 生成默认配置文件
python main.py config --generate config/my_config.json
```

## Python API 使用

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

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `base` | 标准推理 | 单张图片，需要详细时间统计 |
| `multithread` | 多线程推理 | 批量处理，高性能吞吐 |
| `high_res` | 高分辨率分块推理 | 大图像，超出模型输入尺寸 |

## 配置系统

### 配置优先级

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

### AI 核心数配置

根据设备修改 `config/config.py` 中的 `MAX_AI_CORES`：

| 设备型号 | AI 核心数 | MAX_AI_CORES |
|---------|----------|--------------|
| 昇腾 310P | 4 | 4 |
| 昇腾 310（双芯） | 8 | 8 |
| 昇腾 910 | 32 | 32 |

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
│   ├── logger.py     # 日志系统
│   ├── memory_pool.py # 内存池
│   └── exceptions.py  # 异常定义
├── tools/            # 工具模块（已整合到 main.py）
│   └── __init__.py
├── tests/            # 单元测试
│   ├── __init__.py
│   └── test_all.py
├── main.py           # 统一 CLI 入口
├── requirements.txt  # 依赖
├── pyproject.toml    # 项目配置
└── README.md
```

## 测试

```bash
pip install pytest
pytest tests/ -v
```

## 环境要求

- Python 3.6+
- Huawei Ascend AI 处理器
- AscendCL (ACL) 库
- NumPy
- PIL (Pillow)
- 可选：OpenCV

## 最佳实践

1. **使用 JSON 文件管理配置** - 便于版本控制和复用
2. **为不同场景创建配置文件** - 如性能、精度、分辨率等
3. **命令行仅用于临时调整** - 测试不同参数时使用
4. **使用 check 命令验证环境** - 部署前检查环境配置

## 许可证

MIT License
