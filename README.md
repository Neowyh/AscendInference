# AscendInference

基于华为Ascend AI处理器的推理框架，支持YOLO系列模型的高效推理。

## 功能特性

- **多模型支持**：支持YOLOv5、v8、v10等模型
- **多种推理模式**：基础推理、快速推理、多线程并行推理、高分辨率图像推理
- **统一API接口**：提供简单易用的统一接口
- **集中式配置管理**：使用JSON文件管理所有配置参数
- **自动资源管理**：使用上下文管理器自动管理资源
- **详细的日志记录**：完善的错误处理和日志记录

## 目录结构

```
AscendInference/
├── config/            # 配置文件目录
├── src/               # 核心源码
│   ├── base_inference.py          # 推理基类
│   ├── yolo_inference.py          # 基础YOLO推理
│   ├── yolo_inference_fast.py     # 快速YOLO推理
│   ├── yolo_inference_multithread.py  # 多线程YOLO推理
│   ├── yolo_inference_high_res.py     # 高分辨率YOLO推理
│   └── api.py          # 统一API接口
├── utils/             # 工具类
├── examples/          # 示例代码
├── scripts/           # 脚本工具
└── README.md          # 项目说明
```

## 环境要求

- Python 3.6+
- Huawei Ascend AI处理器
- AscendCL (ACL) 库
- NumPy
- PIL (Pillow)
- 可选：OpenCV

## 安装

1. 克隆项目：
   ```bash
   git clone <repository_url>
   cd AscendInference
   ```

2. 安装依赖：
   ```bash
   pip install numpy Pillow
   # 可选
   pip install opencv-python
   ```

3. 配置环境变量：
   ```bash
   export ASCEND_HOME=/usr/local/Ascend
   export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
   ```

## 使用示例

### 1. 使用统一API接口

```python
from src.api import InferenceAPI

# 推理单张图片
results = InferenceAPI.inference_image(
    inference_type='base',
    image_path='test.jpg',
    model_path='models/yolov8s.om',
    device_id=0,
    resolution='640x640'
)
print(results)

# 批量推理
image_paths = ['test1.jpg', 'test2.jpg', 'test3.jpg']
results = InferenceAPI.inference_batch(
    inference_type='multithread',
    image_paths=image_paths,
    model_path='models/yolov8s.om',
    device_id=0,
    resolution='640x640'
)
print(results)
```

### 2. 直接使用推理类

```python
from src.yolo_inference import YOLOInference

# 使用上下文管理器
with YOLOInference(
    model_path='models/yolov8s.om',
    device_id=0,
    resolution='640x640'
) as inference:
    results = inference.inference('test.jpg')
    print(results)
```

## 配置管理

项目使用JSON文件进行配置管理，配置文件位于 `config/default.json`。

### 核心配置参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| model_path | OM模型路径 | 'models/yolov8s.om' |
| device_id | 设备ID | 0 |
| resolution | 输入分辨率 | '640x640' |
| tile_size | 高分辨率推理的分块大小 | 640 |
| overlap | 高分辨率推理的重叠区域 | 100 |
| num_threads | 多线程推理的线程数 | 4 |
| backend | 图像读取后端 | 'pil' |
| conf_threshold | 置信度阈值 | 0.4 |
| iou_threshold | IOU阈值 | 0.5 |

### 配置管理方法

```python
from config import Config

# 获取配置实例
config = Config.get_instance()

# 查看当前配置
print(f"模型路径: {config.model_path}")

# 更新配置
config.update(
    model_path="models/yolov8m.om",
    resolution="1024x1024"
)
```

## 推荐配置文件

- **高性能配置** (`config/high_performance.json`)：适用于需要快速推理的场景
- **高精度配置** (`config/high_accuracy.json`)：适用于对检测精度要求较高的场景
- **高分辨率配置** (`config/high_resolution.json`)：适用于处理大分辨率图片的场景

## 配置文件管理

使用配置管理脚本管理不同的配置文件：

```bash
# 列出所有可用的配置文件
python scripts/config_manager.py list

# 使用指定的配置文件
python scripts/config_manager.py use high_accuracy.json
```

## 性能优化

- **选择合适的推理模式**：根据图片大小和数量选择
- **调整分辨率**：根据模型要求和性能需求调整
- **调整线程数**：根据硬件资源调整
- **使用OpenCV**：加速预处理

## 故障排除

- **模型加载失败**：检查模型路径和格式
- **推理失败**：检查设备ID和ACL配置
- **内存不足**：调整分块大小和线程数
- **性能问题**：使用合适的推理模式和分辨率

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[MIT License](LICENSE)
