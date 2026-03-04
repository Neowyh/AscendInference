# AscendInference

基于华为Ascend AI处理器的推理框架，支持YOLO系列模型的高效推理。

## 功能特性

- **多模型支持**：支持YOLOv5、v8、v10等模型
- **多种推理模式**：
  - 基础推理模式
  - 快速推理模式
  - 多线程并行推理
  - 高分辨率图像推理（支持图像分块处理）
- **统一API接口**：提供简单易用的统一接口
- **集中式配置管理**：所有配置参数集中管理
- **自动资源管理**：使用上下文管理器自动管理资源
- **详细的日志记录**：提供完善的错误处理和日志记录

## 目录结构

```
AscendInference/
├── config.py          # 集中配置管理
├── src/
│   ├── base_inference.py          # 推理基类
│   ├── yolo_inference.py          # 基础YOLO推理
│   ├── yolo_inference_fast.py     # 快速YOLO推理
│   ├── yolo_inference_multithread.py  # 多线程YOLO推理
│   ├── yolo_inference_high_res.py     # 高分辨率YOLO推理
│   └── api.py          # 统一API接口
├── utils/
│   └── acl_utils.py    # ACL工具类
├── examples/           # 示例代码
└── README.md           # 项目说明
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
   # 设置ACL库路径
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

### 3. 使用多线程推理

```python
from src.yolo_inference_multithread import MultithreadInference

# 使用上下文管理器
with MultithreadInference(
    model_path='models/yolov8s.om',
    device_id=0,
    resolution='640x640',
    num_threads=4
) as inference:
    results = inference.inference('test.jpg')
    print(results)
```

### 4. 使用高分辨率推理

```python
from src.yolo_inference_high_res import HighResInference

# 使用上下文管理器
with HighResInference(
    model_path='models/yolov8s.om',
    device_id=0,
    resolution='640x640',
    tile_size=640,
    overlap=100
) as inference:
    results = inference.inference('high_res_image.jpg')
    print(results)
```

## 配置管理

项目使用JSON文件进行配置管理，配置文件位于 `config/default.json`。

### 配置参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| model_path | OM模型路径 | 'models/yolov8s.om' |
| device_id | 设备ID | 0 |
| resolution | 输入分辨率 | '640x640' |
| tile_size | 高分辨率推理的分块大小 | 640 |
| overlap | 高分辨率推理的重叠区域 | 100 |
| num_threads | 多线程推理的线程数 | 4 |
| backend | 图像读取后端 ('pil' 或 'opencv') | 'pil' |
| conf_threshold | 置信度阈值 | 0.4 |
| iou_threshold | IOU阈值 | 0.5 |
| max_detections | 最大检测数量 | 100 |
| enable_logging | 是否启用日志 | true |
| log_level | 日志级别 | 'info' |
| enable_profiling | 是否启用性能分析 | false |

### 配置文件示例

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

### 配置管理示例

```python
from config import Config

# 获取配置实例
config = Config.get_instance()

# 查看当前配置
print(f"模型路径: {config.model_path}")
print(f"设备ID: {config.device_id}")

# 更新配置
config.update(
    model_path="models/yolov8m.om",
    resolution="1024x1024"
)

# 保存配置到文件（自动完成）
print("配置已更新并保存")
```

## 推荐配置文件

项目提供了几个预定义的配置文件，适用于不同的场景：

### 1. 高性能配置 (`config/high_performance.json`)
- **适用场景**：需要快速推理，对精度要求不高的场景
- **特点**：使用轻量级模型，较低的分辨率，关闭日志，最大线程数
- **配置要点**：
  - 模型：yolov8n.om（最轻量级）
  - 分辨率：640x640
  - 线程数：4
  - 后端：opencv
  - 关闭日志

### 2. 高精度配置 (`config/high_accuracy.json`)
- **适用场景**：对检测精度要求较高的场景
- **特点**：使用较大模型，较高的分辨率，较高的置信度阈值
- **配置要点**：
  - 模型：yolov8l.om（较大模型）
  - 分辨率：1024x1024
  - 线程数：2（减少资源竞争）
  - 置信度阈值：0.5
  - IOU阈值：0.6

### 3. 高分辨率配置 (`config/high_resolution.json`)
- **适用场景**：处理大分辨率图片的场景
- **特点**：优化的分块大小和重叠区域，适合处理高分辨率图像
- **配置要点**：
  - 模型：yolov8m.om（平衡性能和精度）
  - 分辨率：640x640
  - 分块大小：640
  - 重叠区域：150
  - 线程数：4

## 配置文件管理

项目提供了配置文件管理脚本，用于管理和切换不同的配置文件：

### 用法

```bash
# 列出所有可用的配置文件
python scripts/config_manager.py list

# 显示配置文件内容
python scripts/config_manager.py show high_performance.json

# 使用指定的配置文件
python scripts/config_manager.py use high_accuracy.json

# 创建新的配置文件
python scripts/config_manager.py create my_config --model_path models/yolov8s.om --resolution 640x640
```

### 示例

```bash
# 切换到高性能配置
python scripts/config_manager.py use high_performance.json

# 查看当前配置
python scripts/config_manager.py show default.json
```

## 性能优化

1. **选择合适的推理模式**：
   - 小图片：使用基础推理模式
   - 中等大小图片：使用快速推理模式
   - 大量图片：使用多线程推理模式
   - 大分辨率图片：使用高分辨率推理模式

2. **调整分辨率**：根据模型要求和性能需求调整输入分辨率

3. **调整线程数**：根据硬件资源调整多线程推理的线程数

4. **使用OpenCV**：如果安装了OpenCV，预处理速度会更快

## 故障排除

1. **模型加载失败**：
   - 检查模型文件路径是否正确
   - 确保模型是针对Ascend平台优化的OM格式

2. **推理失败**：
   - 检查设备ID是否正确
   - 确保ACL库已正确安装和配置

3. **内存不足**：
   - 对于高分辨率图像，调整tile_size和overlap参数
   - 减少线程数

4. **性能问题**：
   - 检查是否使用了合适的推理模式
   - 确保使用了正确的分辨率

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[MIT License](LICENSE)
