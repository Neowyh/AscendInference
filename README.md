# AscendCL YOLO模型推理示例

本示例展示如何使用AscendCL Python接口执行YOLO模型的OM格式推理。

## 项目结构

```
AscendInference/
├── data/            # 数据目录（存放测试图像、临时文件等）
├── src/             # 源代码目录
│   ├── yolo_inference.py              # 完整工程化版本
│   ├── yolo_inference_fast.py         # 核心功能版本
│   ├── yolo_inference_multithread.py  # 多线程性能版本
│   └── image_enhancer.py              # 图像增强工具
├── scripts/         # 脚本目录（使用场景示例）
│   ├── batch_inference.py             # 批量推理脚本
│   ├── realtime_inference.py          # 实时推理脚本
│   └── benchmark.py                   # 性能测试脚本
└── README.md        # 项目说明
```

## 环境准备

### 1. 安装依赖

```bash
# 安装必要的Python库
pip install numpy Pillow

# 如需使用OpenCV后端
pip install opencv-python

# 确保AscendCL Python包已安装
# 通常位于Ascend SDK的python/site-packages目录
```

### 2. 模型转换流程（从PT到OM）

#### 步骤1: 从PT导出ONNX

```bash
# YOLOv5示例
yolo export model=yolov5s.pt format=onnx

# YOLOv8示例
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx')

# YOLOv10示例
python export.py --weights yolov10s.pt --img-size 640 --batch-size 1
```

#### 步骤2: 使用AMCT进行量化（可选）

```bash
# 使用AMCT进行模型量化
amct_onnx --model=yolov5s.onnx --quantize_mode=calib --calibration_data=calibration_dataset --output_dir=quantized_model
```

#### 步骤3: 使用ATC转换为OM

```bash
# 基本转换命令
atc --model=yolov5s.onnx --framework=5 --output=yolov5s --input_shape="images:1,3,640,640" --soc_version=Ascend310B

# 支持不同分辨率
atc --model=yolov5s.onnx --framework=5 --output=yolov5s_1k --input_shape="images:1,3,1024,1024" --soc_version=Ascend310B
```

### 3. 准备测试图像

准备一张测试图像，例如`test.jpg`

## 使用方法

### 基本用法

```bash
# 运行完整版本（带详细输出）
python src/yolo_inference.py test.jpg

# 运行快速版本（无输出，更高效率）
python src/yolo_inference_fast.py test.jpg

# 运行多线程版本（提高吞吐率）
python src/yolo_inference_multithread.py test1.jpg test2.jpg test3.jpg test4.jpg

# 使用图像增强工具
python src/image_enhancer.py test.jpg

# 批量推理
python scripts/batch_inference.py data/test_images

# 实时推理（摄像头）
python scripts/realtime_inference.py camera

# 性能测试
python scripts/benchmark.py data/test_images

# 高分辨率图像处理
python scripts/high_res_inference.py data/high_res_image.jpg
```

### 高级用法（指定模型和分辨率）

```bash
# 使用YOLOv8模型，1k分辨率
python yolo_inference.py test.jpg --model yolov8s.om --resolution 1k

# 使用YOLOv10模型，4k分辨率
python yolo_inference.py test.jpg --model yolov10n.om --resolution 4k

# 指定设备ID
python yolo_inference.py test.jpg --device 1

# 使用OpenCV作为图像读取后端
python yolo_inference.py test.jpg --backend opencv
```

### 支持的分辨率参数

- `640x640` (默认)
- `1k` (1024x1024)
- `1k2k` (1024x2048)
- `2k` (2048x2048)
- `2k4k` (2048x4096)
- `4k` (4096x4096)
- `4k6k` (4096x6144)
- `3k6k` (3072x6144)
- `6k` (6144x6144)

## 高级工具

### 1. 图像增强工具

**功能**：将输入图像扩充到不同的分辨率，用于测试不同分辨率下的模型性能。

**使用方法**：

```bash
# 基本用法（生成所有分辨率）
python image_enhancer.py test.jpg

# 指定分辨率
python image_enhancer.py test.jpg --resolutions 640x640 1k 2k

# 使用OpenCV后端
python image_enhancer.py test.jpg --backend opencv

# 指定输出目录
python image_enhancer.py test.jpg --output my_enhanced_images
```

### 2. 多线程推理版本

**功能**：使用多线程提高端侧设备吞吐率，支持并行推理。

**特点**：
- 支持多个线程并行推理
- 可指定使用不同的AI核
- 参考昇腾310B的AI核数量（4个）
- 支持批量处理多张图像
- 自动计算推理性能指标

**使用方法**：

```bash
# 基本用法（4线程）
python src/yolo_inference_multithread.py test1.jpg test2.jpg test3.jpg test4.jpg

# 指定线程数
python src/yolo_inference_multithread.py test1.jpg test2.jpg --threads 2

# 使用YOLOv8模型和1k分辨率
python src/yolo_inference_multithread.py test*.jpg --model yolov8s.om --resolution 1k

# 使用OpenCV后端
python src/yolo_inference_multithread.py test.jpg --backend opencv
```

### 3. 高分辨率图像推理

**功能**：处理高分辨率图像（如4k、6k等），通过分块并行处理提高效率。

**特点**：
- 将高分辨率图像划分为带交叉冗余的子块
- 利用多线程和多AI核并行处理子块
- 合并检测结果，避免边缘目标漏检
- 支持大分辨率图像的高效处理
- 可调整子块大小和重叠比例

**使用方法**：

```bash
# 基本用法
python scripts/high_res_inference.py high_res_image.jpg

# 指定子块大小和重叠比例
python scripts/high_res_inference.py high_res_image.jpg --tile-size 640 640 --overlap 0.2

# 使用更多线程
python scripts/high_res_inference.py high_res_image.jpg --threads 4

# 使用OpenCV后端
python scripts/high_res_inference.py high_res_image.jpg --backend opencv
```

## 脚本说明

### 主要功能

1. **初始化ACL**：设置设备、创建上下文和流
2. **加载模型**：从文件加载OM模型，获取模型描述
3. **预处理图像**：调整大小、归一化、格式转换
4. **执行推理**：调用模型执行推理
5. **后处理**：获取并解析输出结果
6. **释放资源**：清理所有分配的资源

### 配置参数

- `MODEL_PATH`：OM模型路径
- `DEVICE_ID`：设备ID
- `INPUT_WIDTH`：输入宽度（默认640）
- `INPUT_HEIGHT`：输入高度（默认640）

### 注意事项

1. 确保Ascend设备已正确安装并配置
2. 确保OM模型与脚本中的输入尺寸匹配
3. 对于不同版本的YOLO模型，可能需要调整后处理逻辑

## 故障排除

- **模型加载失败**：检查模型文件路径和权限
- **内存分配失败**：检查设备内存是否充足
- **推理失败**：检查输入数据格式是否正确

## 参考文档

- [AscendCL开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/aclcppdevg/aclcppdevg_0000.html)
