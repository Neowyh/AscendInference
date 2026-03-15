# AscendInference 项目上下文

## 项目概述
昇腾 AscendCL 模型推理工具，提供高性能的模型推理能力，支持单线程、多线程、高分辨率、批处理和流水线并行推理。

## 核心架构

### 模块职责
- **commands/**: 命令行入口实现（infer/check/enhance/package/config）
- **src/inference.py**: 核心推理类（Inference/MultithreadInference/PipelineInference/HighResInference）
- **src/api.py**: 统一 API 层
- **config/config.py**: 配置管理
- **utils/**: 工具函数（acl_utils/validators/memory_pool/logger/exceptions/profiler）

### 性能优化特性
1. **内存池复用**: MemoryPool 类实现，减少预处理阶段内存分配开销
2. **OpenCV 优化**: 默认使用 OpenCV 进行图像预处理，速度提升 30%+
3. **工作窃取多线程**: MultithreadInference 实现负载均衡
4. **批处理推理**: Inference 类支持 batch_size 参数
5. **流水线并行**: PipelineInference 实现预处理/推理/后处理三阶段重叠
6. **高分辨率分块**: split_image 函数带权重融合消除边缘效应

### 异常体系
所有异常继承自 InferenceError 基类，包含：
- error_code: 错误码
- original_error: 原始异常
- details: 详细上下文字典

错误码范围：
- 2xxx: ACL/设备错误
- 3xxx: 输入验证错误
- 4xxx: 其他错误

### 参数验证
所有公共 API 通过 utils/validators.py 进行参数验证，包括：
- 路径安全验证（防止路径遍历）
- 数值范围验证
- 枚举值验证
- 业务参数验证（分辨率、设备ID、批大小等）

## 关键文件位置

### 推理核心
- [src/inference.py](src/inference.py): 所有推理类实现
- [src/api.py](src/api.py): InferenceAPI 统一接口

### ACL 交互
- [utils/acl_utils.py](utils/acl_utils.py): ACL 初始化、模型加载、内存管理

### 配置
- [config/config.py](config/config.py): Config 类和默认配置
- [config/default.json](config/default.json): JSON 默认配置

### 工具
- [utils/validators.py](utils/validators.py): 参数验证
- [utils/memory_pool.py](utils/memory_pool.py): 内存内存池
- [utils/logger.py](utils/logger.py): 结构化日志
- [utils/exceptions.py](utils/exceptions.py): 异常定义

## 常用开发任务

### 添加新的推理模式
1. 在 src/inference.py 中创建新的推理类
2. 在 src/api.py 的 InferenceAPI 中添加模式分支
3. 在 utils/validators.py 中添加模式验证
4. 在 README.md 中更新文档

### 修改 ACL 交互
- 修改 utils/acl_utils.py 中的函数
- 确保调用前检查 HAS_ACL 标志
- 使用 ACLError 抛出异常并记录 error_code

### 添加新的配置项
1. 在 config/config.py 的 Config 类中添加属性
2. 在 config/default.json 中添加默认值
3. 如需验证，在 utils/validators.py 中添加对应验证函数

### 调试内存问题
- 启用调试模式日志: `Logger.log_with_context(logger, "debug", message, ...)`
- 检查资源泄漏: 查看析构函数警告日志
- 使用内存池: MemoryPool 类自动复用内存

## 重要设计决策

### 为什么使用 OpenCV 作为默认后端
OpenCV 的图像处理速度比 PIL 快 30%+，特别是在 resize 和颜色转换操作上。在推理密集场景下性能差异明显。

### 为什么需要内存池
每次预处理都需要分配主机内存，频繁分配/释放导致性能下降。内存池预分配缓冲区并复用，减少系统调用开销。

### 为什么使用工作窃取算法
简单轮询分配在图像处理时间不均匀时会导致负载失衡。工作窃取让空闲 worker 从其他队列偷取任务，实现动态负载均衡。

### 为什么使用流水线并行
传统串行执行：预处理 → 推理 → 后处理
流水线并行：三个阶段同时处理不同批次的图像，CPU 预处理和 NPU 推理完全重叠。

## 测试和调试

### 运行单元测试
```bash
pytest tests/ -v
```

### 验证日志功能
```bash
python test_logger.py
```

### 检查运行环境
```bash
python main.py check
```

### 调试模式
在代码中使用结构化日志：
```python
from utils.logger import LoggerConfig
LoggerConfig.log_with_context(logger, "info", "推理完成",
    image_path="test.jpg",
    inference_time=0.012,
    status="success"
)
```

## 约束和限制
- 仅支持昇腾设备，不可在其他 GPU 上运行
- 需要安装 ACL 库
- 模型格式必须为 .om（昇腾模型格式）
- 单次推理的图像数量受限于 batch_size 和可用 NPU 内存

## 依赖关系
```
src/api.py → src/inference.py → utils/acl_utils.py
                         → utils/memory_pool.py
                         → utils/logger.py
                         → utils/exceptions.py
                         → utils/validators.py
config/config.py → validators.py
```
