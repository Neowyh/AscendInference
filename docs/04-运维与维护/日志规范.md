# 日志使用规范

## 日志级别使用指南

| 级别 | 使用场景 | 示例 |
|------|---------|------|
| DEBUG | 详细调试信息 | 图像尺寸、中间结果、执行流程 |
| INFO | 关键业务节点 | 模型加载成功、推理开始、评测完成 |
| WARNING | 可恢复的异常情况 | 配置使用默认值、性能下降、资源接近上限 |
| ERROR | 需要关注的错误 | 推理失败、资源不足、外部依赖不可用 |
| CRITICAL | 严重错误 | 系统崩溃、数据丢失、无法恢复的错误 |

## 使用原则

### 1. DEBUG 级别
```python
# 用于详细的调试信息
logger.debug(f"图像形状：{image.shape}, 目标尺寸：({width}, {height})")
logger.debug(f"预处理完成，耗时：{elapsed:.2f}ms")
logger.debug(f"推理执行成功，输出大小：{output_size}")
```

### 2. INFO 级别
```python
# 用于关键业务节点
logger.info(f"模型加载成功：{model_path}")
logger.info(f"推理池初始化完成，成功创建 {count} 个实例")
logger.info(f"评测完成，共处理 {total} 张图像")
```

### 3. WARNING 级别
```python
# 用于可恢复的异常情况
logger.warning(f"配置分辨率不在推荐列表中：{resolution}")
logger.warning(f"线程数 ({threads}) 超过最大AI核数 ({max_cores})")
logger.warning(f"资源泄漏检测：实例未正确调用destroy()方法")
```

### 4. ERROR 级别
```python
# 用于需要关注的错误
logger.error(f"模型加载失败：{model_path}")
logger.error(f"推理执行失败，错误码：{error_code}")
logger.error(f"内存分配失败，请求大小：{size}")
```

### 5. CRITICAL 级别
```python
# 用于严重错误（极少使用）
logger.critical(f"系统资源耗尽，无法继续运行")
logger.critical(f"数据损坏，无法恢复：{file_path}")
```

## 异常日志规范

### 使用 logger.exception() 记录异常堆栈
```python
try:
    # 业务逻辑
    pass
except Exception as e:
    logger.exception(f"操作失败: {e}")  # 自动记录堆栈
    raise
```

### 异常信息包含上下文
```python
# 好的做法
logger.error(f"模型加载失败: {model_path}, 原因: {reason}")

# 不好的做法
logger.error("模型加载失败")
```

## 性能考虑

### 避免不必要的字符串格式化
```python
# 好的做法 - 只有当日志级别启用时才格式化
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"详细数据: {expensive_to_compute()}")

# 或者使用惰性格式化（如果日志库支持）
logger.debug("详细数据: %s", lambda: expensive_to_compute())
```

### 批量操作日志
```python
# 批量操作时，使用汇总日志而非逐条日志
logger.info(f"批量处理完成，共 {count} 张图像，成功 {success}，失败 {failed}")

# 而非
for image in images:
    logger.info(f"处理图像: {image}")  # 避免这样做
```

## 敏感信息

### 不要记录敏感信息
```python
# 避免记录敏感信息
# logger.debug(f"用户密码: {password}")  # 错误！
# logger.debug(f"API密钥: {api_key}")    # 错误！

# 可以记录脱敏信息
logger.debug(f"用户ID: {user_id}")
logger.debug(f"API密钥: {api_key[:4]}****")
```

## 结构化日志

### 推荐使用键值对格式
```python
logger.info(
    f"推理完成",
    extra={
        "model": model_name,
        "latency_ms": latency,
        "fps": fps
    }
)
```

## 日志文件配置

### 推荐配置
```python
logging_config = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'inference.log',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    },
}
```
