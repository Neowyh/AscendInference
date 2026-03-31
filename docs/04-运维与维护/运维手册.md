# 昇腾推理工具运维手册

**版本**: 1.1.0  
**日期**: 2026-03-28  
**适用对象**: 运维工程师

---

## 一、部署指南

### 1.1 环境要求

#### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 处理器 | 昇腾310 | 昇腾310/910 |
| 内存 | 8GB | 16GB+ |
| 存储 | 20GB | 100GB+ SSD |
| AI Core | 1个 | 8个+ |

#### 软件要求

| 软件 | 版本要求 |
|------|---------|
| 操作系统 | Ubuntu 18.04+ / CentOS 7+ |
| Python | 3.8+ |
| CANN | 5.0+ |
| ascend-toolkit | 5.0+ |
| numpy | 1.20+ |
| Pillow | 8.0+ |

### 1.2 安装步骤

#### 步骤1: 安装CANN软件栈

```bash
# 下载CANN软件包
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%205.0.4/Ascend-cann-toolkit_5.0.4_linux-x86_64.run

# 安装
chmod +x Ascend-cann-toolkit_5.0.4_linux-x86_64.run
./Ascend-cann-toolkit_5.0.4_linux-x86_64.run --install

# 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 步骤2: 安装Python依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 步骤3: 验证安装

```bash
# 检查环境
python main.py check

# 预期输出
环境检查通过
  - Python版本: 3.8.x
  - ACL可用: 是
  - 设备数量: 1
  - 设备状态: 正常
```

### 1.3 Docker部署

#### Dockerfile

```dockerfile
FROM ubuntu:18.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 复制项目
COPY . /app
WORKDIR /app

# 安装Python依赖
RUN pip3 install -r requirements.txt

# 设置环境变量
ENV ASCEND_HOME=/usr/local/Ascend
ENV LD_LIBRARY_PATH=${ASCEND_HOME}/lib64:$LD_LIBRARY_PATH

# 入口
ENTRYPOINT ["python3", "main.py"]
```

#### 构建和运行

```bash
# 构建镜像
docker build -t ascend-inference:1.1.0 .

# 运行容器
docker run --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/output:/app/output \
    ascend-inference:1.1.0 infer --model /app/models/yolov8s.om --image /app/models/test.jpg
```

---

## 二、配置管理

### 2.1 配置文件位置

| 配置文件 | 用途 |
|---------|------|
| config/default.json | 默认配置 |
| config/extreme.json | 极限性能评测配置 |
| config/production.json | 生产环境配置 |

### 2.2 关键配置项

#### 模型配置

```json
{
    "model_path": "models/yolov8s.om",
    "device_id": 0,
    "resolution": "640x640"
}
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| model_path | 模型文件路径 | 必填 |
| device_id | 设备ID | 0 |
| resolution | 输入分辨率 | 640x640 |

#### 策略配置

```json
{
    "strategies": {
        "multithread": {
            "enabled": true,
            "num_threads": 4,
            "work_stealing": true
        },
        "memory_pool": {
            "enabled": true,
            "pool_size": 10
        }
    }
}
```

#### 评测配置

```json
{
    "benchmark": {
        "iterations": 100,
        "warmup": 5,
        "enable_profiling": true,
        "enable_monitoring": true
    }
}
```

### 2.3 配置验证

```bash
# 验证配置文件
python main.py config --config config/default.json --validate

# 显示当前配置
python main.py config --config config/default.json --show
```

---

## 三、监控与告警

### 3.1 系统监控

#### 资源监控指标

| 指标 | 采集方式 | 告警阈值 |
|------|---------|---------|
| NPU利用率 | acl.rt.get_device_utilization | > 95% 告警 |
| NPU内存 | acl.rt.get_device_memory_info | > 90% 告警 |
| CPU利用率 | psutil.cpu_percent | > 80% 告警 |
| 系统内存 | psutil.virtual_memory | > 85% 告警 |

#### 监控脚本

```python
# scripts/monitor.py
import time
from utils.monitor import ResourceMonitor

monitor = ResourceMonitor(sample_interval=1.0)
monitor.start()

try:
    while True:
        stats = monitor.get_stats()
        
        # 检查告警条件
        if stats.get('npu', {}).get('avg_utilization', 0) > 95:
            send_alert("NPU利用率过高")
        
        if stats.get('memory', {}).get('avg_percent', 0) > 85:
            send_alert("内存使用率过高")
        
        time.sleep(60)
finally:
    monitor.stop()
```

### 3.2 日志管理

#### 日志配置

```python
# config/logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/ascend_inference.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'standard'
        }
    },
    'loggers': {
        'ascend_inference': {
            'level': 'INFO',
            'handlers': ['file']
        }
    }
}
```

#### 日志轮转

```bash
# /etc/logrotate.d/ascend_inference
/var/log/ascend_inference/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

### 3.3 告警配置

#### Prometheus集成

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ascend_inference'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

#### 告警规则

```yaml
# alerts.yml
groups:
  - name: ascend_inference
    rules:
      - alert: HighNPULoad
        expr: npu_utilization > 95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "NPU利用率过高"
          
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "内存使用率过高"
```

---

## 四、故障排查

### 4.1 常见问题

#### 问题1: ACL初始化失败

**错误信息**:
```
ACLError: ACL初始化失败，错误码: 100000
```

**排查步骤**:
```bash
# 1. 检查设备状态
npu-smi info

# 2. 检查驱动版本
cat /usr/local/Ascend/version.info

# 3. 检查环境变量
echo $ASCEND_HOME
echo $LD_LIBRARY_PATH

# 4. 检查设备权限
ls -la /dev/davinci*
```

**解决方案**:
- 确保CANN软件栈正确安装
- 检查环境变量是否配置
- 确保用户有设备访问权限

#### 问题2: 模型加载失败

**错误信息**:
```
ModelLoadError: 模型加载失败: models/yolov8s.om
```

**排查步骤**:
```bash
# 1. 检查模型文件
ls -la models/yolov8s.om

# 2. 检查模型格式
file models/yolov8s.om

# 3. 检查模型大小
du -h models/yolov8s.om
```

**解决方案**:
- 确保模型文件存在且可读
- 检查模型格式是否为OM
- 检查模型是否与CANN版本兼容

#### 问题3: 内存不足

**错误信息**:
```
MemoryError: 设备内存分配失败
```

**排查步骤**:
```bash
# 1. 检查NPU内存使用
npu-smi info -t memory

# 2. 检查系统内存
free -h

# 3. 检查进程内存
ps aux --sort=-%mem | head
```

**解决方案**:
- 减小批处理大小
- 启用内存池策略
- 释放不必要的内存

#### 问题4: 推理超时

**错误信息**:
```
TimeoutError: 推理执行超时
```

**排查步骤**:
```bash
# 1. 检查NPU状态
npu-smi info -t board

# 2. 检查任务队列
# 查看是否有任务堆积

# 3. 检查网络连接（如果使用远程模型）
ping <model_server>
```

**解决方案**:
- 增加超时时间
- 检查NPU是否过载
- 检查网络连接

### 4.2 性能问题排查

#### 排查流程

```
┌─────────────────┐
│  性能问题报告   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  检查资源使用   │────▶│  CPU/NPU/内存   │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  分析性能报告   │────▶│  FPS/延迟分布   │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  定位瓶颈阶段   │────▶│  预处理/推理/后处理│
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  优化建议       │
└─────────────────┘
```

#### 性能分析命令

```bash
# 生成性能报告
python main.py model-bench models/yolov8s.om --images test.jpg \
    --iterations 100 --warmup 5 --output perf_report.txt

# 启用性能分析
python main.py infer --model models/yolov8s.om --image test.jpg \
    --perf-test --enable-profiling
```

---

## 五、备份与恢复

### 5.1 备份策略

| 备份项 | 频率 | 保留时间 |
|--------|------|---------|
| 配置文件 | 每次修改 | 永久 |
| 模型文件 | 每次更新 | 3个版本 |
| 日志文件 | 每日 | 7天 |
| 评测报告 | 每次评测 | 永久 |

### 5.2 备份脚本

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/backup/ascend_inference"
DATE=$(date +%Y%m%d)

# 备份配置
tar -czf ${BACKUP_DIR}/config_${DATE}.tar.gz config/

# 备份模型
tar -czf ${BACKUP_DIR}/models_${DATE}.tar.gz models/

# 备份日志
tar -czf ${BACKUP_DIR}/logs_${DATE}.tar.gz logs/

# 清理旧备份
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +30 -delete
```

### 5.3 恢复步骤

```bash
# 恢复配置
tar -xzf ${BACKUP_DIR}/config_20260328.tar.gz -C /

# 恢复模型
tar -xzf ${BACKUP_DIR}/models_20260328.tar.gz -C /

# 验证恢复
python main.py check
```

---

## 六、安全配置

### 6.1 访问控制

#### 文件权限

```bash
# 设置配置文件权限
chmod 600 config/*.json

# 设置模型文件权限
chmod 644 models/*.om

# 设置日志目录权限
chmod 755 logs/
```

#### 用户权限

```bash
# 创建专用用户
useradd -m -s /bin/bash ascend_user

# 添加到设备组
usermod -a -G HwHiAiUser ascend_user

# 设置目录权限
chown -R ascend_user:ascend_user /app/AscendInference
```

### 6.2 网络安全

#### 防火墙配置

```bash
# 开放必要端口
ufw allow 22/tcp    # SSH
ufw allow 9090/tcp  # Prometheus metrics

# 启用防火墙
ufw enable
```

### 6.3 敏感信息保护

#### 配置文件加密

```bash
# 加密配置文件
openssl enc -aes-256-cbc -salt -in config/production.json -out config/production.json.enc

# 解密配置文件
openssl enc -aes-256-cbc -d -in config/production.json.enc -out config/production.json
```

---

## 七、运维脚本

### 7.1 健康检查脚本

```bash
#!/bin/bash
# scripts/health_check.sh

# 检查服务状态
check_service() {
    if python main.py check > /dev/null 2>&1; then
        echo "服务状态: 正常"
        return 0
    else
        echo "服务状态: 异常"
        return 1
    fi
}

# 检查设备状态
check_device() {
    if npu-smi info > /dev/null 2>&1; then
        echo "设备状态: 正常"
        return 0
    else
        echo "设备状态: 异常"
        return 1
    fi
}

# 检查内存使用
check_memory() {
    MEMORY_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [ $MEMORY_USAGE -gt 90 ]; then
        echo "内存使用: ${MEMORY_USAGE}% (告警)"
        return 1
    else
        echo "内存使用: ${MEMORY_USAGE}% (正常)"
        return 0
    fi
}

# 执行检查
echo "===== 健康检查 $(date) ====="
check_service
check_device
check_memory
echo "============================"
```

### 7.2 日志分析脚本

```bash
#!/bin/bash
# scripts/log_analyze.sh

LOG_FILE="logs/ascend_inference.log"

# 统计错误数量
ERROR_COUNT=$(grep -c "ERROR" $LOG_FILE)
echo "错误数量: $ERROR_COUNT"

# 统计警告数量
WARN_COUNT=$(grep -c "WARN" $LOG_FILE)
echo "警告数量: $WARN_COUNT"

# 显示最近的错误
echo "最近错误:"
grep "ERROR" $LOG_FILE | tail -5

# 显示性能统计
echo "性能统计:"
grep "FPS" $LOG_FILE | tail -5
```

### 7.3 自动化运维

#### Cron定时任务

```bash
# /etc/cron.d/ascend_inference

# 每小时健康检查
0 * * * * ascend_user /app/scripts/health_check.sh >> /var/log/ascend_inference/health.log 2>&1

# 每日日志分析
0 0 * * * ascend_user /app/scripts/log_analyze.sh >> /var/log/ascend_inference/analysis.log 2>&1

# 每周备份
0 2 * * 0 ascend_user /app/scripts/backup.sh >> /var/log/ascend_inference/backup.log 2>&1
```

---

## 八、升级指南

### 8.1 版本升级步骤

```bash
# 1. 备份当前版本
./scripts/backup.sh

# 2. 停止服务
systemctl stop ascend_inference

# 3. 更新代码
git pull origin main

# 4. 更新依赖
pip install -r requirements.txt --upgrade

# 5. 检查配置兼容性
python main.py config --validate

# 6. 运行测试
pytest tests/

# 7. 启动服务
systemctl start ascend_inference

# 8. 验证升级
python main.py check
```

### 8.2 回滚步骤

```bash
# 1. 停止服务
systemctl stop ascend_inference

# 2. 恢复代码
git checkout <previous_version>

# 3. 恢复配置
tar -xzf /backup/ascend_inference/config_<date>.tar.gz -C /

# 4. 启动服务
systemctl start ascend_inference

# 5. 验证回滚
python main.py check
```

---

## 九、联系支持

### 9.1 技术支持

- 项目仓库: <repository_url>
- 问题反馈: GitHub Issues
- 技术文档: docs/

### 9.2 紧急联系

| 问题类型 | 联系方式 |
|---------|---------|
| 系统故障 | <admin_email> |
| 安全问题 | <security_email> |
| 性能问题 | <performance_email> |

---

*文档版本: 1.1.0*
*最后更新: 2026-03-28*
