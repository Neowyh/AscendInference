#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理示例

展示如何使用新的配置管理系统，包括：
- 从JSON文件加载配置
- 更新配置参数
- 查看当前配置
- 配置持久化
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
import json
import os

print("===== 配置管理示例 =====")
print()

# 获取配置实例
config = Config.get_instance()

# 查看当前配置
print("当前配置:")
print(f"模型路径: {config.model_path}")
print(f"设备ID: {config.device_id}")
print(f"输入分辨率: {config.resolution}")
print(f"分块大小: {config.tile_size}")
print(f"重叠区域: {config.overlap}")
print(f"线程数: {config.num_threads}")
print()

# 查看配置文件路径
print("配置文件路径:")
print(f"默认配置文件: {Config.DEFAULT_CONFIG_FILE}")
print()

# 检查配置文件是否存在
if os.path.exists(Config.DEFAULT_CONFIG_FILE):
    print("配置文件内容:")
    with open(Config.DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        config_content = json.load(f)
        print(json.dumps(config_content, indent=2, ensure_ascii=False))
else:
    print("配置文件不存在，将使用默认配置")
print()

# 更新配置
print("更新配置...")
new_config = {
    "model_path": "models/yolov8m.om",
    "device_id": 0,
    "resolution": "1024x1024",
    "tile_size": 1024,
    "overlap": 200,
    "num_threads": 4
}

config.update(**new_config)
print("配置更新成功！")
print()

# 查看更新后的配置
print("更新后的配置:")
print(f"模型路径: {config.model_path}")
print(f"设备ID: {config.device_id}")
print(f"输入分辨率: {config.resolution}")
print(f"分块大小: {config.tile_size}")
print(f"重叠区域: {config.overlap}")
print(f"线程数: {config.num_threads}")
print()

# 检查配置文件是否已更新
if os.path.exists(Config.DEFAULT_CONFIG_FILE):
    print("更新后的配置文件内容:")
    with open(Config.DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
        config_content = json.load(f)
        print(json.dumps(config_content, indent=2, ensure_ascii=False))
print()

# 测试获取配置值
print("测试获取配置值:")
model_path = config.get("model_path", "default_model.om")
device_id = config.get("device_id", 0)
print(f"获取model_path: {model_path}")
print(f"获取device_id: {device_id}")
print(f"获取不存在的配置项: {config.get('non_existent', 'default_value')}")
print()

print("===== 示例完成 =====")
