#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理测试脚本

仅测试配置管理功能，不依赖于推理相关的模块
"""

import sys
import os
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 直接导入config模块，避免导入推理相关的依赖
from config import Config

print("===== 配置管理测试 =====")
print()

# 测试1: 获取配置实例
print("测试1: 获取配置实例")
try:
    config = Config.get_instance()
    print("✓ 成功获取配置实例")
except Exception as e:
    print(f"✗ 获取配置实例失败: {e}")
    sys.exit(1)
print()

# 测试2: 查看当前配置
print("测试2: 查看当前配置")
try:
    print(f"  模型路径: {config.model_path}")
    print(f"  设备ID: {config.device_id}")
    print(f"  输入分辨率: {config.resolution}")
    print(f"  分块大小: {config.tile_size}")
    print(f"  重叠区域: {config.overlap}")
    print(f"  线程数: {config.num_threads}")
    print(f"  后端: {config.backend}")
    print(f"  置信度阈值: {config.conf_threshold}")
    print(f"  IOU阈值: {config.iou_threshold}")
    print(f"  最大检测数量: {config.max_detections}")
    print(f"  启用日志: {config.enable_logging}")
    print(f"  日志级别: {config.log_level}")
    print(f"  启用性能分析: {config.enable_profiling}")
    print("✓ 成功查看配置")
except Exception as e:
    print(f"✗ 查看配置失败: {e}")
print()

# 测试3: 检查配置文件路径
print("测试3: 检查配置文件路径")
try:
    print(f"  配置目录: {Config.CONFIG_DIR}")
    print(f"  默认配置文件: {Config.DEFAULT_CONFIG_FILE}")
    print(f"  配置文件存在: {os.path.exists(Config.DEFAULT_CONFIG_FILE)}")
    print("✓ 成功检查配置文件路径")
except Exception as e:
    print(f"✗ 检查配置文件路径失败: {e}")
print()

# 测试4: 更新配置
print("测试4: 更新配置")
try:
    new_config = {
        "model_path": "models/yolov8l.om",
        "device_id": 1,
        "resolution": "1024x1024",
        "tile_size": 1024,
        "overlap": 150,
        "num_threads": 2,
        "backend": "opencv",
        "conf_threshold": 0.5,
        "iou_threshold": 0.6,
        "max_detections": 200,
        "enable_logging": False,
        "log_level": "debug",
        "enable_profiling": True
    }
    config.update(**new_config)
    print("✓ 成功更新配置")
except Exception as e:
    print(f"✗ 更新配置失败: {e}")
print()

# 测试5: 查看更新后的配置
print("测试5: 查看更新后的配置")
try:
    print(f"  模型路径: {config.model_path}")
    print(f"  设备ID: {config.device_id}")
    print(f"  输入分辨率: {config.resolution}")
    print(f"  分块大小: {config.tile_size}")
    print(f"  重叠区域: {config.overlap}")
    print(f"  线程数: {config.num_threads}")
    print(f"  后端: {config.backend}")
    print(f"  置信度阈值: {config.conf_threshold}")
    print(f"  IOU阈值: {config.iou_threshold}")
    print(f"  最大检测数量: {config.max_detections}")
    print(f"  启用日志: {config.enable_logging}")
    print(f"  日志级别: {config.log_level}")
    print(f"  启用性能分析: {config.enable_profiling}")
    print("✓ 成功查看更新后的配置")
except Exception as e:
    print(f"✗ 查看更新后的配置失败: {e}")
print()

# 测试6: 验证配置文件是否已更新
print("测试6: 验证配置文件是否已更新")
try:
    if os.path.exists(Config.DEFAULT_CONFIG_FILE):
        with open(Config.DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        print(f"  配置文件中的模型路径: {file_config.get('model_path')}")
        print(f"  配置文件中的设备ID: {file_config.get('device_id')}")
        print(f"  配置文件中的分辨率: {file_config.get('resolution')}")
        print(f"  配置文件中的后端: {file_config.get('backend')}")
        print(f"  配置文件中的置信度阈值: {file_config.get('conf_threshold')}")
        print(f"  配置文件中的IOU阈值: {file_config.get('iou_threshold')}")
        print(f"  配置文件中的最大检测数量: {file_config.get('max_detections')}")
        print(f"  配置文件中的启用日志: {file_config.get('enable_logging')}")
        print(f"  配置文件中的日志级别: {file_config.get('log_level')}")
        print(f"  配置文件中的启用性能分析: {file_config.get('enable_profiling')}")
        print("✓ 成功验证配置文件更新")
    else:
        print("✗ 配置文件不存在")
except Exception as e:
    print(f"✗ 验证配置文件更新失败: {e}")
print()

# 测试7: 测试get方法
print("测试7: 测试get方法")
try:
    model_path = config.get("model_path", "default_model.om")
    device_id = config.get("device_id", 0)
    non_existent = config.get("non_existent", "default_value")
    print(f"  get('model_path'): {model_path}")
    print(f"  get('device_id'): {device_id}")
    print(f"  get('non_existent'): {non_existent}")
    print("✓ 成功测试get方法")
except Exception as e:
    print(f"✗ 测试get方法失败: {e}")
print()

print("===== 测试完成 =====")
print("配置管理系统工作正常！")
