#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件管理脚本

用于管理和切换不同的配置文件
"""

import os
import sys
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "default.json")


def list_configs():
    """列出所有可用的配置文件"""
    print("===== 可用配置文件 =====")
    config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
    for i, config_file in enumerate(config_files, 1):
        print(f"{i}. {config_file}")
    print()


def show_config(config_file):
    """显示配置文件内容"""
    config_path = os.path.join(CONFIG_DIR, config_file)
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_file}")
        return
    
    print(f"===== 配置文件: {config_file} =====")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print()


def use_config(config_file):
    """使用指定的配置文件"""
    config_path = os.path.join(CONFIG_DIR, config_file)
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_file}")
        return
    
    # 读取指定的配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # 写入默认配置文件
    with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"已切换到配置文件: {config_file}")
    print("配置已更新，下次启动时将使用新配置")
    print()


def create_config(name, **kwargs):
    """创建新的配置文件"""
    config_path = os.path.join(CONFIG_DIR, f"{name}.json")
    if os.path.exists(config_path):
        print(f"配置文件已存在: {name}.json")
        return
    
    # 获取默认配置
    config = Config.get_instance()
    default_config = {
        "model_path": config.model_path,
        "device_id": config.device_id,
        "resolution": config.resolution,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "num_threads": config.num_threads,
        "backend": config.backend,
        "conf_threshold": config.conf_threshold,
        "iou_threshold": config.iou_threshold,
        "max_detections": config.max_detections,
        "enable_logging": config.enable_logging,
        "log_level": config.log_level,
        "enable_profiling": config.enable_profiling
    }
    
    # 更新配置
    default_config.update(kwargs)
    
    # 保存配置文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"已创建配置文件: {name}.json")
    print()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python scripts/config_manager.py list          # 列出所有配置文件")
        print("  python scripts/config_manager.py show <file>  # 显示配置文件内容")
        print("  python scripts/config_manager.py use <file>   # 使用指定的配置文件")
        print("  python scripts/config_manager.py create <name> --param value  # 创建新配置文件")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_configs()
    elif command == "show" and len(sys.argv) == 3:
        show_config(sys.argv[2])
    elif command == "use" and len(sys.argv) == 3:
        use_config(sys.argv[2])
    elif command == "create" and len(sys.argv) >= 3:
        name = sys.argv[2]
        kwargs = {}
        for i in range(3, len(sys.argv), 2):
            if i + 1 < len(sys.argv):
                key = sys.argv[i].lstrip('--')
                value = sys.argv[i+1]
                # 尝试转换类型
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                kwargs[key] = value
        create_config(name, **kwargs)
    else:
        print("无效的命令或参数")


if __name__ == "__main__":
    main()
