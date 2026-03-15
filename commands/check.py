#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查命令实现
"""
import sys
import os
from config import Config, SUPPORTED_RESOLUTIONS, MAX_AI_CORES


def cmd_check(args):
    """环境检查命令"""
    print("=" * 60)
    print("昇腾推理项目 - 环境检查")
    print("=" * 60)

    errors = []
    warnings = []
    passed = 0
    has_acl = False

    print("\n[1] Python 环境")
    version = sys.version_info
    print(f"  Python 版本：{version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 6:
        print("  [OK] 版本兼容")
        passed += 1
    else:
        print("  [ERROR] 需要 Python 3.6+")
        errors.append("Python 版本过低")

    print("\n[2] 依赖库")
    required = ['numpy', 'PIL']
    optional = ['cv2']

    for lib in required:
        try:
            __import__(lib)
            print(f"  [OK] {lib} 已安装")
            passed += 1
        except ImportError:
            print(f"  [ERROR] {lib} 未安装")
            errors.append(f"缺少库：{lib}")

    for lib in optional:
        try:
            __import__(lib)
            print(f"  [OK] {lib} 已安装 (可选)")
        except ImportError:
            print(f"  [INFO] {lib} 未安装 (可选)")

    print("\n[3] ACL 库")
    try:
        import acl
        print("  [OK] ACL 库可导入")
        has_acl = True
        passed += 1
    except ImportError:
        print("  [WARNING] ACL 库未找到 (仅在昇腾设备上可用)")
        warnings.append("ACL 库未安装 - 如在非昇腾设备测试可忽略")

    print("\n[4] 配置模块")
    try:
        config = Config()
        print(f"  模型路径：{config.model_path}")
        print(f"  设备 ID: {config.device_id}")
        print(f"  分辨率：{config.resolution}")
        print(f"  AI 核心数：{MAX_AI_CORES}")
        print("  [OK] 配置模块正常")
        passed += 1
    except Exception as e:
        print(f"  [ERROR] 配置模块异常：{e}")
        errors.append(f"配置模块：{e}")

    print("\n[5] 模型文件")
    try:
        config = Config()
        if os.path.exists(config.model_path):
            file_size = os.path.getsize(config.model_path)
            print(f"  [OK] 模型文件存在：{config.model_path}")
            print(f"       文件大小：{file_size / 1024 / 1024:.2f} MB")
            passed += 1
        else:
            print(f"  [WARNING] 模型文件不存在：{config.model_path}")
            warnings.append(f"模型文件缺失：{config.model_path}")
    except Exception as e:
        print(f"  [ERROR] 检查失败：{e}")
        errors.append(f"模型检查：{e}")

    print("\n[6] 推理模块")
    try:
        from src.inference import Inference, MultithreadInference, HighResInference
        print("  [OK] 推理类导入成功")
        print("    - Inference (基础推理)")
        print("    - MultithreadInference (多线程推理)")
        print("    - HighResInference (高分辨率推理)")
        passed += 1
    except Exception as e:
        print(f"  [ERROR] 推理模块导入失败：{e}")
        errors.append(f"推理模块：{e}")

    print("\n[7] API 模块")
    try:
        from src.api import InferenceAPI
        print("  [OK] InferenceAPI 导入成功")
        passed += 1
    except Exception as e:
        print(f"  [ERROR] API 模块导入失败：{e}")
        errors.append(f"API 模块：{e}")

    print("\n[8] 支持的分辨率")
    for res, (w, h) in SUPPORTED_RESOLUTIONS.items():
        print(f"  {res}: {w}x{h}")
    passed += 1

    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    print(f"通过：{passed} 项")

    if errors:
        print(f"\n❌ 错误 ({len(errors)} 项):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")

    if warnings:
        print(f"\n⚠️  警告 ({len(warnings)} 项):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")

    print("\n" + "=" * 60)

    if not errors:
        print("[SUCCESS] ✅ 所有必需检查通过！")
        return 0
    else:
        print(f"[FAILED] ❌ {len(errors)} 项检查失败，请修复错误")
        return 1
