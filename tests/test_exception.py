#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常体系验证脚本
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import Inference
from config import Config
from utils.exceptions import (
    InferenceError,
    ModelLoadError,
    DeviceError,
    PreprocessError,
    ACLError,
    InputValidationError
)


def test_model_not_found():
    """测试模型不存在异常"""
    print("\n=== 测试1：模型不存在场景 ===")
    try:
        config = Config(model_path="nonexistent_model.om")
        infer = Inference(config)
        infer.init()
    except ModelLoadError as e:
        print(f"[OK] 正确抛出ModelLoadError")
        print(f"错误信息:\n{e}")
        print(f"错误码: {e.error_code}")
        print(f"包含原始异常: {e.original_error is not None}")
        print(f"详细信息: {e.details}")
        assert "model_path" in e.details
        assert e.details["model_path"] == "nonexistent_model.om"
        return True
    except Exception as e:
        print(f"[ERROR] 错误的异常类型: {type(e).__name__}: {e}")
        return False


def test_image_not_found():
    """测试图像不存在异常"""
    print("\n=== 测试2：图像不存在场景 ===")
    try:
        # 初始化一个mock的推理实例（跳过ACL初始化）
        config = Config()
        infer = Inference(config)
        # 直接调用_load_image方法
        infer._load_image("nonexistent_image.jpg")
    except PreprocessError as e:
        print(f"[OK] 正确抛出PreprocessError")
        print(f"错误信息:\n{e}")
        print(f"错误码: {e.error_code}")
        print(f"详细信息: {e.details}")
        assert "image_path" in e.details
        assert e.details["image_path"] == "nonexistent_image.jpg"
        return True
    except Exception as e:
        print(f"[ERROR] 错误的异常类型: {type(e).__name__}: {e}")
        return False


def test_acl_not_available():
    """测试ACL不可用异常"""
    print("\n=== 测试3：ACL不可用场景 ===")
    try:
        # 模拟ACL不可用
        import src.inference
        original_has_acl = src.inference.HAS_ACL
        src.inference.HAS_ACL = False

        config = Config()
        infer = Inference(config)
        infer.init()
    except ACLError as e:
        print(f"[OK] 正确抛出ACLError")
        print(f"错误信息:\n{e}")
        print(f"错误码: {e.error_code}")
        return True
    except Exception as e:
        print(f"[ERROR] 错误的异常类型: {type(e).__name__}: {e}")
        return False
    finally:
        import src.inference
        src.inference.HAS_ACL = original_has_acl


def test_exception_hierarchy():
    """测试异常继承体系"""
    print("\n=== 测试4：异常继承体系 ===")
    try:
        raise ModelLoadError("测试错误")
    except InferenceError as e:
        print(f"[OK] ModelLoadError正确继承自InferenceError")
        return True
    except Exception as e:
        print(f"[ERROR] 异常体系错误")
        return False


def test_exception_details():
    """测试异常详细信息"""
    print("\n=== 测试5：异常详细信息 ===")
    try:
        raise PreprocessError(
            "预处理失败",
            error_code=2203,
            original_error=RuntimeError("原始错误"),
            details={"image_path": "test.jpg", "backend": "opencv"}
        )
    except PreprocessError as e:
        print(f"[OK] 异常包含所有信息")
        assert "原始错误" in str(e)
        assert "image_path" in e.details
        assert e.error_code == 2203
        print("错误信息格式正确，包含所有详细内容")
        return True


if __name__ == "__main__":
    print("开始验证异常体系...")

    tests = [
        test_exception_hierarchy,
        test_exception_details,
        test_model_not_found,
        test_image_not_found,
        # test_acl_not_available  # 需要在非昇腾环境运行
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"[ERROR] {test.__name__} 测试失败")

    print(f"\n=== 测试结果 ===")
    print(f"总共 {len(tests)} 个测试，通过 {passed} 个，失败 {len(tests) - passed} 个")

    if passed == len(tests):
        print("[OK] 所有测试通过！异常体系工作正常")
    else:
        print("[ERROR] 部分测试失败，请检查异常实现")
        sys.exit(1)
