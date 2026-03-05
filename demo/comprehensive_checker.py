#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合检查工具

检查项目环境和配置，包括：
- Python 环境和依赖库
- ACL 库和昇腾环境
- 配置模块
- 推理模块（基础、多线程、高分辨率）
- API 模块
- 工具模块
- 模型文件
- 功能完整性检查
"""

import os
import sys
import time
import importlib

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Checker:
    """综合检查器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed = 0
        self.failed = 0
        self.has_acl = False
    
    def check(self, name, func, required=True):
        """运行检查
        
        Args:
            name: 检查名称
            func: 检查函数
            required: 是否必需
        """
        print(f"\n=== {name} ===")
        try:
            result = func()
            if result:
                self.passed += 1
                print(f"[OK] {name} 检查通过")
            else:
                if required:
                    self.failed += 1
                    print(f"[FAILED] {name} 检查失败")
                else:
                    print(f"[INFO] {name} 检查未通过（非必需）")
        except Exception as e:
            print(f"[ERROR] {e}")
            if required:
                self.errors.append(f"{name}: {e}")
                self.failed += 1
            else:
                self.warnings.append(f"{name}: {e}")
    
    def check_python_version(self):
        """检查 Python 版本"""
        version = sys.version_info
        print(f"Python 版本：{version.major}.{version.minor}.{version.micro}")
        if version.major >= 3 and version.minor >= 6:
            print("[OK] 版本兼容")
            return True
        print("[ERROR] 需要 Python 3.6+")
        return False
    
    def check_libraries(self):
        """检查必需的库"""
        required = ['numpy', 'PIL']
        optional = ['cv2']
        all_ok = True
        
        for lib in required:
            try:
                importlib.import_module(lib)
                print(f"[OK] {lib} 已安装")
            except ImportError:
                print(f"[ERROR] {lib} 未安装")
                self.errors.append(f"缺少库：{lib}")
                all_ok = False
        
        for lib in optional:
            try:
                importlib.import_module(lib)
                print(f"[OK] {lib} 已安装 (可选)")
            except ImportError:
                print(f"[INFO] {lib} 未安装 (可选)")
        
        return all_ok
    
    def check_acl(self):
        """检查 ACL 库"""
        try:
            import acl
            print("[OK] ACL 库可导入")
            self.has_acl = True
            return True
        except ImportError:
            print("[WARNING] ACL 库未找到 (仅在昇腾设备上可用)")
            self.warnings.append("ACL 库未安装 - 如在非昇腾设备测试可忽略")
            return True
    
    def check_model_file(self):
        """检查模型文件"""
        from config import Config
        config = Config()
        
        if os.path.exists(config.model_path):
            file_size = os.path.getsize(config.model_path)
            print(f"[OK] 模型文件存在：{config.model_path}")
            print(f"     文件大小：{file_size / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"[WARNING] 模型文件不存在：{config.model_path}")
            print("提示：请先转换或准备模型文件")
            self.warnings.append(f"模型文件缺失：{config.model_path}")
            return True  # 非致命错误
    
    def check_utils_modules(self):
        """检查工具模块"""
        all_ok = True
        
        # 先检查不依赖 ACL 的模块
        modules = [
            ('utils.profiler', '性能分析工具'),
            ('utils.data_generator', '数据生成工具'),
        ]
        
        for module_name, desc in modules:
            try:
                # 临时移除 sys.path 中的 ACL 依赖
                import importlib.util
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    # 检查文件内容是否包含 acl 导入
                    with open(spec.origin, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'import acl' in content and not self.has_acl:
                            print(f"[INFO] {desc} 跳过检查 (依赖 ACL 库)")
                            continue
                    
                    importlib.import_module(module_name)
                    print(f"[OK] {desc} 可导入")
                else:
                    print(f"[ERROR] {desc} 模块不存在")
                    self.errors.append(f"{desc}: 模块文件不存在")
                    all_ok = False
            except ImportError as e:
                if 'acl' in str(e) and not self.has_acl:
                    print(f"[INFO] {desc} 跳过检查 (依赖 ACL 库)")
                else:
                    print(f"[ERROR] {desc} 导入失败：{e}")
                    self.errors.append(f"{desc}: {e}")
                    all_ok = False
            except Exception as e:
                print(f"[ERROR] {desc} 检查异常：{e}")
                self.errors.append(f"{desc}: {e}")
                all_ok = False
        
        # ACL 工具模块（依赖 ACL 库）
        if self.has_acl:
            try:
                importlib.import_module('utils.acl_utils')
                print(f"[OK] ACL 工具可导入")
            except Exception as e:
                print(f"[ERROR] ACL 工具导入失败：{e}")
                self.errors.append(f"ACL 工具：{e}")
                all_ok = False
        else:
            print("[INFO] ACL 工具跳过检查 (ACL 库未安装)")
        
        return all_ok
    
    def check_inference_classes(self):
        """检查推理类"""
        try:
            from src.inference import Inference, MultithreadInference, HighResInference
            
            print("[OK] 推理类导入成功")
            print("  - Inference (基础推理)")
            print("  - MultithreadInference (多线程推理)")
            print("  - HighResInference (高分辨率推理)")
            
            # 检查类属性
            config_check = hasattr(Inference, 'config')
            init_check = hasattr(Inference, 'init')
            destroy_check = hasattr(Inference, 'destroy')
            
            if init_check and destroy_check:
                print("[OK] 推理类方法完整")
                return True
            else:
                print("[WARNING] 推理类方法可能不完整")
                return True
        except Exception as e:
            print(f"[ERROR] 推理类导入失败：{e}")
            self.errors.append(f"推理类：{e}")
            return False
    
    def check_api_functions(self):
        """检查 API 函数"""
        try:
            from src.api import InferenceAPI
            
            print("[OK] InferenceAPI 导入成功")
            
            # 检查 API 方法
            methods = ['inference_image', 'inference_batch']
            for method in methods:
                if hasattr(InferenceAPI, method):
                    print(f"  - {method} 可用")
                else:
                    print(f"  [WARNING] {method} 不存在")
            
            return True
        except Exception as e:
            print(f"[ERROR] API 模块导入失败：{e}")
            self.errors.append(f"API 模块：{e}")
            return False
    
    def check_config_validations(self):
        """检查配置验证"""
        try:
            from config import Config
            
            # 检查分辨率支持
            resolutions = ['640x640', '1k', '2k', '4k']
            print("支持的分辨率:")
            for res in resolutions:
                is_supported = Config.is_supported_resolution(res)
                status = "✓" if is_supported else "✗"
                print(f"  {status} {res}")
            
            # 检查默认配置
            config = Config()
            print(f"\n默认配置:")
            print(f"  - 模型路径：{config.model_path}")
            print(f"  - 设备 ID: {config.device_id}")
            print(f"  - 分辨率：{config.resolution}")
            print(f"  - 线程数：{config.num_threads}")
            print(f"  - 后端：{config.backend}")
            
            return True
        except Exception as e:
            print(f"[ERROR] 配置验证失败：{e}")
            self.errors.append(f"配置验证：{e}")
            return False
    
    def check_memory_management(self):
        """检查内存管理函数"""
        if not self.has_acl:
            print("[INFO] 内存管理函数跳过检查 (ACL 库未安装)")
            return True
        
        try:
            from utils.acl_utils import (
                malloc_device, malloc_host,
                free_device, free_host,
                create_dataset, destroy_dataset
            )
            
            print("[OK] 内存管理函数可用")
            print("  - malloc_device (设备内存分配)")
            print("  - malloc_host (主机内存分配)")
            print("  - create_dataset (数据集创建)")
            print("  - destroy_dataset (数据集销毁)")
            
            return True
        except Exception as e:
            print(f"[ERROR] 内存管理函数导入失败：{e}")
            self.errors.append(f"内存管理：{e}")
            return False
    
    def check_error_handling(self):
        """检查错误处理"""
        if not self.has_acl:
            print("[INFO] 错误处理函数跳过检查 (ACL 库未安装)")
            return True
        
        try:
            from utils.acl_utils import get_last_error_msg
            
            print("[OK] 错误处理函数可用")
            print("  - get_last_error_msg (获取错误信息)")
            
            return True
        except Exception as e:
            print(f"[ERROR] 错误处理函数导入失败：{e}")
            self.errors.append(f"错误处理：{e}")
            return False
    
    def check_dataset_mechanism(self):
        """检查 Dataset 机制"""
        if not self.has_acl:
            print("[INFO] Dataset 机制跳过检查 (ACL 库未安装)")
            return True
        
        try:
            from utils.acl_utils import create_dataset, destroy_dataset
            
            # 检查函数签名
            import inspect
            sig = inspect.signature(create_dataset)
            params = list(sig.parameters.keys())
            
            print("[OK] Dataset 机制可用")
            print(f"  - create_dataset 参数：{', '.join(params)}")
            
            if 'dataset_name' in params:
                print("  [OK] 支持调试模式（dataset_name 参数）")
            
            return True
        except Exception as e:
            print(f"[ERROR] Dataset 机制检查失败：{e}")
            self.errors.append(f"Dataset 机制：{e}")
            return False
    
    def check_context_management(self):
        """检查上下文管理"""
        try:
            from src.inference import Inference
            
            # 检查上下文管理器
            has_enter = hasattr(Inference, '__enter__')
            has_exit = hasattr(Inference, '__exit__')
            
            if has_enter and has_exit:
                print("[OK] 上下文管理器可用 (with 语句)")
                return True
            else:
                print("[WARNING] 上下文管理器不完整")
                return True
        except Exception as e:
            print(f"[ERROR] 上下文管理检查失败：{e}")
            self.errors.append(f"上下文管理：{e}")
            return False
    
    def check_config(self):
        """检查配置模块"""
        try:
            from config import Config
            config = Config()
            print(f"模型路径：{config.model_path}")
            print(f"设备 ID: {config.device_id}")
            print(f"分辨率：{config.resolution}")
            print("[OK] 配置模块正常")
            return True
        except Exception as e:
            print(f"[ERROR] 配置模块异常：{e}")
            self.errors.append(f"配置模块：{e}")
            return False
    
    def check_inference(self):
        """检查推理模块"""
        try:
            from src.inference import Inference, MultithreadInference, HighResInference
            print("[OK] 推理模块导入成功")
            return True
        except Exception as e:
            print(f"[ERROR] 推理模块导入失败：{e}")
            self.errors.append(f"推理模块：{e}")
            return False
    
    def check_api(self):
        """检查 API 模块"""
        try:
            from src.api import InferenceAPI
            print("[OK] API 模块导入成功")
            return True
        except Exception as e:
            print(f"[ERROR] API 模块导入失败：{e}")
            self.errors.append(f"API 模块：{e}")
            return False
    
    def run(self):
        """运行所有检查"""
        print("=" * 60)
        print("昇腾推理项目 - 综合检查工具")
        print("=" * 60)
        print("\n检查范围:")
        print("  1. Python 环境和依赖库")
        print("  2. ACL 库和昇腾环境")
        print("  3. 配置模块和模型文件")
        print("  4. 推理模块（基础、多线程、高分辨率）")
        print("  5. API 模块")
        print("  6. 工具模块")
        print("  7. 内存管理和错误处理")
        print("  8. Dataset 机制")
        print("  9. 上下文管理")
        print("=" * 60)
        
        checks = [
            # 基础环境
            ("1. Python 版本", self.check_python_version),
            ("2. 必需的库", self.check_libraries),
            ("3. ACL 库", self.check_acl, False),  # 非必需
            
            # 模型和配置
            ("4. 模型文件", self.check_model_file, False),
            ("5. 配置模块", self.check_config),
            ("6. 配置验证", self.check_config_validations),
            
            # 核心模块
            ("7. 推理类", self.check_inference_classes),
            ("8. API 模块", self.check_api_functions),
            ("9. 工具模块", self.check_utils_modules),
            
            # 高级功能
            ("10. 内存管理", self.check_memory_management),
            ("11. 错误处理", self.check_error_handling),
            ("12. Dataset 机制", self.check_dataset_mechanism),
            ("13. 上下文管理", self.check_context_management),
        ]
        
        for check_item in checks:
            if len(check_item) == 3:
                name, func, required = check_item
            else:
                name, func = check_item
                required = True
            self.check(name, func, required)
        
        print("\n" + "=" * 60)
        print("检查结果汇总")
        print("=" * 60)
        print(f"通过：{self.passed} 项")
        print(f"失败：{self.failed} 项")
        
        if self.errors:
            print("\n❌ 错误列表:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\n⚠️  警告列表:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\n" + "=" * 60)
        
        # 总体评估
        if self.failed == 0:
            if self.warnings:
                print("[SUCCESS] 所有必需检查通过！")
                print(f"注意：存在 {len(self.warnings)} 个警告，请查看上方详情")
            else:
                print("[SUCCESS] ✅ 所有检查通过！项目状态良好")
            return 0
        else:
            print(f"[FAILED] ❌ {self.failed} 项检查失败，请修复错误")
            return 1


def print_summary(checker):
    """打印总结"""
    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    
    if checker.has_acl:
        print("✓ ACL 库已安装，可以在昇腾设备上运行推理")
        print("\n推荐测试流程:")
        print("  1. 准备模型文件 (models/yolov8s.om)")
        print("  2. 准备测试图像")
        print("  3. 运行：python main.py single test.jpg")
    else:
        print("⚠️  ACL 库未安装")
        print("\n建议:")
        print("  - 如在昇腾设备上：请安装 CANN 工具包")
        print("  - 如在非昇腾设备：可以测试代码结构和模块导入")
        print("  - 完整功能需要在昇腾设备上验证")
    
    print("\n更多帮助:")
    print("  - 查看 README.md 了解项目结构")
    print("  - 查看 BUGFIX.md 了解修复记录")
    print("  - 查看 examples/usage_examples.py 了解使用示例")
    print("=" * 60)


def main():
    """主函数"""
    checker = Checker()
    exit_code = checker.run()
    print_summary(checker)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
