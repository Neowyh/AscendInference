#!/usr/bin/env python3
"""
Ascend YOLO推理项目环境检查工具

这是一个独立的、面向过程的脚本，用于检查目标设备是否满足运行Ascend YOLO推理项目的要求。
它包含全面的错误处理，用于识别测试环境中可能存在的问题。
"""

import os
import sys
import subprocess
import importlib

# 全局变量
REQUIRED_LIBRARIES = ['numpy']  # 必需的库
OPTIONAL_LIBRARIES = ['cv2']    # 可选的库
ASCEND_REQUIREMENTS = {
    'ascendcl': 'ACL库',
    'atc': '模型转换工具',
    'amct': '模型量化工具'
}

def check_python_version():
    """检查Python版本是否兼容"""
    print("=== 检查Python版本 ===")
    try:
        version = sys.version_info
        print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
        if version.major >= 3 and version.minor >= 6:
            print("✓ Python版本兼容")
            return True
        else:
            print("✗ Python版本不兼容。需要Python 3.6或更高版本")
            return False
    except Exception as e:
        print(f"✗ 检查Python版本时出错: {e}")
        return False

def check_libraries():
    """检查必需的库是否已安装"""
    print("\n=== 检查必需库 ===")
    all_installed = True
    
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"✓ {lib} 已安装")
        except ImportError:
            print(f"✗ {lib} 未安装")
            all_installed = False
    
    print("\n=== 检查可选库 ===")
    for lib in OPTIONAL_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"✓ {lib} 已安装")
        except ImportError:
            print(f"⚠ {lib} 未安装 (可选)")
    
    return all_installed

def check_ascend_components():
    """检查Ascend组件是否可用"""
    print("\n=== 检查Ascend组件 ===")
    all_available = True
    
    for component, description in ASCEND_REQUIREMENTS.items():
        try:
            # 尝试运行组件以检查是否可用
            result = subprocess.run(
                [component, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ {description} 可用")
            else:
                print(f"✗ {description} 不可用")
                all_available = False
        except FileNotFoundError:
            print(f"✗ {description} 未在PATH中找到")
            all_available = False
        except Exception as e:
            print(f"⚠ 检查{description}时出错: {e}")
            all_available = False
    
    return all_available

def check_environment_variables():
    """检查必需的环境变量是否已设置"""
    print("\n=== 检查环境变量 ===")
    required_vars = [
        'ASCEND_HOME',
        'ASCEND_VERSION',
        'LD_LIBRARY_PATH'
    ]
    
    all_set = True
    for var in required_vars:
        if var in os.environ:
            print(f"✓ {var} 已设置: {os.environ[var]}")
        else:
            print(f"⚠ {var} 未设置")
            all_set = False
    
    return all_set

def check_device_access():
    """检查Ascend设备是否可访问"""
    print("\n=== 检查Ascend设备访问 ===")
    try:
        # 尝试导入ACL并检查设备
        sys.path.append(os.path.join(os.environ.get('ASCEND_HOME', ''), 'ascend-toolkit', 'latest', 'lib64'))
        from acl import acl
        
        # 初始化ACL
        ret = acl.init()
        if ret == 0:
            print("✓ ACL初始化成功")
            
            # 检查设备数量
            device_count = acl.get_device_count()
            print(f"✓ 发现 {device_count} 个Ascend设备")
            
            if device_count > 0:
                # 尝试打开设备0
                ret = acl.rt.set_device(0)
                if ret == 0:
                    print("✓ 设备0打开成功")
                    
                    # 创建上下文
                    context, ret = acl.rt.create_context(0)
                    if ret == 0:
                        print("✓ 上下文创建成功")
                        acl.rt.destroy_context(context)
                    else:
                        print(f"✗ 创建上下文失败: {ret}")
                    
                    acl.rt.reset_device(0)
                else:
                    print(f"✗ 打开设备0失败: {ret}")
            
            acl.finalize()
            return True
        else:
            print(f"✗ ACL初始化失败: {ret}")
            return False
    except ImportError:
        print("✗ ACL库未找到")
        return False
    except Exception as e:
        print(f"✗ 检查设备访问时出错: {e}")
        return False

def main():
    """主函数，运行所有检查"""
    print("Ascend YOLO推理环境检查工具")
    print("==============================")
    print("此脚本检查您的环境是否满足运行Ascend YOLO推理项目的要求。")
    print()
    
    # 运行所有检查
    checks = [
        ('Python版本', check_python_version),
        ('必需库', check_libraries),
        ('Ascend组件', check_ascend_components),
        ('环境变量', check_environment_variables),
        ('设备访问', check_device_access)
    ]
    
    results = []
    for check_name, check_func in checks:
        results.append(check_func())
    
    # 总结
    print("\n=== 检查结果总结 ===")
    print(f"总检查项: {len(checks)}")
    print(f"通过: {sum(results)}")
    print(f"失败: {len(checks) - sum(results)}")
    
    if all(results):
        print("\n🎉 所有检查通过! 您的环境已准备好运行Ascend YOLO推理。")
        return 0
    else:
        print("\n❌ 部分检查失败。请在运行项目前修复这些问题。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
