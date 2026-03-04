#!/usr/bin/env python3
"""
Environment checker demo for Ascend YOLO inference project

This is a self-contained, process-oriented script that checks if the target device
meets the requirements for running the Ascend YOLO inference project. It includes
comprehensive error handling to identify potential issues in the test environment.
"""

import os
import sys
import subprocess
import importlib

# Global variables
REQUIRED_LIBRARIES = ['numpy']
OPTIONAL_LIBRARIES = ['cv2']
ASCEND_REQUIREMENTS = {
    'ascendcl': 'ACL library',
    'atc': 'Model conversion tool',
    'amct': 'Model quantization tool'
}

def check_python_version():
    """Check if Python version is compatible"""
    print("=== Checking Python version ===")
    try:
        version = sys.version_info
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        if version.major >= 3 and version.minor >= 6:
            print("✓ Python version is compatible")
            return True
        else:
            print("✗ Python version is incompatible. Requires Python 3.6 or higher")
            return False
    except Exception as e:
        print(f"✗ Error checking Python version: {e}")
        return False

def check_libraries():
    """Check if required libraries are installed"""
    print("\n=== Checking required libraries ===")
    all_installed = True
    
    for lib in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"✓ {lib} is installed")
        except ImportError:
            print(f"✗ {lib} is not installed")
            all_installed = False
    
    print("\n=== Checking optional libraries ===")
    for lib in OPTIONAL_LIBRARIES:
        try:
            importlib.import_module(lib)
            print(f"✓ {lib} is installed")
        except ImportError:
            print(f"⚠ {lib} is not installed (optional)")
    
    return all_installed

def check_ascend_components():
    """Check if Ascend components are available"""
    print("\n=== Checking Ascend components ===")
    all_available = True
    
    for component, description in ASCEND_REQUIREMENTS.items():
        try:
            # Try to run the component to check if it's available
            result = subprocess.run(
                [component, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ {description} is available")
            else:
                print(f"✗ {description} is not available")
                all_available = False
        except FileNotFoundError:
            print(f"✗ {description} is not found in PATH")
            all_available = False
        except Exception as e:
            print(f"⚠ Error checking {description}: {e}")
            all_available = False
    
    return all_available

def check_environment_variables():
    """Check if required environment variables are set"""
    print("\n=== Checking environment variables ===")
    required_vars = [
        'ASCEND_HOME',
        'ASCEND_VERSION',
        'LD_LIBRARY_PATH'
    ]
    
    all_set = True
    for var in required_vars:
        if var in os.environ:
            print(f"✓ {var} is set: {os.environ[var]}")
        else:
            print(f"⚠ {var} is not set")
            all_set = False
    
    return all_set

def check_device_access():
    """Check if Ascend device is accessible"""
    print("\n=== Checking Ascend device access ===")
    try:
        # Try to import ACL and check device
        sys.path.append(os.path.join(os.environ.get('ASCEND_HOME', ''), 'ascend-toolkit', 'latest', 'lib64'))
        from acl import acl
        
        # Initialize ACL
        ret = acl.init()
        if ret == 0:
            print("✓ ACL initialized successfully")
            
            # Check device count
            device_count = acl.get_device_count()
            print(f"✓ Found {device_count} Ascend device(s)")
            
            if device_count > 0:
                # Try to open device 0
                ret = acl.rt.set_device(0)
                if ret == 0:
                    print("✓ Device 0 opened successfully")
                    
                    # Create context
                    context, ret = acl.rt.create_context(0)
                    if ret == 0:
                        print("✓ Context created successfully")
                        acl.rt.destroy_context(context)
                    else:
                        print(f"✗ Failed to create context: {ret}")
                    
                    acl.rt.reset_device(0)
                else:
                    print(f"✗ Failed to open device 0: {ret}")
            
            acl.finalize()
            return True
        else:
            print(f"✗ Failed to initialize ACL: {ret}")
            return False
    except ImportError:
        print("✗ ACL library not found")
        return False
    except Exception as e:
        print(f"✗ Error checking device access: {e}")
        return False

def main():
    """Main function to run all checks"""
    print("Ascend YOLO Inference Environment Checker")
    print("========================================")
    print("This script checks if your environment meets the requirements for running")
    print("the Ascend YOLO inference project.")
    print()
    
    # Run all checks
    checks = [
        ('Python version', check_python_version),
        ('Required libraries', check_libraries),
        ('Ascend components', check_ascend_components),
        ('Environment variables', check_environment_variables),
        ('Device access', check_device_access)
    ]
    
    results = []
    for check_name, check_func in checks:
        results.append(check_func())
    
    # Summary
    print("\n=== Summary ===")
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(checks) - sum(results)}")
    
    if all(results):
        print("\n🎉 All checks passed! Your environment is ready for Ascend YOLO inference.")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues before running the project.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
