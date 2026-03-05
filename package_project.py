#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目打包脚本

将项目代码打包为 zip 压缩包，保留一级目录结构
"""

import os
import zipfile
from pathlib import Path


def package_project(source_dir, output_zip, exclude_patterns=None):
    """
    打包项目为 zip 文件
    
    Args:
        source_dir: 源目录路径
        output_zip: 输出 zip 文件路径
        exclude_patterns: 要排除的文件/目录模式列表
    """
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo',
            '.git',
            '.gitignore',
            '*.zip',
            'data',
            '*.log',
            '.DS_Store',
            'Thumbs.db'
        ]
    
    source_path = Path(source_dir).resolve()
    output_path = Path(output_zip).resolve()
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果文件已存在，先删除
    if output_path.exists():
        output_path.unlink()
    
    # 要排除的目录
    exclude_dirs = set()
    for pattern in exclude_patterns:
        if not '*' in pattern and not pattern.startswith('.'):
            exclude_dirs.add(pattern)
    
    print(f"正在打包项目：{source_path}")
    print(f"输出文件：{output_path}")
    print(f"排除目录：{exclude_dirs}")
    
    file_count = 0
    total_size = 0
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_path):
            # 移除要排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # 移除以 . 开头的隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_path = Path(root) / file
                
                # 检查是否应该排除
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern.startswith('*'):
                        if file_path.suffix == pattern[1:]:
                            should_exclude = True
                            break
                    elif pattern.startswith('.'):
                        if file_path.name.startswith(pattern):
                            should_exclude = True
                            break
                
                if should_exclude:
                    continue
                
                # 计算相对路径（保留一级目录）
                rel_path = file_path.relative_to(source_path.parent)
                
                # 添加到 zip
                zipf.write(file_path, rel_path)
                
                file_count += 1
                total_size += file_path.stat().st_size
                print(f"  添加：{rel_path}")
    
    # 打印统计信息
    zip_size = output_path.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print(f"\n打包完成!")
    print(f"  文件数量：{file_count}")
    print(f"  原始大小：{total_size / 1024:.2f} KB")
    print(f"  压缩后大小：{zip_size / 1024:.2f} KB")
    print(f"  压缩率：{compression_ratio:.1f}%")
    print(f"  输出文件：{output_path}")


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = Path(__file__).parent.resolve()
    
    # 项目根目录
    project_dir = script_dir
    
    # 输出 zip 文件
    output_zip = script_dir / f"{script_dir.name}_repackaged.zip"
    
    # 排除模式
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.git',
        '.gitignore',
        '*.zip',
        'data',
        '*.log',
        '.DS_Store',
        'Thumbs.db',
        'AscendInference.zip',  # 旧的压缩包
        'REFACTOR_SUMMARY.md'   # 重构总结文档（可选）
    ]
    
    # 执行打包
    package_project(project_dir, output_zip, exclude_patterns)
    
    print(f"\n✓ 项目已成功打包为：{output_zip}")


if __name__ == "__main__":
    main()
