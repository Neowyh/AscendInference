#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目打包命令实现
"""
import os
import zipfile
from pathlib import Path


def cmd_package(args):
    """项目打包命令"""
    script_dir = Path(__file__).parent.parent.resolve()
    output_zip = Path(args.output) if args.output else script_dir / f"{script_dir.name}_packaged.zip"

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
        '.pytest_cache',
        '.mypy_cache',
        '.tox',
        '.venv',
        'venv',
        'env',
        'build',
        'dist',
        '*.egg-info'
    ]

    exclude_dirs = {p for p in exclude_patterns if '*' not in p and not p.startswith('.')}

    print(f"正在打包项目：{script_dir}")
    print(f"输出文件：{output_zip}")
    print()

    file_count = 0
    total_size = 0

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(script_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            for file in files:
                file_path = Path(root) / file

                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern.startswith('*') and file_path.suffix == pattern[1:]:
                        should_exclude = True
                        break
                    elif pattern.startswith('.') and file_path.name.startswith(pattern):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

                rel_path = file_path.relative_to(script_dir.parent)
                zipf.write(file_path, rel_path)

                file_count += 1
                total_size += file_path.stat().st_size

                if file_count <= 20 or file_count % 10 == 0:
                    print(f"  添加：{rel_path}")

    zip_size = output_zip.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0

    print()
    print(f"打包完成!")
    print(f"  文件数量：{file_count}")
    print(f"  原始大小：{total_size / 1024:.2f} KB")
    print(f"  压缩后大小：{zip_size / 1024:.2f} KB")
    print(f"  压缩率：{compression_ratio:.1f}%")
    print(f"  输出文件：{output_zip}")

    return 0
