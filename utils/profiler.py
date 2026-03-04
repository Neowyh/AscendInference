#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能评测工具

用于测量推理时间和性能指标
"""

import time
import statistics
from typing import List, Dict, Any


class Profiler:
    """性能评测器"""
    
    def __init__(self, enable=False):
        """初始化性能评测器
        
        Args:
            enable: 是否启用性能评测
        """
        self.enable = enable
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0
        self.timings = []
    
    def start(self):
        """开始计时"""
        if self.enable:
            self.start_time = time.time()
    
    def stop(self):
        """停止计时"""
        if self.enable:
            self.end_time = time.time()
            self.elapsed_time = self.end_time - self.start_time
            self.timings.append(self.elapsed_time)
    
    def get_elapsed_time(self):
        """获取已用时间
        
        Returns:
            float: 已用时间（秒）
        """
        return self.elapsed_time
    
    def get_average_time(self):
        """获取平均时间
        
        Returns:
            float: 平均时间（秒）
        """
        if not self.timings:
            return 0
        return statistics.mean(self.timings)
    
    def get_total_time(self):
        """获取总时间
        
        Returns:
            float: 总时间（秒）
        """
        return sum(self.timings)
    
    def get_min_time(self):
        """获取最小时间
        
        Returns:
            float: 最小时间（秒）
        """
        if not self.timings:
            return 0
        return min(self.timings)
    
    def get_max_time(self):
        """获取最大时间
        
        Returns:
            float: 最大时间（秒）
        """
        if not self.timings:
            return 0
        return max(self.timings)
    
    def get_std_dev(self):
        """获取标准差
        
        Returns:
            float: 标准差
        """
        if len(self.timings) < 2:
            return 0
        return statistics.stdev(self.timings)
    
    def reset(self):
        """重置评测器"""
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0
        self.timings = []
    
    def get_stats(self):
        """获取性能统计信息
        
        Returns:
            dict: 性能统计信息
        """
        if not self.timings:
            return {}
        
        return {
            "total_time": self.get_total_time(),
            "average_time": self.get_average_time(),
            "min_time": self.get_min_time(),
            "max_time": self.get_max_time(),
            "std_dev": self.get_std_dev(),
            "count": len(self.timings)
        }
    
    def print_stats(self, title="性能评测结果"):
        """打印性能统计信息
        
        Args:
            title: 标题
        """
        if not self.enable or not self.timings:
            return
        
        stats = self.get_stats()
        print(f"\n===== {title} =====")
        print(f"总时间: {stats['total_time']:.4f} 秒")
        print(f"平均时间: {stats['average_time']:.4f} 秒")
        print(f"最小时间: {stats['min_time']:.4f} 秒")
        print(f"最大时间: {stats['max_time']:.4f} 秒")
        print(f"标准差: {stats['std_dev']:.4f} 秒")
        print(f"推理次数: {stats['count']}")
        if stats['count'] > 0:
            print(f"吞吐率: {stats['count'] / stats['total_time']:.2f} 张/秒")
        print("=====================\n")


def profile_inference(func):
    """推理函数装饰器，用于测量推理时间
    
    Args:
        func: 推理函数
        
    Returns:
        function: 装饰后的函数
    """
    def wrapper(*args, **kwargs):
        profiler = Profiler(enable=True)
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        profiler.print_stats(f"{func.__name__} 性能评测")
        return result
    return wrapper
