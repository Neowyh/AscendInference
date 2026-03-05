#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存池模块

提供内存复用机制，减少频繁的内存分配和释放，提高性能
"""

from typing import Optional, List
import threading


try:
    from utils.acl_utils import malloc_device, free_device, malloc_host, free_host
    HAS_ACL = True
except ImportError:
    HAS_ACL = False


class MemoryPool:
    """内存池类
    
    功能：
    - 内存复用，减少分配/释放开销
    - 支持设备内存和主机内存
    - 线程安全
    """
    
    def __init__(self, size: int, device: str = 'host', max_buffers: int = 10):
        """初始化内存池
        
        Args:
            size: 每个缓冲区的大小（字节）
            device: 内存类型 ('host' 或 'device')
            max_buffers: 最大缓冲区数量
        """
        self.size = size
        self.device = device
        self.max_buffers = max_buffers
        self.buffers: List[int] = []
        self.free_buffers: List[int] = []
        self.lock = threading.Lock()
        self._allocated_count = 0
    
    def allocate(self) -> Optional[int]:
        """分配内存
        
        Returns:
            内存缓冲区指针，失败返回 None
        """
        with self.lock:
            # 优先从空闲列表分配
            if self.free_buffers:
                buffer = self.free_buffers.pop()
                return buffer
            
            # 检查是否超过最大缓冲区数量
            if len(self.buffers) >= self.max_buffers:
                return None
            
            # 分配新缓冲区
            if self.device == 'device' and HAS_ACL:
                buffer = malloc_device(self.size)
            elif self.device == 'host' and HAS_ACL:
                buffer = malloc_host(self.size)
            else:
                # 非 ACL 环境，返回 None 或使用其他方式
                return None
            
            if buffer:
                self.buffers.append(buffer)
                self._allocated_count += 1
                return buffer
            
            return None
    
    def free(self, buffer: int) -> None:
        """释放内存到池中
        
        Args:
            buffer: 内存缓冲区指针
        """
        if buffer is None:
            return
        
        with self.lock:
            # 回收到空闲列表，而非真正释放
            if buffer not in self.free_buffers:
                self.free_buffers.append(buffer)
    
    def cleanup(self) -> None:
        """清理所有内存"""
        with self.lock:
            # 释放所有缓冲区
            for buffer in self.buffers + self.free_buffers:
                if buffer:
                    if self.device == 'device' and HAS_ACL:
                        free_device(buffer)
                    elif self.device == 'host' and HAS_ACL:
                        free_host(buffer)
            
            self.buffers.clear()
            self.free_buffers.clear()
            self._allocated_count = 0
    
    @property
    def allocated_count(self) -> int:
        """已分配的缓冲区数量"""
        return len(self.buffers)
    
    @property
    def free_count(self) -> int:
        """空闲的缓冲区数量"""
        return len(self.free_buffers)
    
    @property
    def total_count(self) -> int:
        """总缓冲区数量"""
        return len(self.buffers) + len(self.free_buffers)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


class MultiSizeMemoryPool:
    """多尺寸内存池
    
    功能：
    - 支持多种尺寸的内存分配
    - 自动选择合适的内存池
    """
    
    def __init__(self, sizes: List[int], device: str = 'host'):
        """初始化多尺寸内存池
        
        Args:
            sizes: 支持的内存尺寸列表
            device: 内存类型
        """
        self.device = device
        self.pools = {}
        
        for size in sizes:
            self.pools[size] = MemoryPool(size, device)
    
    def allocate(self, size: int) -> Optional[int]:
        """分配指定大小的内存
        
        Args:
            size: 需要的内存大小
            
        Returns:
            内存缓冲区指针
        """
        # 找到第一个大于等于请求尺寸的池
        for pool_size in sorted(self.pools.keys()):
            if pool_size >= size:
                return self.pools[pool_size].allocate()
        
        # 没有合适的池，返回 None
        return None
    
    def free(self, buffer: int, size: int) -> None:
        """释放内存
        
        Args:
            buffer: 内存缓冲区指针
            size: 内存大小
        """
        if size in self.pools:
            self.pools[size].free(buffer)
    
    def cleanup(self) -> None:
        """清理所有内存池"""
        for pool in self.pools.values():
            pool.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
