#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源监控模块

提供系统资源监控功能，支持：
- NPU 利用率监控（昇腾设备）
- 内存使用监控
- CPU 利用率监控
"""

import time
import threading
import platform
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ResourceSnapshot:
    """资源快照"""
    timestamp: float = 0.0
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    npu_utilization: float = 0.0
    npu_memory_used_mb: float = 0.0
    npu_memory_total_mb: float = 0.0


class ResourceMonitor:
    """资源监控器
    
    支持后台线程持续监控系统资源使用情况
    
    Example:
        monitor = ResourceMonitor()
        monitor.start()
        
        # 执行推理任务...
        
        stats = monitor.get_stats()
        monitor.stop()
    """
    
    def __init__(self, sample_interval: float = 0.5, history_size: int = 100):
        """初始化资源监控器
        
        Args:
            sample_interval: 采样间隔（秒）
            history_size: 历史记录大小
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()
        
        self._has_psutil = False
        self._has_npu = False
        
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """检查依赖库"""
        try:
            import psutil
            self._has_psutil = True
        except ImportError:
            pass
        
        try:
            import acl
            self._has_npu = True
        except ImportError:
            pass
    
    def start(self) -> bool:
        """开始监控
        
        Returns:
            bool: 是否成功启动
        """
        if self._running:
            return True
        
        self._running = True
        self._history.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        return True
    
    def stop(self) -> None:
        """停止监控"""
        self._running = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
            self._monitor_thread = None
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running:
            snapshot = self._collect_snapshot()
            
            with self._lock:
                self._history.append(snapshot)
            
            time.sleep(self.sample_interval)
    
    def _collect_snapshot(self) -> ResourceSnapshot:
        """收集资源快照
        
        Returns:
            ResourceSnapshot: 资源快照
        """
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        if self._has_psutil:
            self._collect_cpu_memory(snapshot)
        
        if self._has_npu:
            self._collect_npu(snapshot)
        
        return snapshot
    
    def _collect_cpu_memory(self, snapshot: ResourceSnapshot) -> None:
        """收集 CPU 和内存信息
        
        Args:
            snapshot: 资源快照
        """
        try:
            import psutil
            
            snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            memory = psutil.virtual_memory()
            snapshot.memory_used_mb = memory.used / (1024 * 1024)
            snapshot.memory_total_mb = memory.total / (1024 * 1024)
            snapshot.memory_percent = memory.percent
            
        except Exception:
            pass
    
    def _collect_npu(self, snapshot: ResourceSnapshot) -> None:
        """收集 NPU 信息
        
        Args:
            snapshot: 资源快照
        """
        try:
            import acl
            
            if hasattr(acl, 'rt'):
                if hasattr(acl.rt, 'get_device_utilization'):
                    snapshot.npu_utilization = acl.rt.get_device_utilization(0)
                
                if hasattr(acl.rt, 'get_device_memory_info'):
                    free, total = acl.rt.get_device_memory_info(0)
                    snapshot.npu_memory_total_mb = total / (1024 * 1024)
                    snapshot.npu_memory_used_mb = (total - free) / (1024 * 1024)
                    
        except Exception:
            pass
    
    def get_current_snapshot(self) -> ResourceSnapshot:
        """获取当前资源快照
        
        Returns:
            ResourceSnapshot: 资源快照
        """
        return self._collect_snapshot()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._lock:
            if not self._history:
                return {}
            
            history = list(self._history)
        
        if not history:
            return {}
        
        cpu_values = [s.cpu_percent for s in history if s.cpu_percent > 0]
        memory_values = [s.memory_percent for s in history if s.memory_percent > 0]
        npu_values = [s.npu_utilization for s in history if s.npu_utilization > 0]
        
        stats = {
            'samples': len(history),
            'duration_seconds': history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0,
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0
            },
            'memory': {
                'avg_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max_percent': max(memory_values) if memory_values else 0,
                'current_mb': history[-1].memory_used_mb,
                'total_mb': history[-1].memory_total_mb
            }
        }
        
        if npu_values:
            stats['npu'] = {
                'avg_utilization': sum(npu_values) / len(npu_values),
                'max_utilization': max(npu_values),
                'min_utilization': min(npu_values),
                'current_memory_mb': history[-1].npu_memory_used_mb,
                'total_memory_mb': history[-1].npu_memory_total_mb
            }
        
        return stats
    
    def get_history(self) -> List[ResourceSnapshot]:
        """获取历史记录
        
        Returns:
            list: 历史记录列表
        """
        with self._lock:
            return list(self._history)
    
    def clear_history(self) -> None:
        """清空历史记录"""
        with self._lock:
            self._history.clear()
    
    def is_running(self) -> bool:
        """检查是否正在监控
        
        Returns:
            bool: 是否正在监控
        """
        return self._running
    
    def has_npu_support(self) -> bool:
        """检查是否支持 NPU 监控
        
        Returns:
            bool: 是否支持
        """
        return self._has_npu


class SimpleResourceMonitor:
    """简单资源监控器
    
    不使用后台线程，手动采样
    """
    
    def __init__(self):
        """初始化简单资源监控器"""
        self._has_psutil = False
        self._has_npu = False
        self._samples: List[ResourceSnapshot] = []
        
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """检查依赖库"""
        try:
            import psutil
            self._has_psutil = True
        except ImportError:
            pass
        
        try:
            import acl
            self._has_npu = True
        except ImportError:
            pass
    
    def sample(self) -> ResourceSnapshot:
        """采样一次
        
        Returns:
            ResourceSnapshot: 资源快照
        """
        snapshot = ResourceSnapshot(timestamp=time.time())
        
        if self._has_psutil:
            try:
                import psutil
                
                snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)
                
                memory = psutil.virtual_memory()
                snapshot.memory_used_mb = memory.used / (1024 * 1024)
                snapshot.memory_total_mb = memory.total / (1024 * 1024)
                snapshot.memory_percent = memory.percent
                
            except Exception:
                pass
        
        self._samples.append(snapshot)
        return snapshot
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self._samples:
            return {}
        
        cpu_values = [s.cpu_percent for s in self._samples if s.cpu_percent > 0]
        memory_values = [s.memory_percent for s in self._samples if s.memory_percent > 0]
        
        return {
            'samples': len(self._samples),
            'duration_seconds': self._samples[-1].timestamp - self._samples[0].timestamp if len(self._samples) > 1 else 0,
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0
            },
            'memory': {
                'avg_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
                'max_percent': max(memory_values) if memory_values else 0,
                'current_mb': self._samples[-1].memory_used_mb,
                'total_mb': self._samples[-1].memory_total_mb
            }
        }
    
    def reset(self) -> None:
        """重置"""
        self._samples.clear()


def get_system_info() -> Dict[str, Any]:
    """获取系统信息
    
    Returns:
        Dict: 系统信息
    """
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': None,
        'memory_total_mb': None
    }
    
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_total_mb'] = psutil.virtual_memory().total / (1024 * 1024)
    except ImportError:
        pass
    
    try:
        import acl
        info['has_npu'] = True
        info['acl_version'] = getattr(acl, '__version__', 'unknown')
    except ImportError:
        info['has_npu'] = False
    
    return info
