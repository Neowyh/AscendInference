#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACL工具类

功能：
- 封装ACL初始化和资源管理
- 提供共享的ACL操作功能
- 减少代码冗余
"""

import acl


class AclManager:
    """ACL管理器"""
    
    def __init__(self, device_id=0):
        """初始化ACL管理器
        
        参数：
            device_id: 设备ID
        """
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.initialized = False
    
    def init(self):
        """初始化ACL"""
        if self.initialized:
            return True
        
        # 初始化ACL
        ret = acl.init()
        if ret != 0:
            return False
        
        # 设置设备
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            return False
        
        # 创建上下文
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            return False
        
        # 创建流
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            return False
        
        self.initialized = True
        return True
    
    def get_context(self):
        """获取上下文"""
        return self.context
    
    def get_stream(self):
        """获取流"""
        return self.stream
    
    def get_device_id(self):
        """获取设备ID"""
        return self.device_id
    
    def is_initialized(self):
        """检查是否已初始化"""
        return self.initialized
    
    def destroy(self):
        """销毁资源"""
        if not self.initialized:
            return
        
        # 销毁流
        if self.stream:
            acl.rt.destroy_stream(self.stream)
            self.stream = None
        
        # 销毁上下文
        if self.context:
            acl.rt.destroy_context(self.context)
            self.context = None
        
        # 重置设备
        acl.rt.reset_device(self.device_id)
        
        # 最终化ACL
        acl.finalize()
        
        self.initialized = False


class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_path):
        """初始化模型管理器
        
        参数：
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None
        self.input_size = 0
        self.output_size = 0
        self.loaded = False
    
    def load(self):
        """加载模型"""
        if self.loaded:
            return True
        
        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            return False
        
        # 创建模型描述
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            return False
        
        # 获取输入大小
        if acl.mdl.get_num_inputs(self.model_desc) != 1:
            return False
        self.input_size = acl.mdl.get_input_size_by_index(self.model_desc, 0)
        
        # 获取输出大小
        if acl.mdl.get_num_outputs(self.model_desc) != 1:
            return False
        self.output_size = acl.mdl.get_output_size_by_index(self.model_desc, 0)
        
        self.loaded = True
        return True
    
    def get_model_id(self):
        """获取模型ID"""
        return self.model_id
    
    def get_model_desc(self):
        """获取模型描述"""
        return self.model_desc
    
    def get_input_size(self):
        """获取输入大小"""
        return self.input_size
    
    def get_output_size(self):
        """获取输出大小"""
        return self.output_size
    
    def is_loaded(self):
        """检查模型是否已加载"""
        return self.loaded
    
    def unload(self):
        """卸载模型"""
        if not self.loaded:
            return
        
        # 销毁模型描述
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None
        
        # 卸载模型
        if self.model_id:
            acl.mdl.unload(self.model_id)
            self.model_id = None
        
        self.loaded = False


class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        """初始化内存管理器"""
        self.buffers = []
    
    def malloc_device(self, size):
        """分配设备内存"""
        buffer, ret = acl.rt.malloc(size, acl.rt.MEMORY_DEVICE)
        if ret != 0:
            return None
        self.buffers.append(('device', buffer))
        return buffer
    
    def malloc_host(self, size):
        """分配主机内存"""
        buffer, ret = acl.rt.malloc_host(size)
        if ret != 0:
            return None
        self.buffers.append(('host', buffer))
        return buffer
    
    def free(self, buffer):
        """释放内存"""
        for i, (buf_type, buf) in enumerate(self.buffers):
            if buf == buffer:
                if buf_type == 'device':
                    acl.rt.free(buffer)
                else:
                    acl.rt.free_host(buffer)
                self.buffers.pop(i)
                break
    
    def free_all(self):
        """释放所有内存"""
        for buf_type, buffer in self.buffers:
            if buf_type == 'device':
                acl.rt.free(buffer)
            else:
                acl.rt.free_host(buffer)
        self.buffers.clear()


# 全局ACL管理器实例
_acl_managers = {}


def get_acl_manager(device_id=0):
    """获取ACL管理器实例"""
    if device_id not in _acl_managers:
        _acl_managers[device_id] = AclManager(device_id)
    return _acl_managers[device_id]


def release_acl_managers():
    """释放所有ACL管理器"""
    for manager in _acl_managers.values():
        manager.destroy()
    _acl_managers.clear()
