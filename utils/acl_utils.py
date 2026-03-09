#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACL 工具函数

提供简化的 ACL 初始化、模型加载和内存管理功能
"""

try:
    import acl
    HAS_ACL = True
except ImportError:
    HAS_ACL = False
    acl = None

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2
MEMCPY_DEVICE_TO_DEVICE = 0

try:
    from utils.logger import LoggerConfig
    logger = LoggerConfig.setup_logger('ascend_inference.acl_utils')
except Exception:
    import logging
    logger = logging.getLogger('ascend_inference.acl_utils')


def init_acl(device_id=0):
    """初始化 ACL 并设置设备
    
    Args:
        device_id: 设备 ID
        
    Returns:
        tuple: (context, stream) 成功返回上下文和流，失败返回 (None, None)
    """
    if not HAS_ACL:
        logger.warning("ACL 库不可用")
        return None, None
    
    ret = acl.init()
    if ret != 0:
        return None, None
    
    ret = acl.rt.set_device(device_id)
    if ret != 0:
        acl.finalize()
        return None, None
    
    context, ret = acl.rt.create_context(device_id)
    if ret != 0:
        acl.rt.reset_device(device_id)
        acl.finalize()
        return None, None
    
    stream, ret = acl.rt.create_stream()
    if ret != 0:
        acl.rt.destroy_context(context)
        acl.rt.reset_device(device_id)
        acl.finalize()
        return None, None
    
    return context, stream


def destroy_acl(context, stream, device_id):
    """销毁 ACL 资源
    
    Args:
        context: 上下文
        stream: 流
        device_id: 设备 ID
        
    Returns:
        bool: 是否成功
    """
    if not HAS_ACL:
        return True
    
    try:
        if context:
            acl.rt.set_context(context)
        
        if stream:
            acl.rt.destroy_stream(stream)
        
        if context:
            acl.rt.destroy_context(context)
        
        acl.rt.reset_device(device_id)
        acl.finalize()
        return True
    except Exception as e:
        logger.error(f"ACL 资源销毁异常：{e}")
        return False


def load_model(model_path):
    """加载模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        tuple: (model_id, model_desc, input_size, output_size) 
               成功返回模型信息，失败返回 (None, None, 0, 0)
    """
    if not HAS_ACL:
        return None, None, 0, 0
    
    model_id, ret = acl.mdl.load_from_file(model_path)
    if ret != 0:
        return None, None, 0, 0
    
    model_desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(model_desc, model_id)
    if ret != 0:
        acl.mdl.unload(model_id)
        return None, None, 0, 0
    
    if acl.mdl.get_num_inputs(model_desc) != 1:
        acl.mdl.destroy_desc(model_desc)
        acl.mdl.unload(model_id)
        return None, None, 0, 0
    
    input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
    
    if acl.mdl.get_num_outputs(model_desc) != 1:
        acl.mdl.destroy_desc(model_desc)
        acl.mdl.unload(model_id)
        return None, None, 0, 0
    
    output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
    
    return model_id, model_desc, input_size, output_size


def unload_model(model_id, model_desc):
    """卸载模型
    
    Args:
        model_id: 模型 ID
        model_desc: 模型描述
        
    Returns:
        bool: 是否成功
    """
    if not HAS_ACL:
        return True
    
    try:
        if model_desc:
            acl.mdl.destroy_desc(model_desc)
        
        if model_id:
            ret = acl.mdl.unload(model_id)
            if ret != 0:
                err_msg = get_last_error_msg()
                logger.error(f"模型卸载失败，错误码：{ret}，错误信息：{err_msg}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"模型卸载异常：{e}")
        return False


def malloc_device(size):
    """分配设备内存
    
    Args:
        size: 内存大小
        
    Returns:
        设备内存指针，失败返回 None
    """
    if not HAS_ACL:
        return None
    
    buffer, ret = acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST)
    if ret != 0:
        return None
    return buffer


def malloc_host(size):
    """分配主机内存
    
    Args:
        size: 内存大小
        
    Returns:
        主机内存指针，失败返回 None
    """
    if not HAS_ACL:
        return None
    
    buffer, ret = acl.rt.malloc_host(size)
    if ret != 0:
        return None
    return buffer


def free_device(buffer):
    """释放设备内存
    
    Args:
        buffer: 设备内存指针
    """
    if not HAS_ACL:
        return
    
    if buffer:
        acl.rt.free(buffer)


def free_host(buffer):
    """释放主机内存
    
    Args:
        buffer: 主机内存指针
    """
    if not HAS_ACL:
        return
    
    if buffer:
        acl.rt.free_host(buffer)


def create_dataset(buffer, size, dataset_name=""):
    """创建数据集
    
    Args:
        buffer: 数据缓冲区指针
        size: 数据大小
        dataset_name: 数据集名称（用于调试）
        
    Returns:
        dataset: 数据集对象，失败返回 None
    """
    if not HAS_ACL:
        return None
    
    try:
        dataset = acl.mdl.create_dataset()
        if dataset is None:
            logger.error(f"{dataset_name}: 创建 dataset 对象失败")
            return None
        
        if buffer is None:
            logger.error(f"{dataset_name}: 缓冲区指针为空")
            acl.mdl.destroy_dataset(dataset)
            return None
        
        if size <= 0:
            logger.error(f"{dataset_name}: 数据大小无效 (size={size})")
            acl.mdl.destroy_dataset(dataset)
            return None
        
        data_buffer = acl.create_data_buffer(buffer, size)
        if data_buffer is None:
            err_msg = get_last_error_msg()
            logger.error(f"{dataset_name}: 创建 data_buffer 失败 (size={size}), 错误：{err_msg}")
            acl.mdl.destroy_dataset(dataset)
            return None
        
        returned_dataset, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
        if returned_dataset != dataset:
            logger.warning(f"{dataset_name}: 返回的 dataset 与输入不一致")
        if ret != 0:
            err_msg = get_last_error_msg()
            logger.error(f"{dataset_name}: 添加 buffer 到 dataset 失败，错误码：{ret}, 错误：{err_msg}")
            acl.destroy_data_buffer(data_buffer)
            acl.mdl.destroy_dataset(dataset)
            return None
        
        return dataset
    except Exception as e:
        logger.error(f"{dataset_name}: 创建 dataset 异常：{e}")
        return None


def destroy_dataset(dataset, context=None):
    """销毁数据集
    
    Args:
        dataset: 数据集对象
        context: 可选的上下文，如果提供则先设置上下文
        
    Returns:
        bool: 是否成功
    """
    if not HAS_ACL:
        return True
    
    if dataset is None:
        return True
    
    try:
        if context:
            acl.rt.set_context(context)
        
        num_buffers = acl.mdl.get_dataset_num_buffers(dataset)
        for i in range(num_buffers):
            data_buffer = acl.mdl.get_dataset_buffer(dataset, i)
            if data_buffer:
                acl.destroy_data_buffer(data_buffer)
        
        acl.mdl.destroy_dataset(dataset)
        return True
    except Exception as e:
        logger.error(f"销毁 dataset 异常：{e}")
        return False


def get_last_error_msg():
    """获取最近的错误信息
    
    Returns:
        str: 错误信息
    """
    if not HAS_ACL:
        return "ACL 库不可用"
    
    try:
        return acl.get_recent_err_msg()
    except:
        return "无法获取错误信息"
