#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理 API

提供简洁的推理接口
"""

from typing import Optional, List, Any
import numpy as np

try:
    from .inference import Inference, MultithreadInference, HighResInference
    HAS_INFERENCE = True
except ImportError:
    HAS_INFERENCE = False

from config import Config

try:
    from utils.profiler import profile
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False


def _profile_wrapper(name: str):
    """简单的性能分析装饰器（当 profiler 不可用时）
    
    Args:
        name: 性能分析名称
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class InferenceAPI:
    """统一推理 API"""
    
    @staticmethod
    def inference_image(
        mode: str, 
        image_path: str, 
        config: Optional[Config] = None
    ) -> Optional[np.ndarray]:
        """推理单张图片
        
        Args:
            mode: 推理模式 ('base', 'multithread', 'high_res')
            image_path: 图片路径
            config: Config 实例，None 则使用默认配置
            
        Returns:
            推理结果 numpy 数组，失败返回 None
            
        Raises:
            ImportError: 推理模块不可用时
            Exception: 推理过程中出现错误
        """
        if not HAS_INFERENCE:
            raise ImportError("推理模块不可用")
        
        config = config or Config()
        
        if HAS_PROFILER:
            from utils.profiler import profile
            profile_decorator = profile(f"单张图片推理 ({mode})")
        else:
            profile_decorator = _profile_wrapper(f"单张图片推理 ({mode})")
        
        @profile_decorator
        def _inference() -> Optional[np.ndarray]:
            if mode == 'high_res':
                inference = HighResInference(config)
                return inference.process_image(image_path, config.backend)
            
            elif mode == 'multithread':
                inference = MultithreadInference(config)
                if not inference.start():
                    raise Exception("无法启动推理")
                inference.add_task(image_path, config.backend)
                inference.wait_completion()
                results = inference.get_results()
                return results[0][1] if results else None
            
            else:
                inference = Inference(config)
                with inference:
                    return inference.run_inference(image_path, config.backend)
        
        return _inference()
    
    @staticmethod
    def inference_batch(
        mode: str, 
        image_paths: List[str], 
        config: Optional[Config] = None
    ) -> List[Optional[np.ndarray]]:
        """批量推理图片（循环单张推理）
        
        Args:
            mode: 推理模式
            image_paths: 图片路径列表
            config: Config 实例，None 则使用默认配置
            
        Returns:
            推理结果列表，失败项为 None
            
        Raises:
            ImportError: 推理模块不可用时
            Exception: 推理过程中出现错误
        """
        if not HAS_INFERENCE:
            raise ImportError("推理模块不可用")
        
        config = config or Config()
        
        if HAS_PROFILER:
            from utils.profiler import profile
            profile_decorator = profile(f"批量图片推理 ({mode})")
        else:
            profile_decorator = _profile_wrapper(f"批量图片推理 ({mode})")
        
        @profile_decorator
        def _inference() -> List[Optional[np.ndarray]]:
            results: List[Optional[np.ndarray]] = []
            
            if mode == 'high_res':
                inference = HighResInference(config)
                try:
                    for image_path in image_paths:
                        result = inference.process_image(image_path, config.backend)
                        results.append(result)
                finally:
                    inference.multithread.stop()
            
            elif mode == 'multithread':
                inference = MultithreadInference(config)
                try:
                    if not inference.start():
                        raise Exception("无法启动推理")
                    for image_path in image_paths:
                        inference.add_task(image_path, config.backend)
                    inference.wait_completion()
                    results_dict = dict(inference.get_results())
                    results = [results_dict.get(path) for path in image_paths]
                finally:
                    inference.stop()
            
            else:
                inference = Inference(config)
                try:
                    if not inference.init():
                        raise Exception("初始化失败")
                    for image_path in image_paths:
                        result = inference.run_inference(image_path, config.backend)
                        results.append(result)
                finally:
                    inference.destroy()
            
            return results
        
        return _inference()
