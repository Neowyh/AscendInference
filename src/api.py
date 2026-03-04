from .base_inference import BaseInference
from .yolo_inference import YOLOInference
from .yolo_inference_fast import YOLOInferenceFast
from .yolo_inference_multithread import MultithreadInference
from .yolo_inference_high_res import HighResInference
from ..config import Config

class InferenceAPI:
    """统一推理API接口"""
    
    @staticmethod
    def create_inference(inference_type, **kwargs):
        """
        创建推理实例
        
        Args:
            inference_type: 推理类型，可选值：'base', 'fast', 'multithread', 'high_res'
            **kwargs: 配置参数，会覆盖默认配置
            
        Returns:
            BaseInference: 推理实例
        """
        # 更新配置
        config = Config.get_instance()
        config.update(kwargs)
        
        # 根据类型创建推理实例
        if inference_type == 'base':
            return YOLOInference()
        elif inference_type == 'fast':
            return YOLOInferenceFast()
        elif inference_type == 'multithread':
            return MultithreadInference()
        elif inference_type == 'high_res':
            return HighResInference()
        else:
            raise ValueError(f"不支持的推理类型: {inference_type}")
    
    @staticmethod
    def inference_image(inference_type, image_path, **kwargs):
        """
        推理单张图片
        
        Args:
            inference_type: 推理类型
            image_path: 图片路径
            **kwargs: 配置参数
            
        Returns:
            list: 检测结果
        """
        with InferenceAPI.create_inference(inference_type, **kwargs) as inference:
            return inference.inference(image_path)
    
    @staticmethod
    def inference_batch(inference_type, image_paths, **kwargs):
        """
        批量推理图片
        
        Args:
            inference_type: 推理类型
            image_paths: 图片路径列表
            **kwargs: 配置参数
            
        Returns:
            list: 检测结果列表
        """
        results = []
        with InferenceAPI.create_inference(inference_type, **kwargs) as inference:
            for image_path in image_paths:
                results.append(inference.inference(image_path))
        return results