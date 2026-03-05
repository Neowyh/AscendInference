#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理核心类

提供统一的模型推理功能，支持：
- 标准推理
- 多线程推理
- 高分辨率图像分块推理
"""

import os
import sys
import time
import threading
import queue
import ctypes
import numpy as np
from PIL import Image

try:
    import acl
    from utils.acl_utils import (
        init_acl, destroy_acl, 
        load_model, unload_model,
        malloc_device, malloc_host,
        free_device, free_host,
        create_dataset, destroy_dataset,
        get_last_error_msg,
        MEMCPY_HOST_TO_DEVICE, MEMCPY_DEVICE_TO_HOST
    )
    HAS_ACL = True
except ImportError:
    HAS_ACL = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from config import Config
from utils.logger import LoggerConfig

# 获取日志记录器
logger = LoggerConfig.setup_logger('ascend_inference.inference')


class Inference:
    """统一推理类"""
    
    def __init__(self, config=None):
        """初始化推理类
        
        Args:
            config: Config 实例，None 则使用默认配置
        """
        self.config = config or Config()
        self.model_path = self.config.model_path
        self.device_id = self.config.device_id
        self.resolution = self.config.resolution
        self.input_width, self.input_height = Config.get_resolution(self.resolution)
        
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None
        self.input_size = 0
        self.output_size = 0
        self.input_buffer = None
        self.output_buffer = None
        self.output_host = None
        self.input_dataset = None
        self.output_dataset = None
        
        self.initialized = False
        self.model_loaded = False
    
    def init(self, warmup: bool = True, warmup_iterations: int = 3):
        """初始化 ACL 和加载模型
        
        Args:
            warmup: 是否进行模型预热
            warmup_iterations: 预热迭代次数
            
        Returns:
            bool: 是否成功
        """
        if not HAS_ACL:
            logger.error("ACL 库不可用")
            logger.error("提示：ACL 库仅在昇腾设备上可用。如在非昇腾设备上测试，请确保已安装 ACL 库或跳过 ACL 相关功能。")
            return False
        
        self.context, self.stream = init_acl(self.device_id)
        if not self.context:
            logger.error(f"ACL 初始化失败 (device_id={self.device_id})")
            logger.error("可能原因：")
            logger.error("  1. 未在昇腾设备上运行")
            logger.error("  2. 设备 ID 不正确")
            logger.error("  3. ACL 环境未正确配置")
            return False
        
        self.initialized = True
        
        if not self._load_model():
            logger.error("模型加载失败，请检查：")
            logger.error(f"  1. 模型文件是否存在：{self.model_path}")
            logger.error("  2. 模型文件是否损坏")
            self.destroy()
            return False
        
        self.output_host = malloc_host(self.output_size)
        if not self.output_host:
            logger.error("分配主机输出内存失败")
            self.destroy()
            return False
        
        # 模型预热
        if warmup and self.config.enable_logging:
            logger.info(f"模型预热 ({warmup_iterations} 次)...")
            self._warmup(iterations=warmup_iterations)
            if self.config.enable_logging:
                logger.info("模型预热完成")
        
        return True
    
    def _warmup(self, iterations: int = 3):
        """模型预热
        
        Args:
            iterations: 预热迭代次数
        """
        if not HAS_ACL:
            return
        
        # 创建虚拟输入（全零数组）
        dummy_input = np.zeros(
            (self.input_height, self.input_width, 3),
            dtype=np.float32
        )
        
        for i in range(iterations):
            # 预处理
            if not self.preprocess(dummy_input, backend='numpy'):
                print(f"预热第 {i+1} 次预处理失败")
                continue
            
            # 执行推理
            if not self.execute():
                print(f"预热第 {i+1} 次推理失败")
                continue
            
            # 获取结果
            result = self.get_result()
            if result is None:
                print(f"预热第 {i+1} 次获取结果失败")
                continue
        
        return True
    
    def _load_model(self):
        """加载模型"""
        if self.model_loaded:
            return True
        
        if not os.path.exists(self.model_path):
            logger.error(f"模型文件不存在：{self.model_path}")
            return False
        
        result = load_model(self.model_path)
        if result[0] is None:
            logger.error("模型加载失败")
            return False
        
        self.model_id, self.model_desc, self.input_size, self.output_size = result
        
        self.input_buffer = malloc_device(self.input_size)
        if not self.input_buffer:
            logger.error("分配输入设备内存失败")
            return False
        
        self.output_buffer = malloc_device(self.output_size)
        if not self.output_buffer:
            logger.error("分配输出设备内存失败")
            return False
        
        self.input_dataset = create_dataset(self.input_buffer, self.input_size, "输入数据集")
        if not self.input_dataset:
            logger.error("创建输入数据集失败，请检查：")
            logger.error("  1. 输入缓冲区是否分配成功")
            logger.error("  2. 输入数据大小是否正确")
            logger.error("  3. ACL 库是否正常工作")
            return False
        
        self.output_dataset = create_dataset(self.output_buffer, self.output_size, "输出数据集")
        if not self.output_dataset:
            logger.error("创建输出数据集失败，请检查：")
            logger.error("  1. 输出缓冲区是否分配成功")
            logger.error("  2. 输出数据大小是否正确")
            logger.error("  3. ACL 库是否正常工作")
            return False
        
        self.model_loaded = True
        logger.info(f"模型加载成功：{self.model_path}")
        return True
    
    def _load_image(self, image_data, backend='pil'):
        """加载图像
        
        Args:
            image_data: 图像路径或 numpy 数组或 PIL 图像
            backend: 图像读取后端
            
        Returns:
            numpy.ndarray: RGB 图像数组
        """
        if isinstance(image_data, str):
            if not os.path.exists(image_data):
                print(f"图像文件不存在：{image_data}")
                return None
            
            try:
                if backend == 'opencv' and HAS_OPENCV:
                    image = cv2.imread(image_data)
                    if image is None:
                        return None
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = Image.open(image_data)
                    image = np.array(image)
            except Exception as e:
                print(f"读取图像异常：{e}")
                return None
        elif isinstance(image_data, np.ndarray):
            image = image_data
        elif isinstance(image_data, Image.Image):
            image = np.array(image_data)
        else:
            print(f"不支持的图像数据类型：{type(image_data)}")
            return None
        
        return image
    
    def _resize_image(self, image, backend='pil'):
        """调整图像大小"""
        try:
            if backend == 'opencv' and HAS_OPENCV:
                return cv2.resize(image, (self.input_width, self.input_height))
            else:
                image = Image.fromarray(image).resize((self.input_width, self.input_height))
                return np.array(image)
        except Exception as e:
            print(f"调整图像大小异常：{e}")
            return None
    
    def preprocess(self, image_data, backend='pil'):
        """预处理图像
        
        Args:
            image_data: 图像数据
            backend: 图像读取后端
            
        Returns:
            bool: 是否成功
        """
        if not HAS_ACL:
            print("ACL 库不可用")
            return False
        
        image = self._load_image(image_data, backend)
        if image is None:
            return False
        
        image = self._resize_image(image, backend)
        if image is None:
            return False
        
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).flatten()
        
        try:
            input_host = malloc_host(self.input_size)
            if not input_host:
                return False
            
            # 直接拷贝数据到主机内存
            ctypes.memmove(input_host, image.ctypes.data, self.input_size)
            
            ret = acl.rt.memcpy(self.input_buffer, self.input_size, input_host, self.input_size, MEMCPY_HOST_TO_DEVICE)
            free_host(input_host)
            
            if ret != 0:
                err_msg = get_last_error_msg()
                print(f"内存拷贝失败，错误码：{ret}，错误信息：{err_msg}")
                return False
        except Exception as e:
            print(f"预处理异常：{e}")
            return False
        
        return True
    
    def execute(self):
        """执行模型推理
        
        Returns:
            bool: 是否成功
        """
        if not HAS_ACL:
            print("ACL 库不可用")
            return False
        
        if not self.model_loaded:
            print("模型未加载")
            return False
        
        try:
            ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
            
            if ret != 0:
                err_msg = get_last_error_msg()
                print(f"推理执行失败，错误码：{ret}，错误信息：{err_msg}")
                return False
            
            ret = acl.rt.synchronize_stream(self.stream)
            if ret != 0:
                err_msg = get_last_error_msg()
                print(f"Stream 同步失败，错误码：{ret}，错误信息：{err_msg}")
                return False
        except Exception as e:
            print(f"推理执行异常：{e}")
            return False
        
        return True
    
    def get_result(self):
        """获取推理结果
        
        Returns:
            np.ndarray: 推理结果
        """
        if not self.model_loaded or not self.output_host:
            return None
        
        ret = acl.rt.memcpy(self.output_host, self.output_size, self.output_buffer, 
                          self.output_size, MEMCPY_DEVICE_TO_HOST)
        if ret != 0:
            err_msg = get_last_error_msg()
            print(f"获取结果失败，错误码：{ret}，错误信息：{err_msg}")
            return None
        
        # Convert memory address to numpy array using ctypes
        buffer = ctypes.cast(self.output_host, ctypes.POINTER(ctypes.c_float))
        return np.ctypeslib.as_array(buffer, shape=(self.output_size // 4,))
    
    def run_inference(self, image_data, backend='pil'):
        """执行完整推理流程
        
        Args:
            image_data: 图像数据
            backend: 图像读取后端
            
        Returns:
            np.ndarray: 推理结果
        """
        if not self.preprocess(image_data, backend):
            return None
        
        if not self.execute():
            return None
        
        return self.get_result()
    
    def destroy(self):
        """销毁资源"""
        if not HAS_ACL:
            return
        
        if self.stream and self.context:
            try:
                acl.rt.set_context(self.context)
                acl.rt.synchronize_stream(self.stream)
            except Exception as e:
                print(f"警告：流同步失败：{e}")
        
        if self.input_dataset:
            if not destroy_dataset(self.input_dataset, self.context):
                print("警告：输入数据集销毁失败")
            self.input_dataset = None
        
        if self.output_dataset:
            if not destroy_dataset(self.output_dataset, self.context):
                print("警告：输出数据集销毁失败")
            self.output_dataset = None
        
        if self.output_host:
            free_host(self.output_host)
            self.output_host = None
        
        if self.input_buffer:
            free_device(self.input_buffer)
            self.input_buffer = None
        
        if self.output_buffer:
            free_device(self.output_buffer)
            self.output_buffer = None
        
        if self.model_id:
            if not unload_model(self.model_id, self.model_desc):
                print("警告：模型卸载失败")
            self.model_id = None
            self.model_desc = None
        
        if self.initialized:
            if not destroy_acl(self.context, self.stream, self.device_id):
                print("警告：ACL 资源销毁失败")
            self.context = None
            self.stream = None
        
        self.initialized = False
        self.model_loaded = False
    
    def __enter__(self):
        if not self.init():
            raise RuntimeError("初始化失败，请查看上方的详细错误信息")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


class MultithreadInference:
    """多线程推理管理器"""
    
    def __init__(self, config=None):
        """初始化多线程推理
        
        Args:
            config: Config 实例
        """
        self.config = config or Config()
        self.num_threads = min(self.config.num_threads, Config.MAX_AI_CORES)
        self.model_path = self.config.model_path
        self.resolution = self.config.resolution
        
        self.workers = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []
        self.running = False
    
    def _init_workers(self):
        """初始化工作线程"""
        for i in range(self.num_threads):
            device_id = i % Config.MAX_AI_CORES
            config = Config(
                model_path=self.model_path,
                device_id=device_id,
                resolution=self.resolution
            )
            worker = Inference(config)
            if worker.init():
                self.workers.append(worker)
                print(f"Worker {i} 初始化成功 (device: {device_id})")
            else:
                print(f"Worker {i} 初始化失败")
        
        return len(self.workers) > 0
    
    def _worker_thread(self, worker):
        """工作线程函数"""
        if HAS_ACL and worker.context:
            acl.rt.set_context(worker.context)
        
        while self.running:
            try:
                task = self.task_queue.get(block=False)
                if task is None:
                    break
                
                image_path, backend = task
                result = worker.run_inference(image_path, backend)
                self.result_queue.put((image_path, result))
                self.task_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """启动多线程"""
        if not self.workers:
            if not self._init_workers():
                return False
        
        self.running = True
        
        for worker in self.workers:
            thread = threading.Thread(target=self._worker_thread, args=(worker,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        return True
    
    def add_task(self, image_path, backend=None):
        """添加推理任务"""
        if backend is None:
            backend = self.config.backend
        self.task_queue.put((image_path, backend))
    
    def get_results(self):
        """获取推理结果"""
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get(block=False)
                results.append(result)
                self.result_queue.task_done()
            except queue.Empty:
                break
        return results
    
    def wait_completion(self):
        """等待所有任务完成"""
        self.task_queue.join()
        while not self.result_queue.empty():
            time.sleep(0.1)
    
    def stop(self):
        """停止多线程"""
        self.running = False
        
        for thread in self.threads:
            thread.join(timeout=5)
        
        for worker in self.workers:
            worker.destroy()
        
        if HAS_ACL:
            acl.finalize()


def split_image(image, tile_size, overlap):
    """将图像划分为带重叠的子块
    
    Args:
        image: 输入图像
        tile_size: 子块大小
        overlap: 重叠比例
        
    Returns:
        tiles: 子块列表
        positions: 子块位置列表
    """
    h, w = image.shape[:2]
    tile_h, tile_w = tile_size
    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)
    step_h = tile_h - overlap_h
    step_w = tile_w - overlap_w
    
    tiles = []
    positions = []
    
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            x1 = x
            y1 = y
            x2 = min(x + tile_w, w)
            y2 = min(y + tile_h, h)
            
            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)
            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)
            
            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
            positions.append((x1, y1, x2 - x1, y2 - y1))
    
    return tiles, positions


class HighResInference:
    """高分辨率图像推理管理器"""
    
    def __init__(self, config=None):
        """初始化高分辨率推理
        
        Args:
            config: Config 实例
        """
        self.config = config or Config()
        self.num_threads = min(self.config.num_threads, Config.MAX_AI_CORES)
        self.model_path = self.config.model_path
        self.tile_size = (self.config.tile_size, self.config.tile_size)
        self.overlap = self.config.overlap / self.config.tile_size if self.config.overlap > 1 else self.config.overlap
        
        self.multithread = MultithreadInference(
            Config(
                model_path=self.model_path,
                num_threads=self.num_threads,
                resolution=f"{self.tile_size[1]}x{self.tile_size[0]}"
            )
        )
    
    def process_image(self, image_path, backend=None):
        """处理高分辨率图像
        
        Args:
            image_path: 图像路径
            backend: 图像读取后端
            
        Returns:
            dict: 推理结果
        """
        if backend is None:
            backend = self.config.backend
        
        if backend == 'opencv' and HAS_OPENCV:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像：{image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            try:
                image = Image.open(image_path)
                image = np.array(image)
            except Exception as e:
                print(f"无法读取图像：{e}")
                return None
        
        print(f"处理图像：{image_path}, 形状：{image.shape}")
        
        start_time = time.time()
        tiles, positions = split_image(image, self.tile_size, self.overlap)
        print(f"分割完成：{len(tiles)} 个子块，耗时：{time.time() - start_time:.2f} 秒")
        
        if not self.multithread.start():
            print("无法启动推理")
            return None
        
        for i, tile in enumerate(tiles):
            self.multithread.add_task(tile, backend)
        
        self.multithread.wait_completion()
        results = self.multithread.get_results()
        
        merged_result = {
            "sub_results": [],
            "image_shape": image.shape,
            "num_tiles": len(tiles)
        }
        
        for tile_id, result in sorted(results, key=lambda x: x[0]):
            if result is not None:
                x, y, w, h = positions[tile_id]
                merged_result["sub_results"].append({
                    "position": (x, y, w, h),
                    "result": result.tolist()[:10]
                })
        
        print(f"推理完成：{len(merged_result['sub_results'])} 个子块成功")
        
        self.multithread.stop()
        return merged_result
