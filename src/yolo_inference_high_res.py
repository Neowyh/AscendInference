#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高分辨率图像推理优化脚本

功能：
- 将高分辨率图像划分为带交叉冗余的子块
- 利用多线程和多AI核并行处理子块
- 合并检测结果，避免边缘目标漏检
- 支持大分辨率图像的高效处理
"""

import os
import sys
import numpy as np
from PIL import Image
import acl
import threading
import queue
import time

# 导入配置和基类
from config import Config
from base_inference import BaseInference

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class InferenceWorker(BaseInference):
    """推理工作线程"""
    
    def __init__(self, worker_id, device_id, model_path, resolution):
        """初始化工作线程
        
        Args:
            worker_id: 工作线程ID
            device_id: 设备ID
            model_path: 模型路径
            resolution: 输入分辨率
        """
        super().__init__(model_path, device_id, resolution)
        self.worker_id = worker_id
        self.output_host = None
    
    def init(self):
        """初始化 worker"""
        # 初始化ACL
        if not self.init_acl():
            return False
        
        # 加载模型
        if not self.load_model():
            return False
        
        # 预分配主机内存用于输出
        output_size = self.model_manager.get_output_size()
        self.output_host = self.memory_manager.malloc_host(output_size)
        if not self.output_host:
            return False
        
        return True
    
    def preprocess(self, image):
        """预处理图像
        
        Args:
            image: 图像数据
            
        Returns:
            bool: 预处理是否成功
        """
        # 调整大小
        if HAS_OPENCV:
            image = cv2.resize(image, (self.input_width, self.input_height))
        else:
            image = Image.fromarray(image).resize((self.input_width, self.input_height))
            image = np.array(image)
        
        # 处理灰度图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # 归一化和通道转换
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).flatten()
        
        # 分配临时主机内存
        input_size = self.model_manager.get_input_size()
        input_host, ret = acl.rt.malloc_host(input_size)
        if ret != 0:
            return False
        
        # 复制数据
        acl.util.vector_to_ptr(image.tobytes(), input_host, input_size)
        
        # 复制到设备
        ret = acl.rt.memcpy(self.input_buffer, input_size, input_host, input_size, acl.rt.MEMCPY_HOST_TO_DEVICE)
        
        # 释放临时内存
        acl.rt.free_host(input_host)
        
        return ret == 0
    
    def get_result(self):
        """获取结果"""
        if not self.model_loaded or not self.output_host:
            return None
        
        output_size = self.model_manager.get_output_size()
        if acl.rt.memcpy(self.output_host, output_size, self.output_buffer, output_size, acl.rt.MEMCPY_DEVICE_TO_HOST) != 0:
            return None
        
        return np.frombuffer(self.output_host, dtype=np.float32)


def split_image(image, tile_size, overlap):
    """
    将图像划分为带重叠的子块
    
    Args:
        image: 输入图像 (H, W, C)
        tile_size: 子块大小 (H, W)
        overlap: 重叠比例
        
    Returns:
        tiles: 子块列表
        positions: 子块在原图中的位置列表 (x, y, w, h)
    """
    h, w = image.shape[:2]
    tile_h, tile_w = tile_size
    
    # 计算重叠像素数
    overlap_h = int(tile_h * overlap)
    overlap_w = int(tile_w * overlap)
    
    # 计算步长
    step_h = tile_h - overlap_h
    step_w = tile_w - overlap_w
    
    tiles = []
    positions = []
    
    # 遍历图像，生成子块
    for y in range(0, h, step_h):
        for x in range(0, w, step_w):
            # 计算子块边界
            x1 = x
            y1 = y
            x2 = min(x + tile_w, w)
            y2 = min(y + tile_h, h)
            
            # 处理边界情况
            if x2 - x1 < tile_w:
                x1 = max(0, x2 - tile_w)
            if y2 - y1 < tile_h:
                y1 = max(0, y2 - tile_h)
            
            # 提取子块
            tile = image[y1:y2, x1:x2]
            tiles.append(tile)
            positions.append((x1, y1, x2 - x1, y2 - y1))
    
    return tiles, positions

def merge_results(results, positions, image_shape):
    """
    合并子块的推理结果
    
    Args:
        results: 子块推理结果列表
        positions: 子块位置列表
        image_shape: 原图形状
        
    Returns:
        merged_result: 合并后的结果
    """
    # 这里简化处理，实际应用中需要根据模型输出格式进行后处理
    # 包括坐标映射、NMS等操作
    merged_result = {
        "sub_results": [],
        "image_shape": image_shape,
        "num_tiles": len(results)
    }
    
    for i, (result, (x, y, w, h)) in enumerate(zip(results, positions)):
        if result is not None:
            merged_result["sub_results"].append({
                "position": (x, y, w, h),
                "result": result.tolist()[:10]  # 只保存部分结果
            })
    
    return merged_result


class HighResInference:
    """高分辨率图像推理管理器"""
    
    def __init__(self, model_path, num_threads=None, tile_size=None, overlap=None):
        """初始化高分辨率图像推理管理器
        
        Args:
            model_path: 模型路径
            num_threads: 线程数量
            tile_size: 子块大小
            overlap: 重叠比例
        """
        config = Config.get_instance()
        self.model_path = model_path or config.model_path
        self.num_threads = min(num_threads or config.num_threads, Config.MAX_AI_CORES)
        self.tile_size = tile_size or (config.tile_size, config.tile_size)
        self.overlap = overlap or 0.2  # 使用相对重叠比例
        
        self.workers = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []
        self.running = False
    
    def init_workers(self):
        """初始化工作线程"""
        for i in range(self.num_threads):
            # 分配不同的设备ID或AI核
            device_id = i % Config.MAX_AI_CORES
            # 使用子块大小作为分辨率
            resolution = f"{self.tile_size[1]}x{self.tile_size[0]}"
            worker = InferenceWorker(i, device_id, self.model_path, resolution)
            if worker.init():
                self.workers.append(worker)
                print(f"Worker {i} 初始化成功 (device: {device_id})")
            else:
                print(f"Worker {i} 初始化失败")
        
        return len(self.workers) > 0
    
    def worker_thread(self, worker):
        """工作线程函数"""
        while self.running:
            try:
                # 从队列获取任务
                task = self.task_queue.get(block=False)
                if task is None:
                    break
                
                tile_id, tile = task
                
                # 预处理
                if not worker.preprocess(tile):
                    self.result_queue.put((tile_id, None))
                    self.task_queue.task_done()
                    continue
                
                # 推理
                if not worker.inference():
                    self.result_queue.put((tile_id, None))
                    self.task_queue.task_done()
                    continue
                
                # 获取结果
                result = worker.get_result()
                self.result_queue.put((tile_id, result))
                self.task_queue.task_done()
                
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Worker error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """启动多线程"""
        if not self.workers:
            if not self.init_workers():
                return False
        
        self.running = True
        
        # 创建并启动线程
        for worker in self.workers:
            thread = threading.Thread(target=self.worker_thread, args=(worker,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        return True
    
    def process_image(self, image_path, backend=Config.DEFAULT_BACKEND):
        """处理高分辨率图像"""
        # 读取图像
        if backend == 'opencv' and HAS_OPENCV:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            try:
                image = Image.open(image_path)
                image = np.array(image)
            except Exception as e:
                print(f"无法读取图像: {e}")
                return None
        
        print(f"处理图像: {image_path}")
        print(f"图像形状: {image.shape}")
        
        # 分割图像
        start_time = time.time()
        tiles, positions = split_image(image, self.tile_size, self.overlap)
        split_time = time.time() - start_time
        
        print(f"分割完成: {len(tiles)} 个子块")
        print(f"分割时间: {split_time:.2f} 秒")
        
        # 添加任务
        inference_start = time.time()
        for i, tile in enumerate(tiles):
            self.task_queue.put((i, tile))
        
        # 等待完成
        self.task_queue.join()
        
        # 获取结果
        results = [None] * len(tiles)
        while not self.result_queue.empty():
            try:
                tile_id, result = self.result_queue.get(block=False)
                results[tile_id] = result
                self.result_queue.task_done()
            except queue.Empty:
                break
        
        inference_time = time.time() - inference_start
        
        # 合并结果
        merge_start = time.time()
        merged_result = merge_results(results, positions, image.shape)
        merge_time = time.time() - merge_start
        
        total_time = time.time() - start_time
        
        print(f"推理完成: {sum(1 for r in results if r is not None)} 个子块成功")
        print(f"推理时间: {inference_time:.2f} 秒")
        print(f"合并时间: {merge_time:.2f} 秒")
        print(f"总时间: {total_time:.2f} 秒")
        
        return merged_result
    
    def stop(self):
        """停止多线程"""
        self.running = False
        
        # 等待线程结束
        for thread in self.threads:
            thread.join(timeout=5)
        
        # 销毁所有worker
        for worker in self.workers:
            worker.destroy()
        
        # 最终化ACL
        acl.finalize()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高分辨率图像推理')
    parser.add_argument('image_path', help='高分辨率图像路径')
    parser.add_argument('--model', default=Config.DEFAULT_MODEL_PATH, help='OM模型文件路径')
    parser.add_argument('--tile-size', type=int, nargs=2, default=Config.DEFAULT_TILE_SIZE,
                        help='子块大小 (高度 宽度)')
    parser.add_argument('--overlap', type=float, default=Config.DEFAULT_OVERLAP, help='重叠比例')
    parser.add_argument('--threads', type=int, default=Config.DEFAULT_THREADS, help='线程数量')
    parser.add_argument('--backend', default=Config.DEFAULT_BACKEND, choices=Config.SUPPORTED_BACKENDS,
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    # 创建高分辨率推理实例
    inference = HighResInference(args.model, args.threads, tuple(args.tile_size), args.overlap)
    
    try:
        # 启动推理
        if not inference.start():
            print("无法启动推理")
            sys.exit(1)
        
        print(f"启动 {len(inference.workers)} 个推理线程")
        
        # 处理图像
        result = inference.process_image(args.image_path, args.backend)
        
        if result:
            print("\n处理完成！")
            print(f"子块数量: {result['num_tiles']}")
            print(f"成功处理: {len(result['sub_results'])} 个子块")
        
    finally:
        # 停止推理
        inference.stop()


if __name__ == "__main__":
    main()
