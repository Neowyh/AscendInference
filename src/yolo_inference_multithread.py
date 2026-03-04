#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AscendCL YOLO模型多线程推理脚本

此脚本使用多线程提高端侧设备吞吐率：
- 支持多个线程并行推理
- 可指定使用不同的AI核
- 参考昇腾310B的AI核数量（4个）
- 支持批量处理多张图像
"""

import os
import sys
import numpy as np
from PIL import Image
import acl
import threading
import queue
import time

# 导入ACL工具类
from utils.acl_utils import AclManager, ModelManager, MemoryManager

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# 配置参数
MODEL_PATH = "yolov5s.om"

# 支持的输入分辨率
SUPPORTED_RESOLUTIONS = {
    "640x640": (640, 640),
    "1k": (1024, 1024),
    "1k2k": (1024, 2048),
    "2k": (2048, 2048),
    "2k4k": (2048, 4096),
    "4k": (4096, 4096),
    "4k6k": (4096, 6144),
    "3k6k": (3072, 6144),
    "6k": (6144, 6144)
}

# 默认分辨率
DEFAULT_RESOLUTION = "640x640"
INPUT_WIDTH, INPUT_HEIGHT = SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION]

# 昇腾310B有4个AI Core
MAX_AI_CORES = 4


class InferenceWorker:
    """推理工作线程"""
    
    def __init__(self, worker_id, device_id, model_path, input_width, input_height):
        self.worker_id = worker_id
        self.device_id = device_id
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        
        # 设备相关
        self.acl_manager = AclManager(self.device_id)  # ACL管理器
        
        # 模型相关
        self.model_manager = ModelManager(self.model_path)  # 模型管理器
        
        # 内存相关
        self.memory_manager = MemoryManager()  # 内存管理器
        self.input_buffer = None
        self.output_buffer = None
        self.output_host = None
    
    def init(self):
        """初始化 worker"""
        # 初始化ACL
        if not self.acl_manager.init():
            return False
        
        # 加载模型
        if not os.path.exists(self.model_path):
            return False
        
        if not self.model_manager.load():
            return False
        
        # 分配内存
        input_size = self.model_manager.get_input_size()
        output_size = self.model_manager.get_output_size()
        
        self.input_buffer = self.memory_manager.malloc_device(input_size)
        if not self.input_buffer:
            return False
        
        self.output_buffer = self.memory_manager.malloc_device(output_size)
        if not self.output_buffer:
            return False
        
        self.output_host = self.memory_manager.malloc_host(output_size)
        if not self.output_host:
            return False
        
        return True
    
    def preprocess(self, image_path, backend='pil'):
        """预处理图像"""
        # 读取图像
        if backend == 'opencv' and HAS_OPENCV:
            image = cv2.imread(image_path)
            if image is None:
                return False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.input_width, self.input_height))
        else:
            image = Image.open(image_path)
            image = image.resize((self.input_width, self.input_height))
            image = np.array(image)
        
        # 处理灰度图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        
        # 归一化和通道转换
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)).flatten()
        
        # 分配临时主机内存
        input_host, ret = acl.rt.malloc_host(self.input_size)
        if ret != 0:
            return False
        
        # 复制数据
        acl.util.vector_to_ptr(image.tobytes(), input_host, self.input_size)
        
        # 复制到设备
        ret = acl.rt.memcpy(self.input_buffer, self.input_size, input_host, self.input_size, acl.rt.MEMCPY_HOST_TO_DEVICE)
        
        # 释放临时内存
        acl.rt.free_host(input_host)
        
        return ret == 0
    
    def inference(self):
        """执行推理"""
        input_data = np.array([self.input_buffer], dtype=np.uintptr)
        output_data = np.array([self.output_buffer], dtype=np.uintptr)
        
        model_id = self.model_manager.get_model_id()
        return acl.mdl.execute(model_id, input_data, output_data) == 0
    
    def get_result(self):
        """获取结果"""
        output_size = self.model_manager.get_output_size()
        if acl.rt.memcpy(self.output_host, output_size, self.output_buffer, output_size, acl.rt.MEMCPY_DEVICE_TO_HOST) != 0:
            return None
        
        return np.frombuffer(self.output_host, dtype=np.float32)
    
    def destroy(self):
        """销毁资源"""
        # 释放内存
        self.memory_manager.free_all()
        
        # 卸载模型
        self.model_manager.unload()
        
        # 销毁ACL资源
        self.acl_manager.destroy()


class MultithreadInference:
    """多线程推理管理器"""
    
    def __init__(self, model_path, num_threads=4, resolution=DEFAULT_RESOLUTION):
        self.model_path = model_path
        self.num_threads = min(num_threads, MAX_AI_CORES)
        self.resolution = resolution
        self.input_width, self.input_height = SUPPORTED_RESOLUTIONS[resolution]
        
        self.workers = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []
        self.running = False
    
    def init_workers(self):
        """初始化工作线程"""
        for i in range(self.num_threads):
            # 分配不同的设备ID或AI核
            device_id = i % MAX_AI_CORES
            worker = InferenceWorker(i, device_id, self.model_path, self.input_width, self.input_height)
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
                
                image_path, backend = task
                
                # 预处理
                if not worker.preprocess(image_path, backend):
                    self.result_queue.put((image_path, None))
                    self.task_queue.task_done()
                    continue
                
                # 推理
                if not worker.inference():
                    self.result_queue.put((image_path, None))
                    self.task_queue.task_done()
                    continue
                
                # 获取结果
                result = worker.get_result()
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
    
    def add_task(self, image_path, backend='pil'):
        """添加推理任务"""
        self.task_queue.put((image_path, backend))
    
    def get_results(self, timeout=None):
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
        
        # 等待所有结果处理完成
        while not self.result_queue.empty():
            time.sleep(0.1)
    
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
    
    parser = argparse.ArgumentParser(description='YOLO模型多线程推理')
    parser.add_argument('image_paths', nargs='+', help='图像文件路径列表')
    parser.add_argument('--model', default=MODEL_PATH, help='OM模型文件路径')
    parser.add_argument('--resolution', default=DEFAULT_RESOLUTION, choices=SUPPORTED_RESOLUTIONS.keys(),
                        help='输入分辨率')
    parser.add_argument('--threads', type=int, default=4, help='线程数量')
    parser.add_argument('--backend', default='pil', choices=['pil', 'opencv'],
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    # 创建多线程推理实例
    inference = MultithreadInference(args.model, args.threads, args.resolution)
    
    try:
        # 启动推理
        if not inference.start():
            print("无法启动推理")
            sys.exit(1)
        
        print(f"启动 {len(inference.workers)} 个推理线程")
        
        # 添加任务
        start_time = time.time()
        for image_path in args.image_paths:
            if os.path.exists(image_path):
                inference.add_task(image_path, args.backend)
            else:
                print(f"图像不存在: {image_path}")
        
        # 等待完成
        inference.wait_completion()
        
        # 获取结果
        results = inference.get_results()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n推理完成: {len(results)} 张图像")
        print(f"总时间: {total_time:.2f} 秒")
        print(f"平均时间: {total_time / len(results):.2f} 秒/张")
        print(f"吞吐率: {len(results) / total_time:.2f} 张/秒")
        
    finally:
        # 停止推理
        inference.stop()


if __name__ == "__main__":
    main()
