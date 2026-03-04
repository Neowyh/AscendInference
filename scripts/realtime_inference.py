#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时推理脚本

功能：
- 模拟实时视频流推理
- 支持摄像头输入或视频文件
- 实时显示推理结果
- 性能监控
"""

import os
import sys
import argparse
import time
import cv2

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.yolo_inference_fast import FastAscendInference

def realtime_inference(source, model_path, resolution, backend):
    """实时推理"""
    # 打开视频源
    if source == 'camera':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return False
    else:
        if not os.path.exists(source):
            print(f"视频文件不存在: {source}")
            return False
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频文件: {source}")
            return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps:.1f} FPS")
    
    # 创建推理实例
    inference = FastAscendInference()
    
    try:
        # 初始化
        if not inference.init():
            print("无法初始化推理")
            return False
        
        # 性能统计
        total_frames = 0
        total_time = 0
        start_time = time.time()
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存帧为临时文件
            temp_file = os.path.join('data', 'temp_frame.jpg')
            cv2.imwrite(temp_file, frame)
            
            # 开始计时
            frame_start = time.time()
            
            # 推理
            if not inference.preprocess(temp_file, backend):
                print("预处理失败")
                continue
            
            if not inference.inference():
                print("推理失败")
                continue
            
            # 获取结果
            result = inference.get_result()
            
            # 结束计时
            frame_time = time.time() - frame_start
            total_time += frame_time
            total_frames += 1
            
            # 计算FPS
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            avg_fps = total_frames / (time.time() - start_time) if total_frames > 0 else 0
            
            # 在帧上显示信息
            cv2.putText(frame, f"Current FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Average FPS: {avg_fps:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame Time: {frame_time*1000:.1f} ms", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('YOLO Inference', frame)
            
            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        
        # 打印性能统计
        if total_frames > 0:
            print(f"\n实时推理完成")
            print(f"处理帧数: {total_frames}")
            print(f"总时间: {total_time:.2f} 秒")
            print(f"平均帧率: {total_frames / total_time:.2f} FPS")
            print(f"平均帧处理时间: {total_time / total_frames * 1000:.2f} ms")
        
        return True
        
    finally:
        # 释放资源
        inference.destroy()
        
        # 清理临时文件
        temp_file = os.path.join('data', 'temp_frame.jpg')
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时推理脚本')
    parser.add_argument('source', default='camera', nargs='?', help='视频源 (camera 或视频文件路径)')
    parser.add_argument('--model', default='yolov5s.om', help='OM模型文件路径')
    parser.add_argument('--resolution', default='640x640', help='输入分辨率')
    parser.add_argument('--backend', default='opencv', choices=['pil', 'opencv'],
                        help='图像读取后端')
    
    args = parser.parse_args()
    
    # 执行实时推理
    if realtime_inference(args.source, args.model, args.resolution, args.backend):
        print("实时推理成功完成")
    else:
        print("实时推理失败")


if __name__ == "__main__":
    main()
