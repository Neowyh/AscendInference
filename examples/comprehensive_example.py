import sys
import os
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.acl_utils import AclManager
from utils.image_enhancer import ImageEnhancer
from utils.data_generator import DataGenerator
from src.yolo_inference import YoloInference
from src.yolo_inference_multithread import YoloInferenceMultithread
from src.yolo_inference_high_res import YoloInferenceHighRes

def test_basic_inference():
    print("=== 测试基础推理 ===")
    try:
        # 初始化ACL
        acl_manager = AclManager()
        acl_manager.init()
        
        # 测试基础YOLO推理
        yolo = YoloInference("./models/yolov8s.om", acl_manager)
        
        # 生成测试图像
        generator = DataGenerator()
        test_image = generator.generate_test_image()
        
        # 运行推理
        results = yolo.infer(test_image)
        print(f"基础推理完成。检测到 {len(results)} 个目标")
        
        # 清理资源
        yolo.destroy()
        acl_manager.destroy()
        return True
    except Exception as e:
        print(f"基础推理出错: {e}")
        return False

def test_multithread_inference():
    print("\n=== 测试多线程推理 ===")
    try:
        # 初始化ACL
        acl_manager = AclManager()
        acl_manager.init()
        
        # 测试多线程YOLO推理
        yolo = YoloInferenceMultithread("./models/yolov8s.om", acl_manager)
        
        # 生成测试图像
        generator = DataGenerator()
        test_images = [generator.generate_test_image() for _ in range(4)]
        
        # 运行推理
        all_results = yolo.batch_infer(test_images)
        print(f"多线程推理完成。处理了 {len(all_results)} 张图像")
        
        # 清理资源
        yolo.destroy()
        acl_manager.destroy()
        return True
    except Exception as e:
        print(f"多线程推理出错: {e}")
        return False

def test_high_res_inference():
    print("\n=== 测试高分辨率推理 ===")
    try:
        # 初始化ACL
        acl_manager = AclManager()
        acl_manager.init()
        
        # 测试高分辨率YOLO推理
        yolo = YoloInferenceHighRes("./models/yolov8s.om", acl_manager)
        
        # 生成高分辨率测试图像
        generator = DataGenerator()
        high_res_image = generator.generate_test_image(width=2048, height=1536)
        
        # 运行推理
        results = yolo.infer(high_res_image)
        print(f"高分辨率推理完成。检测到 {len(results)} 个目标")
        
        # 清理资源
        yolo.destroy()
        acl_manager.destroy()
        return True
    except Exception as e:
        print(f"高分辨率推理出错: {e}")
        return False

def test_image_enhancement():
    print("\n=== 测试图像增强 ===")
    try:
        # 创建增强器
        enhancer = ImageEnhancer()
        
        # 生成测试图像
        generator = DataGenerator()
        test_image = generator.generate_test_image()
        
        # 测试增强到不同分辨率
        resolutions = [(640, 640), (1024, 1024), (2048, 2048)]
        for width, height in resolutions:
            enhanced = enhancer.enhance(test_image, width, height)
            print(f"图像增强到 {width}x{height} - 形状: {enhanced.shape}")
        
        return True
    except Exception as e:
        print(f"图像增强出错: {e}")
        return False

def main():
    print("运行综合示例测试...")
    
    # 运行所有测试
    tests = [
        test_basic_inference,
        test_multithread_inference,
        test_high_res_inference,
        test_image_enhancement
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"总测试数: {len(tests)}")
    print(f"通过: {sum(results)}")
    print(f"失败: {len(tests) - sum(results)}")

if __name__ == "__main__":
    main()
