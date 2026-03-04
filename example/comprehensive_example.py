import sys
import os
import cv2
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.acl_utils import AclManager
from utils.image_enhancer import ImageEnhancer
from utils.data_generator import DataGenerator
from src.yolo_inference import YoloInference
from src.yolo_inference_multithread import YoloInferenceMultithread
from src.yolo_inference_high_res import YoloInferenceHighRes

def test_basic_inference():
    print("=== Testing Basic Inference ===")
    try:
        # Initialize ACL
        acl_manager = AclManager()
        acl_manager.init()
        
        # Test basic YOLO inference
        yolo = YoloInference("./models/yolov8s.om", acl_manager)
        
        # Generate test image
        generator = DataGenerator()
        test_image = generator.generate_test_image()
        
        # Run inference
        results = yolo.infer(test_image)
        print(f"Basic inference completed. Detected {len(results)} objects")
        
        # Cleanup
        yolo.destroy()
        acl_manager.destroy()
        return True
    except Exception as e:
        print(f"Error in basic inference: {e}")
        return False

def test_multithread_inference():
    print("\n=== Testing Multi-threaded Inference ===")
    try:
        # Initialize ACL
        acl_manager = AclManager()
        acl_manager.init()
        
        # Test multi-threaded YOLO inference
        yolo = YoloInferenceMultithread("./models/yolov8s.om", acl_manager)
        
        # Generate test images
        generator = DataGenerator()
        test_images = [generator.generate_test_image() for _ in range(4)]
        
        # Run inference
        all_results = yolo.batch_infer(test_images)
        print(f"Multi-threaded inference completed. Processed {len(all_results)} images")
        
        # Cleanup
        yolo.destroy()
        acl_manager.destroy()
        return True
    except Exception as e:
        print(f"Error in multi-threaded inference: {e}")
        return False

def test_high_res_inference():
    print("\n=== Testing High-resolution Inference ===")
    try:
        # Initialize ACL
        acl_manager = AclManager()
        acl_manager.init()
        
        # Test high-resolution YOLO inference
        yolo = YoloInferenceHighRes("./models/yolov8s.om", acl_manager)
        
        # Generate high-resolution test image
        generator = DataGenerator()
        high_res_image = generator.generate_test_image(width=2048, height=1536)
        
        # Run inference
        results = yolo.infer(high_res_image)
        print(f"High-resolution inference completed. Detected {len(results)} objects")
        
        # Cleanup
        yolo.destroy()
        acl_manager.destroy()
        return True
    except Exception as e:
        print(f"Error in high-resolution inference: {e}")
        return False

def test_image_enhancement():
    print("\n=== Testing Image Enhancement ===")
    try:
        # Create enhancer
        enhancer = ImageEnhancer()
        
        # Generate test image
        generator = DataGenerator()
        test_image = generator.generate_test_image()
        
        # Test enhancement to different resolutions
        resolutions = [(640, 640), (1024, 1024), (2048, 2048)]
        for width, height in resolutions:
            enhanced = enhancer.enhance(test_image, width, height)
            print(f"Enhanced image to {width}x{height} - shape: {enhanced.shape}")
        
        return True
    except Exception as e:
        print(f"Error in image enhancement: {e}")
        return False

def main():
    print("Running comprehensive example tests...")
    
    # Run all tests
    tests = [
        test_basic_inference,
        test_multithread_inference,
        test_high_res_inference,
        test_image_enhancement
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")

if __name__ == "__main__":
    main()
