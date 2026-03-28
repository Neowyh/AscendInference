#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心推理测试模块

测试推理核心功能，包括：
- 初始化测试
- 预处理测试
- 执行器测试
- 后处理测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Tuple
import tempfile
import os

from src.inference.preprocessor import Preprocessor
from src.inference.executor import Executor
from src.inference.postprocessor import Postprocessor, split_image, merge_results
from utils.exceptions import PreprocessError, ACLError, MemoryError


class TestPreprocessor:
    """预处理器测试"""
    
    @pytest.fixture
    def preprocessor(self):
        """创建预处理器实例"""
        return Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
    
    @pytest.fixture
    def sample_image(self):
        """创建测试图像"""
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    def test_init(self, preprocessor):
        """测试初始化"""
        assert preprocessor.input_width == 640
        assert preprocessor.input_height == 640
        assert preprocessor.batch_size == 1
    
    def test_load_image_opencv(self, preprocessor, sample_image, tmp_path):
        """测试图像加载 (OpenCV)"""
        try:
            import cv2
            image_path = str(tmp_path / "test.png")
            cv2.imwrite(image_path, cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR))
            
            result = preprocessor.load_image(image_path, backend='opencv')
            assert result is not None
            assert isinstance(result, np.ndarray)
        except ImportError:
            pytest.skip("OpenCV not available")
    
    def test_resize_image_opencv(self, preprocessor, sample_image):
        """测试图像缩放 (OpenCV)"""
        try:
            result = preprocessor.resize_image(sample_image, backend='opencv')
            assert result.shape == (640, 640, 3)
            assert result.dtype == np.uint8
        except ImportError:
            pytest.skip("OpenCV not available")
    
    def test_resize_image_pil(self, preprocessor, sample_image):
        """测试图像缩放 (PIL)"""
        from PIL import Image
        pil_image = Image.fromarray(sample_image)
        result = preprocessor.resize_image(pil_image, backend='pil')
        assert result.shape == (640, 640, 3)
    
    def test_normalize(self, preprocessor, sample_image):
        """测试归一化"""
        try:
            resized = preprocessor.resize_image(sample_image, 'opencv')
        except Exception:
            from PIL import Image
            pil_image = Image.fromarray(sample_image)
            resized = preprocessor.resize_image(pil_image, 'pil')
        
        normalized = preprocessor.normalize(resized)
        assert normalized.dtype == np.float32
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0
    
    def test_process_single(self, preprocessor, sample_image):
        """测试单张图像处理"""
        result = preprocessor.process_single(sample_image, 'opencv')
        assert result.dtype == np.float32
        assert len(result.shape) == 1


class TestExecutor:
    """执行器测试"""
    
    @pytest.fixture
    def mock_acl_components(self):
        """Mock ACL 组件"""
        model_id = 12345
        stream = Mock()
        input_dataset = Mock()
        output_dataset = Mock()
        output_buffer = 0x1000
        output_size = 8400 * 4
        return model_id, stream, input_dataset, output_dataset, output_buffer, output_size
    
    @pytest.fixture
    def executor(self, mock_acl_components):
        """创建执行器实例"""
        model_id, stream, input_dataset, output_dataset, output_buffer, output_size = mock_acl_components
        return Executor(
            model_id=model_id,
            stream=stream,
            input_dataset=input_dataset,
            output_dataset=output_dataset,
            output_buffer=output_buffer,
            output_size=output_size,
            batch_size=1
        )
    
    def test_init(self, executor, mock_acl_components):
        """测试初始化"""
        model_id, _, _, _, output_buffer, output_size = mock_acl_components
        assert executor.model_id == model_id
        assert executor.output_size == output_size
        assert executor.batch_size == 1
    
    def test_init_output_buffer(self, executor):
        """测试输出缓冲区初始化"""
        mock_host = 0x2000
        executor.init_output_buffer(mock_host)
        assert executor.output_host == mock_host


class TestPostprocessor:
    """后处理器测试"""
    
    @pytest.fixture
    def postprocessor(self):
        """创建后处理器实例"""
        return Postprocessor()
    
    @pytest.fixture
    def sample_output(self):
        """创建测试输出"""
        return np.random.randn(8400).astype(np.float32)
    
    def test_process(self, postprocessor, sample_output):
        """测试处理"""
        result = postprocessor.process(sample_output)
        np.testing.assert_array_equal(result, sample_output)
    
    def test_process_batch(self, postprocessor, sample_output):
        """测试批量处理"""
        outputs = [sample_output for _ in range(4)]
        results = postprocessor.process_batch(outputs)
        assert len(results) == 4


class TestImageSplitting:
    """图像分割测试"""
    
    @pytest.fixture
    def large_image(self):
        """创建大图像"""
        return np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
    
    def test_split_image(self, large_image):
        """测试图像分割"""
        tile_size = (640, 640)
        overlap = 0.25
        
        tiles, positions, weight_map = split_image(large_image, tile_size, overlap)
        
        assert len(tiles) > 0
        assert len(tiles) == len(positions)
        assert weight_map.shape == large_image.shape[:2]
        
        for tile, (x, y, w, h) in zip(tiles, positions):
            assert tile.shape[0] <= tile_size[0]
            assert tile.shape[1] <= tile_size[1]
    
    def test_split_image_no_overlap(self, large_image):
        """测试无重叠分割"""
        tile_size = (640, 640)
        overlap = 0.0
        
        tiles, positions, _ = split_image(large_image, tile_size, overlap)
        
        assert len(tiles) > 0
    
    def test_merge_results(self, large_image):
        """测试结果合并"""
        tile_size = (640, 640)
        overlap = 0.25
        
        tiles, positions, _ = split_image(large_image, tile_size, overlap)
        
        results = [(i, np.random.randn(100).astype(np.float32)) for i in range(len(tiles))]
        
        merged = merge_results(results, positions, large_image.shape)
        
        assert 'sub_results' in merged
        assert 'image_shape' in merged
        assert 'num_tiles' in merged
        assert merged['image_shape'] == large_image.shape
        assert merged['num_tiles'] == len(tiles)


class TestExceptionHandling:
    """异常处理测试"""
    
    def test_preprocess_error_file_not_found(self):
        """测试预处理异常 - 文件不存在"""
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        with pytest.raises(PreprocessError):
            preprocessor.load_image("/nonexistent/path.jpg", "opencv")
    
    def test_invalid_resize(self):
        """测试无效缩放"""
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        invalid_data = "not_an_image"
        with pytest.raises(PreprocessError):
            preprocessor.resize_image(invalid_data, "opencv")


class TestMemoryPool:
    """内存池测试"""
    
    def test_pool_initialization(self):
        """测试内存池初始化"""
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        assert preprocessor.input_host_pool is None
        
        preprocessor.init_pool(max_buffers=3)
        
        assert preprocessor.input_host_pool is not None
    
    def test_pool_cleanup(self):
        """测试内存池清理"""
        preprocessor = Preprocessor(
            input_width=640,
            input_height=640,
            input_size=640 * 640 * 3 * 4,
            batch_size=1
        )
        
        preprocessor.init_pool(max_buffers=3)
        preprocessor.cleanup()
        
        assert preprocessor.input_host_pool is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
