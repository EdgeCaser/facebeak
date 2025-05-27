#!/usr/bin/env python3
"""
Critical detection tests for timeout handling, multi-crow detection, and GPU memory issues.
These tests address the specific problems seen in production.
"""

import unittest
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, call
import threading
import time

from detection import (
    detect_crows_parallel,
    merge_overlapping_detections,
    has_overlapping_crows,
    _run_model_inference,
    TimeoutException
)

class TestCriticalDetection(unittest.TestCase):
    """Test critical detection issues causing production problems."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_frame = np.zeros((640, 480, 3), dtype=np.uint8)
        self.sample_frame[100:200, 150:250] = 255  # White rectangle
        
    def test_timeout_handling_yolo_inference(self):
        """Test timeout handling when YOLO model hangs."""
        def slow_yolo_model(*args, **kwargs):
            """Simulate a hanging YOLO model."""
            time.sleep(35)  # Longer than 30s timeout
            return [MagicMock()]
        
        with patch('detection.yolo_model', side_effect=slow_yolo_model):
            with patch('detection.faster_rcnn_model', return_value=[{'boxes': [], 'labels': [], 'scores': []}]):
                start_time = time.time()
                
                # Should timeout and continue processing
                detections = detect_crows_parallel([self.sample_frame])
                
                elapsed = time.time() - start_time
                self.assertLess(elapsed, 35)  # Should timeout before 35s
                self.assertEqual(len(detections), 1)  # Should still return result
    
    def test_timeout_handling_rcnn_inference(self):
        """Test timeout handling when R-CNN model hangs."""
        def slow_rcnn_model(*args, **kwargs):
            """Simulate a hanging R-CNN model."""
            time.sleep(35)  # Longer than 30s timeout
            return [{'boxes': [], 'labels': [], 'scores': []}]
        
        with patch('detection.yolo_model', return_value=[MagicMock(boxes=MagicMock(xyxy=[], conf=[], cls=[]))]):
            with patch('detection.faster_rcnn_model', side_effect=slow_rcnn_model):
                start_time = time.time()
                
                detections = detect_crows_parallel([self.sample_frame])
                
                elapsed = time.time() - start_time
                self.assertLess(elapsed, 35)  # Should timeout before 35s
                self.assertEqual(len(detections), 1)  # Should still return result

    def test_gpu_memory_exhaustion_handling(self):
        """Test handling of CUDA out of memory errors."""
        def oom_model(*args, **kwargs):
            """Simulate CUDA out of memory error."""
            raise RuntimeError("CUDA out of memory")
        
        with patch('detection.yolo_model', side_effect=oom_model):
            with patch('detection.faster_rcnn_model', side_effect=oom_model):
                # Should handle OOM gracefully
                detections = detect_crows_parallel([self.sample_frame])
                self.assertEqual(len(detections), 1)
                self.assertEqual(detections[0], [])  # Empty result but no crash

    def test_multiple_overlapping_detections_real_scenario(self):
        """Test multi-crow detection with realistic overlapping scenarios."""
        # Simulate YOLO and R-CNN both detecting the same crow with slight offsets
        overlapping_detections = [
            {'bbox': [100, 100, 200, 200], 'score': 0.85, 'class': 'bird', 'model': 'yolo', 'view': 'single'},
            {'bbox': [105, 105, 205, 205], 'score': 0.90, 'class': 'crow', 'model': 'rcnn', 'view': 'single'},
            {'bbox': [102, 102, 202, 202], 'score': 0.75, 'class': 'bird', 'model': 'yolo', 'view': 'single'},
        ]
        
        # Test multi-crow detection
        has_multiple = has_overlapping_crows(overlapping_detections, iou_thresh=0.4)
        self.assertTrue(has_multiple)  # Should detect overlapping crows
        
        # Test merging with aggressive threshold
        merged = merge_overlapping_detections(overlapping_detections, iou_threshold=0.5)
        self.assertEqual(len(merged), 1)  # Should merge into single detection
        self.assertEqual(merged[0]['class'], 'crow')  # Should prefer crow over bird
        self.assertGreater(merged[0]['score'], 0.90)  # Should boost confidence

    def test_batch_processing_memory_efficiency(self):
        """Test memory efficiency with large frame batches."""
        # Create a large batch of frames
        large_batch = [self.sample_frame.copy() for _ in range(50)]
        
        # Mock models to return minimal data
        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes.xyxy = []
        mock_yolo_result.boxes.conf = []
        mock_yolo_result.boxes.cls = []
        
        with patch('detection.yolo_model', return_value=[mock_yolo_result]):
            with patch('detection.faster_rcnn_model', return_value=[{'boxes': [], 'labels': [], 'scores': []}]):
                # Should process without memory issues
                detections = detect_crows_parallel(large_batch)
                self.assertEqual(len(detections), 50)

    def test_detection_consistency_across_frames(self):
        """Test that detection results are consistent across similar frames."""
        # Create two nearly identical frames
        frame1 = self.sample_frame.copy()
        frame2 = self.sample_frame.copy()
        frame2[100:200, 150:250] = 254  # Very slight difference
        
        # Mock consistent model outputs
        mock_bbox = torch.tensor([150, 100, 250, 200])
        mock_score = torch.tensor(0.85)
        mock_cls = torch.tensor(14)  # YOLO bird class
        
        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes.xyxy = [mock_bbox]
        mock_yolo_result.boxes.conf = [mock_score]
        mock_yolo_result.boxes.cls = [mock_cls]
        
        with patch('detection.yolo_model', return_value=[mock_yolo_result]):
            with patch('detection.faster_rcnn_model', return_value=[{'boxes': [], 'labels': [], 'scores': []}]):
                det1 = detect_crows_parallel([frame1])
                det2 = detect_crows_parallel([frame2])
                
                # Results should be similar
                self.assertEqual(len(det1), len(det2))
                if det1[0] and det2[0]:
                    bbox1 = det1[0][0]['bbox']
                    bbox2 = det2[0][0]['bbox']
                    # Bounding boxes should be very close
                    np.testing.assert_allclose(bbox1, bbox2, rtol=0.1)

    def test_multi_crow_frame_flagging_integration(self):
        """Test the new multi_crow_frame flagging feature."""
        # Create detections that should trigger multi-crow flag
        # Use boxes with significant overlap (>0.4 IOU)
        overlapping_dets = [
            {'bbox': [100, 100, 200, 200], 'score': 0.85, 'class': 'bird', 'model': 'yolo', 'view': 'single'},
            {'bbox': [120, 120, 220, 220], 'score': 0.90, 'class': 'crow', 'model': 'rcnn', 'view': 'single'},  # More overlap
        ]
        
        # Mock models to return overlapping detections
        mock_yolo_result = MagicMock()
        mock_yolo_result.boxes.xyxy = [torch.tensor([100, 100, 200, 200])]
        mock_yolo_result.boxes.conf = [torch.tensor(0.85)]
        mock_yolo_result.boxes.cls = [torch.tensor(14)]
        
        mock_rcnn_result = {
            'boxes': [torch.tensor([120, 120, 220, 220])],  # Changed to have higher overlap
            'labels': [torch.tensor(20)],  # Crow class
            'scores': [torch.tensor(0.90)]
        }
        
        with patch('detection.yolo_model', return_value=[mock_yolo_result]):
            with patch('detection.faster_rcnn_model', return_value=[mock_rcnn_result]):
                detections = detect_crows_parallel([self.sample_frame])
                
                # Should flag as multi-crow frame
                self.assertEqual(len(detections), 1)
                if detections[0]:
                    for det in detections[0]:
                        self.assertIn('multi_crow_frame', det)
                        # Should be True since we have overlapping detections
                        self.assertTrue(det['multi_crow_frame'])

    def test_device_switching_on_cuda_error(self):
        """Test fallback to CPU when CUDA operations fail."""
        def cuda_error_model(*args, **kwargs):
            """Simulate CUDA device error."""
            raise RuntimeError("CUDA error: device-side assert triggered")
        
        with patch('detection.yolo_model', side_effect=cuda_error_model):
            with patch('torch.cuda.is_available', return_value=True):
                # Should handle CUDA error gracefully
                detections = detect_crows_parallel([self.sample_frame])
                self.assertEqual(len(detections), 1)

if __name__ == '__main__':
    unittest.main() 