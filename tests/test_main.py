import unittest
from unittest.mock import MagicMock, patch, call
import pytest
import argparse
from main import parse_args, process_video
import cv2
import numpy as np
from pathlib import Path
import tempfile
import os

class TestMainFunctions(unittest.TestCase):
    def test_parse_args(self):
        """Test command line argument parsing."""
        test_args = [
            "--video", "test.mp4",
            "--skip-output", "skip.mp4",
            "--full-output", "full.mp4",
            "--detection-threshold", "0.4",
            "--yolo-threshold", "0.3",
            "--max-age", "10",
            "--min-hits", "3",
            "--iou-threshold", "0.3",
            "--embedding-threshold", "0.8",
            "--skip", "2",
            "--multi-view-stride", "2",
            "--preserve-audio"
        ]
        
        args = parse_args(test_args)
        
        self.assertEqual(args.video, "test.mp4")
        self.assertEqual(args.skip_output, "skip.mp4")
        self.assertEqual(args.full_output, "full.mp4")
        self.assertEqual(args.detection_threshold, 0.4)
        self.assertEqual(args.yolo_threshold, 0.3)
        self.assertEqual(args.max_age, 10)
        self.assertEqual(args.min_hits, 3)
        self.assertEqual(args.iou_threshold, 0.3)
        self.assertEqual(args.embedding_threshold, 0.8)
        self.assertEqual(args.skip, 2)
        self.assertEqual(args.multi_view_stride, 2)
        self.assertTrue(args.preserve_audio)
        
    def test_parse_args_defaults(self):
        """Test command line argument parsing with default values."""
        test_args = [
            "--video", "test.mp4",
            "--skip-output", "skip.mp4",
            "--full-output", "full.mp4"
        ]
        
        args = parse_args(test_args)
        
        self.assertEqual(args.detection_threshold, 0.3)
        self.assertEqual(args.yolo_threshold, 0.2)
        self.assertEqual(args.max_age, 5)
        self.assertEqual(args.min_hits, 2)
        self.assertEqual(args.iou_threshold, 0.2)
        self.assertEqual(args.embedding_threshold, 0.7)
        self.assertEqual(args.skip, 5)
        self.assertEqual(args.multi_view_stride, 1)
        self.assertFalse(args.preserve_audio)

    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    @patch('main.detect_crows_parallel')
    @patch('main.process_frame')
    def test_process_video(self, mock_process_frame, mock_detect_crows, mock_video_writer, mock_video_capture):
        """Test video processing with mocked components."""
        # Create a mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30  # fps
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First frame
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # Second frame
            (False, None)  # End of video
        ]
        mock_video_capture.return_value = mock_cap
        
        # Create a mock video writer
        mock_writer = MagicMock()
        mock_video_writer.return_value = mock_writer
        
        # Mock detection results
        mock_detect_crows.return_value = [
            [{'bbox': [100, 100, 200, 200], 'score': 0.9, 'class': 'crow'}]
        ]
        
        # Mock frame processing
        mock_process_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), [])
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "test.mp4")
            skip_output = os.path.join(temp_dir, "skip.mp4")
            full_output = os.path.join(temp_dir, "full.mp4")
            
            # Create a dummy video file
            with open(video_path, 'wb') as f:
                f.write(b'dummy video data')
            
            # Create test arguments
            args = argparse.Namespace(
                video=video_path,
                skip_output=skip_output,
                full_output=full_output,
                detection_threshold=0.3,
                yolo_threshold=0.2,
                max_age=5,
                min_hits=2,
                iou_threshold=0.2,
                embedding_threshold=0.7,
                skip=1,
                multi_view_stride=1,
                preserve_audio=False
            )
            
            # Process the video
            process_video(video_path, skip_output, full_output, args)
            
            # Verify video capture was opened
            mock_video_capture.assert_called_once_with(video_path)
            
            # Verify video writer was created
            mock_video_writer.assert_called()
            
            # Verify frames were processed
            self.assertEqual(mock_cap.read.call_count, 3)  # Two frames + one to detect end
            self.assertEqual(mock_process_frame.call_count, 2)
            
            # Verify video writer was released
            mock_writer.release.assert_called_once()
            
            # Verify video capture was released
            mock_cap.release.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_process_video_invalid_file(self, mock_video_capture):
        """Test video processing with invalid input file."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, "nonexistent.mp4")
            skip_output = os.path.join(temp_dir, "skip.mp4")
            full_output = os.path.join(temp_dir, "full.mp4")
            
            args = argparse.Namespace(
                video=video_path,
                skip_output=skip_output,
                full_output=full_output,
                detection_threshold=0.3,
                yolo_threshold=0.2,
                max_age=5,
                min_hits=2,
                iou_threshold=0.2,
                embedding_threshold=0.7,
                skip=1,
                multi_view_stride=1,
                preserve_audio=False
            )
            
            with self.assertRaises(Exception):
                process_video(video_path, skip_output, full_output, args)

if __name__ == '__main__':
    unittest.main() 