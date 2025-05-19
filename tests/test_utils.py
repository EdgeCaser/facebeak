import unittest
from unittest.mock import MagicMock, patch, call
import pytest
import numpy as np
import cv2
import os
import tempfile
from pathlib import Path
from utils import extract_frames, save_video_with_labels

class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Create a temporary directory for test files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)
        
        # Create a test video file
        cls.video_path = cls.test_dir / "test_video.mp4"
        cls._create_test_video(cls.video_path)
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Create output directory for each test
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()
        
    @staticmethod
    def _create_test_video(path, num_frames=30, fps=30):
        """Create a test video file with colored frames."""
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
        
        # Write colored frames
        for i in range(num_frames):
            # Create frame with different colors
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 8) % 256  # Blue channel
            frame[:, :, 1] = (i * 4) % 256  # Green channel
            frame[:, :, 2] = (i * 2) % 256  # Red channel
            out.write(frame)
            
        out.release()
        
    def test_extract_frames(self):
        """Test frame extraction from video."""
        # Extract frames
        frames = list(extract_frames(self.video_path, progress_callback=None))
        
        # Verify frame count
        self.assertEqual(len(frames), 30)  # Should match num_frames in test video
        
        # Verify frame properties
        for frame in frames:
            self.assertEqual(frame.shape, (480, 640, 3))  # Should match video dimensions
            self.assertEqual(frame.dtype, np.uint8)
            
    def test_extract_frames_with_callback(self):
        """Test frame extraction with progress callback."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
            
        # Extract frames with callback
        frames = list(extract_frames(self.video_path, progress_callback=progress_callback))
        
        # Verify callback was called
        self.assertGreater(len(progress_values), 0)
        self.assertLessEqual(max(progress_values), 1.0)  # Progress should be <= 100%
        self.assertGreaterEqual(min(progress_values), 0.0)  # Progress should be >= 0%
        
    def test_extract_frames_invalid_file(self):
        """Test frame extraction with invalid video file."""
        invalid_path = self.test_dir / "nonexistent.mp4"
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            list(extract_frames(invalid_path))
            
    def test_save_video_with_labels(self):
        """Test saving video with labels."""
        # Create test frames and labels
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        labels = [
            [{"bbox": [100, 100, 200, 200], "label": "Crow 1", "confidence": 0.95}]
            for _ in range(10)
        ]
        
        # Save video
        output_path = self.output_dir / "labeled_video.mp4"
        save_video_with_labels(
            frames,
            labels,
            output_path,
            fps=30,
            progress_callback=None
        )
        
        # Verify video was created
        self.assertTrue(output_path.exists())
        
        # Verify video properties
        cap = cv2.VideoCapture(str(output_path))
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FPS)), 30)
        cap.release()
        
    def test_save_video_with_callback(self):
        """Test saving video with progress callback."""
        # Create test frames and labels
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
        labels = [[{"bbox": [100, 100, 200, 200], "label": "Crow 1"}] for _ in range(10)]
        
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
            
        # Save video with callback
        output_path = self.output_dir / "labeled_video_callback.mp4"
        save_video_with_labels(
            frames,
            labels,
            output_path,
            fps=30,
            progress_callback=progress_callback
        )
        
        # Verify callback was called
        self.assertGreater(len(progress_values), 0)
        self.assertLessEqual(max(progress_values), 1.0)
        self.assertGreaterEqual(min(progress_values), 0.0)
        
    def test_save_video_empty_frames(self):
        """Test saving video with empty frame list."""
        # Should raise ValueError
        with self.assertRaises(ValueError):
            save_video_with_labels([], [], self.output_dir / "empty.mp4")
            
    def test_save_video_mismatched_frames_labels(self):
        """Test saving video with mismatched frames and labels."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        labels = [[{"bbox": [100, 100, 200, 200], "label": "Crow 1"}] for _ in range(3)]
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            save_video_with_labels(frames, labels, self.output_dir / "mismatched.mp4")
            
    def test_save_video_invalid_labels(self):
        """Test saving video with invalid label format."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
        labels = [[{"invalid": "format"}] for _ in range(5)]
        
        # Should raise KeyError for missing required label fields
        with self.assertRaises(KeyError):
            save_video_with_labels(frames, labels, self.output_dir / "invalid_labels.mp4")

if __name__ == '__main__':
    unittest.main() 