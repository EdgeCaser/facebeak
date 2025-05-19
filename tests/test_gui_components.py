import unittest
from unittest.mock import MagicMock, patch, call
import tkinter as tk
from tkinter import ttk
import pytest
from train_triplet_gui import TrainingGUI
from extract_training_gui import CrowExtractorGUI
from gui_launcher import FacebeakGUI, ensure_requirements
import os
from pathlib import Path

class TestTrainingGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.root = tk.Tk()
        cls.logger = MagicMock()
        cls.session_id = "test123"
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        self.app = TrainingGUI(self.root, self.logger, self.session_id)
        
    def tearDown(self):
        """Clean up after each test."""
        self.app.root.quit()
        
    def test_initialization(self):
        """Test GUI initialization."""
        self.assertEqual(self.app.session_id, self.session_id)
        self.assertEqual(self.app.logger, self.logger)
        self.assertFalse(self.app.training)
        self.assertFalse(self.app.paused)
        self.assertIsNotNone(self.app.metrics)
        
    @patch('tkinter.filedialog.askdirectory')
    def test_directory_selection(self, mock_askdirectory):
        """Test directory selection dialogs."""
        # Test crop directory selection
        mock_askdirectory.return_value = "/test/crop/dir"
        self.app._select_crop_dir()
        self.assertEqual(self.app.crop_dir_var.get(), "/test/crop/dir")
        
        # Test audio directory selection
        mock_askdirectory.return_value = "/test/audio/dir"
        self.app._select_audio_dir()
        self.assertEqual(self.app.audio_dir_var.get(), "/test/audio/dir")
        
        # Test output directory selection
        mock_askdirectory.return_value = "/test/output/dir"
        self.app._select_output_dir()
        self.assertEqual(self.app.output_dir_var.get(), "/test/output/dir")

class TestCrowExtractorGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.root = tk.Tk()
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        self.app = CrowExtractorGUI(self.root)
        
    def tearDown(self):
        """Clean up after each test."""
        self.app.root.quit()
        
    def test_initialization(self):
        """Test GUI initialization."""
        self.assertFalse(self.app.processing)
        self.assertFalse(self.app.paused)
        self.assertIsNone(self.app.cap)
        self.assertIsNone(self.app.current_video)
        
    @patch('tkinter.filedialog.askdirectory')
    def test_directory_selection(self, mock_askdirectory):
        """Test directory selection dialogs."""
        # Test video directory selection
        mock_askdirectory.return_value = "/test/video/dir"
        self.app._select_video_dir()
        self.assertEqual(self.app.video_dir_var.get(), "/test/video/dir")
        
        # Test output directory selection
        mock_askdirectory.return_value = "/test/output/dir"
        self.app._select_output_dir()
        self.assertEqual(self.app.output_dir_var.get(), "/test/output/dir")

class TestFacebeakGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.root = tk.Tk()
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        self.app = FacebeakGUI(self.root)
        
    def tearDown(self):
        """Clean up after each test."""
        self.app.root.quit()
        
    def test_initialization(self):
        """Test GUI initialization."""
        self.assertEqual(self.app.root.title(), "facebeak Launcher")
        self.assertIsNotNone(self.app.video_listbox)
        
    @patch('tkinter.filedialog.askopenfilenames')
    def test_browse_videos(self, mock_askopenfilenames):
        """Test video file browsing."""
        mock_askopenfilenames.return_value = ["video1.mp4", "video2.mp4"]
        self.app.browse_videos()
        self.assertEqual(self.app.video_listbox.size(), 2)
        self.assertEqual(self.app.video_listbox.get(0), "video1.mp4")
        self.assertEqual(self.app.video_listbox.get(1), "video2.mp4")
        
    def test_remove_selected_videos(self):
        """Test removing selected videos."""
        # Add some videos
        self.app.video_listbox.insert(0, "video1.mp4")
        self.app.video_listbox.insert(1, "video2.mp4")
        self.app.video_listbox.insert(2, "video3.mp4")
        
        # Select and remove middle video
        self.app.video_listbox.selection_set(1)
        self.app.remove_selected_videos()
        
        self.assertEqual(self.app.video_listbox.size(), 2)
        self.assertEqual(self.app.video_listbox.get(0), "video1.mp4")
        self.assertEqual(self.app.video_listbox.get(1), "video3.mp4")
        
    def test_clear_videos(self):
        """Test clearing all videos."""
        # Add some videos
        self.app.video_listbox.insert(0, "video1.mp4")
        self.app.video_listbox.insert(1, "video2.mp4")
        
        self.app.clear_videos()
        self.assertEqual(self.app.video_listbox.size(), 0)

@pytest.mark.unit
def test_ensure_requirements():
    """Test requirements installation check."""
    with patch('subprocess.check_call') as mock_check_call:
        # Test when cryptography is not installed
        with patch.dict('sys.modules', {'cryptography': None}):
            ensure_requirements()
            mock_check_call.assert_called_once()
        
        # Test when cryptography is already installed
        mock_check_call.reset_mock()
        with patch.dict('sys.modules', {'cryptography': MagicMock()}):
            ensure_requirements()
            mock_check_call.assert_not_called()

if __name__ == '__main__':
    unittest.main() 