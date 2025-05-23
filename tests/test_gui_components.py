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
        # Create a mock Tk instance with all required attributes
        cls.root = MagicMock(spec=tk.Tk)
        cls.root.title = MagicMock()
        cls.root.quit = MagicMock()
        cls.root.geometry = MagicMock()
        cls.root.grid_columnconfigure = MagicMock()
        cls.root.grid_rowconfigure = MagicMock()
        
        # Mock the tk attribute that Tkinter widgets need
        cls.root.tk = MagicMock()
        cls.root.tk.call = MagicMock()
        cls.root.tk.eval = MagicMock()
        
        # Mock ttk.Style
        cls.mock_style = MagicMock(spec=ttk.Style)
        cls.mock_style.configure = MagicMock()
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Create mock widgets
        self.mock_frame = MagicMock(spec=ttk.Frame)
        self.mock_label_frame = MagicMock(spec=ttk.LabelFrame)
        self.mock_listbox = MagicMock(spec=tk.Listbox)
        self.mock_scrollbar = MagicMock(spec=ttk.Scrollbar)
        self.mock_button = MagicMock(spec=ttk.Button)
        self.mock_entry = MagicMock(spec=ttk.Entry)
        self.mock_checkbutton = MagicMock(spec=ttk.Checkbutton)
        self.mock_text = MagicMock(spec=tk.Text)
        self.mock_label = MagicMock(spec=ttk.Label)
        self.mock_boolean_var = MagicMock(spec=tk.BooleanVar)
        
        # Set up mock widget behaviors
        self.mock_listbox.size = MagicMock(return_value=0)
        self.mock_listbox.get = MagicMock(return_value="")
        self.mock_listbox.insert = MagicMock()
        self.mock_listbox.selection_set = MagicMock()
        self.mock_listbox.delete = MagicMock()
        self.mock_listbox.curselection = MagicMock(return_value=[])
        self.mock_listbox.yview = MagicMock()
        
        self.mock_entry.get = MagicMock(return_value="")
        self.mock_entry.insert = MagicMock()
        
        self.mock_text.get = MagicMock(return_value="")
        self.mock_text.insert = MagicMock()
        self.mock_text.delete = MagicMock()
        
        # Create the app with all mocked components
        with patch('tkinter.ttk.Frame', return_value=self.mock_frame), \
             patch('tkinter.ttk.LabelFrame', return_value=self.mock_label_frame), \
             patch('tkinter.Listbox', return_value=self.mock_listbox), \
             patch('tkinter.ttk.Scrollbar', return_value=self.mock_scrollbar), \
             patch('tkinter.ttk.Button', return_value=self.mock_button), \
             patch('tkinter.ttk.Entry', return_value=self.mock_entry), \
             patch('tkinter.ttk.Checkbutton', return_value=self.mock_checkbutton), \
             patch('tkinter.Text', return_value=self.mock_text), \
             patch('tkinter.ttk.Label', return_value=self.mock_label), \
             patch('tkinter.BooleanVar', return_value=self.mock_boolean_var), \
             patch('tkinter.ttk.Style', return_value=self.mock_style):
            self.app = FacebeakGUI(self.root)
        
    def tearDown(self):
        """Clean up after each test."""
        # No need to check quit for mocked root
        pass
        
    def test_initialization(self):
        """Test GUI initialization."""
        self.root.title.assert_called_with("facebeak Launcher")
        self.root.geometry.assert_called_with("900x1100")
        self.assertIsNotNone(self.app.video_listbox)
        
    @patch('tkinter.filedialog.askopenfilenames')
    def test_browse_videos(self, mock_askopenfilenames):
        """Test video file browsing."""
        mock_askopenfilenames.return_value = ["video1.mp4", "video2.mp4"]
        self.app.browse_videos()
        
        # Verify listbox operations
        self.assertEqual(self.mock_listbox.insert.call_count, 2)
        self.mock_listbox.insert.assert_has_calls([
            call('end', "video1.mp4"),
            call('end', "video2.mp4")
        ])
        
    def test_remove_selected_videos(self):
        """Test removing selected videos."""
        # Setup mock listbox state
        self.mock_listbox.size.return_value = 3
        self.mock_listbox.get.side_effect = ["video1.mp4", "video2.mp4", "video3.mp4"]
        self.mock_listbox.curselection.return_value = [1]  # Select middle item
        
        self.app.remove_selected_videos()
        
        # Verify listbox operations
        self.mock_listbox.delete.assert_called_once_with(1)
        
    def test_clear_videos(self):
        """Test clearing all videos."""
        # Setup mock listbox state
        self.mock_listbox.size.return_value = 2
        
        self.app.clear_videos()
        
        # Verify listbox operations
        self.mock_listbox.delete.assert_called_once_with(0, 'end')

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