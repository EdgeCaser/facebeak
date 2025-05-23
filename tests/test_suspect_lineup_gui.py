import unittest
from unittest.mock import MagicMock, patch, call
import tkinter as tk
from tkinter import ttk
import sys
import os
import tempfile
from pathlib import Path

# Import the suspect lineup module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from suspect_lineup import SuspectLineup

class TestSuspectLineupGUI(unittest.TestCase):
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
        cls.root.protocol = MagicMock()
        cls.root.mainloop = MagicMock()
        
        # Mock the tk attribute that Tkinter widgets need
        cls.root.tk = MagicMock()
        cls.root.tk.call = MagicMock()
        cls.root.tk.eval = MagicMock()
        
        # Create temporary directory for test images
        cls.test_dir = tempfile.mkdtemp()
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Create comprehensive mock widgets
        self.mock_frame = MagicMock(spec=ttk.Frame)
        self.mock_label_frame = MagicMock(spec=ttk.LabelFrame)
        self.mock_listbox = MagicMock(spec=tk.Listbox)
        self.mock_scrollbar = MagicMock(spec=ttk.Scrollbar)
        self.mock_button = MagicMock(spec=ttk.Button)
        self.mock_entry = MagicMock(spec=ttk.Entry)
        self.mock_text = MagicMock(spec=tk.Text)
        self.mock_label = MagicMock(spec=ttk.Label)
        self.mock_string_var = MagicMock(spec=tk.StringVar)
        self.mock_combobox = MagicMock(spec=ttk.Combobox)
        self.mock_radiobutton = MagicMock(spec=ttk.Radiobutton)
        self.mock_canvas = MagicMock(spec=tk.Canvas)
        
        # Set up mock widget behaviors
        self.mock_combobox.get = MagicMock(return_value="")
        self.mock_combobox.set = MagicMock()
        self.mock_combobox['values'] = []
        
        self.mock_listbox.size = MagicMock(return_value=0)
        self.mock_listbox.get = MagicMock(return_value="")
        self.mock_listbox.insert = MagicMock()
        self.mock_listbox.delete = MagicMock()
        self.mock_listbox.curselection = MagicMock(return_value=[])
        self.mock_listbox.selection_set = MagicMock()
        
        self.mock_entry.get = MagicMock(return_value="")
        self.mock_entry.insert = MagicMock()
        self.mock_entry.delete = MagicMock()
        
        self.mock_button.configure = MagicMock()
        
        self.mock_string_var.get = MagicMock(return_value="")
        self.mock_string_var.set = MagicMock()
        
        # Mock database functions
        self.db_patches = [
            patch('suspect_lineup.get_all_crows', return_value=[]),
            patch('suspect_lineup.get_first_crow_image', return_value=None),
            patch('suspect_lineup.get_crow_videos', return_value=[]),
            patch('suspect_lineup.get_crow_images_from_video', return_value=[]),
            patch('suspect_lineup.update_crow_name'),
            patch('suspect_lineup.get_embedding_ids_by_image_paths', return_value={}),
            patch('suspect_lineup.reassign_crow_embeddings'),
            patch('suspect_lineup.create_new_crow_from_embeddings', return_value=1),
            patch('suspect_lineup.delete_crow_embeddings'),
        ]
        
        for patcher in self.db_patches:
            patcher.start()
        
        # Create the app with all mocked components
        with patch('tkinter.ttk.Frame', return_value=self.mock_frame), \
             patch('tkinter.ttk.LabelFrame', return_value=self.mock_label_frame), \
             patch('tkinter.Listbox', return_value=self.mock_listbox), \
             patch('tkinter.ttk.Scrollbar', return_value=self.mock_scrollbar), \
             patch('tkinter.ttk.Button', return_value=self.mock_button), \
             patch('tkinter.ttk.Entry', return_value=self.mock_entry), \
             patch('tkinter.Text', return_value=self.mock_text), \
             patch('tkinter.ttk.Label', return_value=self.mock_label), \
             patch('tkinter.StringVar', return_value=self.mock_string_var), \
             patch('tkinter.ttk.Combobox', return_value=self.mock_combobox), \
             patch('tkinter.ttk.Radiobutton', return_value=self.mock_radiobutton), \
             patch('tkinter.Canvas', return_value=self.mock_canvas), \
             patch('PIL.Image.open'), \
             patch('PIL.ImageTk.PhotoImage'):
            
            self.app = SuspectLineup(self.root)
            
    def tearDown(self):
        """Clean up after each test."""
        # Stop all database patches
        for patcher in self.db_patches:
            try:
                patcher.stop()
            except RuntimeError:
                pass  # Already stopped
                
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_initialization(self):
        """Test GUI initialization."""
        self.root.title.assert_called_with("Suspect Lineup - Crow ID Verification")
        self.root.geometry.assert_called_with("1400x900")
        self.assertIsNotNone(self.app.crow_dropdown)
        self.assertIsNotNone(self.app.video_listbox)
        self.assertFalse(self.app.unsaved_changes)
        self.assertEqual(self.app.image_classifications, {})
    
    @patch('suspect_lineup.get_all_crows')
    def test_load_crows(self, mock_get_all_crows):
        """Test loading crows into dropdown."""
        # Mock crow data
        mock_crows = [
            {'id': 1, 'name': 'Crow 1', 'total_sightings': 10},
            {'id': 2, 'name': 'Crow 2', 'total_sightings': 15},
        ]
        mock_get_all_crows.return_value = mock_crows
        
        self.app.load_crows()
        
        mock_get_all_crows.assert_called_once()
        self.assertEqual(len(self.app.crows), 2)
        
    @patch('suspect_lineup.get_first_crow_image')
    @patch('suspect_lineup.get_crow_videos')
    def test_load_crow(self, mock_get_crow_videos, mock_get_first_crow_image):
        """Test loading a specific crow."""
        # Setup mock data
        self.app.crows = [
            {'id': 1, 'name': 'Crow 1', 'total_sightings': 10}
        ]
        self.mock_combobox.current.return_value = 0
        
        mock_get_first_crow_image.return_value = "test_image.jpg"
        mock_get_crow_videos.return_value = [
            {'video_path': 'video1.mp4', 'sighting_count': 5},
            {'video_path': 'video2.mp4', 'sighting_count': 3},
        ]
        
        with patch.object(self.app, 'display_crow_image'):
            self.app.load_crow()
        
        mock_get_first_crow_image.assert_called_once_with(1)
        mock_get_crow_videos.assert_called_once_with(1)
        
    def test_load_crow_no_selection(self):
        """Test load_crow when no crow is selected."""
        self.mock_combobox.current.return_value = -1
        
        # Should not raise an error
        self.app.load_crow()
    
    @patch('suspect_lineup.get_crow_images_from_video')
    def test_load_videos(self, mock_get_crow_images_from_video):
        """Test loading videos and their images."""
        # Setup mock data
        self.app.current_crow_id = 1
        self.mock_listbox.curselection.return_value = [0]
        self.app.videos = [
            {'video_path': 'video1.mp4', 'sighting_count': 5}
        ]
        
        mock_get_crow_images_from_video.return_value = [
            "crop1.jpg", "crop2.jpg", "crop3.jpg"
        ]
        
        with patch.object(self.app, 'display_images'):
            self.app.load_videos()
        
        mock_get_crow_images_from_video.assert_called_once_with(1, 'video1.mp4')
        
    def test_load_videos_no_crow_selected(self):
        """Test load_videos when no crow is selected."""
        self.app.current_crow_id = None
        
        # Should not raise an error
        self.app.load_videos()
    
    def test_classify_image(self):
        """Test image classification."""
        image_path = "test_image.jpg"
        classification = "same_crow"
        
        self.app.classify_image(image_path, classification)
        
        self.assertEqual(self.app.image_classifications[image_path], classification)
        self.assertTrue(self.app.unsaved_changes)
        
    def test_reset_classifications(self):
        """Test resetting all classifications."""
        # Add some test classifications
        self.app.image_classifications = {
            "image1.jpg": "same_crow",
            "image2.jpg": "different_crow"
        }
        self.app.unsaved_changes = True
        
        self.app.reset_classifications()
        
        self.assertEqual(self.app.image_classifications, {})
        self.assertFalse(self.app.unsaved_changes)
    
    @patch('suspect_lineup.update_crow_name')
    def test_save_crow_name(self, mock_update_crow_name):
        """Test saving crow name."""
        self.app.current_crow_id = 1
        self.mock_entry.get.return_value = "New Crow Name"
        
        self.app.save_crow_name()
        
        mock_update_crow_name.assert_called_once_with(1, "New Crow Name")
    
    def test_save_crow_name_no_crow(self):
        """Test save_crow_name when no crow is selected."""
        self.app.current_crow_id = None
        
        # Should not raise an error
        self.app.save_crow_name()
    
    @patch('tkinter.messagebox.askyesno')
    @patch('suspect_lineup.get_embedding_ids_by_image_paths')
    @patch('suspect_lineup.create_new_crow_from_embeddings')
    @patch('suspect_lineup.delete_crow_embeddings')
    def test_save_changes_create_new_crow(self, mock_delete, mock_create_new, 
                                         mock_get_embedding_ids, mock_askyesno):
        """Test saving changes by creating a new crow."""
        # Setup test data
        self.app.image_classifications = {
            "image1.jpg": "different_crow",
            "image2.jpg": "not_a_crow",
            "image3.jpg": "same_crow"
        }
        
        mock_get_embedding_ids.return_value = {
            "image1.jpg": 1,
            "image2.jpg": 2,
            "image3.jpg": 3
        }
        
        mock_askyesno.return_value = True  # Choose "Create new crow"
        mock_create_new.return_value = 5  # New crow ID
        
        with patch.object(self.app, 'reset_classifications'), \
             patch.object(self.app, 'load_crows'):
            self.app.save_changes()
        
        # Verify different_crow images were moved to new crow
        mock_create_new.assert_called_once_with([1])
        # Verify not_a_crow images were deleted
        mock_delete.assert_called_once_with([2])
    
    @patch('tkinter.messagebox.askyesno')
    @patch('suspect_lineup.get_embedding_ids_by_image_paths')
    @patch('suspect_lineup.reassign_crow_embeddings')
    @patch('suspect_lineup.delete_crow_embeddings')
    def test_save_changes_move_to_existing_crow(self, mock_delete, mock_reassign,
                                               mock_get_embedding_ids, mock_askyesno):
        """Test saving changes by moving to existing crow."""
        # Setup test data
        self.app.image_classifications = {
            "image1.jpg": "different_crow",
            "image2.jpg": "not_a_crow"
        }
        
        mock_get_embedding_ids.return_value = {
            "image1.jpg": 1,
            "image2.jpg": 2
        }
        
        mock_askyesno.return_value = False  # Choose "Move to existing crow"
        
        with patch.object(self.app, 'select_target_crow', return_value=10), \
             patch.object(self.app, 'reset_classifications'), \
             patch.object(self.app, 'load_crows'):
            self.app.save_changes()
        
        # Verify different_crow images were moved to existing crow
        mock_reassign.assert_called_once_with([1], 10)
        # Verify not_a_crow images were deleted
        mock_delete.assert_called_once_with([2])
    
    def test_save_changes_no_changes(self):
        """Test save_changes when there are no changes."""
        self.app.image_classifications = {}
        
        with patch('tkinter.messagebox.showinfo') as mock_showinfo:
            self.app.save_changes()
        
        mock_showinfo.assert_called_once()
    
    @patch('tkinter.simpledialog.askstring')
    def test_select_target_crow(self, mock_askstring):
        """Test selecting target crow for reassignment."""
        self.app.crows = [
            {'id': 1, 'name': 'Crow 1'},
            {'id': 2, 'name': 'Crow 2'},
            {'id': 3, 'name': 'Crow 3'}
        ]
        
        mock_askstring.return_value = "2"
        
        result = self.app.select_target_crow()
        
        self.assertEqual(result, 2)
    
    def test_select_target_crow_invalid_input(self):
        """Test select_target_crow with invalid input."""
        self.app.crows = [{'id': 1, 'name': 'Crow 1'}]
        
        with patch('tkinter.simpledialog.askstring', return_value="invalid"), \
             patch('tkinter.messagebox.showerror') as mock_showerror:
            result = self.app.select_target_crow()
        
        mock_showerror.assert_called()
        self.assertIsNone(result)
    
    def test_select_target_crow_cancelled(self):
        """Test select_target_crow when user cancels."""
        with patch('tkinter.simpledialog.askstring', return_value=None):
            result = self.app.select_target_crow()
        
        self.assertIsNone(result)
    
    def test_update_save_buttons(self):
        """Test updating save button states."""
        # Test with changes
        self.app.unsaved_changes = True
        self.app.update_save_buttons()
        self.mock_button.configure.assert_called()
        
        # Test without changes
        self.app.unsaved_changes = False
        self.app.update_save_buttons()
        self.mock_button.configure.assert_called()
    
    @patch('os.path.exists')
    def test_display_crow_image_file_exists(self, mock_exists):
        """Test displaying crow image when file exists."""
        mock_exists.return_value = True
        
        with patch('PIL.Image.open') as mock_image_open, \
             patch('PIL.ImageTk.PhotoImage') as mock_photo_image:
            
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_img.resize.return_value = mock_img
            mock_image_open.return_value = mock_img
            
            self.app.display_crow_image("test_image.jpg")
            
            mock_image_open.assert_called_once_with("test_image.jpg")
            mock_photo_image.assert_called_once()
    
    @patch('os.path.exists')
    def test_display_crow_image_file_not_exists(self, mock_exists):
        """Test displaying crow image when file doesn't exist."""
        mock_exists.return_value = False
        
        # Should not raise an error
        self.app.display_crow_image("nonexistent_image.jpg")
    
    def test_display_crow_image_none(self):
        """Test displaying crow image with None path."""
        # Should not raise an error
        self.app.display_crow_image(None)
    
    @patch('os.path.exists')
    def test_display_images(self, mock_exists):
        """Test displaying multiple images."""
        mock_exists.return_value = True
        
        image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
        
        with patch('PIL.Image.open') as mock_image_open, \
             patch('PIL.ImageTk.PhotoImage') as mock_photo_image:
            
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_img.resize.return_value = mock_img
            mock_image_open.return_value = mock_img
            
            self.app.display_images(image_paths)
            
            # Should open each image
            self.assertEqual(mock_image_open.call_count, len(image_paths))
    
    def test_display_images_empty_list(self):
        """Test displaying empty image list."""
        # Should not raise an error
        self.app.display_images([])
    
    @patch('tkinter.messagebox.askyesno')
    def test_on_closing_with_unsaved_changes(self, mock_askyesno):
        """Test window closing with unsaved changes."""
        self.app.unsaved_changes = True
        mock_askyesno.return_value = True  # User confirms exit
        
        self.app.on_closing()
        
        mock_askyesno.assert_called_once()
        self.root.quit.assert_called_once()
    
    @patch('tkinter.messagebox.askyesno')
    def test_on_closing_cancel_with_unsaved_changes(self, mock_askyesno):
        """Test window closing cancelled due to unsaved changes."""
        self.app.unsaved_changes = True
        mock_askyesno.return_value = False  # User cancels exit
        
        self.app.on_closing()
        
        mock_askyesno.assert_called_once()
        self.root.quit.assert_not_called()
    
    def test_on_closing_no_unsaved_changes(self):
        """Test window closing without unsaved changes."""
        self.app.unsaved_changes = False
        
        self.app.on_closing()
        
        self.root.quit.assert_called_once()

if __name__ == '__main__':
    unittest.main() 