import unittest
from unittest.mock import MagicMock, patch, call
import pytest
import os
from pathlib import Path
import sys
import json # For test_start_training_logic_kivy

# Add parent dir for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import Kivy GUI classes
from kivy_train_triplet_gui import TrainingApp as KivyTrainingApp, TrainingLayout
from kivy_extract_training_gui import CrowExtractorApp as KivyCrowExtractorApp, ExtractorLayout

from facebeak import FacebeakGUI, ensure_requirements # Keep for test_ensure_requirements

# --- Kivy Mocking Setup ---
class MinimalKivyAppMock:
    def __init__(self, **kwargs):
        self.root = None
        self.config = MagicMock() 
        # Add any app-level Kivy properties used by the tested apps if not already on the app class itself
        # Example: self.title = "" 

    @staticmethod
    def get_running_app():
        # Ensure a consistent mock app instance per test class or globally if appropriate
        if not hasattr(MinimalKivyAppMock, '_instance_gui_components'):
            MinimalKivyAppMock._instance_gui_components = MinimalKivyAppMock()
        return MinimalKivyAppMock._instance_gui_components
    
    def run(self): pass
    def stop(self): pass
    def show_popup(self, title, message): 
        # In tests, we might want to check if popups are called rather than seeing print
        # This can be done by assigning a MagicMock to this method on the instance
        # For now, basic print for debugging if needed during test writing.
        # print(f"Mocked App Popup (GUIComponents): {title} - {message}")
        pass


mock_kivy_clock = MagicMock()
def mock_schedule_once(func, timeout=0): func(0) 
mock_kivy_clock.schedule_once = mock_schedule_once
mock_kivy_clock.schedule_interval = MagicMock()

mock_kivy_window = MagicMock()
mock_kivy_window.width = 1200; mock_kivy_window.height = 900 # Example defaults
mock_kivy_window.bind = MagicMock(); mock_kivy_window.unbind = MagicMock()
mock_kivy_window.dpi = 96

# It's important that these mocks are active *before* the Kivy classes are imported
# or at least before they are instantiated if they use these globals at import time.
# Patching them globally here.
global_kivy_patches_components = [
    patch('kivy.app.App', MinimalKivyAppMock),
    patch('kivy.core.window.Window', mock_kivy_window),
    patch('kivy.clock.Clock', mock_kivy_clock),
    # Mocking Kivy properties if they are used standalone (less common)
    # Usually, they are attributes of Widgets, which would be mocked.
    patch('kivy.properties.StringProperty', lambda default='': default),
    patch('kivy.properties.NumericProperty', lambda default=0, **kwargs: default),
    patch('kivy.properties.BooleanProperty', lambda default=False: default),
    patch('kivy.properties.ListProperty', lambda default_factory=list: default_factory()),
    patch('kivy.properties.DictProperty', lambda default_factory=dict: default_factory()),
    patch('kivy.properties.ObjectProperty', lambda default=None: default),
    # Mocking Kivy UIX components that might be imported globally or used by layouts
    patch('kivy.uix.boxlayout.BoxLayout', MagicMock),
    patch('kivy.uix.gridlayout.GridLayout', MagicMock),
    patch('kivy.uix.scrollview.ScrollView', MagicMock),
    patch('kivy.uix.button.Button', MagicMock),
    patch('kivy.uix.label.Label', MagicMock),
    patch('kivy.uix.textinput.TextInput', MagicMock),
    patch('kivy.uix.slider.Slider', MagicMock),
    patch('kivy.uix.checkbox.CheckBox', MagicMock),
    patch('kivy.uix.progressbar.ProgressBar', MagicMock),
    patch('kivy.uix.image.Image', MagicMock),
    patch('kivy.uix.filechooser.FileChooserListView', MagicMock()), 
    patch('kivy.uix.popup.Popup', MagicMock()), 
    patch('kivy.garden.matplotlib.backend_kivyagg.FigureCanvasKivyAgg', MagicMock()) 
]

def setUpClassGlobalPatches():
    for p in global_kivy_patches_components:
        p.start()

def tearDownClassGlobalPatches():
    for p in global_kivy_patches_components:
        p.stop()

# Apply patches at module level
setUpClassGlobalPatches()

class TestKivyTrainingGUI(unittest.TestCase):
    def setUp(self):
        self.app = KivyTrainingApp()
        self.app.layout = MagicMock(spec=TrainingLayout)
        
        # Mock UI elements on the layout that app methods interact with
        self.app.layout.crop_dir_text_input = MagicMock(text="mock_crop_dir") # This was from TrainingLayout's _create_dir_selector
        self.app.layout.audio_dir_text_input = MagicMock(text="mock_audio_dir")
        self.app.layout.output_dir_text_input = MagicMock(text="mock_output_dir")
        
        # For parameters that are children of a layout created by _create_param_input
        # The TextInput is the second child (index 1, or children[0] if Kivy adds in reverse for some layouts)
        # Let's assume children[0] is the TextInput for simplicity in mocking.
        self.app.layout.batch_size_input = MagicMock(children=[MagicMock(text="32")]) 
        self.app.layout.lr_input = MagicMock(children=[MagicMock(text="0.0001")])
        self.app.layout.epochs_input = MagicMock(children=[MagicMock(text="10")])
        self.app.layout.patience_input = MagicMock(children=[MagicMock(text="3")])
        self.app.layout.embed_dim_input = MagicMock(children=[MagicMock(text="256")])
        
        self.app.layout.val_split_slider = MagicMock(value=0.15)
        self.app.layout.margin_slider = MagicMock(value=0.5)
        
        self.app.layout.start_button = MagicMock()
        self.app.layout.pause_button = MagicMock()
        self.app.layout.stop_button = MagicMock()
        self.app.layout.progress_bar = MagicMock()
        self.app.layout.progress_label = MagicMock()
        self.app.layout.status_label = MagicMock()
        self.app.layout.metrics_labels = { 
            'epoch': MagicMock(), 'train_loss': MagicMock(), 'val_loss': MagicMock(),
            'same_crow_sim': MagicMock(), 'diff_crow_sim': MagicMock(),
            'time_elapsed': MagicMock(), 'best_val_loss': MagicMock()
        }
        self.app.layout.loss_ax = MagicMock()
        self.app.layout.sim_ax = MagicMock()
        self.app.layout.plot_canvas_widget = MagicMock()

        self.app.logger = MagicMock() 
        self.app.session_id = "test_kivy123_train"

    def test_initialization_kivy(self):
        self.assertIsNotNone(self.app)
        self.assertEqual(self.app.session_id, "test_kivy123_train")
        self.assertFalse(self.app.training_active)

    @patch('kivy_train_triplet_gui.os.path.isdir', return_value=True)
    @patch('kivy_train_triplet_gui.os.makedirs')
    @patch('kivy_train_triplet_gui.json.dump')
    @patch("builtins.open", new_callable=MagicMock)
    @patch('kivy_train_triplet_gui.threading.Thread')
    def test_start_training_logic_kivy(self, mock_thread, mock_open_file, mock_json_dump, mock_os_makedirs, mock_os_path_isdir):
        # _get_training_config is called by _start_training. 
        # It reads from self.app.layout.xxx widgets which are mocked in setUp.
        
        self.app._start_training(None) 
        
        self.assertTrue(self.app.training_active)
        # Use the text from the mocked TextInput for assertions
        mock_os_makedirs.assert_called_with(self.app.layout.output_dir_text_input.text, exist_ok=True)
        mock_open_file.assert_called_with(os.path.join(self.app.layout.output_dir_text_input.text, 'training_config.json'), 'w')
        self.assertTrue(mock_json_dump.called)
        
        retrieved_config = self.app._get_training_config() 
        mock_thread.assert_called_once_with(target=self.app._training_loop, args=(retrieved_config,), daemon=True)
        
        self.assertTrue(self.app.layout.start_button.disabled)
        self.assertFalse(self.app.layout.pause_button.disabled)
        self.assertFalse(self.app.layout.stop_button.disabled)
        self.assertEqual(self.app.layout.status_label.text, "Training in progress...")


class TestKivyCrowExtractorGUI(unittest.TestCase):
    def setUp(self):
        self.app = KivyCrowExtractorApp()
        self.app.layout = MagicMock(spec=ExtractorLayout)
        # Mock UI elements that the app logic might interact with from ExtractorLayout
        self.app.layout.video_dir_text = MagicMock(spec=TextInput, text="")
        self.app.layout.output_dir_text = MagicMock(spec=TextInput, text="crow_crops")
        self.app.layout.review_crops_button = MagicMock()
        self.app.layout.enable_audio_check = MagicMock(active=True)
        self.app.layout.audio_duration_slider = MagicMock(value=2.0)
        self.app.layout.orientation_check = MagicMock(active=True)
        # Add any other elements from ExtractorLayout that app methods touch


    def test_initialization_kivy(self):
        self.assertIsNotNone(self.app)
        self.assertEqual(self.app.output_dir, "crow_crops") # Check Kivy StringProperty default

    @patch('kivy_extract_training_gui.os.path.isdir', return_value=True)
    @patch.object(KivyCrowExtractorApp, '_select_dir_popup') 
    def test_select_directory_actions_kivy(self, mock_select_dir_popup_method, mock_os_path_isdir):
        self.app._select_video_dir_action(None) 
        args, _ = mock_select_dir_popup_method.call_args
        self.assertTrue(callable(args[0])) 
        self.assertEqual(args[1], 'video_dir') 
        args[0]("/test/video_kivy_selected") 
        self.assertEqual(self.app.video_dir, "/test/video_kivy_selected")

        mock_select_dir_popup_method.reset_mock()
        with patch.object(self.app, '_initialize_tracker') as mock_init_tracker:
            self.app._select_output_dir_action(None)
            args_out, _ = mock_select_dir_popup_method.call_args
            self.assertTrue(callable(args_out[0]))
            self.assertEqual(args_out[1], 'output_dir')
            args_out[0]("/test/output_kivy_selected")
            self.assertEqual(self.app.output_dir, "/test/output_kivy_selected")
            mock_init_tracker.assert_called_once()


@unittest.skip("FacebeakGUI is Tkinter-based and out of scope for Kivy rewrite")
class TestFacebeakGUI(unittest.TestCase):
    def test_initialization(self): pass
    def test_browse_videos(self): pass
    def test_remove_selected_videos(self): pass
    def test_clear_videos(self): pass

@pytest.mark.unit
def test_ensure_requirements():
    """Test requirements installation check."""
    with patch('subprocess.check_call') as mock_check_call:
        with patch.dict(sys.modules, {'cryptography': None}): 
            ensure_requirements()
            mock_check_call.assert_called_once()
        
        mock_check_call.reset_mock()
        sys.modules['cryptography'] = MagicMock() 
        ensure_requirements()
        mock_check_call.assert_not_called()
        del sys.modules['cryptography'] 

if __name__ == '__main__':
    unittest.main()

# Ensure to call tearDownClassGlobalPatches at the very end if running directly
# or if test runner doesn't handle it. For unittest.main(), it's tricky.
# A common pattern is to register it with atexit if needed for standalone runs.
import atexit
atexit.register(tearDownClassGlobalPatches)
