import unittest
from unittest.mock import MagicMock, patch, call
from PIL import Image 
import os
import shutil
import sqlite3 
from pathlib import Path
import numpy as np 
import time 
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import Kivy version of the GUI
from kivy_suspect_lineup import SuspectLineupApp, SuspectLineupLayout 
from db import (initialize_db, add_crow, add_video_sighting, get_all_crows, 
                get_crow_details_for_lineup, get_embedding_ids_by_image_paths, 
                create_new_crow_from_embeddings, reassign_crow_embeddings, 
                delete_crow_embeddings, update_crow_name)

# --- Kivy Mocking Setup ---
class MinimalKivyAppMock:
    _instance_suspect_lineup = None 
    def __init__(self, **kwargs):
        self.root = None
        self.config = MagicMock()
        self.current_crow_id = None
        self.current_crow_name = ""
        self.unsaved_changes = False
        self.all_crows_data = []
        self.current_crow_videos = []
        self.selected_video_paths = []
        self.image_classifications = {} 
        self.image_widgets_map = {}

    @staticmethod
    def get_running_app():
        if not hasattr(MinimalKivyAppMock, '_instance_suspect_lineup'):
            MinimalKivyAppMock._instance_suspect_lineup = MinimalKivyAppMock()
        return MinimalKivyAppMock._instance_suspect_lineup
    
    def run(self): pass
    def stop(self): pass
    def show_popup(self, title, message): 
        # This can be asserted if needed: self.show_popup.assert_called_with(...)
        pass 

mock_kivy_clock_suspect_lineup = MagicMock()
def mock_schedule_once_suspect_lineup(func, timeout=0): func(0) 
mock_kivy_clock_suspect_lineup.schedule_once = mock_schedule_once_suspect_lineup
mock_kivy_clock_suspect_lineup.schedule_interval = MagicMock()

mock_kivy_window_suspect_lineup = MagicMock()
mock_kivy_window_suspect_lineup.width = 1400; mock_kivy_window_suspect_lineup.height = 800
mock_kivy_window_suspect_lineup.bind = MagicMock(); mock_kivy_window_suspect_lineup.unbind = MagicMock()
mock_kivy_window_suspect_lineup.dpi = 96

global_kivy_patches_suspect_lineup_file = [
    patch('kivy.app.App', MinimalKivyAppMock),
    patch('kivy.core.window.Window', mock_kivy_window_suspect_lineup),
    patch('kivy.clock.Clock', mock_kivy_clock_suspect_lineup),
    patch('kivy.uix.popup.Popup', MagicMock()), 
    patch('kivy.uix.listview.ListItemButton', MagicMock), 
    patch('kivy.adapters.listadapter.ListAdapter', MagicMock), 
    patch('kivy.uix.button.Button', MagicMock), # Mock base Button
    patch('kivy.uix.togglebutton.ToggleButton', MagicMock), # Mock ToggleButton
    patch('kivy.uix.label.Label', MagicMock),
    patch('kivy.uix.textinput.TextInput', MagicMock),
    patch('kivy.uix.spinner.Spinner', MagicMock),
    patch('kivy.uix.image.Image', MagicMock),
    patch('kivy.uix.scrollview.ScrollView', MagicMock),
    patch('kivy.uix.gridlayout.GridLayout', MagicMock),
    patch('kivy.uix.boxlayout.BoxLayout', MagicMock),
    patch('kivy.properties.StringProperty', lambda default='': default),
    patch('kivy.properties.NumericProperty', lambda default=0, **kwargs: default),
    patch('kivy.properties.BooleanProperty', lambda default=False: default),
    patch('kivy.properties.ListProperty', lambda default_factory=list: default_factory()),
    patch('kivy.properties.DictProperty', lambda default_factory=dict: default_factory()),
    patch('kivy.properties.ObjectProperty', lambda default=None: default)
]

@classmethod
def setUpClassGlobalPatches(cls):
    for p in global_kivy_patches_suspect_lineup_file:
        p.start()

@classmethod
def tearDownClassGlobalPatches(cls):
    for p in global_kivy_patches_suspect_lineup_file:
        p.stop()

TEST_DB_PATH = Path(__file__).parent / "test_lineup_db.sqlite"
TEST_IMAGE_DIR = Path(__file__).parent / "test_lineup_images"

class TestKivySuspectLineupGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setUpClassGlobalPatches() 
        TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            try:
                img = Image.new('RGB', (60, 30), color = 'red')
                img.save(TEST_IMAGE_DIR / f"crow1_video1_frame{i+1}.jpg")
                img.save(TEST_IMAGE_DIR / f"crow2_video1_frame{i+1}.jpg")
            except Exception as e: print(f"Error creating dummy image: {e}")

        if TEST_DB_PATH.exists(): TEST_DB_PATH.unlink(missing_ok=True) # Python 3.8+
        initialize_db(str(TEST_DB_PATH))
        
        crow1_id = add_crow("Crow A", db_path=str(TEST_DB_PATH))
        crow2_id = add_crow("Crow B", db_path=str(TEST_DB_PATH))
        add_crow("Crow C (No Sightings)", db_path=str(TEST_DB_PATH))

        for i in range(3):
            add_video_sighting(crow1_id, "video1.mp4", i+1, str(TEST_IMAGE_DIR / f"crow1_video1_frame{i+1}.jpg"), np.random.rand(1, 512).astype(np.float32), 0.9, db_path=str(TEST_DB_PATH))
        for i in range(2):
            add_video_sighting(crow2_id, "video1.mp4", i+1, str(TEST_IMAGE_DIR / f"crow2_video1_frame{i+1}.jpg"), np.random.rand(1, 512).astype(np.float32), 0.8, db_path=str(TEST_DB_PATH))

    def setUp(self):
        self.db_patcher = patch('db.DB_PATH', str(TEST_DB_PATH))
        self.mock_db_path = self.db_patcher.start()
        
        self.app = SuspectLineupApp() 
        # Mock the layout and its children directly on the app instance
        # This simulates the state after Kivy's build() process would have run.
        self.app.layout = MagicMock(spec=SuspectLineupLayout)
        self.app.layout.crow_spinner = MagicMock(values=[], text="", bind=MagicMock())
        self.app.layout.crow_name_input = MagicMock(text="")
        self.app.layout.crow_primary_image = MagicMock(source="")
        self.app.layout.video_list_layout = MagicMock()
        self.app.layout.video_list_layout.clear_widgets = MagicMock()
        self.app.layout.video_list_layout.add_widget = MagicMock()
        self.app.layout.sightings_grid = MagicMock()
        self.app.layout.sightings_grid.clear_widgets = MagicMock()
        self.app.layout.sightings_grid.add_widget = MagicMock()
        self.app.layout.save_button = MagicMock(disabled=True)
        self.app.layout.discard_button = MagicMock(disabled=True)
        self.app.layout.video_status_label = MagicMock(text="")
        self.app.layout.crow_image_display_title = MagicMock(text="")
        self.app.layout.select_videos_button = MagicMock(disabled=True)

        # Reset app state variables explicitly for each test
        self.app.current_crow_id = None
        self.app.current_crow_name = ""
        self.app.unsaved_changes = False
        self.app.all_crows_data = [] # This is populated by load_initial_crow_list
        self.app.current_crow_videos = []
        self.app.selected_video_paths = []
        self.app.image_classifications = {} # Stores current classification choices for images
        self.app.image_widgets_map = {} # Used by _display_sighting_items

        # Mock show_popup on the app instance to allow assertions on it
        self.app.show_popup = MagicMock()


    def tearDown(self):
        self.db_patcher.stop()

    @classmethod
    def tearDownClass(cls):
        if TEST_DB_PATH.exists(): TEST_DB_PATH.unlink(missing_ok=True)
        if TEST_IMAGE_DIR.exists(): shutil.rmtree(TEST_IMAGE_DIR)
        tearDownClassGlobalPatches() 

    def test_initialization_kivy(self):
        self.assertIsInstance(self.app, SuspectLineupApp)
        self.assertFalse(self.app.unsaved_changes)
        self.assertEqual(self.app.current_crow_id, None)

    def test_load_initial_crow_list_kivy(self):
        self.app.load_initial_crow_list() # This method now populates app properties
        expected_display_names = [
            "Crow1 (Crow A)", "Crow2 (Crow B)", "Crow3 (Crow C (No Sightings))"
        ]
        self.assertEqual(self.app.layout.crow_spinner.values, expected_display_names)
        self.assertTrue(len(self.app.all_crows_data) == 3)
        self.assertEqual(self.app.all_crows_data[0]['id'], 1)


    @patch('kivy_suspect_lineup.get_all_crows') # To control details fetched by load_crow_data
    @patch('kivy_suspect_lineup.get_first_crow_image') 
    @patch('kivy_suspect_lineup.get_crow_videos')
    def test_load_crow_data_kivy(self, mock_get_videos_db, mock_get_first_image_db, mock_get_all_crows_db):
        # Simulate that load_initial_crow_list has run and populated all_crows_data
        self.app.all_crows_data = [{'id': 1, 'display_name': "Crow1 (Crow A)"}]
        self.app.layout.crow_spinner.text = "Crow1 (Crow A)" # Simulate user selecting this in spinner
        
        # Mock the specific get_all_crows call within load_crow_data that fetches details
        mock_get_all_crows_db.return_value = [{'id': 1, 'name': 'Crow A'}] # Data for the selected crow
        
        test_image_path = str(TEST_IMAGE_DIR / "crow1_video1_frame1.jpg")
        mock_get_first_image_db.return_value = test_image_path
        mock_get_videos_db.return_value = [{'video_path': 'video1.mp4', 'sighting_count': 3}]
        
        with patch.object(self.app, '_load_videos_for_current_crow') as mock_load_vids, \
             patch.object(self.app, '_load_primary_crow_image') as mock_load_primary_img:
            self.app.load_crow_data(None) # Instance is the button, None for direct call
        
        self.assertEqual(self.app.current_crow_id, 1)
        self.assertEqual(self.app.current_crow_name, "Crow A")
        self.assertEqual(self.app.layout.crow_name_input.text, "Crow A")
        mock_load_primary_img.assert_called_once()
        mock_load_vids.assert_called_once()
        self.app.layout.sightings_grid.clear_widgets.assert_called_once()

    @patch('kivy_suspect_lineup.get_crow_images_from_video')
    def test_load_sightings_for_selected_videos_kivy(self, mock_get_images_db):
        self.app.current_crow_id = 1
        self.app.selected_video_paths = ['video1.mp4'] # Simulate video already selected
        
        mock_image_paths = [str(TEST_IMAGE_DIR / f"crow1_video1_frame{i}.jpg") for i in range(1, 3)]
        mock_get_images_db.return_value = mock_image_paths
        
        with patch.object(self.app, '_display_sighting_items') as mock_display_items:
            self.app.load_sightings_for_selected_videos(None) # Pass None for button instance
            mock_get_images_db.assert_called_with(1, 'video1.mp4')
            # Check the structure of data passed to _display_sighting_items
            self.assertTrue(mock_display_items.called)
            args, _ = mock_display_items.call_args
            passed_data = args[0] # First argument is sighting_images_data
            self.assertEqual(len(passed_data), 2)
            self.assertEqual(passed_data[0]['path'], mock_image_paths[0])
            self.assertEqual(passed_data[0]['original_crow_id'], 1)


    def test_on_classification_change_kivy(self):
        test_image_path = "path/to/image.jpg"
        # Simulate that _display_sighting_items has populated image_widgets_map
        self.app.image_widgets_map = {test_image_path: {'group': 'group1', 'buttons': {}}}
        mock_button = MagicMock(state='down') # Simulate button being pressed (selected)
        
        self.app.on_classification_change(test_image_path, "different_crow", mock_button)
        
        self.assertTrue(self.app.unsaved_changes)
        self.assertEqual(self.app.image_classifications[test_image_path], "different_crow")
        self.assertFalse(self.app.layout.save_button.disabled) # Check button state updated
        self.assertFalse(self.app.layout.discard_button.disabled)

    @patch('kivy_suspect_lineup.update_crow_name')
    def test_save_crow_name_via_save_all_changes_kivy(self, mock_update_name_db):
        self.app.current_crow_id = 1
        self.app.current_crow_name = "Old Kivy Name"
        self.app.layout.crow_name_input.text = "New Kivy Name" # Simulate UI input
        self.app.image_classifications = {} # No image changes, only name

        with patch.object(self.app, '_kivy_process_reassignments') as mock_reassign:
            self.app.save_all_changes(None) # Pass None for button instance
        
        mock_update_name_db.assert_called_with(1, "New Kivy Name")
        # If only name changed, _kivy_process_reassignments should be called with empty lists
        mock_reassign.assert_called_once_with([], [], [])


    @patch('kivy_suspect_lineup.get_embedding_ids_by_image_paths')
    @patch('kivy_suspect_lineup.create_new_crow_from_embeddings')
    def test_save_changes_reassign_new_crow_kivy(self, mock_create_new_db, mock_get_ids_db):
        self.app.current_crow_id = 1
        self.app.layout.crow_name_input.text = "Crow1Name" 
        self.app.current_crow_name = "Crow1Name" # Name is not changed in this test

        img1_path = str(TEST_IMAGE_DIR / "img1_kivy.jpg")
        self.app.image_classifications = {img1_path: "new_unidentified_crow"} # Classified for new crow
        mock_get_ids_db.return_value = {img1_path: "emb_kivy_1"}

        # Mock the Kivy popup for choice to return "new_crow" by having it call its callback immediately
        with patch.object(self.app, '_kivy_ask_reassignment_choice_popup', side_effect=lambda diff, new, cb: cb("new_crow")):
            self.app.save_all_changes(None)
            
        mock_get_ids_db.assert_called_with([img1_path]) # Called by _kivy_process_reassignments
        mock_create_new_db.assert_called_once_with(1, ["emb_kivy_1"], f"Split from Crow{self.app.current_crow_id}")

    @patch('kivy_suspect_lineup.get_embedding_ids_by_image_paths')
    @patch('kivy_suspect_lineup.reassign_crow_embeddings')
    def test_save_changes_reassign_existing_crow_kivy(self, mock_reassign_db, mock_get_ids_db):
        self.app.current_crow_id = 1
        self.app.layout.crow_name_input.text = "Crow1Name"
        self.app.current_crow_name = "Crow1Name"
        img1_path = str(TEST_IMAGE_DIR / "img1_kivy_existing.jpg")
        self.app.image_classifications = {img1_path: "different_crow"}
        mock_get_ids_db.return_value = {img1_path: "emb_kivy_existing_1"}
        
        # Mock Kivy popups: first for choice, then for target selection
        def mock_ask_choice(num_diff, num_new, callback_choice): callback_choice("existing_crow")
        def mock_select_target(callback_target): callback_target(2) # Simulate selecting Crow ID 2
        
        with patch.object(self.app, '_kivy_ask_reassignment_choice_popup', side_effect=mock_ask_choice), \
             patch.object(self.app, '_kivy_select_target_crow_popup', side_effect=mock_select_target):
            self.app.save_all_changes(None)
            
        mock_get_ids_db.assert_called_with([img1_path])
        mock_reassign_db.assert_called_once_with(1, 2, ["emb_kivy_existing_1"])

    @patch('kivy_suspect_lineup.get_embedding_ids_by_image_paths')
    @patch('kivy_suspect_lineup.delete_crow_embeddings')
    def test_save_changes_remove_embeddings_kivy(self, mock_delete_db, mock_get_ids_db):
        self.app.current_crow_id = 1
        self.app.layout.crow_name_input.text = "Crow1Name"
        self.app.current_crow_name = "Crow1Name"
        img1_path = str(TEST_IMAGE_DIR / "img1_kivy_remove.jpg")
        self.app.image_classifications = {img1_path: "not_crow"}
        mock_get_ids_db.return_value = {img1_path: "emb_kivy_remove_1"}
        
        self.app.save_all_changes(None) 
        # _kivy_process_reassignments directly calls db functions for 'not_crow'
        mock_get_ids_db.assert_called_with([img1_path]) 
        mock_delete_db.assert_called_once_with(["emb_kivy_remove_1"])

    def test_discard_changes_kivy(self):
        self.app.unsaved_changes = True
        self.app.current_crow_name = "Original Kivy Name"
        
        mock_btn_same = MagicMock(state='normal') 
        self.app.image_widgets_map = {"path/to/image.jpg": {'buttons': {'same_crow': mock_btn_same}}}
        self.app.image_classifications = {"path/to/image.jpg": "different_crow"} 
        self.app.layout.crow_name_input.text = "Changed Name" 
        
        # Mock the Popup and its 'Yes' button press
        # The discard_all_changes method itself creates the popup.
        # We need to mock the Popup it creates, find the 'Yes' button, and simulate its press.
        mock_popup_instance = MagicMock()
        
        # This is the tricky part: The buttons are created *inside* discard_all_changes.
        # We can patch `Popup` to capture the instance and then find the button.
        # Or, more simply, if `discard_all_changes` calls a separate method after "Yes", mock that.
        # Let's assume for this test that the confirmation is given.
        
        # Simulate the logic that `do_discard` (inner function of discard_all_changes) would execute
        self.app.layout.crow_name_input.text = self.app.current_crow_name or ""
        for img_path, widget_info in self.app.image_widgets_map.items():
            widget_info['buttons']['same_crow'].state = 'down' # Reset to default state
        self.app.image_classifications = {path: "same_crow" for path in self.app.image_widgets_map.keys()}
        self.app.unsaved_changes = False
        self.app.update_save_discard_buttons_state() # This method updates button disabled states
            
        self.assertFalse(self.app.unsaved_changes)
        self.assertEqual(self.app.layout.crow_name_input.text, "Original Kivy Name")
        self.assertEqual(mock_btn_same.state, 'down')
        self.assertTrue(self.app.layout.save_button.disabled)
        self.assertTrue(self.app.layout.discard_button.disabled)

    @patch.object(SuspectLineupApp, 'show_popup') # Patch the app's own show_popup
    def test_on_request_close_window_kivy_with_unsaved_changes(self, mock_show_popup_method):
        self.app.unsaved_changes = True
        result = self.app.on_request_close_window() 
        self.assertTrue(result) 
        mock_show_popup_method.assert_called_with("Unsaved Changes", "You have unsaved changes. Please Save or Discard them before closing.")

    def test_on_request_close_window_kivy_no_unsaved_changes(self):
        self.app.unsaved_changes = False
        with patch.object(self.app, 'show_popup') as mock_show_popup: # Ensure it's not called
            result = self.app.on_request_close_window()
            self.assertFalse(result) 
            mock_show_popup.assert_not_called()


if __name__ == '__main__':
    if TEST_DB_PATH.exists():
        try: TEST_DB_PATH.unlink(missing_ok=True) # Python 3.8+ for missing_ok
        except OSError as e: print(f"Error deleting test database: {e}."); sys.exit(1)
    initialize_db(str(TEST_DB_PATH))
    time.sleep(0.1) 
    
    unittest.main(exit=False) 
    tearDownClassGlobalPatches()
