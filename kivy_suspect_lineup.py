import os
import logging
import threading
from pathlib import Path

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image as KivyImage
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import StringProperty, ObjectProperty, ListProperty, BooleanProperty, NumericProperty
from kivy.core.window import Window
from kivy.clock import Clock

from PIL import Image as PILImage # For image manipulation if needed, Kivy Image handles display

# Assuming db.py is in the same directory and contains the necessary functions
from db import (get_all_crows, get_first_crow_image, get_crow_videos, 
                get_crow_images_from_video, update_crow_name, 
                reassign_crow_embeddings, create_new_crow_from_embeddings,
                get_crow_embeddings, get_embedding_ids_by_image_paths,
                delete_crow_embeddings)

# Configure logging
logging.basicConfig(level=logging.INFO) # Kivy's default logger might be used instead/additionally
logger = logging.getLogger(__name__)


class SuspectLineupLayout(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = 'horizontal'
        self.padding = "10dp"
        self.spacing = "10dp"

        # Left Panel (Controls)
        left_panel = BoxLayout(orientation='vertical', size_hint_x=0.35, spacing="10dp")
        
        # Title
        left_panel.add_widget(Label(text="Select Crow ID", font_size='20sp', size_hint_y=None, height="40dp", bold=True))

        # Crow Selection
        crow_selection_layout = BoxLayout(size_hint_y=None, height="40dp", spacing="5dp")
        self.crow_spinner = Spinner(text="Select a Crow", size_hint_x=0.8)
        crow_selection_layout.add_widget(self.crow_spinner)
        self.load_crow_button = Button(text="Load", size_hint_x=0.2, on_press=self.app.load_crow_data)
        crow_selection_layout.add_widget(self.load_crow_button)
        left_panel.add_widget(crow_selection_layout)

        # Crow Image Display
        self.crow_image_display_frame = BoxLayout(orientation='vertical', size_hint_y=None, height="220dp") # Placeholder size
        self.crow_image_display_title = Label(text="Crow Primary Image", size_hint_y=None, height="20dp")
        self.crow_image_display_frame.add_widget(self.crow_image_display_title)
        self.crow_primary_image = KivyImage(source='', allow_stretch=True, keep_ratio=True)
        self.crow_image_display_frame.add_widget(self.crow_primary_image)
        left_panel.add_widget(self.crow_image_display_frame)
        
        # Crow Name Entry
        left_panel.add_widget(Label(text="Crow Name:", size_hint_y=None, height="20dp", halign='left'))
        self.crow_name_input = TextInput(multiline=False, size_hint_y=None, height="35dp")
        self.crow_name_input.bind(text=self.app.on_input_change)
        left_panel.add_widget(self.crow_name_input)

        # Video File Selection
        left_panel.add_widget(Label(text="Video Files:", size_hint_y=None, height="20dp", halign='left'))
        # Using a ScrollView for video list for now, ListView + Adapter is more robust for many items
        self.video_scroll = ScrollView(size_hint_y=0.3) 
        self.video_list_layout = GridLayout(cols=1, spacing="2dp", size_hint_y=None)
        self.video_list_layout.bind(minimum_height=self.video_list_layout.setter('height'))
        self.video_scroll.add_widget(self.video_list_layout)
        left_panel.add_widget(self.video_scroll)
        
        self.select_videos_button = Button(text="Load Sightings from Selected Videos", size_hint_y=None, height="40dp", on_press=self.app.load_sightings_for_selected_videos, disabled=True)
        left_panel.add_widget(self.select_videos_button)
        self.video_status_label = Label(text="Select a crow to see videos.", size_hint_y=None, height="25dp")
        left_panel.add_widget(self.video_status_label)

        # Save/Discard Buttons
        action_buttons_layout = BoxLayout(size_hint_y=None, height="40dp", spacing="10dp")
        self.save_button = Button(text="Save Changes", on_press=self.app.save_all_changes, disabled=True)
        action_buttons_layout.add_widget(self.save_button)
        self.discard_button = Button(text="Discard Changes", on_press=self.app.discard_all_changes, disabled=True)
        action_buttons_layout.add_widget(self.discard_button)
        left_panel.add_widget(action_buttons_layout)

        self.add_widget(left_panel)

        # Right Panel (Sightings)
        right_panel_layout = BoxLayout(orientation='vertical', spacing="5dp")
        right_panel_layout.add_widget(Label(text="Sightings - Confirm Crow Identity", font_size='20sp', size_hint_y=None, height="40dp", bold=True))
        
        self.sightings_scroll = ScrollView()
        self.sightings_grid = GridLayout(cols=1, spacing="10dp", size_hint_y=None) # Will display items in single column
        self.sightings_grid.bind(minimum_height=self.sightings_grid.setter('height'))
        self.sightings_scroll.add_widget(self.sightings_grid)
        right_panel_layout.add_widget(self.sightings_scroll)
        
        self.add_widget(right_panel_layout)


class SuspectLineupApp(App):
    # Properties to hold current state
    current_crow_id = NumericProperty(None, allownone=True)
    current_crow_name = StringProperty("")
    unsaved_changes = BooleanProperty(False)

    # Data lists
    all_crows_data = ListProperty([]) # List of dicts {id: x, name: 'CrowX (name)'}
    current_crow_videos = ListProperty([]) # List of dicts {video_id: x, video_path: 'path', sighting_count: y}
    selected_video_paths = ListProperty([]) # List of paths for selected videos
    
    # UI element references from layout (can also be accessed via self.root.ids if ids are set in kv)
    # For now, we'll pass `app` to layout and layout can store refs to its children.
    # Or, app can get them via self.root once layout is built.

    def build(self):
        self.title = "Suspect Lineup - Crow ID Verification (Kivy)"
        Window.size = (1400, 800) # Set initial window size
        self.layout = SuspectLineupLayout(app=self)
        Window.bind(on_request_close=self.on_request_close_window)
        return self.layout

    def on_start(self):
        logger.info("SuspectLineupApp started.")
        self.load_initial_crow_list()

    def load_initial_crow_list(self):
        logger.info("Loading initial crow list for spinner.")
        try:
            crows_db = get_all_crows() # [{'id': 1, 'name': 'Crow A', ...}, ...]
            self.all_crows_data = [{'id': c['id'], 'display_name': f"Crow{c['id']} ({c['name'] or 'Unnamed'})"} for c in crows_db]
            self.layout.crow_spinner.values = [c['display_name'] for c in self.all_crows_data]
            if self.all_crows_data:
                self.layout.crow_spinner.text = self.all_crows_data[0]['display_name'] # Default to first crow
            else:
                self.layout.crow_spinner.text = "No crows found"
        except Exception as e:
            logger.error(f"Error loading crow list: {e}", exc_info=True)
            self.show_popup("DB Error", f"Failed to load crow list: {e}")
            self.layout.crow_spinner.values = ["Error loading crows"]
            self.layout.crow_spinner.text = "Error loading crows"

    def load_crow_data(self, instance):
        logger.info(f"Load button pressed. Selected crow from spinner: {self.layout.crow_spinner.text}")
        selected_display_name = self.layout.crow_spinner.text
        
        selected_crow_obj = next((c for c in self.all_crows_data if c['display_name'] == selected_display_name), None)

        if not selected_crow_obj:
            self.show_popup("Selection Error", "Could not find data for selected crow.")
            return

        self.current_crow_id = selected_crow_obj['id']
        # Fetch full crow details again to ensure freshness, or use from all_crows_data if sufficient
        crows_db = get_all_crows() 
        crow_details = next((c for c in crows_db if c['id'] == self.current_crow_id), None)
        if crow_details:
            self.current_crow_name = crow_details['name'] or ""
            self.layout.crow_name_input.text = self.current_crow_name
            self.layout.crow_image_display_title.text = f"Crow {self.current_crow_id} ({self.current_crow_name or 'Unnamed'}) - Primary Image"
        else:
            self.show_popup("Error", f"Could not load details for Crow {self.current_crow_id}")
            return

        logger.info(f"Loading data for Crow ID: {self.current_crow_id}, Name: {self.current_crow_name}")
        self._load_primary_crow_image()
        self._load_videos_for_current_crow()
        self.layout.sightings_grid.clear_widgets() # Clear previous sightings
        self.unsaved_changes = False # Reset unsaved changes flag
        self.update_save_discard_buttons_state()


    def _load_primary_crow_image(self):
        if not self.current_crow_id: return
        try:
            image_path = get_first_crow_image(self.current_crow_id)
            if image_path and os.path.exists(image_path):
                self.layout.crow_primary_image.source = image_path
                self.layout.crow_primary_image.reload()
            else:
                self.layout.crow_primary_image.source = '' # Clear image if not found
                logger.warning(f"Primary image not found for crow {self.current_crow_id} at {image_path}")
        except Exception as e:
            logger.error(f"Error loading primary crow image: {e}", exc_info=True)
            self.layout.crow_primary_image.source = ''
            self.show_popup("Image Load Error", f"Failed to load primary image: {e}")

    def _load_videos_for_current_crow(self):
        if not self.current_crow_id: return
        self.layout.video_list_layout.clear_widgets()
        self.selected_video_paths = [] # Clear previous selections
        self.layout.select_videos_button.disabled = True
        try:
            self.current_crow_videos = get_crow_videos(self.current_crow_id) # [{'video_path': ..., 'sighting_count': ...}, ...]
            if not self.current_crow_videos:
                self.layout.video_status_label.text = "No videos found for this crow."
                return

            for video_info in self.current_crow_videos:
                video_path = video_info['video_path']
                sighting_count = video_info['sighting_count']
                display_text = f"{os.path.basename(video_path)} ({sighting_count} sightings)"
                
                # Using ToggleButton for selection behavior in a list
                video_button = ToggleButton(text=display_text, size_hint_y=None, height="30dp")
                video_button.video_path = video_path # Store path for later retrieval
                video_button.bind(on_press=self.on_video_button_selected)
                self.layout.video_list_layout.add_widget(video_button)
            self.layout.video_status_label.text = f"{len(self.current_crow_videos)} videos available. Select to load sightings."
        except Exception as e:
            logger.error(f"Error loading videos for crow {self.current_crow_id}: {e}", exc_info=True)
            self.show_popup("DB Error", f"Failed to load videos: {e}")
            self.layout.video_status_label.text = "Error loading videos."

    def on_video_button_selected(self, instance_button):
        video_path = instance_button.video_path
        if instance_button.state == 'down': # Selected
            if video_path not in self.selected_video_paths:
                self.selected_video_paths.append(video_path)
        else: # Deselected
            if video_path in self.selected_video_paths:
                self.selected_video_paths.remove(video_path)
        
        self.layout.select_videos_button.disabled = not bool(self.selected_video_paths)
        if self.selected_video_paths:
            self.layout.video_status_label.text = f"{len(self.selected_video_paths)} video(s) selected."
        else:
            self.layout.video_status_label.text = "No videos selected."
        logger.info(f"Selected video paths: {self.selected_video_paths}")


    def load_sightings_for_selected_videos(self, instance):
        logger.info("Loading sightings for selected videos.")
        if not self.current_crow_id or not self.selected_video_paths:
            self.show_popup("Info", "No videos selected or crow not loaded.")
            return
        
        self.layout.sightings_grid.clear_widgets()
        all_sighting_images = []
        for video_path in self.selected_video_paths:
            try:
                # Returns list of image paths for that crow in that video
                images_in_video = get_crow_images_from_video(self.current_crow_id, video_path)
                for img_path in images_in_video:
                    all_sighting_images.append({
                        'path': img_path,
                        'video_path': video_path, # Keep original full video path for context
                        'original_crow_id': self.current_crow_id # Store this for processing changes
                    })
            except Exception as e:
                logger.error(f"Error loading sightings from video {video_path}: {e}", exc_info=True)
                self.show_popup("Load Error", f"Failed to load sightings from {os.path.basename(video_path)}: {e}")
        
        self._display_sighting_items(all_sighting_images)


    def _display_sighting_items(self, sighting_images_data):
        self.layout.sightings_grid.clear_widgets() # Clear previous
        self.image_widgets_map = {} # To store {image_path: {classification_group_id: id, ...}}

        if not sighting_images_data:
            self.layout.sightings_grid.add_widget(Label(text="No sightings found in selected video(s) for this crow."))
            return

        for i, img_data in enumerate(sighting_images_data):
            item_layout = BoxLayout(orientation='vertical', size_hint_y=None, height="300dp", spacing="5dp", padding="5dp") # Increased height
            
            item_layout.add_widget(Label(text=os.path.basename(img_data['path']), size_hint_y=None, height="20dp"))
            
            sighting_image = KivyImage(source=img_data['path'], allow_stretch=True, keep_ratio=True, size_hint_y=0.7)
            item_layout.add_widget(sighting_image)
            
            # Classification ToggleButtons
            classification_group_id = f"classification_group_{i}"
            options_layout = GridLayout(cols=2, size_hint_y=0.3, spacing="2dp") # Adjust cols as needed

            # Store button references for this image
            self.image_widgets_map[img_data['path']] = {'group': classification_group_id, 'buttons': {}}

            btn_this_crow = ToggleButton(text=f"This is Crow {self.current_crow_id}", group=classification_group_id, state='down') # Default
            btn_this_crow.classification_value = "same_crow"
            btn_this_crow.bind(on_press=lambda x, p=img_data['path'], v="same_crow": self.on_classification_change(p,v,x))
            options_layout.add_widget(btn_this_crow)
            self.image_widgets_map[img_data['path']]['buttons']["same_crow"] = btn_this_crow
            
            btn_other_crow = ToggleButton(text="Another Known Crow", group=classification_group_id)
            btn_other_crow.classification_value = "different_crow"
            btn_other_crow.bind(on_press=lambda x, p=img_data['path'], v="different_crow": self.on_classification_change(p,v,x))
            options_layout.add_widget(btn_other_crow)
            self.image_widgets_map[img_data['path']]['buttons']["different_crow"] = btn_other_crow

            btn_new_crow = ToggleButton(text="New Unidentified Crow", group=classification_group_id) # Added this option
            btn_new_crow.classification_value = "new_unidentified_crow"
            btn_new_crow.bind(on_press=lambda x, p=img_data['path'], v="new_unidentified_crow": self.on_classification_change(p,v,x))
            options_layout.add_widget(btn_new_crow)
            self.image_widgets_map[img_data['path']]['buttons']["new_unidentified_crow"] = btn_new_crow

            btn_not_crow = ToggleButton(text="Not a Crow", group=classification_group_id)
            btn_not_crow.classification_value = "not_crow"
            btn_not_crow.bind(on_press=lambda x, p=img_data['path'], v="not_crow": self.on_classification_change(p,v,x))
            options_layout.add_widget(btn_not_crow)
            self.image_widgets_map[img_data['path']]['buttons']["not_crow"] = btn_not_crow

            btn_multi_crow = ToggleButton(text="Multiple Crows", group=classification_group_id)
            btn_multi_crow.classification_value = "multi_crow"
            btn_multi_crow.bind(on_press=lambda x, p=img_data['path'], v="multi_crow": self.on_classification_change(p,v,x))
            options_layout.add_widget(btn_multi_crow)
            self.image_widgets_map[img_data['path']]['buttons']["multi_crow"] = btn_multi_crow
            
            item_layout.add_widget(options_layout)
            self.layout.sightings_grid.add_widget(item_layout)
        
        # Initialize selections state (all are 'same_crow' by default due to ToggleButton state)
        self.image_classifications = {path: "same_crow" for path in self.image_widgets_map.keys()}


    def on_input_change(self, instance, value): # For TextInput or other inputs
        self.unsaved_changes = True
        self.update_save_discard_buttons_state()

    def on_classification_change(self, image_path, classification_value, instance_button):
        if instance_button.state == 'down': # Only update if button is selected
            self.image_classifications[image_path] = classification_value
            self.unsaved_changes = True
            self.update_save_discard_buttons_state()
            logger.info(f"Image {os.path.basename(image_path)} classified as: {classification_value}")


    def update_save_discard_buttons_state(self):
        self.layout.save_button.disabled = not self.unsaved_changes
        self.layout.discard_button.disabled = not self.unsaved_changes


    def save_all_changes(self, instance):
        logger.info("Save changes button pressed.")
        if not self.current_crow_id:
            self.show_popup("Error", "No crow loaded to save changes for.")
            return

        # 1. Update crow name if changed
        new_name = self.layout.crow_name_input.text.strip()
        if new_name != (self.current_crow_name or ""):
            try:
                update_crow_name(self.current_crow_id, new_name if new_name else None)
                self.current_crow_name = new_name
                logger.info(f"Crow {self.current_crow_id} name updated to '{new_name}'.")
                # Update spinner text
                self.load_initial_crow_list() # Reload to reflect name change
                self.layout.crow_spinner.text = f"Crow{self.current_crow_id} ({self.current_crow_name or 'Unnamed'})"

            except Exception as e:
                logger.error(f"Error updating crow name: {e}", exc_info=True)
                self.show_popup("DB Error", f"Failed to update crow name: {e}")
                # Continue with other changes if name update failed

        # 2. Process image reassignments based on self.image_classifications
        # This part needs the complex dialog logic from original `process_reassignments`
        # For now, this is a simplified placeholder.
        # TODO: Adapt the full `process_reassignments` logic with Kivy Popups.
        
        images_to_reassign_to_other = [] # Path list for 'different_crow'
        images_to_create_new_crow = [] # Path list for 'new_unidentified_crow'
        images_to_remove_embeddings = [] # Path list for 'not_crow' or 'multi_crow'

        for img_path, classification in self.image_classifications.items():
            if classification == "different_crow":
                images_to_reassign_to_other.append(img_path)
            elif classification == "new_unidentified_crow":
                images_to_create_new_crow.append(img_path)
            elif classification == "not_crow" or classification == "multi_crow":
                images_to_remove_embeddings.append(img_path)
        
        # Placeholder for calling the adapted process_reassignments logic
        # For now, just log and show success
        logger.info(f"To Reassign (Existing): {len(images_to_reassign_to_other)}")
        logger.info(f"To Reassign (New Crow): {len(images_to_create_new_crow)}")
        logger.info(f"To Remove Embeddings: {len(images_to_remove_embeddings)}")

        # This is where you would call the Kivy version of process_reassignments
        if images_to_reassign_to_other or images_to_create_new_crow or images_to_remove_embeddings:
            self._kivy_process_reassignments(
                images_to_reassign_to_other, 
                images_to_create_new_crow, 
                images_to_remove_embeddings
            )
        else: # Only name might have changed
            self.show_popup("Success", "Crow name updated successfully!")
            self.unsaved_changes = False
            self.update_save_discard_buttons_state()
            self.load_crow_data(None) # Reload to reflect name change in UI title etc.


    def discard_all_changes(self, instance):
        logger.info("Discard changes button pressed.")
        if self.unsaved_changes:
            # Show confirmation popup
            content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
            content.add_widget(Label(text="Are you sure you want to discard all unsaved changes?"))
            buttons = BoxLayout(size_hint_y=None, height="40dp", spacing="10dp")
            yes_btn = Button(text="Yes, Discard")
            no_btn = Button(text="No, Cancel")
            buttons.add_widget(yes_btn)
            buttons.add_widget(no_btn)
            content.add_widget(buttons)
            popup = Popup(title="Confirm Discard", content=content, size_hint=(None, None), size=("400dp", "150dp"), auto_dismiss=False)
            
            def do_discard(btn_instance):
                popup.dismiss()
                self.layout.crow_name_input.text = self.current_crow_name or ""
                # Reset image selections in UI (if UI elements directly store state)
                # For current model, _display_sighting_items would be called on reload if needed
                # Or, iterate self.image_widgets_map and reset ToggleButton states
                for img_path, widget_info in self.image_widgets_map.items():
                    widget_info['buttons']['same_crow'].state = 'down' # Reset to default
                self.image_classifications = {path: "same_crow" for path in self.image_widgets_map.keys()}

                self.unsaved_changes = False
                self.update_save_discard_buttons_state()
                logger.info("Changes discarded.")

            yes_btn.bind(on_press=do_discard)
            no_btn.bind(on_press=popup.dismiss)
            popup.open()
        else:
            self.show_popup("Info", "No changes to discard.")
            

    def on_request_close_window(self, *args, **kwargs):
        if self.unsaved_changes:
            self.show_popup("Unsaved Changes", "You have unsaved changes. Please Save or Discard them before closing.")
            return True  # Prevent closing
        return False # Allow closing

    def show_popup(self, title, message):
        # Ensure popup is created on main Kivy thread if called from elsewhere
        if not isinstance(threading.current_thread(), threading._MainThread):
             Clock.schedule_once(lambda dt: self._do_show_popup(title, message))
        else:
            self._do_show_popup(title, message)

    def _do_show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        # Ensure text_size allows wrapping for longer messages
        msg_label = Label(text=message, text_size=(Window.width*0.45, None), halign='center', valign='middle')
        content.add_widget(msg_label)
        ok_button = Button(text="OK", size_hint_y=None, height="44dp")
        content.add_widget(ok_button)
        
        popup = Popup(title=title, content=content, size_hint=(0.5, None), height="200dp", auto_dismiss=True) # Adjust height
        popup.bind(on_open=lambda instance: setattr(msg_label, 'width', instance.width * 0.9)) # Adjust label width on open
        ok_button.bind(on_press=popup.dismiss)
        popup.open()
        logger.info(f"Popup shown: '{title}'")

    # --- Methods for Reassignment Popups ---
    _reassignment_choice_result = None # Temporary storage for popup result
    _target_crow_id_result = None

    def _kivy_ask_reassignment_choice_popup(self, num_images_different, num_images_new_unidentified, callback):
        self._reassignment_choice_result = None # Reset before showing
        
        total_for_new_or_existing = num_images_different + num_images_new_unidentified
        if total_for_new_or_existing == 0: # Should not happen if called, but good check
            callback(None) # No choice needed
            return

        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        
        text_parts = []
        if num_images_different > 0:
            text_parts.append(f"{num_images_different} image(s) marked 'Another Known Crow'")
        if num_images_new_unidentified > 0:
            text_parts.append(f"{num_images_new_unidentified} image(s) marked 'New Unidentified Crow'")
        
        full_text = (" and ".join(text_parts) + ".\n" +
                     "How would you like to handle these images?")
        
        content.add_widget(Label(text=full_text, text_size=(Window.width*0.4, None))) # Wider for text

        buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
        new_crow_btn = Button(text="Create New Crow(s)")
        existing_crow_btn = Button(text="Move to Existing Crow")
        cancel_btn = Button(text="Cancel This Reassignment")
        buttons.add_widget(new_crow_btn)
        buttons.add_widget(existing_crow_btn)
        buttons.add_widget(cancel_btn)
        content.add_widget(buttons)

        popup = Popup(title="Reassignment Options", content=content, size_hint=(0.6, None), height="250dp", auto_dismiss=False)

        def set_choice(choice, btn_instance):
            self._reassignment_choice_result = choice
            popup.dismiss()

        new_crow_btn.bind(on_press=lambda x: set_choice("new_crow", x))
        existing_crow_btn.bind(on_press=lambda x: set_choice("existing_crow", x))
        cancel_btn.bind(on_press=lambda x: set_choice("cancel", x))
        
        popup.bind(on_dismiss=lambda instance: callback(self._reassignment_choice_result))
        popup.open()

    def _kivy_select_target_crow_popup(self, callback):
        self._target_crow_id_result = None # Reset
        
        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        content.add_widget(Label(text="Select the target crow to move images to:", size_hint_y=None, height="30dp"))

        # Prepare crow data for selection
        try:
            all_crows_db = get_all_crows()
            # Exclude the current crow from the list of targets
            target_crows_data = [
                {'id': c['id'], 'text': f"Crow{c['id']} ({c['name'] or 'Unnamed'}) - {c['total_sightings']} sightings"}
                for c in all_crows_db if c['id'] != self.current_crow_id
            ]
            if not target_crows_data:
                self.show_popup("No Other Crows", "There are no other crows available to move images to.")
                Clock.schedule_once(lambda dt: callback(None)) # Ensure callback is called
                return
        except Exception as e:
            logger.error(f"Error fetching crows for target selection: {e}", exc_info=True)
            self.show_popup("DB Error", "Failed to load other crows.")
            Clock.schedule_once(lambda dt: callback(None)) # Ensure callback is called
            return

        # Create scrollable list of crows using ScrollView + GridLayout
        scroll_view = ScrollView(size_hint_y=1)
        crow_list_layout = GridLayout(cols=1, spacing="2dp", size_hint_y=None)
        crow_list_layout.bind(minimum_height=crow_list_layout.setter('height'))
        
        selected_crow_id = [None]  # Use list to allow modification in nested function
        
        def on_crow_selected(instance):
            # Deselect all other buttons
            for child in crow_list_layout.children:
                if hasattr(child, 'state'):
                    child.state = 'normal'
            # Select this button
            instance.state = 'down'
            selected_crow_id[0] = instance.crow_id
        
        for crow_data in target_crows_data:
            crow_button = ToggleButton(
                text=crow_data['text'], 
                size_hint_y=None, 
                height="30dp",
                group="target_crow_selection"
            )
            crow_button.crow_id = crow_data['id']
            crow_button.bind(on_press=on_crow_selected)
            crow_list_layout.add_widget(crow_button)
        
        scroll_view.add_widget(crow_list_layout)
        content.add_widget(scroll_view)

        buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
        select_btn = Button(text="Select Target Crow")
        cancel_btn = Button(text="Cancel")
        buttons.add_widget(select_btn)
        buttons.add_widget(cancel_btn)
        content.add_widget(buttons)

        popup = Popup(title="Select Target Crow", content=content, size_hint=(0.7, 0.8), auto_dismiss=False)

        def set_target(btn_instance):
            self._target_crow_id_result = selected_crow_id[0]
            popup.dismiss()
        
        def cancel_selection(btn_instance):
            self._target_crow_id_result = None
            popup.dismiss()

        select_btn.bind(on_press=set_target)
        cancel_btn.bind(on_press=cancel_selection)
        
        popup.bind(on_dismiss=lambda instance: callback(self._target_crow_id_result))
        popup.open()


    def _kivy_process_reassignments(self, images_to_reassign_to_other, images_to_create_new_crow, images_to_remove_embeddings):
        logger.info("Starting Kivy process reassignments...")
        
        # Process removals first (no user interaction needed for these beyond initial classification)
        if images_to_remove_embeddings:
            try:
                embedding_map = get_embedding_ids_by_image_paths(images_to_remove_embeddings)
                embedding_ids = list(embedding_map.values())
                if embedding_ids:
                    delete_crow_embeddings(embedding_ids)
                    logger.info(f"Removed {len(embedding_ids)} embeddings for 'not_crow'/'multi_crow' images.")
                else:
                    logger.info("No embeddings found for images marked for removal.")
            except Exception as e:
                logger.error(f"Error removing embeddings: {e}", exc_info=True)
                self.show_popup("DB Error", f"Failed to remove embeddings: {e}")

        # Now handle reassignments that require choices
        images_for_choice = images_to_reassign_to_other + images_to_create_new_crow
        if not images_for_choice:
            logger.info("No images require reassignment choices. Process complete.")
            self.show_popup("Success", "Changes saved successfully!")
            self.unsaved_changes = False
            self.update_save_discard_buttons_state()
            self.load_crow_data(None) # Reload to reflect all changes
            return

        def after_ask_choice(choice):
            if choice == "new_crow":
                logger.info("User chose to create new crow(s) for reassignments.")
                try:
                    # For images initially marked "Another Known Crow" or "New Unidentified"
                    # if they are all going to one new crow:
                    if images_for_choice:
                        embedding_map = get_embedding_ids_by_image_paths(images_for_choice)
                        embedding_ids = list(embedding_map.values())
                        if embedding_ids:
                            new_crow_id = create_new_crow_from_embeddings(self.current_crow_id, embedding_ids, f"Split from Crow{self.current_crow_id}")
                            logger.info(f"Created new Crow ID {new_crow_id} with {len(embedding_ids)} embeddings.")
                            self.show_popup("New Crow Created", f"Created new Crow ID {new_crow_id}.")
                        else:
                            logger.info("No embeddings found for images to create new crow.")
                except Exception as e:
                    logger.error(f"Error creating new crow: {e}", exc_info=True)
                    self.show_popup("DB Error", f"Failed to create new crow: {e}")
                
                # Finalize after this path
                # Finalize after this path
                self.unsaved_changes = False # Reset as an action was taken or attempted
                self.update_save_discard_buttons_state()
                self.load_initial_crow_list() 
                self.load_crow_data(None) 


            elif choice == "existing_crow":
                logger.info("User chose to move images to an existing crow.")
                
                def after_select_target(target_crow_id):
                    action_taken = False
                    if target_crow_id:
                        logger.info(f"Target crow ID {target_crow_id} selected for reassignments.")
                        try:
                            if images_for_choice: # These are the paths for "different_crow" + "new_unidentified"
                                embedding_map = get_embedding_ids_by_image_paths(images_for_choice)
                                embedding_ids = list(embedding_map.values())
                                if embedding_ids:
                                    reassign_crow_embeddings(self.current_crow_id, target_crow_id, embedding_ids)
                                    logger.info(f"Reassigned {len(embedding_ids)} embeddings from Crow {self.current_crow_id} to Crow {target_crow_id}.")
                                    self.show_popup("Reassignment Complete", f"Moved {len(embedding_ids)} images to Crow ID {target_crow_id}.")
                                    action_taken = True
                                else:
                                    logger.info("No embeddings found for images to reassign to existing crow.")
                                    self.show_popup("Info", "No embeddings found for selected images to move.")
                        except Exception as e:
                            logger.error(f"Error reassigning to existing crow: {e}", exc_info=True)
                            self.show_popup("DB Error", f"Failed to reassign to Crow ID {target_crow_id}: {e}")
                    else:
                        logger.info("User cancelled target crow selection for reassignment.")
                        self.show_popup("Cancelled", "Reassignment to existing crow cancelled.")
                    
                    # Finalize after this path
                    self.unsaved_changes = False # Reset as an action was taken, attempted, or explicitly cancelled at target selection
                    self.update_save_discard_buttons_state()
                    self.load_crow_data(None) 
                    if action_taken: # Potentially refresh full crow list if counts changed significantly, though not strictly necessary
                        self.load_initial_crow_list()


                self._kivy_select_target_crow_popup(after_select_target)

            elif choice == "cancel":
                logger.info("User cancelled the reassignment operation at the choice dialog.")
                self.show_popup("Cancelled", "Reassignment operation cancelled.")
                # User explicitly cancelled the multi-image reassignment operation.
                # We might leave unsaved_changes as true if only name was changed, or other pending changes exist.
                # For simplicity now, we assume this cancel applies to the bulk operation.
                # If name was also changed, it's already saved before this step.
                # If only reassignments were pending, then effectively they are "discarded" for now.
                # A more granular unsaved_changes might be needed for complex scenarios.
                # For now, let's assume the main save operation is "done" or "cancelled" at this point.
                # If they cancel here, they might still want to save a name change if that was separate.
                # However, current flow saves name first.
                self.unsaved_changes = False # Or handle more granularly.
                self.update_save_discard_buttons_state()
            
            else: # No choice made or popup closed (e.g. by pressing Esc, if auto_dismiss was true)
                logger.info("No reassignment choice made or dialog dismissed.")
                # Similar to cancel, current operation regarding these images is halted.
                self.unsaved_changes = False 
                self.update_save_discard_buttons_state()


        num_different = len(images_to_reassign_to_other)
        num_new_unidentified = len(images_to_create_new_crow) # These are images user marked as "New Unidentified"
        
        # This logic path assumes that if 'new_unidentified_crow' was chosen by user for an image,
        # it implies they want those specific images to form one or more new crows.
        # The current `_kivy_ask_reassignment_choice_popup` groups "different_crow" and "new_unidentified_crow"
        # into a single question: "Create New Crow(s)" or "Move to Existing Crow".
        # This means if "Create New Crow(s)" is chosen, ALL images in `images_for_choice` go to ONE new crow.
        # This matches the original tkinter logic more closely where 'different_crow' could become a new crow.

        self._kivy_ask_reassignment_choice_popup(num_different, num_new_unidentified, after_ask_choice)


if __name__ == '__main__':
    # Kivy specific setup if needed
    SuspectLineupApp().run()
# print("SuspectLineupApp finished.")
