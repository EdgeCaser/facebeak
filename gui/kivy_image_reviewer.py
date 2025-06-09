import os
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image as KivyImage
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture as KivyTexture


from PIL import Image as PILImage
import cv2 # OpenCV is not directly used in tkinter version, but included in imports
import numpy as np # Included in tkinter imports
from db import get_unlabeled_images, add_image_label, get_training_data_stats # Assuming db.py is in the same directory
import logging
from pathlib import Path
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageReviewerLayout(BoxLayout):
    def __init__(self, app, **kwargs): # Added app argument
        super().__init__(**kwargs)
        self.app = app # Store a reference to the app
        self.orientation = 'horizontal'
        
        # Left panel for controls
        control_panel = BoxLayout(orientation='vertical', size_hint_x=0.3, spacing=10, padding=10)
        
        # Title and instructions
        title_label = Label(text="Crow Image Reviewer", font_size='20sp', bold=True, size_hint_y=None, height=40)
        instructions_text = ("Review images and confirm whether they contain crows.\n"
                             "Images marked as 'Not a crow' will be excluded from training.")
        instructions_label = Label(text=instructions_text, size_hint_y=None, height=60)
        
        control_panel.add_widget(title_label)
        control_panel.add_widget(instructions_label)
        
        # Directory selection
        dir_frame = BoxLayout(orientation='vertical', size_hint_y=None, height=100, spacing=5)
        dir_frame_label = Label(text="Image Directory", bold=True, size_hint_y=None, height=30)
        self.dir_input = TextInput(text="crow_crops", multiline=False, size_hint_y=None, height=30)
        dir_buttons = BoxLayout(orientation='horizontal', size_hint_y=None, height=30, spacing=5)
        browse_button = Button(text="Browse", on_press=self.browse_directory)
        reload_button = Button(text="Reload", on_press=self.load_images_action) # Changed
        dir_buttons.add_widget(browse_button)
        dir_buttons.add_widget(reload_button)
        
        dir_frame.add_widget(dir_frame_label)
        dir_frame.add_widget(self.dir_input)
        dir_frame.add_widget(dir_buttons)
        control_panel.add_widget(dir_frame)

        # Progress info
        progress_frame = BoxLayout(orientation='vertical', size_hint_y=None, height=80, spacing=5)
        progress_frame_label = Label(text="Progress", bold=True, size_hint_y=None, height=30)
        self.progress_label = Label(text="No images loaded", size_hint_y=None, height=20)
        self.progress_bar = ProgressBar(max=100, value=0, size_hint_y=None, height=20)
        
        progress_frame.add_widget(progress_frame_label)
        progress_frame.add_widget(self.progress_label)
        progress_frame.add_widget(self.progress_bar)
        control_panel.add_widget(progress_frame)

        # Navigation buttons
        nav_frame = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=5)
        prev_button = Button(text="← Previous", on_press=self.app.previous_image)
        next_button = Button(text="Next →", on_press=self.app.next_image)
        nav_frame.add_widget(prev_button)
        nav_frame.add_widget(next_button)
        control_panel.add_widget(nav_frame)

        # Labeling options
        label_frame = BoxLayout(orientation='vertical', size_hint_y=None, height=130, spacing=5) # Adjusted height
        label_frame_label = Label(text="Label Current Image", bold=True, size_hint_y=None, height=30)
        
        self.label_spinner = Spinner(
            text='Crow', # Default value
            values=('Crow', 'Not a crow', 'Not sure', 'Multiple crows'),
            size_hint_y=None,
            height=44 # Kivy default height
        )
        submit_button = Button(text="Submit Label", size_hint_y=None, height=44, on_press=self.app.submit_label)
        
        label_frame.add_widget(label_frame_label)
        label_frame.add_widget(self.label_spinner)
        label_frame.add_widget(submit_button)
        control_panel.add_widget(label_frame)

        # Statistics
        stats_frame = BoxLayout(orientation='vertical', size_hint_y=None, height=180, spacing=5) # Adjusted height
        stats_frame_label = Label(text="Statistics", bold=True, size_hint_y=None, height=30)
        self.stats_label = Label(text="Loading statistics...", size_hint_y=None, height=100, halign='left', valign='top')
        self.stats_label.bind(size=self.stats_label.setter('text_size')) # For wrapping
        refresh_stats_button = Button(text="Refresh Stats", size_hint_y=None, height=44, on_press=self.app.update_stats)

        stats_frame.add_widget(stats_frame_label)
        stats_frame.add_widget(self.stats_label)
        stats_frame.add_widget(refresh_stats_button)
        control_panel.add_widget(stats_frame)
        
        # Add a spacer to push everything up
        control_panel.add_widget(BoxLayout(size_hint_y=1))


        # Right panel for image display
        image_panel = BoxLayout(orientation='vertical', spacing=10, padding=10)
        self.image_info_label = Label(text="No image loaded", size_hint_y=None, height=30, font_size='14sp')
        self.image_widget = KivyImage(source='', allow_stretch=True, keep_ratio=True)
        
        image_panel.add_widget(self.image_info_label)
        image_panel.add_widget(self.image_widget)

        self.add_widget(control_panel)
        self.add_widget(image_panel)

    def browse_directory(self, instance):
        logger.info("Browse directory called")
        content = BoxLayout(orientation='vertical')
        # Ensure the filechooser can see system files if needed, adjust path as necessary
        # For user directories, os.path.expanduser('~') is a good starting point.
        filechooser = FileChooserListView(path=os.path.expanduser('~'), dirselect=True)
        
        buttons = BoxLayout(size_hint_y=None, height=40, spacing=5)
        select_button = Button(text="Select Directory")
        cancel_button = Button(text="Cancel")
        
        buttons.add_widget(select_button)
        buttons.add_widget(cancel_button)
        
        content.add_widget(filechooser)
        content.add_widget(buttons)
        
        popup = Popup(title="Select Image Directory", content=content, size_hint=(0.9, 0.9))
        
        def select_dir(instance):
            if filechooser.selection:
                selected_path = filechooser.selection[0]
                self.dir_input.text = selected_path
                logger.info(f"Directory selected: {selected_path}")
            popup.dismiss()

        def cancel_popup(instance):
            popup.dismiss()

        select_button.bind(on_press=select_dir)
        cancel_button.bind(on_press=cancel_popup)
        
        popup.open()

    def load_images_action(self, instance): # New method to call app's load_images
        self.app.load_images()


class ImageReviewerApp(App):
    def build(self):
        self.title = "Crow Image Reviewer (Kivy)"
        self.current_images = []
        self.current_index = 0
        self.labels_queue = queue.Queue()

        self.layout = ImageReviewerLayout(app=self) # Pass app instance
        
        # Other connections and initial calls will be in on_start or here
        return self.layout

    def show_popup(self, title, message):
        """Helper function to show a simple popup."""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text=message, halign='center', valign='middle'))
        ok_button = Button(text="OK", size_hint_y=None, height=44) # Kivy default height
        content.add_widget(ok_button)
        
        popup = Popup(title=title, content=content, size_hint=(0.7, 0.5), auto_dismiss=False)
        ok_button.bind(on_press=popup.dismiss)
        popup.open()

    def load_images(self, instance=None):
        logger.info("Load images called")
        try:
            directory = self.layout.dir_input.text
            if not directory or not os.path.exists(directory):
                self.show_popup("Error", "Please select a valid directory.")
                return
            
            self.current_images = get_unlabeled_images(limit=100, from_directory=directory)
            
            if not self.current_images:
                self.show_popup("Info", "No unlabeled images found in the selected directory.")
                self.layout.progress_label.text = "No images to review"
                self.layout.image_widget.source = ''
                self.layout.image_info_label.text = "No image loaded"
                self.current_images = [] 
                self.current_index = 0
                self.layout.progress_bar.value = 0
                self.layout.progress_bar.max = 1 
                return
                
            self.current_index = 0
            self.layout.progress_bar.max = len(self.current_images)
            self.update_display() 
            
            logger.info(f"Loaded {len(self.current_images)} unlabeled images from '{directory}'")
            
        except Exception as e:
            logger.error(f"Error loading images: {e}", exc_info=True)
            self.show_popup("Error", f"Failed to load images: {str(e)}")
            self.layout.progress_label.text = "Error loading images"
            self.layout.image_widget.source = ''
            self.layout.image_info_label.text = "Error loading images"
            self.current_images = []
            self.current_index = 0

    def update_display(self):
        logger.info(f"Update display called. Index: {self.current_index}, Total: {len(self.current_images)}")
        if not self.current_images:
            self.layout.progress_label.text = "No images to review"
            self.layout.image_widget.source = '' 
            self.layout.image_info_label.text = "No image loaded"
            self.layout.progress_bar.value = 0
            return
            
        if self.current_index >= len(self.current_images):
            self.show_popup("Complete", "All images in this batch have been reviewed! Reloading for more...")
            self.load_images() 
            return
            
        progress_text = f"Image {self.current_index + 1} of {len(self.current_images)}"
        self.layout.progress_label.text = progress_text
        self.layout.progress_bar.value = self.current_index + 1
        
        image_path = self.current_images[self.current_index]
        self.display_image(image_path)
        
        filename = os.path.basename(image_path)
        self.layout.image_info_label.text = f"File: {filename}"
        
        # Reset spinner to default only if it's not already on a valid label for current image (though usually it is reset)
        if self.layout.label_spinner.text not in self.layout.label_spinner.values:
             self.layout.label_spinner.text = 'Crow' # Default label
        elif not self.layout.label_spinner.text: # Handles if text is None or empty
             self.layout.label_spinner.text = 'Crow'


    def display_image(self, image_path):
        logger.info(f"Display image called for: {image_path}")
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image path does not exist: {image_path}")
                self.show_popup("Image Load Error", f"Image file not found:\n{os.path.basename(image_path)}")
                self.layout.image_widget.source = ''
                self.layout.image_info_label.text = f"Error: Not Found {os.path.basename(image_path)}"
                return

            # Verify with PIL first to catch corrupted images before Kivy tries to load them
            # This was good practice from the original Tkinter code.
            img = PILImage.open(image_path)
            img.verify() # Verifies if it's an image
            # Must reopen after verify. PIL documentation suggests this.
            img_pil = PILImage.open(image_path)
            # img_pil.load() # Actually load data to catch more errors, Kivy might do this lazily.

            self.layout.image_widget.source = image_path 
            self.layout.image_widget.reload() # Crucial for Kivy to refresh the image texture

        except FileNotFoundError: 
            logger.error(f"Error displaying image {image_path}: File not found.", exc_info=True)
            self.layout.image_widget.source = '' # Clear image display
            self.layout.image_info_label.text = f"Error: Not Found {os.path.basename(image_path)}"
            self.show_popup("Image Load Error", f"Could not find image:\n{os.path.basename(image_path)}")
        except PILImage.UnidentifiedImageError:
            logger.error(f"Error displaying image {image_path}: Unidentified image format or corrupted.", exc_info=True)
            self.layout.image_widget.source = ''
            self.layout.image_info_label.text = f"Error: Corrupted {os.path.basename(image_path)}"
            self.show_popup("Image Load Error", f"Cannot identify image file (possibly corrupted):\n{os.path.basename(image_path)}")
        except Exception as e:
            logger.error(f"Error displaying image {image_path}: {e}", exc_info=True)
            self.layout.image_widget.source = '' 
            self.layout.image_info_label.text = f"Error loading: {os.path.basename(image_path)}"
            self.show_popup("Image Load Error", f"Could not load image:\n{os.path.basename(image_path)}\nReason: {str(e)}")
        
    def previous_image(self, instance):
        logger.info("Previous image called")
        # Check if there are images and the current index is valid
        if not self.current_images or not (0 <= self.current_index < len(self.current_images)):
            # Optionally, display a message or log if no images/invalid index for previous_image
            logger.info("Previous image: No images or invalid index.")
            return # Do nothing if no images or index is out of bounds

        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
        else:
            # Optionally, provide feedback that it's the first image
            logger.info("Previous image: Already at the first image.")
            
    def next_image(self, instance):
        logger.info("Next image called")
        if not self.current_images or not (0 <= self.current_index < len(self.current_images)):
            logger.info("Next image: No images or invalid index. Attempting to load.")
            self.load_images() # Attempt to load if list is empty or index is bad
            return

        if self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self.update_display()
        else: # Already at the last image of the current batch
            logger.info("Next image: Reached end of batch, reloading.")
            self.show_popup("End of Batch", "Reached end of current batch. Reloading for more images.")
            self.load_images() 

    def set_label_and_submit(self, label_text): # Renamed 'label' to 'label_text' for clarity
        logger.info(f"Set label and submit called with: {label_text}")
        if label_text in self.layout.label_spinner.values:
            self.layout.label_spinner.text = label_text
            self.submit_label(None) # Pass None as instance, as it's not from a button press
        else:
            logger.warning(f"Label '{label_text}' not found in spinner values.")
            self.show_popup("Error", f"Invalid label '{label_text}' for shortcut.")

    def submit_label(self, instance): # instance is the button that called it, or None
        logger.info("Submit label called")
        if not self.current_images or not (0 <= self.current_index < len(self.current_images)):
            self.show_popup("Info", "No image to label or end of list reached.")
            return
            
        image_path = self.current_images[self.current_index]
        label = self.layout.label_spinner.text # Get current text from spinner
        
        # Ensure the label is one of the valid choices, though spinner should enforce this
        if label not in self.layout.label_spinner.values:
            logger.error(f"Invalid label '{label}' selected in spinner somehow.")
            self.show_popup("Error", f"Invalid label selected: {label}")
            return

        self.labels_queue.put((image_path, label))
        logger.info(f"Queued {os.path.basename(image_path)} as {label}")
        
        # Move to next image automatically
        if self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self.update_display()
        else: 
            # This was the last image in the batch. update_display will handle reload.
            self.update_display() 

    def process_queue(self, dt=None): # dt is delta-time from Clock.schedule_interval
        # This method runs periodically, so limit logging for normal operation
        # logger.debug("Process queue check") 
        try:
            while not self.labels_queue.empty():
                image_path, label = self.labels_queue.get_nowait() # Non-blocking
                
                # Determine if this should be training data based on Kivy spinner values
                is_training_data = label == "Crow" or label == "Multiple crows"
                
                add_image_label(image_path, label, is_training_data=is_training_data)
                logger.info(f"DB: Labeled {os.path.basename(image_path)} as '{label}' (Training: {is_training_data})")
                
        except queue.Empty:
            pass # This is expected when the queue is empty
        except Exception as e:
            # Log the error and inform the user if appropriate
            logger.error(f"Error processing label queue: {e}", exc_info=True)
            # Avoid showing too many popups if db fails repeatedly. Maybe a flag?
            self.show_popup("Database Error", f"Error saving label to database:\n{str(e)}")
            
    def update_stats(self, instance=None): # instance is the button or None
        logger.info("Update stats called")
        try:
            stats = get_training_data_stats() 
            
            if stats is None: # db.py might return None on error or if table doesn't exist
                logger.warning("get_training_data_stats returned None.")
                self.layout.stats_label.text = "Could not load statistics (database issue?)"
                return

            if not stats: # Empty dictionary or list
                self.layout.stats_label.text = "No labeled data yet."
                return
            
            # Ensure keys from Kivy spinner are used. Default to 0 if a label type is missing.
            crow_count = stats.get('Crow', {}).get('count', 0)
            not_crow_count = stats.get('Not a crow', {}).get('count', 0)
            not_sure_count = stats.get('Not sure', {}).get('count', 0)
            multi_crow_count = stats.get('Multiple crows', {}).get('count', 0)
            
            total_labeled_from_db = 0
            # Summing up counts from the stats dictionary, only for expected labels
            for lbl_key in self.layout.label_spinner.values: # Use spinner values as the source of truth for labels
                total_labeled_from_db += stats.get(lbl_key, {}).get('count', 0)

            # Determine total training data and excluded based on current logic
            total_training_data = crow_count + multi_crow_count
            # This is an example; actual excluded might be calculated differently in db.py
            total_excluded_from_training = not_crow_count + not_sure_count 

            stats_text = f"""[b]Labeled Images:[/b]
  Crows: {crow_count}
  Not a crow: {not_crow_count}
  Not sure: {not_sure_count}
  Multiple Crows: {multi_crow_count}
---
Total Labeled (DB): {total_labeled_from_db}
Total Training Data: {total_training_data} 
Excluded (examples): {total_excluded_from_training}""" # Using Kivy's markup for bold
            
            self.layout.stats_label.text = stats_text
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}", exc_info=True)
            self.layout.stats_label.text = "Error loading stats."

    def on_start(self):
        """Called when the Kivy app starts. Setup initial state and schedules."""
        # Schedule initial actions after the UI is fully drawn
        Clock.schedule_once(lambda dt: self.load_images(), 0.5) # Load images shortly after start
        Clock.schedule_interval(self.process_queue, 1.0) # Check queue every second
        Clock.schedule_once(lambda dt: self.update_stats(), 1.0) # Initial stats update

        # Bind keyboard shortcuts
        Window.bind(on_keyboard=self._on_keyboard)
        logger.info("ImageReviewerApp started, keyboard shortcuts bound.")

    def on_stop(self):
        """Called when the Kivy app stops. Unbind keyboard."""
        Window.unbind(on_keyboard=self._on_keyboard)
        logger.info("ImageReviewerApp stopped, keyboard shortcuts unbound.")


    def _on_keyboard_closed(self): # This is for soft keyboard, usually not needed for desktop
        pass

    def _on_keyboard(self, window_obj, key_code, scancode, codepoint, modifiers):
        # window_obj is Window, key_code is numeric, scancode is platform specific
        # codepoint is unicode char if available, modifiers is list like ['shift', 'ctrl']
        # logger.debug(f"Key pressed: code={key_code}, char='{codepoint}', mods={modifiers}")

        if key_code == 275:  # Right arrow
            self.next_image(None)
            return True # Consume the event
        elif key_code == 276:  # Left arrow
            self.previous_image(None)
            return True
        # Enter key might be 13 (main keyboard) or 271 (numpad)
        elif key_code == 13 or key_code == 271: 
            self.submit_label(None)
            return True
        elif codepoint == '1': # Use codepoint for number keys for better reliability
            self.set_label_and_submit('Crow')
            return True
        elif codepoint == '2':
            self.set_label_and_submit('Not a crow')
            return True
        elif codepoint == '3':
            self.set_label_and_submit('Not sure')
            return True
        elif codepoint == '4':
            self.set_label_and_submit('Multiple crows')
            return True
        
        return False # Key not handled, allow it to propagate


def main():
    """Main function to run the Kivy image reviewer."""
    # Set window size (optional, can also be done in App class or kv file)
    # Window.size = (1200, 800) 
    ImageReviewerApp().run()

if __name__ == "__main__":
    # This ensures that if db.py also has a main guard, it doesn't run if imported.
    main()
