import logging
from pathlib import Path
import uuid
import os
import sys
import time
import json
from datetime import datetime
import threading
import queue

import torch
# from torch.utils.data import DataLoader, Dataset, random_split # Will be used by training logic
# import torch.nn as nn # Will be used by training logic
# import torch.optim as optim # Will be used by training logic
# from tqdm import tqdm # Not for GUI
import cv2 # Used by dataset
import numpy as np
from models import CrowResNetEmbedder, CrowMultiModalEmbedder # Will be used by training logic
from old_scripts.train_triplet_resnet import CrowTripletDataset, TripletLoss, compute_metrics, custom_triplet_collate # Will be used by training logic
from torchvision import transforms # Used by dataset
from db import get_training_data_stats # If used by the copied logic for dataset info

import matplotlib
matplotlib.use('Agg') # Use Agg backend for Matplotlib to avoid conflicts with Kivy's event loop
import matplotlib.pyplot as plt
import seaborn as sns

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.slider import Slider
from kivy.uix.spinner import Spinner # Might be useful for some params
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.utils import platform

# Preserve the TrainingMetrics class
class TrainingMetrics:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.same_crow_mean = []
        self.same_crow_std = []
        self.diff_crow_mean = []
        self.diff_crow_std = []
        self.val_similarities = [] # List of similarity arrays for each epoch
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_time = 0
        self.start_time = None

# Global logger (will be configured in main)
logger = logging.getLogger(__name__)

class TrainingLayout(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = 'vertical'

        # Main content area (horizontal: controls on left, plots on right)
        main_content = BoxLayout(orientation='horizontal', spacing=10, padding=10)

        # Left Panel (Controls) - Use ScrollView for many controls
        left_scroll = ScrollView(size_hint_x=0.4, do_scroll_x=False)
        self.left_panel = GridLayout(cols=1, spacing=15, size_hint_y=None, padding=5) # Increased spacing
        self.left_panel.bind(minimum_height=self.left_panel.setter('height'))

        # --- Directory Selection ---
        dir_section = GridLayout(cols=1, spacing=5, size_hint_y=None)
        dir_section.add_widget(Label(text="Directories", font_size='18sp', size_hint_y=None, height=30, bold=True))
        
        self.crop_dir_input = self._create_dir_selector("Crow Crops Directory:", "crow_crops")
        dir_section.add_widget(self.crop_dir_input)
        
        self.audio_dir_input = self._create_dir_selector("Audio Directory:", "crow_crops/audio")
        dir_section.add_widget(self.audio_dir_input)
        dir_section.add_widget(Label(text="Audio directory for synchronized audio segments (optional)", 
                                     font_size='10sp', size_hint_y=None, height=20, color=(0.7,0.7,0.7,1)))

        self.output_dir_input = self._create_dir_selector("Output Directory:", "training_output")
        dir_section.add_widget(self.output_dir_input)
        self.left_panel.add_widget(dir_section)

        # --- Training Parameters ---
        train_params_section = GridLayout(cols=1, spacing=5, size_hint_y=None)
        train_params_section.add_widget(Label(text="Training Parameters", font_size='18sp', size_hint_y=None, height=30, bold=True))
        
        self.batch_size_input = self._create_param_input("Batch Size:", "32")
        train_params_section.add_widget(self.batch_size_input)
        
        self.lr_input = self._create_param_input("Learning Rate:", "1e-4")
        train_params_section.add_widget(self.lr_input)

        self.epochs_input = self._create_param_input("Max Epochs:", "50")
        train_params_section.add_widget(self.epochs_input)

        self.val_split_slider, self.val_split_label, val_split_layout = self._create_slider_param("Validation Split (0.1-0.5):", 0.1, 0.5, 0.2)
        train_params_section.add_widget(val_split_layout)

        self.patience_input = self._create_param_input("Early Stopping Patience:", "5")
        train_params_section.add_widget(self.patience_input)
        self.left_panel.add_widget(train_params_section)

        # --- Model Parameters ---
        model_params_section = GridLayout(cols=1, spacing=5, size_hint_y=None)
        model_params_section.add_widget(Label(text="Model Parameters", font_size='18sp', size_hint_y=None, height=30, bold=True))

        self.embed_dim_input = self._create_param_input("Embedding Dimension:", "512")
        model_params_section.add_widget(self.embed_dim_input)

        self.margin_slider, self.margin_label, margin_layout = self._create_slider_param("Triplet Loss Margin (0.1-2.0):", 0.1, 2.0, 1.0)
        model_params_section.add_widget(margin_layout)
        self.left_panel.add_widget(model_params_section)

        # --- Control Buttons ---
        control_buttons_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint_y=None, height=44)
        self.start_button = Button(text="Start Training", on_press=self.app._start_training)
        self.pause_button = ToggleButton(text="Pause", group='training_control', state='normal', on_press=self.app._pause_training, disabled=True)
        self.stop_button = Button(text="Stop", on_press=self.app._stop_training, disabled=True)
        control_buttons_layout.add_widget(self.start_button)
        control_buttons_layout.add_widget(self.pause_button)
        control_buttons_layout.add_widget(self.stop_button)
        self.left_panel.add_widget(control_buttons_layout)

        # --- Progress Tracking ---
        progress_section = GridLayout(cols=1, spacing=5, size_hint_y=None)
        progress_section.add_widget(Label(text="Progress", font_size='18sp', size_hint_y=None, height=30, bold=True))
        self.progress_bar = ProgressBar(max=100, value=0, size_hint_y=None, height=20)
        progress_section.add_widget(self.progress_bar)
        self.progress_label = Label(text="Ready", size_hint_y=None, height=25)
        progress_section.add_widget(self.progress_label)
        self.left_panel.add_widget(progress_section)

        # --- Current Metrics Display ---
        metrics_section = GridLayout(cols=1, spacing=5, size_hint_y=None)
        metrics_section.add_widget(Label(text="Current Metrics", font_size='18sp', size_hint_y=None, height=30, bold=True))
        
        self.metrics_labels = {}
        metrics_definitions = [
            ('epoch', 'Epoch:'), ('train_loss', 'Train Loss:'), ('val_loss', 'Val Loss:'),
            ('same_crow_sim', 'Same Crow Similarity:'), ('diff_crow_sim', 'Diff Crow Similarity:'),
            ('time_elapsed', 'Time Elapsed:'), ('best_val_loss', 'Best Val Loss:')
        ]
        metrics_grid = GridLayout(cols=2, spacing=5, size_hint_y=None)
        for key, text_label in metrics_definitions:
            metrics_grid.add_widget(Label(text=text_label, size_hint_x=0.6, halign='right', height=25))
            self.metrics_labels[key] = Label(text="--", size_hint_x=0.4, halign='left', height=25)
            metrics_grid.add_widget(self.metrics_labels[key])
        metrics_section.add_widget(metrics_grid)
        self.left_panel.add_widget(metrics_section)
        
        # Make sure the left_panel's height is updated correctly for the ScrollView
        self.left_panel.bind(minimum_height=self.left_panel.setter('height'))


        left_scroll.add_widget(self.left_panel)
        main_content.add_widget(left_scroll)

        # Right Panel (Plots)
        self.right_panel = BoxLayout(orientation='vertical', size_hint_x=0.6)
        # Placeholder for plots
        self.right_panel.add_widget(Label(text="Plots Will Go Here"))
        main_content.add_widget(self.right_panel)

        self.add_widget(main_content)

        # Status Bar (Bottom)
        self.status_label = Label(text="Ready", size_hint_y=None, height=30)
        self.add_widget(self.status_label)

        # Initialize Matplotlib figures and axes for Kivy
        self.fig, (self.loss_ax, self.sim_ax) = plt.subplots(2, 1, figsize=(8,6))
        self.fig.tight_layout(pad=3.0) # Add some padding
        self.plot_canvas_widget = FigureCanvasKivyAgg(self.fig)
        self.right_panel.clear_widgets() # Clear placeholder
        self.right_panel.add_widget(self.plot_canvas_widget)

    # Helper methods for creating UI elements consistently
    def _create_dir_selector(self, label_text, default_path):
        layout = BoxLayout(orientation='horizontal', spacing=5, size_hint_y=None, height=35)
        layout.add_widget(Label(text=label_text, size_hint_x=0.4, halign='right'))
        text_input = TextInput(text=default_path, size_hint_x=0.5, multiline=False)
        # Store the text_input widget itself for access later
        setattr(self, f"{label_text.lower().replace(' ', '_').replace(':', '')}_text_input", text_input)
        browse_button = Button(text="Browse", size_hint_x=0.1)
        browse_button.bind(on_press=lambda x: self.app._select_directory_popup(text_input))
        layout.add_widget(text_input)
        layout.add_widget(browse_button)
        return layout

    def _create_param_input(self, label_text, default_value):
        layout = BoxLayout(orientation='horizontal', spacing=5, size_hint_y=None, height=35)
        layout.add_widget(Label(text=label_text, size_hint_x=0.7, halign='right'))
        text_input = TextInput(text=default_value, size_hint_x=0.3, multiline=False)
        layout.add_widget(text_input)
        return layout
        
    def _create_slider_param(self, label_text, min_val, max_val, default_val):
        layout = BoxLayout(orientation='horizontal', spacing=5, size_hint_y=None, height=35)
        layout.add_widget(Label(text=label_text, size_hint_x=0.6, halign='right'))
        slider = Slider(min=min_val, max=max_val, value=default_val, size_hint_x=0.3)
        value_label = Label(text=f"{default_val:.2f}", size_hint_x=0.1)
        def update_label(instance, value):
            value_label.text = f"{instance.value:.2f}"
        slider.bind(value=update_label)
        layout.add_widget(slider)
        layout.add_widget(value_label)
        return slider, value_label, layout


class TrainingApp(App):
    def build(self):
        self.title = "Crow Training GUI (Kivy)"
        Window.size = (1300, 950) # Adjusted size for more controls

        self.training_thread = None
        self.plot_queue = queue.Queue()
        self.plot_updater_thread = None
        self.training_active = False 
        self.training_paused = False 
        self.metrics = TrainingMetrics()
        self.session_id = str(uuid.uuid4())[:8] 

        self._setup_logging()
        logger.info(f"SESSION ID: {self.session_id}")
        self._log_system_info()

        self.layout = TrainingLayout(app=self)
        return self.layout

    def _setup_logging(self):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f'kivy_train_triplet_gui_{self.session_id}.log'
        
        if logger.hasHandlers(): # Clear existing handlers
            logger.handlers.clear()

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True 
        )
        logger.info("Logging configured.")

    def _log_system_info(self):
        logger.info(f'Python executable: {sys.executable}')
        logger.info(f'Torch version: {torch.__version__}')
        cuda_available = torch.cuda.is_available()
        logger.info(f'CUDA available: {cuda_available}')
        if cuda_available:
            try:
                logger.info(f'Device Name: {torch.cuda.get_device_name(0)}')
            except Exception as e:
                logger.error(f"Could not get CUDA device name: {e}")
        logger.info(f'Platform: {sys.platform}') # Using sys.platform for more general info
        # logger.info(f'Kivy Version: {kivy.__version__}') # kivy import might not be available here yet

    def on_start(self):
        logger.info(f"TrainingApp started. Kivy version: {kivy.__version__}")
        # self._start_plot_updater() # Will be implemented later

    def on_stop(self):
        logger.info("TrainingApp stopping.")
        self.training_active = False # Signal training thread to stop
        if self.training_thread and self.training_thread.is_alive():
            logger.info("Joining training thread...")
            self.training_thread.join(timeout=5)
            if self.training_thread.is_alive():
                logger.warning("Training thread did not terminate cleanly.")
        
        if self.plot_updater_thread and self.plot_updater_thread.is_alive():
            logger.info("Signaling plot updater to exit and joining...")
            self.plot_queue.put(None) 
            self.plot_updater_thread.join(timeout=2)
            if self.plot_updater_thread.is_alive():
                logger.warning("Plot updater thread did not terminate cleanly.")
        logger.info("TrainingApp stopped.")
    
    def show_popup(self, title, message):
        # Ensure this runs on the main Kivy thread
        if not isinstance(threading.current_thread(), threading._MainThread):
            Clock.schedule_once(lambda dt: self._do_show_popup(title, message))
        else:
            self._do_show_popup(title, message)

    def _do_show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(text=message, halign='center', valign='middle', text_size=(Window.width * 0.6, None)))
        ok_button = Button(text="OK", size_hint_y=None, height=44)
        content.add_widget(ok_button)
        
        popup = Popup(title=title, content=content, size_hint=(0.7, 0.5), auto_dismiss=False)
        ok_button.bind(on_press=popup.dismiss)
        popup.open()
        logger.info(f"Popup shown: '{title}'")
        popup.dismiss_schedule = Clock.schedule_once(popup.dismiss, 30) # Auto-dismiss after 30s if not interacted with

    def _select_directory_popup(self, text_input_target):
        logger.info(f"Browse directory called for {text_input_target}")
        content = BoxLayout(orientation='vertical', spacing=10)
        
        # Start browsing from user's home directory or current input path if valid
        initial_path = text_input_target.text
        if not os.path.isdir(initial_path):
            initial_path = os.path.expanduser('~')

        filechooser = FileChooserListView(path=initial_path, dirselect=True, size_hint_y=0.8)
        content.add_widget(filechooser)
        
        buttons_layout = BoxLayout(size_hint_y=0.2, height=44, spacing=10)
        select_button = Button(text="Select Directory")
        cancel_button = Button(text="Cancel")
        buttons_layout.add_widget(select_button)
        buttons_layout.add_widget(cancel_button)
        content.add_widget(buttons_layout)
        
        popup = Popup(title="Select Directory", content=content, size_hint=(0.9, 0.9))
        
        def select_dir(instance):
            if filechooser.selection:
                selected_path = filechooser.selection[0]
                text_input_target.text = selected_path
                logger.info(f"Directory selected: {selected_path} for {text_input_target}")
            popup.dismiss()

        select_button.bind(on_press=select_dir)
        cancel_button.bind(on_press=popup.dismiss)
        popup.open()


    def _start_training(self, instance):
        logger.info("Start training button pressed.")
        # Actual implementation will follow: validate inputs, get config, start thread
        pass

    def _training_loop(self, config):
        pass

    def _update_metrics_display_safe(self, *args): # Scheduled by Clock
        pass

    def _update_plots_safe(self, *args): # Scheduled by Clock
        pass
    
    def _start_plot_updater(self): # Manages the plot update thread
        pass

    def _save_training_results(self, config):
        pass

    def _training_complete_safe(self, *args): # Scheduled by Clock
        pass

    def _pause_training(self, instance):
        pass

    def _stop_training(self, instance):
        pass
    
    def _confirm_close(self, *args): # For on_request_close
        pass


if __name__ == '__main__':
    # This check is important for Kivy, especially on Windows.
    # It also ensures that if any part of your app uses multiprocessing,
    # it behaves correctly.
    from kivy import kivy_config_version
    if kivy_config_version < 21: # Check for Kivy config version if needed
        pass # Kivy specific setup if required for older versions
    
    # Ensure Kivy's global objects are properly initialized if running standalone
    # (though App.run() usually handles this)
    
    # Set up environment variables for Kivy if needed (e.g., for specific renderers)
    # os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2' # Example for Windows
    
    TrainingApp().run()
print("Script execution finished.") # For debugging if app doesn't start
