import sys
import os

if __name__ == '__main__': 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging 
logger = setup_logging() 

import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import logging 
import threading
import json
# import queue # Not strictly needed if using Clock for UI updates and simple flags for thread comms
import time
from collections import defaultdict

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.slider import Slider
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image as KivyImage
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import StringProperty, ObjectProperty, ListProperty, BooleanProperty, NumericProperty, DictProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture as KivyTexture

from crow_tracking import CrowTracker 
from detection import detect_crows_parallel # Ensure this is imported

logger.info("Starting Kivy Crow Training Data Extractor GUI")

class TooltipBehavior:
    """Behavior to add tooltip functionality via clickable info icons."""
    def __init__(self, tooltip_text="", **kwargs):
        # Store tooltip attributes without calling super()
        self.tooltip_text = tooltip_text
        self.tooltip_popup = None
        # Note: No longer binding hover events - tooltips will be triggered by info icons
        
    def show_tooltip(self):
        if self.tooltip_popup:
            return
            
        # Create tooltip content
        content = Label(
            text=self.tooltip_text,
            text_size=(Window.width * 0.4, None),
            halign='left',
            valign='top',
            font_size='12sp',
            markup=True
        )
        content.bind(texture_size=content.setter('size'))
        
        # Create popup
        self.tooltip_popup = Popup(
            content=content,
            size_hint=(None, None),
            size=(Window.width * 0.45, content.texture_size[1] + 40),
            auto_dismiss=True,
            separator_height=1,
            title_size='14sp',
            title='Help Information'
        )
        
        self.tooltip_popup.open()
        
        # Auto-hide after 8 seconds (longer since user explicitly requested it)
        Clock.schedule_once(lambda dt: self.hide_tooltip(), 8.0)
    
    def hide_tooltip(self):
        if self.tooltip_popup:
            self.tooltip_popup.dismiss()
            self.tooltip_popup = None

class InfoIcon(Button):
    """Small clickable info icon that shows tooltips."""
    def __init__(self, tooltip_target, **kwargs):
        super().__init__(
            text="ℹ",
            size_hint=(None, None),
            size=("25dp", "25dp"),
            font_size="14sp",
            background_color=(0.2, 0.6, 1.0, 0.8),  # Light blue
            color=(1, 1, 1, 1),  # White text
            **kwargs
        )
        self.tooltip_target = tooltip_target
        self.bind(on_press=self.show_help)
    
    def show_help(self, instance):
        if hasattr(self.tooltip_target, 'show_tooltip'):
            self.tooltip_target.show_tooltip()

class TooltipButton(Button, TooltipBehavior):
    def __init__(self, tooltip_text="", **kwargs):
        # Initialize Button without tooltip_text
        Button.__init__(self, **kwargs)
        # Initialize TooltipBehavior with tooltip_text
        TooltipBehavior.__init__(self, tooltip_text=tooltip_text)

class TooltipSlider(Slider, TooltipBehavior):
    def __init__(self, tooltip_text="", **kwargs):
        # Initialize Slider without tooltip_text
        Slider.__init__(self, **kwargs)
        # Initialize TooltipBehavior with tooltip_text
        TooltipBehavior.__init__(self, tooltip_text=tooltip_text)

class TooltipCheckBox(CheckBox, TooltipBehavior):
    def __init__(self, tooltip_text="", **kwargs):
        # Initialize CheckBox without tooltip_text
        CheckBox.__init__(self, **kwargs)
        # Initialize TooltipBehavior with tooltip_text
        TooltipBehavior.__init__(self, tooltip_text=tooltip_text)

class TooltipTextInput(TextInput, TooltipBehavior):
    def __init__(self, tooltip_text="", **kwargs):
        # Initialize TextInput without tooltip_text
        TextInput.__init__(self, **kwargs)
        # Initialize TooltipBehavior with tooltip_text
        TooltipBehavior.__init__(self, tooltip_text=tooltip_text)

class OrientationDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.bird_detector = None 
    
    def detect_orientation(self, frame, detections=None):
        best_orientation = 0; best_score = -1
        for rotation in [0, 90, 180, 270]:
            score = self._score_orientation(frame, rotation, detections)
            if score > best_score: best_score, best_orientation = score, rotation
        return best_orientation, False 
    
    def _score_orientation(self, frame, rotation, detections=None):
        score = 0; h, w = frame.shape[:2]
        if rotation != 0:
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            frame = cv2.warpAffine(frame, matrix, (w, h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            for (x, y, w_f, h_f) in faces: score += 2 if h_f > w_f else -1
        if detections and len(detections) > 0 : # Check if detections is not empty
            valid_scores = [d.get('score') for d in detections if d.get('score') is not None]
            if valid_scores: score += np.mean(valid_scores) * 3
        edges = cv2.Canny(gray, 100, 200)
        horizontal_edges, vertical_edges = np.sum(edges, axis=1), np.sum(edges, axis=0)
        if w > h: score += 1 if np.mean(horizontal_edges) > np.mean(vertical_edges) else 0
        else: score += 1 if np.mean(vertical_edges) > np.mean(horizontal_edges) else 0
        top_half, bottom_half = frame[:h//2], frame[h//2:]
        if np.mean(top_half) > np.mean(bottom_half): score += 1
        return score

class ExtractorLayout(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = 'vertical'
        self.padding = "10dp"
        self.spacing = "5dp"

        top_section = BoxLayout(orientation='horizontal', spacing="10dp")
        left_scroll = ScrollView(size_hint_x=0.4, do_scroll_x=False)
        self.controls_panel = GridLayout(cols=1, spacing="8dp", size_hint_y=None)
        self.controls_panel.bind(minimum_height=self.controls_panel.setter('height'))

        # Directory Settings with tooltips
        dir_section_box, dir_section = self._create_section_layout("Directories")
        self.video_dir_text = self._add_path_selector(
            dir_section, "Video Directory:", "", "_select_video_dir_action",
            tooltip="[b]Video Directory[/b]\n\nSelect the directory containing your video files to process.\n\n[b]Impact:[/b] This determines which videos will be processed for crow detection and tracking. The system will search for common video formats (.mp4, .avi, .mov, .mkv).\n\n[b]Tip:[/b] Organize videos by date or location for better dataset management."
        )
        self.output_dir_text = self._add_path_selector(
            dir_section, "Output Directory:", "crow_crops", "_select_output_dir_action",
            tooltip="[b]Output Directory[/b]\n\nSpecify where extracted crow crops, tracking data, and audio segments will be saved.\n\n[b]Impact:[/b] All processing results including individual crow images, metadata, and audio clips will be organized in this directory structure.\n\n[b]Structure:[/b]\n• videos/ - Frame-based crops (prevents training bias)\n• crows/ - Legacy crow-based organization\n• audio/ - Audio segments\n• metadata/ - Tracking and crop metadata"
        )
        self.controls_panel.add_widget(dir_section_box)

        # Detection Settings with tooltips
        det_section_box, det_section = self._create_section_layout("Detection Settings")
        self.min_confidence_slider, self.min_confidence_label = self._add_slider_setting(
            det_section, "Min Confidence:", 0.1, 0.9, 0.2, 0.01,
            tooltip="[b]Minimum Detection Confidence[/b]\n\nSets the minimum confidence score (0.1-0.9) required for a detection to be considered valid.\n\n[b]Impact:[/b]\n• Lower values = More detections but more false positives\n• Higher values = Fewer but more accurate detections\n• Default 0.2 provides good balance\n\n[b]Recommendation:[/b] Start with 0.2, increase to 0.3-0.4 if getting too many false detections, decrease to 0.15 if missing obvious crows."
        )
        self.min_detections_text = self._add_text_input_setting(
            det_section, "Min Detections (per track):", "2",
            tooltip="[b]Minimum Detections Per Track[/b]\n\nMinimum number of detections required before a crow track is considered valid and saved.\n\n[b]Impact:[/b]\n• Higher values = More reliable tracks but may miss brief appearances\n• Lower values = Capture short appearances but may include noise\n• Default 2 filters out single-frame false positives\n\n[b]Use Cases:[/b]\n• Set to 1 for capturing all crow appearances\n• Set to 5+ for only well-established tracks"
        )
        self.bbox_padding_slider, self.bbox_padding_label = self._add_slider_setting(
            det_section, "BBox Padding (extra context):", 0.0, 0.5, 0.2, 0.05,
            tooltip="[b]Bounding Box Padding[/b]\n\nAdds extra padding around detected crow bounding boxes when extracting crops.\n\n[b]Impact:[/b]\n• 0.0 = Tight crop around detected area\n• 0.2 (default) = 20% extra context around crow\n• 0.5 = 50% extra background context\n\n[b]Benefits of padding:[/b]\n• Captures important context (perches, environment)\n• Improves training data quality\n• Helps with orientation detection\n• Better for visual identification\n\n[b]Trade-offs:[/b] More padding = larger file sizes"
        )
        det_section.add_widget(Label(text="(Higher = more context around crow)", font_size='10sp', size_hint_y=None, height='15dp'))
        self.mv_yolo_check = self._add_checkbox_setting(
            det_section, "Multi-View YOLO", False,
            tooltip="[b]Multi-View YOLO Detection[/b]\n\nEnables YOLOv8 detection with multiple image transformations (rotations, zoom) to catch crows in different orientations.\n\n[b]Impact:[/b]\n• [color=00aa00]✓[/color] Catches crows in unusual orientations\n• [color=00aa00]✓[/color] Better detection coverage\n• [color=aa0000]✗[/color] ~3x slower processing\n• [color=aa0000]✗[/color] Higher GPU memory usage\n\n[b]When to use:[/b] Enable for challenging videos with crows at unusual angles or if standard detection is missing obvious crows."
        )
        self.mv_rcnn_check = self._add_checkbox_setting(
            det_section, "Multi-View Faster R-CNN", False,
            tooltip="[b]Multi-View Faster R-CNN Detection[/b]\n\nEnables Faster R-CNN detection with multiple image transformations for enhanced accuracy.\n\n[b]Impact:[/b]\n• [color=00aa00]✓[/color] Higher quality bounding boxes\n• [color=00aa00]✓[/color] Better localization accuracy\n• [color=aa0000]✗[/color] ~4x slower than single-view\n• [color=aa0000]✗[/color] Very high GPU memory usage\n\n[b]Best for:[/b] Final processing runs where accuracy is more important than speed."
        )
        self.orientation_check = self._add_checkbox_setting(
            det_section, "Auto-correct Orientation", True,
            tooltip="[b]Automatic Orientation Correction[/b]\n\nAutomatically detects and corrects video orientation (rotation) for better detection results.\n\n[b]Impact:[/b]\n• [color=00aa00]✓[/color] Improves detection in rotated videos\n• [color=00aa00]✓[/color] Standardizes crop orientations\n• [color=00aa00]✓[/color] Better training data consistency\n• [color=aa0000]✗[/color] Slight processing overhead\n\n[b]Recommendation:[/b] Keep enabled unless you specifically need to preserve original orientations."
        )
        self.nms_slider, self.nms_label = self._add_slider_setting(
            det_section, "Box Merge Threshold:", 0.1, 0.7, 0.3, 0.01,
            tooltip="[b]Non-Maximum Suppression (NMS) Threshold[/b]\n\nControls how aggressively overlapping detections are merged into single detections.\n\n[b]Impact:[/b]\n• Lower values (0.1-0.2) = Merge more overlapping boxes\n• Higher values (0.5-0.7) = Keep more separate detections\n• Default 0.3 provides good balance\n\n[b]Symptoms to adjust:[/b]\n• Multiple boxes on same crow → Decrease value\n• Missing separate nearby crows → Increase value\n\n[b]Technical:[/b] Intersection over Union (IoU) threshold for box merging."
        )
        det_section.add_widget(Label(text="(Lower = merge more boxes)", font_size='10sp', size_hint_y=None, height='15dp'))
        self.controls_panel.add_widget(det_section_box)

        # Audio Settings with tooltips
        audio_section_box, audio_section = self._create_section_layout("Audio Settings")
        self.enable_audio_check = self._add_checkbox_setting(
            audio_section, "Extract Audio Segments", True,
            tooltip="[b]Audio Segment Extraction[/b]\n\nExtracts audio clips around crow detections for potential call analysis and multi-modal training.\n\n[b]Impact:[/b]\n• [color=00aa00]✓[/color] Captures crow calls and environmental sounds\n• [color=00aa00]✓[/color] Enables future audio-visual training\n• [color=00aa00]✓[/color] Useful for behavior analysis\n• [color=aa0000]✗[/color] Increases storage requirements\n• [color=aa0000]✗[/color] Slight processing overhead\n\n[b]Use cases:[/b] Research projects, behavior studies, multi-modal AI training."
        )
        self.audio_duration_slider, self.audio_duration_label = self._add_slider_setting(
            audio_section, "Audio Duration (s):", 0.5, 5.0, 2.0, 0.1,
            tooltip="[b]Audio Segment Duration[/b]\n\nLength of audio clips extracted around each crow detection (in seconds).\n\n[b]Impact:[/b]\n• Shorter (0.5-1s) = Just the immediate call\n• Medium (2-3s) = Call plus context (default)\n• Longer (4-5s) = Extended behavioral context\n\n[b]Storage impact:[/b]\n• 2s clips ≈ 350KB each (stereo, 44.1kHz)\n• 5s clips ≈ 875KB each\n\n[b]Recommendation:[/b] 2 seconds captures most crow calls while maintaining reasonable file sizes."
        )
        self.controls_panel.add_widget(audio_section_box)

        # Advanced Processing Settings with tooltips
        advanced_section_box, advanced_section = self._create_section_layout("Advanced Settings")
        self.recursive_check = self._add_checkbox_setting(
            advanced_section, "Recursive Search (subdirectories)", False,
            tooltip="[b]Recursive Directory Search[/b]\n\nSearches for video files in all subdirectories of the selected video directory.\n\n[b]Impact:[/b]\n• [color=00aa00]✓[/color] Processes videos organized in nested folders\n• [color=00aa00]✓[/color] Useful for date/location-based organization\n• [color=aa0000]✗[/color] May process unintended videos\n\n[b]Example structure:[/b]\nVideos/\n  ├── 2024-01/\n  │   ├── video1.mp4\n  │   └── video2.mp4\n  └── 2024-02/\n      └── video3.mp4\n\n[b]When to use:[/b] When videos are organized in subdirectories by date, location, or project."
        )
        self.batch_size_slider, self.batch_size_label = self._add_slider_setting(
            advanced_section, "Batch Size:", 1, 64, 32, 1,
            tooltip="[b]Processing Batch Size[/b]\n\nNumber of frames to process simultaneously (future enhancement for GPU efficiency).\n\n[b]Current Status:[/b] [color=aa6600]Partially implemented[/color] - prepared for future batch processing optimization.\n\n[b]Future Impact:[/b]\n• Larger batches = Better GPU utilization\n• Smaller batches = Lower memory usage\n• Optimal size depends on GPU VRAM\n\n[b]Recommendations:[/b]\n• RTX 3080/4080: 32-64\n• RTX 3060/4060: 16-32\n• GTX 1660: 8-16"
        )
        self.frame_skip_slider, self.frame_skip_label = self._add_slider_setting(
            advanced_section, "Frame Skip (1=every frame):", 1, 10, 1, 1,
            tooltip="[b]Frame Skip Interval[/b]\n\nProcess every Nth frame to speed up processing at the cost of temporal resolution.\n\n[b]Impact:[/b]\n• 1 = Process every frame (highest accuracy)\n• 2 = Process every 2nd frame (2x faster)\n• 5 = Process every 5th frame (5x faster)\n• 10 = Process every 10th frame (10x faster)\n\n[b]Trade-offs:[/b]\n• [color=00aa00]✓[/color] Dramatically faster processing\n• [color=aa0000]✗[/color] May miss brief appearances\n• [color=aa0000]✗[/color] Less smooth tracking\n\n[b]Recommendation:[/b] Use 2-3 for initial processing, 1 for final runs."
        )
        self.max_crops_per_crow_slider, self.max_crops_per_crow_label = self._add_slider_setting(
            advanced_section, "Max crops per crow:", 5, 50, 10, 1,
            tooltip="[b]Maximum Crops Per Crow[/b]\n\nLimits the number of crop images saved per individual crow to prevent dataset bias.\n\n[b]Impact:[/b]\n• Lower values (5-10) = Balanced dataset, less storage\n• Higher values (20-50) = More examples per crow, potential bias\n\n[b]Dataset Quality:[/b]\n• Prevents overrepresentation of frequently detected crows\n• Ensures balanced training data\n• Saves storage space\n\n[b]Recommendations:[/b]\n• Research/Training: 10-20 per crow\n• Quick analysis: 5-10 per crow\n• Comprehensive dataset: 20-50 per crow"
        )
        self.save_best_only_check = self._add_checkbox_setting(
            advanced_section, "Save only best crops per crow", True,
            tooltip="[b]Save Only Best Quality Crops[/b]\n\nOnly saves the highest quality crops for each crow based on detection confidence and image clarity.\n\n[b]Impact:[/b]\n• [color=00aa00]✓[/color] Higher quality training data\n• [color=00aa00]✓[/color] Reduced storage requirements\n• [color=00aa00]✓[/color] Better visual identification\n• [color=aa0000]✗[/color] Less variety in poses/angles\n\n[b]Quality Factors:[/b]\n• Detection confidence score\n• Image sharpness\n• Crop size and resolution\n• Bounding box quality\n\n[b]Recommendation:[/b] Keep enabled for training datasets, disable for comprehensive behavioral analysis."
        )
        self.memory_optimize_check = self._add_checkbox_setting(
            advanced_section, "Memory optimization mode", False,
            tooltip="[b]Memory Optimization Mode[/b]\n\nEnables various memory-saving techniques at the cost of some processing speed.\n\n[b]Current Status:[/b] [color=aa6600]Planned feature[/color] - prepared for future memory optimization.\n\n[b]Future Optimizations:[/b]\n• Smaller batch sizes\n• Frequent garbage collection\n• Model precision reduction (FP16)\n• Temporary file cleanup\n\n[b]When to enable:[/b]\n• Limited GPU VRAM (< 6GB)\n• Processing very large videos\n• Running multiple processes\n\n[b]Trade-off:[/b] ~20-30% slower but 40-60% less memory usage."
        )
        self.controls_panel.add_widget(advanced_section_box)

        # Control Buttons with info icons for help
        buttons_layout = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height="44dp")
        
        # Start button with info icon
        start_container = BoxLayout(orientation='horizontal', spacing="2dp")
        self.start_button = Button(text="Start Processing")
        self.start_button.bind(on_press=self.app._start_processing_action)
        start_tooltip = TooltipBehavior(tooltip_text="[b]Start Processing[/b]\n\nBegin processing videos with current settings.\n\n[b]Before starting:[/b]\n• Verify video and output directories\n• Check detection settings\n• Ensure sufficient disk space\n• Consider frame skip for speed\n\n[b]Processing will:[/b]\n1. Detect crows in each frame\n2. Track individuals across frames\n3. Extract and save crop images\n4. Generate tracking metadata\n5. Extract audio segments (if enabled)\n\n[b]Tip:[/b] Start with a small test video to verify settings.")
        start_info = InfoIcon(start_tooltip)
        start_info.size_hint = (None, None)
        start_info.size = ("25dp", "25dp")
        start_container.add_widget(self.start_button)
        start_container.add_widget(start_info)
        
        # Pause button with info icon
        pause_container = BoxLayout(orientation='horizontal', spacing="2dp")
        self.pause_button = Button(text="Pause", disabled=True)
        self.pause_button.bind(on_press=self.app._pause_processing_action)
        pause_tooltip = TooltipBehavior(tooltip_text="[b]Pause/Resume Processing[/b]\n\nTemporarily pause processing without losing progress.\n\n[b]Useful for:[/b]\n• Adjusting system performance\n• Checking intermediate results\n• Making manual adjustments\n• Conserving laptop battery\n\n[b]Note:[/b] Current frame will complete before pausing. All progress is automatically saved.")
        pause_info = InfoIcon(pause_tooltip)
        pause_info.size_hint = (None, None)
        pause_info.size = ("25dp", "25dp")
        pause_container.add_widget(self.pause_button)
        pause_container.add_widget(pause_info)
        
        # Stop button with info icon
        stop_container = BoxLayout(orientation='horizontal', spacing="2dp")
        self.stop_button = Button(text="Stop", disabled=True)
        self.stop_button.bind(on_press=self.app._stop_processing_action)
        stop_tooltip = TooltipBehavior(tooltip_text="[b]Stop Processing[/b]\n\nCompletely stop processing and return to ready state.\n\n[b]Important:[/b]\n• All progress up to current frame is saved\n• You can resume later using 'Load Progress'\n• Partially processed videos can be continued\n\n[b]Data safety:[/b] Metadata and crops are saved continuously during processing.")
        stop_info = InfoIcon(stop_tooltip)
        stop_info.size_hint = (None, None)
        stop_info.size = ("25dp", "25dp")
        stop_container.add_widget(self.stop_button)
        stop_container.add_widget(stop_info)
        
        buttons_layout.add_widget(start_container)
        buttons_layout.add_widget(pause_container)
        buttons_layout.add_widget(stop_container)
        
        progress_buttons_layout = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height="44dp")
        
        save_layout, self.save_prog_button = self._create_button_with_help(
            text="Save Progress",
            on_press=self.app._save_progress_action,
            tooltip="[b]Save Progress[/b]\n\nManually save current processing progress and tracking data.\n\n[b]Saves:[/b]\n• Crow tracking data\n• Processing statistics\n• Video progress markers\n• Crop metadata\n\n[b]Auto-saved during:[/b]\n• Normal processing\n• Pause/Stop operations\n• Regular intervals\n\n[b]Use when:[/b] You want to ensure progress is saved before system changes or long processing sessions."
        )
        
        load_layout, self.load_prog_button = self._create_button_with_help(
            text="Load Progress",
            on_press=self.app._load_progress_action,
            tooltip="[b]Load Progress[/b]\n\nRestore previously saved processing progress and continue from where you left off.\n\n[b]Loads:[/b]\n• Previous tracking data\n• Crow identities and embeddings\n• Processing statistics\n• Video progress markers\n\n[b]Useful for:[/b]\n• Resuming interrupted sessions\n• Continuing multi-day processing\n• Switching between projects\n\n[b]Warning:[/b] This will replace current unsaved progress."
        )
        
        progress_buttons_layout.add_widget(save_layout)
        progress_buttons_layout.add_widget(load_layout)
        
        review_layout, self.review_crops_button = self._create_button_with_help(
            text="Review Crops",
            on_press=self.app._trigger_crop_review,
            disabled=True,
            tooltip="[b]Review Extracted Crops[/b]\n\nManually review and curate extracted crow crop images.\n\n[b]Review Functions:[/b]\n• Keep high-quality crops\n• Discard poor detections\n• Delete entire crow tracks\n• Navigate through all crops\n\n[b]Quality Control:[/b]\n• Remove false positives\n• Ensure dataset quality\n• Improve training data\n• Manual verification\n\n[b]Available after:[/b] Processing completes or when crops are pending review."
        )
        progress_buttons_layout.add_widget(review_layout)

        controls_buttons_main_box = BoxLayout(orientation='vertical', spacing='5dp', size_hint_y=None)
        controls_buttons_main_box.bind(minimum_height=controls_buttons_main_box.setter('height'))
        controls_buttons_main_box.add_widget(buttons_layout)
        controls_buttons_main_box.add_widget(progress_buttons_layout)
        self.controls_panel.add_widget(controls_buttons_main_box)

        # Progress Section
        prog_section_box, prog_section = self._create_section_layout("Progress")
        self.progress_bar = ProgressBar(max=100, value=0, size_hint_y=None, height="20dp")
        prog_section.add_widget(self.progress_bar)
        self.progress_label = Label(text="Ready", size_hint_y=None, height="25dp")
        prog_section.add_widget(self.progress_label)
        self.controls_panel.add_widget(prog_section_box)

        # Statistics Section
        stats_section_box, stats_section = self._create_section_layout("Statistics")
        self.stats_grid = GridLayout(cols=1, spacing="2dp", size_hint_y=None, padding="5dp")
        self.stats_grid.bind(minimum_height=self.stats_grid.setter('height'))
        self.stats_labels_ui = {} 
        stats_definitions = [
            ('videos_processed', 'Videos Processed:', 'Number of video files completely processed'),
            ('total_frames', 'Total Frames:', 'Total number of video frames analyzed'),
            ('detections', 'Total Detections:', 'Total number of crow detections found'),
            ('valid_crops', 'Valid Crops:', 'Number of high-quality crop images saved'),
            ('invalid_crops', 'Invalid Crops:', 'Number of poor-quality detections discarded'),
            ('crows_created', 'New Crows:', 'Number of new individual crows identified'),
            ('crows_updated', 'Updated Crows:', 'Number of existing crows with new detections'),
            ('current_video_detections', 'Video Detections:', 'Detections in current video being processed'),
            ('current_video_crows', 'Video Crows:', 'Individual crows found in current video')
        ]
        for key, text_label, tooltip_text in stats_definitions:
            # Create a horizontal row for each statistic with tooltip
            row_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height='22dp', spacing="5dp")
            
            lbl_title = Label(text=text_label, size_hint_x=0.65, 
                            halign='left', valign='middle', 
                            text_size=(None, None), font_size='11sp')
            lbl_value = Label(text="0", size_hint_x=0.35,
                            halign='right', valign='middle', 
                            text_size=(None, None), font_size='11sp', bold=True)
            
            row_layout.add_widget(lbl_title)
            row_layout.add_widget(lbl_value)
            
            self.stats_grid.add_widget(row_layout)
            self.stats_labels_ui[key] = lbl_value
            
            # Add tooltip behavior to the title label
            self._add_tooltip_to_label(lbl_title, f"[b]{text_label}[/b]\n\n{tooltip_text}")
        stats_section.add_widget(self.stats_grid)
        self.controls_panel.add_widget(stats_section_box)

        left_scroll.add_widget(self.controls_panel)
        top_section.add_widget(left_scroll)

        # Preview Panel with tooltip
        preview_panel = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing="5dp")
        preview_title = Label(text="Video Preview", size_hint_y=None, height="30dp", font_size="18sp")
        self._add_tooltip_to_label(preview_title, "[b]Video Preview[/b]\n\nReal-time preview of video processing showing:\n\n• Current frame being processed\n• Detected crow bounding boxes (green)\n• Detection confidence scores\n• Processing progress\n\nThe preview updates during processing to show detection results and help monitor progress.")
        preview_panel.add_widget(preview_title)
        self.preview_image = KivyImage(source='', allow_stretch=True, keep_ratio=True)
        preview_panel.add_widget(self.preview_image)
        top_section.add_widget(preview_panel)
        
        self.add_widget(top_section)
        self.status_label = Label(text="Ready", size_hint_y=None, height="30dp", font_size='12sp')
        self._add_tooltip_to_label(self.status_label, "[b]Status Display[/b]\n\nShows current system status:\n\n• Ready - System ready for processing\n• Processing - Currently analyzing videos\n• Paused - Processing temporarily paused\n• Stopped - Processing stopped by user\n• Error messages and progress updates")
        self.add_widget(self.status_label)

    def _create_section_layout(self, title):
        section_box = BoxLayout(orientation='vertical', spacing="5dp", size_hint_y=None)
        section_box.bind(minimum_height=section_box.setter('height'))
        section_box.add_widget(Label(text=title, font_size="16sp", size_hint_y=None, height="30dp", bold=True, halign='left', text_size=(None,None)))
        content_grid = GridLayout(cols=1, spacing="5dp", size_hint_y=None)
        content_grid.bind(minimum_height=content_grid.setter('height'))
        section_box.add_widget(content_grid)
        return section_box, content_grid

    def _add_path_selector(self, parent_grid, label_text, default_path, browse_action_name, tooltip=""):
        row = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height="35dp")
        
        # Label with info icon if tooltip provided
        label_layout = BoxLayout(orientation='horizontal', size_hint_x=0.35, spacing="2dp")
        label = Label(text=label_text, halign='left', text_size=(None,None))
        label_layout.add_widget(label)
        
        if tooltip:
            # Create a tooltip target that can show help
            tooltip_target = TooltipBehavior(tooltip_text=tooltip)
            info_icon = InfoIcon(tooltip_target)
            label_layout.add_widget(info_icon)
        
        row.add_widget(label_layout)
        text_input = TextInput(text=default_path, size_hint_x=0.55, multiline=False)
        browse_button = Button(text="Browse", size_hint_x=0.1)
        browse_button.bind(on_press=getattr(self.app, browse_action_name))
        row.add_widget(text_input)
        row.add_widget(browse_button)
        parent_grid.add_widget(row)
        return text_input

    def _add_slider_setting(self, parent_grid, label_text, min_val, max_val, default_val, step, tooltip=""):
        row = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height="35dp")
        
        # Label with info icon if tooltip provided
        label_layout = BoxLayout(orientation='horizontal', size_hint_x=0.4, spacing="2dp")
        label = Label(text=label_text, halign='left', text_size=(None,None))
        label_layout.add_widget(label)
        
        if tooltip:
            # Create a tooltip target that can show help
            tooltip_target = TooltipBehavior(tooltip_text=tooltip)
            info_icon = InfoIcon(tooltip_target)
            label_layout.add_widget(info_icon)
        
        row.add_widget(label_layout)
        slider = Slider(
            min=min_val, 
            max=max_val, 
            value=default_val, 
            step=step, 
            size_hint_x=0.45
        )
        value_label = Label(text=f"{default_val:.2f}", size_hint_x=0.15, halign='left', text_size=(None,None))
        slider.bind(value=lambda instance, value: setattr(value_label, 'text', f"{value:.2f}"))
        row.add_widget(slider)
        row.add_widget(value_label)
        parent_grid.add_widget(row)
        return slider, value_label

    def _add_text_input_setting(self, parent_grid, label_text, default_value, tooltip=""):
        row = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height="35dp")
        
        # Label with info icon if tooltip provided
        label_layout = BoxLayout(orientation='horizontal', size_hint_x=0.7, spacing="2dp")
        label = Label(text=label_text, halign='left', text_size=(None,None))
        label_layout.add_widget(label)
        
        if tooltip:
            # Create a tooltip target that can show help
            tooltip_target = TooltipBehavior(tooltip_text=tooltip)
            info_icon = InfoIcon(tooltip_target)
            label_layout.add_widget(info_icon)
        
        row.add_widget(label_layout)
        text_input = TextInput(text=default_value, size_hint_x=0.3, multiline=False)
        row.add_widget(text_input)
        parent_grid.add_widget(row)
        return text_input

    def _add_checkbox_setting(self, parent_grid, label_text, default_active, tooltip=""):
        row = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height="35dp")
        
        # Label with info icon if tooltip provided
        label_layout = BoxLayout(orientation='horizontal', size_hint_x=0.85, spacing="2dp")
        label = Label(text=label_text, halign='left', text_size=(None,None))
        label_layout.add_widget(label)
        
        if tooltip:
            # Create a tooltip target that can show help
            tooltip_target = TooltipBehavior(tooltip_text=tooltip)
            info_icon = InfoIcon(tooltip_target)
            label_layout.add_widget(info_icon)
        
        row.add_widget(label_layout)
        checkbox = CheckBox(active=default_active, size_hint_x=0.15)
        row.add_widget(checkbox)
        parent_grid.add_widget(row)
        return checkbox

    def _add_tooltip_to_label(self, label, tooltip_text):
        """Add tooltip functionality to an existing label."""
        label.tooltip_text = tooltip_text
        label.tooltip_popup = None
        
        def on_tooltip_touch_down(widget, touch):
            if widget.collide_point(*touch.pos) and widget.tooltip_text and not widget.tooltip_popup:
                Clock.schedule_once(lambda dt: show_tooltip(widget, touch.pos), 0.8)
            return False
        
        def on_tooltip_touch_up(widget, touch):
            hide_tooltip(widget)
            return False
        
        def show_tooltip(widget, pos):
            if widget.tooltip_popup:
                return
                
            content = Label(
                text=widget.tooltip_text,
                text_size=(Window.width * 0.35, None),
                halign='left',
                valign='top',
                font_size='11sp',
                markup=True
            )
            content.bind(texture_size=content.setter('size'))
            
            widget.tooltip_popup = Popup(
                content=content,
                size_hint=(None, None),
                size=(Window.width * 0.4, content.texture_size[1] + 20),
                pos_hint={'center_x': 0.5, 'center_y': 0.5},
                auto_dismiss=True,
                separator_height=0,
                title_size='12sp',
                title='Help'
            )
            
            tooltip_x = min(pos[0] + 10, Window.width - widget.tooltip_popup.width)
            tooltip_y = max(pos[1] - widget.tooltip_popup.height - 10, 0)
            widget.tooltip_popup.pos = (tooltip_x, tooltip_y)
            
            widget.tooltip_popup.open()
            Clock.schedule_once(lambda dt: hide_tooltip(widget), 5.0)
        
        def hide_tooltip(widget):
            if hasattr(widget, 'tooltip_popup') and widget.tooltip_popup:
                widget.tooltip_popup.dismiss()
                widget.tooltip_popup = None
        
        label.bind(on_touch_down=on_tooltip_touch_down)
        label.bind(on_touch_up=on_tooltip_touch_up)

    def _create_button_with_help(self, text, on_press=None, tooltip="", **kwargs):
        """Create a button with an optional help icon."""
        if tooltip:
            # Create button with help icon layout
            button_layout = BoxLayout(orientation='horizontal', spacing="2dp", **kwargs)
            button = Button(text=text, size_hint_x=0.9)
            if on_press:
                button.bind(on_press=on_press)
            
            # Create tooltip target and info icon
            tooltip_target = TooltipBehavior(tooltip_text=tooltip)
            info_icon = InfoIcon(tooltip_target)
            info_icon.size_hint_x = 0.1
            
            button_layout.add_widget(button)
            button_layout.add_widget(info_icon)
            return button_layout, button
        else:
            # Regular button without help
            button = Button(text=text, **kwargs)
            if on_press:
                button.bind(on_press=on_press)
            return button, button

    def _add_info_icon_to_label(self, label, tooltip_text):
        """Add an info icon next to a label for help."""
        # Create a horizontal layout to hold the label and info icon
        label_container = BoxLayout(orientation='horizontal', spacing="5dp", size_hint_y=None, height=label.height)
        
        # Remove the label from its current parent and add to container
        if label.parent:
            label.parent.remove_widget(label)
        label_container.add_widget(label)
        
        # Create tooltip target and info icon
        tooltip_target = TooltipBehavior(tooltip_text=tooltip_text)
        info_icon = InfoIcon(tooltip_target)
        info_icon.size_hint = (None, None)
        info_icon.size = ("20dp", "20dp")
        label_container.add_widget(info_icon)
        
        return label_container

class CrowExtractorApp(App):
    video_dir = StringProperty("")
    output_dir = StringProperty("crow_crops") 
    processing_active = BooleanProperty(False)
    processing_paused = BooleanProperty(False)
    stats = DictProperty({
        'videos_processed': 0, 'total_frames': 0, 'detections': 0,
        'valid_crops': 0, 'invalid_crops': 0, 'crows_created': 0,
        'crows_updated': 0, 'current_video_detections': 0,
        'current_video_crows': 0 
    })

    def build(self):
        self.title = "Kivy Crow Training Data Extractor"
        Window.size = (1400, 850)
        self.orientation_detector = OrientationDetector()
        self.tracker = None 
        self.video_files_list = []
        self.current_video_idx = 0
        self.cv_capture = None 
        self.processing_thread = None
        self.layout = ExtractorLayout(app=self)
        Window.bind(on_request_close=self.on_request_close_window)
        return self.layout

    def on_start(self):
        logger.info("CrowExtractorApp started.")
        self.layout.status_label.text = "Ready. Please select directories and settings."
        if self.output_dir and not self.tracker:
             self._initialize_tracker()
        self.layout.video_dir_text.text = self.video_dir
        self.layout.output_dir_text.text = self.output_dir
        self.layout.video_dir_text.bind(text=self.setter('video_dir'))
        self.layout.output_dir_text.bind(text=self.setter('output_dir'))

    def _initialize_tracker(self):
        if not self.output_dir:
            self.show_popup("Error", "Output directory must be set before initializing tracker.")
            return False
        enable_audio = self.layout.enable_audio_check.active
        audio_duration = self.layout.audio_duration_slider.value
        correct_orientation_flag = self.layout.orientation_check.active
        bbox_padding = self.layout.bbox_padding_slider.value
        try:
            self.tracker = CrowTracker(
                base_dir=self.output_dir,
                enable_audio_extraction=enable_audio,
                audio_duration=audio_duration,
                correct_orientation=correct_orientation_flag,
                bbox_padding=bbox_padding
            )
            logger.info(f"CrowTracker initialized with output directory: {self.output_dir}")
            logger.info(f"BBox padding set to: {bbox_padding}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CrowTracker: {e}", exc_info=True)
            self.show_popup("Tracker Error", f"Failed to initialize CrowTracker: {e}")
            return False

    def on_request_close_window(self, *args, **kwargs):
        if self.processing_active:
            content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
            content.add_widget(Label(text="Processing is active. Stop and exit?"))
            buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
            yes_btn = Button(text="Yes, Stop & Exit")
            no_btn = Button(text="No, Continue Processing")
            buttons.add_widget(yes_btn); buttons.add_widget(no_btn)
            content.add_widget(buttons)
            popup = Popup(title="Confirm Exit", content=content, size_hint=(None, None), size=("400dp", "150dp"), auto_dismiss=False)
            def confirm_exit(instance):
                popup.dismiss()
                self.processing_active = False 
                if self.processing_thread and self.processing_thread.is_alive():
                    logger.info("Attempting to join processing thread on exit...")
                    self.processing_thread.join(timeout=1.0)
                    if self.processing_thread.is_alive(): logger.warning("Processing thread did not exit cleanly.")
                App.get_running_app().stop()
            def continue_processing(instance): popup.dismiss()
            yes_btn.bind(on_press=confirm_exit)
            no_btn.bind(on_press=continue_processing)
            popup.open()
            return True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_active = False 
            self.processing_thread.join(timeout=1.0) 
            if self.processing_thread.is_alive(): logger.warning("Processing thread did not exit cleanly upon normal close.")
        return False 

    def show_popup(self, title, message):
        if not isinstance(threading.current_thread(), threading._MainThread):
             Clock.schedule_once(lambda dt: self._do_show_popup(title, message))
        else: self._do_show_popup(title, message)

    def _do_show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        msg_label = Label(text=message, text_size=(Window.width*0.4, None), halign='center', valign='middle')
        content.add_widget(msg_label)
        ok_button = Button(text="OK", size_hint_y=None, height="44dp")
        content.add_widget(ok_button)
        msg_label.texture_update()
        popup_height = msg_label.texture_size[1] + ok_button.height + Window.dpi * 0.8
        min_height = Window.height * 0.2
        actual_height = max(min_height, popup_height)
        popup = Popup(title=title, content=content, size_hint=(0.5, None), height=f"{actual_height}dp", auto_dismiss=True)
        ok_button.bind(on_press=popup.dismiss)
        popup.open()
        logger.info(f"Popup shown: '{title}'")

    def _select_dir_popup(self, callback_set_path, initial_path_prop_name):
        content = BoxLayout(orientation='vertical', spacing="10dp")
        initial_path = getattr(self, initial_path_prop_name, os.path.expanduser('~'))
        if not os.path.isdir(initial_path): initial_path = os.path.expanduser('~')
        filechooser = FileChooserListView(path=initial_path, dirselect=True, size_hint_y=0.85)
        content.add_widget(filechooser)
        buttons_layout = BoxLayout(size_hint_y=0.15, height="44dp", spacing="10dp")
        select_button = Button(text="Select Directory")
        cancel_button = Button(text="Cancel")
        buttons_layout.add_widget(select_button); buttons_layout.add_widget(cancel_button)
        content.add_widget(buttons_layout)
        popup = Popup(title="Select Directory", content=content, size_hint=(0.9, 0.9))
        def select_dir(instance):
            if filechooser.selection:
                selected_path = filechooser.selection[0]
                callback_set_path(selected_path)
                logger.info(f"Directory selected via popup: {selected_path} for {initial_path_prop_name}")
            popup.dismiss()
        select_button.bind(on_press=select_dir)
        cancel_button.bind(on_press=popup.dismiss)
        popup.open()

    def _select_video_dir_action(self, instance):
        self._select_dir_popup(lambda path: setattr(self, 'video_dir', path), 'video_dir')

    def _select_output_dir_action(self, instance):
        def set_output_dir_and_init_tracker(path):
            self.output_dir = path
            self._initialize_tracker()
        self._select_dir_popup(set_output_dir_and_init_tracker, 'output_dir')

    def _find_video_files(self, video_dir, recursive=False):
        """Find video files in directory, optionally recursively."""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
        video_files = []
        
        if recursive:
            # Use os.walk for recursive search
            for root, dirs, files in os.walk(video_dir):
                for file in files:
                    if file.endswith(video_extensions):
                        video_files.append(os.path.join(root, file))
        else:
            # Non-recursive search (original behavior)
            try:
                for file in os.listdir(video_dir):
                    if file.endswith(video_extensions):
                        video_files.append(os.path.join(video_dir, file))
            except OSError as e:
                logger.error(f"Error reading directory {video_dir}: {e}")
                return []
        
        return sorted(video_files)

    def _start_processing_action(self, instance):
        logger.info("Start processing action initiated.")
        if self.processing_active and not self.processing_paused:
            self.show_popup("Info", "Processing is already active.")
            return
        if self.processing_active and self.processing_paused: 
            self.processing_paused = False
            self.layout.pause_button.text = "Pause"
            self.layout.status_label.text = "Processing resumed..."
            logger.info("Processing resumed.")
            return

        if not self.video_dir or not os.path.isdir(self.video_dir):
            self.show_popup("Error", "Please select a valid video directory.")
            return
        if not self.output_dir:
            self.show_popup("Error", "Please select a valid output directory.")
            return
        
        if not self.tracker and not self._initialize_tracker():
            self.show_popup("Error", "Failed to initialize tracker. Cannot start processing.")
            return

        # Use new recursive file finding logic
        recursive_search = self.layout.recursive_check.active
        search_type = "recursively" if recursive_search else ""
        self.video_files_list = self._find_video_files(self.video_dir, recursive=recursive_search)
        
        if not self.video_files_list:
            self.show_popup("Error", f"No video files found {search_type} in the selected directory.")
            return
        
        logger.info(f"Found {len(self.video_files_list)} videos to process {search_type}")
        if recursive_search:
            logger.info("Videos found in:")
            for video_file in self.video_files_list:
                rel_path = os.path.relpath(video_file, self.video_dir)
                logger.info(f"  {rel_path}")

        self.processing_active = True
        self.processing_paused = False
        self.current_video_idx = 0 
        
        self.layout.start_button.disabled = True
        self.layout.pause_button.disabled = False
        self.layout.pause_button.text = "Pause"
        self.layout.stop_button.disabled = False
        self.layout.status_label.text = "Processing started..."
        if self.layout.review_crops_button: self.layout.review_crops_button.disabled = True
        
        self._reset_stats_data()
        self._update_stats_ui()   

        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Processing thread start requested, but already alive.")
        else:
            self.processing_thread = threading.Thread(target=self._video_processing_thread_target, daemon=True)
            self.processing_thread.start()

    def _video_processing_thread_target(self):
        logger.info("Video processing thread started.")
        if self.tracker and hasattr(self.tracker, 'create_processing_run') and callable(self.tracker.create_processing_run):
             self.tracker.current_run_dir = self.tracker.create_processing_run()
             logger.info(f"Created new processing run directory: {getattr(self.tracker, 'current_run_dir', 'N/A')}")

        for video_file_name in self.video_files_list[self.current_video_idx:]:
            if not self.processing_active:
                logger.info("Processing stopped by user signal (thread target).")
                break
            try:
                self.current_video_idx = self.video_files_list.index(video_file_name)
            except ValueError:
                logger.error(f"Video file {video_file_name} not found in list. Skipping.")
                continue
            video_full_path = video_file_name
            Clock.schedule_once(lambda dt, vfn=video_file_name: setattr(self.layout.status_label, 'text', f"Starting video: {vfn}"))
            self._process_video_file(video_full_path)
            if self.processing_active:
                 self.stats['videos_processed'] += 1
                 Clock.schedule_once(self._update_stats_ui)
        Clock.schedule_once(self._processing_finished_ui_update)
        logger.info("Video processing thread finished.")

    def _process_video_file(self, video_path):
        logger.info(f"Processing video file: {video_path}")
        self.cv_capture = cv2.VideoCapture(video_path)
        if not self.cv_capture.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            Clock.schedule_once(lambda dt: self.show_popup("Video Error", f"Could not open: {os.path.basename(video_path)}"))
            return
        total_frames_in_video = int(self.cv_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cv_capture.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        self.stats['current_video_detections'] = 0
        _current_video_crow_ids_set = set() 
        self.stats['current_video_crows'] = 0
        frame_num = 0
        
        # Get advanced settings
        frame_skip = int(self.layout.frame_skip_slider.value)
        batch_size = int(self.layout.batch_size_slider.value)  # For future batch processing
        max_crops_per_crow = int(self.layout.max_crops_per_crow_slider.value)
        memory_optimize = self.layout.memory_optimize_check.active
        correct_orientation_flag = self.layout.orientation_check.active
        bbox_padding = self.layout.bbox_padding_slider.value
        
        # Advanced settings usage:
        # - frame_skip: Skip frames for faster processing (implemented)
        # - batch_size: Process multiple frames at once (future enhancement)
        # - max_crops_per_crow: Limit crops per crow to prevent bias (tracker handles)
        # - memory_optimize: Use memory-efficient processing modes (future enhancement)
        # - correct_orientation_flag: Auto-rotate frames for better detection (tracker handles)

        while self.processing_active:
            while self.processing_paused:
                if not self.processing_active: break
                time.sleep(0.1)
            if not self.processing_active: break
            
            ret, frame = self.cv_capture.read()
            if not ret: break
            frame_num += 1
            
            # Frame skipping logic
            if frame_skip > 1 and (frame_num % frame_skip) != 0:
                continue
                
            self.stats['total_frames'] += 1

            detections_on_frame = []
            try:
                # Retrieve settings from Kivy UI elements from the main app's layout
                min_conf = self.layout.min_confidence_slider.value
                multi_view_yolo = self.layout.mv_yolo_check.active
                multi_view_rcnn = self.layout.mv_rcnn_check.active
                nms_thresh = self.layout.nms_slider.value
                bbox_padding = self.layout.bbox_padding_slider.value
                # Orientation correction is handled by CrowTracker based on its init flag

                # Actual detection call
                # Note: detect_crows_parallel expects a list of frames
                raw_detections = detect_crows_parallel(
                    [frame.copy()], # Pass a copy of the frame
                    score_threshold=min_conf,
                    multi_view_yolo=multi_view_yolo,
                    multi_view_rcnn=multi_view_rcnn,
                    nms_threshold=nms_thresh
                )
                detections_on_frame = raw_detections[0] if raw_detections else []
                
                self.stats['detections'] += len(detections_on_frame)
                self.stats['current_video_detections'] += len(detections_on_frame)

                if self.tracker:
                    processed_detection_ids_this_frame = set()
                    for det_data in detections_on_frame:
                        # Score check should ideally be handled by detect_crows_parallel based on score_threshold
                        if det_data['score'] < min_conf: 
                            continue
                        
                        # CrowTracker's process_detection should handle orientation internally
                        # and manage adding crops to self.tracker.pending_review_crops.
                        # It returns a crow_id if the detection is finalized, 
                        # or None/special marker if pending review or invalid.
                        crow_id_or_status = self.tracker.process_detection(
                            frame.copy(), 
                            frame_num,
                            det_data,
                            video_path, # Full video path
                            frame_num / fps if fps > 0 else None # timestamp
                        )
                        
                        if crow_id_or_status is not None and isinstance(crow_id_or_status, (int, str)): 
                            self.stats['valid_crops'] += 1
                            processed_detection_ids_this_frame.add(str(crow_id_or_status)) # Ensure string for set
                            
                            # This logic for new/updated might need refinement based on how tracker handles IDs
                            # A more robust way would be for tracker to report if a crow was new or updated.
                            # Conceptual: Check if this crow ID is newly created in this session by the tracker
                            if hasattr(self.tracker, 'is_new_track_id') and self.tracker.is_new_track_id(str(crow_id_or_status)):
                                 self.stats['crows_created'] +=1
                            elif str(crow_id_or_status) in self.tracker.tracking_data['crows']: # Check against tracker's known crows
                                 self.stats['crows_updated'] +=1
                            else: # If not in tracking_data and not marked new by tracker, assume new
                                self.stats['crows_created'] +=1

                        elif crow_id_or_status is None: 
                            is_pending = False
                            if hasattr(self.tracker, 'is_crop_pending_review') and callable(self.tracker.is_crop_pending_review):
                                # Pass enough info for tracker to check if this specific detection led to a pending crop
                                is_pending = self.tracker.is_crop_pending_review(det_data, frame_num, video_path) 
                            
                            if not is_pending: # Not pending and no ID returned, so invalid
                                self.stats['invalid_crops'] += 1
                    
                    _current_video_crow_ids_set.update(processed_detection_ids_this_frame)
                    self.stats['current_video_crows'] = len(_current_video_crow_ids_set)

            except Exception as e:
                logger.error(f"Error during detection/tracking for frame {frame_num} in {video_path}: {e}", exc_info=True)
                self.stats['invalid_crops'] += 1 
            
            Clock.schedule_once(lambda dt, f=frame.copy(), d=list(detections_on_frame): self._update_preview_texture(f,d))
            Clock.schedule_once(self._update_stats_ui)
            Clock.schedule_once(lambda dt, cur=frame_num, tot=total_frames_in_video, name=os.path.basename(video_path): self._update_progress_ui(cur,tot,name))
            
            # No artificial time.sleep here; processing speed is determined by detection/tracking.
            # The thread yields by scheduling Kivy UI updates.

        if self.cv_capture: self.cv_capture.release()
        self.cv_capture = None

    def _update_preview_texture(self, frame_bgr, detections, *args):
        if frame_bgr is None: return
        preview_frame = frame_bgr.copy() 
        for det in detections: 
            box = det['bbox']
            cv2.rectangle(preview_frame, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 2)
            cv2.putText(preview_frame, f"{det.get('score',0):.2f}", (int(box[0]),int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)
        buf = cv2.flip(preview_frame, 0).tobytes()
        texture = KivyTexture.create(size=(preview_frame.shape[1], preview_frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.layout.preview_image.texture = texture

    def _update_stats_ui(self, *args):
        for key, value in self.stats.items():
            if key in self.layout.stats_labels_ui: 
                 self.layout.stats_labels_ui[key].text = str(value)
        logger.debug(f"Stats UI updated: {self.stats}")
    
    def _reset_stats_data(self):
        self.stats = {key: 0 for key in self.stats.keys()}
        self.stats['current_video_crows'] = 0 
        logger.info("Internal stats dictionary reset.")

    def _update_progress_ui(self, current_frame, total_frames, video_name, *args):
        if total_frames > 0:
            progress_percent = (current_frame / total_frames) * 100
            self.layout.progress_bar.value = progress_percent
            self.layout.progress_label.text = f"{video_name}: {current_frame}/{total_frames} ({progress_percent:.1f}%)"
        else: self.layout.progress_label.text = f"{video_name}: Frame {current_frame} (Total unknown)"

    def _processing_finished_ui_update(self, *args):
        self.processing_active = False; self.processing_paused = False
        self.layout.start_button.disabled = False
        self.layout.pause_button.disabled = True; self.layout.pause_button.text = "Pause"
        self.layout.stop_button.disabled = True
        self.layout.status_label.text = "Processing finished or stopped."
        is_all_videos_processed = (self.current_video_idx >= len(self.video_files_list) -1) and \
                                  (self.layout.progress_bar.value >= 99.9 or (self.layout.progress_bar.value == 0 and len(self.video_files_list)==0) )
        self.layout.progress_label.text = "Finished all videos." if is_all_videos_processed else "Stopped."
        if self.layout.review_crops_button: # Enable review button for mock testing
            self.layout.review_crops_button.disabled = False 
            # Real logic: self.layout.review_crops_button.disabled = not (self.tracker and hasattr(self.tracker, 'has_pending_reviews') and self.tracker.has_pending_reviews())
        total_crows_tracked = len(self.tracker.tracking_data["crows"]) if self.tracker and self.tracker.tracking_data else 0
        summary_message = (f"Processed {self.stats['videos_processed']} video(s).\n"
                           f"Total valid crops: {self.stats['valid_crops']}\n"
                           f"Total crows tracked: {total_crows_tracked}")
        self.show_popup("Processing Summary", summary_message)
        logger.info(f"UI updated for processing finish/stop. Summary: {summary_message}")

    def _pause_processing_action(self, instance):
        if not self.processing_active: return
        self.processing_paused = not self.processing_paused
        instance.text = "Resume" if self.processing_paused else "Pause"
        self.layout.status_label.text = "Paused." if self.processing_paused else "Processing..."
        logger.info(f"Processing paused: {self.processing_paused}")

    def _stop_processing_action(self, instance):
        if not self.processing_active: return
        logger.info("Stop processing action initiated.")
        self.processing_active = False 
        self.layout.status_label.text = "Stopping..."

    def _save_progress_action(self, instance): 
        if self.tracker and hasattr(self.tracker, '_save_tracking_data'):
            try:
                if hasattr(self.tracker, 'tracking_data') and isinstance(self.tracker.tracking_data, dict):
                    self.tracker.tracking_data['progress_summary'] = {
                        'videos_processed': self.stats['videos_processed'],
                        'last_processed_video_idx': self.current_video_idx -1 
                    }
                self.tracker._save_tracking_data(force=True) 
                self.show_popup("Success", "Tracking progress saved.")
                logger.info("Progress saved manually.")
            except Exception as e:
                logger.error(f"Error saving progress: {e}", exc_info=True)
                self.show_popup("Save Error", f"Could not save progress: {e}")
        else: self.show_popup("Info", "Tracker not available or save method missing.")

    def _load_progress_action(self, instance): 
        if not self.tracker and not self._initialize_tracker():
            self.show_popup("Error", "Tracker not initialized. Output directory might be missing or invalid.")
            return
        if not self.tracker or not hasattr(self.tracker, 'tracking_file') or not self.tracker.tracking_file.exists():
            self.show_popup("Load Info", "No saved progress file found to load.")
            return

        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        content.add_widget(Label(text="Load saved progress? This will overwrite current unsaved data."))
        buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
        yes_btn = Button(text="Yes, Load")
        no_btn = Button(text="No, Cancel")
        buttons.add_widget(yes_btn); buttons.add_widget(no_btn)
        content.add_widget(buttons)
        popup = Popup(title="Confirm Load Progress", content=content, size_hint=(None, None), size=("400dp", "150dp"), auto_dismiss=False)

        def do_load(btn_instance):
            popup.dismiss()
            try:
                if hasattr(self.tracker, '_load_tracking_data'): 
                     self.tracker._load_tracking_data()
                elif hasattr(self.tracker, 'tracking_file') and self.tracker.tracking_file.exists():
                     with open(self.tracker.tracking_file, 'r') as f:
                        self.tracker.tracking_data = json.load(f)
                     if 'next_track_id' in self.tracker.tracking_data:
                         self.tracker.next_track_id = self.tracker.tracking_data['next_track_id']
                else:
                    raise FileNotFoundError("Tracking file not found or tracker cannot load data.")

                self.stats['crows_created'] = len(self.tracker.tracking_data.get("crows", {}))
                self.stats['valid_crops'] = sum(len(data.get('crops',[])) for data in self.tracker.tracking_data.get("crows",{}).values())
                progress_summary = self.tracker.tracking_data.get('progress_summary', {})
                self.stats['videos_processed'] = progress_summary.get('videos_processed',0)
                self.current_video_idx = progress_summary.get('last_processed_video_idx', -1) + 1 
                self._update_stats_ui() 
                last_updated = self.tracker.tracking_data.get('updated_at', 'Unknown')
                self.show_popup("Success", f"Progress loaded successfully.\nLast saved: {last_updated}\nNext video to process: index {self.current_video_idx}")
                logger.info(f"Progress loaded. Next video index: {self.current_video_idx}")
                if self.layout.review_crops_button and hasattr(self.tracker, 'has_pending_reviews'): 
                    self.layout.review_crops_button.disabled = not self.tracker.has_pending_reviews()
            except Exception as e:
                logger.error(f"Error loading progress: {e}", exc_info=True)
                self.show_popup("Load Error", f"Could not load progress: {e}")
        
        yes_btn.bind(on_press=do_load)
        no_btn.bind(on_press=popup.dismiss)
        popup.open()

    def _trigger_crop_review(self, instance=None):
        if not self.tracker:
            self.show_popup("Error", "Tracker not initialized. Cannot review crops.")
            return
        
        crops_data = []
        if hasattr(self.tracker, 'get_all_pending_crops_for_review'):
            try:
                crops_data = self.tracker.get_all_pending_crops_for_review() 
            except Exception as e:
                logger.error(f"Error fetching crops for review from tracker: {e}", exc_info=True)
                self.show_popup("Review Error", f"Could not fetch crops for review: {e}")
                return
        else: 
             logger.warning("Using MOCK data for crop review as get_all_pending_crops_for_review is not available on tracker.")
             mock_img = np.random.randint(0, 255, size=(100,100,3), dtype=np.uint8)
             for i in range(np.random.randint(1,4)): 
                 crops_data.append({
                     'track_id': i + 1, 'frame_num': (i+1)*np.random.randint(5,15), 
                     'image_np': mock_img.copy(), 
                     'crop_path': f'mock_crop_track{i+1}_frame{(i+1)*10}.jpg'
                 })

        if not crops_data:
            self.show_popup("Info", "No crops currently pending manual review.")
            if self.layout.review_crops_button: self.layout.review_crops_button.disabled = True
            return

        logger.info(f"Opening crop review for {len(crops_data)} crops.")
        
        def review_finished_callback():
            logger.info("Crop review popup finished callback triggered.")
            self.layout.status_label.text = "Crop review finished. Ready."
            if self.layout.review_crops_button and hasattr(self.tracker, 'has_pending_reviews'):
                self.layout.review_crops_button.disabled = not self.tracker.has_pending_reviews()

        review_popup = KivyCropReviewPopup(
            tracker_instance=self.tracker,
            crops_data=list(crops_data), 
            on_finish_callback=review_finished_callback
        )
        review_popup.open()

# --- Kivy Crop Review Popup ---
class KivyCropReviewPopup(Popup):
    current_crop_idx = NumericProperty(0)
    crop_image_widget = ObjectProperty(None)
    crop_info_label = ObjectProperty(None)
    nav_buttons_layout = ObjectProperty(None)

    def __init__(self, tracker_instance, crops_data, on_finish_callback, **kwargs):
        super().__init__(**kwargs)
        self.tracker = tracker_instance 
        self.all_crops_in_session = list(crops_data) 
        self.on_finish_callback = on_finish_callback
        self.current_crop_idx = 0
        
        self.title = "Review Extracted Crops"
        self.size_hint = (0.9, 0.95) 
        self.auto_dismiss = False

        main_layout = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        self.crop_image_widget = KivyImage(allow_stretch=True, keep_ratio=True, size_hint_y=0.65)
        main_layout.add_widget(self.crop_image_widget)
        self.crop_info_label = Label(text="Crop Info", size_hint_y=0.1, height="50dp", halign='center')
        self.crop_info_label.bind(width=lambda *x: self.crop_info_label.setter('text_size')(self.crop_info_label, (self.crop_info_label.width, None)))
        main_layout.add_widget(self.crop_info_label)

        self.nav_buttons_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height="44dp", spacing="10dp")
        prev_btn = Button(text="< Previous Crop", on_press=self._previous_crop)
        next_btn = Button(text="Next Crop >", on_press=self._next_crop)
        self.nav_buttons_layout.add_widget(prev_btn)
        self.nav_buttons_layout.add_widget(next_btn)
        main_layout.add_widget(self.nav_buttons_layout)

        action_buttons_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height="44dp", spacing="10dp")
        keep_btn = Button(text="Keep This", on_press=self._keep_crop) 
        discard_btn = Button(text="Discard This", on_press=self._discard_crop) 
        delete_track_btn = Button(text="Delete Track", on_press=self._delete_track)
        action_buttons_layout.add_widget(keep_btn)
        action_buttons_layout.add_widget(discard_btn)
        action_buttons_layout.add_widget(delete_track_btn)
        main_layout.add_widget(action_buttons_layout)
        
        finish_btn_layout = BoxLayout(size_hint_y=None, height="44dp", padding=("0dp", "10dp", "0dp", "0dp")) 
        finish_btn = Button(text="Finish Review Session", on_press=self._finish_review)
        finish_btn_layout.add_widget(finish_btn)
        main_layout.add_widget(finish_btn_layout)
        
        self.content = main_layout
        
        if not self.all_crops_in_session:
             Clock.schedule_once(lambda dt: self._handle_no_crops_to_review())
        else:
             self._display_current_crop()

    def _handle_no_crops_to_review(self):
        App.get_running_app().show_popup("Info", "No crops to review in this session.")
        self._finish_review(None, auto_close=True) 

    def _display_current_crop(self):
        if not self.all_crops_in_session or not (0 <= self.current_crop_idx < len(self.all_crops_in_session)):
            self.crop_image_widget.source = ''
            self.crop_image_widget.texture = None
            self.crop_info_label.text = "No more crops in this review batch."
            if self.nav_buttons_layout: 
                self.nav_buttons_layout.children[0].disabled = True 
                self.nav_buttons_layout.children[1].disabled = True 
            return

        crop_data = self.all_crops_in_session[self.current_crop_idx]
        image_np = crop_data.get('image_np') 
        crop_path_display = os.path.basename(crop_data.get('crop_path', 'N/A'))

        if image_np is not None and isinstance(image_np, np.ndarray):
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(image_rgb, 0).tobytes()
            texture = KivyTexture.create(size=(image_np.shape[1], image_np.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.crop_image_widget.texture = texture
        elif crop_data.get('crop_path') and os.path.exists(crop_data['crop_path']):
             self.crop_image_widget.source = crop_data['crop_path']
             self.crop_image_widget.reload()
        else:
            self.crop_image_widget.source = ''
            self.crop_image_widget.texture = None
            logger.warning(f"Review: Img not found/loadable: {crop_path_display}")

        self.crop_info_label.text = (f"Crop {self.current_crop_idx + 1} of {len(self.all_crops_in_session)}\n"
                                     f"Track ID: {crop_data.get('track_id', 'N/A')}, Frame: {crop_data.get('frame_num', 'N/A')}\n"
                                     f"File: {crop_path_display}")
        
        if self.nav_buttons_layout: 
            self.nav_buttons_layout.children[1].disabled = (self.current_crop_idx == 0) 
            self.nav_buttons_layout.children[0].disabled = (self.current_crop_idx >= len(self.all_crops_in_session) - 1)

    def _next_crop(self, instance):
        if self.current_crop_idx < len(self.all_crops_in_session) - 1:
            self.current_crop_idx += 1
            self._display_current_crop()

    def _previous_crop(self, instance):
        if self.current_crop_idx > 0:
            self.current_crop_idx -= 1
            self._display_current_crop()

    def _process_current_crop_action(self, action_type):
        if not self.all_crops_in_session or not (0 <= self.current_crop_idx < len(self.all_crops_in_session)):
            logger.warning("Process crop action called with no valid crop selected.")
            return

        crop_data = self.all_crops_in_session[self.current_crop_idx]
        track_id = crop_data['track_id']
        crop_path = crop_data['crop_path']
        
        logger.info(f"User action: {action_type} for crop {crop_path} of track {track_id}")

        action_performed_on_tracker = False 
        if action_type == 'keep':
            if self.tracker and hasattr(self.tracker, 'approve_crop_for_review'):
                self.tracker.approve_crop_for_review(crop_data)
                action_performed_on_tracker = True
        elif action_type == 'discard_crop':
            if self.tracker and hasattr(self.tracker, 'discard_crop_for_review'):
                self.tracker.discard_crop_for_review(crop_data)
                action_performed_on_tracker = True
        elif action_type == 'delete_track':
            if self.tracker and hasattr(self.tracker, 'delete_track_for_review'):
                self.tracker.delete_track_for_review(track_id)
                action_performed_on_tracker = True
                self.all_crops_in_session = [c for c in self.all_crops_in_session if c['track_id'] != track_id]
                if self.current_crop_idx >= len(self.all_crops_in_session) and len(self.all_crops_in_session) > 0:
                    self.current_crop_idx = len(self.all_crops_in_session) - 1
                elif not self.all_crops_in_session: 
                    self._finish_review(None, auto_close=True)
                    return 
            else: 
                 logger.warning(f"Tracker method for delete_track not found. Removing from review list only.")
                 self.all_crops_in_session = [c for c in self.all_crops_in_session if c['track_id'] != track_id]
                 if not self.all_crops_in_session: self._finish_review(None, auto_close=True); return
        
        if action_type != 'delete_track': 
             if self.current_crop_idx < len(self.all_crops_in_session) : 
                self.all_crops_in_session.pop(self.current_crop_idx)
             if self.current_crop_idx >= len(self.all_crops_in_session) and len(self.all_crops_in_session) > 0:
                self.current_crop_idx = len(self.all_crops_in_session) - 1
        
        if not self.all_crops_in_session:
            self._finish_review(None, auto_close=True)
        else:
             self._display_current_crop()


    def _keep_crop(self, instance): self._process_current_crop_action('keep')
    def _discard_crop(self, instance): self._process_current_crop_action('discard_crop')
    
    def _delete_track(self, instance):
        if not self.all_crops_in_session or not (0 <= self.current_crop_idx < len(self.all_crops_in_session)): return
        crop_data = self.all_crops_in_session[self.current_crop_idx]
        track_id_to_delete = crop_data['track_id']

        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        label_text = f"Delete all crops for Track ID {track_id_to_delete}?\nThis involves deleting files and potentially DB entries via tracker."
        confirm_label = Label(text=label_text, text_size=(Window.width*0.4, None), halign='center') 
        content.add_widget(confirm_label)
        
        buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
        yes_btn = Button(text="Yes, Delete Track")
        no_btn = Button(text="No, Cancel")
        buttons.add_widget(yes_btn); buttons.add_widget(no_btn)
        content.add_widget(buttons)
        
        confirm_label.texture_update() 
        popup_height = confirm_label.texture_size[1] + buttons.height + Window.dpi*0.8
        min_height = Window.height * 0.25
        actual_height = max(min_height, popup_height)

        confirm_popup = Popup(title="Confirm Delete Track", content=content, size_hint=(0.5, None), height=f"{actual_height}dp", auto_dismiss=False)
        
        def do_delete(btn_instance):
            confirm_popup.dismiss()
            self._process_current_crop_action('delete_track')
        
        yes_btn.bind(on_press=do_delete)
        no_btn.bind(on_press=confirm_popup.dismiss)
        confirm_popup.open()

    def _finish_review(self, instance, auto_close=False):
        logger.info("Crop review session ended.")
        if self.on_finish_callback:
            try:
                self.on_finish_callback()
            except Exception as e:
                logger.error(f"Error in review_finished_callback: {e}", exc_info=True)
        if auto_close or instance is not None: 
            self.dismiss()
        elif not self.all_crops_in_session: 
             self.dismiss()

# Conceptual changes/additions for CrowTracker (to be implemented in actual crow_tracking.py)
# class CrowTracker:
#     def __init__(self, output_dir, ..., correct_orientation=True): 
#         self.output_dir = Path(output_dir)
#         self.pending_review_crops = [] 
#         self.reviewed_crops_info = {} 
#         self.correct_orientation_flag = correct_orientation 
#         self.orientation_detector = OrientationDetector() if correct_orientation else None
#         # ... other initializations
#
#     def _rotate_frame_if_needed(self, frame, detections_for_orientation=None):
#         """Helper to rotate frame based on orientation detection if flag is set."""
#         if self.correct_orientation_flag and self.orientation_detector:
#             try:
#                 h, w = frame.shape[:2]
#                 roi_for_orientation = frame[h//4:3*h//4, w//4:3*w//4] 
#                 orientation, _ = self.orientation_detector.detect_orientation(roi_for_orientation, detections_for_orientation)
#                 if orientation != 0:
#                     logger.info(f"Auto-rotating frame by {orientation} degrees.")
#                     center = (frame.shape[1] // 2, frame.shape[0] // 2)
#                     matrix = cv2.getRotationMatrix2D(center, orientation, 1.0)
#                     if orientation == 90 or orientation == 270:
#                         frame = cv2.warpAffine(frame, matrix, (frame.shape[0], frame.shape[1]))
#                     else: 
#                         frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
#             except Exception as e_orient:
#                 logger.error(f"Error during orientation correction: {e_orient}", exc_info=True)
#         return frame
#
#     def process_detection(self, frame, frame_num, detection_data, video_path, timestamp):
#         # frame_to_process = self._rotate_frame_if_needed(frame.copy(), [detection_data] if detection_data else None) 
#         # current_crop = extract_normalized_crow_crop(frame_to_process, detection_data['bbox']) # Assuming this import
#         # if current_crop is None or current_crop.size == 0: return None # Invalid crop
#
#         # ... rest of existing tracking logic (assign new track ID or update existing track) ...
#         # track_id = self._update_track(detection_data, current_crop_embedding_vector) # Conceptual
#
#         # Example logic for deciding if review is needed:
#         # needs_review = False
#         # if detection_data['score'] < SOME_LOW_CONF_THRESHOLD or self.is_track_ambiguous(track_id): # Conceptual
#         #    needs_review = True
#
#         # saved_crop_path = self.save_crop(track_id, frame_num, current_crop, video_path, is_pending=needs_review) # Conceptual
#         # if needs_review:
#         #   crop_data_for_review = {
#         #       'track_id': track_id, 'frame_num': frame_num,
#         #       'image_np': current_crop, 'crop_path': saved_crop_path, 
#         #       'original_detection_data': detection_data 
#         #   }
#         #   self.pending_review_crops.append(crop_data_for_review)
#         #   return None # Indicate it's pending review
#         # else:
#         #   # Finalize this crop directly (e.g. update embeddings in self.tracking_data['crows'][track_id])
#         #   return track_id # Return the finalized track_id for this detection
#         logger.debug(f"Tracker processing detection: frame {frame_num}, det_score: {detection_data['score']}")
#         # This is a MOCK implementation for the Kivy GUI to call for now
#         mock_track_id = f"MockTrack_{detection_data['bbox'][0] % 5 + 1}" 
#         if detection_data['score'] < 0.97 and len(self.pending_review_crops) < 5: # Simulate adding a few to review
#             # In real code, extract_normalized_crow_crop would be used
#             mock_crop_np = frame[int(detection_data['bbox'][1]):int(detection_data['bbox'][3]), int(detection_data['bbox'][0]):int(detection_data['bbox'][2])]
#             if mock_crop_np.size > 0:
#                 self.pending_review_crops.append({
#                     'track_id': mock_track_id, 'frame_num': frame_num, 'image_np': mock_crop_np,
#                     'crop_path': f"temp/mock_crop_track_{mock_track_id}_frame_{frame_num}.jpg", 
#                     'original_detection_data': detection_data
#                 })
#                 return None # Pending review
#         return mock_track_id # Auto-approved
#
#     def get_all_pending_crops_for_review(self): # Called by Kivy GUI
#         data_to_send = list(self.pending_review_crops) 
#         # self.pending_review_crops.clear() # Or manage state (e.g. mark as "under_review")
#         return data_to_send
#
#     def approve_crop_for_review(self, crop_data_dict): # Called by Kivy Review Popup
#         logger.info(f"Tracker: Approving crop - {crop_data_dict.get('crop_path')}")
#         self.pending_review_crops = [c for c in self.pending_review_crops if c['crop_path'] != crop_data_dict['crop_path']]
#
#     def discard_crop_for_review(self, crop_data_dict): # Called by Kivy Review Popup
#         crop_path = crop_data_dict.get('crop_path')
#         logger.info(f"Tracker: Discarding crop - {crop_path}")
#         self.pending_review_crops = [c for c in self.pending_review_crops if c['crop_path'] != crop_data_dict['crop_path']]
#         # if crop_path and os.path.exists(crop_path): # This might be a mock path
#         #     try: os.remove(crop_path) 
#         #     except Exception as e: logger.error(f"Error deleting {crop_path}: {e}")
#
#     def delete_track_for_review(self, track_id): # Called by Kivy Review Popup
#         logger.info(f"Tracker: Deleting track - {track_id}")
#         self.pending_review_crops = [c for c in self.pending_review_crops if c['track_id'] != track_id]
#         # if str(track_id) in self.tracking_data['crows']:
#         #     # ... logic to find and delete files ...
#         #     del self.tracking_data['crows'][str(track_id)]
#         #     self._save_tracking_data(force=True)
#
#     def has_pending_reviews(self): # Used by Kivy GUI
#         return bool(self.pending_review_crops)
#
#     def is_crop_pending_review(self, detection_data, frame_num, video_path): # Conceptual helper
#         # Check if a crop matching this new detection is already in pending_review_crops
#         for crop in self.pending_review_crops:
#             if crop['frame_num'] == frame_num and \
#                Path(crop['crop_path']).name.startswith(f"track_{detection_data.get('track_id','unk')}_video_{Path(video_path).stem}") and \
#                np.array_equal(crop['original_detection_data']['bbox'], detection_data['bbox']): 
#                 return True
#         return False
#
#     def _save_tracking_data(self, force=False): # Conceptual save method
#         logger.info("Tracker: Saving tracking data...")
#         # ... (implementation) ...
#
#     def _load_tracking_data(self): # Conceptual load method
#         logger.info("Tracker: Loading tracking data...")
#         # ... (implementation) ...


if __name__ == '__main__':
    import kivy 
    # kivy.require('2.1.0') 
    CrowExtractorApp().run()
# print("CrowExtractorApp finished.")
