# Set up logging first, before any other imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import setup_logging
logger = setup_logging()

import os
import cv2
import numpy as np
from tqdm import tqdm
from detection import detect_crows_parallel
from tracking import extract_normalized_crow_crop, load_faster_rcnn
from crow_tracking import CrowTracker
import torch
from pathlib import Path
import shutil
from datetime import datetime
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import json
import queue
from PIL import Image, ImageTk
import time
from collections import defaultdict
from video_orientation import apply_video_orientation, get_video_orientation

# Log startup
logger.info("Starting Crow Training Data Extractor GUI")

def find_video_files(video_dir, recursive=False):
    """
    Find video files in directory, optionally recursively.
    
    Args:
        video_dir: Directory to search for videos
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of video file paths
    """
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

class OrientationDetector:
    def __init__(self):
        # Initialize face detector for orientation detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Initialize bird detector for orientation detection
        self.bird_detector = None  # Will be initialized when needed
    
    def detect_orientation(self, frame, detections=None):
        """
        Automatically detect the correct orientation of a frame.
        Uses multiple heuristics:
        1. Face detection (humans should be upright)
        2. Bird detection confidence (birds should be detected with higher confidence in correct orientation)
        3. Motion vectors (if available)
        4. Edge orientation (horizontal edges should be more common in landscape)
        """
        best_orientation = 0
        best_score = -1
        
        # Try each orientation
        for rotation in [0, 90, 180, 270]:
            score = self._score_orientation(frame, rotation, detections)
            if score > best_score:
                best_score = score
                best_orientation = rotation
        
        # Also check if flipping helps
        for flip in [False, True]:
            flipped = cv2.flip(frame, 1) if flip else frame
            score = self._score_orientation(flipped, best_orientation, detections)
            if score > best_score:
                best_score = score
                return best_orientation, flip
        
        return best_orientation, False
    
    def _score_orientation(self, frame, rotation, detections=None):
        """Score how well a particular orientation works"""
        score = 0
        h, w = frame.shape[:2]
        
        # Rotate frame if needed
        if rotation != 0:
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            frame = cv2.warpAffine(frame, matrix, (w, h))
        
        # 1. Check face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            # Faces should be upright (height > width)
            for (x, y, w, h) in faces:
                if h > w:  # Face is upright
                    score += 2
                else:
                    score -= 1
        
        # 2. Check bird detection confidence if available
        if detections:
            # Birds should be detected with higher confidence in correct orientation
            avg_confidence = np.mean([d['score'] for d in detections])
            score += avg_confidence * 3
        
        # 3. Check edge orientation
        edges = cv2.Canny(gray, 100, 200)
        horizontal_edges = np.sum(edges, axis=1)
        vertical_edges = np.sum(edges, axis=0)
        
        # In landscape orientation, horizontal edges should be more prominent
        if w > h:  # Landscape
            if np.mean(horizontal_edges) > np.mean(vertical_edges):
                score += 1
        else:  # Portrait
            if np.mean(vertical_edges) > np.mean(horizontal_edges):
                score += 1
        
        # 4. Check for sky/ground separation
        # Sky is usually brighter and at the top
        top_half = frame[:h//2]
        bottom_half = frame[h//2:]
        if np.mean(top_half) > np.mean(bottom_half):
            score += 1
        
        return score

class CropReviewWindow:
    def __init__(self, parent, crops, track_id, on_delete):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Review Crops - Track {track_id}")
        self.window.geometry("800x600")
        
        self.crops = crops
        self.on_delete = on_delete
        self.current_idx = 0
        
        # Create GUI
        self.create_gui()
        self.show_current_crop()
    
    def create_gui(self):
        # Image display
        self.canvas = tk.Canvas(self.window, width=400, height=400, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Navigation
        nav_frame = ttk.Frame(self.window)
        nav_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_crop).pack(side=tk.LEFT, padx=5)
        self.crop_label = ttk.Label(nav_frame, text="")
        self.crop_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_crop).pack(side=tk.LEFT, padx=5)
        
        # Actions
        action_frame = ttk.LabelFrame(self.window, text="Actions", padding="5")
        action_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Button(action_frame, text="Delete Track", 
                  command=lambda: self.on_delete(self.crops[0][0])).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Close", 
                  command=self.window.destroy).pack(side=tk.LEFT, padx=5)
    
    def show_current_crop(self):
        frame_num, crop = self.crops[self.current_idx]
        # Resize crop to fit canvas
        h, w = crop.shape[:2]
        scale = min(400/w, 400/h)
        new_size = (int(w*scale), int(h*scale))
        crop_resized = cv2.resize(crop, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to PhotoImage
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        self.canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo
        
        # Update label
        self.crop_label.configure(text=f"Crop {self.current_idx + 1}/{len(self.crops)} (Frame {frame_num})")
    
    def next_crop(self):
        if self.current_idx < len(self.crops) - 1:
            self.current_idx += 1
            self.show_current_crop()
    
    def prev_crop(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_crop()

class CrowExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Crow Training Data Extractor")
        
        # Initialize logger
        logger.info("Starting Crow Training Data Extractor GUI")
        
        # Initialize processing state
        self.processing = False
        self.paused = False
        self.skip_current_video = False  # Flag for skipping current video
        self.cap = None
        self.current_video = None
        
        # Initialize progress tracking for resume functionality
        self.processing_progress = {
            'video_files': [],           # List of all video files to process
            'current_video_index': 0,    # Index of current video being processed
            'current_frame_num': 0,      # Current frame number within video
            'processed_videos': [],      # List of fully processed video paths
            'last_save_time': None,      # When progress was last saved
            'total_videos': 0            # Total number of videos in current session
        }
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left panel for controls
        self.left_panel = ttk.Frame(self.main_frame, padding="5")
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Directory selection
        dir_frame = ttk.LabelFrame(self.left_panel, text="Directories", padding="5")
        dir_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Input directory
        ttk.Label(dir_frame, text="Video Directory:").grid(row=0, column=0, sticky=tk.W)
        self.video_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.video_dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._select_video_dir).grid(row=0, column=2)
        
        # Output directory
        ttk.Label(dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="crow_crops")
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._select_output_dir).grid(row=1, column=2)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(self.left_panel, text="Detection Settings", padding="5")
        settings_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.min_confidence_var = tk.DoubleVar(value=0.2)
        ttk.Label(settings_frame, text="Min Confidence:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(settings_frame, from_=0.1, to=0.9, variable=self.min_confidence_var, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.min_confidence_var).grid(row=0, column=2)
        
        self.min_detections_var = tk.IntVar(value=2)
        ttk.Label(settings_frame, text="Min Detections:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.min_detections_var, 
                   width=5).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Multi-view checkboxes
        self.mv_yolo_var = tk.BooleanVar(value=False)
        self.mv_rcnn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Enable Multi-View for YOLO", variable=self.mv_yolo_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(settings_frame, text="Enable Multi-View for Faster R-CNN", variable=self.mv_rcnn_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Orientation correction checkbox  
        self.orientation_correction_var = tk.BooleanVar(value=False)  # DISABLED by default
        ttk.Checkbutton(settings_frame, text="Auto-correct crow orientation", variable=self.orientation_correction_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Video orientation correction checkbox
        self.video_orientation_var = tk.BooleanVar(value=True)  # ENABLED by default for video orientation
        ttk.Checkbutton(settings_frame, text="Auto-correct video orientation (portrait→landscape)", variable=self.video_orientation_var).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Recursive search checkbox
        self.recursive_search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Search subdirectories recursively", variable=self.recursive_search_var).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # NMS merge threshold
        self.nms_threshold_var = tk.DoubleVar(value=0.3)
        ttk.Label(settings_frame, text="Box Merge Threshold:").grid(row=7, column=0, sticky=tk.W)
        ttk.Scale(settings_frame, from_=0.1, to=0.7, variable=self.nms_threshold_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=7, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.nms_threshold_var).grid(row=7, column=2)
        
        # Tooltip-like label for NMS threshold
        ttk.Label(settings_frame, text="(Lower = merge more boxes)", font=('TkDefaultFont', 8)).grid(row=8, column=0, columnspan=3, sticky=tk.W)
        
        # Bounding box padding slider
        self.bbox_padding_var = tk.DoubleVar(value=0.3)
        ttk.Label(settings_frame, text="BBox Padding:").grid(row=9, column=0, sticky=tk.W)
        ttk.Scale(settings_frame, from_=0.1, to=0.8, variable=self.bbox_padding_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=9, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.bbox_padding_var).grid(row=9, column=2)
        
        # Tooltip-like label for bbox padding
        ttk.Label(settings_frame, text="(Higher = more context around crow)", font=('TkDefaultFont', 8)).grid(row=10, column=0, columnspan=3, sticky=tk.W)
        
        # Audio settings frame
        audio_frame = ttk.LabelFrame(self.left_panel, text="Audio Settings", padding="5")
        audio_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Enable audio extraction
        self.enable_audio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(audio_frame, text="Extract audio segments", variable=self.enable_audio_var).grid(row=0, column=0, sticky=tk.W)
        
        # Audio duration
        ttk.Label(audio_frame, text="Audio duration (seconds):").grid(row=1, column=0, sticky=tk.W)
        self.audio_duration_var = tk.DoubleVar(value=2.0)
        ttk.Scale(audio_frame, from_=0.5, to=5.0, variable=self.audio_duration_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=5)
        ttk.Label(audio_frame, textvariable=self.audio_duration_var).grid(row=1, column=2)
        
        # Control buttons
        control_frame = ttk.Frame(self.left_panel)
        control_frame.grid(row=3, column=0, columnspan=3, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self._start_processing)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self._pause_processing, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.skip_button = ttk.Button(control_frame, text="Skip Video", command=self._skip_current_video, state=tk.DISABLED)
        self.skip_button.grid(row=0, column=2, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="Save Progress", command=self._save_progress)
        self.save_button.grid(row=0, column=4, padx=5)
        
        self.load_button = ttk.Button(control_frame, text="Load Progress", command=self._load_progress)
        self.load_button.grid(row=0, column=5, padx=5)
        
        # Progress tracking
        progress_frame = ttk.LabelFrame(self.left_panel, text="Progress", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(self.left_panel, text="Statistics", padding="5")
        stats_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Initialize statistics
        self.stats = {
            'videos_processed': 0,
            'total_frames': 0,
            'detections': 0,
            'valid_crops': 0,
            'invalid_crops': 0,
            'crows_created': 0,
            'crows_updated': 0,
            'current_video_detections': 0,
            'current_video_crows': set()
        }
        
        # Create statistics labels
        self.stats_labels = {}
        stats = [
            ('videos_processed', 'Videos Processed:'),
            ('total_frames', 'Total Frames:'),
            ('detections', 'Total Detections:'),
            ('valid_crops', 'Valid Crops:'),
            ('invalid_crops', 'Invalid Crops:'),
            ('crows_created', 'New Crows:'),
            ('crows_updated', 'Updated Crows:'),
            ('current_video_detections', 'Current Video Detections:'),
            ('current_video_crows', 'Current Video Crows:')
        ]
        
        for i, (key, label) in enumerate(stats):
            ttk.Label(stats_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=1)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0")
            self.stats_labels[key].grid(row=i, column=1, sticky=tk.W, padx=5, pady=1)
        
        # Right panel - Preview
        preview_frame = ttk.LabelFrame(self.main_frame, text="Preview", padding="5")
        preview_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.left_panel.columnconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Set window size - increased width to accommodate wider preview
        self.root.geometry("1600x800")
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        """Handle window closing."""
        if self.processing:
            if messagebox.askyesno("Quit", "Processing is in progress. Are you sure you want to quit?"):
                self._stop_processing()
                self.root.destroy()
        else:
            self.root.destroy()

    def _select_video_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.video_dir_var.set(directory)
    
    def _select_output_dir(self):
        """Select output directory for crow crops."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
            # Reinitialize tracker with new output directory
            self.tracker = CrowTracker(base_dir=directory)
            logger.info(f"Set output directory to: {directory}")

    def _start_processing(self):
        """Start processing videos."""
        if not self.video_dir_var.get():
            messagebox.showerror("Error", "Please select a video directory")
            return
        
        if not self.output_dir_var.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        video_dir = self.video_dir_var.get()
        output_dir = self.output_dir_var.get()
        logger.info(f"Starting processing in directory: {video_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize tracker with audio settings
        enable_audio = self.enable_audio_var.get()
        audio_duration = self.audio_duration_var.get()
        correct_orientation = self.orientation_correction_var.get()
        bbox_padding = self.bbox_padding_var.get()
        
        self.tracker = CrowTracker(
            base_dir=output_dir, 
            enable_audio_extraction=enable_audio,
            audio_duration=audio_duration,
            correct_orientation=correct_orientation,
            bbox_padding=bbox_padding
        )
        
        logger.info(f"Audio extraction: {'enabled' if enable_audio else 'disabled'}")
        if enable_audio:
            logger.info(f"Audio duration: {audio_duration} seconds")
        logger.info(f"Orientation correction: {'enabled' if correct_orientation else 'disabled'}")
        logger.info(f"BBox padding: {bbox_padding}")
        
        # Enable save button when processing starts
        self.save_button.config(state=tk.NORMAL)
        
        # Check if we're resuming from saved progress
        resume_from_progress = (
            self.processing_progress.get('video_files') and 
            self.processing_progress.get('current_video_index', 0) < len(self.processing_progress['video_files']) and
            self.processing_progress.get('total_videos', 0) > 0
        )
        
        if resume_from_progress:
            # Resume from saved progress
            self.video_files = self.processing_progress['video_files'].copy()
            self.current_video_index = self.processing_progress['current_video_index']
            resume_frame = self.processing_progress['current_frame_num']
            
            logger.info(f"Resuming from saved progress:")
            logger.info(f"  Video {self.current_video_index + 1} of {len(self.video_files)}")
            logger.info(f"  Starting from frame {resume_frame}")
            
            # Verify video files still exist and are accessible
            missing_videos = []
            for video_file in self.video_files:
                if os.path.isabs(video_file):
                    video_path = video_file
                else:
                    video_path = os.path.join(video_dir, video_file)
                
                if not os.path.exists(video_path):
                    missing_videos.append(video_file)
            
            if missing_videos:
                error_msg = f"Cannot resume: {len(missing_videos)} video(s) from saved progress are missing:\n"
                for video in missing_videos[:5]:  # Show first 5
                    error_msg += f"  • {video}\n"
                if len(missing_videos) > 5:
                    error_msg += f"  • ... and {len(missing_videos) - 5} more\n"
                error_msg += "\nPlease ensure all videos are accessible or start fresh processing."
                
                messagebox.showerror("Missing Videos", error_msg)
                return
        
        else:
            # Start fresh - get video files from directory
            recursive_search = self.recursive_search_var.get()
            logger.info(f"Recursive search: {'enabled' if recursive_search else 'disabled'}")
            
            # Find video files using the new function
            video_file_paths = find_video_files(video_dir, recursive=recursive_search)
            
            if not video_file_paths:
                logger.error("No video files found in selected directory")
                messagebox.showerror("Error", "No video files found in selected directory")
                return
            
            # Convert to relative paths for display and processing
            self.video_files = []
            for video_path in video_file_paths:
                if recursive_search:
                    # For recursive search, store the full path
                    self.video_files.append(video_path)
                else:
                    # For non-recursive, store just the filename for compatibility
                    self.video_files.append(os.path.basename(video_path))
            
            logger.info(f"Found {len(self.video_files)} video files to process")
            if recursive_search:
                logger.info("Videos found in:")
                for video_file in video_file_paths:
                    rel_path = os.path.relpath(video_file, video_dir)
                    logger.info(f"  {rel_path}")
            
            # Initialize processing progress for new session
            self.processing_progress = {
                'video_files': self.video_files.copy(),
                'current_video_index': 0,
                'current_frame_num': 0,
                'processed_videos': [],
                'last_save_time': None,
                'total_videos': len(self.video_files)
            }
            
            self.current_video_index = 0
            resume_frame = 0
        
        # Create processing run directory
        self.run_dir = self.tracker.create_processing_run()
        logger.info(f"Created processing run directory: {self.run_dir}")
        
        self.processing = True
        self.paused = False
        self.skip_current_video = False  # Reset skip flag
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.skip_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        # Reset statistics only if starting fresh
        if not resume_from_progress:
            self._reset_stats()
        
        # Start processing from the correct video and frame
        logger.info("Starting video processing")
        self._process_next_video(self.video_files, self.current_video_index, resume_frame)
    
    def _process_next_video(self, video_files, current_index, resume_frame=0):
        if not self.processing or current_index >= len(video_files):
            self._stop_processing()
            return
        
        # Update current video index for pause/resume
        self.current_video_index = current_index
        
        video_file = video_files[current_index]
        
        # Handle both full paths (recursive) and filenames (non-recursive)
        if os.path.isabs(video_file):
            # Full path (recursive search)
            self.current_video = video_file
            display_name = os.path.relpath(video_file, self.video_dir_var.get())
        else:
            # Filename only (non-recursive search)
            self.current_video = os.path.join(self.video_dir_var.get(), video_file)
            display_name = video_file
        
        self.status_var.set(f"Processing {display_name}...")
        logger.info(f"Processing video: {display_name}")
        
        self.cap = cv2.VideoCapture(self.current_video)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {self.current_video}")
            self._process_next_video(video_files, current_index + 1, 0)
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_num = resume_frame
        
        # Skip to resume frame if needed
        if resume_frame > 0:
            logger.info(f"Skipping to frame {resume_frame} for resume")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
        
        self.progress_var.set(0)
        
        # Reset video-specific stats
        self.stats['current_video_detections'] = 0
        self.stats['current_video_crows'].clear()
        self._update_stats()
        
        # Process frames
        self._process_frame(video_files, current_index)
    
    def _process_frame(self, video_files, current_video_index):
        if not self.processing or self.paused:
            return
        
        # Check if we should skip the current video
        if self.skip_current_video:
            logger.info(f"Skipping video: {os.path.basename(self.current_video)}")
            self.skip_current_video = False  # Reset flag
            if self.cap:
                self.cap.release()
            # Mark current video as processed and move to next
            current_video_file = self.video_files[self.current_video_index]
            if current_video_file not in self.processing_progress['processed_videos']:
                self.processing_progress['processed_videos'].append(current_video_file)
                logger.info(f"Marked skipped video as completed: {current_video_file}")
            
            self.stats['videos_processed'] += 1
            self.stats['current_video_detections'] = 0
            self.stats['current_video_crows'].clear()
            self._update_stats()
            self._process_next_video(video_files, current_video_index + 1, 0)
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Finished processing video {self.current_video}")
                self.cap.release()
                # Update stats for completed video
                self.stats['videos_processed'] += 1
                
                # Mark current video as fully processed
                current_video_file = self.video_files[self.current_video_index]
                if current_video_file not in self.processing_progress['processed_videos']:
                    self.processing_progress['processed_videos'].append(current_video_file)
                    logger.info(f"Marked video as completed: {current_video_file}")
                
                self.stats['current_video_detections'] = 0
                self.stats['current_video_crows'].clear()
                self._update_stats()
                self._process_next_video(video_files, current_video_index + 1, 0)
                return
            
            # Apply video orientation correction if enabled
            if self.video_orientation_var.get():
                frame = apply_video_orientation(frame, self.current_video)
            
            # Update progress
            self.current_frame_num += 1
            self.stats['total_frames'] += 1
            progress = (self.current_frame_num / self.total_frames) * 100
            self.progress_var.set(progress)
            self.progress_label.configure(
                text=f"Processing {os.path.basename(self.current_video)}: "
                     f"{self.current_frame_num}/{self.total_frames} frames"
            )
            
            # Update processing progress
            self.processing_progress['current_video_index'] = self.current_video_index
            self.processing_progress['current_frame_num'] = self.current_frame_num
            
            # Save progress periodically (every 100 frames)
            if self.current_frame_num % 100 == 0:
                try:
                    # Quick save without showing dialog
                    progress_data = {
                        'processing_progress': self.processing_progress.copy(),
                        'current_settings': {
                            'video_dir': self.video_dir_var.get(),
                            'output_dir': self.output_dir_var.get(), 
                            'min_confidence': self.min_confidence_var.get(),
                            'min_detections': self.min_detections_var.get(),
                            'mv_yolo': self.mv_yolo_var.get(),
                            'mv_rcnn': self.mv_rcnn_var.get(),
                            'orientation_correction': self.orientation_correction_var.get(),
                            'video_orientation': self.video_orientation_var.get(),
                            'recursive_search': self.recursive_search_var.get(),
                            'nms_threshold': self.nms_threshold_var.get(),
                            'bbox_padding': self.bbox_padding_var.get(),
                            'enable_audio': self.enable_audio_var.get(),
                            'audio_duration': self.audio_duration_var.get()
                        },
                        'session_stats': self._get_serializable_stats(),
                        'saved_at': datetime.now().isoformat()
                    }
                    
                    progress_data['processing_progress']['last_save_time'] = datetime.now().isoformat()
                    
                    # Save to processing progress file
                    progress_file = self.tracker.metadata_dir / "processing_progress.json"
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f, indent=2)
                    
                    logger.debug(f"Auto-saved progress at frame {self.current_frame_num}")
                    
                except Exception as e:
                    logger.warning(f"Failed to auto-save progress: {e}")
            
            # Detect crows with timeout
            try:
                detections = detect_crows_parallel(
                    [frame],
                    score_threshold=self.min_confidence_var.get(),
                    multi_view_yolo=self.mv_yolo_var.get(),
                    multi_view_rcnn=self.mv_rcnn_var.get(),
                    nms_threshold=self.nms_threshold_var.get()
                )
                frame_dets = detections[0]
                logger.debug(f"Frame {self.current_frame_num}: Found {len(frame_dets)} detections")
                
                self.stats['detections'] += len(frame_dets)
                self.stats['current_video_detections'] += len(frame_dets)
                
                # Process each detection
                for det in frame_dets:
                    if det['score'] < self.min_confidence_var.get():
                        continue
                    
                    try:
                        # Process detection and get crow_id
                        crow_id = self.tracker.process_detection(
                            frame, 
                            self.current_frame_num,
                            det,
                            self.current_video,
                            self.current_frame_num / self.fps if self.fps > 0 else None
                        )
                        
                        if crow_id:
                            logger.debug(f"Frame {self.current_frame_num}: Processed detection as {crow_id}")
                            self.stats['valid_crops'] += 1
                            self.stats['current_video_crows'].add(crow_id)
                            if crow_id not in self.tracker.tracking_data["crows"]:
                                self.stats['crows_created'] += 1
                                logger.info(f"Created new crow: {crow_id}")
                            else:
                                self.stats['crows_updated'] += 1
                                logger.debug(f"Updated existing crow: {crow_id}")
                        else:
                            self.stats['invalid_crops'] += 1
                            logger.debug(f"Frame {self.current_frame_num}: Invalid crop")
                    except Exception as e:
                        logger.error(f"Error processing detection: {str(e)}", exc_info=True)
                        self.stats['invalid_crops'] += 1
                
                # Update preview and stats
                self._update_preview(frame, frame_dets)
                self._update_stats()
                
            except TimeoutError:
                logger.error(f"Frame {self.current_frame_num}: Detection timed out")
                self.stats['invalid_crops'] += 1
                self._update_stats()
            except Exception as e:
                logger.error(f"Error detecting crows in frame {self.current_frame_num}: {str(e)}", exc_info=True)
                self.stats['invalid_crops'] += 1
                self._update_stats()
            
            # Schedule next frame with a more reasonable interval (100ms)
            # This gives more time for processing and UI updates
            self.root.after(100, lambda: self._process_frame(video_files, current_video_index))
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)
            self._stop_processing()
            messagebox.showerror("Error", f"Error processing frame: {str(e)}")
    
    def _update_preview(self, frame, detections):
        # Draw detections
        preview = frame.copy()
        for det in detections:
            if det['score'] < self.min_confidence_var.get():
                continue
            box = det['bbox']
            cv2.rectangle(preview, 
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0), 2)
            cv2.putText(preview,
                       f"{det['score']:.2f}",
                       (int(box[0]), int(box[1] - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
        
        # Convert to PhotoImage
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(preview)
        preview.thumbnail((1000, 700))  # Resize to fit wider window
        photo = ImageTk.PhotoImage(preview)
        
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # Keep reference
    
    def _pause_processing(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        if not self.paused:
            # Resume processing from where we left off
            self._process_frame(self.video_files, self.current_video_index)
    
    def _skip_current_video(self):
        """Skip the current video and move to the next one."""
        if self.processing and not self.paused:
            self.skip_current_video = True
            logger.info(f"Skipping current video: {os.path.basename(self.current_video) if self.current_video else 'Unknown'}")
            self.status_var.set("Skipping to next video...")
    
    def _stop_processing(self):
        """Stop processing videos."""
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.skip_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        # Keep save button enabled after stopping
        self.save_button.config(state=tk.NORMAL)
        
        logger.info("Stopping processing")
        if self.cap:
            self.cap.release()
        
        # Clean up processing directory
        if hasattr(self, 'run_dir'):
            logger.info(f"Cleaning up processing directory: {self.run_dir}")
            self.tracker.cleanup_processing_dir(self.run_dir)
        
        # Update UI
        self.status_var.set("Ready")
        
        # Show summary
        total_crows = len(self.tracker.list_crows())
        summary = (
            f"Processing complete!\n\n"
            f"Total crows in database: {total_crows}\n"
            f"Videos processed: {self.stats['videos_processed']}\n"
            f"Total frames: {self.stats['total_frames']}\n"
            f"Total detections: {self.stats['detections']}\n"
            f"Valid crops: {self.stats['valid_crops']}\n"
            f"New crows: {self.stats['crows_created']}\n"
            f"Updated crows: {self.stats['crows_updated']}"
        )
        logger.info("Processing summary:\n" + summary)
        messagebox.showinfo("Processing Complete", summary)
    
    def _update_stats(self):
        """Update statistics display"""
        for key, label in self.stats_labels.items():
            if key == 'current_video_crows':
                value = len(self.stats[key])
            else:
                value = self.stats[key]
            label.configure(text=str(value))
    
    def _get_serializable_stats(self):
        """Get a JSON-serializable copy of stats."""
        serializable_stats = self.stats.copy()
        if 'current_video_crows' in serializable_stats:
            serializable_stats['current_video_crows'] = list(serializable_stats['current_video_crows'])
        return serializable_stats
    
    def _reset_stats(self):
        """Reset all statistics to zero."""
        for key in self.stats:
            if key == 'current_video_crows':
                self.stats[key] = set()
            else:
                self.stats[key] = 0
        self._update_stats()
    
    def _save_progress(self):
        """Manually save the current tracking data and processing progress."""
        try:
            # Save crow tracking data
            self.tracker._save_tracking_data(force=True)
            
            # Save processing progress
            progress_data = {
                'processing_progress': self.processing_progress.copy(),
                'current_settings': {
                    'video_dir': self.video_dir_var.get(),
                    'output_dir': self.output_dir_var.get(), 
                    'min_confidence': self.min_confidence_var.get(),
                    'min_detections': self.min_detections_var.get(),
                    'mv_yolo': self.mv_yolo_var.get(),
                    'mv_rcnn': self.mv_rcnn_var.get(),
                    'orientation_correction': self.orientation_correction_var.get(),
                    'video_orientation': self.video_orientation_var.get(),
                    'recursive_search': self.recursive_search_var.get(),
                    'nms_threshold': self.nms_threshold_var.get(),
                    'bbox_padding': self.bbox_padding_var.get(),
                    'enable_audio': self.enable_audio_var.get(),
                    'audio_duration': self.audio_duration_var.get()
                },
                'session_stats': self._get_serializable_stats(),
                'saved_at': datetime.now().isoformat()
            }
            
            # Update processing progress with current state if processing
            if self.processing and hasattr(self, 'video_files'):
                progress_data['processing_progress']['video_files'] = self.video_files.copy()
                progress_data['processing_progress']['current_video_index'] = getattr(self, 'current_video_index', 0)
                progress_data['processing_progress']['current_frame_num'] = getattr(self, 'current_frame_num', 0)
                progress_data['processing_progress']['total_videos'] = len(self.video_files)
                
            progress_data['processing_progress']['last_save_time'] = datetime.now().isoformat()
            
            # Save to processing progress file
            progress_file = self.tracker.metadata_dir / "processing_progress.json"
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            messagebox.showinfo("Save Progress", 
                f"Progress saved successfully!\n\n"
                f"Crow tracking data: {len(self.tracker.tracking_data['crows'])} crows\n"
                f"Processing progress: Video {getattr(self, 'current_video_index', 0) + 1} of {len(getattr(self, 'video_files', []))}\n"
                f"Current frame: {getattr(self, 'current_frame_num', 0)}")
            
            logger.info("Manual save completed - both tracking data and processing progress saved")
            
        except Exception as e:
            error_msg = f"Error saving progress: {str(e)}"
            messagebox.showerror("Save Error", error_msg)
            logger.error(error_msg)

    def _load_progress(self):
        """Load previously saved tracking data and processing progress."""
        try:
            # Initialize tracker if it doesn't exist
            if not hasattr(self, 'tracker'):
                if not self.output_dir_var.get():
                    messagebox.showerror("Error", "Please select an output directory first")
                    return
                
                # Initialize tracker with current settings
                enable_audio = self.enable_audio_var.get()
                audio_duration = self.audio_duration_var.get()
                correct_orientation = self.orientation_correction_var.get()
                bbox_padding = self.bbox_padding_var.get()
                
                self.tracker = CrowTracker(
                    base_dir=self.output_dir_var.get(), 
                    enable_audio_extraction=enable_audio,
                    audio_duration=audio_duration,
                    correct_orientation=correct_orientation,
                    bbox_padding=bbox_padding
                )
            
            # Check if tracking file exists
            if not self.tracker.tracking_file.exists():
                messagebox.showinfo("Load Progress", "No saved progress found.")
                return
            
            # Load processing progress if available
            progress_file = self.tracker.metadata_dir / "processing_progress.json"
            has_processing_progress = progress_file.exists()
            
            if has_processing_progress:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                
                saved_processing_progress = progress_data.get('processing_progress', {})
                saved_settings = progress_data.get('current_settings', {})
                saved_stats = progress_data.get('session_stats', {})
                
                # Show detailed progress info and ask user what to do
                current_video_idx = saved_processing_progress.get('current_video_index', 0)
                current_frame = saved_processing_progress.get('current_frame_num', 0) 
                total_videos = saved_processing_progress.get('total_videos', 0)
                processed_videos = len(saved_processing_progress.get('processed_videos', []))
                
                progress_info = (
                    f"Found saved processing progress!\n\n"
                    f"Last session:\n"
                    f"• Total videos: {total_videos}\n"
                    f"• Fully processed: {processed_videos}\n"
                    f"• Was processing video {current_video_idx + 1} of {total_videos}\n"
                    f"• At frame {current_frame}\n"
                    f"• Last saved: {progress_data.get('saved_at', 'Unknown')}\n\n"
                    f"Do you want to:\n"
                    f"• Resume from where you left off, or\n"
                    f"• Load crow data but start processing fresh?"
                )
                
                result = messagebox.askyesnocancel("Resume Processing?", 
                    progress_info + "\n\nYes = Resume, No = Start Fresh, Cancel = Cancel Load")
                
                if result is None:  # Cancel
                    return
                elif result:  # Yes - Resume
                    # Load complete processing state
                    self.processing_progress = saved_processing_progress.copy()
                    
                    # Apply saved settings to GUI
                    if saved_settings:
                        self.video_dir_var.set(saved_settings.get('video_dir', ''))
                        self.min_confidence_var.set(saved_settings.get('min_confidence', 0.2))
                        self.min_detections_var.set(saved_settings.get('min_detections', 2))
                        self.mv_yolo_var.set(saved_settings.get('mv_yolo', False))
                        self.mv_rcnn_var.set(saved_settings.get('mv_rcnn', False))
                        self.orientation_correction_var.set(saved_settings.get('orientation_correction', False))
                        self.video_orientation_var.set(saved_settings.get('video_orientation', True))
                        self.recursive_search_var.set(saved_settings.get('recursive_search', False))
                        self.nms_threshold_var.set(saved_settings.get('nms_threshold', 0.3))
                        self.bbox_padding_var.set(saved_settings.get('bbox_padding', 0.3))
                        self.enable_audio_var.set(saved_settings.get('enable_audio', True))
                        self.audio_duration_var.set(saved_settings.get('audio_duration', 2.0))
                    
                    # Restore session stats
                    if saved_stats:
                        # Convert sets back from lists for current_video_crows
                        if 'current_video_crows' in saved_stats and isinstance(saved_stats['current_video_crows'], list):
                            saved_stats['current_video_crows'] = set(saved_stats['current_video_crows'])
                        self.stats.update(saved_stats)
                    
                    resume_msg = (
                        f"Progress loaded and ready to resume!\n\n"
                        f"Will continue from:\n"
                        f"• Video {current_video_idx + 1} of {total_videos}\n"
                        f"• Frame {current_frame}\n\n"
                        f"Click 'Start Processing' to resume from this point."
                    )
                    messagebox.showinfo("Resume Ready", resume_msg)
                    logger.info(f"Loaded progress for resume: video {current_video_idx + 1}, frame {current_frame}")
                    
                else:  # No - Start Fresh (just load crow data)
                    has_processing_progress = False
            
            # Always load crow tracking data
            with open(self.tracker.tracking_file, 'r') as f:
                self.tracker.tracking_data = json.load(f)
            
            # Update statistics from loaded crow data
            total_crows = len(self.tracker.tracking_data["crows"])
            total_detections = sum(crow["total_detections"] for crow in self.tracker.tracking_data["crows"].values())
            
            if not has_processing_progress:
                # Just update crow-related stats
                self.stats['crows_created'] = total_crows
                self.stats['detections'] = total_detections
                self.stats['valid_crops'] = total_detections
            
            self._update_stats()
            
            if not has_processing_progress:
                # Show summary for fresh start
                summary = (
                    f"Crow data loaded successfully!\n\n"
                    f"Total crows: {total_crows}\n"
                    f"Total detections: {total_detections}\n"
                    f"Last updated: {self.tracker.tracking_data.get('updated_at', 'Unknown')}\n\n"
                    f"You can now start processing videos. Existing crows will be recognized."
                )
                messagebox.showinfo("Load Progress", summary)
                logger.info(f"Loaded crow tracking data: {total_crows} crows, {total_detections} detections")
            
        except Exception as e:
            error_msg = f"Error loading progress: {str(e)}"
            messagebox.showerror("Load Error", error_msg)
            logger.error(error_msg)

def main():
    root = tk.Tk()
    app = CrowExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 