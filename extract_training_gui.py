# Set up logging first, before any other imports
from logging_config import setup_logging
logger = setup_logging()

import os
import cv2
import numpy as np
from tqdm import tqdm
from detection import detect_crows_parallel as parallel_detect_birds
from tracking import extract_crow_image, load_faster_rcnn
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

# Log startup
logger.info("Starting Crow Training Data Extractor GUI")

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
        crop_resized = cv2.resize(crop, new_size)
        
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
        self.cap = None
        self.current_video = None
        
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
        
        # Control buttons
        control_frame = ttk.Frame(self.left_panel)
        control_frame.grid(row=4, column=0, columnspan=3, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self._start_processing)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self._pause_processing, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="Save Progress", command=self._save_progress)
        self.save_button.grid(row=0, column=3, padx=5)
        
        self.load_button = ttk.Button(control_frame, text="Load Progress", command=self._load_progress)
        self.load_button.grid(row=0, column=4, padx=5)
        
        # Progress tracking
        progress_frame = ttk.LabelFrame(self.left_panel, text="Progress", padding="5")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(self.left_panel, text="Statistics", padding="5")
        stats_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
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
        
        # Set window size
        self.root.geometry("1200x800")
        
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
            self.tracker = CrowTracker(directory)
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
        
        # Initialize tracker with output directory
        self.tracker = CrowTracker(output_dir)
        
        # Enable save button when processing starts
        self.save_button.config(state=tk.NORMAL)
        
        video_files = [f for f in os.listdir(video_dir) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            logger.error("No video files found in selected directory")
            messagebox.showerror("Error", "No video files found in selected directory")
            return
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        # Create processing run directory
        self.run_dir = self.tracker.create_processing_run()
        logger.info(f"Created processing run directory: {self.run_dir}")
        
        self.processing = True
        self.paused = False
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        
        # Reset statistics
        self._reset_stats()
        
        # Start processing the first video
        logger.info("Starting video processing")
        self._process_next_video(video_files, 0)
    
    def _process_next_video(self, video_files, current_index):
        if not self.processing or current_index >= len(video_files):
            self._stop_processing()
            return
        
        video_file = video_files[current_index]
        self.current_video = os.path.join(self.video_dir_var.get(), video_file)
        
        self.status_var.set(f"Processing {video_file}...")
        logger.info(f"Processing video: {video_file}")
        
        self.cap = cv2.VideoCapture(self.current_video)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {self.current_video}")
            self._process_next_video(video_files, current_index + 1)
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_num = 0
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
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"Finished processing video {self.current_video}")
                self.cap.release()
                # Update stats for completed video
                self.stats['videos_processed'] += 1
                self.stats['current_video_detections'] = 0
                self.stats['current_video_crows'].clear()
                self._update_stats()
                self._process_next_video(video_files, current_video_index + 1)
                return
            
            # Update progress
            self.current_frame_num += 1
            self.stats['total_frames'] += 1
            progress = (self.current_frame_num / self.total_frames) * 100
            self.progress_var.set(progress)
            self.progress_label.configure(
                text=f"Processing {os.path.basename(self.current_video)}: "
                     f"{self.current_frame_num}/{self.total_frames} frames"
            )
            
            # Detect crows with timeout
            try:
                detections = parallel_detect_birds(
                    [frame],
                    score_threshold=self.min_confidence_var.get(),
                    multi_view_yolo=self.mv_yolo_var.get(),
                    multi_view_rcnn=self.mv_rcnn_var.get()
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
        preview.thumbnail((800, 600))  # Resize to fit window
        photo = ImageTk.PhotoImage(preview)
        
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # Keep reference
    
    def _pause_processing(self):
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        if not self.paused:
            self._process_frame(self.current_video_files, self.current_video_index)
    
    def _stop_processing(self):
        """Stop processing videos."""
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
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
    
    def _reset_stats(self):
        """Reset all statistics to zero."""
        for key in self.stats:
            if key == 'current_video_crows':
                self.stats[key] = set()
            else:
                self.stats[key] = 0
        self._update_stats()
    
    def _save_progress(self):
        """Manually save the current tracking data."""
        try:
            self.tracker._save_tracking_data(force=True)
            messagebox.showinfo("Save Progress", "Tracking data saved successfully!")
            logger.info("Manual save completed")
        except Exception as e:
            error_msg = f"Error saving tracking data: {str(e)}"
            messagebox.showerror("Save Error", error_msg)
            logger.error(error_msg)

    def _load_progress(self):
        """Load previously saved tracking data."""
        try:
            # Check if tracking file exists
            if not self.tracker.tracking_file.exists():
                messagebox.showinfo("Load Progress", "No saved progress found.")
                return
            
            # Confirm with user
            if not messagebox.askyesno("Load Progress", 
                "This will replace current tracking data with saved data. Continue?"):
                return
            
            # Load the tracking data
            with open(self.tracker.tracking_file, 'r') as f:
                self.tracker.tracking_data = json.load(f)
            
            # Update statistics
            total_crows = len(self.tracker.tracking_data["crows"])
            total_detections = sum(crow["total_detections"] for crow in self.tracker.tracking_data["crows"].values())
            
            # Update stats display
            self.stats_labels['crows_created'].configure(text=str(total_crows))
            self.stats_labels['detections'].configure(text=str(total_detections))
            
            # Show summary
            summary = (
                f"Progress loaded successfully!\n\n"
                f"Total crows: {total_crows}\n"
                f"Total detections: {total_detections}\n"
                f"Last updated: {self.tracker.tracking_data['updated_at']}"
            )
            messagebox.showinfo("Load Progress", summary)
            logger.info(f"Loaded tracking data: {total_crows} crows, {total_detections} detections")
            
        except Exception as e:
            error_msg = f"Error loading tracking data: {str(e)}"
            messagebox.showerror("Load Error", error_msg)
            logger.error(error_msg)

def main():
    root = tk.Tk()
    app = CrowExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 