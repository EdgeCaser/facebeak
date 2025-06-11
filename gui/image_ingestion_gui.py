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

# Log startup
logger.info("Starting Crow Image Ingestion GUI")

def find_image_files(image_dir, recursive=False):
    """
    Find image files in directory, optionally recursively.
    
    Args:
        image_dir: Directory to search for images
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of image file paths
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
                       '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.TIF')
    image_files = []
    
    if recursive:
        # Use os.walk for recursive search
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        # Non-recursive search
        try:
            for file in os.listdir(image_dir):
                if file.endswith(image_extensions):
                    image_files.append(os.path.join(image_dir, file))
        except OSError as e:
            logger.error(f"Error reading directory {image_dir}: {e}")
            return []
    
    return sorted(image_files)

class CropReviewWindow:
    def __init__(self, parent, image_path, detections, on_approve, on_reject, on_skip):
        self.window = tk.Toplevel(parent)
        self.window.title(f"Review Detections - {os.path.basename(image_path)}")
        self.window.geometry("1200x800")
        
        self.image_path = image_path
        self.detections = detections
        self.on_approve = on_approve
        self.on_reject = on_reject
        self.on_skip = on_skip
        self.current_idx = 0
        self.labels = {}  # Store labels for each detection
        
        # Manual drawing variables
        self.drawing_mode = False
        self.start_x = None
        self.start_y = None
        self.current_bbox = None
        self.manual_detections = []  # Store manually drawn detections
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not load image: {image_path}")
            self.window.destroy()
            return
        
        # Create GUI
        self.create_gui()
        self.show_current_detection()
    
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image display
        self.canvas = tk.Canvas(main_frame, width=800, height=600, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)
        
        # Bind mouse events for manual drawing
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_detection).pack(side=tk.LEFT, padx=5)
        self.detection_label = ttk.Label(nav_frame, text="")
        self.detection_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_detection).pack(side=tk.LEFT, padx=5)
        
        # Manual drawing controls
        drawing_frame = ttk.LabelFrame(main_frame, text="Manual Drawing", padding="5")
        drawing_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        self.drawing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(drawing_frame, text="Enable Manual Drawing", 
                       variable=self.drawing_var, 
                       command=self.toggle_drawing_mode).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(drawing_frame, text="Click and drag to draw square bounding boxes").pack(side=tk.LEFT, padx=10)
        
        ttk.Button(drawing_frame, text="Clear Manual Boxes", 
                  command=self.clear_manual_detections).pack(side=tk.RIGHT, padx=5)
        
        # Labeling frame
        label_frame = ttk.LabelFrame(main_frame, text="Labeling", padding="5")
        label_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        
        # Label dropdown
        ttk.Label(label_frame, text="Label:").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        # Canonical label set
        self.predefined_labels = [
            "crow",
            "not_a_crow",
            "not_sure",
            "multi_crow"
        ]
        self.label_var = tk.StringVar()
        self.label_dropdown = ttk.Combobox(label_frame, textvariable=self.label_var, 
                                          values=self.predefined_labels, 
                                          state="readonly", width=20)
        self.label_dropdown.grid(row=0, column=1, padx=5)
        self.label_dropdown.set("crow")  # Default selection
        
        # Confidence display
        self.confidence_label = ttk.Label(label_frame, text="")
        self.confidence_label.grid(row=0, column=2, padx=20)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, columnspan=3, pady=5)
        
        ttk.Button(action_frame, text="Approve & Save", 
                  command=self.approve_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Reject", 
                  command=self.reject_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Skip Image", 
                  command=self.skip_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Close", 
                  command=self.window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def toggle_drawing_mode(self):
        """Toggle manual drawing mode on/off."""
        self.drawing_mode = self.drawing_var.get()
        if self.drawing_mode:
            self.canvas.config(cursor="crosshair")
        else:
            self.canvas.config(cursor="")
    
    def on_mouse_down(self, event):
        """Handle mouse button press for drawing."""
        if not self.drawing_mode:
            return
        
        self.start_x = event.x
        self.start_y = event.y
        self.current_bbox = None
    
    def on_mouse_drag(self, event):
        """Handle mouse drag for drawing square bounding box."""
        if not self.drawing_mode or self.start_x is None:
            return
        self.canvas.delete("temp_rect")
        width = abs(event.x - self.start_x)
        height = abs(event.y - self.start_y)
        size = max(width, height)
        if event.x >= self.start_x:
            x2 = self.start_x + size
        else:
            x2 = self.start_x - size
        if event.y >= self.start_y:
            y2 = self.start_y + size
        else:
            y2 = self.start_y - size
        # Store the square coordinates for use in on_mouse_up
        self._square_coords = (self.start_x, self.start_y, x2, y2)
        self.canvas.create_rectangle(
            self.start_x, self.start_y, x2, y2,
            outline="red", width=2, tags="temp_rect"
        )
    
    def on_mouse_up(self, event):
        """Handle mouse button release to finalize square bounding box."""
        if not self.drawing_mode or self.start_x is None:
            return
            
        # Use the square coordinates calculated during drag
        if hasattr(self, '_square_coords'):
            x1, y1, x2, y2 = self._square_coords
            # Ensure proper order (but maintain square shape)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
        else:
            # Fallback: calculate square from current mouse position
            width = abs(event.x - self.start_x)
            height = abs(event.y - self.start_y)
            size = max(width, height)
            
            if event.x >= self.start_x:
                x2 = self.start_x + size
            else:
                x2 = self.start_x - size
            if event.y >= self.start_y:
                y2 = self.start_y + size
            else:
                y2 = self.start_y - size
                
            x1, y1, x2, y2 = self.start_x, self.start_y, x2, y2
            # Ensure proper order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
        
        # Verify it's a square and has minimum size
        width = x2 - x1
        height = y2 - y1
        if width == height and width > 20:  # Must be square and have minimum size
            img_bbox = self.canvas_to_image_coords(x1, y1, x2, y2)
            manual_detection = {
                'bbox': img_bbox,
                'score': 1.0,
                'manual': True
            }
            self.manual_detections.append(manual_detection)
            self.show_current_detection()
            
        self.canvas.delete("temp_rect")
        self.start_x = None
        self.start_y = None
        if hasattr(self, '_square_coords'):
            del self._square_coords
    
    def canvas_to_image_coords(self, x1, y1, x2, y2):
        """Convert canvas coordinates to original image coordinates."""
        if not hasattr(self, 'canvas_offset_x'):
            # Fallback to old method if offsets not set
            h, w = self.original_image.shape[:2]
            canvas_w = 800
            canvas_h = 600
            
            # Calculate scale factors
            scale_x = w / canvas_w
            scale_y = h / canvas_h
            
            # Use the same scale factor to preserve square shapes
            scale = min(scale_x, scale_y)
            
            # Convert coordinates using the same scale
            img_x1 = int(x1 * scale)
            img_y1 = int(y1 * scale)
            img_x2 = int(x2 * scale)
            img_y2 = int(y2 * scale)
            
            # Clamp to image boundaries
            img_x1 = max(0, min(img_x1, w))
            img_y1 = max(0, min(img_y1, h))
            img_x2 = max(0, min(img_x2, w))
            img_y2 = max(0, min(img_y2, h))
            
            return [img_x1, img_y1, img_x2, img_y2]
        
        # Account for image centering offset
        x1_adjusted = x1 - self.canvas_offset_x
        y1_adjusted = y1 - self.canvas_offset_y
        x2_adjusted = x2 - self.canvas_offset_x
        y2_adjusted = y2 - self.canvas_offset_y
        
        # Calculate the original square size before any clamping
        original_width = x2_adjusted - x1_adjusted
        original_height = y2_adjusted - y1_adjusted
        original_size = max(original_width, original_height)
        
        # Clamp to image boundaries on canvas while preserving square shape
        canvas_img_w, canvas_img_h = self.canvas_image_size
        
        # Ensure the square fits within the canvas image bounds
        x1_adjusted = max(0, min(x1_adjusted, canvas_img_w - original_size))
        y1_adjusted = max(0, min(y1_adjusted, canvas_img_h - original_size))
        x2_adjusted = x1_adjusted + original_size
        y2_adjusted = y1_adjusted + original_size
        
        # Final clamp to ensure we don't exceed bounds
        x1_adjusted = max(0, min(x1_adjusted, canvas_img_w))
        y1_adjusted = max(0, min(y1_adjusted, canvas_img_h))
        x2_adjusted = max(0, min(x2_adjusted, canvas_img_w))
        y2_adjusted = max(0, min(y2_adjusted, canvas_img_h))
        
        # Convert from canvas image coordinates to original image coordinates
        orig_img_w, orig_img_h = self.original_image_size
        
        scale_x = orig_img_w / canvas_img_w
        scale_y = orig_img_h / canvas_img_h
        
        # Use the same scale factor to preserve square shapes
        scale = min(scale_x, scale_y)
        
        img_x1 = int(x1_adjusted * scale)
        img_y1 = int(y1_adjusted * scale)
        img_x2 = int(x2_adjusted * scale)
        img_y2 = int(y2_adjusted * scale)
        
        # Clamp to original image boundaries while preserving square shape
        img_size = img_x2 - img_x1
        
        # Ensure the square fits within the original image bounds
        img_x1 = max(0, min(img_x1, orig_img_w - img_size))
        img_y1 = max(0, min(img_y1, orig_img_h - img_size))
        img_x2 = img_x1 + img_size
        img_y2 = img_y1 + img_size
        
        # Final clamp to ensure we don't exceed bounds
        img_x1 = max(0, min(img_x1, orig_img_w))
        img_y1 = max(0, min(img_y1, orig_img_h))
        img_x2 = max(0, min(img_x2, orig_img_w))
        img_y2 = max(0, min(img_y2, orig_img_h))
        
        return [img_x1, img_y1, img_x2, img_y2]
    
    def clear_manual_detections(self):
        """Clear all manually drawn detections."""
        self.manual_detections.clear()
        self.show_current_detection()
    
    def _get_combined_detections(self):
        """Get combined detections with manual detections overriding overlapping automatic ones."""
        all_detections = []
        
        # Start with automatic detections
        for det in self.detections:
            all_detections.append(det.copy())
        
        # Add manual detections, replacing any overlapping automatic ones
        for manual_det in self.manual_detections:
            manual_bbox = manual_det['bbox']
            manual_center_x = (manual_bbox[0] + manual_bbox[2]) / 2
            manual_center_y = (manual_bbox[1] + manual_bbox[3]) / 2
            manual_size = max(manual_bbox[2] - manual_bbox[0], manual_bbox[3] - manual_bbox[1])
            
            # Check if this manual detection overlaps with any automatic detection
            overlapping_auto_idx = None
            for i, auto_det in enumerate(all_detections):
                if auto_det.get('manual', False):  # Skip if already a manual detection
                    continue
                    
                auto_bbox = auto_det['bbox']
                auto_center_x = (auto_bbox[0] + auto_bbox[2]) / 2
                auto_center_y = (auto_bbox[1] + auto_bbox[3]) / 2
                auto_size = max(auto_bbox[2] - auto_bbox[0], auto_bbox[3] - auto_bbox[1])
                
                # Calculate distance between centers
                distance = ((manual_center_x - auto_center_x) ** 2 + (manual_center_y - auto_center_y) ** 2) ** 0.5
                
                # Consider overlapping if centers are close relative to box sizes
                overlap_threshold = min(manual_size, auto_size) * 0.5
                if distance < overlap_threshold:
                    overlapping_auto_idx = i
                    break
            
            # Replace overlapping automatic detection or add new manual detection
            if overlapping_auto_idx is not None:
                all_detections[overlapping_auto_idx] = manual_det
            else:
                all_detections.append(manual_det)
        
        return all_detections

    def show_current_detection(self):
        """Show current detection with all manual detections."""
        # Get combined detections with smart override logic
        all_detections = self._get_combined_detections()
        
        if not all_detections:
            # No detections, just show the image
            self._display_image(self.original_image.copy())
            self.detection_label.configure(text="No detections - Draw manually or skip")
            self.confidence_label.configure(text="")
            return
        
        if self.current_idx >= len(all_detections):
            self.current_idx = len(all_detections) - 1
        
        detection = all_detections[self.current_idx]
        box = detection['bbox']
        score = detection['score']
        is_manual = detection.get('manual', False)
        
        # Draw detection on image
        preview = self.original_image.copy()
        
        # Draw all detections
        for i, det in enumerate(all_detections):
            det_box = det['bbox']
            det_score = det['score']
            det_manual = det.get('manual', False)
            
            # Choose color based on type
            if det_manual:
                color = (0, 0, 255)  # Red for manual
                label = f"Manual {i + 1}"
            else:
                color = (0, 255, 0)  # Green for automatic
                label = f"Auto {i + 1}"
            
            # Draw box
            cv2.rectangle(preview, 
                        (int(det_box[0]), int(det_box[1])),
                        (int(det_box[2]), int(det_box[3])),
                        color, 3)
            
            # Draw label
            cv2.putText(preview,
                       label,
                       (int(det_box[0]), int(det_box[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2)
        
        # Highlight current detection
        current_box = detection['bbox']
        highlight_color = (255, 255, 0)  # Yellow highlight
        cv2.rectangle(preview, 
                    (int(current_box[0]), int(current_box[1])),
                    (int(current_box[2]), int(current_box[3])),
                    highlight_color, 5)
        
        self._display_image(preview)
        
        # Update labels
        detection_type = "Manual" if is_manual else "Automatic"
        self.detection_label.configure(
            text=f"{detection_type} Detection {self.current_idx + 1}/{len(all_detections)}")
        self.confidence_label.configure(
            text=f"Confidence: {score:.3f}")
        
        # Load existing label if any
        if self.current_idx in self.labels:
            saved_label = self.labels[self.current_idx]
            if saved_label in self.predefined_labels:
                self.label_var.set(saved_label)
            else:
                self.label_var.set("crow")  # Default if saved label is not in predefined list
        else:
            self.label_var.set("crow")  # Default selection
    
    def _display_image(self, image):
        """Display image on canvas."""
        # Resize to fit canvas
        h, w = image.shape[:2]
        scale = min(800/w, 600/h)
        new_size = (int(w*scale), int(h*scale))
        image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate centering offset
        canvas_w, canvas_h = 800, 600
        img_w, img_h = image_resized.shape[1], image_resized.shape[0]
        offset_x = (canvas_w - img_w) // 2
        offset_y = (canvas_h - img_h) // 2
        
        # Store the offset for coordinate conversion
        self.canvas_offset_x = offset_x
        self.canvas_offset_y = offset_y
        self.canvas_image_size = (img_w, img_h)
        self.original_image_size = (w, h)
        
        # Convert to PhotoImage
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo
    
    def next_detection(self):
        # Save current label
        current_label = self.label_var.get()
        if current_label and current_label in self.predefined_labels:
            self.labels[self.current_idx] = current_label
        
        all_detections = self._get_combined_detections()
        if self.current_idx < len(all_detections) - 1:
            self.current_idx += 1
            self.show_current_detection()
    
    def prev_detection(self):
        # Save current label
        current_label = self.label_var.get()
        if current_label and current_label in self.predefined_labels:
            self.labels[self.current_idx] = current_label
        
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_detection()
    
    def approve_detection(self):
        # Save current label
        current_label = self.label_var.get()
        if current_label and current_label in self.predefined_labels:
            self.labels[self.current_idx] = current_label
        
        # Get all detections
        all_detections = self._get_combined_detections()
        
        # Call approve callback with current detection and label
        detection = all_detections[self.current_idx]
        label = self.labels.get(self.current_idx, "crow")  # Default to "crow" if no label
        self.on_approve(detection, label)
        
        # Move to next detection or close if last
        if self.current_idx < len(all_detections) - 1:
            self.current_idx += 1
            self.show_current_detection()
        else:
            # Don't destroy window here - let the ReviewSession handle it
            # The ReviewSession will call on_finish and then destroy the window
            pass
    
    def reject_detection(self):
        # Get all detections
        all_detections = self._get_combined_detections()
        
        # Call reject callback with current detection
        detection = all_detections[self.current_idx]
        self.on_reject(detection)
        
        # Move to next detection or close if last
        if self.current_idx < len(all_detections) - 1:
            self.current_idx += 1
            self.show_current_detection()
        else:
            # Don't destroy window here - let the ReviewSession handle it
            pass
    
    def skip_image(self):
        # Call skip callback
        self.on_skip()
        self.window.destroy()

class ReviewSession:
    def __init__(self, parent, image_batches, on_finish):
        self.parent = parent
        self.image_batches = image_batches  # List of (image_path, detections, image)
        self.on_finish = on_finish
        self.current_image_idx = 0
        self.current_detection_idx = 0
        self.review_window = None
        self.labels = {}  # {(image_idx, det_idx): label}
        self._open_review_window()

    def _open_review_window(self):
        image_path, detections, image = self.image_batches[self.current_image_idx]
        def on_approve(detection, label):
            self.labels[(self.current_image_idx, self.current_detection_idx)] = label
            self._next_detection()
        def on_reject(detection):
            self._next_detection()
        def on_skip():
            self._next_image()
        self.review_window = CropReviewWindow(
            self.parent, image_path, detections, on_approve, on_reject, on_skip
        )
        self.review_window.current_idx = self.current_detection_idx
        self.review_window.show_current_detection()

    def _next_detection(self):
        image_path, detections, image = self.image_batches[self.current_image_idx]
        if self.current_detection_idx < len(detections) - 1:
            self.current_detection_idx += 1
            self.review_window.current_idx = self.current_detection_idx
            self.review_window.show_current_detection()
        else:
            self._next_image()

    def _next_image(self):
        if self.review_window:
            self.review_window.window.destroy()
        if self.current_image_idx < len(self.image_batches) - 1:
            self.current_image_idx += 1
            self.current_detection_idx = 0
            self._open_review_window()
        else:
            # All images have been reviewed - call on_finish and show completion message
            if self.on_finish:
                self.on_finish(self.labels)
            # Show completion message
            messagebox.showinfo("Review Complete", 
                              f"Review session completed!\n\n"
                              f"Images reviewed: {len(self.image_batches)}\n"
                              f"Total labels applied: {len(self.labels)}")

class CrowImageIngestionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Crow Image Ingestion Tool")
        
        # Initialize logger
        logger.info("Starting Crow Image Ingestion GUI")
        
        # Initialize processing state
        self.processing = False
        self.current_image_index = 0
        self.image_files = []
        
        # Initialize tracker
        self.tracker = None
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'detections_found': 0,
            'crops_saved': 0,
            'crops_rejected': 0,
            'images_skipped': 0
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
        ttk.Label(dir_frame, text="Image Directory:").grid(row=0, column=0, sticky=tk.W)
        self.image_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.image_dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._select_image_dir).grid(row=0, column=2)
        
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
        
        # Multi-view checkboxes
        self.mv_yolo_var = tk.BooleanVar(value=False)
        self.mv_rcnn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Enable Multi-View for YOLO", variable=self.mv_yolo_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(settings_frame, text="Enable Multi-View for Faster R-CNN", variable=self.mv_rcnn_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Orientation correction checkbox
        self.orientation_correction_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Auto-correct crow orientation", variable=self.orientation_correction_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # Recursive search checkbox
        self.recursive_search_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Search subdirectories recursively", variable=self.recursive_search_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # NMS merge threshold
        self.nms_threshold_var = tk.DoubleVar(value=0.3)
        ttk.Label(settings_frame, text="Box Merge Threshold:").grid(row=5, column=0, sticky=tk.W)
        ttk.Scale(settings_frame, from_=0.1, to=0.7, variable=self.nms_threshold_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=5, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.nms_threshold_var).grid(row=5, column=2)
        
        # Bounding box padding slider
        self.bbox_padding_var = tk.DoubleVar(value=0.3)
        ttk.Label(settings_frame, text="BBox Padding:").grid(row=6, column=0, sticky=tk.W)
        ttk.Scale(settings_frame, from_=0.1, to=0.8, variable=self.bbox_padding_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=6, column=1, padx=5)
        ttk.Label(settings_frame, textvariable=self.bbox_padding_var).grid(row=6, column=2)
        
        # Control buttons
        control_frame = ttk.Frame(self.left_panel)
        control_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self._start_processing)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self._stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Progress tracking
        progress_frame = ttk.LabelFrame(self.left_panel, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(self.left_panel, text="Statistics", padding="5")
        stats_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Create statistics labels
        self.stats_labels = {}
        stats = [
            ('images_processed', 'Images Processed:'),
            ('detections_found', 'Detections Found:'),
            ('crops_saved', 'Crops Saved:'),
            ('crops_rejected', 'Crops Rejected:'),
            ('images_skipped', 'Images Skipped:')
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
        self.root.geometry("1400x800")
        
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

    def _select_image_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.image_dir_var.set(directory)
    
    def _select_output_dir(self):
        """Select output directory for crow crops."""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
            # Reinitialize tracker with new output directory
            self.tracker = CrowTracker(base_dir=directory)
            logger.info(f"Set output directory to: {directory}")

    def _start_processing(self):
        """Start processing images."""
        if not self.image_dir_var.get():
            messagebox.showerror("Error", "Please select an image directory")
            return
        
        if not self.output_dir_var.get():
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        image_dir = self.image_dir_var.get()
        output_dir = self.output_dir_var.get()
        logger.info(f"Starting image processing in directory: {image_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Initialize tracker
        correct_orientation = self.orientation_correction_var.get()
        bbox_padding = self.bbox_padding_var.get()
        
        self.tracker = CrowTracker(
            base_dir=output_dir, 
            enable_audio_extraction=False,  # No audio for images
            correct_orientation=correct_orientation,
            bbox_padding=bbox_padding
        )
        
        logger.info(f"Orientation correction: {'enabled' if correct_orientation else 'disabled'}")
        logger.info(f"BBox padding: {bbox_padding}")
        
        # Find image files
        recursive_search = self.recursive_search_var.get()
        logger.info(f"Recursive search: {'enabled' if recursive_search else 'disabled'}")
        
        self.image_files = find_image_files(image_dir, recursive=recursive_search)
        
        if not self.image_files:
            logger.error("No image files found in selected directory")
            messagebox.showerror("Error", "No image files found in selected directory")
            return
        
        logger.info(f"Found {len(self.image_files)} image files to process")
        
        # Reset statistics
        self._reset_stats()
        
        # Start processing
        self.processing = True
        self.current_image_index = 0
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self._process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_images(self):
        """Process images in a separate thread and collect all detections for review."""
        image_batches = []
        try:
            for i, image_path in enumerate(self.image_files):
                if not self.processing:
                    break
                self.current_image_index = i
                progress = (i / len(self.image_files)) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda p=image_path: self.progress_label.configure(
                    text=f"Processing {os.path.basename(p)} ({i+1}/{len(self.image_files)})"))
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Could not load image: {image_path}")
                    self.stats['images_skipped'] += 1
                    continue
                detections = detect_crows_parallel(
                    [image],
                    score_threshold=self.min_confidence_var.get(),
                    multi_view_yolo=self.mv_yolo_var.get(),
                    multi_view_rcnn=self.mv_rcnn_var.get(),
                    nms_threshold=self.nms_threshold_var.get()
                )[0]
                logger.info(f"Found {len(detections)} detections in {os.path.basename(image_path)}")
                self.stats['detections_found'] += len(detections)
                if detections:
                    image_batches.append((image_path, detections, image))
                else:
                    self.stats['images_processed'] += 1
                    logger.info(f"No detections found in {os.path.basename(image_path)}")
                self.root.after(0, self._update_stats)
        except Exception as e:
            logger.error(f"Error in processing thread: {str(e)}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing error: {str(e)}"))
        finally:
            self.root.after(0, lambda: self._start_review_session(image_batches))

    def _start_review_session(self, image_batches):
        if not image_batches:
            messagebox.showinfo("No Detections", "No detections found in any images. Processing complete!")
            self._stop_processing()
            return
        
        def on_finish(labels):
            # Process the labels and save crops
            logger.info(f"Review session completed with {len(labels)} labels")
            
            # Update statistics
            self.stats['images_processed'] += len(image_batches)
            
            # Process the approved detections
            for (image_idx, det_idx), label in labels.items():
                if image_idx < len(image_batches):
                    image_path, detections, image = image_batches[image_idx]
                    if det_idx < len(detections):
                        detection = detections[det_idx]
                        try:
                            # Save the crop
                            crop_path = self.tracker.save_crop(image, detection, label)
                            if crop_path:
                                self.stats['crops_saved'] += 1
                                logger.info(f"Saved crop: {os.path.basename(crop_path)}")
                            else:
                                self.stats['crops_rejected'] += 1
                        except Exception as e:
                            logger.error(f"Error saving crop: {e}")
                            self.stats['crops_rejected'] += 1
            
            # Update stats and stop processing
            self._update_stats()
            self._stop_processing()
        
        # Start the review session
        ReviewSession(self.root, image_batches, on_finish)

    def _stop_processing(self):
        """Stop processing images."""
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        logger.info("Stopping image processing")
        
        # Update UI
        self.status_var.set("Ready")
        
        # Show summary
        summary = (
            f"Processing complete!\n\n"
            f"Images processed: {self.stats['images_processed']}\n"
            f"Detections found: {self.stats['detections_found']}\n"
            f"Crops saved: {self.stats['crops_saved']}\n"
            f"Crops rejected: {self.stats['crops_rejected']}\n"
            f"Images skipped: {self.stats['images_skipped']}"
        )
        logger.info("Processing summary:\n" + summary)
        messagebox.showinfo("Processing Complete", summary)

    def _update_stats(self):
        """Update statistics display"""
        for key, label in self.stats_labels.items():
            label.configure(text=str(self.stats[key]))

    def _reset_stats(self):
        """Reset all statistics to zero."""
        for key in self.stats:
            self.stats[key] = 0
        self._update_stats()

def main():
    root = tk.Tk()
    app = CrowImageIngestionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 