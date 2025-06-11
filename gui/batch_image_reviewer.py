# Set up path for importing from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageTk
import threading
import queue
from pathlib import Path
from db import get_unlabeled_images, add_image_label, get_training_data_stats, get_image_label, get_connection
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_images_from_directory(directory, limit=None):
    """
    Get image files from a directory, with optional limiting for performance.
    Prioritizes unlabeled images to ensure they're available for review.
    Uses randomization to maximize diversity of labeled images for better training data.
    
    Args:
        directory (str): Directory to scan for images
        limit (int): Maximum number of images to return (None for all images)
        
    Returns:
        list: List of image file paths
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return []
        
        # Get all image files from the directory
        search_dir_path = Path(directory)
        image_files = []
        for root, dirs, files in os.walk(search_dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path_obj = Path(root) / file
                    image_files.append(full_path_obj.as_posix())
        
        # Get labeled images from database to prioritize unlabeled ones
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT image_path FROM image_labels')
            labeled_paths = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            # Separate labeled and unlabeled images
            labeled_images = [img for img in image_files if img in labeled_paths]
            unlabeled_images = [img for img in image_files if img not in labeled_paths]
            
            # If no limit specified, return all images
            if limit is None:
                # Randomize for diversity while prioritizing unlabeled images
                random.shuffle(unlabeled_images)
                random.shuffle(labeled_images)
                
                all_images = unlabeled_images + labeled_images
                logger.info(f"Loaded all {len(all_images)} images from {directory}")
                logger.info(f"  - {len(unlabeled_images)} unlabeled images")
                logger.info(f"  - {len(labeled_images)} labeled images")
                return all_images
            
            # If limit specified, prioritize unlabeled images
            selected_images = []
            
            # Add as many unlabeled images as possible first
            unlabeled_count = min(len(unlabeled_images), limit)
            if unlabeled_count > 0:
                # Randomize unlabeled images for diversity
                random.shuffle(unlabeled_images)
                selected_images.extend(unlabeled_images[:unlabeled_count])
            
            # Fill remaining slots with labeled images
            remaining_slots = limit - len(selected_images)
            if remaining_slots > 0 and labeled_images:
                # Randomize labeled images for diversity
                random.shuffle(labeled_images)
                selected_images.extend(labeled_images[:remaining_slots])
            
            # Shuffle the final list for display variety
            random.shuffle(selected_images)
            
            logger.info(f"Loaded {len(selected_images)} of {len(image_files)} total images from {directory}")
            logger.info(f"  - {len([img for img in selected_images if img in unlabeled_images])} unlabeled images")
            logger.info(f"  - {len([img for img in selected_images if img in labeled_images])} labeled images")
            
            return selected_images
            
        except Exception as db_e:
            logger.warning(f"Could not prioritize images due to database error: {db_e}")
            # Fallback to loading all images or random sampling
            if limit is None:
                # Randomize for diversity in fallback case
                random.shuffle(image_files)
                logger.info(f"Loaded all {len(image_files)} images from {directory} (randomized fallback)")
                return image_files
            else:
                random.shuffle(image_files)
                limited_files = image_files[:limit]
                logger.info(f"Loaded {len(limited_files)} of {len(image_files)} total images from {directory} (random fallback)")
                return limited_files
        
    except Exception as e:
        logger.error(f"Error getting images from directory: {e}")
        return []

def get_multiple_image_labels(image_paths):
    """Get labels for multiple images in one database query for performance."""
    if not image_paths:
        return {}
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Convert to POSIX paths for database query
        posix_paths = [Path(path).as_posix() for path in image_paths]
        
        # Create placeholders for IN query
        placeholders = ','.join('?' * len(posix_paths))
        cursor.execute(f'''
            SELECT image_path, label, confidence, reviewer_notes, 
                   timestamp, is_training_data, created_at, updated_at
            FROM image_labels 
            WHERE image_path IN ({placeholders})
        ''', posix_paths)
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = {
                'label': row[1],
                'confidence': row[2],
                'reviewer_notes': row[3],
                'timestamp': row[4],
                'is_training_data': bool(row[5]),
                'created_at': row[6],
                'updated_at': row[7]
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting multiple image labels: {e}")
        return {}
    finally:
        if conn:
            conn.close()

class BatchImageReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Crow Image Reviewer")
        self.root.geometry("1400x900")
        
        # Initialize state
        self.all_images = []  # All available images
        self.displayed_images = []  # Currently displayed images
        self.photo_objects = []  # Keep references to prevent garbage collection
        self.selected_images = set()
        self.thumbnail_size = (120, 120)
        self.columns = 8
        self.loading = False
        self.labels_queue = queue.Queue()
        self.checkbox_vars = {}  # Track checkbox variables by image path
        
        # Performance optimization: caching
        self.filter_cache = {}  # Cache filtered results
        self.label_cache = {}   # Cache database lookups
        self.cache_valid = True # Track if cache needs refresh
        
        # Pagination settings
        self.images_per_page = 100  # Show 100 images at a time (manageable for Tkinter)
        self.current_page = 0
        
        # Image viewer window reference
        self.viewer_window = None
        
        # Add to __init__:
        self.last_clicked_index = None  # Track last clicked image index for shift selection
        
        # Create main layout
        self.create_layout()
        
        # Start background processing
        self.process_queue()
        
    def create_layout(self):
        """Create the main GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Image grid row
        
        # Top control panel
        control_frame = ttk.Frame(main_frame, padding="5")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        control_frame.columnconfigure(1, weight=1)
        
        # Directory selection (first row)
        ttk.Label(control_frame, text="Directory:").grid(row=0, column=0, padx=(0, 5))
        self.dir_var = tk.StringVar(value="dataset")
        dir_entry = ttk.Entry(control_frame, textvariable=self.dir_var, width=40)
        dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(control_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(control_frame, text="Load Images", command=self.load_images).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(control_frame, text="Refresh", command=self.refresh_images).grid(row=0, column=4)
        
        # Quick-select buttons
        quick_select_frame = ttk.Frame(control_frame)
        quick_select_frame.grid(row=2, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(5, 0))
        ttk.Label(quick_select_frame, text="Quick Select:").grid(row=0, column=0, padx=(0, 5))
        ttk.Button(quick_select_frame, text="Crows (generic)", command=lambda: self.dir_var.set("dataset/crows/generic")).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(quick_select_frame, text="Not Crow", command=lambda: self.dir_var.set("dataset/not_crow")).grid(row=0, column=2, padx=(0, 5))
        
        # Filter options (second row)
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Image filter dropdown
        ttk.Label(filter_frame, text="Show:").grid(row=0, column=0, sticky=tk.W)
        
        self.filter_var = tk.StringVar(value="All Images")
        filter_options = [
            "All Images",
            "Unlabeled Only", 
            "Crow Only",
            "Not Crow Only",
            "Bad Crow Only", 
            "Not Sure Only",
            "Multi Crow Only"
        ]
        
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, 
                                   values=filter_options, state="readonly", width=15)
        filter_combo.grid(row=0, column=1, padx=(5, 0), sticky=tk.W)
        filter_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        # Thumbnail size control
        ttk.Label(filter_frame, text="  |  Thumbnail Size:").grid(row=0, column=2, padx=(20, 5))
        
        self.thumb_size_var = tk.IntVar(value=120)
        thumb_scale = ttk.Scale(filter_frame, from_=80, to=200, orient=tk.HORIZONTAL, 
                               variable=self.thumb_size_var, length=100,
                               command=self.on_thumbnail_size_change)
        thumb_scale.grid(row=0, column=3, padx=(0, 5))
        
        self.thumb_size_label = ttk.Label(filter_frame, text="120px")
        self.thumb_size_label.grid(row=0, column=4, padx=(0, 10))
        
        ttk.Label(filter_frame, text="  |  Legend: ").grid(row=0, column=5, padx=(20, 0))
        
        # Legend
        legend_frame = ttk.Frame(filter_frame)
        legend_frame.grid(row=0, column=6, sticky=tk.W, padx=(5, 0))
        
        legend_items = [
            ("✓ Crow", "#28a745"),
            ("✗ Not Crow", "#dc3545"), 
            ("🚫 Bad Crow", "#fd7e14"),
            ("? Unsure", "#ffc107"),
            ("👥 Multi", "#17a2b8"),
            ("Unlabeled", "#6c757d")
        ]
        
        for i, (text, color) in enumerate(legend_items):
            legend_label = tk.Label(legend_frame, text=text, fg=color, font=("Arial", 8, "bold"))
            legend_label.grid(row=0, column=i, padx=(0, 10))
        
        # Progress and stats
        progress_frame = ttk.Frame(main_frame, padding="5")
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="No images loaded")
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        # Pagination controls
        pagination_frame = ttk.Frame(progress_frame)
        pagination_frame.grid(row=0, column=1, padx=(20, 0))
        
        self.prev_btn = ttk.Button(pagination_frame, text="← Previous", command=self.previous_page, state='disabled')
        self.prev_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.page_label = ttk.Label(pagination_frame, text="Page 0 of 0")
        self.page_label.grid(row=0, column=1, padx=(0, 5))
        
        self.next_btn = ttk.Button(pagination_frame, text="Next →", command=self.next_page, state='disabled')
        self.next_btn.grid(row=0, column=2, padx=(0, 5))
        
        self.stats_label = ttk.Label(progress_frame, text="")
        self.stats_label.grid(row=0, column=2, sticky=tk.E)
        progress_frame.columnconfigure(2, weight=1)
        
        # Image grid with scrollbar
        grid_frame = ttk.Frame(main_frame)
        grid_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        
        # Create scrollable canvas
        self.canvas = tk.Canvas(grid_frame, bg='white')
        scrollbar = ttk.Scrollbar(grid_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Create window in canvas for scrollable frame
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bind canvas resize
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.scrollable_frame.bind('<Configure>', self.on_frame_configure)
        
        # Bottom control panel
        bottom_frame = ttk.Frame(main_frame, padding="5")
        bottom_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Selection info
        self.selection_label = ttk.Label(bottom_frame, text="No images selected")
        self.selection_label.grid(row=0, column=0, sticky=tk.W)
        
        # Selection controls
        select_frame = ttk.Frame(bottom_frame)
        select_frame.grid(row=0, column=1, padx=(20, 0))
        
        ttk.Button(select_frame, text="Select All", command=self.select_all).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(select_frame, text="Clear Selection", command=self.clear_selection).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(select_frame, text="Invert Selection", command=self.invert_selection).grid(row=0, column=2)
        
        # Labeling controls
        label_frame = ttk.LabelFrame(bottom_frame, text="Label Selected Images", padding="5")
        label_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Label buttons
        button_frame = ttk.Frame(label_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        ttk.Button(button_frame, text="✓ Crow (1)", 
                   command=lambda: self.label_selected("crow")).grid(row=0, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="✗ Not Crow (2)", 
                   command=lambda: self.label_selected("not_a_crow")).grid(row=0, column=1, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="🚫 Bad Crow (3)", 
                   command=lambda: self.label_selected("bad_crow")).grid(row=0, column=2, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="? Not Sure (4)", 
                   command=lambda: self.label_selected("not_sure")).grid(row=0, column=3, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="👥 Multiple (5)", 
                   command=lambda: self.label_selected("multi_crow")).grid(row=0, column=4, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # Progress bar for batch operations
        self.batch_progress = ttk.Progressbar(label_frame, mode='determinate')
        self.batch_progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Keyboard shortcuts
        self.root.bind('<Key-1>', lambda e: self.label_selected("crow"))
        self.root.bind('<Key-2>', lambda e: self.label_selected("not_a_crow"))
        self.root.bind('<Key-3>', lambda e: self.label_selected("bad_crow"))
        self.root.bind('<Key-4>', lambda e: self.label_selected("not_sure"))
        self.root.bind('<Key-5>', lambda e: self.label_selected("multi_crow"))
        self.root.bind('<Control-a>', lambda e: self.select_all())
        self.root.bind('<Control-d>', lambda e: self.clear_selection())
        self.root.bind('<Control-i>', lambda e: self.invert_selection())
        self.root.bind('<Left>', lambda e: self.previous_page())
        self.root.bind('<Right>', lambda e: self.next_page())
        self.root.bind('<Prior>', lambda e: self.previous_page())  # Page Up
        self.root.bind('<Next>', lambda e: self.next_page())  # Page Down
        
        # Make window focusable for keyboard shortcuts
        self.root.focus_set()
        
        # Update initial stats
        self.update_stats()
        
    def on_canvas_configure(self, event):
        """Handle canvas resize."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        
        # Account for scrollbar width and padding
        scrollbar_width = 20  # Approximate scrollbar width
        available_width = canvas_width - scrollbar_width - 10  # Extra margin for safety
        
        # Calculate actual space needed per thumbnail (including all padding and borders)
        # Frame border (3px each side = 6px) + padx (2px each side = 4px) + inner padding (~20px) 
        thumbnail_total_width = self.thumbnail_size[0] + 30  # Conservative estimate
        
        # Recalculate columns based on available width
        new_columns = max(1, available_width // thumbnail_total_width)
        if new_columns != self.columns:
            self.columns = new_columns
            if self.displayed_images:
                self.display_images()
                
    def on_frame_configure(self, event):
        """Handle scrollable frame resize."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def browse_directory(self):
        """Browse for image directory."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.dir_var.set(directory)
    
    def refresh_images(self):
        """Refresh the current image list to pick up new images."""
        if not self.dir_var.get():
            messagebox.showinfo("Info", "Please select a directory first")
            return
        
        # Store current filter to maintain user's view
        current_filter = self.filter_var.get()
        
        # Reload images
        self.load_images()
        
        # Restore filter after loading
        self.root.after(500, lambda: self._restore_filter_view(current_filter))
    
    def _restore_filter_view(self, filter_name):
        """Helper to restore filter view after refresh."""
        self.filter_var.set(filter_name)
        self.display_current_page()
            
    def load_images(self):
        """Load images from the specified directory."""
        if self.loading:
            messagebox.showinfo("Loading", "Images are already being loaded. Please wait.")
            return
            
        directory = self.dir_var.get()
        if not directory or not os.path.exists(directory):
            messagebox.showerror("Error", "Please select a valid directory")
            return
            
        self.loading = True
        self.progress_label.config(text="Loading images...")
        
        # Clear existing data
        self.all_images.clear()
        self.photo_objects.clear()
        self.selected_images.clear()
        self.clear_grid()
        
        # Clear caches when loading new images
        self.filter_cache.clear()
        self.label_cache.clear()
        self.cache_valid = True
        
        # Load images in background thread (load all images, not limited)
        threading.Thread(target=self._load_images_background, args=(directory,), daemon=True).start()
        
    def _load_images_background(self, directory):
        """Load images in background thread."""
        try:
            # Get all images from the directory (no limit to ensure all are loaded)
            all_images = get_all_images_from_directory(directory, limit=None)
            
            if not all_images:
                self.root.after(0, lambda: self.progress_label.config(text="No images found"))
                self.loading = False
                return
                
            # Store image paths
            self.all_images = all_images
            
            # Update UI in main thread
            self.root.after(0, self._on_images_loaded)
            
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load images: {str(e)}"))
            self.loading = False
            
    def _on_images_loaded(self):
        """Called when images are loaded."""
        self.loading = False
        self.current_page = 0
        self.update_pagination_info()
        self.display_current_page()
        self.update_selection_label()
        
    def update_pagination_info(self):
        """Update pagination controls and labels."""
        if not self.all_images:
            self.page_label.config(text="Page 0 of 0")
            self.prev_btn.config(state='disabled')
            self.next_btn.config(state='disabled')
            self.progress_label.config(text="No images loaded")
            return
            
        total_pages = (len(self.all_images) + self.images_per_page - 1) // self.images_per_page
        current_page_display = self.current_page + 1
        
        self.page_label.config(text=f"Page {current_page_display} of {total_pages}")
        self.prev_btn.config(state='normal' if self.current_page > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_page < total_pages - 1 else 'disabled')
        
        # Update progress label with current page info
        start_idx = self.current_page * self.images_per_page + 1
        end_idx = min((self.current_page + 1) * self.images_per_page, len(self.all_images))
        self.progress_label.config(text=f"Showing {start_idx}-{end_idx} of {len(self.all_images)} images")
        
    def display_current_page(self):
        """Display the current page of images."""
        # Use optimized filtering instead of individual database queries
        working_images = self.get_filtered_images(self.filter_var.get())
        
        # Calculate pagination for filtered list
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(working_images))
        
        self.displayed_images = working_images[start_idx:end_idx]
        
        # Update pagination info based on filtered list
        self.update_pagination_info_filtered(working_images)
        
        self.display_images()
        
    def update_pagination_info_filtered(self, working_images):
        """Update pagination controls and labels for filtered images."""
        if not working_images:
            self.page_label.config(text="Page 0 of 0")
            self.prev_btn.config(state='disabled')
            self.next_btn.config(state='disabled')
            filter_text = f" ({self.filter_var.get()})" if self.filter_var.get() != "All Images" else ""
            refresh_hint = " - Click 'Refresh' to check for new images" if self.filter_var.get() == "Unlabeled Only" else ""
            self.progress_label.config(text=f"No images found{filter_text}{refresh_hint}")
            return
            
        total_pages = (len(working_images) + self.images_per_page - 1) // self.images_per_page
        current_page_display = self.current_page + 1
        
        self.page_label.config(text=f"Page {current_page_display} of {total_pages}")
        self.prev_btn.config(state='normal' if self.current_page > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_page < total_pages - 1 else 'disabled')
        
        # Update progress label with current page info
        start_idx = self.current_page * self.images_per_page + 1
        end_idx = min((self.current_page + 1) * self.images_per_page, len(working_images))
        filter_text = f" {self.filter_var.get()}" if self.filter_var.get() != "All Images" else ""
        total_text = f" of {len(self.all_images)} total" if self.filter_var.get() != "All Images" else ""
        self.progress_label.config(text=f"Showing {start_idx}-{end_idx} of {len(working_images)}{filter_text} images{total_text}")
        
    def previous_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_current_page()
            
    def next_page(self):
        """Go to next page."""
        # Use optimized filtering instead of individual database queries
        filtered_images = self.get_filtered_images(self.filter_var.get())
            
        total_pages = (len(filtered_images) + self.images_per_page - 1) // self.images_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.display_current_page()
        
    def display_images(self):
        """Display images in grid layout."""
        # Clear existing grid
        self.clear_grid()
        
        if not self.displayed_images:
            return
            
        # Create thumbnails in background
        threading.Thread(target=self._create_thumbnails, args=(self.displayed_images,), daemon=True).start()
        
    def _create_thumbnails(self, image_paths):
        """Create thumbnails for images."""
        thumbnails = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load and resize image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    # Create thumbnail
                    img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(img)
                    thumbnails.append((image_path, photo))
                    
            except Exception as e:
                logger.warning(f"Failed to load thumbnail for {image_path}: {e}")
                # Create placeholder
                placeholder = Image.new('RGB', self.thumbnail_size, color='gray')
                photo = ImageTk.PhotoImage(placeholder)
                thumbnails.append((image_path, photo))
        
        # Update UI in main thread
        self.root.after(0, lambda: self._display_thumbnails(thumbnails))
        
    def _display_thumbnails(self, thumbnails):
        """Display thumbnails in the grid."""
        row = 0
        col = 0
        
        # Define status colors and labels
        status_colors = {
            'crow': '#28a745',           # Green
            'not_a_crow': '#dc3545',     # Red  
            'bad_crow': '#fd7e14',       # Orange
            'not_sure': '#ffc107',       # Yellow
            'multi_crow': '#17a2b8',     # Blue
            None: '#6c757d'              # Gray for unlabeled
        }
        
        status_text = {
            'crow': '✓ CROW',
            'not_a_crow': '✗ NOT CROW',
            'bad_crow': '🚫 BAD CROW',
            'not_sure': '? UNSURE',
            'multi_crow': '👥 MULTI',
            None: ''
        }
        
        for idx, (image_path, photo) in enumerate(thumbnails):
            # Check current label status
            label_info = get_image_label(image_path)
            current_label = label_info['label'] if label_info else None
            
            # Create frame for image and checkbox with colored border
            border_color = status_colors.get(current_label, status_colors[None])
            img_frame = tk.Frame(self.scrollable_frame, bg=border_color, bd=3, relief='solid')
            img_frame.grid(row=row, column=col, padx=2, pady=2)
            
            # Inner frame for content (white background)
            inner_frame = ttk.Frame(img_frame, padding="2")
            inner_frame.grid(row=0, column=0, padx=2, pady=2)
            
            # Create checkbox for selection
            var = tk.BooleanVar()
            # Set initial state based on current selection
            var.set(image_path in self.selected_images)
            self.checkbox_vars[image_path] = var  # Store reference
            
            checkbox = ttk.Checkbutton(inner_frame, variable=var)
            checkbox.grid(row=0, column=0, sticky=tk.W)
            
            # Bind <Button-1> to checkbox for shift selection
            # This handler will do range selection if shift is held, and prevent default toggle

            def make_checkbox_click_handler(path, check_var, idx):
                def handler(event):
                    if event.state & 0x0001:  # Shift is held
                        self.toggle_selection(path, not check_var.get(), index=idx, event=event)
                        check_var.set(path in self.selected_images)
                        return "break"  # Prevent default toggle
                    # Otherwise, allow default toggle, but track last clicked
                    self.toggle_selection(path, not check_var.get(), index=idx, event=event)
                    # Let default behavior proceed
                return handler
            checkbox.bind("<Button-1>", make_checkbox_click_handler(image_path, var, idx))
            
            # Add status text if labeled
            if current_label:
                status_label = tk.Label(inner_frame, text=status_text[current_label], 
                                        font=("Arial", 7, "bold"), 
                                        fg=border_color, bg='white')
                status_label.grid(row=0, column=1, sticky=tk.E, padx=(5, 0))
            
            # Create label for image
            img_label = ttk.Label(inner_frame, image=photo)
            img_label.grid(row=1, column=0, columnspan=2)
            
            # CRITICAL: Store reference to prevent garbage collection
            img_label.image = photo  # Keep a reference on the label itself
            
            # Add filename
            filename = os.path.basename(image_path)
            name_label = ttk.Label(inner_frame, text=filename[:15] + ("..." if len(filename) > 15 else ""),
                                   font=("Arial", 8))
            name_label.grid(row=2, column=0, columnspan=2)
            
            # Store reference to prevent garbage collection
            self.photo_objects.append(photo)
            
            # Store checkbox variable for later updates
            checkbox.image_path = image_path
            
            # Bind click events to image for easy selection
            def make_click_handler(path, check_var, idx):
                def handler(event):
                    # Toggle selection with shift support
                    self.toggle_selection(path, not check_var.get(), index=idx, event=event)
                    check_var.set(path in self.selected_images)
                return handler
            
            img_label.bind("<Button-1>", make_click_handler(image_path, var, idx))
            
            # Bind double-click to show enlarged image
            img_label.bind("<Double-Button-1>", lambda e, path=image_path: self.show_enlarged_image(path))
            
            # Move to next position
            col += 1
            if col >= self.columns:
                col = 0
                row += 1
                
        # Update scroll region
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def clear_grid(self):
        """Clear the image grid."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.photo_objects.clear()
        self.checkbox_vars.clear()  # Clear checkbox variable references
        
    def toggle_selection(self, image_path, selected, index=None, event=None):
        """Toggle selection of an image, with optional shift-click support."""
        if event and hasattr(event, 'state') and (event.state & 0x0001):  # Shift is held
            if self.last_clicked_index is not None and index is not None:
                start = min(self.last_clicked_index, index)
                end = max(self.last_clicked_index, index)
                for i in range(start, end + 1):
                    path = self.displayed_images[i]
                    self.selected_images.add(path)
                    if path in self.checkbox_vars:
                        self.checkbox_vars[path].set(True)
            else:
                if selected:
                    self.selected_images.add(image_path)
                else:
                    self.selected_images.discard(image_path)
        else:
            if selected:
                self.selected_images.add(image_path)
            else:
                self.selected_images.discard(image_path)
            if index is not None:
                self.last_clicked_index = index
        self.update_selection_label()
        
    def select_all(self):
        """Select all visible images."""
        for image_path in self.displayed_images:
            self.selected_images.add(image_path)
            # Update checkbox visual state
            if image_path in self.checkbox_vars:
                self.checkbox_vars[image_path].set(True)
                
        self.update_selection_label()
        
    def clear_selection(self):
        """Clear all selections."""
        self.selected_images.clear()
        
        # Update checkbox visual states
        for image_path, var in self.checkbox_vars.items():
            var.set(False)
            
        self.update_selection_label()
        
    def invert_selection(self):
        """Invert current selection."""
        # Invert selection for currently displayed images only
        new_selection = set(self.selected_images)
        
        for image_path in self.displayed_images:
            if image_path in self.selected_images:
                new_selection.discard(image_path)
            else:
                new_selection.add(image_path)
        
        self.selected_images = new_selection
        
        # Update checkbox visual states for displayed images
        for image_path in self.displayed_images:
            if image_path in self.checkbox_vars:
                self.checkbox_vars[image_path].set(image_path in self.selected_images)
                
        self.update_selection_label()
        
    def update_selection_label(self):
        """Update the selection count label."""
        count = len(self.selected_images)
        if count == 0:
            self.selection_label.config(text="No images selected")
        elif count == 1:
            self.selection_label.config(text="1 image selected")
        else:
            self.selection_label.config(text=f"{count} images selected")
            
    def label_selected(self, label):
        """Label all selected images."""
        if not self.selected_images:
            messagebox.showinfo("No Selection", "Please select images to label first.")
            return
            
        count = len(self.selected_images)
        if not messagebox.askyesno("Confirm", f"Label {count} images as '{label.replace('_', ' ')}'?"):
            return
        
        logger.info(f"Starting batch labeling of {count} images as '{label}'")
        
        # Show progress
        self.batch_progress['maximum'] = count
        self.batch_progress['value'] = 0
        
        # Disable UI during processing to prevent conflicts
        self.root.config(cursor="wait")
        
        # Process labels in background
        selected_list = list(self.selected_images)
        self.labels_queue.put(('batch_label', selected_list, label))
        
        # Don't clear selection here - wait for completion
        
    def process_queue(self):
        """Process the labeling queue in background."""
        try:
            while not self.labels_queue.empty():
                action, data, label = self.labels_queue.get_nowait()
                
                if action == 'batch_label':
                    self._process_batch_labels(data, label)
                    
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing label queue: {e}")
            
        # Schedule next queue processing
        self.root.after(1000, self.process_queue)
        
    def _process_batch_labels(self, image_paths, label):
        """Process batch labels in background thread."""
        def background_process():
            try:
                total = len(image_paths)
                successful = 0
                failed = 0
                
                for i, image_path in enumerate(image_paths):
                    try:
                        # Determine if this should be training data
                        is_training_data = label == "crow"
                        
                        # Save label to database
                        add_image_label(image_path, label, is_training_data=is_training_data)
                        logger.info(f"Labeled {os.path.basename(image_path)} as {label}")
                        successful += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to label {os.path.basename(image_path)}: {e}")
                        failed += 1
                    
                    # Update progress
                    progress = i + 1
                    self.root.after(0, lambda p=progress: self.batch_progress.config(value=p))
                    
                # Log final results
                logger.info(f"Batch labeling complete: {successful} successful, {failed} failed")
                
                # Update UI when done
                self.root.after(0, self._on_batch_complete)
                
            except Exception as e:
                logger.error(f"Error in batch labeling: {e}")
                self.root.after(0, lambda: self._on_batch_error(str(e)))
                
        threading.Thread(target=background_process, daemon=True).start()
        
    def _on_batch_complete(self):
        """Called when batch labeling is complete."""
        self.batch_progress['value'] = 0
        
        # Restore normal cursor
        self.root.config(cursor="")
        
        # Clear selection
        self.selected_images.clear()
        for var in self.checkbox_vars.values():
            var.set(False)
        
        # Invalidate cache so filters refresh with new labels
        self.cache_valid = False
        self.filter_cache.clear()
        self.label_cache.clear()
        
        # Only refresh current page (not entire dataset)
        logger.info("Batch labeling complete, refreshing current page...")
        self.display_current_page()
        self.update_stats()
        
        # Show completion message
        messagebox.showinfo("Complete", "Batch labeling completed successfully!")
        
    def _on_batch_error(self, error_message):
        """Called when batch labeling encounters an error."""
        self.batch_progress['value'] = 0
        
        # Restore normal cursor
        self.root.config(cursor="")
        
        # Clear selection
        self.selected_images.clear()
        for var in self.checkbox_vars.values():
            var.set(False)
        
        # Try to refresh display anyway in case some labels succeeded
        self.display_current_page()
        self.update_stats()
        
        # Show error message
        messagebox.showerror("Error", f"Batch labeling failed: {error_message}")
        
    def update_stats(self):
        """Update the statistics display."""
        try:
            # Use the currently selected directory for stats
            stats = get_training_data_stats(self.dir_var.get())
            if not stats:
                self.stats_label.config(text="No labeled data yet")
                return
            crow_count = stats.get('crow', {}).get('count', 0)
            not_crow_count = stats.get('not_a_crow', {}).get('count', 0)
            bad_crow_count = stats.get('bad_crow', {}).get('count', 0)
            not_sure_count = stats.get('not_sure', {}).get('count', 0)
            multi_crow_count = stats.get('multi_crow', {}).get('count', 0)
            stats_text = f"Labeled: {crow_count} crows, {not_crow_count} not-crows, {bad_crow_count} bad-crows, {not_sure_count} unsure, {multi_crow_count} multi"
            self.stats_label.config(text=stats_text)
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
            self.stats_label.config(text="Error loading stats")

    def on_filter_change(self, event):
        """Handle filter change."""
        self.current_page = 0
        # Cache is still valid, just switching to different filter
        self.display_current_page()

    def on_thumbnail_size_change(self, event):
        """Handle thumbnail size change."""
        new_size = int(self.thumb_size_var.get())
        self.thumbnail_size = (new_size, new_size)
        self.thumb_size_label.config(text=f"{new_size}px")
        
        # Recalculate columns for new thumbnail size
        if hasattr(self, 'canvas'):
            canvas_width = self.canvas.winfo_width()
            if canvas_width > 1:  # Only if canvas is properly initialized
                scrollbar_width = 20
                available_width = canvas_width - scrollbar_width - 10
                thumbnail_total_width = self.thumbnail_size[0] + 30
                new_columns = max(1, available_width // thumbnail_total_width)
                if new_columns != self.columns:
                    self.columns = new_columns
        
        # Refresh display if we have images loaded
        if self.displayed_images:
            self.display_current_page()

    def show_enlarged_image(self, image_path):
        """Show the enlarged image in a new window."""
        try:
            # Close previous viewer window if open
            if self.viewer_window and self.viewer_window.winfo_exists():
                self.viewer_window.destroy()
            
            # Load the image
            with Image.open(image_path) as img:
                # Get original image dimensions
                orig_width, orig_height = img.size
                
                # Calculate display size (max 800x600 while maintaining aspect ratio)
                max_width, max_height = 800, 600
                ratio = min(max_width / orig_width, max_height / orig_height)
                
                if ratio < 1:  # Only resize if image is larger than max size
                    display_width = int(orig_width * ratio)
                    display_height = int(orig_height * ratio)
                    display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
                else:
                    display_img = img.copy()
                    display_width, display_height = orig_width, orig_height
                
                # Create a new window for the enlarged image
                self.viewer_window = tk.Toplevel(self.root)
                self.viewer_window.title(f"Enlarged Image: {os.path.basename(image_path)}")
                self.viewer_window.geometry(f"{display_width + 40}x{display_height + 80}")
                
                # Create main frame
                main_frame = ttk.Frame(self.viewer_window, padding="10")
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                # Add image info
                info_label = ttk.Label(main_frame, 
                                      text=f"File: {os.path.basename(image_path)}\nOriginal Size: {orig_width}x{orig_height}")
                info_label.pack(pady=(0, 10))
                
                # Convert to PhotoImage and display
                photo = ImageTk.PhotoImage(display_img)
                img_label = ttk.Label(main_frame, image=photo)
                img_label.pack()
                
                # Keep reference to prevent garbage collection
                img_label.image = photo
                
                # Add close button
                close_btn = ttk.Button(main_frame, text="Close", 
                                     command=self.viewer_window.destroy)
                close_btn.pack(pady=(10, 0))
                
                # Center the window
                self.viewer_window.transient(self.root)
                self.viewer_window.grab_set()
                
        except Exception as e:
            logger.error(f"Error showing enlarged image {image_path}: {e}")
            messagebox.showerror("Error", f"Failed to display enlarged image: {str(e)}")

    def get_filtered_images(self, filter_type):
        """Get filtered images using cache for performance."""
        # Check cache first
        if self.cache_valid and filter_type in self.filter_cache:
            return self.filter_cache[filter_type]
        
        # If not cached or cache invalid, compute
        if filter_type == "All Images":
            filtered_images = self.all_images
        elif filter_type == "Unlabeled Only":
            # Batch query for all labels at once
            all_labels = get_multiple_image_labels(self.all_images)
            filtered_images = [img for img in self.all_images 
                             if Path(img).as_posix() not in all_labels]
        else:
            # For specific label filters
            label_map = {
                "Crow Only": "crow",
                "Not Crow Only": "not_a_crow", 
                "Bad Crow Only": "bad_crow",
                "Not Sure Only": "not_sure",
                "Multi Crow Only": "multi_crow"
            }
            target_label = label_map.get(filter_type)
            if target_label:
                all_labels = get_multiple_image_labels(self.all_images)
                filtered_images = []
                for img in self.all_images:
                    img_posix = Path(img).as_posix()
                    if img_posix in all_labels and all_labels[img_posix].get('label') == target_label:
                        filtered_images.append(img)
            else:
                filtered_images = self.all_images
        
        # Cache the result
        self.filter_cache[filter_type] = filtered_images
        logger.info(f"Filtered {len(filtered_images)} images for '{filter_type}'")
        return filtered_images

def main():
    """Main function to run the batch image reviewer."""
    root = tk.Tk()
    app = BatchImageReviewer(root)
    
    # Add help dialog
    help_text = """Batch Crow Image Reviewer

Instructions:
1. Select directory containing crow crop images
2. Click 'Load Images' to display all available thumbnails
3. Use 'Refresh' button to reload images after new crops are added
4. Use pagination controls to navigate through pages
5. Use the "Show" dropdown to filter by label type:
   - All Images: Show everything
   - Unlabeled Only: Show only unlabeled images
   - Crow Only: Show only good crow images  
   - Not Crow Only: Show only non-crow images
   - Bad Crow Only: Show only poor quality crow images
   - Not Sure Only: Show only uncertain images
   - Multi Crow Only: Show only multi-crow images
6. Adjust thumbnail size using the slider in the filter bar
7. Click on images or checkboxes to select them
8. Double-click any image to view it enlarged
9. Use selection controls:
   - Select All / Clear Selection / Invert Selection
10. Choose label and apply to all selected images

Performance Features:
• Loads ALL images from directory in random order for diversity
• Prioritizes unlabeled images while maintaining random ordering
• Intelligent caching for instant filter switching
• Batch database queries for faster labeling
• Smart refresh - automatically updates after labeling
• Refresh button to pick up newly processed crops

Label Categories:
• Crow - Good quality crow images suitable for training
• Not Crow - Images that aren't crows at all (false positives)
• Bad Crow - Images that are crows but poor quality/unusable
• Not Sure - Uncertain cases that need review
• Multiple - Images containing multiple crows

Keyboard Shortcuts:
• 1 - Label selected as Crow
• 2 - Label selected as Not Crow  
• 3 - Label selected as Bad Crow
• 4 - Label selected as Not Sure
• 5 - Label selected as Multiple Crows
• Ctrl+A - Select All (current page)
• Ctrl+D - Clear Selection
• Ctrl+I - Invert Selection (all images)
• ← → Arrow Keys - Navigate pages
• Page Up/Down - Navigate pages

Features:
• Fast batch processing of multiple images
• Paginated display (100 images per page)
• Filter by specific label categories
• Adjustable thumbnail sizes (80-200px)
• Double-click images to view enlarged
• Visual thumbnail grid with selection
• Automatic progress tracking
• Memory-efficient loading
• Improved column layout prevents spillover
• Performance optimized for large datasets
"""
    
    def show_help():
        messagebox.showinfo("Help - Batch Image Reviewer", help_text)
    
    # Add help menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Instructions", command=show_help)
    
    root.mainloop()

if __name__ == "__main__":
    main() 
    