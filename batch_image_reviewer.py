import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageTk
import threading
import queue
from pathlib import Path
from db import get_unlabeled_images, add_image_label, get_training_data_stats, get_image_label
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Pagination settings
        self.images_per_page = 100  # Show 100 images at a time (manageable for Tkinter)
        self.current_page = 0
        
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
        self.dir_var = tk.StringVar(value="crow_crops")
        dir_entry = ttk.Entry(control_frame, textvariable=self.dir_var, width=40)
        dir_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(control_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(control_frame, text="Load Images", command=self.load_images).grid(row=0, column=3)
        
        # Filter options (second row)
        filter_frame = ttk.Frame(control_frame)
        filter_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.show_only_unlabeled = tk.BooleanVar(value=False)
        unlabeled_check = ttk.Checkbutton(filter_frame, text="Show only unlabeled images", 
                                          variable=self.show_only_unlabeled,
                                          command=self.apply_filter)
        unlabeled_check.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(filter_frame, text="  |  Legend: ").grid(row=0, column=1, padx=(20, 0))
        
        # Legend
        legend_frame = ttk.Frame(filter_frame)
        legend_frame.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
        
        legend_items = [
            ("✓ Crow", "#28a745"),
            ("✗ Not Crow", "#dc3545"), 
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
        button_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        ttk.Button(button_frame, text="✓ Crow (1)", 
                   command=lambda: self.label_selected("crow")).grid(row=0, column=0, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="✗ Not Crow (2)", 
                   command=lambda: self.label_selected("not_a_crow")).grid(row=0, column=1, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="? Not Sure (3)", 
                   command=lambda: self.label_selected("not_sure")).grid(row=0, column=2, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="👥 Multiple (4)", 
                   command=lambda: self.label_selected("multi_crow")).grid(row=0, column=3, padx=2, pady=2, sticky=(tk.W, tk.E))
        
        # Progress bar for batch operations
        self.batch_progress = ttk.Progressbar(label_frame, mode='determinate')
        self.batch_progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Keyboard shortcuts
        self.root.bind('<Key-1>', lambda e: self.label_selected("crow"))
        self.root.bind('<Key-2>', lambda e: self.label_selected("not_a_crow"))
        self.root.bind('<Key-3>', lambda e: self.label_selected("not_sure"))
        self.root.bind('<Key-4>', lambda e: self.label_selected("multi_crow"))
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
        
        # Recalculate columns based on canvas width
        new_columns = max(1, canvas_width // (self.thumbnail_size[0] + 10))
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
        
        # Load images in background thread
        threading.Thread(target=self._load_images_background, args=(directory,), daemon=True).start()
        
    def _load_images_background(self, directory):
        """Load images in background thread."""
        try:
            # Get all unlabeled images (no limit for batch processing)
            unlabeled_images = get_unlabeled_images(limit=10000, from_directory=directory)
            
            if not unlabeled_images:
                self.root.after(0, lambda: self.progress_label.config(text="No unlabeled images found"))
                self.loading = False
                return
                
            # Store image paths
            self.all_images = unlabeled_images
            
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
        # Apply filter if needed
        if self.show_only_unlabeled.get():
            # Filter to show only unlabeled images
            filtered_images = []
            for img_path in self.all_images:
                label_info = get_image_label(img_path)
                if not label_info:  # No label = unlabeled
                    filtered_images.append(img_path)
            working_images = filtered_images
        else:
            # Show all images
            working_images = self.all_images
        
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
            filter_text = " (filtered)" if self.show_only_unlabeled.get() else ""
            self.progress_label.config(text=f"No images found{filter_text}")
            return
            
        total_pages = (len(working_images) + self.images_per_page - 1) // self.images_per_page
        current_page_display = self.current_page + 1
        
        self.page_label.config(text=f"Page {current_page_display} of {total_pages}")
        self.prev_btn.config(state='normal' if self.current_page > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_page < total_pages - 1 else 'disabled')
        
        # Update progress label with current page info
        start_idx = self.current_page * self.images_per_page + 1
        end_idx = min((self.current_page + 1) * self.images_per_page, len(working_images))
        filter_text = " unlabeled" if self.show_only_unlabeled.get() else ""
        total_text = f" of {len(self.all_images)} total" if self.show_only_unlabeled.get() else ""
        self.progress_label.config(text=f"Showing {start_idx}-{end_idx} of {len(working_images)}{filter_text} images{total_text}")
        
    def previous_page(self):
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_current_page()
            
    def next_page(self):
        """Go to next page."""
        # Calculate total pages for current filter
        if self.show_only_unlabeled.get():
            filtered_images = []
            for img_path in self.all_images:
                label_info = get_image_label(img_path)
                if not label_info:
                    filtered_images.append(img_path)
            working_images = filtered_images
        else:
            working_images = self.all_images
            
        total_pages = (len(working_images) + self.images_per_page - 1) // self.images_per_page
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
            'not_sure': '#ffc107',       # Yellow
            'multi_crow': '#17a2b8',     # Blue
            None: '#6c757d'              # Gray for unlabeled
        }
        
        status_text = {
            'crow': '✓ CROW',
            'not_a_crow': '✗ NOT CROW',
            'not_sure': '? UNSURE',
            'multi_crow': '👥 MULTI',
            None: ''
        }
        
        for image_path, photo in thumbnails:
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
            checkbox = ttk.Checkbutton(inner_frame, variable=var,
                                       command=lambda path=image_path, v=var: self.toggle_selection(path, v.get()))
            checkbox.grid(row=0, column=0, sticky=tk.W)
            
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
            def make_click_handler(path, check_var):
                def handler(event):
                    check_var.set(not check_var.get())
                    self.toggle_selection(path, check_var.get())
                return handler
            
            img_label.bind("<Button-1>", make_click_handler(image_path, var))
            
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
        
    def toggle_selection(self, image_path, selected):
        """Toggle selection of an image."""
        if selected:
            self.selected_images.add(image_path)
        else:
            self.selected_images.discard(image_path)
        self.update_selection_label()
        
    def select_all(self):
        """Select all visible images."""
        for image_path in self.displayed_images:
            self.selected_images.add(image_path)
            
        # Update checkboxes
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        child.state(['selected'])
                        
        self.update_selection_label()
        
    def clear_selection(self):
        """Clear all selections."""
        self.selected_images.clear()
        
        # Update checkboxes
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        child.state(['!selected'])
                        
        self.update_selection_label()
        
    def invert_selection(self):
        """Invert current selection."""
        new_selection = set(self.all_images) - self.selected_images
        self.selected_images = new_selection
        
        # Update checkboxes
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                checkbox = None
                image_path = None
                
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Checkbutton):
                        checkbox = child
                    elif isinstance(child, ttk.Label) and len(child.cget('text')) > 10:  # filename label
                        # Find corresponding image path
                        filename = child.cget('text').replace("...", "")
                        for path in self.all_images:
                            if filename in os.path.basename(path):
                                image_path = path
                                break
                                
                if checkbox and image_path:
                    if image_path in self.selected_images:
                        checkbox.state(['selected'])
                    else:
                        checkbox.state(['!selected'])
                        
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
            
        # Show progress
        self.batch_progress['maximum'] = count
        self.batch_progress['value'] = 0
        
        # Process labels in background
        selected_list = list(self.selected_images)
        self.labels_queue.put(('batch_label', selected_list, label))
        
        # Clear selection
        self.clear_selection()
        
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
                for i, image_path in enumerate(image_paths):
                    # Determine if this should be training data
                    is_training_data = label == "crow"
                    
                    # Save label to database
                    add_image_label(image_path, label, is_training_data=is_training_data)
                    logger.info(f"Labeled {os.path.basename(image_path)} as {label}")
                    
                    # Update progress
                    progress = i + 1
                    self.root.after(0, lambda p=progress: self.batch_progress.config(value=p))
                    
                # Update UI when done
                self.root.after(0, self._on_batch_complete)
                
            except Exception as e:
                logger.error(f"Error in batch labeling: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", f"Batch labeling failed: {str(e)}"))
                
        threading.Thread(target=background_process, daemon=True).start()
        
    def _on_batch_complete(self):
        """Called when batch labeling is complete."""
        self.batch_progress['value'] = 0
        messagebox.showinfo("Complete", "Batch labeling completed successfully!")
        
        # Clear selection
        self.selected_images.clear()
        
        # Refresh the current page to show updated status
        self.display_current_page()
        self.update_stats()
        
    def update_stats(self):
        """Update the statistics display."""
        try:
            stats = get_training_data_stats()
            
            if not stats:
                self.stats_label.config(text="No labeled data yet")
                return
                
            crow_count = stats.get('crow', {}).get('count', 0)
            not_crow_count = stats.get('not_a_crow', {}).get('count', 0)
            not_sure_count = stats.get('not_sure', {}).get('count', 0)
            multi_crow_count = stats.get('multi_crow', {}).get('count', 0)
            
            stats_text = f"Labeled: {crow_count} crows, {not_crow_count} not-crows, {not_sure_count} unsure, {multi_crow_count} multi"
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
            self.stats_label.config(text="Error loading stats")

    def apply_filter(self):
        """Apply filter to show only unlabeled images."""
        # Reset to first page when filter changes
        self.current_page = 0
        self.display_current_page()

def main():
    """Main function to run the batch image reviewer."""
    root = tk.Tk()
    app = BatchImageReviewer(root)
    
    # Add help dialog
    help_text = """Batch Crow Image Reviewer

Instructions:
1. Select directory containing crow crop images
2. Click 'Load Images' to display thumbnails (100 per page)
3. Use pagination controls to navigate through pages
4. Click on images or checkboxes to select them
5. Use selection controls:
   - Select All / Clear Selection / Invert Selection
6. Choose label and apply to all selected images

Keyboard Shortcuts:
• 1 - Label selected as Crow
• 2 - Label selected as Not Crow  
• 3 - Label selected as Not Sure
• 4 - Label selected as Multiple Crows
• Ctrl+A - Select All (current page)
• Ctrl+D - Clear Selection
• Ctrl+I - Invert Selection (all images)
• ← → Arrow Keys - Navigate pages
• Page Up/Down - Navigate pages

Features:
• Fast batch processing of multiple images
• Paginated display (100 images per page)
• Visual thumbnail grid with selection
• Automatic progress tracking
• Memory-efficient loading
• Labeled images automatically removed
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