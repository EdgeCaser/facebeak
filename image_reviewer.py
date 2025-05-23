import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageTk
import cv2
import numpy as np
from db import get_unlabeled_images, add_image_label, get_training_data_stats
import logging
from pathlib import Path
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Crow Image Reviewer")
        self.root.geometry("1200x800")
        
        # Initialize state
        self.current_images = []
        self.current_index = 0
        self.labels_queue = queue.Queue()
        self.processing = False
        
        # Create main layout
        self.create_layout()
        
        # Load initial images
        self.load_images()
        
        # Start processing queue
        self.process_queue()
        
    def create_layout(self):
        """Create the main GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel for controls
        control_panel = ttk.Frame(main_frame, padding="5")
        control_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title and instructions
        title_label = ttk.Label(control_panel, text="Crow Image Reviewer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        instructions = ttk.Label(control_panel, 
                                text="Review images and confirm whether they contain crows.\n"
                                     "Images marked as 'Not a crow' will be excluded from training.",
                                wraplength=300, justify=tk.LEFT)
        instructions.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Directory selection
        dir_frame = ttk.LabelFrame(control_panel, text="Image Directory", padding="5")
        dir_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.dir_var = tk.StringVar(value="crow_crops")
        ttk.Entry(dir_frame, textvariable=self.dir_var, width=30).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=1)
        ttk.Button(dir_frame, text="Reload", command=self.load_images).grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        # Progress info
        progress_frame = ttk.LabelFrame(control_panel, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="No images loaded")
        self.progress_label.grid(row=0, column=0, columnspan=2)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_panel)
        nav_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Button(nav_frame, text="← Previous", command=self.previous_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(nav_frame, text="Next →", command=self.next_image).grid(row=0, column=1)
        
        # Labeling buttons
        label_frame = ttk.LabelFrame(control_panel, text="Label Current Image", padding="5")
        label_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Radio buttons for labeling
        self.label_var = tk.StringVar(value="crow")
        ttk.Radiobutton(label_frame, text="Crow", variable=self.label_var, 
                       value="crow").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(label_frame, text="Not a crow", variable=self.label_var, 
                       value="not_a_crow").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(label_frame, text="Not sure", variable=self.label_var, 
                       value="not_sure").grid(row=2, column=0, sticky=tk.W)
        
        # Submit button
        ttk.Button(label_frame, text="Submit Label", 
                  command=self.submit_label).grid(row=3, column=0, pady=(10, 0))
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_panel, text="Statistics", padding="5")
        stats_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="Loading statistics...")
        self.stats_label.grid(row=0, column=0)
        
        # Update stats button
        ttk.Button(stats_frame, text="Refresh Stats", 
                  command=self.update_stats).grid(row=1, column=0, pady=(5, 0))
        
        # Right panel for image display
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(1, weight=1)
        
        # Image info
        self.image_info_label = ttk.Label(image_frame, text="No image loaded", 
                                         font=("Arial", 12))
        self.image_info_label.grid(row=0, column=0, pady=(0, 10))
        
        # Image display canvas
        self.canvas = tk.Canvas(image_frame, bg='black', width=600, height=600)
        self.canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind keyboard shortcuts
        self.root.bind('<Key-1>', lambda e: self.set_label_and_submit("crow"))
        self.root.bind('<Key-2>', lambda e: self.set_label_and_submit("not_a_crow"))
        self.root.bind('<Key-3>', lambda e: self.set_label_and_submit("not_sure"))
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Return>', lambda e: self.submit_label())
        
        # Make window focusable for keyboard shortcuts
        self.root.focus_set()
        
        # Update initial stats
        self.update_stats()
        
    def browse_directory(self):
        """Browse for image directory."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.dir_var.set(directory)
            
    def load_images(self):
        """Load unlabeled images from the specified directory."""
        try:
            directory = self.dir_var.get()
            if not directory or not os.path.exists(directory):
                messagebox.showerror("Error", "Please select a valid directory")
                return
                
            # Get unlabeled images
            self.current_images = get_unlabeled_images(limit=100, from_directory=directory)
            
            if not self.current_images:
                messagebox.showinfo("Info", "No unlabeled images found in the selected directory")
                self.progress_label.config(text="No images to review")
                return
                
            self.current_index = 0
            self.progress_bar['maximum'] = len(self.current_images)
            self.update_display()
            
            logger.info(f"Loaded {len(self.current_images)} unlabeled images")
            
        except Exception as e:
            logger.error(f"Error loading images: {e}")
            messagebox.showerror("Error", f"Failed to load images: {str(e)}")
            
    def update_display(self):
        """Update the image display and progress."""
        if not self.current_images:
            return
            
        if self.current_index >= len(self.current_images):
            # All images reviewed
            messagebox.showinfo("Complete", "All images have been reviewed!")
            self.load_images()  # Reload to get more images
            return
            
        # Update progress
        progress_text = f"Image {self.current_index + 1} of {len(self.current_images)}"
        self.progress_label.config(text=progress_text)
        self.progress_bar['value'] = self.current_index + 1
        
        # Load and display current image
        image_path = self.current_images[self.current_index]
        self.display_image(image_path)
        
        # Update image info
        filename = os.path.basename(image_path)
        self.image_info_label.config(text=f"File: {filename}")
        
        # Reset label selection
        self.label_var.set("crow")
        
    def display_image(self, image_path):
        """Display an image on the canvas."""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Calculate size to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width() or 600
            canvas_height = self.canvas.winfo_height() or 600
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Calculate scaling factor
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            # Resize image
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            x = canvas_width // 2
            y = canvas_height // 2
            self.canvas.create_image(x, y, image=self.photo, anchor=tk.CENTER)
            
        except Exception as e:
            logger.error(f"Error displaying image {image_path}: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(300, 300, text=f"Error loading image:\n{str(e)}", 
                                  fill="white", font=("Arial", 12))
            
    def previous_image(self):
        """Go to previous image."""
        if self.current_images and self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            
    def next_image(self):
        """Go to next image."""
        if self.current_images and self.current_index < len(self.current_images) - 1:
            self.current_index += 1
            self.update_display()
        elif self.current_images and self.current_index >= len(self.current_images) - 1:
            # At the end, try to load more images
            self.load_images()
            
    def set_label_and_submit(self, label):
        """Set label and submit immediately."""
        self.label_var.set(label)
        self.submit_label()
        
    def submit_label(self):
        """Submit the current label."""
        if not self.current_images or self.current_index >= len(self.current_images):
            return
            
        image_path = self.current_images[self.current_index]
        label = self.label_var.get()
        
        # Add to queue for background processing
        self.labels_queue.put((image_path, label))
        
        # Move to next image
        self.current_index += 1
        self.update_display()
        
    def process_queue(self):
        """Process the labeling queue in background."""
        try:
            while not self.labels_queue.empty():
                image_path, label = self.labels_queue.get_nowait()
                
                # Determine if this should be training data
                is_training_data = label == "crow"
                
                # Save label to database
                add_image_label(image_path, label, is_training_data=is_training_data)
                logger.info(f"Labeled {os.path.basename(image_path)} as {label}")
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing label queue: {e}")
            
        # Schedule next queue processing
        self.root.after(1000, self.process_queue)
        
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
            total_labeled = stats.get('total_labeled', 0)
            total_excluded = stats.get('total_excluded', 0)
            
            stats_text = f"""Labeled Images:
• Crows: {crow_count}
• Not crows: {not_crow_count}
• Not sure: {not_sure_count}
• Total training data: {total_labeled}
• Excluded from training: {total_excluded}"""
            
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
            self.stats_label.config(text="Error loading stats")

def main():
    """Main function to run the image reviewer."""
    root = tk.Tk()
    app = ImageReviewer(root)
    
    # Add keyboard shortcut help
    help_text = """Keyboard Shortcuts:
1 - Mark as Crow
2 - Mark as Not a Crow  
3 - Mark as Not Sure
← → - Navigate images
Enter - Submit current label"""
    
    def show_help():
        messagebox.showinfo("Keyboard Shortcuts", help_text)
    
    # Add help menu
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    help_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Keyboard Shortcuts", command=show_help)
    
    root.mainloop()

if __name__ == "__main__":
    main() 