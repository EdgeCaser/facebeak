#!/usr/bin/env python3
"""
GUI for extracting hard negative crops (non-bird objects) from videos.
This tool provides an easy-to-use interface for sampling random crops while avoiding bird areas.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from pathlib import Path
import logging
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utilities.extract_hard_negatives import process_video_file, add_crops_to_database

class HardNegativesExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hard Negatives Extractor")
        self.root.geometry("700x600")
        
        # Configure logging to capture to GUI
        self.setup_logging()
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="crow_crops/hard_negatives")
        self.max_crops_per_video = tk.IntVar(value=200)
        self.frame_skip = tk.IntVar(value=30)
        self.processing = False
        
        self.create_widgets()
        
    def setup_logging(self):
        """Setup logging to capture output in GUI."""
        self.log_handler = GUILogHandler(self)
        self.log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.log_handler.setFormatter(formatter)
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)
        
    def create_widgets(self):
        """Create and layout GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="Hard Negatives Extractor", 
                               font=("TkDefaultFont", 16, "bold"))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # Input path selection
        ttk.Label(main_frame, text="Input Path:").grid(row=row, column=0, sticky=tk.W, pady=5)
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(input_frame, text="Browse Video", command=self.browse_video).grid(row=0, column=1, padx=(5, 0))
        ttk.Button(input_frame, text="Browse Folder", command=self.browse_folder).grid(row=0, column=2, padx=(5, 0))
        row += 1
        
        # Output directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=row, column=0, sticky=tk.W, pady=5)
        output_frame = ttk.Frame(main_frame)
        output_frame.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=1, padx=(5, 0))
        row += 1
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Extraction Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        params_frame.columnconfigure(1, weight=1)
        row += 1
        
        param_row = 0
        
        # Max crops per video
        ttk.Label(params_frame, text="Max crops per video:").grid(row=param_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(params_frame, from_=50, to=1000, textvariable=self.max_crops_per_video, width=10).grid(
            row=param_row, column=1, sticky=tk.W, pady=2)
        param_row += 1
        
        # Frame skip
        ttk.Label(params_frame, text="Frame skip:").grid(row=param_row, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(params_frame, from_=1, to=300, textvariable=self.frame_skip, width=10).grid(
            row=param_row, column=1, sticky=tk.W, pady=2)
        param_row += 1
        
        # Crop size info (fixed)
        ttk.Label(params_frame, text="Crop size:").grid(row=param_row, column=0, sticky=tk.W, pady=2)
        ttk.Label(params_frame, text="512x512 (fixed)", font=('Arial', 9, 'italic')).grid(
            row=param_row, column=1, sticky=tk.W, pady=2)
        param_row += 1
        
        # Database and extraction options
        db_frame = ttk.LabelFrame(main_frame, text="Database & Labeling", padding="10")
        db_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Add to database checkbox
        self.add_to_db_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(db_frame, text="Add to database", 
                       variable=self.add_to_db_var).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Label (fixed to prevent user errors)
        ttk.Label(db_frame, text="Label:").grid(row=1, column=0, sticky=tk.W, pady=(5,0))
        ttk.Label(db_frame, text="not_a_crow", 
                 font=('TkDefaultFont', 9, 'bold'),
                 foreground='red').grid(row=1, column=1, sticky=tk.W, padx=(5,0), pady=(5,0))
        
        # Confidence
        ttk.Label(db_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, pady=(5,0))
        self.confidence_var = tk.DoubleVar(value=0.9)
        confidence_scale = ttk.Scale(db_frame, from_=0.1, to=1.0, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5,0), pady=(5,0))
        self.confidence_label = ttk.Label(db_frame, text="0.90")
        self.confidence_label.grid(row=2, column=2, padx=(5,0), pady=(5,0))
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        row += 1
        
        self.start_button = ttk.Button(button_frame, text="Start Extraction", 
                                      command=self.start_extraction, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                     command=self.stop_extraction, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        row += 1
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1
        
        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="5")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(row, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def browse_video(self):
        """Browse for a single video file."""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MOV"),
                      ("All files", "*.*")]
        )
        if filename:
            self.input_path.set(filename)
            
    def browse_folder(self):
        """Browse for a folder containing videos."""
        dirname = filedialog.askdirectory(title="Select Folder with Videos")
        if dirname:
            self.input_path.set(dirname)
            
    def browse_output_dir(self):
        """Browse for output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            
    def log_message(self, message):
        """Add message to log output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """Clear the log output."""
        self.log_text.delete(1.0, tk.END)
        
    def update_status(self, status):
        """Update status label."""
        self.status_label.config(text=status)
        self.root.update_idletasks()
        
    def start_extraction(self):
        """Start the extraction process in a separate thread."""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input path")
            return
            
        if self.processing:
            return
            
        self.processing = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress.start()
        self.update_status("Processing...")
        
        # Start extraction in separate thread
        self.extraction_thread = threading.Thread(target=self.run_extraction)
        self.extraction_thread.daemon = True
        self.extraction_thread.start()
        
    def stop_extraction(self):
        """Stop the extraction process."""
        self.processing = False
        self.update_status("Stopping...")
        
    def run_extraction(self):
        """Run the extraction process."""
        try:
            input_path = Path(self.input_path.get())
            output_dir = Path(self.output_dir.get())
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            total_extracted = 0
            
            if input_path.is_file():
                # Process single video
                video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MOV"]
                if input_path.suffix.lower() in [ext.lower() for ext in video_extensions]:
                    total_extracted = process_video_file(
                        str(input_path), str(output_dir),
                        self.max_crops_per_video.get(), self.frame_skip.get()
                    )
                else:
                    self.log_message(f"Error: {input_path} is not a supported video format")
                    return
                    
            elif input_path.is_dir():
                # Process all videos in directory
                video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MOV"]
                video_files = []
                for ext in video_extensions:
                    video_files.extend(input_path.rglob(f"*{ext}"))
                    video_files.extend(input_path.rglob(f"*{ext.upper()}"))
                
                self.log_message(f"Found {len(video_files)} video files to process")
                
                for i, video_file in enumerate(video_files):
                    if not self.processing:
                        break
                        
                    self.update_status(f"Processing video {i+1}/{len(video_files)}")
                    extracted = process_video_file(
                        str(video_file), str(output_dir),
                        self.max_crops_per_video.get(), self.frame_skip.get()
                    )
                    total_extracted += extracted
            else:
                self.log_message(f"Error: Input path {input_path} does not exist")
                return
                
            self.log_message(f"Total extracted crops: {total_extracted}")
            
            # Add to database if requested
            if self.add_to_db_var.get() and total_extracted > 0:
                self.update_status("Adding to database...")
                added_count = add_crops_to_database(str(output_dir), "not_a_crow", self.confidence_var.get())
                self.log_message(f"Database updated with {added_count} new entries")
                
            self.update_status("Completed!")
            messagebox.showinfo("Success", f"Extraction completed!\nExtracted {total_extracted} crops")
            
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.processing = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.progress.stop()
            self.update_status("Ready")

    def update_confidence_label(self, event):
        """Update the confidence label when the confidence scale is changed."""
        self.confidence_label.config(text=f"{self.confidence_var.get():.2f}")

class GUILogHandler(logging.Handler):
    """Custom log handler to display logs in GUI."""
    
    def __init__(self, gui):
        super().__init__()
        self.gui = gui
        
    def emit(self, record):
        try:
            msg = self.format(record)
            # Schedule GUI update in main thread
            self.gui.root.after(0, lambda: self.gui.log_message(msg))
        except Exception:
            self.handleError(record)

def main():
    root = tk.Tk()
    
    # Try to set a more modern theme
    try:
        style = ttk.Style()
        style.theme_use('clam')
    except:
        pass
    
    app = HardNegativesExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 