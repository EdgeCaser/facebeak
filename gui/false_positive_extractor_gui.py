#!/usr/bin/env python3
"""
GUI for False Positive Extraction Tool

A user-friendly interface for extracting low-confidence detections 
from videos for manual review as potential "not_a_crow" examples.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.extract_false_positive_crops import (
    extract_false_positive_crops,
    process_video_directory,
    validate_extraction_results
)

class FalsePositiveExtractorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("False Positive Extractor - Facebeak")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Variables
        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar(value="potential_not_crow_crops")
        self.recursive_var = tk.BooleanVar(value=False)
        self.min_confidence_var = tk.DoubleVar(value=0.2)
        self.max_confidence_var = tk.DoubleVar(value=0.6)
        self.frame_skip_var = tk.IntVar(value=15)
        self.max_crops_var = tk.IntVar(value=50)
        self.batch_size_var = tk.IntVar(value=10)
        
        # Processing state
        self.is_processing = False
        self.extraction_thread = None
        
        self.setup_ui()
        self.setup_logging()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="False Positive Extractor", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input selection section
        self.create_input_section(main_frame, row=1)
        
        # Output section
        self.create_output_section(main_frame, row=2)
        
        # Parameters section
        self.create_parameters_section(main_frame, row=3)
        
        # Control buttons
        self.create_control_section(main_frame, row=4)
        
        # Progress section
        self.create_progress_section(main_frame, row=5)
        
        # Log section
        self.create_log_section(main_frame, row=6)
        
        # Configure row weights for resizing
        main_frame.rowconfigure(6, weight=1)
        
    def create_input_section(self, parent, row):
        """Create input file/folder selection section."""
        # Input section frame
        input_frame = ttk.LabelFrame(parent, text="Input Selection", padding="10")
        input_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # Input path
        ttk.Label(input_frame, text="Video/Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_path_var, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10)
        )
        
        # Buttons frame
        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=0, column=2, sticky=tk.W)
        
        ttk.Button(buttons_frame, text="Select Video", 
                  command=self.select_video_file).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(buttons_frame, text="Select Folder", 
                  command=self.select_folder).grid(row=0, column=1)
        
        # Recursive search option
        ttk.Checkbutton(input_frame, text="Recursive search (include subfolders)", 
                       variable=self.recursive_var).grid(row=1, column=0, columnspan=3, 
                                                        sticky=tk.W, pady=(10, 0))
        
    def create_output_section(self, parent, row):
        """Create output directory selection section."""
        output_frame = ttk.LabelFrame(parent, text="Output Settings", padding="10")
        output_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        # Output directory
        ttk.Label(output_frame, text="Output Dir:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10)
        )
        ttk.Button(output_frame, text="Browse", 
                  command=self.select_output_folder).grid(row=0, column=2)
        
    def create_parameters_section(self, parent, row):
        """Create parameters configuration section."""
        params_frame = ttk.LabelFrame(parent, text="Extraction Parameters", padding="10")
        params_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create two columns for parameters
        left_frame = ttk.Frame(params_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 20))
        
        right_frame = ttk.Frame(params_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.N))
        
        # Left column parameters
        row_idx = 0
        
        # Confidence range
        ttk.Label(left_frame, text="Min Confidence:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        conf_min_spinbox = ttk.Spinbox(left_frame, from_=0.1, to=0.9, increment=0.1, 
                                      textvariable=self.min_confidence_var, width=10)
        conf_min_spinbox.grid(row=row_idx, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        row_idx += 1
        
        ttk.Label(left_frame, text="Max Confidence:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        conf_max_spinbox = ttk.Spinbox(left_frame, from_=0.2, to=1.0, increment=0.1, 
                                      textvariable=self.max_confidence_var, width=10)
        conf_max_spinbox.grid(row=row_idx, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        row_idx += 1
        
        # Frame skip
        ttk.Label(left_frame, text="Frame Skip:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        skip_spinbox = ttk.Spinbox(left_frame, from_=1, to=60, increment=1, 
                                  textvariable=self.frame_skip_var, width=10)
        skip_spinbox.grid(row=row_idx, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Right column parameters
        row_idx = 0
        
        # Max crops per video
        ttk.Label(right_frame, text="Max Crops/Video:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        crops_spinbox = ttk.Spinbox(right_frame, from_=10, to=500, increment=10, 
                                   textvariable=self.max_crops_var, width=10)
        crops_spinbox.grid(row=row_idx, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        row_idx += 1
        
        # Batch size
        ttk.Label(right_frame, text="Batch Size:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        batch_spinbox = ttk.Spinbox(right_frame, from_=1, to=50, increment=1, 
                                   textvariable=self.batch_size_var, width=10)
        batch_spinbox.grid(row=row_idx, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Help text
        help_text = ("Min/Max Confidence: Range of detection confidence to extract (lower = more false positives)\n"
                    "Frame Skip: Process every Nth frame (higher = faster but may miss detections)\n"
                    "Max Crops/Video: Maximum number of crops to extract per video file\n"
                    "Batch Size: Number of frames to process together (affects memory usage)")
        
        help_label = ttk.Label(params_frame, text=help_text, font=('Arial', 8), 
                              foreground='gray', justify=tk.LEFT)
        help_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_control_section(self, parent, row):
        """Create control buttons section."""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        
        self.extract_button = ttk.Button(control_frame, text="Start Extraction", 
                                        command=self.start_extraction, width=15)
        self.extract_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self.stop_extraction, width=10, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=(0, 10))
        
        self.validate_button = ttk.Button(control_frame, text="Validate Results", 
                                         command=self.validate_results, width=15)
        self.validate_button.grid(row=0, column=2, padx=(0, 10))
        
        self.clear_log_button = ttk.Button(control_frame, text="Clear Log", 
                                          command=self.clear_log, width=10)
        self.clear_log_button.grid(row=0, column=3)
        
    def create_progress_section(self, parent, row):
        """Create progress bar section."""
        progress_frame = ttk.Frame(parent)
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def create_log_section(self, parent, row):
        """Create log display section."""
        log_frame = ttk.LabelFrame(parent, text="Log", padding="5")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def setup_logging(self):
        """Setup logging to display in the GUI."""
        # Create custom handler for GUI
        self.gui_handler = GUILogHandler(self.log_text)
        self.gui_handler.setLevel(logging.INFO)
        
        # Setup formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%H:%M:%S')
        self.gui_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger = logging.getLogger('utilities.extract_false_positive_crops')
        logger.addHandler(self.gui_handler)
        logger.setLevel(logging.INFO)
        
        # Initial log message
        self.log_message("False Positive Extractor GUI started", "INFO")
        
    def log_message(self, message, level="INFO"):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"{timestamp} - {level} - {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared", "INFO")
        
    def select_video_file(self):
        """Select a single video file."""
        filetypes = [
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.flv *.wmv'),
            ('All files', '*.*')
        ]
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        if filename:
            self.input_path_var.set(filename)
            self.recursive_var.set(False)  # Not applicable for single file
            
    def select_folder(self):
        """Select a folder containing videos."""
        folder = filedialog.askdirectory(title="Select Folder with Videos")
        if folder:
            self.input_path_var.set(folder)
            
    def select_output_folder(self):
        """Select output folder for extracted crops."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path_var.set(folder)
            
    def validate_inputs(self):
        """Validate user inputs before starting extraction."""
        input_path = self.input_path_var.get().strip()
        if not input_path:
            messagebox.showerror("Error", "Please select an input video file or folder.")
            return False
            
        if not os.path.exists(input_path):
            messagebox.showerror("Error", f"Input path does not exist: {input_path}")
            return False
            
        output_path = self.output_path_var.get().strip()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output directory.")
            return False
            
        # Validate confidence range
        min_conf = self.min_confidence_var.get()
        max_conf = self.max_confidence_var.get()
        if min_conf >= max_conf:
            messagebox.showerror("Error", "Minimum confidence must be less than maximum confidence.")
            return False
            
        if max_conf > 0.8:
            result = messagebox.askyesno(
                "Warning", 
                f"Maximum confidence is {max_conf:.1f}, which may include many real crows.\n"
                "Continue anyway?"
            )
            if not result:
                return False
                
        return True
        
    def start_extraction(self):
        """Start the extraction process."""
        if not self.validate_inputs():
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Extraction is already in progress.")
            return
            
        # Update UI state
        self.is_processing = True
        self.extract_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set("Starting extraction...")
        self.progress_bar.start()
        
        # Start extraction in separate thread
        self.extraction_thread = threading.Thread(target=self.run_extraction, daemon=True)
        self.extraction_thread.start()
        
    def run_extraction(self):
        """Run the extraction process in a separate thread."""
        try:
            input_path = Path(self.input_path_var.get().strip())
            output_dir = self.output_path_var.get().strip()
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Get parameters
            params = {
                'min_confidence': self.min_confidence_var.get(),
                'max_confidence': self.max_confidence_var.get(),
                'frame_skip': self.frame_skip_var.get(),
                'max_crops_per_video': self.max_crops_var.get(),
                'crop_size': (224, 224),
                'batch_size': self.batch_size_var.get()
            }
            
            self.log_message(f"Starting extraction with parameters: {params}")
            
            total_crops = 0
            
            if input_path.is_file():
                # Process single video
                self.log_message(f"Processing single video: {input_path.name}")
                total_crops = extract_false_positive_crops(
                    input_path,
                    output_dir,
                    **params
                )
            elif input_path.is_dir():
                # Process directory
                self.log_message(f"Processing directory: {input_path}")
                if self.recursive_var.get():
                    self.log_message("Recursive search enabled")
                    total_crops = self.process_directory_recursive(input_path, output_dir, **params)
                else:
                    total_crops = process_video_directory(input_path, output_dir, **params)
            
            # Show results
            self.root.after(0, self.extraction_completed, total_crops, output_dir)
            
        except Exception as e:
            error_msg = f"Error during extraction: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.root.after(0, self.extraction_failed, error_msg)
            
    def process_directory_recursive(self, input_dir, output_dir, **params):
        """Process directory recursively."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        total_crops = 0
        
        for video_file in input_dir.rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                if not self.is_processing:  # Check if stopped
                    break
                    
                try:
                    self.log_message(f"Processing: {video_file.relative_to(input_dir)}")
                    crops = extract_false_positive_crops(video_file, output_dir, **params)
                    total_crops += crops
                    self.log_message(f"Extracted {crops} crops from {video_file.name}")
                except Exception as e:
                    self.log_message(f"Error processing {video_file}: {e}", "ERROR")
                    
        return total_crops
        
    def extraction_completed(self, total_crops, output_dir):
        """Called when extraction completes successfully."""
        self.is_processing = False
        self.extract_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_var.set("Extraction completed")
        
        self.log_message(f"Extraction completed! Total crops extracted: {total_crops}")
        self.log_message(f"Results saved to: {output_dir}")
        
        # Show completion dialog
        messagebox.showinfo(
            "Extraction Complete",
            f"Successfully extracted {total_crops} potential false positive crops.\n\n"
            f"Results saved to: {output_dir}\n\n"
            "Next steps:\n"
            "1. Review crops using the image reviewer\n"
            "2. Label false positives as 'not_a_crow'\n"
            "3. Use labeled data for training"
        )
        
    def extraction_failed(self, error_msg):
        """Called when extraction fails."""
        self.is_processing = False
        self.extract_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_var.set("Extraction failed")
        
        messagebox.showerror("Extraction Failed", error_msg)
        
    def stop_extraction(self):
        """Stop the extraction process."""
        if self.is_processing:
            self.is_processing = False
            self.log_message("Stopping extraction...", "WARNING")
            self.progress_var.set("Stopping...")
            
    def validate_results(self):
        """Validate extraction results in output directory."""
        output_dir = self.output_path_var.get().strip()
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory to validate.")
            return
            
        if not os.path.exists(output_dir):
            messagebox.showwarning("Warning", f"Output directory does not exist: {output_dir}")
            return
            
        self.log_message(f"Validating results in: {output_dir}")
        try:
            validate_extraction_results(output_dir)
            self.log_message("Validation completed")
        except Exception as e:
            self.log_message(f"Error during validation: {e}", "ERROR")
            
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

class GUILogHandler(logging.Handler):
    """Custom logging handler to display logs in GUI."""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        """Emit a log record to the text widget."""
        try:
            msg = self.format(record)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.update_idletasks()
        except Exception:
            pass  # Ignore errors in logging

def main():
    """Main entry point."""
    app = FalsePositiveExtractorGUI()
    app.run()

if __name__ == "__main__":
    main() 