#!/usr/bin/env python3
"""
GUI for applying trained crow classifier model to unlabeled images.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import json
from datetime import datetime
from pathlib import Path
import logging
from PIL import Image, ImageTk

# Import the core functionality
from apply_model_to_unlabeled import CrowClassifier, apply_model_to_unlabeled
from db import get_unlabeled_images, add_image_label

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelApplicationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Crow Classifier - Apply Model to Unlabeled Images")
        self.root.geometry("1400x800")
        
        # State variables
        self.processing = False
        self.results = None
        self.current_predictions = []
        self.current_image_index = -1
        self.current_image_data = None
        
        self.create_widgets()
        self.update_status()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S), padx=(0, 10))
        
        # Middle panel for results
        middle_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="5")
        middle_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for image review
        right_frame = ttk.LabelFrame(main_frame, text="Image Review", padding="5")
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === LEFT PANEL ===
        
        # Model selection
        model_frame = ttk.LabelFrame(left_frame, text="Model Selection", padding="5")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model File:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value="crow_classifier.pth")
        model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=30)
        model_entry.grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).grid(row=0, column=2)
        
        # Processing parameters
        params_frame = ttk.LabelFrame(left_frame, text="Processing Parameters", padding="5")
        params_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(params_frame, text="Display Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.75)
        confidence_scale = ttk.Scale(params_frame, from_=0.5, to=0.99, variable=self.confidence_var, 
                                   orient=tk.HORIZONTAL, length=120)
        confidence_scale.grid(row=0, column=1, padx=5)
        self.confidence_label = ttk.Label(params_frame, text="75%")
        self.confidence_label.grid(row=0, column=2)
        
        ttk.Label(params_frame, text="Auto-label Min:").grid(row=1, column=0, sticky=tk.W)
        self.auto_label_threshold_var = tk.DoubleVar(value=0.90)
        auto_label_scale = ttk.Scale(params_frame, from_=0.5, to=0.99, variable=self.auto_label_threshold_var, 
                                   orient=tk.HORIZONTAL, length=120)
        auto_label_scale.grid(row=1, column=1, padx=5)
        self.auto_label_threshold_label = ttk.Label(params_frame, text="90%")
        self.auto_label_threshold_label.grid(row=1, column=2)
        
        # Update confidence labels when scales change
        confidence_scale.configure(command=self.update_confidence_label)
        auto_label_scale.configure(command=self.update_auto_label_threshold_label)
        
        ttk.Label(params_frame, text="Max Images:").grid(row=2, column=0, sticky=tk.W)
        self.max_images_var = tk.IntVar(value=1000)
        ttk.Entry(params_frame, textvariable=self.max_images_var, width=10).grid(row=2, column=1, padx=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, padx=5, sticky=tk.W)
        
        # Directory selection
        dir_frame = ttk.LabelFrame(left_frame, text="Target Directory", padding="5")
        dir_frame.pack(fill=tk.X, pady=5)
        
        self.use_all_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(dir_frame, text="All unlabeled images", variable=self.use_all_var, 
                       value=True, command=self.toggle_directory).pack(anchor=tk.W)
        ttk.Radiobutton(dir_frame, text="Specific directory:", variable=self.use_all_var, 
                       value=False, command=self.toggle_directory).pack(anchor=tk.W)
        
        self.directory_var = tk.StringVar()
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.directory_var, width=30, state=tk.DISABLED)
        self.dir_entry.pack(fill=tk.X, pady=2)
        self.dir_browse_btn = ttk.Button(dir_frame, text="Browse Directory", 
                                        command=self.browse_directory, state=tk.DISABLED)
        self.dir_browse_btn.pack()
        
        # Processing options
        options_frame = ttk.LabelFrame(left_frame, text="Processing Options", padding="5")
        options_frame.pack(fill=tk.X, pady=5)
        
        self.auto_label_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable auto-labeling", 
                       variable=self.auto_label_var).pack(anchor=tk.W)
        
        # Add explanation label
        explanation_label = ttk.Label(options_frame, 
                                    text="(Only predictions ≥ auto-label threshold will be labeled)",
                                    font=('TkDefaultFont', 8))
        explanation_label.pack(anchor=tk.W, padx=20)
        
        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Dry run (preview only)", 
                       variable=self.dry_run_var).pack(anchor=tk.W)
        
        # Control buttons
        control_frame = ttk.LabelFrame(left_frame, text="Control", padding="5")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_btn.pack(pady=2, fill=tk.X)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(pady=2, fill=tk.X)
        
        # Status and progress
        status_frame = ttk.LabelFrame(left_frame, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        self.stats_label = ttk.Label(status_frame, text="", font=('TkDefaultFont', 9))
        self.stats_label.pack(anchor=tk.W)
        
        # === MIDDLE PANEL ===
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(1, weight=1)
        
        # Summary display
        summary_frame = ttk.Frame(middle_frame)
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        summary_frame.columnconfigure(1, weight=1)
        
        self.summary_text = tk.Text(summary_frame, height=6, width=40)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        self.summary_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        summary_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Predictions table
        table_frame = ttk.LabelFrame(middle_frame, text="Predictions (click to review)", padding="5")
        table_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Create treeview for predictions
        columns = ('filename', 'prediction', 'confidence', 'status')
        self.predictions_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Define headings
        self.predictions_tree.heading('filename', text='Filename')
        self.predictions_tree.heading('prediction', text='Prediction')
        self.predictions_tree.heading('confidence', text='Confidence')
        self.predictions_tree.heading('status', text='Status')
        
        # Configure column widths
        self.predictions_tree.column('filename', width=150)
        self.predictions_tree.column('prediction', width=80)
        self.predictions_tree.column('confidence', width=70)
        self.predictions_tree.column('status', width=100)
        
        # Add scrollbar to treeview
        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.predictions_tree.yview)
        self.predictions_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.predictions_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind double-click event to tree
        self.predictions_tree.bind('<Double-1>', self.on_prediction_select)
        self.predictions_tree.bind('<Button-1>', self.on_prediction_click)
        
        # Buttons for results
        button_frame = ttk.Frame(middle_frame)
        button_frame.grid(row=2, column=0, pady=5)
        
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=2)
        
        # === RIGHT PANEL - IMAGE REVIEW ===
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        # Image display
        image_display_frame = ttk.Frame(right_frame)
        image_display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        image_display_frame.columnconfigure(0, weight=1)
        image_display_frame.rowconfigure(0, weight=1)
        
        # Canvas for image with scrollbars
        self.image_canvas = tk.Canvas(image_display_frame, bg='white', width=300, height=300)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image info
        self.image_info_label = ttk.Label(right_frame, text="Select a prediction to review", 
                                         font=('TkDefaultFont', 10))
        self.image_info_label.grid(row=1, column=0, pady=5)
        
        # Labeling controls
        label_frame = ttk.LabelFrame(right_frame, text="Label Controls", padding="5")
        label_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Current prediction display
        self.current_prediction_label = ttk.Label(label_frame, text="Predicted: --", 
                                                 font=('TkDefaultFont', 10, 'bold'))
        self.current_prediction_label.pack(pady=2)
        
        self.current_confidence_label = ttk.Label(label_frame, text="Confidence: --")
        self.current_confidence_label.pack(pady=2)
        
        # Label buttons
        button_grid = ttk.Frame(label_frame)
        button_grid.pack(pady=5)
        
        self.confirm_btn = ttk.Button(button_grid, text="✓ Confirm", 
                                     command=self.confirm_label, state=tk.DISABLED)
        self.confirm_btn.grid(row=0, column=0, padx=2, pady=2)
        
        ttk.Button(button_grid, text="Crow", 
                  command=lambda: self.set_manual_label("crow")).grid(row=0, column=1, padx=2, pady=2)
        
        ttk.Button(button_grid, text="Not Crow", 
                  command=lambda: self.set_manual_label("not_a_crow")).grid(row=1, column=0, padx=2, pady=2)
        
        ttk.Button(button_grid, text="Multi Crow", 
                  command=lambda: self.set_manual_label("multi_crow")).grid(row=1, column=1, padx=2, pady=2)
        
        ttk.Button(button_grid, text="Bad Crow", 
                  command=lambda: self.set_manual_label("bad_crow")).grid(row=2, column=0, columnspan=2, padx=2, pady=2)
        
        # Navigation
        nav_frame = ttk.Frame(right_frame)
        nav_frame.grid(row=3, column=0, pady=5)
        
        self.prev_btn = ttk.Button(nav_frame, text="◀ Previous", 
                                  command=self.previous_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.next_btn = ttk.Button(nav_frame, text="Next ▶", 
                                  command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=2)
        
        # Status
        self.review_status_label = ttk.Label(right_frame, text="", font=('TkDefaultFont', 9))
        self.review_status_label.grid(row=4, column=0, pady=2)
        
    def update_confidence_label(self, value):
        """Update confidence percentage label"""
        self.confidence_label.config(text=f"{int(float(value)*100)}%")
        
    def update_auto_label_threshold_label(self, value):
        """Update auto-label threshold percentage label"""
        self.auto_label_threshold_label.config(text=f"{int(float(value)*100)}%")
        
    def toggle_directory(self):
        """Toggle directory entry based on selection"""
        if self.use_all_var.get():
            self.dir_entry.config(state=tk.DISABLED)
            self.dir_browse_btn.config(state=tk.DISABLED)
        else:
            self.dir_entry.config(state=tk.NORMAL)
            self.dir_browse_btn.config(state=tk.NORMAL)
            
    def browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("PyTorch models", "*.pth"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.model_path_var.set(filename)
            
    def browse_directory(self):
        """Browse for target directory"""
        directory = filedialog.askdirectory(title="Select Directory to Process")
        if directory:
            self.directory_var.set(directory)
            
    def update_status(self):
        """Update status display with current unlabeled count"""
        try:
            # Get count of unlabeled images
            directory = None if self.use_all_var.get() else self.directory_var.get()
            unlabeled = get_unlabeled_images(limit=10, from_directory=directory)
            count = len(get_unlabeled_images(limit=10000, from_directory=directory))
            
            if directory and not os.path.exists(directory):
                self.stats_label.config(text="Directory not found")
            else:
                self.stats_label.config(text=f"~{count} unlabeled images available")
                
        except Exception as e:
            self.stats_label.config(text=f"Error checking images: {str(e)[:50]}")
            
        # Schedule next update
        self.root.after(5000, self.update_status)  # Update every 5 seconds
        
    def start_processing(self):
        """Start the model application process"""
        # Validate inputs
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
            
        if not self.use_all_var.get():
            directory = self.directory_var.get()
            if not directory or not os.path.exists(directory):
                messagebox.showerror("Error", "Please select a valid directory")
                return
        
        # Update UI state
        self.processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Starting processing...")
        self.progress_var.set(0)
        
        # Clear previous results
        self.clear_results()
        
        # Start processing in background thread
        self.processing_thread = threading.Thread(target=self.run_processing)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def run_processing(self):
        """Run the actual model application in background"""
        try:
            # Get parameters
            model_path = self.model_path_var.get()
            confidence_threshold = self.confidence_var.get()  # Display threshold
            auto_label_threshold = self.auto_label_threshold_var.get()  # Auto-label threshold
            max_images = self.max_images_var.get()
            batch_size = self.batch_size_var.get()
            directory = None if self.use_all_var.get() else self.directory_var.get()
            auto_label = self.auto_label_var.get() and not self.dry_run_var.get()
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(text="Loading model..."))
            
            # Run the processing with the display threshold (for now - we'll implement dual threshold logic later)
            results = apply_model_to_unlabeled(
                model_path=model_path,
                confidence_threshold=auto_label_threshold if auto_label else confidence_threshold,
                auto_label=auto_label,
                max_images=max_images,
                batch_size=batch_size,
                from_directory=directory
            )
            
            # Update UI with results
            if results:
                self.root.after(0, lambda: self.display_results(results, auto_label_threshold))
            else:
                self.root.after(0, lambda: self.status_label.config(text="No results - check logs"))
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
            
        finally:
            self.root.after(0, self.stop_processing)
            
    def display_results(self, results, auto_label_threshold=None):
        """Display processing results in the GUI"""
        self.results = results
        
        # Update summary
        self.summary_text.delete(1.0, tk.END)
        
        summary = f"""Processing Complete!

📊 Summary:
• Total processed: {results['total_processed']} images
• High confidence (≥{results['confidence_threshold']:.0%}): {results['high_confidence_count']}
• Need manual review: {results['uncertain_count']}

🏷️ Predictions:
"""
        
        for label, count in results['predictions_summary'].items():
            summary += f"• {label}: {count}\n"
            
        if results['high_confidence_count'] > 0:
            summary += f"\n🎯 High Confidence:\n"
            for label, count in results['high_confidence_summary'].items():
                summary += f"• {label}: {count}\n"
                
        if self.auto_label_var.get() and not self.dry_run_var.get():
            summary += f"\n✅ Auto-labeled: {results['high_confidence_count']} images"
        elif self.dry_run_var.get():
            summary += f"\n🔍 DRY RUN - No labels were added"
            
        self.summary_text.insert(1.0, summary)
        
        # Store predictions for image review
        self.current_predictions = results['detailed_results']
        
        # Display sample predictions in table
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
            
        # Show all predictions (they can scroll if needed)
        sample_predictions = self.current_predictions
        
        for pred in sample_predictions:
            filename = Path(pred['image_path']).name
            prediction = pred['predicted_label']
            confidence = f"{pred['confidence']:.3f}"
            
            # Determine status
            if pred['confidence'] >= results['confidence_threshold']:
                status = "High confidence" if not self.dry_run_var.get() else "Would auto-label"
            else:
                status = "Needs review"
                
            # Color code rows
            item = self.predictions_tree.insert('', tk.END, values=(filename, prediction, confidence, status))
            
            # Tag for coloring
            if pred['confidence'] >= results['confidence_threshold']:
                self.predictions_tree.set(item, 'status', '✅ ' + status)
            else:
                self.predictions_tree.set(item, 'status', '⚠️ ' + status)
                
        self.status_label.config(text=f"Complete! Processed {results['total_processed']} images")
        self.progress_var.set(100)
        
        # Show completion message
        completion_msg = (
            f"Processing completed!\n\n"
            f"📊 Results:\n"
            f"• Processed: {results['total_processed']} images\n"
            f"• High confidence: {results['high_confidence_count']}\n"
            f"• Need review: {results['uncertain_count']}\n\n"
        )
        
        if self.auto_label_var.get() and not self.dry_run_var.get():
            completion_msg += f"✅ Auto-labeled {results['high_confidence_count']} images!\n\n"
        
        completion_msg += f"📁 Detailed results saved to:\n{results.get('results_file', 'results file')}"
        
        messagebox.showinfo("Processing Complete", completion_msg)
        
    def stop_processing(self):
        """Stop processing and reset UI"""
        self.processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if hasattr(self, 'processing_thread'):
            # Note: Can't safely kill thread, but setting flag will help it exit gracefully
            pass
            
    def export_results(self):
        """Export results to file"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['filename', 'prediction', 'confidence', 'image_path'])
                        for pred in self.results['detailed_results']:
                            writer.writerow([
                                Path(pred['image_path']).name,
                                pred['predicted_label'],
                                pred['confidence'],
                                pred['image_path']
                            ])
                else:
                    with open(filename, 'w') as f:
                        json.dump(self.results, f, indent=2)
                        
                messagebox.showinfo("Success", f"Results exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
                
    def review_uncertain(self):
        """Open batch reviewer for uncertain predictions"""
        if not self.results or self.results['uncertain_count'] == 0:
            messagebox.showinfo("Info", "No uncertain predictions to review")
            return
            
        try:
            # This would ideally open the batch reviewer with filtered uncertain images
            messagebox.showinfo("Review Uncertain", 
                              f"Found {self.results['uncertain_count']} uncertain predictions.\n\n"
                              f"Use the batch_image_reviewer.py with 'Unlabeled Only' filter "
                              f"to review these images manually.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open reviewer: {e}")
            
    def clear_results(self):
        """Clear all results and reset display"""
        self.results = None
        self.current_predictions = []
        self.current_image_index = -1
        self.current_image_data = None
        
        self.summary_text.delete(1.0, tk.END)
        
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
            
        # Clear image display
        self.image_canvas.delete("all")
        self.image_info_label.config(text="Select a prediction to review")
        self.current_prediction_label.config(text="Predicted: --")
        self.current_confidence_label.config(text="Confidence: --")
        self.confirm_btn.config(state=tk.DISABLED)
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.review_status_label.config(text="")
            
        self.progress_var.set(0)
        if not self.processing:
            self.status_label.config(text="Ready")
            
    def on_prediction_click(self, event):
        """Handle single click on prediction"""
        item = self.predictions_tree.selection()[0] if self.predictions_tree.selection() else None
        if item:
            # Get the index of the clicked item
            children = self.predictions_tree.get_children()
            self.current_image_index = children.index(item)
            self.load_current_image()
            
    def on_prediction_select(self, event):
        """Handle double-click on prediction"""
        self.on_prediction_click(event)
        
    def load_current_image(self):
        """Load and display the current image"""
        if not self.current_predictions or self.current_image_index < 0:
            return
            
        try:
            prediction = self.current_predictions[self.current_image_index]
            image_path = prediction['image_path']
            
            # Load image
            pil_image = Image.open(image_path)
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = 300
            canvas_height = 300
            
            # Calculate scaling
            img_width, img_height = pil_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image_data = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.current_image_data)
            
            # Update info labels
            filename = Path(image_path).name
            self.image_info_label.config(text=f"Image: {filename}")
            
            pred_label = prediction['predicted_label']
            confidence = prediction['confidence']
            
            self.current_prediction_label.config(text=f"Predicted: {pred_label}")
            self.current_confidence_label.config(text=f"Confidence: {confidence:.3f}")
            
            # Enable confirm button
            self.confirm_btn.config(state=tk.NORMAL)
            
            # Update navigation buttons
            self.prev_btn.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_image_index < len(self.current_predictions) - 1 else tk.DISABLED)
            
            # Update status
            self.review_status_label.config(text=f"Image {self.current_image_index + 1} of {len(self.current_predictions)}")
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Failed to load image: {e}")
            
    def previous_image(self):
        """Go to previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
            # Update selection in tree
            children = self.predictions_tree.get_children()
            if self.current_image_index < len(children):
                self.predictions_tree.selection_set(children[self.current_image_index])
                self.predictions_tree.see(children[self.current_image_index])
                
    def next_image(self):
        """Go to next image"""
        if self.current_image_index < len(self.current_predictions) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
            # Update selection in tree
            children = self.predictions_tree.get_children()
            if self.current_image_index < len(children):
                self.predictions_tree.selection_set(children[self.current_image_index])
                self.predictions_tree.see(children[self.current_image_index])
                
    def confirm_label(self):
        """Confirm the predicted label"""
        if self.current_image_index < 0 or not self.current_predictions:
            return
            
        prediction = self.current_predictions[self.current_image_index]
        predicted_label = prediction['predicted_label']
        
        self.save_label(predicted_label, f"Confirmed prediction (conf: {prediction['confidence']:.3f})")
        
    def set_manual_label(self, label):
        """Set a manual label for the current image"""
        if self.current_image_index < 0 or not self.current_predictions:
            return
            
        prediction = self.current_predictions[self.current_image_index]
        self.save_label(label, f"Manual override of '{prediction['predicted_label']}' (conf: {prediction['confidence']:.3f})")
        
    def save_label(self, label, notes):
        """Save the label to database"""
        try:
            prediction = self.current_predictions[self.current_image_index]
            image_path = prediction['image_path']
            confidence = prediction['confidence']
            
            # Determine if this should be training data
            is_training_data = label == "crow"  # Only good quality crows are training data
            
            # Save to database
            add_image_label(
                image_path=image_path,
                label=label,
                confidence=confidence,
                reviewer_notes=notes,
                is_training_data=is_training_data
            )
            
            # Update the prediction in our local data
            prediction['user_labeled'] = True
            prediction['final_label'] = label
            
            # Update the tree display
            children = self.predictions_tree.get_children()
            if self.current_image_index < len(children):
                item = children[self.current_image_index]
                filename = Path(image_path).name
                self.predictions_tree.set(item, 'status', f'✅ Labeled: {label}')
                
            # Show success message
            self.review_status_label.config(text=f"✅ Saved: {label}", foreground="green")
            
            # Auto-advance to next image
            self.root.after(1000, self.auto_advance)
            
            logger.info(f"Labeled {Path(image_path).name} as {label}")
            
        except Exception as e:
            logger.error(f"Error saving label: {e}")
            messagebox.showerror("Error", f"Failed to save label: {e}")
            
    def auto_advance(self):
        """Auto-advance to next unlabeled image"""
        # Reset status color
        self.review_status_label.config(foreground="black")
        
        # Find next unlabeled image
        start_index = self.current_image_index + 1
        for i in range(start_index, len(self.current_predictions)):
            if not self.current_predictions[i].get('user_labeled', False):
                self.current_image_index = i
                self.load_current_image()
                
                # Update tree selection
                children = self.predictions_tree.get_children()
                self.predictions_tree.selection_set(children[i])
                self.predictions_tree.see(children[i])
                return
                
        # If no more unlabeled images, stay on current
        self.review_status_label.config(text="All remaining images labeled!")

def main():
    root = tk.Tk()
    app = ModelApplicationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 