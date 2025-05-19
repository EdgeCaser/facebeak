import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
from typing import List
import os
import sys
import pkg_resources
import subprocess
from db import clear_database

def ensure_requirements():
    """Ensure all required packages are installed."""
    try:
        # Try to import cryptography to check if it's installed
        import cryptography
    except ImportError:
        # If not installed, install requirements
        try:
            requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to install required packages: {e}")
            sys.exit(1)

def get_venv_python():
    """Get the path to the Python interpreter."""
    return sys.executable

class FacebeakGUI:
    def __init__(self, root):
        self.root = root
        root.title("facebeak Launcher")
        root.geometry("900x1100")  # Increased window height for more vertical space

        # Configure grid weights for resizing
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(11, weight=1)  # Make the output box row expandable
        root.grid_rowconfigure(13, weight=2)  # Make the output box row even more expandable
        root.grid_rowconfigure(14, weight=0)  # Database section (no expansion)

        # Create main frame for controls
        control_frame = ttk.Frame(root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")

        # Video list frame
        video_list_frame = ttk.LabelFrame(control_frame, text="Selected Videos", padding="5")
        video_list_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)
        video_list_frame.grid_columnconfigure(0, weight=1)
        video_list_frame.grid_rowconfigure(0, weight=1)

        # Video list with scrollbar
        self.video_listbox = tk.Listbox(video_list_frame, height=5, selectmode=tk.EXTENDED)
        self.video_listbox.grid(row=0, column=0, sticky="ew")
        video_scrollbar = ttk.Scrollbar(video_list_frame, orient="vertical", command=self.video_listbox.yview)
        video_scrollbar.grid(row=0, column=1, sticky="ns")
        self.video_listbox.configure(yscrollcommand=video_scrollbar.set)

        # Video list buttons
        video_buttons_frame = ttk.Frame(video_list_frame)
        video_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        ttk.Button(video_buttons_frame, text="Add Videos", command=self.browse_videos).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_buttons_frame, text="Remove Selected", command=self.remove_selected_videos).pack(side=tk.LEFT, padx=2)
        ttk.Button(video_buttons_frame, text="Clear All", command=self.clear_videos).pack(side=tk.LEFT, padx=2)

        # Output directory
        ttk.Label(control_frame, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=2)
        self.output_dir_entry = ttk.Entry(control_frame, width=50)
        self.output_dir_entry.grid(row=1, column=1, sticky="ew", padx=5)
        self.output_dir_entry.insert(0, "output")
        ttk.Button(control_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5)

        # Detection threshold
        ttk.Label(control_frame, text="Detection Threshold (0.0-1.0):").grid(row=2, column=0, sticky="w", pady=2)
        self.det_thresh = ttk.Entry(control_frame, width=10)
        self.det_thresh.grid(row=2, column=1, sticky="w", padx=5)
        self.det_thresh.insert(0, "0.3")
        ttk.Label(control_frame, text="Lower = more sensitive to detecting birds", font=("Arial", 8)).grid(row=2, column=2, sticky="w")

        # YOLO threshold
        ttk.Label(control_frame, text="YOLO Threshold (0.0-1.0):").grid(row=3, column=0, sticky="w", pady=2)
        self.yolo_thresh = ttk.Entry(control_frame, width=10)
        self.yolo_thresh.grid(row=3, column=1, sticky="w", padx=5)
        self.yolo_thresh.insert(0, "0.2")
        ttk.Label(control_frame, text="Lower = more initial detections", font=("Arial", 8)).grid(row=3, column=2, sticky="w")

        # Max age
        ttk.Label(control_frame, text="Max Age (frames):").grid(row=4, column=0, sticky="w", pady=2)
        self.max_age = ttk.Entry(control_frame, width=10)
        self.max_age.grid(row=4, column=1, sticky="w", padx=5)
        self.max_age.insert(0, "5")
        ttk.Label(control_frame, text="How many frames a crow can disappear before losing its ID.", font=("Arial", 8)).grid(row=4, column=2, sticky="w")

        # Min hits
        ttk.Label(control_frame, text="Min Hits:").grid(row=5, column=0, sticky="w", pady=2)
        self.min_hits = ttk.Entry(control_frame, width=10)
        self.min_hits.grid(row=5, column=1, sticky="w", padx=5)
        self.min_hits.insert(0, "2")
        ttk.Label(control_frame, text="Minimum detections to start tracking", font=("Arial", 8)).grid(row=5, column=2, sticky="w")

        # IOU threshold
        ttk.Label(control_frame, text="IOU Threshold (0.0-1.0):").grid(row=6, column=0, sticky="w", pady=2)
        self.iou_thresh = ttk.Entry(control_frame, width=10)
        self.iou_thresh.grid(row=6, column=1, sticky="w", padx=5)
        self.iou_thresh.insert(0, "0.2")
        ttk.Label(control_frame, text="Lower = more lenient tracking", font=("Arial", 8)).grid(row=6, column=2, sticky="w")

        # Embedding threshold
        ttk.Label(control_frame, text="Embedding Threshold (0.0-1.0):").grid(row=7, column=0, sticky="w", pady=2)
        self.embed_thresh = ttk.Entry(control_frame, width=10)
        self.embed_thresh.grid(row=7, column=1, sticky="w", padx=5)
        self.embed_thresh.insert(0, "0.7")
        ttk.Label(control_frame, text="Higher = stricter visual matching", font=("Arial", 8)).grid(row=7, column=2, sticky="w")

        # Skip
        ttk.Label(control_frame, text="Frame Skip:").grid(row=8, column=0, sticky="w", pady=2)
        self.skip_entry = ttk.Entry(control_frame, width=10)
        self.skip_entry.grid(row=8, column=1, sticky="w", padx=5)
        self.skip_entry.insert(0, "5")
        ttk.Label(control_frame, text="Higher = faster processing but might miss quick movements", font=("Arial", 8)).grid(row=8, column=2, sticky="w")

        # Multi-View Stride
        ttk.Label(control_frame, text="Multi-View Stride:").grid(row=9, column=0, sticky="w", pady=2)
        self.mv_stride_entry = ttk.Entry(control_frame, width=10)
        self.mv_stride_entry.grid(row=9, column=1, sticky="w", padx=5)
        self.mv_stride_entry.insert(0, "1")
        ttk.Label(control_frame, text="How often to run multi-view extraction (1 = every frame, 2 = every other frame, etc.). Use 1 unless you want to speed up processing by skipping some frames for multi-view analysis.", wraplength=400, font=("Arial", 8)).grid(row=9, column=2, sticky="w")

        # Multi-view checkboxes
        self.mv_yolo_var = tk.BooleanVar(value=False)
        self.mv_rcnn_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Enable Multi-View for YOLO", variable=self.mv_yolo_var).grid(row=10, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(control_frame, text="Enable Multi-View for Faster R-CNN", variable=self.mv_rcnn_var).grid(row=10, column=2, columnspan=1, sticky="w", pady=2)

        # Run button (moved to class variable for access in other methods)
        self.run_button = ttk.Button(control_frame, text="Process Videos", command=self.run_facebeak)
        self.run_button.grid(row=11, column=0, columnspan=3, pady=10)

        # Add clustering section AFTER processing button
        clustering_frame = ttk.LabelFrame(control_frame, text="Crow Clustering", padding="5")
        clustering_frame.grid(row=12, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Clustering parameters
        ttk.Label(clustering_frame, text="Eps Range:").grid(row=0, column=0, sticky="w", pady=2)
        eps_frame = ttk.Frame(clustering_frame)
        eps_frame.grid(row=0, column=1, sticky="w", padx=5)
        self.eps_min = ttk.Entry(eps_frame, width=6)
        self.eps_min.pack(side=tk.LEFT, padx=2)
        self.eps_min.insert(0, "0.2")
        ttk.Label(eps_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.eps_max = ttk.Entry(eps_frame, width=6)
        self.eps_max.pack(side=tk.LEFT, padx=2)
        self.eps_max.insert(0, "0.5")
        ttk.Label(clustering_frame, text="DBSCAN eps parameter range for clustering", font=("Arial", 8)).grid(row=0, column=2, sticky="w")
        
        ttk.Label(clustering_frame, text="Min Samples Range:").grid(row=1, column=0, sticky="w", pady=2)
        samples_frame = ttk.Frame(clustering_frame)
        samples_frame.grid(row=1, column=1, sticky="w", padx=5)
        self.min_samples_min = ttk.Entry(samples_frame, width=6)
        self.min_samples_min.pack(side=tk.LEFT, padx=2)
        self.min_samples_min.insert(0, "2")
        ttk.Label(samples_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.min_samples_max = ttk.Entry(samples_frame, width=6)
        self.min_samples_max.pack(side=tk.LEFT, padx=2)
        self.min_samples_max.insert(0, "5")
        ttk.Label(clustering_frame, text="DBSCAN min_samples parameter range", font=("Arial", 8)).grid(row=1, column=2, sticky="w")
        
        # Clustering button
        self.cluster_button = ttk.Button(clustering_frame, text="Run Clustering", command=self.run_clustering)
        self.cluster_button.grid(row=2, column=0, columnspan=3, pady=5)

        # Output box with scrollbar (made taller)
        output_frame = ttk.Frame(root)
        output_frame.grid(row=13, column=0, sticky="nsew", padx=10, pady=5)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Output box with scrollbar (increased height)
        self.output_box = tk.Text(output_frame, height=20, width=70, yscrollcommand=scrollbar.set)
        self.output_box.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.output_box.yview)

        # Add database management section at the bottom
        db_frame = ttk.LabelFrame(root, text="Database Management", padding="5")
        db_frame.grid(row=14, column=0, sticky="ew", padx=10, pady=5)
        
        # Add warning label
        warning_label = ttk.Label(db_frame, 
            text="WARNING: Clearing the database will permanently delete all crow data.\nThis includes all crow identities, embeddings, and sighting history.",
            wraplength=400,
            foreground='red',
            font=('Arial', 9, 'bold'))
        warning_label.pack(pady=5)
        
        # Add clear database button with danger style
        style = ttk.Style()
        style.configure('Danger.TButton', foreground='red', font=('Arial', 9, 'bold'))
        ttk.Button(db_frame, text="Clear Database", command=self.clear_db, style='Danger.TButton').pack(pady=5)

        # Configure style
        style = ttk.Style()
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("TButton", padding=5)

    def browse_videos(self):
        filenames = filedialog.askopenfilenames(
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")],
            title="Select Videos"
        )
        for filename in filenames:
            if filename not in self.video_listbox.get(0, tk.END):
                self.video_listbox.insert(tk.END, filename)

    def remove_selected_videos(self):
        selected = self.video_listbox.curselection()
        for index in reversed(selected):
            self.video_listbox.delete(index)

    def clear_videos(self):
        self.video_listbox.delete(0, tk.END)

    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dirname)

    def run_facebeak(self):
        videos = list(self.video_listbox.get(0, tk.END))
        if not videos:
            messagebox.showerror("Error", "Please select at least one video to process")
            return

        output_dir = self.output_dir_entry.get()
        det_thresh = self.det_thresh.get()
        yolo_thresh = self.yolo_thresh.get()
        max_age = self.max_age.get()
        min_hits = self.min_hits.get()
        iou_thresh = self.iou_thresh.get()
        embed_thresh = self.embed_thresh.get()
        skip = self.skip_entry.get()
        mv_stride = self.mv_stride_entry.get()
        mv_yolo = self.mv_yolo_var.get()
        mv_rcnn = self.mv_rcnn_var.get()
        
        # Validate thresholds
        try:
            det_thresh_float = float(det_thresh)
            yolo_thresh_float = float(yolo_thresh)
            iou_thresh_float = float(iou_thresh)
            embed_thresh_float = float(embed_thresh)
            max_age_int = int(max_age)
            min_hits_int = int(min_hits)
            skip_int = int(skip)
            mv_stride_int = int(mv_stride)
            
            if not (0 <= det_thresh_float <= 1 and 0 <= yolo_thresh_float <= 1 and 
                    0 <= iou_thresh_float <= 1 and 0 <= embed_thresh_float <= 1):
                messagebox.showerror("Invalid Input", "Thresholds must be between 0.0 and 1.0")
                return
            if max_age_int < 1 or min_hits_int < 1 or skip_int < 0 or mv_stride_int < 1:
                messagebox.showerror("Invalid Input", "Max age and min hits must be positive integers; frame skip must be zero or positive; multi-view stride must be >= 1.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for all parameters")
            return

        # Disable run button while processing
        self.run_button.configure(state='disabled')
        
        # Start processing in a separate thread
        threading.Thread(target=self._process_videos, args=(videos, output_dir, {
            'det_thresh': det_thresh,
            'yolo_thresh': yolo_thresh,
            'max_age': max_age,
            'min_hits': min_hits,
            'iou_thresh': iou_thresh,
            'embed_thresh': embed_thresh,
            'skip': skip,
            'mv_stride': mv_stride,
            'mv_yolo': mv_yolo,
            'mv_rcnn': mv_rcnn
        })).start()

    def _process_videos(self, videos: List[str], output_dir: str, params: dict):
        try:
            python_exe = get_venv_python()
            for video in videos:
                # Generate output filenames based on input video name
                video_name = os.path.splitext(os.path.basename(video))[0]
                skip_output = os.path.join(output_dir, f"{video_name}_skip.mp4")  # Frame-skipped detection video
                full_output = os.path.join(output_dir, f"{video_name}_full.mp4")  # Full-frame interpolated video
                
                cmd = [python_exe, "main.py", 
                       "--video", video, 
                       "--skip-output", skip_output,  # New argument for skip-frame output
                       "--full-output", full_output,  # New argument for full-frame output
                       "--detection-threshold", params['det_thresh'],
                       "--yolo-threshold", params['yolo_thresh'],
                       "--max-age", params['max_age'],
                       "--min-hits", params['min_hits'],
                       "--iou-threshold", params['iou_thresh'],
                       "--embedding-threshold", params['embed_thresh'],
                       "--skip", params['skip'],
                       "--multi-view-stride", params['mv_stride']]
                if params.get('mv_yolo'):
                    cmd.append('--multi-view-yolo')
                if params.get('mv_rcnn'):
                    cmd.append('--multi-view-rcnn')
                cmd.append("--preserve-audio")  # New argument to preserve audio
                
                self.output_box.insert(tk.END, f"\nProcessing: {video}\n")
                self.output_box.insert(tk.END, f"Running: {' '.join(cmd)}\n")
                self.output_box.see(tk.END)
                
                # Use Popen with real-time output capture
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )
                
                # Read output in real-time
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        # Update GUI in thread-safe way
                        self.root.after(0, lambda l=line: self._update_output(l))
                
                # Get final return code
                return_code = process.wait()
                
                if return_code == 0:
                    self.root.after(0, lambda: self._update_output(f"\n[Successfully processed: {video}]\n"))
                    self.root.after(0, lambda: self._update_output(f"Generated videos:\n"))
                    self.root.after(0, lambda: self._update_output(f"- Skip-frame detection: {skip_output}\n"))
                    self.root.after(0, lambda: self._update_output(f"- Full-frame interpolated: {full_output}\n"))
                else:
                    self.root.after(0, lambda: self._update_output(f"\n[Error processing: {video}]\n"))
                self.root.after(0, lambda: self.output_box.see(tk.END))
            
            self.root.after(0, lambda: self._update_output("\n[All videos processed]\n"))
            self.root.after(0, lambda: self.output_box.see(tk.END))
            
        finally:
            # Re-enable run button
            self.root.after(0, lambda: self.run_button.configure(state='normal'))
    
    def _update_output(self, text):
        """Thread-safe method to update the output box."""
        self.output_box.insert(tk.END, text)
        self.output_box.see(tk.END)

    def clear_db(self):
        """Clear all data from the database with two-step confirmation."""
        # First confirmation with detailed warning
        if not messagebox.askyesno("Clear Database - First Warning", 
                                 "WARNING: You are about to delete ALL data from the database.\n\n"
                                 "This will permanently delete:\n"
                                 "- All crow identities\n"
                                 "- All crow embeddings\n"
                                 "- All sighting history\n"
                                 "- All tracking data\n\n"
                                 "This action CANNOT be undone.\n\n"
                                 "Are you absolutely sure you want to proceed?"):
            return
            
        # Second confirmation with different wording
        if not messagebox.askyesno("Clear Database - Second Warning", 
                                 "FINAL WARNING: You are about to permanently delete ALL data.\n\n"
                                 "This will reset the database to its initial state.\n"
                                 "All crow tracking history will be lost.\n\n"
                                 "Are you completely certain you want to continue?"):
            return
            
        # Show a text input dialog for the final confirmation
        dialog = tk.Toplevel(self.root)
        dialog.title("Final Confirmation Required")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Add warning text
        warning_frame = ttk.Frame(dialog, padding="10")
        warning_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(warning_frame, 
                 text="FINAL CONFIRMATION REQUIRED",
                 font=('Arial', 12, 'bold'),
                 foreground='red').pack(pady=(0, 10))
        
        ttk.Label(warning_frame,
                 text="To confirm deletion of ALL database data,\nplease type 'DELETE ALL DATA' exactly as shown:",
                 wraplength=350).pack(pady=5)
        
        entry = ttk.Entry(warning_frame, width=30)
        entry.pack(pady=10)
        
        def on_confirm():
            if entry.get().strip() == 'DELETE ALL DATA':
                dialog.destroy()
                try:
                    if clear_database():
                        messagebox.showinfo("Success", "Database cleared successfully")
                    else:
                        messagebox.showerror("Error", "Failed to clear database")
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred while clearing the database:\n{str(e)}")
            else:
                messagebox.showerror("Error", "Confirmation text does not match exactly")
                dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
            
        button_frame = ttk.Frame(warning_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Confirm Deletion", command=on_confirm, style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Set focus to entry and wait for dialog
        entry.focus_set()
        self.root.wait_window(dialog)

    def run_clustering(self):
        """Run clustering on processed videos."""
        # Get clustering parameters
        try:
            eps_min = float(self.eps_min.get())
            eps_max = float(self.eps_max.get())
            min_samples_min = int(self.min_samples_min.get())
            min_samples_max = int(self.min_samples_max.get())
            
            if not (0 < eps_min < eps_max <= 1.0):
                messagebox.showerror("Invalid Input", "Eps range must be between 0 and 1, with min < max")
                return
            if not (0 < min_samples_min <= min_samples_max):
                messagebox.showerror("Invalid Input", "Min samples range must be positive integers, with min <= max")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for clustering parameters")
            return
        
        # Get output directory
        output_dir = self.output_dir_entry.get()
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory")
            return
        
        # Disable clustering button while processing
        self.cluster_button.configure(state='disabled')
        
        # Start clustering in a separate thread
        threading.Thread(target=self._run_clustering_thread, 
                       args=(output_dir, eps_min, eps_max, min_samples_min, min_samples_max)).start()
    
    def _run_clustering_thread(self, output_dir: str, eps_min: float, eps_max: float, 
                             min_samples_min: int, min_samples_max: int):
        """Run clustering in a separate thread."""
        try:
            python_exe = get_venv_python()
            
            # Create clustering output directory
            cluster_dir = os.path.join(output_dir, "clustering_results")
            os.makedirs(cluster_dir, exist_ok=True)
            
            # Get list of processed videos
            processed_videos = []
            for video in self.video_listbox.get(0, tk.END):
                video_name = os.path.splitext(os.path.basename(video))[0]
                output_video = os.path.join(output_dir, f"{video_name}_output.mp4")
                if os.path.exists(output_video):
                    processed_videos.append(video)
            
            if not processed_videos:
                self.root.after(0, lambda: messagebox.showwarning("Warning", 
                    "No processed videos found. Please process videos first."))
                return
            
            # Run clustering on each processed video
            for video in processed_videos:
                self.output_box.insert(tk.END, f"\nRunning clustering on: {video}\n")
                self.output_box.see(tk.END)
                
                cmd = [python_exe, "crow_clustering.py",
                       "--video", video,
                       "--output-dir", cluster_dir,
                       "--eps-min", str(eps_min),
                       "--eps-max", str(eps_max),
                       "--min-samples-min", str(min_samples_min),
                       "--min-samples-max", str(min_samples_max)]
                
                self.output_box.insert(tk.END, f"Running: {' '.join(cmd)}\n")
                self.output_box.see(tk.END)
                
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.output_box.insert(tk.END, line)
                    self.output_box.see(tk.END)
                proc.wait()
                
                if proc.returncode == 0:
                    self.output_box.insert(tk.END, f"\n[Successfully clustered: {video}]\n")
                else:
                    self.output_box.insert(tk.END, f"\n[Error clustering: {video}]\n")
                self.output_box.see(tk.END)
            
            self.output_box.insert(tk.END, "\n[All videos clustered]\n")
            self.output_box.see(tk.END)
            
            # Show results directory
            self.root.after(0, lambda: messagebox.showinfo("Clustering Complete", 
                f"Clustering results saved to:\n{cluster_dir}\n\n"
                "Check clustering_results.json for detailed metrics and "
                "crow_clusters_visualization.png for cluster visualization."))
            
        finally:
            # Re-enable clustering button
            self.root.after(0, lambda: self.cluster_button.configure(state='normal'))

if __name__ == "__main__":
    # Ensure all requirements are installed before starting the GUI
    ensure_requirements()
    
    root = tk.Tk()
    app = FacebeakGUI(root)
    root.mainloop() 