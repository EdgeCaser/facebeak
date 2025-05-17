import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import threading
from typing import List

class FacebeakGUI:
    def __init__(self, root):
        self.root = root
        root.title("facebeak Launcher")
        root.geometry("800x800")  # Made window taller to accommodate video list

        # Configure grid weights for resizing
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(9, weight=1)  # Make the output box row expandable

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

        # Run button (moved to class variable for access in other methods)
        self.run_button = ttk.Button(control_frame, text="Process Videos", command=self.run_facebeak)
        self.run_button.grid(row=9, column=0, columnspan=3, pady=10)

        # Output box with scrollbar
        output_frame = ttk.Frame(root)
        output_frame.grid(row=10, column=0, sticky="nsew", padx=10, pady=5)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Output box with scrollbar
        self.output_box = tk.Text(output_frame, height=10, width=70, yscrollcommand=scrollbar.set)
        self.output_box.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.output_box.yview)

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
        
        # Validate thresholds
        try:
            det_thresh_float = float(det_thresh)
            yolo_thresh_float = float(yolo_thresh)
            iou_thresh_float = float(iou_thresh)
            embed_thresh_float = float(embed_thresh)
            max_age_int = int(max_age)
            min_hits_int = int(min_hits)
            skip_int = int(skip)
            
            if not (0 <= det_thresh_float <= 1 and 0 <= yolo_thresh_float <= 1 and 
                    0 <= iou_thresh_float <= 1 and 0 <= embed_thresh_float <= 1):
                messagebox.showerror("Invalid Input", "Thresholds must be between 0.0 and 1.0")
                return
            if max_age_int < 1 or min_hits_int < 1 or skip_int < 0:
                messagebox.showerror("Invalid Input", "Max age and min hits must be positive integers; frame skip must be zero or positive.")
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
            'skip': skip
        })).start()

    def _process_videos(self, videos: List[str], output_dir: str, params: dict):
        try:
            for video in videos:
                # Generate output filename based on input video name
                import os
                video_name = os.path.splitext(os.path.basename(video))[0]
                output = os.path.join(output_dir, f"{video_name}_output.mp4")
                
                cmd = ["python", "main.py", 
                       "--video", video, 
                       "--output", output, 
                       "--detection-threshold", params['det_thresh'],
                       "--yolo-threshold", params['yolo_thresh'],
                       "--max-age", params['max_age'],
                       "--min-hits", params['min_hits'],
                       "--iou-threshold", params['iou_thresh'],
                       "--embedding-threshold", params['embed_thresh'],
                       "--skip", params['skip']]
                
                self.output_box.insert(tk.END, f"\nProcessing: {video}\n")
                self.output_box.insert(tk.END, f"Running: {' '.join(cmd)}\n")
                self.output_box.see(tk.END)
                
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.output_box.insert(tk.END, line)
                    self.output_box.see(tk.END)
                proc.wait()
                
                if proc.returncode == 0:
                    self.output_box.insert(tk.END, f"\n[Successfully processed: {video}]\n")
                else:
                    self.output_box.insert(tk.END, f"\n[Error processing: {video}]\n")
                self.output_box.see(tk.END)
            
            self.output_box.insert(tk.END, "\n[All videos processed]\n")
            self.output_box.see(tk.END)
            
        finally:
            # Re-enable run button
            self.root.after(0, lambda: self.run_button.configure(state='normal'))

if __name__ == "__main__":
    root = tk.Tk()
    app = FacebeakGUI(root)
    root.mainloop() 