import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading

class FacebeakGUI:
    def __init__(self, root):
        self.root = root
        root.title("facebeak Launcher")
        root.geometry("800x650")  # Increased height for new control

        # Video file selection
        tk.Label(root, text="Input Video:").pack()
        self.video_entry = tk.Entry(root, width=50)
        self.video_entry.pack()
        tk.Button(root, text="Browse", command=self.browse_video).pack()

        # Output file
        tk.Label(root, text="Output Video:").pack()
        self.output_entry = tk.Entry(root, width=50)
        self.output_entry.insert(0, "output.mp4")
        self.output_entry.pack()

        # Detection threshold
        tk.Label(root, text="Detection Threshold (0.0-1.0):").pack()
        self.det_thresh = tk.Entry(root, width=10)
        self.det_thresh.insert(0, "0.3")
        self.det_thresh.pack()
        tk.Label(root, text="Lower = more sensitive to detecting birds", font=("Arial", 8)).pack()

        # Similarity threshold
        tk.Label(root, text="Similarity Threshold (0.0-1.0):").pack()
        self.sim_thresh = tk.Entry(root, width=10)
        self.sim_thresh.insert(0, "0.85")
        self.sim_thresh.pack()
        tk.Label(root, text="Higher = stricter matching between frames", font=("Arial", 8)).pack()

        # Skip
        tk.Label(root, text="Frame Skip:").pack()
        self.skip_entry = tk.Entry(root, width=10)
        self.skip_entry.insert(0, "1")
        self.skip_entry.pack()

        # Run button
        tk.Button(root, text="Run facebeak", command=self.run_facebeak).pack(pady=10)

        # Output box
        self.output_box = tk.Text(root, height=10, width=70)
        self.output_box.pack()

    def browse_video(self):
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
        if filename:
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, filename)

    def run_facebeak(self):
        video = self.video_entry.get()
        output = self.output_entry.get()
        det_thresh = self.det_thresh.get()
        sim_thresh = self.sim_thresh.get()
        skip = self.skip_entry.get()
        
        # Validate thresholds
        try:
            det_thresh_float = float(det_thresh)
            sim_thresh_float = float(sim_thresh)
            if not (0 <= det_thresh_float <= 1 and 0 <= sim_thresh_float <= 1):
                messagebox.showerror("Invalid Input", "Thresholds must be between 0.0 and 1.0")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Thresholds must be valid numbers")
            return
            
        cmd = ["python", "main.py", 
               "--video", video, 
               "--output", output, 
               "--detection-threshold", det_thresh,
               "--similarity-threshold", sim_thresh,
               "--skip", skip]
        self.output_box.insert(tk.END, f"Running: {' '.join(cmd)}\n")
        self.output_box.see(tk.END)
        threading.Thread(target=self._run_cmd, args=(cmd,)).start()

    def _run_cmd(self, cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            self.output_box.insert(tk.END, line)
            self.output_box.see(tk.END)
        proc.wait()
        self.output_box.insert(tk.END, "\n[Done]\n")
        self.output_box.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = FacebeakGUI(root)
    root.mainloop() 