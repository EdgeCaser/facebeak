"""
DictosPlus: Audio Transcription and Speaker Diarization Tool

Usage:
- Run this script to launch the GUI for audio transcription, speaker diarization, and relabeling.
- Requires ffmpeg, Python 3.8+, and the dependencies listed in requirements.txt.
- Supports Whisper ASR, SpeechBrain speaker embeddings, and MFCC fallback clustering.
- Optionally supports relabeling and speaker database improvement via relabeled transcripts.

Dependencies:
- torch, torchaudio, numpy, scikit-learn, ffmpeg-python, pydub, soundfile, noisereduce, faster-whisper, pyannote.audio, transformers, python-dotenv, huggingface-hub, sounddevice, librosa, logging

Thread Safety:
- All queue operations are thread-safe. Shared mutable state is only accessed from the main thread or protected by locks where needed.
- All background threads are daemonized to ensure clean shutdown.

"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import time
import os
import sys
import shutil
import json
import queue
import threading
import warnings
import pickle
import noisereduce as nr
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine
from transformers import pipeline
from speechbrain.inference.speaker import SpeakerRecognition
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
import logging
import sounddevice as sd
import librosa.display
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment, Timeline
import tempfile
from faster_whisper import WhisperModel  # "Don't Stop Me Now" - faster ASR!
import subprocess

# Disable specific transformers warning to prevent logging error
logging.getLogger("transformers").setLevel(logging.ERROR)

# Filter out specific warnings
warnings.filterwarnings('ignore', message='.*torchaudio._backend.*')
warnings.filterwarnings('ignore', message='.*backend.common.AudioMetaData.*')
warnings.filterwarnings('ignore', message='.*Model was trained with.*')

# --- CONFIGURATION ---
CONFIG = {
    'SIMILARITY_THRESHOLD': 0.7,
    'MAX_CHUNK_SEC': 90,
    'SILENCE_THRESHOLD': 3.0,
    'ASR_BATCH_SIZE': 8,
    'EMBEDDING_BATCH_SIZE': 32,
    'MIN_SEGMENT_DURATION': 1.0,
    'STATUS_TEXT_MAX_LINES': 1000,
    'STATUS_TEXT_TRIM_TO': 500,
}
SETTINGS_FILE = "optimizer_config.json"
def load_optimized_settings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('best_settings', CONFIG)
    except Exception:
        return CONFIG
OPTIMIZED_SETTINGS = load_optimized_settings()

def find_ffmpeg():
    print("Starting find_ffmpeg...")
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path:
        ffmpeg_exe = os.path.join(ffmpeg_path, 'ffmpeg.exe' if sys.platform == "win32" else 'ffmpeg')
        if os.path.exists(ffmpeg_exe):
            print(f"Found ffmpeg via FFMPEG_PATH: {ffmpeg_path}")
            return ffmpeg_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [script_dir, os.path.dirname(script_dir)]
    print(f"Searching for ffmpeg in directories: {search_dirs}")
    for search_dir in search_dirs:
        for ffmpeg_dir in ['ffmpeg', 'ffmpeg-*']:
            matches = [d for d in os.listdir(search_dir) if d.startswith(ffmpeg_dir)]
            for match in matches:
                bin_path = os.path.join(search_dir, match, 'bin')
                ffmpeg_exe = os.path.join(bin_path, 'ffmpeg.exe' if sys.platform == "win32" else 'ffmpeg')
                if os.path.exists(ffmpeg_exe):
                    print(f"Found ffmpeg in directory: {bin_path}")
                    return bin_path
    try:
        ffmpeg_exe = shutil.which("ffmpeg")
        if ffmpeg_exe:
            print(f"Found ffmpeg in system PATH: {ffmpeg_exe}")
            return os.path.dirname(ffmpeg_exe)
    except Exception as e:
        print(f"Error checking system PATH for ffmpeg: {str(e)}")
    raise RuntimeError("Could not find ffmpeg. Please ensure ffmpeg is installed in a subdirectory, set in the FFMPEG_PATH environment variable, or available in your system PATH.")

# Find and set ffmpeg path
print("Initializing ffmpeg path...")
try:
    ffmpeg_path = find_ffmpeg()
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
    print(f"ffmpeg path set: {ffmpeg_path}")
except Exception as e:
    messagebox.showerror("Error", str(e))
    sys.exit(1)

# Set device globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load environment variables
load_dotenv()

# Check for required environment variables
if not os.getenv('HF_TOKEN'):
    messagebox.showerror("Error", "HF_TOKEN environment variable not set. Please check your .env file.")
    sys.exit(1)

# Get Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN loaded from environment.")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Clear and initialize logging for this session
log_file = 'dictosplus.log'
if os.path.exists(log_file):
    try:
        with open(log_file, 'w') as f:
            f.write('')  # Clear the file
    except Exception as e:
        print(f"Warning: Could not clear log file: {str(e)}")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    force=True  # Force reconfiguration of logging
)
logging.info("=== New DictosPlus Session Started ===")

class SpeakerDatabase:
    """Database for storing and matching speaker embeddings and metadata."""
    SIMILARITY_THRESHOLD = OPTIMIZED_SETTINGS['SIMILARITY_THRESHOLD']
    
    def __init__(self, db_path="speaker_database.pkl"):
        """Initialize the speaker database and load the embedding model."""
        print("Initializing SpeakerDatabase...")
        self.db_path = Path(db_path)
        self.speakers = self._load_database()
        # "I Want It All" - initialize embedding cache
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        print("Speaker database loaded.")
        
        # Check for HF_TOKEN
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN environment variable not set. Please set it with your Hugging Face token.")
        
        # Initialize embedding model (optional for transcription)
        self.embedding_model = None
        try:
            if DEVICE == "cuda":
                # "I Want It All" - enable TF32 and FP16 for RTX 3080
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True  # "Don't Stop Me Now" - optimize CUDA kernels
            
            print(f"Loading speaker embedding model 'speechbrain/spkrec-xvect-voxceleb' on device: {DEVICE}...")
            model_dir = Path("pretrained_models/spkrec-xvect-voxceleb")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # "The Show Must Go On" - initialize with GPU optimization
            self.embedding_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",  # Lighter model for faster processing
                savedir=str(model_dir),
                run_opts={"device": DEVICE}
            )
            self.embedding_model.eval()
            
            # Move to GPU and enable FP16 if available
            if DEVICE == "cuda":
                self.embedding_model = self.embedding_model.to(DEVICE)
                try:
                    # "Another One Bites the Dust" - enable FP16 for faster processing
                    self.embedding_model = self.embedding_model.half()
                    print("FP16 enabled for speaker embedding model.")
                except Exception as e:
                    print(f"Could not enable FP16: {str(e)}. Continuing with FP32.")
            
            print("Speaker embedding model loaded successfully.")
        except Exception as e:
            print(f"Failed to initialize speaker embedding model: {str(e)}. Transcription will proceed without speaker diarization.")
            self.embedding_model = None
    
    def _load_database(self):
        """Load the speaker database with all metadata."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'speakers' in data:
                        self.embedding_cache = data.get('cache', {})
                        return data['speakers']
                    return data
            except Exception as e:
                print(f"Failed to load speaker database: {str(e)}. Starting with empty database.")
                return {}
        return {}
    
    def save_database(self):
        """Save the speaker database with all metadata."""
        try:
            # Convert sets to lists for pickle compatibility
            speakers_serializable = {}
            for name, data in self.speakers.items():
                speakers_serializable[name] = dict(data)
                if 'audio_paths' in data:
                    speakers_serializable[name]['audio_paths'] = list(data['audio_paths'])
            with open(self.db_path, 'wb') as f:
                pickle.dump({
                    'speakers': speakers_serializable,
                    'cache': self.embedding_cache,
                    'cache_stats': {
                        'hits': self.cache_hits,
                        'misses': self.cache_misses
                    }
                }, f)
        except Exception as e:
            print(f"Failed to save speaker database: {str(e)}")
    
    def get_cached_embedding(self, audio_hash):
        """Get cached embedding if available."""
        if audio_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[audio_hash]
        self.cache_misses += 1
        return None

    def cache_embedding(self, audio_hash, embedding):
        """Cache an embedding."""
        self.embedding_cache[audio_hash] = embedding
        # "Don't Stop Me Now" - limit cache size to prevent memory issues
        if len(self.embedding_cache) > 1000:  # Keep last 1000 embeddings
            self.embedding_cache = dict(list(self.embedding_cache.items())[-1000:])

    def compute_audio_hash(self, audio_data):
        """Compute a hash for audio data to use as cache key."""
        # "We Will Rock You" - use a fast hash of audio statistics
        return hash((
            np.mean(audio_data),
            np.std(audio_data),
            len(audio_data)
        ))

    def encode_batch(self, audio_batch, sr_batch):
        """Encode a batch of audio segments into speaker embeddings with caching."""
        if self.embedding_model is None:
            return None
            
        try:
            # "Keep Yourself Alive" - check cache first
            embeddings = []
            to_process = []
            to_process_indices = []
            
            for i, audio in enumerate(audio_batch):
                audio_hash = self.compute_audio_hash(audio)
                cached = self.get_cached_embedding(audio_hash)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    to_process.append(audio)
                    to_process_indices.append(i)
            
            if to_process:
                # Process uncached audio
                with torch.no_grad():
                    if DEVICE == "cuda":
                        audio_tensors = [torch.FloatTensor(audio).to(DEVICE).half() 
                                       if hasattr(self.embedding_model, 'half') 
                                       else torch.FloatTensor(audio).to(DEVICE) 
                                       for audio in to_process]
                    else:
                        audio_tensors = [torch.FloatTensor(audio).to(DEVICE) 
                                       for audio in to_process]
                    
                    # Process batch
                    new_embeddings = self.embedding_model.encode_batch(audio_tensors)
                    
                    # Convert to numpy and handle batch dimension
                    if isinstance(new_embeddings, torch.Tensor):
                        new_embeddings = new_embeddings.cpu().numpy()
                        if len(new_embeddings.shape) > 1:
                            new_embeddings = np.mean(new_embeddings, axis=0)
                    
                    # Cache new embeddings
                    for audio, embedding in zip(to_process, new_embeddings):
                        self.cache_embedding(self.compute_audio_hash(audio), embedding)
                    
                    # Insert new embeddings at correct positions
                    for idx, embedding in zip(to_process_indices, new_embeddings):
                        embeddings.insert(idx, embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Error in batch encoding: {str(e)}")
            return None

    def find_matching_speaker(self, embedding):
        """Find matching speaker using optimized cosine similarity."""
        if not self.speakers:
            return None, 0
            
        # "Another One Bites the Dust" - use vectorized operations
        best_match = None
        best_score = 0
        
        # Convert embeddings to numpy arrays for faster computation
        speaker_embeddings = {}
        for name, data in self.speakers.items():
            if data.get('embeddings'):
                speaker_embeddings[name] = np.array(data['embeddings'])
        
        # Compute similarities for all speakers at once
        for name, known_embeddings in speaker_embeddings.items():
            if (hasattr(embedding, 'shape') and 
                hasattr(known_embeddings, 'shape') and 
                embedding.shape == known_embeddings.shape[-1] and
                not np.any(np.isnan(embedding)) and 
                not np.any(np.isnan(known_embeddings))):
                
                # Vectorized cosine similarity
                similarities = 1 - np.array([cosine(embedding, e) for e in known_embeddings])
                max_similarity = np.max(similarities)
                
                if max_similarity > self.SIMILARITY_THRESHOLD and max_similarity > best_score:
                    best_score = max_similarity
                    best_match = name
        
        return best_match, best_score

    def optimize_clustering(self, embeddings, num_speakers=None):
        """Optimized clustering for speaker diarization."""
        if len(embeddings) < 10:
            # "The Show Must Go On" - skip clustering for few segments
            return np.zeros(len(embeddings), dtype=int)
            
        try:
            # "I Want It All" - use optimized clustering
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            
            # Normalize embeddings
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(embeddings)
            
            # Use optimal number of clusters if not specified
            if num_speakers is None:
                from sklearn.metrics import silhouette_score
                max_clusters = min(10, len(embeddings) - 1)
                if max_clusters < 2:
                    return np.zeros(len(embeddings), dtype=int)
                    
                # Find optimal number of clusters
                scores = []
                for n in range(2, max_clusters + 1):
                    clustering = AgglomerativeClustering(n_clusters=n)
                    labels = clustering.fit_predict(normalized_embeddings)
                    score = silhouette_score(normalized_embeddings, labels)
                    scores.append((n, score))
                
                # Choose number of clusters with best silhouette score
                num_speakers = max(scores, key=lambda x: x[1])[0]
            
            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward',  # "Don't Stop Me Now" - use ward linkage for better clusters
                compute_full_tree=True
            )
            
            return clustering.fit_predict(normalized_embeddings)
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}")
            # Fallback to single speaker
            return np.zeros(len(embeddings), dtype=int)

    def add_speaker(self, name, embedding, audio_path=None):
        """Add or update a speaker in the database with embedding and audio provenance."""
        now = datetime.now().isoformat()
        if name not in self.speakers:
            self.speakers[name] = {
                'embeddings': [],
                'audio_paths': set(),
                'speaking_time': 0.0,
                'last_seen': now,
                'segment_count': 0
            }
        self.speakers[name]['embeddings'].append(np.array(embedding))
        if audio_path:
            self.speakers[name]['audio_paths'].add(audio_path)
        self.speakers[name]['last_seen'] = now
        self.speakers[name]['segment_count'] += 1
        self.save_database()

    def update_speaking_time(self, name, duration):
        """Update the total speaking time for a speaker."""
        if name in self.speakers:
            self.speakers[name]['speaking_time'] += duration
            self.speakers[name]['last_seen'] = datetime.now().isoformat()
            self.save_database()

class SpeakerLabelDialog:
    """Dialog for labeling speakers after diarization or clustering."""
    def __init__(self, parent, speaker_samples, speaker_embeddings, speaker_db, audio_data=None, sample_rate=None):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Label Speakers")
        self.dialog.geometry("800x1000")
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.speaker_names = {}
        self.speaker_embeddings = speaker_embeddings
        self.speaker_db = speaker_db
        self.sample_segment_times = {}  # Store start/end times for each sample for playback
        self.create_widgets(speaker_samples)
    
    def create_widgets(self, speaker_samples):
        tk.Label(self.dialog, text="Label Speakers - Known speakers will be pre-filled", 
                font=("Arial", 12)).pack(pady=10)
        self.name_entries = {}
        self.show_more_vars = {}
        for speaker, samples in speaker_samples.items():
            frame = ttk.LabelFrame(self.dialog, text=f"Speaker {speaker}")
            frame.pack(padx=10, pady=5, fill="x")
            # Try to find matching speaker
            embedding = self.speaker_embeddings[speaker]
            matching_speaker, confidence = self.speaker_db.find_matching_speaker(embedding) if embedding is not None else (None, 0)
            # Show match confidence if found
            if matching_speaker:
                tk.Label(frame, text=f"Matched to: {matching_speaker} (Confidence: {confidence:.2f})", fg="green").pack(anchor="w")
                # Show speaker info if available
                info = self.speaker_db.speakers.get(matching_speaker, {})
                if info:
                    tk.Label(frame, text=f"Total speaking time: {info.get('speaking_time', 0):.1f}s, Segments: {info.get('segment_count', 0)}, Last seen: {info.get('last_seen', 'N/A')}", fg="blue").pack(anchor="w")
            # Show up to 3 different segments, with option to show more
            self.sample_segment_times[speaker] = []
            show_more_var = tk.BooleanVar(value=False)
            self.show_more_vars[speaker] = show_more_var
            def make_show_more_callback(sp=speaker, fr=frame, smp=samples, var=show_more_var):
                def callback():
                    for widget in fr.winfo_children():
                        if getattr(widget, 'is_sample', False):
                            widget.destroy()
                    for i, (sample_text, (start, end)) in enumerate(smp):
                        if i < 3 or var.get():
                            lbl = tk.Label(fr, text=f"Sample {i+1}: {sample_text[:100]}...")
                            lbl.is_sample = True
                            lbl.pack(anchor="w")
                            self.sample_segment_times[sp].append((start, end))
                            if self.audio_data is not None and self.sample_rate is not None:
                                btn = tk.Button(fr, text=f"Play Sample {i+1}", command=lambda idx=i, spk=sp: self.play_sample(spk, idx))
                                btn.is_sample = True
                                btn.pack(anchor="w")
                return callback
            # Initial display
            make_show_more_callback()( )
            if len(samples) > 3:
                show_more_btn = tk.Checkbutton(frame, text="Show More", variable=show_more_var, command=make_show_more_callback())
                show_more_btn.pack(anchor="w")
            tk.Label(frame, text="Enter name:").pack(anchor="w")
            entry = tk.Entry(frame)
            entry.insert(0, matching_speaker if matching_speaker else f"Speaker {speaker}")
            entry.pack(anchor="w", padx=5, pady=2)
            self.name_entries[speaker] = entry
        # Add checkbox to save to database
        self.save_to_db = tk.BooleanVar(value=True)
        tk.Checkbutton(self.dialog, text="Save speakers to database", 
                      variable=self.save_to_db).pack(pady=5)
        tk.Button(self.dialog, text="Confirm Labels", 
                 command=self.confirm_labels).pack(pady=10)
    
    def play_sample(self, speaker, idx):
        # Play the audio for the idx-th segment of this speaker
        if self.audio_data is None or self.sample_rate is None:
            messagebox.showerror("Error", "No audio loaded.")
            return
        segs = self.sample_segment_times.get(speaker, [])
        if not segs or idx >= len(segs):
            return
        start, end = segs[idx]
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        segment = self.audio_data[start_idx:end_idx]
        threading.Thread(target=lambda: sd.play(segment, self.sample_rate)).start()
    
    def confirm_labels(self):
        for speaker, entry in self.name_entries.items():
            name = entry.get()
            self.speaker_names[speaker] = name
            # Save to database if checkbox is checked
            if self.save_to_db.get():
                self.speaker_db.add_speaker(
                    name, 
                    self.speaker_embeddings[speaker],
                    audio_path=getattr(self, 'audio_path', None)
                )
        self.dialog.destroy()

class TranscriptionApp:
    """Main application class for DictosPlus GUI and transcription workflow."""
    def __init__(self, root):
        print("Initializing TranscriptionApp...")
        logging.info("Application started.")
        self.root = root
        self.root.title("Audio Transcription with Speaker Diarization (HuggingFace + pyannote)")
        self.root.geometry("600x800")  # Increased height to ensure button visibility
        self.audio_path = tk.StringVar()
        self.audio_duration = tk.DoubleVar(value=0.0)
        self.max_duration_var = tk.StringVar()
        self.num_speakers_var = tk.StringVar(value="2")
        self.status_queue = queue.Queue()
        self.language_var = tk.StringVar(value="auto")
        self.noise_reduction_var = tk.BooleanVar(value=True)
        self.mono_method_var = tk.StringVar(value="average")

        tk.Label(self.root, text="Audio Transcription Tool", font=("Arial", 16)).pack(pady=10)
        tk.Label(self.root, text="Audio File:").pack()
        tk.Entry(self.root, textvariable=self.audio_path, width=50, state='readonly').pack()
        tk.Button(self.root, text="Select Audio File", command=self.select_file).pack(pady=5)
        tk.Label(self.root, text="Duration to Process (seconds, 0 for full audio):").pack()
        tk.Entry(self.root, textvariable=self.max_duration_var, width=20).pack()
        tk.Label(self.root, text="Estimated Number of Speakers (leave blank for auto):").pack()
        tk.Entry(self.root, textvariable=self.num_speakers_var, width=20).pack()
        tk.Label(self.root, text="Transcription Language:").pack()
        language_options = [
            ("Auto", "auto"),
            ("English", "en"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Portuguese", "pt"),
            ("Russian", "ru"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
            ("Korean", "ko"),
            ("Arabic", "ar"),
        ]
        lang_dropdown = ttk.Combobox(self.root, textvariable=self.language_var, state="readonly", width=20)
        lang_dropdown['values'] = [name for name, code in language_options]
        lang_dropdown.current(0)
        lang_dropdown.pack()
        self.language_code_map = {name: code for name, code in language_options}
        self.start_button = tk.Button(self.root, text="Start Transcription", command=self.start_transcription)
        self.start_button.pack(pady=10)
        tk.Label(self.root, text="Progress:").pack()
        self.progress = ttk.Progressbar(self.root, length=400, mode='determinate')
        self.progress.pack()
        tk.Label(self.root, text="Status:").pack()
        self.status_text = tk.Text(self.root, height=10, width=60, state='disabled')
        self.status_text.pack(pady=5)
        
        # Placeholder for the "Label Speakers" button (to be added after transcription)
        self.label_speakers_button = None
        
        # Create the button container frame to maintain layout
        self.button_container = tk.Frame(self.root)
        self.button_container.pack(pady=5)
        
        self.root.after(100, self.check_status_queue)
        
        # Initialize speaker database
        print("Creating SpeakerDatabase instance...")
        self.speaker_db = SpeakerDatabase()
        print("SpeakerDatabase initialized.")

        # Initialize variables to store transcription data
        self.current_segments = None
        self.current_speaker_segments = None
        self.current_audio_file = None
        self.current_audio_data = None
        self.current_sample_rate = None

        # Add menu for importing relabeled transcripts
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Import Relabeled Transcript", command=self.import_relabeled_transcript)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

        self._playback_lock = threading.Lock()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Add progress bar for labeling process
        self.labeling_progress_frame = tk.Frame(self.root)
        self.labeling_progress_frame.pack(pady=5)
        self.labeling_progress_label = tk.Label(self.labeling_progress_frame, text="Labeling Progress:")
        self.labeling_progress_label.pack()
        self.labeling_progress = ttk.Progressbar(self.labeling_progress_frame, length=400, mode='determinate')
        self.labeling_progress.pack()
        self.labeling_status = tk.Label(self.labeling_progress_frame, text="")
        self.labeling_status.pack()
        # Hide the labeling progress initially
        self.labeling_progress_frame.pack_forget()

        # Initialize VAD pipeline
        self.vad_pipeline = None
        try:
            # First check if we have access to the model
            from huggingface_hub import HfApi
            api = HfApi()
            try:
                # Check if we can access the model
                api.model_info("pyannote/voice-activity-detection", token=HF_TOKEN)
                self.update_status("Checking VAD model access...")
                
                # Initialize the pipeline
                self.vad_pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection",
                    use_auth_token=HF_TOKEN
                )
                
                # Move to GPU if available
                if DEVICE == "cuda":
                    try:
                        self.vad_pipeline = self.vad_pipeline.to(torch.device("cuda"))
                        self.update_status("VAD pipeline initialized successfully on GPU.")
                    except Exception as gpu_error:
                        self.update_status(f"Warning: Could not move VAD pipeline to GPU: {str(gpu_error)}")
                        self.update_status("VAD pipeline initialized on CPU instead.")
                else:
                    self.update_status("VAD pipeline initialized successfully on CPU.")
                    
            except Exception as model_error:
                if "401" in str(model_error):
                    self.update_status("Error: No access to VAD model. Please accept the model's terms of use at:")
                    self.update_status("https://huggingface.co/pyannote/voice-activity-detection")
                    self.update_status("Then try restarting the application.")
                elif "404" in str(model_error):
                    self.update_status("Error: VAD model not found. Please check your internet connection.")
                else:
                    self.update_status(f"Error accessing VAD model: {str(model_error)}")
                self.vad_pipeline = None
                
        except ImportError:
            self.update_status("Warning: huggingface_hub not installed. Installing required package...")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                self.update_status("Please restart the application to use VAD features.")
            except Exception as install_error:
                self.update_status(f"Failed to install huggingface_hub: {str(install_error)}")
            self.vad_pipeline = None
        except Exception as e:
            self.update_status(f"Warning: Failed to initialize VAD pipeline: {str(e)}")
            self.update_status("The application will use fallback segmentation method.")
            self.vad_pipeline = None

        # Add optimization buttons
        self.optimize_button = tk.Button(self.root, text="Re-optimize Settings", command=self.run_optimizer)
        self.optimize_button.pack(pady=2)
        self.reset_button = tk.Button(self.root, text="Reset to Defaults", command=self.reset_settings)
        self.reset_button.pack(pady=2)

    def on_close(self):
        """Handle application close: clean up resources and exit."""
        logging.info("Application closed by user.")
        try:
            sd.stop()
        except Exception:
            pass
        self.root.destroy()

    def select_file(self):
        """Prompt user to select an audio file and update state."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File to Transcribe",
            filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")]
        )
        if file_path:
            self.audio_path.set(file_path)
            duration = self.get_audio_duration(file_path)
            if duration == 0.0:
                messagebox.showerror("Error", "Failed to load audio file.")
                logging.error(f"Failed to load audio file: {file_path}")
                return
            self.audio_duration.set(duration)
            self.max_duration_var.set(str(duration))
            self.update_status(f"Selected audio file: {file_path}\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    def get_audio_duration(self, file_path: str) -> float:
        try:
            info = sf.info(file_path)
            return info.duration
        except Exception as e:
            self.update_status(f"Error determining audio duration: {str(e)}")
            return 0.0

    def update_status(self, message: str):
        """Update the status text and log the message."""
        self.status_queue.put(message)
        logging.info(message)
        # Limit status text growth
        self.status_text.config(state='normal')
        lines = self.status_text.get("1.0", tk.END).splitlines()
        if len(lines) > OPTIMIZED_SETTINGS['STATUS_TEXT_MAX_LINES']:
            self.status_text.delete("1.0", f"{len(lines)-OPTIMIZED_SETTINGS['STATUS_TEXT_TRIM_TO']}.0")
        self.status_text.config(state='disabled')

    def check_status_queue(self):
        """Check the status queue and update the GUI in the main thread."""
        try:
            while True:
                message = self.status_queue.get_nowait()
                self.status_text.config(state='normal')
                self.status_text.insert(tk.END, message + "\n")
                self.status_text.see(tk.END)
                self.status_text.config(state='disabled')
        except queue.Empty:
            pass
        self.root.after(100, self.check_status_queue)

    def start_transcription(self):
        """Start the transcription process in a background thread."""
        try:
            max_duration = float(self.max_duration_var.get())
            if max_duration < 0:
                messagebox.showerror("Invalid Input", "Please enter a non-negative number for duration (or 0 for full audio).")
                return
            if max_duration > self.audio_duration.get():
                messagebox.showerror("Invalid Input", f"Duration cannot exceed audio length ({self.audio_duration.get():.1f} seconds).")
                return
            num_speakers_input = self.num_speakers_var.get().strip()
            num_speakers = int(num_speakers_input) if num_speakers_input else None
            if num_speakers is not None and num_speakers <= 0:
                messagebox.showerror("Invalid Input", "Number of speakers must be a positive integer (or leave blank for auto-detection).")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for duration and number of speakers.")
            return
        # Disable the start button and hide the label speakers button if it exists
        self.start_button.config(state='disabled')
        if self.label_speakers_button:
            self.label_speakers_button.pack_forget()
        self.progress['value'] = 0
        self.update_status("Starting transcription process...")
        thread = threading.Thread(target=self.run_transcription, args=(
            self.audio_path.get(), max_duration, num_speakers
        ))
        thread.daemon = True  # Daemon thread for clean shutdown
        thread.start()

    def detect_speech_segments(self, audio_path: str) -> Timeline:
        """Detect speech segments using VAD with parallel processing."""
        if self.vad_pipeline is None:
            self.update_status("VAD pipeline not available. Using parallel energy-based detection.")
            return self._parallel_energy_vad(audio_path)
        try:
            self.update_status("Running Voice Activity Detection in parallel...")
            audio, sr = sf.read(audio_path)
            duration = len(audio) / sr
            num_chunks = min(10, os.cpu_count() or 4)
            chunk_size = len(audio) // num_chunks
            chunks = []
            for i in range(0, len(audio), chunk_size):
                chunk_end = min(i + chunk_size, len(audio))
                chunk_audio = audio[i:chunk_end]
                chunk_start = i / sr
                chunks.append((chunk_audio, sr, chunk_start))
            from concurrent.futures import ThreadPoolExecutor
            timeline = Timeline()
            def process_chunk(chunk_info):
                chunk_audio, chunk_sr, chunk_start = chunk_info
                try:
                    fd, temp_path = tempfile.mkstemp(suffix='.wav')
                    os.close(fd)
                    sf.write(temp_path, chunk_audio, chunk_sr)
                    chunk_output = self.vad_pipeline(temp_path)
                    os.remove(temp_path)
                    for segment in chunk_output.get_timeline():
                        if segment.duration >= OPTIMIZED_SETTINGS['MIN_SEGMENT_DURATION']:
                            adjusted_segment = Segment(
                                segment.start + chunk_start,
                                segment.end + chunk_start
                            )
                            timeline.add(adjusted_segment)
                except Exception as e:
                    self.update_status(f"Error processing VAD chunk: {str(e)}")
                    return None
            with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                list(executor.map(process_chunk, chunks))
            if len(timeline) == 0:
                self.update_status("No speech segments detected. Using fallback method.")
                return self._parallel_energy_vad(audio_path)
            timeline = timeline.support()
            merged_timeline = Timeline()
            for segment in timeline:
                if not merged_timeline or (segment.start - merged_timeline[-1].end) > OPTIMIZED_SETTINGS['SILENCE_THRESHOLD']:
                    merged_timeline.add(segment)
                else:
                    prev_segment = merged_timeline[-1]
                    merged_timeline.add(Segment(prev_segment.start, segment.end))
            self.update_status(f"VAD detected {len(merged_timeline)} speech segments.")
            return merged_timeline
        except Exception as e:
            self.update_status(f"Error in VAD processing: {str(e)}")
            self.update_status("Falling back to parallel energy-based detection.")
            return self._parallel_energy_vad(audio_path)

    def _parallel_energy_vad(self, audio_path: str) -> Timeline:
        """Fallback VAD using parallel energy-based detection."""
        self.update_status("Running parallel energy-based speech detection...")
        try:
            audio, sr = sf.read(audio_path)
            duration = len(audio) / sr
            frame_length = int(0.03 * sr)
            hop_length = int(0.01 * sr)
            energy = np.array([np.sum(audio[i:i+frame_length]**2) 
                             for i in range(0, len(audio)-frame_length, hop_length)])
            energy_threshold = np.percentile(energy, 25) * 2
            num_chunks = min(10, os.cpu_count() or 4)
            chunk_size = len(audio) // num_chunks
            chunks = []
            for i in range(0, len(audio), chunk_size):
                chunk_end = min(i + chunk_size, len(audio))
                chunk_audio = audio[i:chunk_end]
                chunk_start = i / sr
                chunks.append((chunk_audio, sr, chunk_start, frame_length, hop_length, energy_threshold))
            from concurrent.futures import ProcessPoolExecutor
            timeline = Timeline()
            with ProcessPoolExecutor(max_workers=num_chunks) as executor:
                chunk_results = list(executor.map(process_energy_chunk, chunks))
            for segments in chunk_results:
                for segment in segments:
                    timeline.add(segment)
            timeline = timeline.support()
            merged_timeline = Timeline()
            for segment in timeline:
                if not merged_timeline or (segment.start - merged_timeline[-1].end) > OPTIMIZED_SETTINGS['SILENCE_THRESHOLD']:
                    merged_timeline.add(segment)
                else:
                    prev_segment = merged_timeline[-1]
                    merged_timeline.add(Segment(prev_segment.start, segment.end))
            self.update_status(f"Energy-based VAD detected {len(merged_timeline)} speech segments.")
            return merged_timeline
        except Exception as e:
            self.update_status(f"Error in energy-based VAD: {str(e)}")
            return Timeline([Segment(start, min(start + OPTIMIZED_SETTINGS['MAX_CHUNK_SEC'], duration))
                           for start in np.arange(0, duration, OPTIMIZED_SETTINGS['MAX_CHUNK_SEC'])])

    def run_transcription(self, file_path: str, max_duration: float, num_speakers: int):
        """Run the transcription pipeline in a background thread."""
        try:
            self.progress['value'] = 5
            self.root.update()
            
            self.update_status("Preprocessing audio...")
            try:
                audio, sr = sf.read(file_path)
                if len(audio.shape) > 1:
                    method = self.mono_method_var.get()
                    if method == "left":
                        audio = audio[:, 0]
                    elif method == "right":
                        audio = audio[:, -1]
                    else:
                        audio = np.mean(audio, axis=1)
                
                if max_duration > 0:
                    samples = int(max_duration * sr)
                    audio = audio[:samples]
                
                if self.noise_reduction_var.get():
                    self.update_status("Applying noise reduction...")
                    audio = nr.reduce_noise(y=audio, sr=sr)
                
                duration = len(audio) / sr
            except Exception as e:
                self.update_status(f"Error during audio preprocessing: {str(e)}")
                messagebox.showerror("Error", f"Failed to process audio file: {str(e)}")
                self.start_button.config(state='normal')
                return

            self.current_audio_file = file_path
            self.current_audio_data = audio
            self.current_sample_rate = sr

            # Detect speech segments using VAD
            speech_timeline = self.detect_speech_segments(file_path)
            
            if speech_timeline is None:
                # Fallback to fixed chunks if VAD fails
                self.update_status("Using fallback fixed-chunk segmentation...")
                max_chunk_sec = OPTIMIZED_SETTINGS['MAX_CHUNK_SEC']
                speech_segments = [(start, min(start + max_chunk_sec, duration)) 
                                 for start in np.arange(0, duration, max_chunk_sec)]
            else:
                # Convert Timeline to list of (start, end) tuples
                speech_segments = [(segment.start, segment.end) for segment in speech_timeline]
            
            self.update_status(f"Processing {len(speech_segments)} speech segments.")
            
            # Create temporary directory for chunks
            temp_chunks_dir = os.path.join(os.path.dirname(file_path), "temp_chunks")
            os.makedirs(temp_chunks_dir, exist_ok=True)

            selected_lang_name = self.language_var.get()
            selected_lang_code = self.language_code_map.get(selected_lang_name, "auto")
            
            self.update_status("Initializing Faster Whisper ASR model on GPU...")
            try:
                asr_model = WhisperModel(
                    "medium.en",  # Use medium model for better accuracy
                    device="cuda" if DEVICE == "cuda" else "cpu",
                    compute_type="float16" if DEVICE == "cuda" else "float32",
                    download_root=os.path.join(os.path.dirname(__file__), "models")
                )
            except Exception as e:
                self.update_status(f"Failed to initialize Faster Whisper model: {str(e)}")
                messagebox.showerror("Error", f"Failed to initialize Faster Whisper model: {str(e)}")
                self.start_button.config(state='normal')
                return

            segments = []
            batch_size = OPTIMIZED_SETTINGS['ASR_BATCH_SIZE']
            chunk_batches = []
            current_batch = []
            
            # Track processed segments and their boundaries
            processed_segments = set()
            segment_boundaries = {}  # Maps segment key to (start, end, text)
            
            # Log segment durations before batching
            self.update_status(f"Logging segment durations before batching:")
            for i, (seg_start, seg_end) in enumerate(speech_segments):
                seg_duration = seg_end - seg_start
                self.update_status(f"Segment {i+1}: {seg_duration:.2f} seconds")
                
                # Only process segments that haven't been processed
                segment_key = (seg_start, seg_end)
                if segment_key in processed_segments:
                    continue
                processed_segments.add(segment_key)
                
                # Process segment as a single chunk if it's short enough
                if seg_duration <= OPTIMIZED_SETTINGS['MAX_CHUNK_SEC']:
                    start_sample = int(seg_start * sr)
                    end_sample = int(seg_end * sr)
                    chunk_audio = audio[start_sample:end_sample]
                    if len(chunk_audio) == 0:
                        continue
                    chunk_audio = chunk_audio.astype(np.float32)
                    current_batch.append((chunk_audio, sr, seg_start, seg_end, segment_key))
                else:
                    # Split longer segments into chunks with overlap
                    overlap = 2.0  # 2 second overlap between chunks
                    chunk_starts = np.arange(seg_start, seg_end, OPTIMIZED_SETTINGS['MAX_CHUNK_SEC'] - overlap)
                    for chunk_start in chunk_starts:
                        chunk_end = min(chunk_start + OPTIMIZED_SETTINGS['MAX_CHUNK_SEC'], seg_end)
                        start_sample = int(chunk_start * sr)
                        end_sample = int(chunk_end * sr)
                        chunk_audio = audio[start_sample:end_sample]
                        if len(chunk_audio) == 0:
                            continue
                        chunk_audio = chunk_audio.astype(np.float32)
                        current_batch.append((chunk_audio, sr, chunk_start, chunk_end, segment_key))
                
                if len(current_batch) >= batch_size:
                    chunk_batches.append(current_batch)
                    current_batch = []
            
            if current_batch:
                chunk_batches.append(current_batch)
            
            # Process batches
            for batch_idx, batch in enumerate(chunk_batches):
                self.update_status(f"Processing batch {batch_idx + 1}/{len(chunk_batches)}...")
                try:
                    for chunk_audio, chunk_sr, chunk_start, chunk_end, segment_key in batch:
                        # Normalize audio if needed
                        if np.max(np.abs(chunk_audio)) > 1.0:
                            chunk_audio = chunk_audio / np.max(np.abs(chunk_audio))
                        
                        segments_result, info = asr_model.transcribe(
                            chunk_audio,
                            language=selected_lang_code if selected_lang_code != "auto" else None,
                            beam_size=10,  # Increased beam size for better accuracy
                            vad_filter=False,  # Disable VAD filter since we're doing VAD preprocessing
                            condition_on_previous_text=True,  # Enable context awareness
                            temperature=0.0,  # Disable sampling for more deterministic output
                            compression_ratio_threshold=2.4,  # Stricter compression ratio
                            no_speech_threshold=0.6  # Higher threshold to avoid false positives
                        )
                        
                        # Convert generator to list and process segments
                        segments_list = list(segments_result)
                        
                        # Group segments by their original segment key
                        if segment_key not in segment_boundaries:
                            segment_boundaries[segment_key] = []
                        
                        for seg in segments_list:
                            segment_start = float(seg.start) + chunk_start
                            segment_end = float(seg.end) + chunk_start
                            
                            # Only add non-empty segments with valid timestamps
                            if segment_end > segment_start and seg.text.strip():
                                segment_boundaries[segment_key].append({
                                    "start": segment_start,
                                    "end": segment_end,
                                    "text": self.postprocess_text(seg.text)
                                })
                    
                    # After processing each batch, merge segments within their boundaries
                    for segment_key, seg_list in segment_boundaries.items():
                        if not seg_list:
                            continue
                        
                        # Sort segments by start time
                        seg_list.sort(key=lambda x: x["start"])
                        
                        # Merge overlapping segments
                        merged_segments = []
                        current_seg = seg_list[0]
                        
                        for next_seg in seg_list[1:]:
                            if next_seg["start"] <= current_seg["end"] + 0.5:  # 0.5s overlap threshold
                                # Merge segments
                                current_seg["end"] = max(current_seg["end"], next_seg["end"])
                                current_seg["text"] = current_seg["text"] + " " + next_seg["text"]
                            else:
                                merged_segments.append(current_seg)
                                current_seg = next_seg
                        
                        merged_segments.append(current_seg)
                        
                        # Add merged segments to final output
                        for seg in merged_segments:
                            segments.append({
                                "start": seg["start"],
                                "end": seg["end"],
                                "text": seg["text"],
                                "speaker": "UNKNOWN"
                            })
                    
                    self.update_status(f"Finished batch {batch_idx + 1}/{len(chunk_batches)}. Segments so far: {len(segments)}")
                except Exception as e:
                    print(f"[DEBUG] Error in batch {batch_idx + 1}: {str(e)}")
                    self.update_status(f"ASR batch failed: {str(e)}")
                    continue
                
                self.progress['value'] = 20 + int(30 * (batch_idx + 1) / max(1, len(chunk_batches)))
                self.root.update()

            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])
            
            # Clean up temporary files
            shutil.rmtree(temp_chunks_dir, ignore_errors=True)
            
            # Continue with speaker diarization...
            self.progress['value'] = 50
            self.root.update()
            speaker_segments = {"SPEAKER_1": []}
            for seg in segments:
                seg["speaker"] = "SPEAKER_1"
                speaker_segments["SPEAKER_1"].append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                })
            self.update_status("Saving initial transcription...")
            txt_output, json_output = self.save_transcription(file_path, segments, speaker_segments, {})
            print(f"Transcription saved to: {txt_output}")
            print(f"JSON output saved to: {json_output}")
            self.update_status("Transcription completed and saved!")
            self.progress['value'] = 100
            self.root.update()
            self.current_segments = segments
            self.current_speaker_segments = speaker_segments
            if hasattr(self.speaker_db, 'embedding_model') and self.speaker_db.embedding_model is not None:
                if hasattr(self, 'label_speakers_button') and self.label_speakers_button:
                    self.label_speakers_button.pack_forget()
                self.label_speakers_button = tk.Button(
                    self.button_container,
                    text="Label Speakers",
                    command=lambda: self.start_label_speakers(num_speakers, file_path, txt_output, json_output)
                )
                self.label_speakers_button.pack(pady=10, padx=10, fill=tk.X)
                messagebox.showinfo(
                    "Transcription Complete",
                    f"Transcription completed and saved to:\n{txt_output}\n\nDetailed JSON output saved to:\n{json_output}\n\nClick 'Label Speakers' to identify speakers."
                )
            else:
                messagebox.showinfo(
                    "Transcription Complete",
                    f"Transcription completed and saved to:\n{txt_output}\n\nDetailed JSON output saved to:\n{json_output}\n\nSpeaker labeling is unavailable due to embedding model initialization failure."
                )
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror(
                "Error",
                f"An error occurred: {str(e)}"
            )
        finally:
            self.start_button.config(state='normal')

    def start_label_speakers(self, num_speakers, file_path, txt_output, json_output):
        """Start the label_speakers process in a separate thread to prevent GUI freezing."""
        self.label_speakers_button.config(state='disabled')
        # Show and reset the labeling progress
        self.labeling_progress_frame.pack(pady=5)
        self.labeling_progress['value'] = 0
        self.labeling_status.config(text="Starting speaker labeling process...")
        self.root.update()

        thread = threading.Thread(target=self.label_speakers, args=(
            num_speakers, file_path, txt_output, json_output
        ))
        thread.daemon = True
        thread.start()

    def update_labeling_progress(self, value, status_text):
        """Update the labeling progress bar and status text."""
        self.labeling_progress['value'] = value
        self.labeling_status.config(text=status_text)
        self.root.update()

    def label_speakers(self, num_speakers, file_path, txt_output, json_output):
        """Perform speaker diarization and labeling after transcription."""
        try:
            self.update_labeling_progress(5, "Extracting speaker embeddings with caching...")
            audio = self.current_audio_data
            sr = self.current_sample_rate
            segments = self.current_segments
            # Skip processing if too few segments
            if len(segments) < 10:
                self.update_status("Too few segments for speaker diarization. Assigning all to one speaker.")
                for seg in segments:
                    seg["speaker"] = "SPEAKER_00"
                speaker_segments = {"SPEAKER_00": [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} 
                                                  for seg in segments]}
                self.current_speaker_segments = speaker_segments
                # Only show dialog if there are valid segments
                if any(len(segs) > 0 for segs in speaker_segments.values()):
                    self._show_labeling_dialog()
                else:
                    self.update_status("No valid segments for speaker labeling. Skipping dialog.")
                    messagebox.showinfo("Speaker Labeling", "No valid segments for speaker labeling. All assigned to one speaker.")
                return
            # Resample if needed
            if sr != 16000:
                self.update_labeling_progress(10, "Resampling audio to 16 kHz for embedding extraction...")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Prepare segments for batch processing
            segment_batches = []
            current_batch = []
            segment_times = []
            
            for seg in segments:
                start = seg["start"]
                end = seg["end"]
                duration = end - start
                if duration < OPTIMIZED_SETTINGS['MIN_SEGMENT_DURATION']:
                    continue
                    
                start_idx = int(start * sr)
                end_idx = int(end * sr)
                if end_idx <= start_idx:
                    continue
                    
                segment_audio = audio[start_idx:end_idx]
                if len(segment_audio) == 0:
                    continue
                    
                current_batch.append(segment_audio)
                segment_times.append((start, end))
                
                if len(current_batch) >= OPTIMIZED_SETTINGS['EMBEDDING_BATCH_SIZE']:
                    segment_batches.append(current_batch)
                    current_batch = []
            
            if current_batch:
                segment_batches.append(current_batch)

            # Process batches with caching
            segment_embeddings = []
            total_batches = len(segment_batches)
            processed_segments = 0
            start_time = time.time()

            for batch_idx, batch in enumerate(segment_batches):
                progress = 15 + int(30 * (batch_idx + 1) / max(1, total_batches))
                self.update_labeling_progress(progress, 
                    f"Processing embedding batch {batch_idx + 1}/{total_batches} (with caching)...")
                
                # Get embeddings with caching
                batch_embeddings = self.speaker_db.encode_batch(batch, [sr] * len(batch))
                if batch_embeddings is not None:
                    segment_embeddings.extend(batch_embeddings)
                
                processed_segments += len(batch)
                elapsed = time.time() - start_time
                eta = (elapsed / processed_segments) * (len(segment_times) - processed_segments) if processed_segments > 0 else 0
                
                cache_stats = f"Cache hits: {self.speaker_db.cache_hits}, misses: {self.speaker_db.cache_misses}"
                self.update_status(f"Processed {processed_segments}/{len(segment_times)} segments | "
                                 f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | {cache_stats}")

            if not segment_embeddings:
                self.update_status("No embeddings extracted. Skipping speaker diarization.")
                return

            # Perform optimized clustering
            self.update_status("Performing optimized clustering...")
            self.update_labeling_progress(50, "Clustering embeddings...")
            
            labels = self.speaker_db.optimize_clustering(
                np.stack(segment_embeddings),
                num_speakers if num_speakers else None
            )
            
            # Assign labels to segments
            speaker_segments = {}
            for (start, end), label in zip(segment_times, labels):
                speaker = f"SPEAKER_{label:02d}"
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                    
                for seg in self.current_segments:
                    if abs(seg["start"] - start) < 0.1 and abs(seg["end"] - end) < 0.1:
                        seg["speaker"] = speaker
                        speaker_segments[speaker].append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"]
                        })

            self.current_speaker_segments = speaker_segments
            
            # Show labeling dialog
            self._show_labeling_dialog()
            
        except Exception as e:
            self.update_status(f"Error during speaker labeling: {str(e)}")
            def show_error():
                messagebox.showerror(
                    "Error",
                    f"An error occurred during speaker labeling: {str(e)}"
                )
                self.label_speakers_button.config(state='normal')
                self.labeling_progress_frame.pack_forget()
            self.root.after(0, show_error)

    def _show_labeling_dialog(self):
        """Show the speaker labeling dialog with optimized speaker selection."""
        self.update_status("Preparing speaker labeling dialog...")
        self.update_labeling_progress(90, "Preparing speaker labeling dialog...")
        
        # For each speaker, select the most representative segments
        speaker_samples = {}
        for speaker, segs in self.current_speaker_segments.items():
            # "The Show Must Go On" - select diverse samples
            seen = set()
            samples = []
            for seg in segs:
                key = (seg["text"], seg["start"], seg["end"])
                if key not in seen:
                    samples.append((seg["text"], (seg["start"], seg["end"])))
                    seen.add(key)
                if len(samples) == 3:  # Show up to 3 different segments per speaker
                    break
            speaker_samples[speaker] = samples

        # Get speaker embeddings for matching
        speaker_embeddings = {}
        for speaker, segs in self.current_speaker_segments.items():
            # Use the first segment's embedding as representative
            if segs:
                start, end = segs[0]["start"], segs[0]["end"]
                start_idx = int(start * self.current_sample_rate)
                end_idx = int(end * self.current_sample_rate)
                audio = self.current_audio_data[start_idx:end_idx]
                embedding = self.speaker_db.encode_batch([audio], [self.current_sample_rate])
                if embedding is not None:
                    speaker_embeddings[speaker] = embedding[0]

        def show_dialog():
            self.update_status("Opening speaker labeling dialog...")
            try:
                speaker_dialog = SpeakerLabelDialog(
                    self.root,
                    speaker_samples,
                    speaker_embeddings,
                    self.speaker_db,
                    self.current_audio_data,
                    self.current_sample_rate
                )
                self.root.wait_window(speaker_dialog.dialog)
                
                # Update speaker labels and save to database
                self.update_status("Updating speaker labels...")
                for speaker, speaker_segs in self.current_speaker_segments.items():
                    name = speaker_dialog.speaker_names.get(speaker)
                    if name:
                        # Update speaking time
                        total_duration = sum(float(seg["end"]) - float(seg["start"]) 
                                          for seg in speaker_segs)
                        self.speaker_db.update_speaking_time(name, total_duration)
                        
                        # Update segment labels
                        for seg in speaker_segs:
                            if "speaker" in seg and seg["speaker"] == speaker:
                                seg["speaker"] = name
                
                # Save updated transcription
                self.update_status("Saving updated transcription...")
                self.update_labeling_progress(98, "Saving updated transcription...")
                txt_output, json_output = self.save_transcription(
                    self.current_audio_file,
                    self.current_segments,
                    self.current_speaker_segments,
                    speaker_embeddings
                )
                
                self.update_labeling_progress(100, "Labeling complete!")
                messagebox.showinfo(
                    "Labeling Complete",
                    f"Speaker labeling completed and updated transcription saved to:\n{txt_output}\n\n"
                    f"Updated JSON output saved to:\n{json_output}"
                )
                
            except Exception as e:
                self.update_status(f"Error in speaker labeling dialog: {str(e)}")
                messagebox.showerror("Error", f"An error occurred in the speaker labeling dialog: {str(e)}")
            finally:
                self.label_speakers_button.config(state='normal')
                self.labeling_progress_frame.pack_forget()
        
        self.root.after(0, show_dialog)

    def save_transcription(self, file_path: str, segments, speaker_segments, speaker_embeddings):
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(file_path), "transcriptions")
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_output = os.path.join(output_dir, f"{base_name}_{timestamp}.txt")
        json_output = os.path.join(output_dir, f"{base_name}_{timestamp}.json")

        # Save text output
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write("Transcription result with speakers:\n")
            f.write("-" * 50 + "\n")
            for speaker, segs in speaker_segments.items():
                f.write(f"\nSpeaker: {speaker}\n")
                for seg in segs:
                    start_time = self.format_time(seg["start"])
                    end_time = self.format_time(seg["end"])
                    f.write(f"[{start_time} - {end_time}] {seg['text']}\n")
            f.write("-" * 50 + "\n")

        # Save JSON output with embeddings for future labeling
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump({
                "segments": segments,
                "speaker_segments": speaker_segments,
                "speaker_embeddings": speaker_embeddings
            }, f, indent=2, ensure_ascii=False)

        # Show the transcription in the status text
        self.update_status("\nTranscription result:")
        for speaker, segs in speaker_segments.items():
            self.update_status(f"\nSpeaker: {speaker}")
            for seg in segs[:3]:  # Show first 3 segments per speaker
                start_time = self.format_time(seg["start"])
                end_time = self.format_time(seg["end"])
                self.update_status(f"[{start_time} - {end_time}] {seg['text']}")
            if len(segs) > 3:
                self.update_status("...")

        return txt_output, json_output

    def format_time(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def postprocess_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        # Capitalize first letter
        text = text[0].upper() + text[1:]
        # Add period if missing
        if text[-1] not in '.!?':
            text += '.'
        return text

    def play_segment(self, start, end):
        if self.current_audio_data is None or self.current_sample_rate is None:
            messagebox.showerror("Error", "No audio loaded.")
            return
        start_idx = int(start * self.current_sample_rate)
        end_idx = int(end * self.current_sample_rate)
        segment = self.current_audio_data[start_idx:end_idx]
        with self._playback_lock:
            try:
                sd.stop()
                sd.play(segment, self.current_sample_rate)
            except Exception as e:
                messagebox.showerror("Error", f"Playback failed: {str(e)}")

    def import_relabeled_transcript(self):
        json_path = filedialog.askopenfilename(
            title="Select Relabeled Transcript JSON File",
            filetypes=[("JSON Files", "*.json")]
        )
        if not json_path:
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        segments = transcript.get("segments", [])
        speaker_embeddings = transcript.get("speaker_embeddings", {})
        # For each speaker, add embeddings to the database
        for speaker, embedding in speaker_embeddings.items():
            if isinstance(embedding, list):
                self.speaker_db.add_speaker(speaker, np.array(embedding), "imported")
        messagebox.showinfo("Import Complete", "Relabeled transcript imported and speaker database updated.")

    def run_optimizer(self):
        # Run DictosOptimizer.py as a subprocess
        self.update_status("Running optimizer. This may take several minutes...")
        try:
            subprocess.run([sys.executable, "DictosOptimizer.py"], check=True)
            self.update_status("Optimization complete. Please restart the application to use new settings.")
            messagebox.showinfo("Optimization Complete", "Optimization complete. Please restart the application to use new settings.")
        except Exception as e:
            self.update_status(f"Optimizer failed: {str(e)}")
            messagebox.showerror("Optimizer Failed", f"Optimizer failed: {str(e)}")

    def reset_settings(self):
        # Remove optimizer_config.json and reload defaults
        try:
            if os.path.exists(SETTINGS_FILE):
                os.remove(SETTINGS_FILE)
            self.update_status("Settings reset to defaults. Please restart the application.")
            messagebox.showinfo("Settings Reset", "Settings reset to defaults. Please restart the application.")
        except Exception as e:
            self.update_status(f"Failed to reset settings: {str(e)}")
            messagebox.showerror("Reset Failed", f"Failed to reset settings: {str(e)}")

if __name__ == "__main__":
    print("Starting main application...")
    root = tk.Tk()
    app = TranscriptionApp(root)
    print("Application initialized, entering main loop...")
    root.mainloop()
    print("Application closed.")