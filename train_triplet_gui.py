import logging
from pathlib import Path
import uuid

# Remove top-level logging setup and session ID
# ... existing code ...

import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import numpy as np
from models import CrowResNetEmbedder, CrowMultiModalEmbedder
from train_triplet_resnet import CrowTripletDataset, TripletLoss, compute_metrics
from torchvision import transforms
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue
import time
import sys

# Remove diagnostic printout and logging at startup
python_exec = sys.executable
torch_version = torch.__version__
cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
device = torch.device('cuda' if cuda_available else 'cpu')
pin_memory_setting = True if cuda_available else False

# Remove print statements and logging calls here
# ... existing code ...

class TrainingMetrics:
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.same_crow_mean = []
        self.same_crow_std = []
        self.diff_crow_mean = []
        self.diff_crow_std = []
        self.val_similarities = []
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_time = 0
        self.start_time = None

class TrainingGUI:
    def __init__(self, root, logger, session_id):
        self.logger = logger
        self.session_id = session_id
        self.logger.info("Initializing Training GUI")
        self.root = root
        self.root.title("Crow Training GUI")
        
        # Initialize training state
        self.training = False
        self.paused = False
        self.metrics = TrainingMetrics()
        self.plot_queue = queue.Queue()
        
        self.logger.debug("Creating GUI components")
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left panel for controls
        self.left_panel = ttk.Frame(self.main_frame, padding="5")
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Directory selection
        dir_frame = ttk.LabelFrame(self.left_panel, text="Directories", padding="5")
        dir_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Input directory (crow_crops)
        ttk.Label(dir_frame, text="Crow Crops Directory:").grid(row=0, column=0, sticky=tk.W)
        self.crop_dir_var = tk.StringVar(value="crow_crops")
        ttk.Entry(dir_frame, textvariable=self.crop_dir_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._select_crop_dir).grid(row=0, column=2)
        
        # Audio directory
        ttk.Label(dir_frame, text="Audio Directory:").grid(row=1, column=0, sticky=tk.W)
        self.audio_dir_var = tk.StringVar(value="crow_audio")
        ttk.Entry(dir_frame, textvariable=self.audio_dir_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._select_audio_dir).grid(row=1, column=2)
        
        # Output directory
        ttk.Label(dir_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W)
        self.output_dir_var = tk.StringVar(value="training_output")
        ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=40).grid(row=2, column=1, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self._select_output_dir).grid(row=2, column=2)
        
        # Training parameters
        params_frame = ttk.LabelFrame(self.left_panel, text="Training Parameters", padding="5")
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=0, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(params_frame, from_=8, to=128, textvariable=self.batch_size_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=1e-4)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Epochs
        ttk.Label(params_frame, text="Max Epochs:").grid(row=2, column=0, sticky=tk.W)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Spinbox(params_frame, from_=1, to=1000, textvariable=self.epochs_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Validation split
        ttk.Label(params_frame, text="Validation Split:").grid(row=3, column=0, sticky=tk.W)
        self.val_split_var = tk.DoubleVar(value=0.2)
        ttk.Scale(params_frame, from_=0.1, to=0.5, variable=self.val_split_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=3, column=1, padx=5)
        ttk.Label(params_frame, textvariable=self.val_split_var).grid(row=3, column=2)
        
        # Early stopping patience
        ttk.Label(params_frame, text="Early Stopping Patience:").grid(row=4, column=0, sticky=tk.W)
        self.patience_var = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=1, to=20, textvariable=self.patience_var, width=5).grid(row=4, column=1, sticky=tk.W, padx=5)
        
        # Model parameters
        model_frame = ttk.LabelFrame(self.left_panel, text="Model Parameters", padding="5")
        model_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Embedding dimension
        ttk.Label(model_frame, text="Embedding Dimension:").grid(row=0, column=0, sticky=tk.W)
        self.embed_dim_var = tk.IntVar(value=512)
        ttk.Spinbox(model_frame, from_=128, to=1024, textvariable=self.embed_dim_var, width=5).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Triplet loss margin
        ttk.Label(model_frame, text="Triplet Loss Margin:").grid(row=1, column=0, sticky=tk.W)
        self.margin_var = tk.DoubleVar(value=1.0)
        ttk.Scale(model_frame, from_=0.1, to=2.0, variable=self.margin_var, 
                 orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=5)
        ttk.Label(model_frame, textvariable=self.margin_var).grid(row=1, column=2)
        
        # Control buttons
        control_frame = ttk.Frame(self.left_panel)
        control_frame.grid(row=3, column=0, columnspan=2, pady=5)
        
        self.start_button = ttk.Button(control_frame, text="Start Training", command=self._start_training)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = ttk.Button(control_frame, text="Pause", command=self._pause_training, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self._stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, padx=5)
        
        # Progress tracking
        progress_frame = ttk.LabelFrame(self.left_panel, text="Progress", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        
        # Training metrics
        metrics_frame = ttk.LabelFrame(self.left_panel, text="Current Metrics", padding="5")
        metrics_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.metrics_labels = {}
        metrics = [
            ('epoch', 'Epoch:'),
            ('train_loss', 'Train Loss:'),
            ('val_loss', 'Val Loss:'),
            ('same_crow_sim', 'Same Crow Similarity:'),
            ('diff_crow_sim', 'Diff Crow Similarity:'),
            ('time_elapsed', 'Time Elapsed:'),
            ('best_val_loss', 'Best Val Loss:')
        ]
        
        for i, (key, label) in enumerate(metrics):
            ttk.Label(metrics_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=1)
            self.metrics_labels[key] = ttk.Label(metrics_frame, text="--")
            self.metrics_labels[key].grid(row=i, column=1, sticky=tk.W, padx=5, pady=1)
        
        # Right panel - Plots
        plots_frame = ttk.LabelFrame(self.main_frame, text="Training Plots", padding="5")
        plots_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Create figure for plots
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create subplots
        self.loss_ax = self.fig.add_subplot(211)
        self.sim_ax = self.fig.add_subplot(212)
        self.fig.tight_layout()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.left_panel.columnconfigure(1, weight=1)
        plots_frame.columnconfigure(0, weight=1)
        plots_frame.rowconfigure(0, weight=1)
        
        # Set window size
        self.root.geometry("1200x1000")
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start plot update thread
        self._start_plot_updater()
        
        self.logger.info("GUI initialization complete")

    def _select_crop_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.crop_dir_var.set(directory)
    
    def _select_audio_dir(self):
        """Select audio directory."""
        directory = filedialog.askdirectory()
        if directory:
            self.audio_dir_var.set(directory)
    
    def _select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir_var.set(directory)
    
    def _start_training(self):
        """Start the training process."""
        self.logger.info("Starting training process")
        
        if not self.crop_dir_var.get():
            self.logger.error("No crow crops directory selected")
            messagebox.showerror("Error", "Please select a crow crops directory")
            return
        
        if not self.output_dir_var.get():
            self.logger.error("No output directory selected")
            messagebox.showerror("Error", "Please select an output directory")
            return
        
        # Create output directory
        output_dir = self.output_dir_var.get()
        self.logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training config
        config = {
            'crop_dir': self.crop_dir_var.get(),
            'audio_dir': self.audio_dir_var.get(),
            'output_dir': output_dir,
            'batch_size': self.batch_size_var.get(),
            'learning_rate': self.lr_var.get(),
            'epochs': self.epochs_var.get(),
            'val_split': self.val_split_var.get(),
            'embed_dim': self.embed_dim_var.get(),
            'margin': self.margin_var.get(),
            'patience': self.patience_var.get(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id
        }
        
        self.logger.info("Training configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        config_path = os.path.join(output_dir, 'training_config.json')
        self.logger.info(f"Saving training config to: {config_path}")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Reset metrics
        self.logger.debug("Resetting training metrics")
        self.metrics = TrainingMetrics()
        self.metrics.start_time = time.time()
        
        # Update UI
        self.logger.debug("Updating UI for training start")
        self.training = True
        self.paused = False
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        self.progress_label.configure(text="Starting training...")
        self.status_var.set("Training in progress...")
        
        # Start training thread
        self.logger.info("Starting training thread")
        self.training_thread = threading.Thread(target=self._training_loop, args=(config,))
        self.training_thread.start()
    
    def _training_loop(self, config):
        """Main training loop running in a separate thread."""
        try:
            self.logger.info("Initializing training loop")
            
            # Device setup with improved logging
            device = torch.device(config['device'])
            if device.type == 'cuda':
                if not torch.cuda.is_available():
                    self.logger.warning("CUDA requested but not available, falling back to CPU")
                    device = torch.device('cpu')
                    config['device'] = 'cpu'
                else:
                    torch.cuda.empty_cache()  # Clear any existing allocations
                    self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.logger.info("Using CPU device")
            
            # Create datasets
            self.logger.info(f"Loading dataset from: {config['crop_dir']}")
            full_dataset = CrowTripletDataset(
                config['crop_dir'],
                audio_dir=config['audio_dir'] if config['audio_dir'] else None
            )
            self.logger.info(f"Total dataset size: {len(full_dataset)}")
            if config['audio_dir']:
                self.logger.info(f"Crows with audio: {len(full_dataset.crow_to_audio)}")
            
            val_size = int(len(full_dataset) * config['val_split'])
            train_size = len(full_dataset) - val_size
            self.logger.info(f"Train size: {train_size}, Validation size: {val_size}")
            
            train_dataset, val_dataset = random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create data loaders with device-aware settings
            self.logger.info("Creating data loaders")
            pin_memory = device.type == 'cuda'  # Only use pin_memory with CUDA
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=0,  # Keep at 0 for stability
                pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=0,  # Keep at 0 for stability
                pin_memory=pin_memory
            )
            
            # Model initialization with improved error handling
            self.logger.info("Initializing model")
            try:
                model = CrowMultiModalEmbedder(
                    visual_embed_dim=config['embed_dim'] // 2,
                    audio_embed_dim=config['embed_dim'] // 2,
                    final_embed_dim=config['embed_dim'],
                    device=device
                )
                self.logger.info(f"Model initialized on device: {device}")
                
                # Load checkpoint if available
                best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
                if os.path.exists(best_model_path):
                    self.logger.info(f"Loading model weights from checkpoint: {best_model_path}")
                    try:
                        # Load to CPU first to avoid device mismatch
                        state_dict = torch.load(best_model_path, map_location='cpu')
                        model_state_dict = model.state_dict()
                        filtered_state_dict = {}
                        
                        # Filter and validate state dict
                        for k, v in state_dict.items():
                            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                                filtered_state_dict[k] = v
                            else:
                                self.logger.warning(f"Skipping incompatible key: {k}")
                        
                        # Load filtered state dict
                        model.load_state_dict(filtered_state_dict, strict=False)
                        self.logger.info(f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} model parameters")
                        
                        # Verify model is on correct device after loading
                        model = model.to(device)
                        param_devices = set(str(p.device) for p in model.parameters())
                        if device.type == 'cuda':
                            if not all(d.startswith('cuda') for d in param_devices):
                                raise RuntimeError("Model parameters not on CUDA after loading checkpoint")
                        else:
                            if not all(d == str(device) for d in param_devices):
                                raise RuntimeError(f"Model parameters not on {device} after loading checkpoint")
                            
                    except Exception as e:
                        self.logger.error(f"Failed to load checkpoint: {str(e)}")
                        self.logger.warning("Using randomly initialized model")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize model: {str(e)}")
                messagebox.showerror("Model Initialization Error", f"Failed to initialize model: {str(e)}")
                self.training = False
                self.root.after(0, self._training_complete)
                return
            
            self.logger.info(f"Model architecture:\n{model}")
            
            # Initialize optimizer and loss function
            self.logger.info("Initializing optimizer and loss function")
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = TripletLoss(margin=config['margin'], mining_type='hard')
            
            # Training loop with improved device handling
            self.logger.info("Starting training epochs")
            patience_counter = 0
            for epoch in range(config['epochs']):
                if not self.training:
                    self.logger.info("Training stopped by user")
                    break
                    
                self.logger.info(f"Starting epoch {epoch + 1}/{config['epochs']}")
                
                # Training phase
                model.train()
                total_loss = 0
                batch_count = 0
                
                for batch_idx, (imgs, audio, _) in enumerate(train_loader):
                    # Check for pause/stop
                    while self.paused:
                        time.sleep(0.1)
                        if not self.training:
                            break
                    if not self.training:
                        break
                        
                    try:
                        # Unpack and move data to device efficiently
                        anchor_imgs, pos_imgs, neg_imgs = imgs
                        anchor_audio, pos_audio, neg_audio = audio
                        
                        # Move tensors to device only if needed
                        if anchor_imgs.device != device:
                            anchor_imgs = anchor_imgs.to(device)
                            pos_imgs = pos_imgs.to(device)
                            neg_imgs = neg_imgs.to(device)
                        
                        if anchor_audio is not None:
                            if isinstance(anchor_audio, dict):
                                anchor_audio = {k: v.to(device) if v.device != device else v 
                                              for k, v in anchor_audio.items()}
                                pos_audio = {k: v.to(device) if v.device != device else v 
                                           for k, v in pos_audio.items()}
                                neg_audio = {k: v.to(device) if v.device != device else v 
                                           for k, v in neg_audio.items()}
                            else:
                                if anchor_audio.device != device:
                                    anchor_audio = anchor_audio.to(device)
                                    pos_audio = pos_audio.to(device)
                                    neg_audio = neg_audio.to(device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        anchor_emb = model(anchor_imgs, anchor_audio)
                        pos_emb = model(pos_imgs, pos_audio)
                        neg_emb = model(neg_imgs, neg_audio)
                        
                        # Compute loss
                        loss = criterion(anchor_emb, pos_emb, neg_emb)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        total_loss += loss.item() * anchor_imgs.size(0)
                        batch_count += 1
                        
                        if batch_idx % 10 == 0:
                            self.logger.debug(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                        # UI updates
                        self.root.after(0, self._update_metrics_display)
                        self.root.after(0, self._update_plots)
                        
                    except Exception as e:
                        self.logger.error(f"Error in training batch {batch_idx}: {str(e)}", exc_info=True)
                        continue
                        
                # Compute epoch metrics
                avg_train_loss = total_loss / len(train_loader.dataset)
                self.metrics.train_loss.append(avg_train_loss)
                self.logger.info(f"Epoch {epoch + 1} training loss: {avg_train_loss:.4f}")
                
                # Validation phase
                self.logger.info(f"Starting validation for epoch {epoch + 1}")
                model.eval()
                val_loss = 0
                val_batch_count = 0
                
                with torch.no_grad():
                    for batch_idx, (imgs, audio, _) in enumerate(val_loader):
                        try:
                            # Unpack and move data to device efficiently
                            anchor_imgs, pos_imgs, neg_imgs = imgs
                            anchor_audio, pos_audio, neg_audio = audio
                            
                            # Move tensors to device only if needed
                            if anchor_imgs.device != device:
                                anchor_imgs = anchor_imgs.to(device)
                                pos_imgs = pos_imgs.to(device)
                                neg_imgs = neg_imgs.to(device)
                            
                            if anchor_audio is not None:
                                if isinstance(anchor_audio, dict):
                                    anchor_audio = {k: v.to(device) if v.device != device else v 
                                                  for k, v in anchor_audio.items()}
                                    pos_audio = {k: v.to(device) if v.device != device else v 
                                               for k, v in pos_audio.items()}
                                    neg_audio = {k: v.to(device) if v.device != device else v 
                                               for k, v in neg_audio.items()}
                                else:
                                    if anchor_audio.device != device:
                                        anchor_audio = anchor_audio.to(device)
                                        pos_audio = pos_audio.to(device)
                                        neg_audio = neg_audio.to(device)
                            
                            # Forward pass
                            anchor_emb = model(anchor_imgs, anchor_audio)
                            pos_emb = model(pos_imgs, pos_audio)
                            neg_emb = model(neg_imgs, neg_audio)
                            
                            # Compute loss
                            loss = criterion(anchor_emb, pos_emb, neg_emb)
                            val_loss += loss.item() * anchor_imgs.size(0)
                            val_batch_count += 1
                            
                            if batch_idx % 10 == 0:
                                self.logger.debug(f"Validation batch {batch_idx}, Loss: {loss.item():.4f}")
                                
                        except Exception as e:
                            self.logger.error(f"Error in validation batch {batch_idx}: {str(e)}", exc_info=True)
                            continue
                            
                # Compute validation metrics
                avg_val_loss = val_loss / len(val_loader.dataset)
                self.metrics.val_loss.append(avg_val_loss)
                self.logger.info(f"Epoch {epoch + 1} validation loss: {avg_val_loss:.4f}")
                
                # Compute additional validation metrics
                self.logger.info("Computing validation metrics")
                try:
                    val_metrics, similarities, _ = compute_metrics(model, val_loader, device)
                    self.logger.info(f"Validation metrics: {val_metrics}")
                    self.metrics.val_similarities.append(similarities)
                    self.metrics.same_crow_mean.append(val_metrics['same_crow_mean'])
                    self.metrics.same_crow_std.append(val_metrics['same_crow_std'])
                    self.metrics.diff_crow_mean.append(val_metrics['diff_crow_mean'])
                    self.metrics.diff_crow_std.append(val_metrics['diff_crow_std'])
                except Exception as e:
                    self.logger.error(f"Error computing validation metrics: {str(e)}", exc_info=True)
                    val_metrics = {}
                    similarities = []
                
                # Update metrics
                self.metrics.current_epoch = epoch + 1
                self.metrics.training_time = time.time() - self.metrics.start_time
                
                # Update UI
                self.root.after(0, self._update_metrics_display)
                self.root.after(0, self._update_plots)
                
                # Save checkpoint
                checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
                self.logger.info(f"Saving checkpoint to: {checkpoint_path}")
                try:
                    # Save checkpoint to CPU to ensure compatibility
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'metrics': val_metrics
                    }
                    torch.save(checkpoint, checkpoint_path)
                except Exception as e:
                    self.logger.error(f"Error saving checkpoint: {str(e)}", exc_info=True)
                
                # Early stopping
                if avg_val_loss < self.metrics.best_val_loss:
                    self.metrics.best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_path = os.path.join(config['output_dir'], 'best_model.pth')
                    self.logger.info(f"Saving new best model to: {best_model_path}")
                    try:
                        # Save best model to CPU to ensure compatibility
                        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_model_path)
                    except Exception as e:
                        self.logger.error(f"Error saving best model: {str(e)}", exc_info=True)
                else:
                    patience_counter += 1
                    self.logger.info(f"Early stopping patience: {patience_counter}/{config['patience']}")
                    if patience_counter >= config['patience']:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                # Update progress
                progress = ((epoch + 1) / config['epochs']) * 100
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.progress_label.configure(
                    text=f"Epoch {epoch + 1}/{config['epochs']} "
                         f"(Best Val Loss: {self.metrics.best_val_loss:.4f})"
                ))
                
                # Clear GPU memory after each epoch if using CUDA
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    self.logger.info("Cleared GPU cache")
                
            self.logger.info("Training loop complete. Saving final results.")
            
            # Save final metrics
            self._save_training_results(config)
            
            # Update UI
            self.root.after(0, self._training_complete)
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}", exc_info=True)
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.root.after(0, self._training_complete)
    
    def _update_metrics_display(self):
        """Update the metrics display in the GUI."""
        self.logger.info("_update_metrics_display called")
        self.logger.info(f"Current metrics: epoch={self.metrics.current_epoch}, train_loss={self.metrics.train_loss}, val_loss={self.metrics.val_loss}, same_crow_mean={self.metrics.same_crow_mean}, diff_crow_mean={self.metrics.diff_crow_mean}, time_elapsed={self.metrics.training_time}, best_val_loss={self.metrics.best_val_loss}")
        self.metrics_labels['epoch'].configure(text=str(self.metrics.current_epoch))
        self.metrics_labels['train_loss'].configure(text=f"{self.metrics.train_loss[-1]:.4f}" if self.metrics.train_loss else "--")
        self.metrics_labels['val_loss'].configure(text=f"{self.metrics.val_loss[-1]:.4f}" if self.metrics.val_loss else "--")
        if self.metrics.same_crow_mean and self.metrics.same_crow_std:
            self.metrics_labels['same_crow_sim'].configure(
                text=f"{self.metrics.same_crow_mean[-1]:.4f} ± {self.metrics.same_crow_std[-1]:.4f}"
            )
        else:
            self.metrics_labels['same_crow_sim'].configure(text="--")
        if self.metrics.diff_crow_mean and self.metrics.diff_crow_std:
            self.metrics_labels['diff_crow_sim'].configure(
                text=f"{self.metrics.diff_crow_mean[-1]:.4f} ± {self.metrics.diff_crow_std[-1]:.4f}"
            )
        else:
            self.metrics_labels['diff_crow_sim'].configure(text="--")
        self.metrics_labels['time_elapsed'].configure(
            text=f"{self.metrics.training_time/60:.1f} min"
        )
        self.metrics_labels['best_val_loss'].configure(
            text=f"{self.metrics.best_val_loss:.4f}" if self.metrics.best_val_loss != float('inf') else "--"
        )
    
    def _update_plots(self):
        """Update the training plots."""
        self.logger.info("_update_plots called")
        self.logger.info(f"Plotting metrics: train_loss={self.metrics.train_loss}, val_loss={self.metrics.val_loss}, same_crow_mean={self.metrics.same_crow_mean}, diff_crow_mean={self.metrics.diff_crow_mean}")
        # Put plot update in queue
        self.plot_queue.put(True)
    
    def _start_plot_updater(self):
        """Start the plot update thread."""
        def update_plots():
            while True:
                try:
                    # Wait for plot update signal
                    self.plot_queue.get(timeout=0.1)
                    
                    # Update loss plot
                    self.loss_ax.clear()
                    self.loss_ax.plot(self.metrics.train_loss, label='Train')
                    self.loss_ax.plot(self.metrics.val_loss, label='Validation')
                    self.loss_ax.set_title('Loss')
                    self.loss_ax.set_xlabel('Epoch')
                    self.loss_ax.set_ylabel('Loss')
                    self.loss_ax.legend()
                    
                    # Update similarity plot
                    self.sim_ax.clear()
                    self.sim_ax.plot(self.metrics.same_crow_mean, label='Same Crow')
                    self.sim_ax.plot(self.metrics.diff_crow_mean, label='Different Crow')
                    self.sim_ax.fill_between(
                        range(len(self.metrics.same_crow_mean)),
                        np.array(self.metrics.same_crow_mean) - np.array(self.metrics.same_crow_std),
                        np.array(self.metrics.same_crow_mean) + np.array(self.metrics.same_crow_std),
                        alpha=0.2
                    )
                    self.sim_ax.fill_between(
                        range(len(self.metrics.diff_crow_mean)),
                        np.array(self.metrics.diff_crow_mean) - np.array(self.metrics.diff_crow_std),
                        np.array(self.metrics.diff_crow_mean) + np.array(self.metrics.diff_crow_std),
                        alpha=0.2
                    )
                    self.sim_ax.set_title('Embedding Similarities')
                    self.sim_ax.set_xlabel('Epoch')
                    self.sim_ax.set_ylabel('Cosine Similarity')
                    self.sim_ax.legend()
                    
                    # Update canvas
                    self.fig.tight_layout()
                    self.canvas.draw()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Plot update error: {str(e)}", exc_info=True)
        
        # Start plot update thread
        self.plot_thread = threading.Thread(target=update_plots, daemon=True)
        self.plot_thread.start()
    
    def _save_training_results(self, config):
        """Save training results and metrics."""
        # Save metrics history
        metrics_history = {
            'train_loss': self.metrics.train_loss,
            'val_loss': self.metrics.val_loss,
            'same_crow_mean': self.metrics.same_crow_mean,
            'same_crow_std': self.metrics.same_crow_std,
            'diff_crow_mean': self.metrics.diff_crow_mean,
            'diff_crow_std': self.metrics.diff_crow_std,
            'val_similarities': [s.tolist() for s in self.metrics.val_similarities] if self.metrics.val_similarities else []
        }
        with open(os.path.join(config['output_dir'], 'metrics_history.json'), 'w') as f:
            json.dump(metrics_history, f, indent=2)
        # Save final plot
        plt.figure(figsize=(15, 10))
        # Plot loss
        plt.subplot(2, 2, 1)
        if self.metrics.train_loss and self.metrics.val_loss:
            plt.plot(self.metrics.train_loss, label='Train')
            plt.plot(self.metrics.val_loss, label='Validation')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        else:
            plt.title('Loss (no data)')
        # Plot similarities
        plt.subplot(2, 2, 2)
        if self.metrics.same_crow_mean and self.metrics.diff_crow_mean:
            plt.plot(self.metrics.same_crow_mean, label='Same Crow')
            plt.plot(self.metrics.diff_crow_mean, label='Different Crow')
            if self.metrics.same_crow_std:
                plt.fill_between(
                    range(len(self.metrics.same_crow_mean)),
                    np.array(self.metrics.same_crow_mean) - np.array(self.metrics.same_crow_std),
                    np.array(self.metrics.same_crow_mean) + np.array(self.metrics.same_crow_std),
                    alpha=0.2
                )
            if self.metrics.diff_crow_std:
                plt.fill_between(
                    range(len(self.metrics.diff_crow_mean)),
                    np.array(self.metrics.diff_crow_mean) - np.array(self.metrics.diff_crow_std),
                    np.array(self.metrics.diff_crow_mean) + np.array(self.metrics.diff_crow_std),
                    alpha=0.2
                )
            plt.title('Embedding Similarities')
            plt.xlabel('Epoch')
            plt.ylabel('Cosine Similarity')
            plt.legend()
        else:
            plt.title('Embedding Similarities (no data)')
        # Plot similarity distributions
        plt.subplot(2, 2, 3)
        if self.metrics.val_similarities:
            sns.kdeplot(self.metrics.val_similarities[-1], label='Validation')
            plt.title('Similarity Distribution (Last Epoch)')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Density')
        else:
            self.logger.warning('No validation similarities to plot.')
            plt.title('Similarity Distribution (no data)')
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(config['output_dir'], 'training_metrics.png'))
        plt.close()
    
    def _training_complete(self):
        """Handle training completion."""
        self.training = False
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Training complete")
        
        # Show summary
        summary = (
            f"Training complete!\n\n"
            f"Total epochs: {self.metrics.current_epoch}\n"
            f"Best validation loss: {self.metrics.best_val_loss:.4f}\n"
            f"Training time: {self.metrics.training_time/60:.1f} minutes\n"
            f"Results saved to: {self.output_dir_var.get()}"
        )
        messagebox.showinfo("Training Complete", summary)
    
    def _pause_training(self):
        """Pause/resume training."""
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
        self.status_var.set("Training paused" if self.paused else "Training in progress...")
    
    def _stop_training(self):
        """Stop training."""
        if messagebox.askyesno("Stop Training", "Are you sure you want to stop training?"):
            self.training = False
            self.status_var.set("Stopping training...")
    
    def _on_closing(self):
        """Handle window closing."""
        if self.training:
            if messagebox.askyesno("Quit", "Training is in progress. Are you sure you want to quit?"):
                self.training = False
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    try:
        # Set up logging at the very top, before any other imports
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'train_triplet_gui.log'
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ],
            force=True  # Ensures this config is always used
        )
        logger = logging.getLogger(__name__)
        logger.info('=== Starting new training session ===')
        logger.info('TEST: Logging is working!')

        # Generate session ID
        session_id = str(uuid.uuid4())[:8]
        logger.info(f"SESSION ID: {session_id}")

        # Diagnostic printout and logging at startup
        python_exec = sys.executable
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
        device = torch.device('cuda' if cuda_available else 'cpu')
        pin_memory_setting = True if cuda_available else False

        print(f'Python executable: {python_exec}')
        print(f'Torch version: {torch_version}')
        print(f'CUDA available: {cuda_available}')
        print(f'Device: {device_name}')

        logger.info(f'Python executable: {python_exec}')
        logger.info(f'Torch version: {torch_version}')
        logger.info(f'CUDA available: {cuda_available}')
        logger.info(f'Device: {device_name}')

        logger.info("Starting application")
        root = tk.Tk()
        app = TrainingGUI(root, logger, session_id)
        root.mainloop()
    except Exception as e:
        logger.critical(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 