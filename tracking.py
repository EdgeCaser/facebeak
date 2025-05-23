import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, deque
from db import save_crow_embedding, get_all_crows, get_crow_history, add_behavioral_marker
from sort import Sort
from scipy.spatial.distance import cdist
from models import CrowResNetEmbedder, create_model
from color_normalization import AdaptiveNormalizer, create_normalizer
from multi_view import create_multi_view_extractor
from ultralytics import YOLO
import torchvision
from torchvision.models import resnet18
import torch.nn as nn
import logging
import time
import threading
import gc
from functools import wraps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from pathlib import Path
from contextlib import contextmanager
import signal
import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch.nn.functional as F
from threading import Thread, Event
from multi_view import MultiViewExtractor
from color_normalization import ColorNormalizer
import functools
import platform
import json
from datetime import datetime
import shutil

# Custom exception classes
class TrackingError(Exception):
    """Base exception for tracking errors."""
    pass

class DeviceError(TrackingError):
    """Exception raised for device-related errors."""
    pass

class ModelError(TrackingError):
    """Exception raised for model-related errors."""
    pass

class EmbeddingError(TrackingError):
    """Exception raised for embedding-related errors."""
    pass

class TimeoutException(TrackingError):
    """Exception raised for operation timeouts."""
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PATCH: Platform check for signal.SIGALRM ---
IS_WINDOWS = platform.system() == 'Windows'

def timeout(seconds):
    """Decorator for function timeout."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if IS_WINDOWS:
                # SIGALRM not available, skip timeout
                return func(*args, **kwargs)
            def handler(signum, frame):
                raise TimeoutException(f"Function {func.__name__} timed out after {seconds} seconds")
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

# Load models
print("[INFO] Loading models...")
# Crow embedding model
model = CrowResNetEmbedder(embedding_dim=512)
try:
    model.load_state_dict(torch.load('crow_resnet_triplet.pth'))
    print("[INFO] Loaded trained crow embedding model")
except:
    print("[WARNING] Could not load trained model, using untrained model")
model.eval()

# Toy detection model
try:
    toy_model = YOLO('yolov8n_toys.pt')
    print("[INFO] Loaded toy detection model")
except:
    print("[WARNING] Could not load toy detection model, toy detection disabled")
    toy_model = None

if torch.cuda.is_available():
    model = model.cuda()
    if toy_model:
        toy_model.to('cuda')
    print("[INFO] Models loaded on GPU")
else:
    print("[INFO] Models loaded on CPU")

# Load ResNet model for embeddings
model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print("[INFO] Tracking model loaded on GPU")
else:
    print("[INFO] Tracking model loaded on CPU")

# Define a simple super-resolution model
class SuperResolutionModel(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
    def forward(self, x):
        return self.upsample(x)

# Initialize super-resolution model
sr_model = SuperResolutionModel(scale_factor=2)
if torch.cuda.is_available():
    sr_model = sr_model.cuda()
sr_model.eval()
print("[INFO] Super-resolution model initialized")

def apply_super_resolution(img_tensor, min_size=100):
    """Apply super-resolution to small images."""
    h, w = img_tensor.shape[-2:]
    if h >= min_size and w >= min_size:
        return img_tensor
    with torch.no_grad():
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        enhanced = sr_model(img_tensor)
        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced.cpu()

def compute_embedding(img_tensors):
    """Compute feature embeddings for both head and body regions with improved quality and error handling."""
    try:
        embeddings = {}
        with torch.no_grad():
            # Move tensors to device and normalize
            for key, tensor in img_tensors.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Expected torch.Tensor for {key}, got {type(tensor)}")
                if tensor.dim() != 4:  # Add batch dimension if missing
                    tensor = tensor.unsqueeze(0)
                if tensor.size(1) != 3:  # Ensure RGB channels
                    raise ValueError(f"Expected 3 channels for {key}, got {tensor.size(1)}")
                
                # Normalize tensor
                tensor = tensor.float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                tensor = (tensor - mean) / std
                
                # Move to device
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                img_tensors[key] = tensor
            
            # Compute embeddings
            for key, tensor in img_tensors.items():
                try:
                    features = model(tensor)
                    # Normalize embeddings
                    features = F.normalize(features, p=2, dim=1)
                    embeddings[key] = features.squeeze().cpu().numpy()
                except Exception as e:
                    raise EmbeddingError(f"Error computing embedding for {key}: {str(e)}")
        
        # Combine embeddings with weighted average
        combined = np.concatenate([
            0.7 * embeddings['full'],  # Weight full body more
            0.3 * embeddings['head']   # Weight head less
        ])
        
        # Normalize combined embedding
        combined = combined / np.linalg.norm(combined)
        
        return combined, embeddings
        
    except Exception as e:
        if isinstance(e, EmbeddingError):
            raise
        raise EmbeddingError(f"Error in compute_embedding: {str(e)}")

def extract_normalized_crow_crop(frame, bbox, expected_size=(224, 224)):
    """Extract and normalize a crop of a crow from a frame.
    
    Args:
        frame: Input frame as numpy array
        bbox: Bounding box [x1, y1, x2, y2]
        expected_size: Expected output size (height, width)
        
    Returns:
        dict: Dictionary containing normalized full crop and head crop
    """
    try:
        # Validate input
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
            
        if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
            raise ValueError("Bbox must be a list/tuple/array of 4 values [x1, y1, x2, y2]")
            
        # Convert bbox to integers
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bbox coordinates
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bbox coordinates: {bbox} (x1 >= x2 or y1 >= y2)")
            
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            raise ValueError(f"Invalid bbox coordinates: {bbox} for frame size {frame.shape[1]}x{frame.shape[0]}")
            
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        # Resize to expected size
        crop = cv2.resize(crop, (expected_size[1], expected_size[0]))
        
        # Convert to float32 and normalize to [0, 1]
        crop = crop.astype(np.float32) / 255.0
        
        # Extract head region (top third of the crop)
        head_height = expected_size[0] // 3
        head_crop = crop[:head_height, :]
        
        # Resize head crop to expected size
        head_crop = cv2.resize(head_crop, (expected_size[1], expected_size[0]))
        
        return {
            'full': crop,
            'head': head_crop
        }
        
    except Exception as e:
        logger.error(f"Error extracting normalized crow crop: {str(e)}")
        return None

class EnhancedTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, 
                 embedding_threshold=0.7, model_path=None, conf_threshold=0.5,
                 multi_view_stride=5, strict_mode=False):
        """Initialize the enhanced tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track without detection
            min_hits: Minimum number of detections before a track is confirmed
            iou_threshold: IOU threshold for matching detections to tracks
            embedding_threshold: Threshold for embedding similarity
            model_path: Path to the model weights file
            conf_threshold: Confidence threshold for detections
            multi_view_stride: Number of frames between multi-view updates
            strict_mode: Whether to enforce strict model initialization
        """
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Store parameters first
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.conf_threshold = conf_threshold
        self.multi_view_stride = multi_view_stride
        self.strict_mode = strict_mode
        
        # Initialize models
        self.multi_view_extractor = None
        self.color_normalizer = None
        self._initialize_models(model_path, strict_mode)
        
        # Initialize tracking state
        self.track_history = {}
        self.track_embeddings = {}
        self.track_head_embeddings = {}
        self.track_ages = {}
        self.track_confidences = {}
        self.track_bboxes = {}
        self.track_temporal_consistency = {}
        
        # Initialize processing directory
        self.processing_dir = Path("processing")
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking state
        self.frame_count = 0
        self.track_id_changes = {}
        self.last_cleanup_frame = 0
        self.cleanup_interval = 30
        self.max_track_history = 100
        self.max_behavior_history = 50
        self.behavior_window = 10
        self.max_movement = 100.0
        self.next_id = 0  # Start from 0, will be incremented to 1 on first use
        self.expected_size = (100, 100)
        self.max_retries = 3
        self.retry_delay = 1.0
        self.tracking_file = "tracking_data.json"
        self.tracking_data = {
            "metadata": {
                "last_id": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "crows": {}
        }

        # Initialize SORT tracker
        self.tracker = Sort(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold
        )

        # Assign extract_normalized_crow_crop as instance method
        self.extract_normalized_crow_crop = extract_normalized_crow_crop

    def _configure_logger(self, level=logging.DEBUG):
        """Configure logger with specified level."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # Add new handler with formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        self.logger.debug("Logger configured successfully")

    def set_log_level(self, level):
        """Set logger level dynamically."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self._configure_logger(level)
        self.logger.info(f"Log level set to {logging.getLevelName(level)}")

    @contextmanager
    def device_context(self, device):
        """Context manager for temporary device changes."""
        original_device = self.device
        try:
            # Only allow device changes if CUDA is not available
            if not torch.cuda.is_available():
                self.device = torch.device(device)
                yield
            else:
                # If CUDA is available, ignore device changes
                yield
        finally:
            if not torch.cuda.is_available():
                self.device = original_device

    def to_device(self, device):
        """Move models to specified device."""
        if not torch.cuda.is_available():
            # Only allow CPU if CUDA is not available
            self.model = self.model.to('cpu')
            self.device = torch.device('cpu')
            self.logger.info("Moved model to CPU (CUDA not available)")
            return True
        else:
            # If CUDA is available, ignore device changes
            self.logger.info("Device change ignored: CUDA enforced")
            return True

    def _retry_operation(self, operation, max_retries=None, delay=None):
        """Retry an operation with exponential backoff."""
        if max_retries is None:
            max_retries = self.max_retries
        if delay is None:
            delay = self.retry_delay
        
        last_error = None
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    self.logger.info(f"Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
        
        raise last_error

    def _load_model(self, model_path=None):
        """Load model with improved error handling.
        
        Args:
            model_path: Optional path to model weights
            
        Returns:
            torch.nn.Module: Loaded model
            
        Raises:
            ModelError: If model loading fails
        """
        try:
            if model_path is None:
                # Load default torchvision ResNet18
                import torchvision.models as models
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                # Remove the final classification layer for embedding
                model = torch.nn.Sequential(*list(model.children())[:-1])
                
                # Verify model architecture
                if not isinstance(model, torch.nn.Module):
                    raise ModelError("Invalid model architecture")
                return model
                
            if not os.path.exists(model_path):
                raise ModelError(f"Model file not found: {model_path}")
                
            try:
                # Load model with proper device placement
                model = torch.load(model_path, map_location=self.device)
                if not isinstance(model, torch.nn.Module):
                    raise ModelError("Loaded file is not a valid PyTorch model")
                    
                # Verify model parameters
                if not any(p.requires_grad for p in model.parameters()):
                    self.logger.warning("Model has no trainable parameters")
                    
                return model
                
            except Exception as e:
                raise ModelError(f"Failed to load model from {model_path}: {str(e)}")
                
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    @timeout(5.0)  # 5 second timeout for embedding processing
    def _process_tracking_batch(self, frame, detections, frame_num):
        """Process a batch of detections with improved error handling.
        
        Args:
            frame: Input frame
            detections: Array of detections
            frame_num: Current frame number
            
        Returns:
            np.ndarray: Processed tracks
        """
        try:
            if not isinstance(detections, np.ndarray) or len(detections.shape) != 2:
                raise ValueError("Invalid detections format: expected 2D numpy array")
                
            if not isinstance(frame, np.ndarray) or len(frame.shape) != 3:
                raise ValueError("Invalid frame format: expected 3D numpy array")
                
            # Process detections through SORT
            tracks = self.tracker.update(detections)
            
            # Update track history and embeddings
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'history': deque(maxlen=self.max_track_history),
                        'behaviors': deque(maxlen=self.max_behavior_history),
                        'crow_id': None,
                        'video_path': None,
                        'frame_time': None
                    }
                
                # Add to history
                self.track_history[track_id]['history'].append({
                    'bbox': bbox,
                    'frame_num': frame_num,
                    'confidence': track[4] if len(track) > 4 else 1.0,
                    'embedding_factor': 1.0,
                    'behavior_score': 0.73  # Default behavior score
                })
                
                # Update track age
                self.track_ages[track_id] = frame_num
                
                # Clean up old tracks
                if frame_num - self.last_cleanup_frame >= self.cleanup_interval:
                    self._cleanup_old_tracks(frame_num)
                    self.last_cleanup_frame = frame_num
            
            return tracks
            
        except Exception as e:
            self.logger.error(f"Detection batch processing failed: {str(e)}")
            raise

    def _process_detection_batch(self, frame, detections):
        """Process a batch of detections to compute embeddings.
        
        Args:
            frame: Input frame (numpy array)
            detections: Array of detections in format [x1, y1, x2, y2, score]
            
        Returns:
            Dictionary containing 'full' and 'head' embeddings for each detection
        """
        try:
            if len(detections) == 0:
                return {'full': [], 'head': []}
                
            # Extract crops for each detection
            full_crops = []
            head_crops = []
            
            for det in detections:
                bbox = det[:4]  # Get bbox coordinates
                # Extract normalized crops
                crops = self.extract_normalized_crow_crop(frame, bbox)
                if crops is not None:
                    full_crops.append(crops['full'])
                    head_crops.append(crops['head'])
                else:
                    # If extraction fails, use zero tensors
                    full_crops.append(np.zeros((224, 224, 3), dtype=np.float32))
                    head_crops.append(np.zeros((224, 224, 3), dtype=np.float32))
            
            # Convert to tensors
            full_tensor = torch.from_numpy(np.stack(full_crops)).permute(0, 3, 1, 2).float()
            head_tensor = torch.from_numpy(np.stack(head_crops)).permute(0, 3, 1, 2).float()
            
            # Compute embeddings
            with torch.no_grad():
                if torch.cuda.is_available():
                    full_tensor = full_tensor.cuda()
                    head_tensor = head_tensor.cuda()
                
                # Compute embeddings
                full_embeddings = self.model(full_tensor)
                head_embeddings = self.model(head_tensor)
                
                # Normalize embeddings
                full_embeddings = F.normalize(full_embeddings, p=2, dim=1)
                head_embeddings = F.normalize(head_embeddings, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                full_embeddings = full_embeddings.cpu().numpy()
                head_embeddings = head_embeddings.cpu().numpy()
            
            return {
                'full': [emb for emb in full_embeddings],
                'head': [emb for emb in head_embeddings]
            }
            
        except Exception as e:
            self.logger.error(f"Error in _process_detection_batch: {str(e)}")
            # Return zero embeddings on error
            zero_emb = np.zeros(512, dtype=np.float32)
            return {
                'full': [zero_emb.copy() for _ in range(len(detections))],
                'head': [zero_emb.copy() for _ in range(len(detections))]
            }

    def _cleanup_old_tracks(self, current_frame):
        """Clean up old tracks that haven't been updated recently."""
        try:
            tracks_to_remove = []
            for track_id, age in self.track_ages.items():
                if current_frame - age > self.max_age:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                self._cleanup_track(track_id)
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old tracks: {str(e)}")
            if self.strict_mode:
                raise

    def _initialize_models(self, model_path=None, strict_mode=False):
        """Initialize models with proper error handling and support for strict mode.
        
        Args:
            model_path: Path to model weights file
            strict_mode: Whether to enforce strict model loading
            
        Raises:
            ModelError: If model initialization fails in strict mode
        """
        try:
            # Initialize multi-view extractor
            self.multi_view_extractor = create_multi_view_extractor()
            if self.multi_view_extractor is None:
                if strict_mode:
                    raise ModelError("Model initialization failed: create_multi_view_extractor() returned None")
                self.logger.warning("Model initialization failed, using default model")
                self.multi_view_extractor = create_multi_view_extractor()  # Try again
                
            # Move model to device if it has a 'to' method
            if hasattr(self.multi_view_extractor, 'to'):
                self.multi_view_extractor.to(self.device)
            else:
                # If multi_view_extractor doesn't support device placement, set device attribute
                self.multi_view_extractor.device = self.device
                
            if hasattr(self.multi_view_extractor, 'eval'):
                self.multi_view_extractor.eval()
            
            # Load model weights if provided
            if model_path:
                if not os.path.exists(model_path):
                    if strict_mode:
                        raise ModelError(f"Model file not found: {model_path}")
                    self.logger.warning(f"Model file not found: {model_path}, using default weights")
                else:
                    try:
                        state_dict = torch.load(model_path, map_location=self.device)
                        if not isinstance(state_dict, dict):
                            raise ModelError(f"Invalid model file format: {model_path}")
                        self.multi_view_extractor.load_state_dict(state_dict)
                        self.logger.info(f"Loaded model weights from {model_path}")
                    except Exception as e:
                        if strict_mode:
                            raise ModelError(f"Error loading model weights: {str(e)}")
                        self.logger.warning(f"Error loading model weights: {str(e)}, using default weights")
                        
            # Initialize color normalizer
            self.color_normalizer = create_normalizer()
            if self.color_normalizer is None:
                if strict_mode:
                    raise ModelError("Color normalizer initialization failed")
                self.logger.warning("Color normalizer initialization failed, using default normalizer")
                self.color_normalizer = create_normalizer()  # Try again
                
            # Set device for color normalizer
            if hasattr(self.color_normalizer, 'to'):
                self.color_normalizer.to(self.device)
            else:
                # If normalizer doesn't support device placement, set device attribute
                self.color_normalizer.device = self.device
                
            self.logger.info("Color normalizer initialized successfully")
            
            # Initialize SORT tracker
            self.tracker = Sort(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )
            self.logger.info("SORT tracker initialized successfully")
            
        except Exception as e:
            if strict_mode:
                raise ModelError(f"Model initialization failed: {str(e)}")
            self.logger.error(f"Error initializing models: {str(e)}")
            # Use default models if initialization fails
            self.multi_view_extractor = create_multi_view_extractor()
            self.color_normalizer = create_normalizer()
            if hasattr(self.color_normalizer, 'to'):
                self.color_normalizer.to(self.device)
            self.tracker = Sort(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )

    def _init_multi_view_extractor(self, strict_mode=False):
        """Initialize multi-view extractor with proper error handling."""
        try:
            if not hasattr(self, 'model') or self.model is None:
                if strict_mode:
                    raise ModelError("Model required for multi-view extraction")
                self.logger.warning("Model not available for multi-view extraction")
                self.multi_view_extractor = None
                return
            
            # Create multi-view extractor without stride parameter
            self.multi_view_extractor = create_multi_view_extractor()
            self.logger.info("Multi-view extractor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize multi-view extractor: {str(e)}")
            if strict_mode:
                raise ModelError(f"Multi-view extractor initialization failed: {str(e)}")
            self.multi_view_extractor = None

    def _get_embedding_dim(self, state_dict):
        """Get embedding dimension from state dict."""
        for k, v in state_dict.items():
            if k.endswith('fc.weight'):
                return v.size(0)
        raise ModelError("Could not determine embedding dimension from state dict")

    def _load_state_dict(self, state_dict, strict_mode):
        """Load state dict with improved error handling."""
        if strict_mode:
            self.model.load_state_dict(state_dict)
            return
        
        model_state = self.model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
        
        # Handle size mismatches
        size_mismatches = []
        for k, v in filtered_state.items():
            if v.size() != model_state[k].size():
                size_mismatches.append(f"{k}: expected {model_state[k].size()}, got {v.size()}")
        
        if size_mismatches:
            self.logger.warning("Size mismatches in state dict:")
            for mismatch in size_mismatches:
                self.logger.warning(f"  {mismatch}")
            
            # Adapt mismatched layers
            for k, v in filtered_state.items():
                if v.size() != model_state[k].size():
                    if k.endswith('fc.weight'):
                        if v.size(1) != model_state[k].size(1):
                            raise ModelError(f"Input dimension mismatch in {k}: expected {model_state[k].size(1)}, got {v.size(1)}")
                        new_weight = torch.zeros(model_state[k].size(), device=v.device)
                        min_dim = min(v.size(0), model_state[k].size(0))
                        new_weight[:min_dim] = v[:min_dim]
                        filtered_state[k] = new_weight
                        self.logger.info(f"Adapted {k} weights from {v.size()} to {model_state[k].size()}")
                    elif k.endswith('fc.bias'):
                        new_bias = torch.zeros(model_state[k].size(), device=v.device)
                        min_dim = min(v.size(0), model_state[k].size(0))
                        new_bias[:min_dim] = v[:min_dim]
                        filtered_state[k] = new_bias
                        self.logger.info(f"Adapted {k} bias from {v.size()} to {model_state[k].size()}")
                    elif v.numel() == model_state[k].numel():
                        filtered_state[k] = v.view(model_state[k].size())
                    else:
                        raise ModelError(f"Size mismatch in {k}: expected {model_state[k].size()}, got {v.size()}")
        
        # Load adapted state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state, strict=False)
        
        if missing_keys:
            self.logger.warning("Missing keys in state dict:")
            for key in missing_keys:
                self.logger.warning(f"  {key}")
        
        if unexpected_keys:
            self.logger.warning("Unexpected keys in state dict:")
            for key in unexpected_keys:
                self.logger.warning(f"  {key}")

    @timeout(5.0)  # 5 second timeout for embedding processing
    def _compute_feature_embedding(self, img_tensors, key=None, return_tensor=False):
        """Compute feature embeddings for both head and body regions using the instance's model and device.
        
        Args:
            img_tensors: Dictionary containing 'full' and 'head' tensors or single tensor
            key: Optional key to process single region
            return_tensor: Whether to return PyTorch tensor or numpy array
            
        Returns:
            Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]]:
                Feature embeddings in requested format
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Model not initialized")
                zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                return zero_emb if return_tensor else {'full': zero_emb.detach().cpu().numpy(), 'head': zero_emb.detach().cpu().numpy()}

            with torch.no_grad():
                if key is not None:
                    # Process single region
                    if not isinstance(img_tensors, (torch.Tensor, np.ndarray)):
                        self.logger.error(f"Invalid input type for {key}: {type(img_tensors)}")
                        zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                        return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()

                    # Convert to numpy array if needed
                    if isinstance(img_tensors, torch.Tensor):
                        img_tensors = img_tensors.detach().cpu().numpy()

                    # Ensure uint8 type
                    if img_tensors.dtype != np.uint8:
                        if img_tensors.max() <= 1.0:
                            img_tensors = (img_tensors * 255).astype(np.uint8)
                        else:
                            img_tensors = img_tensors.astype(np.uint8)

                    # Convert to tensor and ensure proper shape
                    img_tensors = torch.from_numpy(img_tensors).float()
                    if img_tensors.dim() != 4:
                        img_tensors = img_tensors.unsqueeze(0)
                    if img_tensors.size(1) != 3:
                        img_tensors = img_tensors.permute(0, 3, 1, 2)  # Convert HWC to CHW if needed

                    # Move to device and normalize
                    img_tensors = img_tensors.to(self.device)
                    if hasattr(self, 'color_normalizer'):
                        # Convert tensor to numpy for normalization
                        img_np = img_tensors.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format
                        normalized = np.stack([self.color_normalizer.normalize(img) for img in img_np])
                        img_tensors = torch.from_numpy(normalized.transpose(0, 3, 1, 2)).to(self.device)  # Convert back to CHW format
                    else:
                        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                        img_tensors = img_tensors / 255.0
                        img_tensors = (img_tensors - mean) / std

                    try:
                        # Ensure model is on correct device
                        if next(self.model.parameters()).device != self.device:
                            self.model = self.model.to(self.device)
                        
                        features = self.model(img_tensors)
                        features = F.normalize(features, p=2, dim=1)
                        features = features.squeeze()
                        
                        # Verify features are on correct device
                        if features.device != self.device:
                            features = features.to(self.device)
                            
                    except Exception as e:
                        self.logger.error(f"Error computing embedding for {key}: {str(e)}")
                        zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                        return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()

                    # Return tensor or numpy array based on return_tensor flag
                    if return_tensor:
                        return features
                    else:
                        return features.detach().cpu().numpy()

                else:
                    # Process both regions
                    if not isinstance(img_tensors, dict) or not all(k in img_tensors for k in ['full', 'head']):
                        self.logger.error("Expected dict with 'full' and 'head' keys")
                        zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                        return {'full': zero_emb, 'head': zero_emb} if return_tensor else {
                            'full': zero_emb.detach().cpu().numpy(),
                            'head': zero_emb.detach().cpu().numpy()
                        }

                    # Process full body and head on device
                    try:
                        full_emb = self._compute_feature_embedding(img_tensors['full'], 'full', return_tensor=True)
                        head_emb = self._compute_feature_embedding(img_tensors['head'], 'head', return_tensor=True)
                        
                        # Verify embeddings are on correct device
                        if full_emb.device != self.device:
                            full_emb = full_emb.to(self.device)
                        if head_emb.device != self.device:
                            head_emb = head_emb.to(self.device)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing embeddings: {str(e)}")
                        zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                        return {'full': zero_emb, 'head': zero_emb} if return_tensor else {
                            'full': zero_emb.detach().cpu().numpy(),
                            'head': zero_emb.detach().cpu().numpy()
                        }

                    # Ensure proper synchronization
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Return tensors or numpy arrays based on return_tensor flag
                    if return_tensor:
                        return {'full': full_emb, 'head': head_emb}
                    else:
                        return {
                            'full': full_emb.detach().cpu().numpy(),
                            'head': head_emb.detach().cpu().numpy()
                        }

        except TimeoutError:
            self.logger.error("Embedding computation timed out")
            zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
            if key is not None:
                return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()
            else:
                return {'full': zero_emb, 'head': zero_emb} if return_tensor else {
                    'full': zero_emb.detach().cpu().numpy(),
                    'head': zero_emb.detach().cpu().numpy()
                }
        except Exception as e:
            self.logger.error(f"Error in embedding processing: {str(e)}")
            zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
            if key is not None:
                return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()
            else:
                return {'full': zero_emb, 'head': zero_emb} if return_tensor else {
                    'full': zero_emb.detach().cpu().numpy(),
                    'head': zero_emb.detach().cpu().numpy()
                }

    def get_track_history(self, track_id):
        """Return the history for a given track as a list (not a deque)."""
        if track_id in self.track_history:
            history = self.track_history[track_id]
            # Convert all deques to lists
            def convert(obj):
                if isinstance(obj, deque):
                    return list(obj)
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                else:
                    return obj
            history_copy = {k: convert(v) for k, v in history.items()}
            # Also convert track_id_changes if present
            if track_id in self.track_id_changes:
                history_copy['track_id_changes'] = {k: convert(v) for k, v in self.track_id_changes[track_id].items()}
            return history_copy
        return {}

    def _trim_list(self, lst, maxlen):
        """Trim a list to the specified maximum length (keep most recent)."""
        if len(lst) > maxlen:
            del lst[:-maxlen]

    def _calculate_temporal_consistency(self, track_id, bbox, embedding):
        """Calculate temporal consistency score for a track update."""
        try:
            track_state = self.track_history[track_id]
            
            # Get previous state
            if not track_state['history']:
                return 1.0
                
            prev_state = track_state['history'][-1]
            prev_bbox = prev_state['bbox']
            
            if prev_bbox is None:
                return 1.0
                
            # Calculate movement consistency
            movement = np.linalg.norm(np.array(bbox[:2]) - np.array(prev_bbox[:2]))
            movement_score = 1.0 - min(1.0, movement / self.max_movement)
            
            # Calculate embedding consistency if available
            embedding_score = 1.0
            if embedding is not None and track_state['last_valid_embedding'] is not None:
                try:
                    similarity = F.cosine_similarity(
                        embedding.unsqueeze(0),
                        track_state['last_valid_embedding'].unsqueeze(0)
                    )
                    embedding_score = float(similarity.item())
                except Exception as e:
                    self.logger.warning(f"Error calculating embedding similarity: {str(e)}")
                    embedding_score = 0.0
            
            # Combine scores with embedding factor
            embedding_factor = track_state.get('embedding_factor', 1.0)
            temporal_consistency = (
                movement_score * (1.0 - embedding_factor * 0.5) +
                embedding_score * embedding_factor * 0.5
            )
            
            return float(temporal_consistency)
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal consistency: {str(e)}")
            return 0.0
            
    def _calculate_embedding_quality(self, embedding):
        """Calculate embedding quality score."""
        try:
            if embedding is None:
                return 0.0
            
            # Handle both tensor and numpy array inputs
            if isinstance(embedding, dict):
                # If embedding is a dict, use the 'full' embedding
                if 'full' in embedding:
                    embedding = embedding['full']
                else:
                    return 0.0
            
            # Convert to tensor if needed
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding).to(self.device)
            elif not isinstance(embedding, torch.Tensor):
                return 0.0
            
            # Ensure embedding is on correct device
            if embedding.device != self.device:
                embedding = embedding.to(self.device)
            
            # Check if embedding is normalized
            if not torch.allclose(torch.norm(embedding, p=2), torch.tensor(1.0)):
                embedding = F.normalize(embedding, p=2, dim=0)
            
            # Calculate quality based on embedding norm and variance
            norm = float(torch.norm(embedding, p=2).item())
            variance = float(torch.var(embedding).item())
            
            # Additional quality metrics
            min_val = float(torch.min(embedding).item())
            max_val = float(torch.max(embedding).item())
            range_score = 1.0 - (max_val - min_val) / 2.0  # Penalize if values are too spread out
            
            # Quality decreases with high variance, deviates from unit norm, and poor range
            quality = 1.0 - min(1.0, abs(norm - 1.0) + variance + (1.0 - range_score))
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            self.logger.error(f"Error calculating embedding quality: {str(e)}")
            return 0.0
            
    def _calculate_size_score(self, bbox):
        """Calculate size score based on bbox dimensions."""
        try:
            if bbox is None:
                return 0.0
                
            # Calculate area
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # Score based on area relative to expected size
            expected_area = self.expected_size[0] * self.expected_size[1]
            size_ratio = area / expected_area
            
            # Penalize both too small and too large detections
            if size_ratio < 0.1:  # Too small
                return 0.0
            elif size_ratio > 10.0:  # Too large
                return 0.0
            else:
                # Optimal size is close to expected
                return float(np.exp(-(size_ratio - 1.0) ** 2))
                
        except Exception as e:
            self.logger.error(f"Error calculating size score: {str(e)}")
            return 0.0
            
    def _calculate_movement_score(self, track_id, frame):
        """Calculate movement score based on track history."""
        try:
            track_state = self.track_history[track_id]
            
            if not track_state['history']:
                return 0.0
                
            prev_state = track_state['history'][-1]
            prev_bbox = prev_state['bbox']
            
            if prev_bbox is None:
                return 0.0
                
            # Calculate movement
            movement = np.linalg.norm(np.array(prev_bbox[:2]) - np.array(prev_bbox[2:]))
            
            # Normalize by frame size
            frame_size = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
            normalized_movement = movement / frame_size
            
            return min(1.0, normalized_movement * 10)  # Scale up for better visibility
            
        except Exception as e:
            self.logger.error(f"Error calculating movement score: {str(e)}")
            return 0.0
            
    def _calculate_behavior_score(self, track_id):
        """Calculate behavior score based on track history."""
        if track_id not in self.track_history:
            return 0.0
            
        history = self.track_history[track_id]['history']
        if len(history) < 2:
            return 0.5
            
        # Compute movement consistency
        bboxes = [h['bbox'] for h in history]
        centers = np.array([(b[0] + b[2])/2, (b[1] + b[3])/2] for b in bboxes)
        velocities = np.diff(centers, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        if len(velocity_magnitudes) == 0:
            return 0.5
            
        # Normalize velocity magnitudes
        max_velocity = np.max(velocity_magnitudes)
        if max_velocity > 0:
            velocity_magnitudes = velocity_magnitudes / max_velocity
            
        # Compute consistency score
        consistency = 1.0 - np.std(velocity_magnitudes)
        return max(0.0, min(1.0, consistency))
        
    def _update_temporal_consistency(self, track_id):
        """Update temporal consistency score for a track."""
        if track_id not in self.track_history:
            return
            
        history = self.track_history[track_id]['history']
        if len(history) < 2:
            self.track_temporal_consistency[track_id] = 1.0
            return
            
        # Get recent history entries
        recent_history = list(history)[-5:]
        
        # Compute consistency factors
        embedding_factors = [h['embedding_factor'] for h in recent_history]
        behavior_scores = [h['behavior_score'] for h in recent_history]
        movement_scores = [h['movement_score'] for h in recent_history]
        
        # Compute weighted average
        weights = np.array([0.4, 0.3, 0.3])  # Weights for embedding, behavior, movement
        factors = np.array([
            np.mean(embedding_factors),
            np.mean(behavior_scores),
            np.mean(movement_scores)
        ])
        
        consistency = np.sum(weights * factors)
        self.track_temporal_consistency[track_id] = max(0.0, min(1.0, consistency))
        
    def _cleanup_old_tracks(self):
        """Clean up old tracks and their resources."""
        try:
            current_time = self.frame_count
            tracks_to_remove = []
            
            for track_id in list(self.track_history.keys()):
                if current_time - self.track_ages[track_id] > self.max_age:
                    tracks_to_remove.append(track_id)
                    
            for track_id in tracks_to_remove:
                del self.track_history[track_id]
                del self.track_embeddings[track_id]
                del self.track_head_embeddings[track_id]
                del self.track_ages[track_id]
                del self.track_confidences[track_id]
                del self.track_bboxes[track_id]
                del self.track_temporal_consistency[track_id]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old tracks: {str(e)}")
            raise

    def update(self, frame, detections):
        """Update tracking with new detections.
        
        Args:
            frame: Input frame as numpy array
            detections: Array of detections [x1, y1, x2, y2, score, ...]
            
        Returns:
            list: List of active tracks with their IDs and bounding boxes
        """
        try:
            # Validate input
            if not isinstance(frame, np.ndarray):
                raise ValueError("Frame must be a numpy array")
                
            if not isinstance(detections, np.ndarray):
                detections = np.array(detections)
                
            if len(detections.shape) != 2 or detections.shape[1] < 5:
                raise ValueError("Detections must be a 2D array with at least 5 columns [x1, y1, x2, y2, score]")
                
            # Update frame count
            self.frame_count += 1
            
            # Filter detections by confidence
            mask = detections[:, 4] >= self.conf_threshold
            detections = detections[mask]
            
            if len(detections) == 0:
                # Update track ages and clean up old tracks
                self._cleanup_old_tracks()
                return []
                
            # Update SORT tracker
            tracks = self.tracker.update(detections)
            
            # Process each track
            active_tracks = []
            for track in tracks:
                track_id = int(track[4])
                bbox = track[:4]
                score = track[4] if len(track) > 4 else 1.0
                
                # Initialize track if new
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'history': deque(maxlen=100),
                        'last_valid_embedding': None,
                        'temporal_consistency': 1.0
                    }
                    self.track_embeddings[track_id] = deque(maxlen=5)
                    self.track_head_embeddings[track_id] = deque(maxlen=5)
                    self.track_ages[track_id] = 0
                    self.track_confidences[track_id] = score
                    self.track_bboxes[track_id] = bbox
                    self.track_temporal_consistency[track_id] = 1.0
                    
                # Update track age and confidence
                self.track_ages[track_id] += 1
                self.track_confidences[track_id] = score
                self.track_bboxes[track_id] = bbox
                
                # Extract and process crops if possible
                crops = extract_normalized_crow_crop(frame, bbox)
                if crops is not None:
                    # Convert crops to tensors for embedding computation
                    full_tensor = torch.from_numpy(crops['full']).permute(2, 0, 1).unsqueeze(0)
                    head_tensor = torch.from_numpy(crops['head']).permute(2, 0, 1).unsqueeze(0)
                    
                    # Create tensor dict for compute_embedding
                    crop_tensors = {
                        'full': full_tensor,
                        'head': head_tensor
                    }
                    
                    try:
                        combined_embedding, individual_embeddings = compute_embedding(crop_tensors)
                        full_embedding = individual_embeddings['full']
                        head_embedding = individual_embeddings['head']
                        
                        if full_embedding is not None:
                            self.track_embeddings[track_id].append(full_embedding)
                            self.track_history[track_id]['last_valid_embedding'] = torch.from_numpy(full_embedding).to(self.device)
                            
                        if head_embedding is not None:
                            self.track_head_embeddings[track_id].append(head_embedding)
                    except Exception as e:
                        self.logger.warning(f"Error computing embeddings for track {track_id}: {str(e)}")
                
                # Update track history
                history_entry = {
                    'bbox': bbox,
                    'confidence': score,
                    'age': self.track_ages[track_id],
                    'embedding_factor': 1.0 if len(self.track_embeddings[track_id]) > 0 else 0.0,
                    'behavior_score': self._compute_behavior_score(track_id),
                    'movement_score': self._calculate_movement_score(track_id, frame),
                    'temporal_consistency': self.track_temporal_consistency[track_id]
                }
                self.track_history[track_id]['history'].append(history_entry)
                
                # Update temporal consistency
                self._update_temporal_consistency(track_id)
                
                # Add to active tracks
                active_tracks.append({
                    'id': track_id,
                    'bbox': bbox,
                    'score': score,
                    'age': self.track_ages[track_id],
                    'embedding_count': len(self.track_embeddings[track_id]),
                    'temporal_consistency': self.track_temporal_consistency[track_id]
                })
                
            # Clean up old tracks
            self._cleanup_old_tracks()
            
            return active_tracks
            
        except Exception as e:
            self.logger.error(f"Error updating tracking: {str(e)}")
            raise

    def _generate_crow_id(self):
        """Generate a unique crow ID in format 'crow_XXXX'."""
        self.next_id += 1
        crow_id = f"crow_{self.next_id:04d}"
        self.tracking_data["metadata"]["last_id"] = self.next_id
        return crow_id

    def get_crow_info(self, crow_id):
        """Get information about a specific crow.
        
        Args:
            crow_id: The ID of the crow to get information for
            
        Returns:
            dict: Crow information or None if not found
        """
        return self.tracking_data["crows"].get(crow_id)

    def list_crows(self):
        """List all tracked crows.
        
        Returns:
            list: List of crow IDs
        """
        return list(self.tracking_data["crows"].keys())

    def _save_tracking_data(self, force=False):
        """Save tracking data to disk.
        
        Args:
            force: If True, save even if no changes
        """
        try:
            # Create metadata directory if it doesn't exist and tracking_file has a directory
            tracking_file_dir = os.path.dirname(self.tracking_file)
            if tracking_file_dir:  # Only create directory if there's a directory path
                os.makedirs(tracking_file_dir, exist_ok=True)
            
            # Save tracking data
            with open(self.tracking_file, "w") as f:
                json.dump(self.tracking_data, f, indent=2)
                
            self.logger.info("Tracking data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving tracking data: {str(e)}")
            raise

    def create_processing_run(self):
        """Create a processing run directory for compatibility with CrowTracker interface."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.processing_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created processing run directory: {run_dir}")
        return run_dir

    def process_detection(self, frame, frame_num, detection, video_path, frame_time):
        """Process a single detection and update tracking.
        
        Args:
            frame: Input frame
            frame_num: Frame number
            detection: Detection in format [[x1, y1, x2, y2, score]] or dict
            video_path: Path to the video file
            frame_time: Frame timestamp
            
        Returns:
            str: Crow ID if assigned, None otherwise
        """
        try:
            # Convert detection to proper format for SORT
            if isinstance(detection, dict):
                det_array = np.array([[
                    detection['bbox'][0], detection['bbox'][1], 
                    detection['bbox'][2], detection['bbox'][3], 
                    detection['score']
                ]])
            elif isinstance(detection, np.ndarray):
                if detection.ndim == 1:
                    det_array = detection.reshape(1, -1)
                elif detection.ndim == 2:
                    det_array = detection
                else:
                    raise ValueError(f"Invalid detection array shape: {detection.shape}")
            else:
                raise ValueError(f"Invalid detection format: {type(detection)}")
            
            # Validate detection format
            if det_array.shape[1] < 5:
                raise ValueError("Detection must have at least 5 columns [x1, y1, x2, y2, score]")
            
            # Process through tracking system
            tracks = self.update(frame, det_array)
            if len(tracks) == 0:
                return None
            
            # Get the first track (should only be one since we passed one detection)
            track = tracks[0]
            track_id = track['id']
            
            # Assign crow ID if this is a new track or update existing
            if track_id not in self.track_history:
                return None
            
            # Update track history with metadata
            if 'crow_id' not in self.track_history[track_id]:
                crow_id = self._generate_crow_id()
                self.track_history[track_id]['crow_id'] = crow_id
                self.track_history[track_id]['video_path'] = video_path
                self.track_history[track_id]['frame_time'] = frame_time
                
                # Store in tracking data
                self.tracking_data["crows"][crow_id] = {
                    "track_id": track_id,
                    "first_seen": frame_time,
                    "last_seen": frame_time,
                    "video_path": video_path,
                    "total_detections": 1,
                    "first_frame": frame_num,
                    "last_frame": frame_num,
                    "detections": [{
                        "frame": frame_num,
                        "bbox": det_array[0][:4].tolist(),
                        "score": float(det_array[0][4]),
                        "timestamp": frame_time
                    }]
                }
                self._save_tracking_data()
                return crow_id
            else:
                # Update existing crow
                crow_id = self.track_history[track_id]['crow_id']
                if crow_id in self.tracking_data["crows"]:
                    crow_data = self.tracking_data["crows"][crow_id]
                    crow_data["last_seen"] = frame_time
                    crow_data["last_frame"] = frame_num
                    crow_data["total_detections"] += 1
                    crow_data["detections"].append({
                        "frame": frame_num,
                        "bbox": det_array[0][:4].tolist(),
                        "score": float(det_array[0][4]),
                        "timestamp": frame_time
                    })
                    self._save_tracking_data()
                return crow_id
            
        except Exception as e:
            self.logger.error(f"Error processing detection: {str(e)}")
            return None

def compute_bbox_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    if box1.size == 0 or box2.size == 0:
        return 0.0
    
    x1, y1, x2, y2 = box1[:4]
    xx1, yy1, xx2, yy2 = box2[:4]
    
    # Calculate areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    
    if area1 <= 0 or area2 <= 0:
        return 0.0
    
    # Calculate intersection
    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area

def _convert_detection_to_array(detection):
    """Convert a detection dictionary to the required numpy array format.
    
    Args:
        detection: Dictionary with 'bbox' and 'score' keys
        
    Returns:
        np.ndarray: Detection in format [x1, y1, x2, y2, score]
    """
    if isinstance(detection, dict):
        bbox = detection['bbox']
        score = detection['score']
        return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], score]])
    elif isinstance(detection, np.ndarray):
        if len(detection.shape) == 1:
            return detection.reshape(1, -1)
        return detection
    else:
        raise ValueError("Invalid detection format")

def create_model():
    """Create and return a ResNet18 model for feature extraction."""
    import torchvision.models as models
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Remove the final classification layer for embedding
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model

def assign_crow_ids(frames, detections_list, video_path=None, max_age=5, min_hits=2, iou_threshold=0.2, embedding_threshold=0.7, return_track_history=False, multi_view_stride=1):
    print("[INFO] Starting enhanced crow tracking...")
    labeled_frames = []
    tracker = EnhancedTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        embedding_threshold=embedding_threshold,
        multi_view_stride=multi_view_stride
    )
    known_crows = get_all_crows()
    crow_ids = {crow['id']: crow['id'] for crow in known_crows}
    track_history_per_frame = []
    for frame_idx, (frame, detections) in enumerate(zip(frames, detections_list)):
        t0 = time.time()
        tracks = tracker.update(frame, detections)
        t1 = time.time()
        print(f"[INFO] Frame {frame_idx+1}/{len(frames)}: tracking/embedding took {t1-t0:.3f} seconds")
        frame_copy = frame.copy()
        frame_tracks = {}
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            track_id = int(track_id)
            if track_id in tracker.track_embeddings and tracker.track_embeddings[track_id]:
                embedding = tracker.track_embeddings[track_id][-1]
                # Ensure confidence is a valid float
                conf = track[4]
                if isinstance(conf, bytes):
                    try:
                        conf = float(conf.decode('utf-8'))
                    except Exception:
                        conf = 0.0
                elif not isinstance(conf, (int, float, np.floating)):
                    try:
                        conf = float(conf)
                    except Exception:
                        conf = 0.0
                db_crow_id = save_crow_embedding(
                    embedding,
                    video_path=video_path,
                    frame_number=frame_idx,
                    confidence=conf
                )
                crow_id = crow_ids.get(db_crow_id, db_crow_id)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = f"Crow {crow_id}"
                cv2.putText(frame_copy, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                history = get_crow_history(db_crow_id)
                if history:
                    info = f"Seen in {history['video_count']} videos, {history['total_sightings']} times"
                    cv2.putText(frame_copy, info, (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                frame_tracks[crow_id] = [x1, y1, x2, y2]
        labeled_frames.append(frame_copy)
        track_history_per_frame.append(frame_tracks)
    print("[INFO] Enhanced tracking complete.")
    if return_track_history:
        return labeled_frames, track_history_per_frame
    return labeled_frames

def load_faster_rcnn():
    """Load and configure the Faster R-CNN model."""
    try:
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load pre-trained model
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Configure for bird detection
        model.roi_heads.score_thresh = 0.5
        model.roi_heads.nms_thresh = 0.3
        
        logger.info(f"Faster R-CNN model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading Faster R-CNN model: {str(e)}")
        raise

def load_triplet_model():
    """Load and configure the triplet network model for crow embeddings."""
    try:
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model
        model = CrowResNetEmbedder(embedding_dim=512)
        
        # Try to load trained weights
        try:
            # Load weights to the appropriate device
            state_dict = torch.load('crow_resnet_triplet.pth', map_location=device)
            
            # Handle state dict mismatch
            if 'fc.weight' in state_dict:
                # Remove the fc layer weights if they exist
                state_dict.pop('fc.weight', None)
                state_dict.pop('fc.bias', None)
                logger.info("Removed fc layer weights from state dict")
            
            # Load the state dict with strict=False to handle any remaining mismatches
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded trained triplet network model on {device}")
        except Exception as e:
            logger.warning(f"Could not load trained model weights: {e}")
            logger.info("Using untrained model")
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Add get_embedding method for convenience
        def get_embedding(self, img_tensor):
            """Get embedding for a single image tensor."""
            # Move input to same device as model
            if isinstance(img_tensor, dict):
                # Handle both full and head regions
                full_tensor = img_tensor['full'].to(device)
                head_tensor = img_tensor['head'].to(device)
                with torch.no_grad():
                    full_emb = self(full_tensor)
                    head_emb = self(head_tensor)
                # Combine embeddings (70% full, 30% head)
                combined = 0.7 * full_emb + 0.3 * head_emb
                return combined
            else:
                # Handle single tensor
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    return self(img_tensor.unsqueeze(0))
        
        # Add method to model
        model.get_embedding = get_embedding.__get__(model)
        
        logger.info(f"Triplet network model initialized successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Error loading triplet network model: {str(e)}")
        raise

# Export for test visibility
__all__ = [
    'TrackingError', 'ModelError', 'DeviceError', 'EmbeddingError', 'TimeoutException',
    'EnhancedTracker', 'extract_normalized_crow_crop', 'assign_crow_ids', 'compute_bbox_iou', 'load_faster_rcnn', 'load_triplet_model'
]
