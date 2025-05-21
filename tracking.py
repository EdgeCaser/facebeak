import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, deque
from db import save_crow_embedding, get_all_crows, get_crow_history, add_behavioral_marker
from sort import Sort
from scipy.spatial.distance import cdist
from models import CrowResNetEmbedder
from ultralytics import YOLO
import torchvision
from torchvision.models import resnet18
import torch.nn as nn
import logging
import time
import threading
import gc
from functools import wraps
from color_normalization import create_normalizer
from multi_view import create_multi_view_extractor
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

def extract_crow_image(frame, bbox, padding=0.3, min_size=10):
    """Extract a cropped image of a crow from the frame.
    
    Args:
        frame: Input frame (numpy array)
        bbox: Bounding box in [x1, y1, x2, y2] format
        padding: Padding factor around the bbox (default: 0.3)
        min_size: Minimum size for valid bbox (default: 10)
        
    Returns:
        Dictionary containing 'full' and 'head' tensors, or None if extraction fails
    """
    try:
        # Convert bbox to numpy array and ensure it's the right shape
        bbox = np.asarray(bbox, dtype=np.float32)
        if bbox.size != 4:
            logger.warning(f"Invalid bbox size: {bbox.size}, expected 4 values")
            return None
            
        # Ensure bbox is in [x1, y1, x2, y2] format and coordinates are valid
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid bbox coordinates: {bbox} (x1 >= x2 or y1 >= y2)")
            return None
            
        h, w = frame.shape[:2]
        
        # Validate bbox coordinates
        if (x1 < 0 or y1 < 0 or x2 > w or y2 > h or  # Box outside frame
            x2 - x1 < min_size or y2 - y1 < min_size):  # Box too small
            logger.warning(f"Invalid bbox coordinates: {bbox} for frame size {w}x{h}")
            return None
            
        # Calculate padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        x1 = int(max(0, x1 - pad_w))
        y1 = int(max(0, y1 - pad_h))
        x2 = int(min(w, x2 + pad_w))
        y2 = int(min(h, y2 + pad_h))
        
        # Extract and validate crops
        try:
            crow_img = frame[y1:y2, x1:x2].copy()  # Make a copy to ensure contiguous array
            if not isinstance(crow_img, np.ndarray) or crow_img.size == 0:
                logger.warning(f"Invalid crow crop extracted for bbox {bbox}")
                return None
            
            # Calculate head region (top third of the crow)
            head_height = max(1, int((y2 - y1) * 0.33))  # Ensure at least 1 pixel
            head_img = frame[y1:y1 + head_height, x1:x2].copy()
            if not isinstance(head_img, np.ndarray) or head_img.size == 0:
                logger.warning(f"Invalid head crop extracted for bbox {bbox}")
                return None
                
            # Convert to tensors and normalize
            def to_tensor(img):
                # Ensure image is in RGB format
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
                # Resize to 224x224 (standard input size for ResNet)
                img = cv2.resize(img, (224, 224))
                
                # Convert to tensor and normalize
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                return img
            
            return {
                'full': to_tensor(crow_img),
                'head': to_tensor(head_img)
            }
            
        except Exception as e:
            logger.error(f"Error extracting crops: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting crow image: {str(e)}")
        return None

class EnhancedTracker:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, embedding_threshold=0.5, model_path=None, conf_threshold=0.5, multi_view_stride=1, strict_mode=False):
        """Initialize the enhanced tracker with improved parameters.
        
        Args:
            max_age: Maximum number of frames to keep a track without detection
            min_hits: Minimum number of detections before a track is confirmed
            iou_threshold: IoU threshold for track-detection association
            embedding_threshold: Threshold for embedding similarity
            model_path: Path to model weights file
            conf_threshold: Confidence threshold for detections
            multi_view_stride: Stride for multi-view processing
            strict_mode: If True, require exact state dict match during model initialization
        """
        try:
            # Tracking parameters with improved defaults
            self.max_age = max_age
            self.min_hits = min_hits
            self.iou_threshold = iou_threshold
            self.embedding_threshold = embedding_threshold
            self.conf_threshold = conf_threshold
            self.multi_view_stride = multi_view_stride
            self.strict_mode = strict_mode
            
            # Add embedding quality threshold
            self.min_embedding_quality = 0.3  # Minimum quality threshold for embeddings
            
            # Set track limits based on max_age to ensure proper cleanup
            self.max_track_history = max_age * 2  # Keep twice the max_age worth of history
            self.max_embedding_history = max_age  # Keep max_age worth of embeddings
            self.max_behavior_history = max_age  # Keep max_age worth of behavior history
            self.cleanup_interval = max(1, max_age // 2)  # Clean up at least every max_age/2 frames
            self.max_tracks = 10  # Enforce maximum number of active tracks
            
            # Retry parameters
            self.max_retries = 3
            self.retry_delay = 0.1  # seconds
            
            # Initialize logger
            self._configure_logger()
            
            # Initialize device with improved handling
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                torch.cuda.set_device(0)  # Set default device to 0
                self.logger.info("CUDA device initialized")
            else:
                self.device = torch.device('cpu')
                self.logger.info("Using CPU device (CUDA not available)")
            
            # Initialize SORT tracker
            from sort import Sort
            self.tracker = Sort(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold
            )
            self.logger.info("SORT tracker initialized successfully")
            
            # Initialize tracking state with improved memory management
            self.frame_count = 0
            self.track_embeddings = defaultdict(lambda: deque(maxlen=self.max_embedding_history))
            self.track_head_embeddings = defaultdict(lambda: deque(maxlen=self.max_embedding_history))
            self.track_history = {}
            self.track_id_changes = {}
            self.track_ages = {}
            self.next_id = 0
            self.active_tracks = set()
            self.last_cleanup_frame = 0
            
            # Initialize models with improved error handling
            self._initialize_models(model_path, strict_mode=strict_mode)
            
            # Initialize multi-view extractor if needed
            self._init_multi_view_extractor(strict_mode)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {str(e)}")
            # Clean up any partially initialized resources
            if hasattr(self, 'model'):
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise

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
    def _process_detection_batch(self, frame, detections, frame_idx):
        """Process a batch of detections with improved track management and error handling."""
        try:
            # Validate input format
            if not isinstance(detections, np.ndarray) or detections.ndim != 2:
                raise ValueError("Invalid detections format: expected 2D numpy array")
            
            # Get tracked objects from SORT
            tracked_objects = self.tracker.update(detections)
            
            # Process each tracked object
            processed_tracks = []
            for track in tracked_objects:
                try:
                    x1, y1, x2, y2, track_id = track
                    bbox = [float(x1), float(y1), float(x2), float(y2)]  # Ensure float type
                    
                    # Get detection confidence using IOU
                    det_idx = np.argmax([compute_iou(bbox, d[:4]) for d in detections])
                    confidence = float(detections[det_idx, 4])  # Ensure float type
                    
                    # Skip low confidence detections
                    if confidence < self.conf_threshold:
                        continue
                    
                    # Initialize track if needed
                    if track_id not in self.track_history:
                        self._initialize_track(track_id, frame_idx)
                    
                    # Extract crow image and compute embedding
                    crow_img = self.extract_crow_image(frame, bbox)
                    if crow_img is None:
                        continue
                    
                    # Process embedding
                    embedding = self._process_embedding(crow_img)
                    if embedding is None:
                        continue
                    
                    # Update track
                    if self._update_track(track_id, frame_idx, bbox, confidence, embedding):
                        processed_tracks.append(track)
                
                except Exception as e:
                    self.logger.error(f"Error processing tracked object: {str(e)}")
                    continue
            
            return np.array(processed_tracks) if processed_tracks else np.empty((0, 5))
            
        except Exception as e:
            self.logger.error(f"Detection batch processing failed: {str(e)}")
            return np.empty((0, 5))

    def _update_track(self, track_id, frame_idx, bbox, confidence, embedding=None, head_embedding=None):
        """Update track state with proper embedding and history management."""
        try:
            if track_id not in self.track_history:
                self.logger.warning(f"Track {track_id} not found in history")
                return False
                
            # Get current track state
            track_state = self.track_history[track_id]
            current_time = time.time()
            
            # Update track age
            self.track_ages[track_id] = frame_idx - track_state['history'][0]['frame']
            
            # Calculate temporal consistency and embedding quality
            temporal_consistency = self._calculate_temporal_consistency(track_id, bbox, embedding)
            embedding_quality = self._calculate_embedding_quality(embedding) if embedding is not None else 0.0
            
            # Update embedding history if valid
            if embedding is not None and embedding_quality > self.min_embedding_quality:
                # Ensure embedding is on correct device
                embedding = embedding.to(self.device)
                head_embedding = head_embedding.to(self.device) if head_embedding is not None else None
                
                # Update embedding history with proper device handling
                self.track_embeddings[track_id].append(embedding.clone())
                if head_embedding is not None:
                    self.track_head_embeddings[track_id].append(head_embedding.clone())
                
                # Update last valid embedding
                track_state['last_valid_embedding'] = embedding.clone()
                track_state['last_embedding_quality'] = embedding_quality
                
                # Update embedding factor for temporal consistency
                track_state['embedding_factor'] = min(1.0, embedding_quality / self.min_embedding_quality)
            
            # Calculate track quality metrics
            size_score = self._calculate_size_score(bbox)
            movement_score = self._calculate_movement_score(track_id, bbox)
            behavior_score = self._calculate_behavior_score(track_id)
            
            # Create new state entry
            new_state = {
                'frame': frame_idx,
                'time': current_time,
                'bbox': bbox,
                'confidence': confidence,
                'embedding_quality': embedding_quality,
                'temporal_consistency': temporal_consistency,
                'track_quality': min(1.0, (confidence + temporal_consistency + embedding_quality) / 3.0),
                'size_score': size_score,
                'movement_score': movement_score,
                'behavior_score': behavior_score,
                'embedding_factor': track_state.get('embedding_factor', 1.0)
            }
            
            # Update history collections
            track_state['history'].append(new_state)
            track_state['confidence_history'].append(confidence)
            track_state['bbox_history'].append(bbox)
            
            # Update track state
            track_state['last_bbox'] = bbox
            track_state['last_confidence'] = confidence
            track_state['last_temporal_consistency'] = temporal_consistency
            track_state['last_track_quality'] = new_state['track_quality']
            track_state['last_size_score'] = size_score
            track_state['last_movement_score'] = movement_score
            track_state['last_behavior_score'] = behavior_score
            track_state['update_count'] += 1
            
            # Update track ID changes if needed
            if track_id in self.track_id_changes:
                self.track_id_changes[track_id].update(track_state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating track {track_id}: {str(e)}")
            # Add error to history
            if track_id in self.track_history:
                self.track_history[track_id]['errors'].append({
                    'frame': frame_idx,
                    'time': time.time(),
                    'error': str(e)
                })
            return False

    def _cleanup_track(self, track_id):
        """Clean up a single track and its resources."""
        try:
            if track_id in self.track_history:
                # Log final state
                history = self.track_history[track_id]
                self.logger.debug(f"Cleaning up track {track_id} with age {self.track_ages.get(track_id, 0)}")
                
                # Clear embeddings
                if track_id in self.track_embeddings:
                    for emb in self.track_embeddings[track_id]:
                        if isinstance(emb, torch.Tensor):
                            emb.cpu()
                            del emb
                    self.track_embeddings.pop(track_id)
                
                if track_id in self.track_head_embeddings:
                    for emb in self.track_head_embeddings[track_id]:
                        if isinstance(emb, torch.Tensor):
                            emb.cpu()
                            del emb
                    self.track_head_embeddings.pop(track_id)
                
                # Clear history collections
                for collection in ['history', 'behaviors', 'embedding_history', 
                                 'confidence_history', 'bbox_history', 'behavior_sequence']:
                    if collection in history:
                        history[collection].clear()
                
                # Remove from all collections
                self.track_history.pop(track_id, None)
                self.track_id_changes.pop(track_id, None)
                self.track_ages.pop(track_id, None)
                self.active_tracks.discard(track_id)
                
                # Force CUDA cache clear if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up track {track_id}: {str(e)}", exc_info=True)

    def _cleanup_old_tracks(self):
        """Clean up old tracks and free memory with improved persistence handling."""
        try:
            current_frame = self.frame_count
            tracks_to_remove = set()
            
            # First pass: identify tracks to remove based on multiple criteria
            for track_id in list(self.track_history.keys()):
                history = self.track_history[track_id]
                track_age = current_frame - history['created_frame']
                frames_since_seen = current_frame - history['last_seen']
                
                # Calculate track health score with improved weighting
                health_score = (
                    0.4 * history['temporal_consistency'] +  # Increased weight for temporal consistency
                    0.3 * history['average_confidence'] +    # Keep high weight for confidence
                    0.2 * history['embedding_quality'] +     # Keep moderate weight for embedding quality
                    0.1 * (1.0 - min(1.0, history['consecutive_misses'] / self.max_age))  # Reduced weight for misses
                )
                
                # Remove tracks based on multiple criteria with improved thresholds
                should_remove = False
                reason = []
                
                # Only remove if significantly past max age and health score is low
                if track_age > self.max_age * 1.5 and health_score < 0.3:
                    should_remove = True
                    reason.append(f"age={track_age} health={health_score:.2f}")
                
                # Only remove if many frames since last seen and health score is low
                if frames_since_seen > self.max_age and health_score < 0.3:
                    should_remove = True
                    reason.append(f"unseen={frames_since_seen} health={health_score:.2f}")
                
                # Remove if track quality is very poor
                if history['track_quality'] < 0.1 and track_age > self.max_age:
                    should_remove = True
                    reason.append(f"quality={history['track_quality']:.2f}")
                
                if should_remove:
                    self.logger.debug(f"Marking track {track_id} for removal: {', '.join(reason)}")
                    tracks_to_remove.add(track_id)
            
            # Second pass: remove marked tracks
            for track_id in tracks_to_remove:
                self._cleanup_track(track_id)
            
            # Update cleanup frame counter
            self.last_cleanup_frame = current_frame
            
        except Exception as e:
            self.logger.error(f"Error in track cleanup: {str(e)}", exc_info=True)

    def _initialize_track(self, track_id, frame_idx):
        """Initialize a new track with proper history management."""
        try:
            # Initialize track history with proper collections
            self.track_history[track_id] = {
                'history': deque(maxlen=self.max_track_history),
                'confidence_history': deque(maxlen=self.max_track_history),
                'bbox_history': deque(maxlen=self.max_track_history),
                'embedding_history': deque(maxlen=self.max_embedding_history),
                'behavior_sequence': deque(maxlen=self.max_behavior_history),
                'errors': deque(maxlen=self.max_track_history),
                'behavioral_markers': set(),
                'behavior_transitions': 0,
                'behavior_duration': 0,
                'update_count': 0,
                'temporal_consistency': 1.0,
                'size_score': 1.0,
                'movement_score': 0.0,
                'average_embedding_quality': 1.0,
                'last_valid_embedding': None,
                'last_bbox': None,
                'last_confidence': 1.0,
                'last_embedding_quality': 1.0,
                'last_temporal_consistency': 1.0,
                'last_track_quality': 1.0,
                'last_size_score': 1.0,
                'last_movement_score': 0.0,
                'last_behavior_score': 0.0,
                'last_behavior': None,
                'embedding_factor': 1.0,  # Add embedding factor for temporal consistency
                'created_frame': frame_idx,  # Add created_frame for track age calculation
                'last_seen': frame_idx,  # Add last_seen for track cleanup
                'average_confidence': 1.0,  # Add average_confidence for track health calculation
                'consecutive_misses': 0,  # Add consecutive_misses for track health calculation
                'track_quality': 1.0  # Add track_quality for track health calculation
            }
            
            # Initialize track ID change history with same fields
            self.track_id_changes[track_id] = self.track_history[track_id].copy()
            
            # Initialize embeddings collections with proper device handling
            self.track_embeddings[track_id] = deque(maxlen=self.max_embedding_history)
            self.track_head_embeddings[track_id] = deque(maxlen=self.max_embedding_history)
            
            # Add initial zero embedding with proper device placement
            zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
            self.track_embeddings[track_id].append(zero_emb)
            self.track_head_embeddings[track_id].append(zero_emb)
            
            # Store initial embedding as last valid
            self.track_history[track_id]['last_valid_embedding'] = zero_emb.clone()
            self.track_id_changes[track_id]['last_valid_embedding'] = zero_emb.clone()
            
            # Add to active tracks and initialize age
            self.active_tracks.add(track_id)
            self.track_ages[track_id] = 0
            
            # Initialize history collections with initial values
            initial_state = {
                'frame': frame_idx,
                'time': time.time(),
                'bbox': None,
                'confidence': 1.0,
                'embedding_quality': 1.0,
                'temporal_consistency': 1.0,
                'track_quality': 1.0,
                'size_score': 1.0,
                'movement_score': 0.0,
                'behavior_score': 0.0,
                'embedding_factor': 1.0  # Add embedding factor to initial state
            }
            
            # Add initial state to all history collections
            self.track_history[track_id]['history'].append(initial_state.copy())
            self.track_history[track_id]['confidence_history'].append(1.0)
            self.track_history[track_id]['bbox_history'].append(None)
            self.track_history[track_id]['behavior_sequence'].append(None)
            
            # Ensure track is properly indexed
            if track_id not in self.track_history:
                raise ValueError(f"Failed to initialize track {track_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing track {track_id}: {str(e)}")
            # Clean up any partially initialized state
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_id_changes:
                del self.track_id_changes[track_id]
            if track_id in self.track_embeddings:
                del self.track_embeddings[track_id]
            if track_id in self.track_head_embeddings:
                del self.track_head_embeddings[track_id]
            self.active_tracks.discard(track_id)
            self.track_ages.pop(track_id, None)
            return False

    def _update_track_behavior(self, track_id, bbox, frame_idx):
        """Update track behavior with improved temporal consistency."""
        try:
            history = self.track_history[track_id]
            changes = self.track_id_changes[track_id]
            
            # Calculate size score for temporal consistency
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if len(history['bbox_history']) > 0:
                prev_bbox = history['bbox_history'][-1]
                prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
                size_ratio = min(bbox_area, prev_area) / max(bbox_area, prev_area)
                history['size_score'] = size_ratio
                changes['size_score'] = size_ratio
            
            # Update behavior sequence
            if len(history['bbox_history']) > 0:
                prev_bbox = history['bbox_history'][-1]
                movement = np.linalg.norm(np.array(bbox[:2]) - np.array(prev_bbox[:2]))
                history['movement_score'] = movement
                changes['movement_score'] = movement
                
                # Detect behavior based on movement
                if movement < 5:
                    behavior = 'stationary'
                elif movement < 20:
                    behavior = 'walking'
                else:
                    behavior = 'running'
                
                # Update behavior sequence
                if behavior != history['last_behavior']:
                    history['behavior_transitions'] += 1
                    changes['behavior_transitions'] += 1
                    history['behavior_duration'] = 0
                else:
                    history['behavior_duration'] += 1
                
                history['last_behavior'] = behavior
                changes['last_behavior'] = behavior
                history['behavior_sequence'].append(behavior)
                changes['behavior_sequence'].append(behavior)
                
                # Update behavioral markers
                if behavior == 'stationary' and history['behavior_duration'] > 10:
                    history['behavioral_markers'].add('long_stationary')
                    changes['behavioral_markers'].add('long_stationary')
                elif behavior == 'running' and history['behavior_duration'] > 5:
                    history['behavioral_markers'].add('sustained_running')
                    changes['behavioral_markers'].add('sustained_running')
                
                return behavior
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error updating track behavior for {track_id}: {str(e)}")
            return None

    def update(self, frame, detections):
        """Update tracks with new detections."""
        try:
            # Validate input
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                raise ValueError("Invalid frame format: expected 3D numpy array")
            if not isinstance(detections, (list, np.ndarray)):
                raise ValueError("Invalid detections format: expected list or numpy array")
            
            # Convert detections to proper format
            if isinstance(detections, list):
                # Ensure each detection has required fields
                for det in detections:
                    if not isinstance(det, dict):
                        raise ValueError("Each detection must be a dictionary")
                    if 'bbox' not in det:
                        raise ValueError("Each detection must have a 'bbox' field")
                    if 'score' not in det:
                        raise ValueError("Each detection must have a 'score' field")
                    if not isinstance(det['bbox'], (list, np.ndarray)) or len(det['bbox']) != 4:
                        raise ValueError("Each detection bbox must be a list/array of 4 values")
                    if not isinstance(det['score'], (int, float)):
                        raise ValueError("Each detection score must be a number")
                
                # Convert to numpy array with proper types
                detections = np.array([
                    [float(x) for x in det['bbox']] + [float(det['score'])]
                    for det in detections
                ])
            
            # Process detections in batches
            processed_tracks = self._process_detection_batch(frame, detections, self.frame_count)
            
            # Update frame count
            self.frame_count += 1
            
            # Clean up old tracks periodically
            if self.frame_count - self.last_cleanup_frame >= self.cleanup_interval:
                self._cleanup_old_tracks()
            
            return processed_tracks
            
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}", exc_info=True)
            return np.empty((0, 5))

    def extract_crow_image(self, frame, bbox, padding=0.3, min_size=10):
        """Extract a cropped image of a crow from the frame.
        
        Args:
            frame: Input frame (numpy array)
            bbox: Bounding box in [x1, y1, x2, y2] format
            padding: Padding factor around the bbox (default: 0.3)
            min_size: Minimum size for valid bbox (default: 10)
            
        Returns:
            Dictionary containing 'full' and 'head' tensors, or None if extraction fails
        """
        try:
            # Use the robust standalone function for extraction
            return extract_crow_image(frame, bbox, padding=padding, min_size=min_size)
        except Exception as e:
            self.logger.error(f"Error extracting crow image: {str(e)}")
            return None

    def _initialize_models(self, model_path, strict_mode=False):
        """Initialize models with proper error handling and strict mode support."""
        try:
            # Create base model first
            try:
                self.model = create_model()
                if self.model is None:
                    if strict_mode:
                        raise ModelError("create_model() returned None")
                    self.logger.warning("create_model() returned None, using default ResNet18")
                    import torchvision.models as models
                    self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                    self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            except Exception as e:
                if strict_mode:
                    raise ModelError(f"Failed to create base model: {str(e)}")
                self.logger.warning(f"Failed to create base model, using default ResNet18: {str(e)}")
                import torchvision.models as models
                self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Initialize color normalizer
            try:
                self.color_normalizer = create_normalizer()
                if self.color_normalizer is None:
                    if strict_mode:
                        raise ModelError("create_normalizer() returned None")
                    self.logger.warning("create_normalizer() returned None, using default normalization")
                    self.color_normalizer = lambda x: x / 255.0
            except Exception as e:
                if strict_mode:
                    raise ModelError(f"Failed to create color normalizer: {str(e)}")
                self.logger.warning(f"Failed to create color normalizer, using default normalization: {str(e)}")
                self.color_normalizer = lambda x: x / 255.0
            
            if model_path is not None:
                try:
                    if not os.path.exists(model_path):
                        if strict_mode:
                            raise ModelError(f"Model file not found: {model_path}")
                        self.logger.warning(f"Model file not found: {model_path}, using default weights")
                        return
                    
                    state_dict = torch.load(model_path, map_location=self.device)
                    if not isinstance(state_dict, dict):
                        if strict_mode:
                            raise ModelError(f"Invalid model state dict format from {model_path}")
                        self.logger.warning(f"Invalid model state dict format from {model_path}, using default weights")
                        return
                    
                    self._load_state_dict(state_dict, strict_mode)
                except Exception as e:
                    if strict_mode:
                        raise ModelError(f"Failed to load model from {model_path}: {str(e)}")
                    self.logger.warning(f"Failed to load model from {model_path}, using default weights: {str(e)}")
            
            # Initialize multi-view extractor immediately after model creation
            self._init_multi_view_extractor(strict_mode)
            
        except ModelError:
            # Re-raise ModelError without wrapping
            raise
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            if strict_mode:
                raise ModelError(f"Model initialization failed: {str(e)}")
            raise

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
    def _process_embedding(self, img_tensors, key=None, return_tensor=False):
        """Compute feature embeddings for both head and body regions using the instance's model and device."""
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
                        full_emb = self._process_embedding(img_tensors['full'], 'full', return_tensor=True)
                        head_emb = self._process_embedding(img_tensors['head'], 'head', return_tensor=True)
                        
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
            
    def _calculate_movement_score(self, track_id, bbox):
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
            movement = np.linalg.norm(np.array(bbox[:2]) - np.array(prev_bbox[:2]))
            
            # Score based on movement relative to max allowed
            movement_score = 1.0 - min(1.0, movement / self.max_movement)
            
            # Apply temporal smoothing
            prev_movement_score = track_state.get('last_movement_score', 0.0)
            smoothed_score = 0.7 * prev_movement_score + 0.3 * movement_score
            
            return float(smoothed_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating movement score: {str(e)}")
            return 0.0
            
    def _calculate_behavior_score(self, track_id):
        """Calculate behavior score based on track history."""
        try:
            track_state = self.track_history[track_id]
            
            if not track_state['behavior_sequence']:
                return 0.0
                
            # Get recent behaviors
            recent_behaviors = list(track_state['behavior_sequence'])[-self.behavior_window:]
            if not recent_behaviors:
                return 0.0
                
            # Calculate behavior stability
            unique_behaviors = len(set(recent_behaviors))
            behavior_stability = 1.0 - (unique_behaviors - 1) / len(recent_behaviors)
            
            # Calculate behavior duration
            current_behavior = recent_behaviors[-1]
            behavior_duration = sum(1 for b in recent_behaviors if b == current_behavior)
            duration_score = min(1.0, behavior_duration / self.behavior_window)
            
            # Combine scores
            behavior_score = 0.7 * behavior_stability + 0.3 * duration_score
            
            return float(behavior_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating behavior score: {str(e)}")
            return 0.0

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU score between 0 and 1
    """
    # Convert to numpy arrays if needed
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return float(iou)

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
    'EnhancedTracker', 'extract_crow_image', 'assign_crow_ids', 'compute_iou', 'load_faster_rcnn', 'load_triplet_model'
]
