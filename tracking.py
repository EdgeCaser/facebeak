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

def extract_crow_image(frame, bbox, padding=0.3, min_size=100):
    """Extract a cropped image of a crow from the frame.
    
    Args:
        frame: Input frame (numpy array)
        bbox: Bounding box in [x1, y1, x2, y2] format
        padding: Padding factor around the bbox (default: 0.3)
        min_size: Minimum size for valid bbox (default: 100)
        
    Returns:
        Dictionary containing 'full' and 'head' tensors, or None if extraction fails
    """
    try:
        # Convert bbox to numpy array and ensure it's the right shape
        bbox = np.asarray(bbox, dtype=np.float32)
        if bbox.size != 4:
            logger.warning(f"Invalid bbox size: {bbox.size}, expected 4 values")
            return None
            
        # Ensure bbox is in [x1, y1, x2, y2] format
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            logger.warning(f"Invalid bbox coordinates: {bbox}")
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Validate bbox coordinates
        if (x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h or
            x2 - x1 < min_size or y2 - y1 < min_size):
            logger.warning(f"Invalid bbox coordinates: {bbox} for frame size {w}x{h}")
            return None
            
        # Calculate padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Extract and validate crops
        try:
            crow_img = frame[y1:y2, x1:x2].copy()  # Make a copy to ensure contiguous array
            if not isinstance(crow_img, np.ndarray) or crow_img.size == 0 or crow_img.shape[0] == 0 or crow_img.shape[1] == 0:
                logger.warning(f"Invalid crow crop extracted for bbox {bbox}")
                return None
            
            head_height = int((y2 - y1) * 0.33)
            if head_height <= 0:
                logger.warning(f"Invalid head height {head_height} for bbox {bbox}")
                return None
            
            head_img = frame[y1:y1 + head_height, x1:x2].copy()  # Make a copy to ensure contiguous array
            if not isinstance(head_img, np.ndarray) or head_img.size == 0 or head_img.shape[0] == 0 or head_img.shape[1] == 0:
                logger.warning(f"Invalid head crop extracted for bbox {bbox}")
                return None
        except Exception as e:
            logger.error(f"Error extracting crops: {str(e)}")
            return None
            
        def enhance_image(img):
            """Simple image enhancement using only normalization."""
            try:
                if not isinstance(img, np.ndarray) or img.size == 0:
                    return None
                
                # Basic validation
                if len(img.shape) != 3 or img.shape[2] != 3:
                    logger.warning(f"Invalid image shape: {img.shape}")
                    return None
                
                # Simple contrast enhancement using normalization only
                img = img.astype(np.float32)
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                img = (img * 255).astype(np.uint8)
                return img
                
            except Exception as e:
                logger.error(f"Error in enhance_image: {str(e)}")
                return None
            
        # Enhance images
        crow_img = enhance_image(crow_img)
        head_img = enhance_image(head_img)
        if crow_img is None or head_img is None:
            logger.warning(f"Image enhancement failed for bbox {bbox}")
            return None
            
        def resize_with_aspect(img, target_size=224):
            try:
                if not isinstance(img, np.ndarray) or img.size == 0:
                    return None
                
                h, w = img.shape[:2]
                if h == 0 or w == 0:
                    return None
                
                scale = target_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h == 0 or new_w == 0:
                    return None
                
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2
                square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                return square
                
            except Exception as e:
                logger.error(f"Error in resize_with_aspect: {str(e)}")
                return None
            
        # Resize images
        crow_square = resize_with_aspect(crow_img)
        head_square = resize_with_aspect(head_img)
        if crow_square is None or head_square is None:
            logger.warning(f"Image resizing failed for bbox {bbox}")
            return None
            
        # Convert to tensors
        try:
            crow_tensor = torch.from_numpy(crow_square).permute(2, 0, 1).float() / 255.0
            head_tensor = torch.from_numpy(head_square).permute(2, 0, 1).float() / 255.0
            return {
                'full': crow_tensor,
                'head': head_tensor
            }
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
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
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                raise ValueError("Invalid frame format")
            if not isinstance(detections, (list, np.ndarray)):
                raise ValueError("Invalid detections format")
            
            # Convert detections to numpy array if needed
            if isinstance(detections, list):
                detections = np.array([d['bbox'] + [d.get('score', 0.0)] for d in detections])
            
            # Update SORT tracker
            tracked_objects = self.tracker.update(detections)
            processed_tracks = []
            
            # Process each tracked object
            for track in tracked_objects:
                try:
                    x1, y1, x2, y2, track_id = track
                    bbox = [x1, y1, x2, y2]
                    
                    # Get detection confidence
                    det_idx = np.argmax([iou(bbox, d[:4]) for d in detections])
                    confidence = detections[det_idx, 4]
                    
                    # Skip low confidence detections
                    if confidence < self.conf_threshold:
                        continue
                    
                    # Initialize track if needed
                    if track_id not in self.track_history:
                        self._initialize_track(track_id, frame_idx)
                    
                    # Extract region for embedding
                    try:
                        region = self._extract_region(frame, bbox)
                        if region is None or region.size == 0:
                            self.logger.warning(f"Empty region for track {track_id}")
                            # Store zero embedding for empty region
                            zero_emb = torch.zeros(512, device=self.device)
                            self.track_embeddings[track_id].append(zero_emb)
                            continue
                        
                        # Get embedding with retry logic
                        embedding = None
                        for attempt in range(self.max_retries):
                            try:
                                embedding = self._process_embedding(region)
                                if embedding is not None:
                                    break
                            except Exception as e:
                                if attempt == self.max_retries - 1:
                                    self.logger.error(f"Failed to compute embedding after {self.max_retries} attempts: {str(e)}")
                                    # Store zero embedding on final failure
                                    zero_emb = torch.zeros(512, device=self.device)
                                    self.track_embeddings[track_id].append(zero_emb)
                                    raise
                                time.sleep(self.retry_delay)
                        
                        if embedding is None:
                            # Store zero embedding for None result
                            zero_emb = torch.zeros(512, device=self.device)
                            self.track_embeddings[track_id].append(zero_emb)
                            continue
                        
                        # Get view information if available
                        view_info = None
                        if self.multi_view is not None:
                            try:
                                view_info = self.multi_view.get_view_info(frame, bbox)
                            except Exception as e:
                                self.logger.warning(f"Failed to get view info: {str(e)}")
                        
                        # Update or create track
                        track_info = self._update_track(
                            track_id=int(track_id),
                            bbox=bbox,
                            embedding=embedding,
                            confidence=float(confidence),
                            frame_idx=frame_idx,
                            view_info=view_info
                        )
                        
                        if track_info is not None:
                            processed_tracks.append(track_info)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing track {track_id}: {str(e)}")
                        # Store zero embedding for any error
                        zero_emb = torch.zeros(512, device=self.device)
                        if track_id not in self.track_embeddings:
                            self.track_embeddings[track_id] = deque(maxlen=self.max_embedding_history)
                        self.track_embeddings[track_id].append(zero_emb)
                        continue
                    
                except Exception as e:
                    self.logger.error(f"Error processing tracked object: {str(e)}")
                    continue
            
            return processed_tracks
            
        except Exception as e:
            self.logger.error(f"Detection batch processing failed: {str(e)}")
            if isinstance(e, TimeoutException):
                raise
            return []

    def _update_track(self, track_id, bbox, embedding, confidence, frame_idx, view_info=None):
        """Update track information with improved persistence and temporal consistency."""
        try:
            # Check age limit first
            if track_id in self.track_ages and self.track_ages[track_id] > self.max_age:
                self.logger.debug(f"Track {track_id} exceeded max age {self.max_age}")
                return False
                
            # Check track count limit
            if len(self.track_embeddings) >= self.max_tracks:
                self.logger.debug(f"Track count limit reached ({self.max_tracks})")
                return False
            
            # Get track history
            history = self.track_history[track_id]
            changes = self.track_id_changes[track_id]
            
            # Update track statistics
            history['update_count'] += 1
            history['total_updates'] += 1
            history['consecutive_misses'] = 0
            history['last_update_time'] = time.time()
            history['last_update_frame'] = frame_idx
            
            # Convert embedding to tensor and move to device with proper error handling
            try:
                if isinstance(embedding, np.ndarray):
                    embedding = torch.from_numpy(embedding).float().to(self.device)
                elif isinstance(embedding, torch.Tensor):
                    embedding = embedding.detach().float().to(self.device)
                else:
                    embedding = torch.tensor(embedding, dtype=torch.float32, device=self.device)
                
                # Ensure embedding is normalized
                embedding = F.normalize(embedding, p=2, dim=0)
                
                # Calculate embedding quality
                if len(history['embedding_history']) > 0:
                    prev_emb = history['embedding_history'][-1]
                    similarity = torch.dot(embedding, prev_emb) / (torch.norm(embedding) * torch.norm(prev_emb))
                    embedding_quality = max(0.0, min(1.0, similarity.item()))
                else:
                    embedding_quality = 1.0
                
                # Update embedding quality metrics
                history['embedding_quality'] = embedding_quality
                history['average_embedding_quality'] = (
                    (history['average_embedding_quality'] * (history['total_updates'] - 1) + embedding_quality) 
                    / history['total_updates']
                )
                
            except Exception as e:
                self.logger.error(f"Error processing embedding for track {track_id}: {str(e)}")
                embedding_quality = 0.0
                embedding = torch.zeros(512, device=self.device)
            
            # Store last valid embedding (ensure it's a clone to avoid reference issues)
            history['last_valid_embedding'] = embedding.clone()
            changes['last_valid_embedding'] = embedding.clone()
            
            # Update track history with proper device handling
            history_entry = {
                'frame_idx': int(frame_idx),
                'bbox': [float(x) for x in bbox],  # Ensure bbox values are float
                'confidence': float(confidence),
                'embedding_quality': float(embedding_quality),
                'embedding_factor': float(history['embedding_factor']),
                'age': int(self.track_ages[track_id]),
                'temporal_consistency': float(history['temporal_consistency']),
                'track_quality': float(history['track_quality']),
                'size_score': float(history['size_score']),
                'movement_score': float(history['movement_score']),
                'behavior_score': float(history.get('behavior_score', 0.0))
            }
            
            # Update history collections
            history['history'].append(history_entry)
            history['embedding_history'].append(embedding)
            history['confidence_history'].append(float(confidence))
            history['bbox_history'].append([float(x) for x in bbox])
            
            # Update track age and last seen
            history['last_seen'] = frame_idx
            self.track_ages[track_id] = frame_idx - history['created_frame']
            
            # Update confidence metrics
            history['average_confidence'] = (
                (history['average_confidence'] * (history['total_updates'] - 1) + confidence) 
                / history['total_updates']
            )
            
            # Update temporal consistency
            if len(history['bbox_history']) > 1:
                prev_bbox = history['bbox_history'][-2]
                curr_bbox = history['bbox_history'][-1]
                
                # Calculate movement score
                movement = np.linalg.norm(np.array(curr_bbox[:2]) - np.array(prev_bbox[:2]))
                history['movement_score'] = float(movement)
                
                # Calculate size score
                curr_area = (curr_bbox[2] - curr_bbox[0]) * (curr_bbox[3] - curr_bbox[1])
                prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
                size_ratio = min(curr_area, prev_area) / max(curr_area, prev_area)
                history['size_score'] = float(size_ratio)
                
                # Update temporal consistency with weighted factors
                temporal_consistency = (
                    0.4 * embedding_quality +  # Embedding similarity
                    0.3 * size_ratio +        # Size consistency
                    0.2 * (1.0 - min(1.0, movement / 100.0)) +  # Movement consistency
                    0.1 * confidence          # Detection confidence
                )
                
                # Apply exponential moving average
                alpha = 0.7
                history['temporal_consistency'] = float(
                    alpha * temporal_consistency + 
                    (1 - alpha) * history['temporal_consistency']
                )
            
            # Update track quality based on multiple factors
            track_quality = (
                0.3 * history['temporal_consistency'] +
                0.3 * history['average_confidence'] +
                0.2 * history['embedding_quality'] +
                0.1 * history['size_score'] +
                0.1 * (1.0 - min(1.0, history['consecutive_misses'] / self.max_age))
            )
            history['track_quality'] = float(track_quality)
            
            # Update persistence score
            persistence_score = (
                0.4 * track_quality +
                0.3 * (1.0 - history['consecutive_misses'] / self.max_age) +
                0.2 * history['average_confidence'] +
                0.1 * history['embedding_quality']
            )
            history['persistence_score'] = float(persistence_score)
            
            # Update track ID changes
            changes.update(history)
            
            # Store last update state
            history['last_bbox'] = [float(x) for x in bbox]
            history['last_confidence'] = float(confidence)
            history['last_embedding_quality'] = float(embedding_quality)
            history['last_temporal_consistency'] = float(history['temporal_consistency'])
            history['last_track_quality'] = float(track_quality)
            history['last_size_score'] = float(history['size_score'])
            history['last_movement_score'] = float(history['movement_score'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating track {track_id}: {str(e)}", exc_info=True)
            # Store error in track history
            if track_id in self.track_history:
                self.track_history[track_id]['errors'].append({
                    'frame': frame_idx,
                    'error': str(e),
                    'time': time.time()
                })
            return False

    def _is_occluded(self, track, bbox):
        """Check if a track is occluded based on its history."""
        try:
            if len(track['history']) < 2:
                return False
            
            # Get recent bboxes
            recent_bboxes = track['history'][-5:]  # Look at last 5 frames
            
            # Calculate IoU with current bbox
            ious = [iou(bbox, b['bbox']) for b in recent_bboxes]
            max_iou = max(ious)
            
            # Check for significant overlap with other tracks
            for other_id, other_track in self.track_history.items():
                if other_id == track['id']:
                    continue
                
                if not other_track['history']:
                    continue
                
                other_bbox = other_track['history'][-1]['bbox']
                other_iou = iou(bbox, other_bbox)
                
                if other_iou > self.iou_threshold:
                    return True
            
            # Check for sudden size changes
            if len(track['history']) >= 2:
                prev_size = (track['history'][-2]['bbox'][2] - track['history'][-2]['bbox'][0]) * \
                           (track['history'][-2]['bbox'][3] - track['history'][-2]['bbox'][1])
                curr_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                size_ratio = curr_size / prev_size
                
                if size_ratio < 0.5 or size_ratio > 2.0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Occlusion check failed: {str(e)}")
            return False

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
                
                # Calculate track health score
                health_score = (
                    0.3 * history['temporal_consistency'] +
                    0.3 * history['average_confidence'] +
                    0.2 * history['embedding_quality'] +
                    0.1 * history['size_score'] +
                    0.1 * (1.0 - min(1.0, history['consecutive_misses'] / self.max_age))
                )
                
                # Remove tracks based on multiple criteria
                should_remove = False
                reason = []
                
                if track_age > self.max_age:
                    should_remove = True
                    reason.append(f"age={track_age}")
                if frames_since_seen > self.max_age:
                    should_remove = True
                    reason.append(f"unseen={frames_since_seen}")
                if health_score < 0.3 and track_age > self.max_age // 2:
                    should_remove = True
                    reason.append(f"health={health_score:.2f}")
                if len(self.active_tracks) > self.max_tracks:
                    should_remove = True
                    reason.append("track_limit")
                if history['consecutive_misses'] > self.max_age:
                    should_remove = True
                    reason.append(f"misses={history['consecutive_misses']}")
                
                if should_remove:
                    tracks_to_remove.add(track_id)
                    self.logger.debug(
                        f"Marking track {track_id} for removal: {', '.join(reason)}, "
                        f"age={track_age}, last_seen={frames_since_seen}, "
                        f"health={health_score:.2f}"
                    )
            
            # If we're over the track limit, prioritize keeping healthy tracks
            if len(self.active_tracks) > self.max_tracks:
                # Sort tracks by health score
                track_health = {
                    tid: (
                        0.3 * self.track_history[tid]['temporal_consistency'] +
                        0.3 * self.track_history[tid]['average_confidence'] +
                        0.2 * self.track_history[tid]['embedding_quality'] +
                        0.1 * self.track_history[tid]['size_score'] +
                        0.1 * (1.0 - min(1.0, self.track_history[tid]['consecutive_misses'] / self.max_age))
                    )
                    for tid in self.active_tracks
                }
                sorted_tracks = sorted(
                    self.active_tracks,
                    key=lambda x: (track_health.get(x, 0), -self.track_history[x]['created_frame'])
                )
                excess_tracks = sorted_tracks[:len(self.active_tracks) - self.max_tracks]
                tracks_to_remove.update(excess_tracks)
                self.logger.debug(f"Removing excess tracks based on health: {excess_tracks}")
            
            # Second pass: remove identified tracks and clean up resources
            for track_id in tracks_to_remove:
                try:
                    # Log final state before removal
                    history = self.track_history[track_id]
                    final_state = {
                        'age': current_frame - history['created_frame'],
                        'last_seen': history['last_seen'],
                        'embedding_count': len(history['embedding_history']),
                        'behavior_count': len(history['behaviors']),
                        'temporal_consistency': history['temporal_consistency'],
                        'average_confidence': history['average_confidence'],
                        'embedding_quality': history['embedding_quality'],
                        'health_score': (
                            0.3 * history['temporal_consistency'] +
                            0.3 * history['average_confidence'] +
                            0.2 * history['embedding_quality'] +
                            0.1 * history['size_score'] +
                            0.1 * (1.0 - min(1.0, history['consecutive_misses'] / self.max_age))
                        )
                    }
                    self.logger.debug(f"Removing track {track_id} with final state: {final_state}")
                    
                    # Clear embeddings from GPU memory
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
                    
                    # Clear last valid embedding
                    if 'last_valid_embedding' in history:
                        emb = history['last_valid_embedding']
                        if isinstance(emb, torch.Tensor):
                            emb.cpu()
                            del emb
                    
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
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning up track {track_id}: {str(e)}", exc_info=True)
            
            # Force CUDA cache clear if needed
            if torch.cuda.is_available() and tracks_to_remove:
                torch.cuda.empty_cache()
                self.logger.debug("Cleared CUDA cache after track cleanup")
            
            # Update last cleanup frame
            self.last_cleanup_frame = current_frame
                
        except Exception as e:
            self.logger.error(f"Error in track cleanup: {str(e)}", exc_info=True)
            # Try to recover by forcing CUDA cache clear
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    self.logger.debug("Forced CUDA cache clear after cleanup error")
                except Exception as cache_error:
                    self.logger.error(f"Failed to clear CUDA cache: {str(cache_error)}")

    def _initialize_track(self, track_id, frame_idx):
        """Initialize a new track with improved persistence and state management."""
        try:
            # Initialize track history with all required fields and proper type casting
            self.track_history[track_id] = {
                'created_frame': int(frame_idx),
                'last_seen': int(frame_idx),
                'history': deque(maxlen=self.max_track_history),  # Use deque for automatic size limiting
                'behaviors': deque(maxlen=self.max_behavior_history),
                'views': set(),
                'temporal_consistency': float(1.0),
                'device_transitions': [],
                'errors': [],
                'occlusion_count': int(0),
                'last_occlusion_frame': None,
                'behavioral_markers': set(),
                'embedding_history': deque(maxlen=self.max_embedding_history),
                'confidence_history': deque(maxlen=self.max_track_history),
                'bbox_history': deque(maxlen=self.max_track_history),
                'persistence_score': float(1.0),
                'last_valid_embedding': torch.zeros(512, dtype=torch.float32, device=self.device),
                'embedding_quality': float(1.0),
                'embedding_factor': float(1.0),
                'movement_score': float(0.0),
                'behavior_sequence': deque(maxlen=self.max_behavior_history),
                'last_behavior': None,
                'behavior_duration': int(0),
                'behavior_transitions': int(0),
                'track_quality': float(1.0),
                'size_score': float(1.0),
                'frame_size': (int(1), int(1)),
                'device': str(self.device),
                'device_type': str(self.device.type),
                'track_id': int(track_id),  # Store original track ID
                'id_changes': [],  # Track ID change history
                'last_update_time': time.time(),
                'update_count': int(0),
                'consecutive_misses': int(0),
                'max_consecutive_misses': int(0),
                'total_updates': int(0),
                'total_misses': int(0),
                'average_confidence': float(0.0),
                'average_embedding_quality': float(0.0),
                'last_bbox': None,
                'last_confidence': float(0.0),
                'last_embedding_quality': float(0.0),
                'last_temporal_consistency': float(1.0),
                'last_track_quality': float(1.0),
                'last_size_score': float(1.0),
                'last_movement_score': float(0.0),
                'last_behavior_score': float(0.0),
                'last_update_frame': int(frame_idx),
                'last_update_bbox': None,
                'last_update_confidence': float(0.0),
                'last_update_embedding_quality': float(0.0),
                'last_update_temporal_consistency': float(1.0),
                'last_update_track_quality': float(1.0),
                'last_update_size_score': float(1.0),
                'last_update_movement_score': float(0.0),
                'last_update_behavior_score': float(0.0)
            }
            
            # Initialize track ID change history with same fields
            self.track_id_changes[track_id] = self.track_history[track_id].copy()
            
            # Initialize embeddings collections with proper device handling and size limits
            self.track_embeddings[track_id] = deque(maxlen=self.max_embedding_history)
            self.track_head_embeddings[track_id] = deque(maxlen=self.max_embedding_history)
            
            # Add initial zero embedding with proper device placement
            zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
            self.track_embeddings[track_id].append(zero_emb)
            self.track_head_embeddings[track_id].append(zero_emb)
            
            # Add to active tracks and initialize age
            self.active_tracks.add(track_id)
            self.track_ages[track_id] = int(0)
            
            # Log initialization with detailed state
            self.logger.debug(
                f"Initialized track {track_id} at frame {frame_idx} on device {self.device} "
                f"with max history {self.max_track_history}, max embeddings {self.max_embedding_history}, "
                f"max behaviors {self.max_behavior_history}"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing track {track_id}: {str(e)}", exc_info=True)
            # Clean up any partially initialized state
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.track_id_changes:
                del self.track_id_changes[track_id]
            if track_id in self.track_embeddings:
                del self.track_embeddings[track_id]
            if track_id in self.track_head_embeddings:
                del self.track_head_embeddings[track_id]
            if track_id in self.track_ages:
                del self.track_ages[track_id]
            self.active_tracks.discard(track_id)
            raise

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
            if frame is None or not isinstance(frame, np.ndarray):
                raise ValueError("Invalid frame input")
            
            # Handle empty detections
            if detections is None or len(detections) == 0:
                detections = np.empty((0, 5), dtype=np.float32)
            
            # Convert detections to proper format if needed
            if isinstance(detections, list):
                # Handle list of dictionaries
                if isinstance(detections[0], dict):
                    detections = np.array([[float(d['bbox'][0]), float(d['bbox'][1]), 
                                          float(d['bbox'][2]), float(d['bbox'][3]), 
                                          float(d.get('score', 1.0))] for d in detections], 
                                         dtype=np.float32)
                # Handle list of lists/arrays
                else:
                    detections = np.array(detections, dtype=np.float32)
            
            # Ensure detections is 2D array with correct shape
            if detections.ndim == 1:
                detections = detections.reshape(1, -1)
            
            # Validate shape and add confidence if missing
            if detections.shape[1] == 4:  # Only bbox coordinates
                confidence = np.ones((detections.shape[0], 1), dtype=np.float32)
                detections = np.hstack([detections, confidence])
            elif detections.shape[1] != 5:
                raise ValueError(f"Invalid detections shape: {detections.shape}. Expected (N, 4) or (N, 5)")
            
            # Ensure all values are float32
            detections = detections.astype(np.float32)
            
            # Increment frame counter
            self.frame_count += 1
            
            # Update tracker with retry logic
            tracked_objects = None
            for attempt in range(self.max_retries):
                try:
                    tracked_objects = self.tracker.update(detections)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"Failed to update tracker after {self.max_retries} attempts: {str(e)}")
                        raise
                    time.sleep(self.retry_delay)
            
            if tracked_objects is None or len(tracked_objects) == 0:
                return np.empty((0, 5), dtype=np.float32)
            
            # Process each tracked object
            active_tracks = []
            for track in tracked_objects:
                try:
                    x1, y1, x2, y2, track_id = track
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    confidence = float(track[4]) if len(track) > 4 else 1.0
                    
                    # Initialize track if needed
                    if track_id not in self.track_history:
                        self._initialize_track(track_id, self.frame_count)
                    
                    # Update temporal consistency
                    if track_id in self.track_history:
                        # Calculate temporal consistency based on embedding history
                        if len(self.track_embeddings[track_id]) > 1:
                            prev_emb = self.track_embeddings[track_id][-2]
                            curr_emb = self.track_embeddings[track_id][-1]
                            
                            # Ensure both embeddings are on device and float type
                            if isinstance(prev_emb, np.ndarray):
                                prev_emb = torch.from_numpy(prev_emb).float().to(self.device)
                            elif isinstance(prev_emb, torch.Tensor):
                                prev_emb = prev_emb.detach().float().to(self.device)
                            else:
                                prev_emb = torch.tensor(prev_emb, dtype=torch.float32, device=self.device)
                                
                            if isinstance(curr_emb, np.ndarray):
                                curr_emb = torch.from_numpy(curr_emb).float().to(self.device)
                            elif isinstance(curr_emb, torch.Tensor):
                                curr_emb = curr_emb.detach().float().to(self.device)
                            else:
                                curr_emb = torch.tensor(curr_emb, dtype=torch.float32, device=self.device)
                            
                            # Calculate cosine similarity
                            similarity = torch.dot(prev_emb, curr_emb) / (torch.norm(prev_emb) * torch.norm(curr_emb))
                            temporal_consistency = max(0.0, min(1.0, similarity.item()))
                            
                            # Update temporal consistency with exponential moving average
                            alpha = 0.7  # Weight for new value
                            old_consistency = float(self.track_history[track_id]['temporal_consistency'])
                            self.track_history[track_id]['temporal_consistency'] = float(
                                alpha * temporal_consistency + (1 - alpha) * old_consistency
                            )
                            
                            # Update track quality based on temporal consistency
                            self.track_history[track_id]['track_quality'] = float(self.track_history[track_id]['temporal_consistency'])
                            self.track_id_changes[track_id]['track_quality'] = float(self.track_history[track_id]['temporal_consistency'])
                    
                    # Update track history with proper device handling
                    history_entry = {
                        'frame_idx': int(self.frame_count),
                        'bbox': bbox,
                        'confidence': float(confidence),
                        'temporal_consistency': float(self.track_history[track_id]['temporal_consistency']),
                        'track_quality': float(self.track_history[track_id]['track_quality']),
                        'age': int(self.track_ages.get(track_id, 0))
                    }
                    
                    # Update history lists with proper limits
                    if len(self.track_history[track_id]['history']) >= self.max_track_history:
                        self.track_history[track_id]['history'].pop(0)
                        self.track_id_changes[track_id]['history'].pop(0)
                    self.track_history[track_id]['history'].append(history_entry)
                    self.track_id_changes[track_id]['history'].append(history_entry)
                    
                    # Add to active tracks with proper type conversion
                    active_tracks.append([float(x1), float(y1), float(x2), float(y2), float(track_id)])
                
                except Exception as e:
                    self.logger.error(f"Error processing track: {str(e)}")
                    continue
            
            return np.array(active_tracks, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in update: {str(e)}")
            return np.empty((0, 5), dtype=np.float32)

    def extract_crow_image(self, frame, bbox, padding=0.3, min_size=100):
        """Extract a cropped image of a crow from the frame using robust validation and enhancement logic.
        Args:
            frame: Input frame (numpy array)
            bbox: Bounding box in [x1, y1, x2, y2] format
            padding: Padding factor around the bbox (default: 0.3)
            min_size: Minimum size for valid bbox (default: 100)
        Returns:
            Dictionary containing 'full' and 'head' tensors, or None if extraction fails
        """
        try:
            # Use the robust standalone function for extraction
            return extract_crow_image(frame, bbox, padding=padding, min_size=min_size)
        except Exception as e:
            self.logger.error(f"Error extracting crow image: {str(e)}")
            return None

    def _initialize_models(self, model_path=None, strict_mode=False):
        """Initialize models with improved error handling and device management."""
        try:
            # Create model on CPU first
            from models import CrowResNetEmbedder
            self.model = CrowResNetEmbedder(embedding_dim=512)
            
            if model_path is not None:
                # Load state dict with improved error handling
                try:
                    # Always load to CPU first
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # Handle state dict mismatch
                    model_state_dict = self.model.state_dict()
                    filtered_state_dict = {}
                    
                    # Only load matching keys and shapes
                    for k, v in state_dict.items():
                        if k in model_state_dict and v.shape == model_state_dict[k].shape:
                            filtered_state_dict[k] = v
                        else:
                            self.logger.warning(f"Skipping incompatible key: {k}")
                    
                    if strict_mode:
                        # In strict mode, require all keys to match
                        if len(filtered_state_dict) != len(model_state_dict):
                            raise ModelError(f"Strict mode: Model state dict mismatch. Expected {len(model_state_dict)} parameters, got {len(filtered_state_dict)}")
                        # Also verify all shapes match
                        for k, v in filtered_state_dict.items():
                            if v.shape != model_state_dict[k].shape:
                                raise ModelError(f"Strict mode: Shape mismatch for key {k}. Expected {model_state_dict[k].shape}, got {v.shape}")
                        # Load with strict=True in strict mode
                        self.model.load_state_dict(filtered_state_dict, strict=True)
                    else:
                        # In non-strict mode, load what we can
                        self.model.load_state_dict(filtered_state_dict, strict=False)
                        self.logger.info(f"Loaded {len(filtered_state_dict)}/{len(model_state_dict)} model parameters")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load state dict: {str(e)}")
                    if strict_mode:
                        # In strict mode, raise the error and don't fall back
                        raise ModelError(f"Failed to load state dict in strict mode: {str(e)}")
                    else:
                        self.logger.warning("Using randomly initialized model")
                
            # Move model to target device
            self.model = self.model.to(self.device)
            
            # Set model to eval mode
            self.model.eval()
            
            # Verify embedding dimension
            try:
                test_input = torch.randn(1, 3, 224, 224, device=self.device)
                with torch.no_grad():
                    test_output = self.model(test_input)
                embedding_dim = test_output.shape[1]
                if embedding_dim != 512:
                    if strict_mode:
                        raise ModelError(f"Model embedding dimension mismatch: expected 512, got {embedding_dim}")
                    else:
                        self.logger.warning(f"Model embedding dimension mismatch: expected 512, got {embedding_dim}")
                
                self.logger.info(f"Model loaded successfully with embedding dimension {embedding_dim}")
                
            except Exception as e:
                self.logger.error(f"Failed to verify embedding dimension: {str(e)}")
                if strict_mode:
                    raise ModelError(f"Failed to verify embedding dimension: {str(e)}")
                else:
                    self.logger.warning("Skipping embedding dimension verification")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            if hasattr(self, 'model'):
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if strict_mode:
                raise ModelError(f"Model initialization failed: {str(e)}")
            else:
                self.logger.warning("Using randomly initialized model")
                self.model = CrowResNetEmbedder(embedding_dim=512).to(self.device)

    def _init_multi_view_extractor(self, strict_mode):
        """Initialize multi-view extractor if stride > 1."""
        try:
            if self.multi_view_stride > 1:
                if not isinstance(self.multi_view_stride, int) or self.multi_view_stride < 1:
                    raise ValueError(f"Invalid multi_view_stride: {self.multi_view_stride}")
                self.multi_view = create_multi_view_extractor(stride=self.multi_view_stride)
                self.logger.info(f"Multi-view extractor initialized with stride {self.multi_view_stride}")
            else:
                self.multi_view = None
                self.logger.info("Multi-view extractor disabled (stride <= 1)")
        except Exception as e:
            self.logger.error(f"Failed to initialize multi-view extractor: {str(e)}")
            self.multi_view = None
            if strict_mode:
                raise ModelError(f"Failed to initialize multi-view extractor: {str(e)}")
            else:
                self.logger.warning("Multi-view extractor initialization failed, continuing without it")

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

    def _process_embedding(self, img_tensors, key=None, return_tensor=False):
        """Compute feature embeddings for both head and body regions using the instance's model and device."""
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.logger.error("Model not initialized")
                zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()

            with torch.no_grad():
                if key is not None:
                    # Process single region
                    if not isinstance(img_tensors, (torch.Tensor, np.ndarray)):
                        self.logger.error(f"Invalid input type for {key}: {type(img_tensors)}")
                        zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                        return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()

                    # Convert to tensor if needed and ensure proper type
                    if isinstance(img_tensors, np.ndarray):
                        img_tensors = torch.from_numpy(img_tensors).float()
                    elif isinstance(img_tensors, torch.Tensor):
                        img_tensors = img_tensors.float()
                    else:
                        img_tensors = torch.tensor(img_tensors, dtype=torch.float32)

                    # Ensure tensor is on device and has correct shape
                    if img_tensors.dim() != 4:
                        img_tensors = img_tensors.unsqueeze(0)
                    if img_tensors.size(1) != 3:
                        self.logger.error(f"Invalid channels for {key}: {img_tensors.size(1)}")
                        zero_emb = torch.zeros(512, dtype=torch.float32, device=self.device)
                        return zero_emb if return_tensor else zero_emb.detach().cpu().numpy()

                    # Move to device and normalize
                    img_tensors = img_tensors.to(self.device)
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

        except Exception as e:
            self.logger.error(f"Error in _process_embedding: {str(e)}")
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

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

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
            if track_id in tracker.track_embeddings and tracker.track_embeddings[track_id]:
                embedding = tracker.track_embeddings[track_id][-1]
                db_crow_id = save_crow_embedding(
                    embedding,
                    video_path=video_path,
                    frame_number=frame_idx,
                    confidence=track[4]
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
