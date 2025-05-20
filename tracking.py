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

def timeout(seconds):
    """Decorator for function timeout."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set signal handler
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable alarm
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
    """Compute feature embeddings for both head and body regions."""
    embeddings = {}
    with torch.no_grad():
        if torch.cuda.is_available():
            for key, tensor in img_tensors.items():
                img_tensors[key] = tensor.cuda()
        for key, tensor in img_tensors.items():
            features = model(tensor)
            embeddings[key] = features.squeeze().cpu().numpy()
    combined = np.concatenate([
        embeddings['full'],
        embeddings['head']
    ])
    return combined, embeddings

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
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3, embedding_threshold=0.5, model_path=None, conf_threshold=0.5, multi_view_stride=1):
        """Initialize the enhanced tracker with improved parameters."""
        try:
            # Tracking parameters with improved defaults
            self.max_age = max_age
            self.min_hits = min_hits
            self.iou_threshold = iou_threshold
            self.embedding_threshold = embedding_threshold
            self.conf_threshold = conf_threshold
            self.multi_view_stride = multi_view_stride
            
            # Memory management parameters
            self.max_track_history = 100  # Maximum number of history entries per track
            self.max_embedding_history = 100  # Maximum number of embeddings per track
            self.max_behavior_history = 50  # Maximum number of behaviors per track
            self.cleanup_interval = 10  # Frames between cleanup operations
            
            # Retry parameters
            self.max_retries = 3
            self.retry_delay = 0.1  # seconds
            
            # Initialize logger
            self._configure_logger()
            
            # Initialize device with improved handling
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, using CPU")
                self.device = torch.device('cpu')
            else:
                try:
                    self.device = torch.device('cuda')
                    # Test CUDA device
                    torch.zeros(1).to(self.device)
                except Exception as e:
                    self.logger.error(f"CUDA initialization failed: {str(e)}")
                    raise DeviceError(f"Failed to initialize CUDA device: {str(e)}")
            
            self.logger.info(f"Using device: {self.device}")
            
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
            self._initialize_models(model_path)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracker: {str(e)}")
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
        """Context manager for temporary device changes with improved synchronization.
        
        Args:
            device: Target device (torch.device or str)
            
        Yields:
            None
        """
        original_device = self.device
        try:
            if self.to_device(device):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                yield
            else:
                raise RuntimeError(f"Failed to move models to device: {device}")
        finally:
            if self.to_device(original_device):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
    def to_device(self, device):
        """Move models to specified device with improved error handling and synchronization.
        
        Args:
            device: Target device (torch.device or str)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            DeviceError: If device movement fails
        """
        try:
            if isinstance(device, str):
                device = torch.device(device)
            if not isinstance(device, torch.device):
                raise ValueError(f"Invalid device type: {type(device)}")
                
            # Move models to device with synchronization
            if hasattr(self, 'model'):
                self.model = self.model.to(device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            if hasattr(self, 'multi_view_extractor'):
                # MultiViewExtractor is not a PyTorch model, so we don't need to move it to device
                # Verify multi-view extractor with a test image
                test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                test_views = self.multi_view_extractor.extract(test_img)
                if not isinstance(test_views, list) or len(test_views) == 0:
                    raise ModelError("Multi-view extractor failed to generate views")
            except Exception as e:
                raise ModelError(f"Failed to initialize multi-view extractor: {str(e)}")
            
            # Initialize color normalizer
            try:
                self.color_norm = ColorNormalizer()
                # Test color normalization
                test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                norm_img = self.color_norm.normalize(test_img)
                if norm_img.shape != test_img.shape:
                    raise ModelError("Color normalization shape mismatch")
            except Exception as e:
                raise ModelError(f"Failed to initialize color normalizer: {str(e)}")
            
            self.logger.info("All models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

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
    def _process_detection_batch(self, frame, detections):
        """Process a batch of detections with improved error handling and retries."""
        if not detections:
            return None
            
        try:
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                embeddings = {}
                for det in detections:
                    bbox = det['bbox']
                    crow_img = self.extract_crow_image(frame, bbox)
                    if crow_img is None:
                        continue
                        
                    # Process full body embedding
                    full_emb = self._process_embedding(crow_img, 'full')
                    if full_emb is not None:
                        full_emb = full_emb.squeeze()  # Remove batch dimension
                        if full_emb.shape != (512,):
                            full_emb = full_emb.reshape(512)
                        embeddings['full'] = [full_emb]
                        
                    # Process head embedding if available
                    head_img = self.extract_head_image(crow_img)
                    if head_img is not None:
                        head_emb = self._process_embedding(head_img, 'head')
                        if head_emb is not None:
                            head_emb = head_emb.squeeze()  # Remove batch dimension
                            if head_emb.shape != (512,):
                                head_emb = head_emb.reshape(512)
                            embeddings['head'] = [head_emb]
                            
                return embeddings
                
        except Exception as e:
            self.logger.error(f"Error processing embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to process embeddings: {str(e)}")

    def _process_embedding(self, img_tensor, region='full'):
        """Process a single image tensor to get its embedding.
        
        Args:
            img_tensor: Input tensor of shape (3, H, W)
            region: Region type ('full' or 'head')
            
        Returns:
            numpy.ndarray: Normalized embedding vector of shape (512,)
            
        Raises:
            EmbeddingError: If embedding processing fails
        """
        try:
            with torch.no_grad():
                # Ensure tensor is on correct device
                img_tensor = img_tensor.to(self.device)
                
                # Add batch dimension if needed
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                
                # Get embedding with proper device placement
                embedding = self.model(img_tensor)
                
                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)
                
                # Verify normalization
                norm = torch.norm(embedding, p=2, dim=1)
                if not torch.allclose(norm, torch.ones_like(norm), atol=1e-6):
                    raise EmbeddingError("Embedding normalization failed")
                
                # Convert to numpy and ensure correct shape
                embedding = embedding.squeeze().cpu().numpy()
                if embedding.shape != (512,):
                    embedding = embedding.reshape(512)
                
                return embedding
                
        except Exception as e:
            self.logger.error(f"Error processing embedding: {str(e)}")
            raise EmbeddingError(f"Failed to process embedding: {str(e)}")

    def _cleanup_old_tracks(self):
        """Clean up old tracks and free memory."""
        try:
            current_frame = self.frame_count
            if current_frame - self.last_cleanup_frame < self.cleanup_interval:
                return
            
            self.logger.debug("Running track cleanup...")
            
            # Remove old tracks
            tracks_to_remove = set()
            for track_id, history in self.track_history.items():
                if current_frame - history['last_seen'] > self.max_age:
                    tracks_to_remove.add(track_id)
            
            # Clean up track data
            for track_id in tracks_to_remove:
                # Remove from all collections
                self.track_history.pop(track_id, None)
                self.track_embeddings.pop(track_id, None)
                self.track_head_embeddings.pop(track_id, None)
                self.track_id_changes.pop(track_id, None)
                self.track_ages.pop(track_id, None)
                self.active_tracks.discard(track_id)
                
                # Log cleanup
                self.logger.debug(f"Removed old track {track_id}")
            
            # Force garbage collection
            if len(tracks_to_remove) > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.last_cleanup_frame = current_frame
            self.logger.debug(f"Cleanup complete: removed {len(tracks_to_remove)} tracks")
            
        except Exception as e:
            self.logger.error(f"Error during track cleanup: {str(e)}")

    def update(self, frame, detections):
        """Update tracks with new detections."""
        try:
            self.frame_count += 1
            
            # Run cleanup if needed
            self._cleanup_old_tracks()
            
            # Convert detections to proper format if needed
            if isinstance(detections, list):
                # Convert list of dicts to numpy array
                det_boxes = np.array([d['bbox'] for d in detections if d['score'] >= self.conf_threshold])
                if len(det_boxes) == 0:
                    self.logger.debug("No valid detections in frame")
                    return np.empty((0, 5), dtype=np.float32)
            elif isinstance(detections, np.ndarray):
                # Ensure detections are in correct format (x1, y1, x2, y2)
                if detections.shape[1] >= 4:
                    det_boxes = detections[:, :4]
                else:
                    self.logger.error("Invalid detection array shape")
                    return np.empty((0, 5), dtype=np.float32)
            else:
                self.logger.error(f"Unsupported detections type: {type(detections)}")
                return np.empty((0, 5), dtype=np.float32)
            
            self.logger.debug(f"Processing frame {self.frame_count} with {len(det_boxes)} detections")
            
            # Get tracked objects from SORT
            tracked_objects = self.tracker.update(det_boxes)
            
            if len(tracked_objects) == 0:
                return np.empty((0, 5), dtype=np.float32)
            
            # Process tracked objects with improved embedding handling
            active_tracks = []
            for track in tracked_objects:
                track_id = int(track[4])
                bbox = track[:4]
                
                # Initialize or update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'created_frame': self.frame_count,
                        'last_seen': self.frame_count,
                        'history': deque(maxlen=self.max_track_history),
                        'behaviors': deque(maxlen=self.max_behavior_history),
                        'views': set(),
                        'temporal_consistency': 0.0
                    }
                    # Initialize track ID change history
                    self.track_id_changes[track_id] = {
                        'created_frame': self.frame_count,
                        'last_seen': self.frame_count,
                        'history': deque(maxlen=self.max_track_history),
                        'behaviors': deque(maxlen=self.max_behavior_history),
                        'device_transitions': [],
                        'errors': []
                    }
                    self.active_tracks.add(track_id)
                else:
                    # Update last seen
                    self.track_history[track_id]['last_seen'] = self.frame_count
                    self.track_id_changes[track_id]['last_seen'] = self.frame_count
                
                # Add history entry
                history_entry = {
                    'frame': self.frame_count,
                    'event': 'updated',
                    'age': self.frame_count - self.track_history[track_id]['created_frame'],
                    'bbox': bbox.tolist(),
                    'device': self.device.type
                }
                
                # Add behavior if present
                if isinstance(detections, list):
                    for det in detections:
                        if compute_iou(bbox, det['bbox']) > self.iou_threshold and 'behavior' in det:
                            history_entry['behavior'] = det['behavior']
                            self.track_history[track_id]['behaviors'].append(det['behavior'])
                            self.track_id_changes[track_id]['behaviors'].append(det['behavior'])
                            break
                
                self.track_history[track_id]['history'].append(history_entry)
                self.track_id_changes[track_id]['history'].append(history_entry)
                
                # Process embeddings
                try:
                    if isinstance(detections, list):
                        matching_det = None
                        best_iou = 0
                        for det in detections:
                            iou = compute_iou(bbox, det['bbox'])
                            if iou > max(self.iou_threshold, best_iou):
                                matching_det = det
                                best_iou = iou
                        
                        if matching_det:
                            # Extract and process crow image
                            crow_img = self.extract_crow_image(frame, matching_det['bbox'])
                            if crow_img is not None:
                                # Process full body embedding
                                full_emb = self._process_embedding(crow_img['full'], 'full')
                                if full_emb is not None:
                                    self.track_embeddings[track_id].append(full_emb)
                                
                                # Process head embedding if available
                                if 'head' in crow_img:
                                    head_emb = self._process_embedding(crow_img['head'], 'head')
                                    if head_emb is not None:
                                        self.track_head_embeddings[track_id].append(head_emb)
                except Exception as e:
                    self.logger.error(f"Error processing embeddings for track {track_id}: {str(e)}")
                    self.track_id_changes[track_id]['errors'].append({
                        'frame': self.frame_count,
                        'error': str(e),
                        'type': 'embedding'
                    })
                
                active_tracks.append(track)
            
            return np.array(active_tracks)
            
        except Exception as e:
            self.logger.error(f"Error updating tracks: {str(e)}")
            raise

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

    def _initialize_models(self, model_path=None):
        """Initialize models with improved error handling.
        
        Raises:
            ModelError: If model initialization fails
        """
        try:
            self.logger.info("Initializing models...")
            
            # Initialize SORT tracker
            self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
            
            # Initialize main model
            try:
                self.model = self._load_model(model_path)
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Verify model output shape
                test_input = torch.randn(1, 3, 224, 224, device=self.device)
                with torch.no_grad():
                    test_output = self.model(test_input)
                    if test_output.shape[1] != 512:
                        raise ModelError(f"Model output shape mismatch: expected 512, got {test_output.shape[1]}")
            except Exception as e:
                raise ModelError(f"Failed to load or initialize model: {str(e)}")
            
            # Initialize multi-view extractor
            try:
                self.multi_view_extractor = create_multi_view_extractor()
                # MultiViewExtractor is not a PyTorch model, so we don't need to move it to device
                # Verify multi-view extractor with a test image
                test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                test_views = self.multi_view_extractor.extract(test_img)
                if not isinstance(test_views, list) or len(test_views) == 0:
                    raise ModelError("Multi-view extractor failed to generate views")
            except Exception as e:
                raise ModelError(f"Failed to initialize multi-view extractor: {str(e)}")
            
            # Initialize color normalizer
            try:
                self.color_norm = ColorNormalizer()
                # Test color normalization
                test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                norm_img = self.color_norm.normalize(test_img)
                if norm_img.shape != test_img.shape:
                    raise ModelError("Color normalization shape mismatch")
            except Exception as e:
                raise ModelError(f"Failed to initialize color normalizer: {str(e)}")
            
            self.logger.info("All models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise

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
