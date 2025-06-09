import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, deque
from db import save_crow_embedding, get_all_crows, get_crow_history, add_behavioral_marker
from sort import Sort
from scipy.spatial.distance import cdist
from models import CrowResNetEmbedder # Assuming CrowResNetEmbedder is correctly defined in models.py
# from model import CrowResNetEmbedder as CrowResNetEmbedder_model_py # If needed to check model.py version
from utilities.color_normalization import AdaptiveNormalizer, create_normalizer
from multi_view import create_multi_view_extractor
from ultralytics import YOLO
import torchvision
# from torchvision.models import resnet18 # We will use CrowResNetEmbedder primarily
import torch.nn as nn
import logging
import time
import threading
import gc
from functools import wraps
import types # Required for MethodType binding in load_triplet_model
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
from utilities.color_normalization import ColorNormalizer
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

# Load configuration at the start of the script
CONFIG = {}
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    logger.warning("config.json not found in tracking.py. Using default model paths.")
except json.JSONDecodeError:
    logger.warning("Error decoding config.json in tracking.py. Using default model paths.")

# --- PATCH: Platform check for signal.SIGALRM ---
IS_WINDOWS = platform.system() == 'Windows'

def timeout(seconds):
    """Decorator for function timeout."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if IS_WINDOWS:
                return func(*args, **kwargs) # SIGALRM not available, skip timeout
            def handler(signum, frame):
                raise TimeoutException(f"Function {func.__name__} timed out after {seconds} seconds")
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0) # Reset alarm
            return result
        return wrapper
    return decorator

# --- Model Loading Section ---
logger.info("--- Initializing and Loading Application Models ---")

# --- Model Loading Section ---
logger.info("--- Initializing and Loading Application Models ---")

# --- Model Loading Section ---
logger.info("--- Initializing and Loading Application Models ---")

# Determine model directory
model_dir_str = CONFIG.get('model_dir')
if not model_dir_str: # Handles None or empty string
    model_dir_path = Path('.') # Default to current directory if not specified or empty
else:
    model_dir_path = Path(model_dir_str)

# 1. Primary Crow Embedding Model
logger.info("Initializing primary Crow Embedding Model (CrowResNetEmbedder)...")
CROW_EMBEDDING_MODEL = CrowResNetEmbedder(embedding_dim=512) # Assuming embedding_dim is a parameter
crow_embedding_model_path_obj = model_dir_path / 'crow_resnet_triplet.pth'
try:
    # Load weights on CPU first to prevent potential CUDA OOM if model was saved on GPU
    CROW_EMBEDDING_MODEL.cpu()
    CROW_EMBEDDING_MODEL.load_state_dict(torch.load(str(crow_embedding_model_path_obj), map_location='cpu'))
    logger.info(f"Successfully loaded trained weights for CrowResNetEmbedder from '{str(crow_embedding_model_path_obj)}'.")
except FileNotFoundError:
    logger.warning(f"'{str(crow_embedding_model_path_obj)}' not found. CrowResNetEmbedder will use its initial (random or default pre-trained) weights. Ensure 'model_dir' in config.json is correct or the model is in the default location.")
except Exception as e:
    logger.warning(f"Could not load weights for CrowResNetEmbedder from '{str(crow_embedding_model_path_obj)}' due to an error: {e}. Using initial weights.")
CROW_EMBEDDING_MODEL.eval()
logger.info("CrowResNetEmbedder set to evaluation mode.")

# 2. Toy Detection Model (YOLO)
TOY_DETECTION_MODEL = None
toy_model_path_obj = model_dir_path / 'yolov8n_toys.pt'
try:
    logger.info(f"Loading YOLO model for toy detection ('{str(toy_model_path_obj)}')...")
    TOY_DETECTION_MODEL = YOLO(str(toy_model_path_obj)) # YOLO handles its own device placement
    logger.info(f"YOLO toy detection model ('{str(toy_model_path_obj)}') loaded successfully.")
except Exception as e:
    logger.warning(f"Failed to load YOLO toy detection model ('{str(toy_model_path_obj)}'): {e}. Toy detection will be disabled. Ensure 'model_dir' in config.json is correct or the model is in the default location.")

# 3. Super Resolution Model
logger.info("Initializing SuperResolutionModel...")
class SuperResolutionModelDef(nn.Module): # Renamed to avoid conflict if SuperResolutionModel is imported
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

SUPER_RESOLUTION_MODEL = SuperResolutionModelDef(scale_factor=2)
SUPER_RESOLUTION_MODEL.eval()
logger.info("SuperResolutionModel initialized and set to evaluation mode.")

# --- Device Management for All Models ---
_target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"--- Moving models to target device: {_target_device} ---")

if CROW_EMBEDDING_MODEL:
    CROW_EMBEDDING_MODEL = CROW_EMBEDDING_MODEL.to(_target_device)
    logger.info(f"CROW_EMBEDDING_MODEL moved to {_target_device}.")
    # logger.info(f"CROW_EMBEDDING_MODEL is on device: {next(CROW_EMBEDDING_MODEL.parameters()).device}")


if TOY_DETECTION_MODEL:
    try:
        TOY_DETECTION_MODEL.to(_target_device) # YOLO's method to move model
        # YOLOv8 models store device internally, e.g. TOY_DETECTION_MODEL.device or TOY_DETECTION_MODEL.model.device
        logger.info(f"TOY_DETECTION_MODEL attempted to move to {_target_device}.")
    except Exception as e:
        logger.warning(f"Could not move TOY_DETECTION_MODEL to {_target_device}: {e}")


if SUPER_RESOLUTION_MODEL:
    SUPER_RESOLUTION_MODEL = SUPER_RESOLUTION_MODEL.to(_target_device)
    logger.info(f"SUPER_RESOLUTION_MODEL moved to {_target_device}.")
    # logger.info(f"SUPER_RESOLUTION_MODEL is on device: {next(SUPER_RESOLUTION_MODEL.parameters()).device}")

logger.info("--- Model loading and device management complete. ---")

# Create a global model reference for backward compatibility with tests
model = CROW_EMBEDDING_MODEL


def apply_super_resolution(img_tensor, min_size=100):
    """Apply super-resolution to small images."""
    h, w = img_tensor.shape[-2:]
    if h >= min_size and w >= min_size:
        return img_tensor
    
    if SUPER_RESOLUTION_MODEL is None:
        logger.warning("SuperResolutionModel is not available. Skipping super-resolution.")
        return img_tensor

    with torch.no_grad():
        device = next(SUPER_RESOLUTION_MODEL.parameters()).device
        img_tensor_device = img_tensor.to(device)
        enhanced = SUPER_RESOLUTION_MODEL(img_tensor_device)
        enhanced = torch.clamp(enhanced, 0, 1)
        return enhanced.cpu() # Original behavior: return to CPU

def compute_embedding(img_tensors):
    """Compute feature embeddings for both head and body regions using CROW_EMBEDDING_MODEL."""
    if CROW_EMBEDDING_MODEL is None:
        raise ModelError("CROW_EMBEDDING_MODEL is not available for computing embeddings.")
    try:
        embeddings = {}
        with torch.no_grad():
            model_device = next(CROW_EMBEDDING_MODEL.parameters()).device
            for key, tensor in img_tensors.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Expected torch.Tensor for {key}, got {type(tensor)}")
                if tensor.dim() != 4:  # Add batch dimension if missing
                    tensor = tensor.unsqueeze(0)
                if tensor.size(1) != 3:  # Ensure RGB channels
                    raise ValueError(f"Expected 3 channels for {key}, got {tensor.size(1)}")
                
                # Normalize tensor (assuming input tensor is [0,1] range, float)
                # Standard ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
                normalized_tensor = (tensor.to(model_device) - mean) / std
                
                img_tensors[key] = normalized_tensor # Store normalized tensor (already on device)
            
            for key, tensor_on_device in img_tensors.items():
                try:
                    features = CROW_EMBEDDING_MODEL(tensor_on_device)
                    features = F.normalize(features, p=2, dim=1) # L2 normalize
                    embeddings[key] = features.squeeze().cpu().numpy() # Return to CPU as per original
                except Exception as e:
                    raise EmbeddingError(f"Error computing embedding for {key} with CROW_EMBEDDING_MODEL: {str(e)}")
        
        if 'full' not in embeddings or 'head' not in embeddings:
             raise EmbeddingError("Full and head embeddings must be computed.")

        # Combine embeddings with weighted average
        combined = np.concatenate([
            0.7 * embeddings['full'],
            0.3 * embeddings['head']
        ])
        
        combined = combined / np.linalg.norm(combined) # Normalize combined embedding
        return combined, embeddings
        
    except Exception as e:
        if isinstance(e, EmbeddingError):
            raise
        raise EmbeddingError(f"Error in compute_embedding: {str(e)}")

def extract_normalized_crow_crop(frame, bbox, expected_size=(512, 512), correct_orientation=True, padding=0.2):
    """Extract and normalize a crop of a crow from a frame. Output is float32 [0,1] HWC."""
    try:
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
        if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
            raise ValueError("Bbox must be a list/tuple/array of 4 values [x1, y1, x2, y2]")
            
        x1, y1, x2, y2 = map(int, bbox)
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"Invalid bbox coordinates: {bbox} (x1 >= x2 or y1 >= y2)")
        
        # Add padding to ensure entire crow is captured and centered
        h, w = frame.shape[:2]
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Calculate padding
        pad_w = int(bbox_w * padding)
        pad_h = int(bbox_h * padding)
        
        # Apply padding and clamp to frame boundaries
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(w, x2 + pad_w)
        y2_padded = min(h, y2 + pad_h)
        
        # Ensure we still have a valid box after padding
        if x1_padded >= x2_padded or y1_padded >= y2_padded:
            # Fallback to original bbox if padding causes issues
            logger.warning(f"Padding resulted in invalid bbox, using original: {bbox}")
            x1_padded, y1_padded, x2_padded, y2_padded = x1, y1, x2, y2
            
        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        if crop.size == 0:
            raise ValueError(f"Crop resulted in empty image for bbox {bbox} with padding")

        if correct_orientation:
            try:
                from crow_orientation import correct_crow_crop_orientation
                crop = correct_crow_crop_orientation(crop)
            except ImportError:
                logger.warning("Crow orientation module not available, skipping orientation correction.")
            except Exception as e:
                logger.warning(f"Error applying orientation correction: {e}. Using original crop.")
        
        # Apply super-resolution if crop is small (before final resize)
        crop_h, crop_w = crop.shape[:2]
        if crop_h < 100 or crop_w < 100:
            try:
                logger.debug(f"Applying super-resolution to small crop: {crop_w}x{crop_h}")
                # Convert to tensor format for super-resolution
                crop_tensor = torch.from_numpy(crop.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> BCHW
                enhanced_tensor = apply_super_resolution(crop_tensor, min_size=100)
                # Convert back to numpy HWC format
                crop = (enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                logger.debug(f"Super-resolution enhanced crop to: {crop.shape[1]}x{crop.shape[0]}")
            except Exception as e:
                logger.warning(f"Super-resolution failed, using original crop: {e}")
        
        crop_resized = cv2.resize(crop, (expected_size[1], expected_size[0]), interpolation=cv2.INTER_LANCZOS4)
        crop_normalized = crop_resized.astype(np.float32) / 255.0 # HWC, [0,1]
        
        head_height = expected_size[0] // 3
        head_crop_region = crop_normalized[:head_height, :, :] # Slice from already normalized full crop
        head_crop_resized = cv2.resize(head_crop_region, (expected_size[1], expected_size[0]), interpolation=cv2.INTER_LANCZOS4)
        
        return {'full': crop_normalized, 'head': head_crop_resized}
        
    except Exception as e:
        logger.error(f"Error extracting normalized crow crop for bbox {bbox}: {str(e)}")
        return None

class EnhancedTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, 
                 embedding_threshold=0.7, conf_threshold=0.5, 
                 multi_view_stride=5, strict_mode=False):
        self.logger = logging.getLogger(f"{__name__}.EnhancedTracker")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Tracker instance device preference
        self.logger.info(f"EnhancedTracker initialized. Instance preferred device: {self.device}")
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.conf_threshold = conf_threshold
        self.multi_view_stride = multi_view_stride
        self.strict_mode = strict_mode
        
        # self.model (for embeddings) is no longer initialized here; global CROW_EMBEDDING_MODEL will be used.
        self.multi_view_extractor = None
        self.color_normalizer = None
        self._initialize_models() # Call without model_path and strict_mode
        
        self.track_history = {}
        self.track_embeddings = {}
        self.track_head_embeddings = {}
        self.track_ages = {}
        self.track_confidences = {}
        self.track_bboxes = {}
        self.track_temporal_consistency = {}
        
        self.processing_dir = Path("processing")
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_count = 0
        self.track_id_changes = {}
        self.last_cleanup_frame = 0
        self.cleanup_interval = 30 
        self.max_track_history = 10
        self.max_behavior_history = 50
        self.behavior_window = 10
        self.max_movement = 100.0
        self.next_id = 0
        self.expected_size = (100, 100)
        self.max_retries = 3
        self.retry_delay = 1.0
        self.tracking_file = self.processing_dir / "tracking_data.json"
        self.tracking_data = {
            "metadata": {"last_id": 0, "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()},
            "crows": {}
        }

        self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_threshold)
        self.extract_normalized_crow_crop = extract_normalized_crow_crop # Use module-level function

    def _initialize_models(self):
        """Initializes auxiliary models for the tracker. Embedding model is global."""
        # Embedding model (self.model) is no longer handled here. Global CROW_EMBEDDING_MODEL is used directly.
        self.logger.info("EnhancedTracker will use the global CROW_EMBEDDING_MODEL for embeddings.")
        if CROW_EMBEDDING_MODEL is None:
            error_msg = "Global CROW_EMBEDDING_MODEL is not available! Embedding functionality will be impaired."
            self.logger.error(error_msg)
            if self.strict_mode:
                raise ModelError(f"Model initialization failed: {error_msg}")
        # No need to move CROW_EMBEDDING_MODEL to self.device, as it's already on _target_device.
        # Calculations using it should ensure tensors are moved to CROW_EMBEDDING_MODEL's device.

        self.logger.info("Initializing multi-view extractor for EnhancedTracker...")
        try:
            self.multi_view_extractor = create_multi_view_extractor() # This function should handle its own model loading/config
            if self.multi_view_extractor is None:
                error_msg = "Multi-view extractor could not be created."
                self.logger.warning(error_msg)
                if self.strict_mode:
                    raise ModelError(f"Model initialization failed: {error_msg}")
            elif hasattr(self.multi_view_extractor, 'to'):
                self.multi_view_extractor.to(self.device)
                if hasattr(self.multi_view_extractor, 'eval'): self.multi_view_extractor.eval()
                self.logger.info("Multi-view extractor configured for EnhancedTracker.")
        except Exception as e:
            error_msg = f"Failed to initialize multi-view extractor: {e}"
            self.logger.error(error_msg)
            if self.strict_mode:
                raise ModelError(f"Model initialization failed: {error_msg}")
            
        self.logger.info("Initializing color normalizer for EnhancedTracker...")
        try:
            self.color_normalizer = create_normalizer() # This might return None or a configured object
            if self.color_normalizer and hasattr(self.color_normalizer, 'to'): # If it's a PyTorch module
                self.color_normalizer.to(self.device)
                self.logger.info("Color normalizer (if PyTorch module) moved to device.")
            elif self.color_normalizer:
                self.logger.info("Color normalizer initialized (non-PyTorch module or no 'to' method).")
            else:
                error_msg = "Color normalizer could not be initialized."
                self.logger.warning(error_msg)
                if self.strict_mode:
                    raise ModelError(f"Model initialization failed: {error_msg}")
        except Exception as e:
            error_msg = f"Failed to initialize color normalizer: {e}"
            self.logger.error(error_msg)
            if self.strict_mode:
                raise ModelError(f"Model initialization failed: {error_msg}")

    # _load_model_from_path is removed as EnhancedTracker no longer loads its own embedding model.

    def _process_detection_batch(self, frame, detections):
        """Process a batch of detections to compute embeddings using the global CROW_EMBEDDING_MODEL."""
        if CROW_EMBEDDING_MODEL is None:
            self.logger.warning("Global CROW_EMBEDDING_MODEL not available in _process_detection_batch. Returning zero embeddings.")
            # Return zero embeddings with the correct structure and embedding dimension
            # Assuming embedding_dim is 512, as per CROW_EMBEDDING_MODEL's typical initialization
            zero_emb_list = [np.zeros(512, dtype=np.float32) for _ in range(len(detections))]
            return {'full': zero_emb_list, 'head': zero_emb_list}
        
        try:
            if len(detections) == 0: return {'full': [], 'head': []}
            
            full_crops_np, head_crops_np = [], []
            for det in detections: # detections are expected to be [x1,y1,x2,y2, possibly score]
                bbox = det[:4]
                crops = extract_normalized_crow_crop(frame, bbox, correct_orientation=True, padding=0.3) 
                if crops:
                    full_crops_np.append(crops['full'])
                    head_crops_np.append(crops['head'])
                else:
                    # Use a consistent size for zero arrays, e.g., 512x512x3
                    full_crops_np.append(np.zeros((512, 512, 3), dtype=np.float32))
                    head_crops_np.append(np.zeros((512, 512, 3), dtype=np.float32))

            full_tensor = torch.stack([torch.from_numpy(crop).permute(2,0,1) for crop in full_crops_np]).float()
            head_tensor = torch.stack([torch.from_numpy(crop).permute(2,0,1) for crop in head_crops_np]).float()

            # Use device of the global CROW_EMBEDDING_MODEL
            model_device = next(CROW_EMBEDDING_MODEL.parameters()).device
            full_tensor = full_tensor.to(model_device)
            head_tensor = head_tensor.to(model_device)

            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            full_tensor = (full_tensor - mean) / std
            head_tensor = (head_tensor - mean) / std
            
            with torch.no_grad():
                full_embeddings = CROW_EMBEDDING_MODEL(full_tensor)
                head_embeddings = CROW_EMBEDDING_MODEL(head_tensor)
                
            full_embeddings = F.normalize(full_embeddings, p=2, dim=1).cpu().numpy()
            head_embeddings = F.normalize(head_embeddings, p=2, dim=1).cpu().numpy()
            
            return {'full': list(full_embeddings), 'head': list(head_embeddings)}
            
        except Exception as e:
            self.logger.error(f"Error in EnhancedTracker._process_detection_batch using CROW_EMBEDDING_MODEL: {str(e)}", exc_info=True)
            zero_emb = np.zeros(512, dtype=np.float32) # Fallback to zero embeddings
            return {'full': [zero_emb.copy() for _ in range(len(detections))], 
                    'head': [zero_emb.copy() for _ in range(len(detections))]}

    def update(self, frame, detections_input):
        """Update tracking with new detections."""
        try:
            detections = self._normalize_detections(detections_input)
            self.frame_count += 1
            
            if len(detections) > 0:
                detections = detections[detections[:, 4] >= self.conf_threshold]
            
            if len(detections) == 0:
                self._cleanup_old_tracks_by_age() # Renamed for clarity
                return np.empty((0, 5))
                
            tracks_from_sort = self.tracker.update(detections) # SORT returns [x1,y1,x2,y2,track_id]
            
            active_tracks_output = []
            batch_embeddings = None # Initialize batch_embeddings

            if len(tracks_from_sort) > 0:
                # Check if global CROW_EMBEDDING_MODEL is available before processing for embeddings
                if CROW_EMBEDDING_MODEL:
                    # Process all current tracks in a batch for embeddings
                    # _process_detection_batch expects detections in a format like [x1,y1,x2,y2,score]
                    # tracks_from_sort is [x1,y1,x2,y2,track_id]
                    # We might need to append a dummy score if _process_detection_batch strictly needs it,
                    # or adjust _process_detection_batch. For now, assume it can handle it or adapt.
                    # Let's pass tracks_from_sort directly, as _process_detection_batch extracts bbox via det[:4]
                    batch_embeddings = self._process_detection_batch(frame, tracks_from_sort)
                else:
                    self.logger.warning("CROW_EMBEDDING_MODEL not available during update. Embeddings will not be computed.")

            for i, track_data in enumerate(tracks_from_sort):
                track_id = int(track_data[4])
                bbox = track_data[:4]
                # Score might be part of track_data if SORT provides it, or from original detection
                # For simplicity, using a default or deriving if needed. Here, assume track_data[4] is ID.
                # Original detections had score at detections[:, 4]
                # We need to map original detection scores to tracks if SORT doesn't preserve/return them.
                # This simplified version doesn't explicitly map scores back post-SORT for embeddings.
                
                score = 1.0 # Placeholder, ideally map from original detection or use SORT's confidence
                # Find original detection for score (approximate by IoU or center distance if necessary)
                # For now, we'll use a default score for history.
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = {'history': deque(maxlen=self.max_track_history), 
                                                    'behaviors': deque(maxlen=self.max_behavior_history),
                                                    'last_valid_embedding_tensor': None, # Store tensor for similarity
                                                    'temporal_consistency': 1.0}
                    self.track_embeddings[track_id] = deque(maxlen=4) # Store numpy embeddings for DB
                    self.track_head_embeddings[track_id] = deque(maxlen=4)
                    self.track_ages[track_id] = 0 # Frames since first detection
                    self.track_confidences[track_id] = score 
                    self.track_bboxes[track_id] = bbox
                    self.track_temporal_consistency[track_id] = 1.0

                self.track_ages[track_id] += 1
                self.track_confidences[track_id] = score # Update with current confidence
                self.track_bboxes[track_id] = bbox

                embedding_available_for_track = False
                if batch_embeddings and batch_embeddings['full'] and i < len(batch_embeddings['full']):
                    full_np_embedding = batch_embeddings['full'][i]
                    head_np_embedding = batch_embeddings['head'][i]

                    if full_np_embedding is not None: # Should be a numpy array now
                        self.track_embeddings[track_id].append(full_np_embedding)
                        # For internal similarity, ensure it's a tensor on the instance's device.
                        # Note: CROW_EMBEDDING_MODEL's device might be different from self.device.
                        # For consistency, similarity checks should probably also use CROW_EMBEDDING_MODEL
                        # or ensure tensors are on a common device. For now, use self.device.
                        self.track_history[track_id]['last_valid_embedding_tensor'] = torch.from_numpy(full_np_embedding).to(self.device)
                        embedding_available_for_track = True
                    if head_np_embedding is not None:
                        self.track_head_embeddings[track_id].append(head_np_embedding)
                
                history_entry = {
                    'bbox': bbox.tolist(), 'frame_idx': self.frame_count, 'confidence': score,
                    'age': self.track_ages[track_id],
                    'embedding_factor': 1.0 if len(self.track_embeddings.get(track_id, [])) > 0 else 0.0,
                    'behavior_score': self._calculate_behavior_score(track_id),
                    'movement_score': self._calculate_movement_score(track_id, frame), # Pass frame
                    'size_score': self._calculate_size_score(bbox),
                    'temporal_consistency': self.track_temporal_consistency.get(track_id, 1.0)
                }
                self.track_history[track_id]['history'].append(history_entry)
                self._update_temporal_consistency(track_id) # Update based on new history
                active_tracks_output.append(track_data)
                
            self._cleanup_old_tracks_by_age()
            return np.array(active_tracks_output) if active_tracks_output else np.empty((0, 5))
            
        except Exception as e:
            self.logger.error(f"Error in EnhancedTracker.update: {str(e)}", exc_info=True)
            # Depending on desired robustness, either raise or return empty
            # self.strict_mode was removed, so we make a choice here (e.g. always raise for debugging, or always return empty for production)
            # For now, let's re-raise to make issues visible during development
            raise 
            # return np.empty((0, 5)) 

    def _cleanup_old_tracks_by_age(self):
        """Clean up old tracks that haven't been updated recently based on frame_count and max_age."""
        current_frame_count = self.frame_count # Using internal frame count
        tracks_to_remove = [
            tid for tid, last_seen_frame in self.track_ages.items() 
            if current_frame_count - last_seen_frame > self.max_age
        ]
        for track_id in tracks_to_remove:
            self._cleanup_track_data(track_id) # New method to actually delete data

    def _cleanup_track_data(self, track_id):
        """Helper to remove all data associated with a track_id."""
        self.logger.debug(f"Cleaning up data for old track ID: {track_id}")
        self.track_history.pop(track_id, None)
        self.track_embeddings.pop(track_id, None)
        self.track_head_embeddings.pop(track_id, None)
        self.track_ages.pop(track_id, None)
        self.track_confidences.pop(track_id, None)
        self.track_bboxes.pop(track_id, None)
        self.track_temporal_consistency.pop(track_id, None)
        # Potentially other cleanup like removing from self.tracker.tracks if necessary and safe

    def _normalize_detections(self, detections_input):
        """Normalizes various detection formats to a standard np.array [[x1,y1,x2,y2,score], ...]."""
        if detections_input is None: return np.empty((0, 5))
        if isinstance(detections_input, list):
            if not detections_input: return np.empty((0, 5))
            # Assuming list of dicts [{'bbox': [x,y,x,y], 'score': s}]
            if isinstance(detections_input[0], dict):
                processed_dets = []
                for d in detections_input:
                    bbox = d.get('bbox')
                    score = d.get('score', 1.0) # Default score if not present
                    if bbox and len(bbox) == 4:
                        processed_dets.append(list(bbox) + [score])
                    else:
                        self.logger.warning(f"Skipping invalid detection dict: {d}")
                return np.array(processed_dets).astype(float) if processed_dets else np.empty((0,5))
            else: # Assuming list of lists/tuples
                return np.array(detections_input).astype(float)
        elif isinstance(detections_input, np.ndarray):
            return detections_input.astype(float)
        
        self.logger.warning(f"Unknown detection format type: {type(detections_input)}. Returning empty.")
        return np.empty((0, 5))

    def _calculate_behavior_score(self, track_id): return 0.5 # Placeholder
    def _calculate_movement_score(self, track_id, frame): return 0.0 # Placeholder
    def _calculate_size_score(self, bbox): return 1.0 # Placeholder
    def _update_temporal_consistency(self, track_id): pass # Placeholder

    # Other methods like _generate_crow_id, get_crow_info, list_crows, _save_tracking_data, etc.
    # would remain largely the same but should use self.logger.

    def _generate_crow_id(self):
        self.next_id += 1
        crow_id = f"crow_{self.next_id:04d}"
        self.tracking_data["metadata"]["last_id"] = self.next_id
        return crow_id

    def get_crow_info(self, crow_id):
        return self.tracking_data["crows"].get(crow_id)

    def list_crows(self):
        return list(self.tracking_data["crows"].keys())

    def _save_tracking_data(self, force=False): # Add force param for compatibility if needed
        try:
            tracking_file_path = Path(self.tracking_file)
            tracking_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tracking_file_path, "w") as f:
                json.dump(self.tracking_data, f, indent=2)
            self.logger.info(f"Tracking data saved to {tracking_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving tracking data: {str(e)}")
            # Optionally re-raise if this is critical path
            # raise

    def create_processing_run(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.processing_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created processing run directory: {run_dir}")
        return run_dir
        
    def process_detection(self, frame, frame_num, detection_input, video_path, frame_time):
        """Processes a single detection, updates tracking, and assigns/updates a crow ID."""
        try:
            # Normalize the single detection to the format expected by self.update
            # self.update expects a batch, so wrap it
            normalized_detection = self._normalize_detections([detection_input] if not isinstance(detection_input, list) else detection_input)
            
            if normalized_detection.shape[0] == 0:
                self.logger.warning("process_detection received an empty or invalid detection.")
                return None

            # Update tracker with this single detection (as a batch of one)
            active_tracks = self.update(frame, normalized_detection)

            if not active_tracks.any():
                return None # No track confirmed or updated for this detection

            # Assuming the first (and likely only) track corresponds to the input detection
            track_data = active_tracks[0]
            track_id = int(track_data[4]) # SORT track ID
            bbox = track_data[:4].tolist()
            score = normalized_detection[0, 4] # Score from the input detection

            # Check if this SORT track_id has an associated crow_id in the tracker's history
            current_crow_id = self.track_history[track_id].get('crow_id')

            if current_crow_id is None: # New crow for this SORT track
                current_crow_id = self._generate_crow_id()
                self.track_history[track_id]['crow_id'] = current_crow_id
                self.track_history[track_id]['video_path'] = video_path # Store first video path
                self.track_history[track_id]['frame_time'] = frame_time # Store first frame time
                
                self.tracking_data["crows"][current_crow_id] = {
                    "internal_sort_id": track_id, # Link to SORT ID for this session
                    "first_seen_timestamp": frame_time,
                    "last_seen_timestamp": frame_time,
                    "first_video_path": video_path,
                    "last_video_path": video_path,
                    "total_detections": 1,
                    "detection_history": [{
                        "frame_num": frame_num, "bbox": bbox, 
                        "score": score, "timestamp": frame_time, "video_path": video_path
                    }]
                }
                self.logger.info(f"New crow {current_crow_id} (SORT ID {track_id}) detected.")
            else: # Existing crow for this SORT track
                crow_record = self.tracking_data["crows"][current_crow_id]
                crow_record["last_seen_timestamp"] = frame_time
                crow_record["last_video_path"] = video_path
                crow_record["total_detections"] += 1
                crow_record["detection_history"].append({
                    "frame_num": frame_num, "bbox": bbox, 
                    "score": score, "timestamp": frame_time, "video_path": video_path
                })
                # Trim history if it gets too long
                if len(crow_record["detection_history"]) > 100: # Example limit
                     crow_record["detection_history"] = crow_record["detection_history"][-100:]
                self.logger.debug(f"Updated crow {current_crow_id} (SORT ID {track_id}).")

            self._save_tracking_data() # Save changes
            return current_crow_id

        except Exception as e:
            self.logger.error(f"Error in process_detection: {str(e)}", exc_info=True)
            return None


def compute_bbox_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    # Ensure numpy arrays for vectorized operations
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)

    if box1.size == 0 or box2.size == 0: return 0.0
    
    x1, y1, x2, y2 = box1[:4]
    xx1, yy1, xx2, yy2 = box2[:4]
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    if area1 <= 0 or area2 <= 0: return 0.0
    
    inter_x1, inter_y1 = max(x1, xx1), max(y1, yy1)
    inter_x2, inter_y2 = min(x2, xx2), min(y2, yy2)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2: return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = area1 + area2 - inter_area
    if union_area == 0: return 0.0 # Avoid division by zero
    
    return inter_area / union_area


def create_model():
    """Creates and returns a new instance of the primary embedding model (CrowResNetEmbedder)."""
    logger.info("Global create_model() called. Creating a new instance of CrowResNetEmbedder.")
    # This function should be consistent with the main CROW_EMBEDDING_MODEL initialization
    new_model_instance = CrowResNetEmbedder(embedding_dim=512)
    new_model_instance.eval() # Set to eval mode by default
    
    # Move to the target device (_target_device is defined at the top level)
    new_model_instance = new_model_instance.to(_target_device)
    logger.info(f"New model instance (CrowResNetEmbedder) created by create_model() is on device: {next(new_model_instance.parameters()).device if list(new_model_instance.parameters()) else 'CPU (no params)'}")
    return new_model_instance


def assign_crow_ids(frames, detections_list, video_path=None, max_age=5, min_hits=2, iou_threshold=0.2, embedding_threshold=0.7, return_track_history=False, multi_view_stride=1):
    logger.info("Starting crow tracking and ID assignment process (assign_crow_ids)...")
    labeled_frames_output = []
    # Initialize EnhancedTracker. It will use global CROW_EMBEDDING_MODEL by default.
    tracker_instance = EnhancedTracker(
        max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold,
        embedding_threshold=embedding_threshold, multi_view_stride=multi_view_stride
    )
    
    # This part seems more related to DB interaction, might be simplified if tracker handles IDs internally
    # known_crows_from_db = get_all_crows() 
    # resolved_crow_ids_map = {crow['id']: crow['id'] for crow in known_crows_from_db}
    
    frame_track_data_history = []

    for frame_idx, (frame_image, detections_for_frame) in enumerate(tqdm(zip(frames, detections_list), total=len(frames), desc="Processing frames")):
        t_start = time.time()
        
        # EnhancedTracker.update expects detections typically as list of dicts or numpy array
        # Example: [{'bbox': [x1,y1,x2,y2], 'score': conf}, ...] or np.array([[x1,y1,x2,y2,score],...])
        # Ensure detections_for_frame matches this.
        # The _normalize_detections method in EnhancedTracker should handle various formats.
        current_tracks = tracker_instance.update(frame_image, detections_for_frame)
        
        t_end = time.time()
        logger.debug(f"Frame {frame_idx+1}/{len(frames)}: tracking update took {t_end-t_start:.3f}s. Found {len(current_tracks)} tracks.")
        
        output_frame_copy = frame_image.copy()
        current_frame_tracks_info = {}

        for track_info in current_tracks: # track_info is [x1, y1, x2, y2, track_id] from SORT
            x1, y1, x2, y2, track_id_int = map(int, track_info[:5])
            
            # Retrieve the persistent crow_id associated with this SORT track_id by the tracker
            # This requires EnhancedTracker to manage this association.
            # For now, let's assume tracker.track_history[track_id_int]['crow_id'] holds it
            # Or, we use a simpler approach for this example:
            
            assigned_crow_id_for_track = tracker_instance.track_history.get(track_id_int, {}).get('crow_id')
            if not assigned_crow_id_for_track: # If tracker hasn't assigned one yet (e.g. new track)
                 # This part is tricky: assign_crow_ids might need to call process_detection
                 # or replicate its ID generation logic if it's meant to be self-contained for ID logic.
                 # For simplicity, let's assume the tracker's internal ID logic is primary.
                 # If EnhancedTracker.update doesn't create/assign persistent IDs, this needs more.
                 # A call to process_detection for each track would be one way:
                 # temp_detection_obj = {'bbox': [x1,y1,x2,y2], 'score': 1.0} # Need score
                 # assigned_crow_id_for_track = tracker_instance.process_detection(frame_image, frame_idx, temp_detection_obj, video_path, time.time())
                 pass # If no ID, just draw box with SORT ID.

            label_text = f"SORT ID: {track_id_int}"
            if assigned_crow_id_for_track:
                label_text = f"Crow {assigned_crow_id_for_track} (S:{track_id_int})"

            cv2.rectangle(output_frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame_copy, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_frame_tracks_info[track_id_int] = {'bbox': [x1,y1,x2,y2], 'assigned_crow_id': assigned_crow_id_for_track}
            
            # Example of using track_embeddings (if available and populated by EnhancedTracker.update)
            # This part (DB saving) was in the original, keep if EnhancedTracker doesn't do it.
            # Check if CROW_EMBEDDING_MODEL is available if embeddings are relevant here.
            # if CROW_EMBEDDING_MODEL and track_id_int in tracker_instance.track_embeddings and tracker_instance.track_embeddings[track_id_int]:
            #     latest_embedding_np = tracker_instance.track_embeddings[track_id_int][-1] # Get latest numpy embedding
            #     # db_crow_id = save_crow_embedding(embedding_np, ...) # etc.
            #     # This suggests assign_crow_ids might be more about DB interaction than pure tracking.

        labeled_frames_output.append(output_frame_copy)
        frame_track_data_history.append(current_frame_tracks_info)

        # --- CUDA Cache Clearing ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug(f"Frame {frame_idx} (assign_crow_ids): Cleared CUDA cache.")
        
    logger.info("assign_crow_ids processing complete.")
    if return_track_history:
        return labeled_frames_output, frame_track_data_history
    return labeled_frames_output


def load_faster_rcnn():
    """Load and configure the Faster R-CNN model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading Faster R-CNN model on device: {device}")
        
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        faster_rcnn_model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        faster_rcnn_model = faster_rcnn_model.to(device)
        faster_rcnn_model.eval()
        
        # Example configurations (adjust as needed)
        faster_rcnn_model.roi_heads.score_thresh = 0.5 
        faster_rcnn_model.roi_heads.nms_thresh = 0.3
        
        logger.info(f"Faster R-CNN model loaded successfully on {device}.")
        return faster_rcnn_model
    except Exception as e:
        logger.error(f"Error loading Faster R-CNN model: {str(e)}", exc_info=True)
        raise # Or return None if preferred

def load_triplet_model():
    """Returns the globally configured CROW_EMBEDDING_MODEL.
    Ensures the 'get_embedding' method is available on it.
    """
    logger.info("load_triplet_model() called.")
    if CROW_EMBEDDING_MODEL is None:
        logger.error("Global CROW_EMBEDDING_MODEL is not initialized!")
        raise ModelError("CROW_EMBEDDING_MODEL is None, cannot be returned by load_triplet_model.")

    # Ensure 'get_embedding' method is attached (idempotent addition)
    if not hasattr(CROW_EMBEDDING_MODEL, 'get_embedding'):
        logger.info("Dynamically adding 'get_embedding' method to CROW_EMBEDDING_MODEL instance.")
        
        def get_embedding_method(self_model_instance, img_tensor_input):
            model_device = next(self_model_instance.parameters()).device
            
            if isinstance(img_tensor_input, dict): # Handle dict for full and head
                full_in = img_tensor_input['full'].to(model_device)
                head_in = img_tensor_input['head'].to(model_device)
                if full_in.dim() == 3: full_in = full_in.unsqueeze(0)
                if head_in.dim() == 3: head_in = head_in.unsqueeze(0)

                # Normalize before feeding to model
                mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
                full_in = (full_in - mean) / std
                head_in = (head_in - mean) / std
                
                with torch.no_grad():
                    full_emb = self_model_instance(full_in)
                    head_emb = self_model_instance(head_in)
                full_emb_norm = F.normalize(full_emb, p=2, dim=1)
                head_emb_norm = F.normalize(head_emb, p=2, dim=1)
                combined = 0.7 * full_emb_norm + 0.3 * head_emb_norm
                return F.normalize(combined, p=2, dim=1)
            else: # Handle single tensor input
                single_in = img_tensor_input.to(model_device)
                if single_in.dim() == 3: single_in = single_in.unsqueeze(0)
                
                mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
                single_in = (single_in - mean) / std

                with torch.no_grad():
                    return F.normalize(self_model_instance(single_in), p=2, dim=1)

        # Bind method to the instance of CROW_EMBEDDING_MODEL
        CROW_EMBEDDING_MODEL.get_embedding = types.MethodType(get_embedding_method, CROW_EMBEDDING_MODEL)
        logger.info("'get_embedding' method added to CROW_EMBEDDING_MODEL.")
    
    logger.info(f"Returning global CROW_EMBEDDING_MODEL (type: {type(CROW_EMBEDDING_MODEL).__name__}).")
    return CROW_EMBEDDING_MODEL

# Export for test visibility and programmatic access
__all__ = [
    'TrackingError', 'ModelError', 'DeviceError', 'EmbeddingError', 'TimeoutException',
    'EnhancedTracker', 
    'extract_normalized_crow_crop', 
    'assign_crow_ids', 
    'compute_bbox_iou', 
    'load_faster_rcnn', 
    'load_triplet_model', 'create_model', # create_model is now consistent
    'apply_super_resolution', 'compute_embedding', # Core functions
    # Global model variables
    'CROW_EMBEDDING_MODEL', 
    'TOY_DETECTION_MODEL', 
    'SUPER_RESOLUTION_MODEL'
]

# Quick test for get_embedding if CROW_EMBEDDING_MODEL is available
if __name__ == '__main__':
    logger.info("Running a quick test for CROW_EMBEDDING_MODEL.get_embedding...")
    if CROW_EMBEDDING_MODEL and hasattr(CROW_EMBEDDING_MODEL, 'get_embedding'):
        try:
            # Create dummy tensors (ensure they are on the correct device, matching model)
            dummy_device = next(CROW_EMBEDDING_MODEL.parameters()).device
            dummy_full_tensor = torch.rand(1, 3, 512, 512).to(dummy_device) # B,C,H,W
            dummy_head_tensor = torch.rand(1, 3, 512, 512).to(dummy_device)
            
            # Test with dict input
            embedding_dict_out = CROW_EMBEDDING_MODEL.get_embedding({'full': dummy_full_tensor, 'head': dummy_head_tensor})
            logger.info(f"get_embedding with dict output shape: {embedding_dict_out.shape}")

            # Test with single tensor input
            embedding_single_out = CROW_EMBEDDING_MODEL.get_embedding(dummy_full_tensor)
            logger.info(f"get_embedding with single tensor output shape: {embedding_single_out.shape}")
            logger.info("get_embedding method seems to work.")
        except Exception as e:
            logger.error(f"Error during get_embedding test: {e}", exc_info=True)
    elif CROW_EMBEDDING_MODEL:
        logger.warning("CROW_EMBEDDING_MODEL is loaded, but get_embedding method is not attached.")
    else:
        logger.warning("CROW_EMBEDDING_MODEL not loaded, skipping get_embedding test.")

    # Test for model device placement
    if CROW_EMBEDDING_MODEL:
        logger.info(f"CROW_EMBEDDING_MODEL is on: {next(CROW_EMBEDDING_MODEL.parameters()).device if list(CROW_EMBEDDING_MODEL.parameters()) else 'No params/Unknown'}")
    if TOY_DETECTION_MODEL:
         # For YOLO, device is often an attribute of the model or its sub-modules
        yolo_device = 'Unknown'
        if hasattr(TOY_DETECTION_MODEL, 'device'): yolo_device = TOY_DETECTION_MODEL.device
        elif hasattr(TOY_DETECTION_MODEL, 'model') and hasattr(TOY_DETECTION_MODEL.model, 'device'): yolo_device = TOY_DETECTION_MODEL.model.device
        logger.info(f"TOY_DETECTION_MODEL is on: {yolo_device}")
    if SUPER_RESOLUTION_MODEL:
        logger.info(f"SUPER_RESOLUTION_MODEL is on: {next(SUPER_RESOLUTION_MODEL.parameters()).device if list(SUPER_RESOLUTION_MODEL.parameters()) else 'No params/Unknown'}")
