import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout(seconds):
    """Windows-compatible timeout decorator using threading."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = []
            error = []
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                thread._stop()
                raise TimeoutException(f"Function timed out after {seconds} seconds")
            if error:
                raise error[0]
            return result[0] if result else None
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

def extract_crow_image(frame, box, padding=0.3, min_size=100):
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        
        # Validate box coordinates
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            logger.warning(f"Invalid box coordinates: {box} for frame size {w}x{h}")
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
                logger.warning(f"Invalid crow crop extracted for box {box}")
                return None
                
            head_height = int((y2 - y1) * 0.33)
            if head_height <= 0:
                logger.warning(f"Invalid head height {head_height} for box {box}")
                return None
                
            head_img = frame[y1:y1 + head_height, x1:x2].copy()  # Make a copy to ensure contiguous array
            if not isinstance(head_img, np.ndarray) or head_img.size == 0 or head_img.shape[0] == 0 or head_img.shape[1] == 0:
                logger.warning(f"Invalid head crop extracted for box {box}")
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
            logger.warning(f"Image enhancement failed for box {box}")
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
            logger.warning(f"Image resizing failed for box {box}")
            return None
            
        # Convert to tensors
        try:
            crow_tensor = torch.from_numpy(crow_square).permute(2, 0, 1).float() / 255.0
            head_tensor = torch.from_numpy(head_square).permute(2, 0, 1).float() / 255.0
            return {
                'full': crow_tensor.unsqueeze(0),
                'head': head_tensor.unsqueeze(0)
            }
        except Exception as e:
            logger.error(f"Error converting to tensor: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error in extract_crow_image: {str(e)}")
        return None

class EnhancedTracker:
    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.15, embedding_threshold=0.6, model_path=None, conf_threshold=0.5, multi_view_stride=1):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.embedding_threshold = embedding_threshold
        self.conf_threshold = conf_threshold
        self.multi_view_stride = multi_view_stride
        self.frame_count = 0
        self.track_embeddings = defaultdict(list)
        self.track_head_embeddings = defaultdict(list)
        self.track_ages = defaultdict(int)
        self.track_history = defaultdict(list)
        self.last_embeddings = {}
        self.last_boxes = {}
        try:
            logger.info("Initializing models...")
            self.model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
            self.multi_view_extractor = create_multi_view_extractor()
            self.color_normalizer = create_normalizer(method='adaptive', use_gpu=True)
            logger.info("All models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
        self.max_batch_size = 4
        self.gpu_timeout = 5
        self.embedding_times = []
        self.tracking_times = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    @timeout(5)
    def _process_detection_batch(self, frame, boxes, indices):
        batch_embeddings = {'full': [], 'head': []}
        for box, idx in zip(boxes, indices):
            try:
                img_tensors = self.extract_crow_image(frame, box)
                if img_tensors is None:
                    raise ValueError("Image extraction failed")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                full_tensor = img_tensors['full'].to(device)
                head_tensor = img_tensors['head'].to(device)
                with torch.no_grad():
                    full_emb = self.model(full_tensor).squeeze(-1).squeeze(-1)
                    head_emb = self.model(head_tensor).squeeze(-1).squeeze(-1)
                full_emb = full_emb.cpu().numpy()
                head_emb = head_emb.cpu().numpy()
                # Use weighted average instead of concatenation to maintain 512 dimensions
                combined_emb = full_emb * 0.7 + head_emb * 0.3
                # Normalize the combined embedding
                combined_emb = combined_emb / np.linalg.norm(combined_emb)
                batch_embeddings['full'].append(combined_emb)
                batch_embeddings['head'].append(head_emb)
                self.last_embeddings[idx] = (combined_emb, {'head': head_emb})
                self.last_boxes[idx] = box
            except Exception as e:
                logger.error(f"Error processing detection {idx}: {str(e)}")
                batch_embeddings['full'].append(np.zeros(512, dtype=np.float32))
                batch_embeddings['head'].append(np.zeros(512, dtype=np.float32))
        return batch_embeddings
    def update(self, frame, detections):
        frame_start_time = time.time()
        self.frame_count += 1
        logger.info(f"Starting update for frame {self.frame_count}")
        try:
            dets = []
            det_boxes = []
            for det in detections:
                x1, y1, x2, y2 = det['box']
                score = det['score']
                dets.append([x1, y1, x2, y2, score])
                det_boxes.append(det['box'])
            dets = np.array(dets) if dets else np.empty((0, 5))
            logger.info(f"Prepared {len(dets)} detections")
            det_embeddings = []
            det_head_embeddings = []
            for i in range(0, len(det_boxes), self.max_batch_size):
                batch_boxes = det_boxes[i:i + self.max_batch_size]
                batch_indices = list(range(i, min(i + self.max_batch_size, len(det_boxes))))
                try:
                    batch_embeddings = self._process_detection_batch(frame, batch_boxes, batch_indices)
                    det_embeddings.extend(batch_embeddings['full'])
                    det_head_embeddings.extend(batch_embeddings['head'])
                except TimeoutException:
                    logger.error(f"Timeout processing batch {i//self.max_batch_size}")
                    for _ in batch_indices:
                        det_embeddings.append(np.zeros(512, dtype=np.float32))
                        det_head_embeddings.append(np.zeros(512, dtype=np.float32))
                except Exception as e:
                    logger.error(f"Error processing batch {i//self.max_batch_size}: {str(e)}")
                    for _ in batch_indices:
                        det_embeddings.append(np.zeros(512, dtype=np.float32))
                        det_head_embeddings.append(np.zeros(512, dtype=np.float32))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            logger.info("Updating SORT tracker")
            tracks = self.tracker.update(dets)
            logger.info(f"SORT tracker returned {len(tracks)} tracks")
            updated_tracks = []
            for track in tracks:
                try:
                    x1, y1, x2, y2, track_id = track
                    track_id = int(track_id)
                    det_idx = None
                    best_iou = 0
                    for i, det in enumerate(dets):
                        iou = compute_iou(det[:4], track[:4])
                        if iou > best_iou:
                            best_iou = iou
                            det_idx = i
                    if det_idx is not None and det_idx < len(det_embeddings):
                        if track_id not in self.track_embeddings:
                            self.track_embeddings[track_id] = []
                            self.track_head_embeddings[track_id] = []
                        self.track_embeddings[track_id].append(det_embeddings[det_idx])
                        self.track_head_embeddings[track_id].append(det_head_embeddings[det_idx])
                        max_embeddings = min(5, 3 + self.track_ages[track_id] // 30)
                        self.track_embeddings[track_id] = self.track_embeddings[track_id][-max_embeddings:]
                        self.track_head_embeddings[track_id] = self.track_head_embeddings[track_id][-max_embeddings:]
                    self.track_history[track_id].append(track[:4])
                    max_history = min(10, 5 + self.track_ages[track_id] // 30)
                    self.track_history[track_id] = self.track_history[track_id][-max_history:]
                    self.track_ages[track_id] += 1
                    updated_tracks.append(track)
                except Exception as e:
                    logger.error(f"Error updating track {track_id}: {str(e)}")
            frame_time = time.time() - frame_start_time
            logger.info(f"Completed frame {self.frame_count} in {frame_time:.3f} seconds")
            return np.array(updated_tracks) if updated_tracks else np.empty((0, 5))
        except Exception as e:
            logger.error(f"Error in tracker update: {str(e)}")
            return np.empty((0, 5))
    def extract_crow_image(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            pad_w = int((x2 - x1) * 0.3)
            pad_h = int((y2 - y1) * 0.3)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            crow_img = frame[y1:y2, x1:x2]
            head_height = int((y2 - y1) * 0.33)
            head_img = frame[y1:y1 + head_height, x1:x2]
            crow_img = cv2.resize(crow_img, (224, 224))
            head_img = cv2.resize(head_img, (224, 224))
            crow_tensor = torch.from_numpy(crow_img).permute(2, 0, 1).float() / 255.0
            head_tensor = torch.from_numpy(head_img).permute(2, 0, 1).float() / 255.0
            return {
                'full': crow_tensor.unsqueeze(0),
                'head': head_tensor.unsqueeze(0)
            }
        except Exception as e:
            logger.error(f"Error extracting crow image: {str(e)}")
            return None

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
