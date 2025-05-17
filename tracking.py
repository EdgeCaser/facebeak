import cv2
import numpy as np
import torchvision
from torchvision.models import resnet18
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from db import save_crow_embedding, get_all_crows, get_crow_history
from sort import Sort
from scipy.spatial.distance import cdist
from color_normalization import create_normalizer
from multi_view import create_multi_view_extractor
import logging
import time
import threading
import gc
from functools import wraps

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
                # Thread is still running, meaning it timed out
                thread._stop()
                raise TimeoutException(f"Function timed out after {seconds} seconds")
            
            if error:
                raise error[0]
            
            return result[0] if result else None
        
        return wrapper
    return decorator

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
        
        # Upsampling layers
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
    # Check if image is too small
    h, w = img_tensor.shape[-2:]
    if h >= min_size and w >= min_size:
        return img_tensor
    
    with torch.no_grad():
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Apply super-resolution
        enhanced = sr_model(img_tensor)
        
        # Ensure output is in [0, 1] range
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced.cpu()

def compute_embedding(img_tensors):
    """Compute feature embeddings for both head and body regions."""
    embeddings = {}
    with torch.no_grad():
        if torch.cuda.is_available():
            for key, tensor in img_tensors.items():
                img_tensors[key] = tensor.cuda()
        
        # Compute embeddings for each region
        for key, tensor in img_tensors.items():
            features = model(tensor)
            embeddings[key] = features.squeeze().cpu().numpy()
    
    # Combine embeddings (concatenate or average)
    # Using concatenation for now, but we could experiment with other methods
    combined = np.concatenate([
        embeddings['full'],  # Full body embedding
        embeddings['head']   # Head embedding
    ])
    
    return combined, embeddings  # Return both combined and individual embeddings

def extract_crow_image(frame, box, padding=0.3, min_size=100):
    """Extract and preprocess crow image for embedding computation, including separate head and body regions."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # Add padding, ensuring we don't go out of frame
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    # Extract full crow image
    crow_img = frame[y1:y2, x1:x2]
    
    # Define head region (upper third of the body)
    head_height = int((y2 - y1) * 0.33)  # Head is roughly 1/3 of body height
    head_img = frame[y1:y1 + head_height, x1:x2]
    
    # Apply adaptive histogram equalization to enhance details
    def enhance_image(img):
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img
    
    crow_img = enhance_image(crow_img)
    head_img = enhance_image(head_img)
    
    # Resize maintaining aspect ratio
    def resize_with_aspect(img, target_size=224):
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to square
        square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return square
    
    crow_square = resize_with_aspect(crow_img)
    head_square = resize_with_aspect(head_img)
    
    # Convert to tensors
    crow_tensor = torch.from_numpy(crow_square).permute(2, 0, 1).float() / 255.0
    head_tensor = torch.from_numpy(head_square).permute(2, 0, 1).float() / 255.0
    
    # Apply super-resolution if needed
    crow_tensor = apply_super_resolution(crow_tensor.unsqueeze(0), min_size).squeeze(0)
    head_tensor = apply_super_resolution(head_tensor.unsqueeze(0), min_size).squeeze(0)
    
    return {
        'full': crow_tensor.unsqueeze(0),  # Add batch dimension
        'head': head_tensor.unsqueeze(0)
    }

class EnhancedTracker:
    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.15, embedding_threshold=0.6, model_path=None, conf_threshold=0.5, multi_view_stride=1):
        """Initialize tracker with simplified processing and timeout handling."""
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
        
        # Initialize models with error handling
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
        
        # Performance settings
        self.max_batch_size = 4  # Reduced batch size
        self.gpu_timeout = 5  # 5 second timeout for GPU operations
        self.embedding_times = []
        self.tracking_times = []
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @timeout(5)  # 5 second timeout for batch processing
    def _process_detection_batch(self, frame, boxes, indices):
        """Process a batch of detections with simplified embedding computation."""
        batch_embeddings = {'full': [], 'head': []}
        
        for box, idx in zip(boxes, indices):
            try:
                # Extract image
                img_tensors = self.extract_crow_image(frame, box)
                if img_tensors is None:
                    raise ValueError("Image extraction failed")

                # Move to GPU
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                full_tensor = img_tensors['full'].to(device)
                head_tensor = img_tensors['head'].to(device)

                # Compute embeddings
                with torch.no_grad():
                    full_emb = self.model(full_tensor).squeeze(-1).squeeze(-1)
                    head_emb = self.model(head_tensor).squeeze(-1).squeeze(-1)

                # Move to CPU and store
                full_emb = full_emb.cpu().numpy()
                head_emb = head_emb.cpu().numpy()
                
                # Combine embeddings
                combined_emb = np.concatenate([
                    full_emb * 0.7,
                    head_emb * 0.3
                ])
                
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
        """Update tracks with simplified processing and timeout handling."""
        frame_start_time = time.time()
        self.frame_count += 1
        logger.info(f"Starting update for frame {self.frame_count}")

        try:
            # Prepare detections
            dets = []
            det_boxes = []
            for det in detections:
                x1, y1, x2, y2 = det['box']
                score = det['score']
                dets.append([x1, y1, x2, y2, score])
                det_boxes.append(det['box'])
            dets = np.array(dets) if dets else np.empty((0, 5))
            logger.info(f"Prepared {len(dets)} detections")

            # Process detections in smaller batches
            det_embeddings = []
            det_head_embeddings = []
            
            for i in range(0, len(det_boxes), self.max_batch_size):
                batch_boxes = det_boxes[i:i + self.max_batch_size]
                batch_indices = list(range(i, min(i + self.max_batch_size, len(det_boxes))))
                
                try:
                    # Process batch with timeout
                    batch_embeddings = self._process_detection_batch(frame, batch_boxes, batch_indices)
                    det_embeddings.extend(batch_embeddings['full'])
                    det_head_embeddings.extend(batch_embeddings['head'])
                except TimeoutException:
                    logger.error(f"Timeout processing batch {i//self.max_batch_size}")
                    # Add zero embeddings for failed batch
                    for _ in batch_indices:
                        det_embeddings.append(np.zeros(512, dtype=np.float32))
                        det_head_embeddings.append(np.zeros(512, dtype=np.float32))
                except Exception as e:
                    logger.error(f"Error processing batch {i//self.max_batch_size}: {str(e)}")
                    for _ in batch_indices:
                        det_embeddings.append(np.zeros(512, dtype=np.float32))
                        det_head_embeddings.append(np.zeros(512, dtype=np.float32))
                
                # Clear GPU memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update SORT tracker
            logger.info("Updating SORT tracker")
            tracks = self.tracker.update(dets)
            logger.info(f"SORT tracker returned {len(tracks)} tracks")

            # Update track information
            updated_tracks = []
            for track in tracks:
                try:
                    x1, y1, x2, y2, track_id = track
                    track_id = int(track_id)
                    
                    # Find matching detection
                    det_idx = None
                    best_iou = 0
                    for i, det in enumerate(dets):
                        iou = compute_iou(det[:4], track[:4])
                        if iou > best_iou:
                            best_iou = iou
                            det_idx = i

                    if det_idx is not None and det_idx < len(det_embeddings):
                        # Update track embeddings
                        if track_id not in self.track_embeddings:
                            self.track_embeddings[track_id] = []
                            self.track_head_embeddings[track_id] = []

                        self.track_embeddings[track_id].append(det_embeddings[det_idx])
                        self.track_head_embeddings[track_id].append(det_head_embeddings[det_idx])
                        
                        # Limit history size
                        max_embeddings = min(5, 3 + self.track_ages[track_id] // 30)
                        self.track_embeddings[track_id] = self.track_embeddings[track_id][-max_embeddings:]
                        self.track_head_embeddings[track_id] = self.track_head_embeddings[track_id][-max_embeddings:]

                    # Update track history
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
        """Simplified image extraction with error handling."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            
            # Add padding
            pad_w = int((x2 - x1) * 0.3)
            pad_h = int((y2 - y1) * 0.3)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            # Extract and resize
            crow_img = frame[y1:y2, x1:x2]
            head_height = int((y2 - y1) * 0.33)
            head_img = frame[y1:y1 + head_height, x1:x2]
            
            # Resize to 224x224
            crow_img = cv2.resize(crow_img, (224, 224))
            head_img = cv2.resize(head_img, (224, 224))
            
            # Convert to tensors
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
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def assign_crow_ids(frames, detections_list, video_path=None, max_age=5, min_hits=2, iou_threshold=0.2, embedding_threshold=0.7, return_track_history=False, multi_view_stride=1):
    """
    Use enhanced tracking to assign consistent IDs to detected crows across frames and videos.
    Combines SORT with visual embeddings for better identity tracking.
    If return_track_history is True, also return a list of per-frame track box dicts for interpolation.
    multi_view_stride: only run multi-view extraction every N frames (default 1 = every frame)
    """
    print("[INFO] Starting enhanced crow tracking...")
    labeled_frames = []
    tracker = EnhancedTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        embedding_threshold=embedding_threshold,
        multi_view_stride=multi_view_stride
    )

    # Get known crows for display
    known_crows = get_all_crows()
    crow_ids = {crow['id']: crow['id'] for crow in known_crows}

    track_history_per_frame = []  # List of {track_id: [x1, y1, x2, y2]} for each frame

    for frame_idx, (frame, detections) in enumerate(zip(frames, detections_list)):
        t0 = time.time()
        # Update tracks
        tracks = tracker.update(frame, detections)
        t1 = time.time()
        print(f"[INFO] Frame {frame_idx+1}/{len(frames)}: tracking/embedding took {t1-t0:.3f} seconds")
        frame_copy = frame.copy()
        frame_tracks = {}
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            # Get crow identity from database
            if track_id in tracker.track_embeddings and tracker.track_embeddings[track_id]:
                embedding = tracker.track_embeddings[track_id][-1]  # Use most recent embedding
                db_crow_id = save_crow_embedding(
                    embedding,
                    video_path=video_path,
                    frame_number=frame_idx,
                    confidence=track[4]  # Use track confidence
                )
                crow_id = crow_ids.get(db_crow_id, db_crow_id)
                # Draw box and label
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
